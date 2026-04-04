from __future__ import annotations

from typing import Any, Generator
from unittest.mock import patch

import pytest
import requests as requests_lib

import scripts.bench.runners as bench_runners
from scripts.bench.models import RequestPlan
from scripts.bench.runners import (
    _endpoint_path,
    _full_url,
    _make_result,
    _parse_sse_chunk,
    _request_runner,
    _run_stream_request,
    _run_sync_request,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DUMMY_PLAN = RequestPlan(
    ordinal=5,
    scenario_name="test",
    payload={"prompt": "x"},
    prompt_length_chars=1,
    prompt_source="short",
    metadata={"key": "value"},
)


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self.status_code = 200
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = False) -> Generator[str, None, None]:
        yield from self._lines

    def __enter__(self) -> "_FakeStreamResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class _FakeSyncResponse:
    def __init__(
        self,
        status_code: int,
        json_data: dict[str, Any] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._json_data = json_data
        self.text = text

    def json(self) -> dict[str, Any]:
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


# ---------------------------------------------------------------------------
# _endpoint_path / _full_url
# ---------------------------------------------------------------------------


class TestEndpointPath:
    @pytest.mark.parametrize(
        "endpoint, expected",
        [
            ("generate", "/generate"),
            ("generate_v2", "/generate_v2"),
            ("stream", "/generate/stream"),
            ("stream_v2", "/generate/stream_v2"),
            ("generate/stream", "/generate/stream"),
            ("generate/stream_v2", "/generate/stream_v2"),
        ],
    )
    def test_all_endpoints(self, endpoint: str, expected: str) -> None:
        assert _endpoint_path(endpoint) == expected

    def test_already_prefixed(self) -> None:
        assert _endpoint_path("/custom") == "/custom"


class TestFullUrl:
    def test_basic(self) -> None:
        assert _full_url("http://host:8000", "generate") == "http://host:8000/generate"

    def test_strips_trailing_slash(self) -> None:
        assert _full_url("http://host:8000/", "generate") == "http://host:8000/generate"


# ---------------------------------------------------------------------------
# _parse_sse_chunk
# ---------------------------------------------------------------------------


class TestParseSseChunk:
    def test_valid(self) -> None:
        assert _parse_sse_chunk('data: {"token_str":"hi"}') == {"token_str": "hi"}

    def test_non_sse(self) -> None:
        assert _parse_sse_chunk("event: message") is None

    def test_empty(self) -> None:
        assert _parse_sse_chunk("") is None


# ---------------------------------------------------------------------------
# _make_result
# ---------------------------------------------------------------------------


class TestMakeResult:
    def test_computes_latency_from_timestamps(self) -> None:
        r = _make_result(
            run_id="r1",
            endpoint="generate",
            mode="closed",
            plan=_DUMMY_PLAN,
            start_ts=1.0,
            end_ts=1.5,
        )
        assert r.latency_ms == pytest.approx(500.0)
        assert r.ordinal == 5
        assert r.metadata == {"key": "value"}

    def test_uses_explicit_latency(self) -> None:
        r = _make_result(
            run_id="r1",
            endpoint="generate",
            mode="closed",
            plan=_DUMMY_PLAN,
            start_ts=1.0,
            end_ts=1.5,
            latency_ms=999.0,
        )
        assert r.latency_ms == 999.0

    def test_error_fields_propagated(self) -> None:
        r = _make_result(
            run_id="r1",
            endpoint="generate",
            mode="closed",
            plan=_DUMMY_PLAN,
            start_ts=1.0,
            end_ts=2.0,
            ok=False,
            error_type="timeout",
            error="request timed out",
        )
        assert r.ok is False
        assert r.error_type == "timeout"
        assert r.error == "request timed out"

    def test_defaults_to_not_ok(self) -> None:
        r = _make_result(
            run_id="r1",
            endpoint="generate",
            mode="closed",
            plan=_DUMMY_PLAN,
            start_ts=0.0,
            end_ts=0.1,
        )
        assert r.ok is False
        assert r.http_status is None
        assert r.error_type is None

    def test_none_optional_fields(self) -> None:
        r = _make_result(
            run_id="r1",
            endpoint="generate",
            mode="closed",
            plan=_DUMMY_PLAN,
            start_ts=0.0,
            end_ts=0.1,
        )
        assert r.first_token_ts is None
        assert r.ttft_ms is None
        assert r.tpot_ms is None
        assert r.output_tokens is None
        assert r.prompt_tokens is None
        assert r.tokens_per_s is None
        assert r.queue_wait_ms is None
        assert r.execution_ms is None
        assert r.response_text_chars is None


# ---------------------------------------------------------------------------
# _request_runner
# ---------------------------------------------------------------------------


class TestRequestRunner:
    def test_stream_endpoints(self) -> None:
        assert _request_runner("stream") is _run_stream_request
        assert _request_runner("stream_v2") is _run_stream_request
        assert _request_runner("generate/stream") is _run_stream_request
        assert _request_runner("generate/stream_v2") is _run_stream_request

    def test_sync_endpoints(self) -> None:
        assert _request_runner("generate") is _run_sync_request
        assert _request_runner("generate_v2") is _run_sync_request


# ---------------------------------------------------------------------------
# _run_stream_request
# ---------------------------------------------------------------------------


def test_run_stream_request_uses_done_chunk_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        'data: {"token_str":"Hello","is_first":true,"is_done":false}\n',
        (
            'data: {"token_str":" world","is_first":false,"is_done":true,'
            '"prompt_tokens":11,"output_tokens":2,"ttft_ms":80.0,"total_ms":180.0,'
            '"tokens_per_s":11.1,"queue_wait_ms":25.0,"execution_ms":155.0}\n'
        ),
    ]

    def fake_post(*args: Any, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(chunks)

    monkeypatch.setattr(bench_runners.requests, "post", fake_post)

    plan = RequestPlan(
        ordinal=0,
        scenario_name="short_short",
        payload={
            "prompt": "hello",
            "max_new_tokens": 2,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        prompt_length_chars=5,
        prompt_source="short",
        metadata={},
    )
    result = _run_stream_request(
        base_url="http://127.0.0.1:8000",
        endpoint="stream_v2",
        timeout_seconds=5.0,
        run_id="run-1",
        mode="closed",
        plan=plan,
    )

    assert result.ok is True
    assert result.output_tokens == 2
    assert result.prompt_tokens == 11
    assert result.ttft_ms == 80.0
    assert result.latency_ms == 180.0
    assert result.queue_wait_ms == 25.0
    assert result.execution_ms == 155.0
    assert result.tokens_per_s == 11.1


# ---------------------------------------------------------------------------
# _run_sync_request
# ---------------------------------------------------------------------------


class TestRunSyncRequest:
    def test_success_parses_metrics(self) -> None:
        fake = _FakeSyncResponse(
            200,
            json_data={
                "text": "hello world",
                "ttft_ms": 50.0,
                "total_ms": 200.0,
                "output_tokens": 5,
                "prompt_tokens": 10,
                "tokens_per_s": 25.0,
                "queue_wait_ms": 15.0,
                "execution_ms": 185.0,
            },
        )
        with patch("scripts.bench.runners.requests.post", return_value=fake):
            result = _run_sync_request(
                base_url="http://localhost:8000",
                endpoint="generate_v2",
                timeout_seconds=5.0,
                run_id="run-1",
                mode="closed",
                plan=_DUMMY_PLAN,
            )
        assert result.ok is True
        assert result.http_status == 200
        assert result.ttft_ms == 50.0
        assert result.latency_ms == 200.0
        assert result.output_tokens == 5
        assert result.prompt_tokens == 10
        assert result.tokens_per_s == 25.0
        assert result.queue_wait_ms == 15.0
        assert result.execution_ms == 185.0
        assert result.response_text_chars == len("hello world")

    def test_success_computes_tpot(self) -> None:
        fake = _FakeSyncResponse(
            200,
            json_data={
                "text": "abc",
                "ttft_ms": 40.0,
                "total_ms": 200.0,
                "output_tokens": 5,
                "prompt_tokens": 8,
                "tokens_per_s": 25.0,
            },
        )
        with patch("scripts.bench.runners.requests.post", return_value=fake):
            result = _run_sync_request(
                base_url="http://localhost:8000",
                endpoint="generate",
                timeout_seconds=5.0,
                run_id="run-1",
                mode="closed",
                plan=_DUMMY_PLAN,
            )
        # tpot = (total_ms - ttft_ms) / (output_tokens - 1) = (200 - 40) / 4 = 40.0
        assert result.tpot_ms == pytest.approx(40.0)

    def test_http_error_with_json(self) -> None:
        fake = _FakeSyncResponse(500, json_data={"error": "internal failure"})
        with patch("scripts.bench.runners.requests.post", return_value=fake):
            result = _run_sync_request(
                base_url="http://localhost:8000",
                endpoint="generate",
                timeout_seconds=5.0,
                run_id="run-1",
                mode="closed",
                plan=_DUMMY_PLAN,
            )
        assert result.ok is False
        assert result.error_type == "http_error"
        assert result.error == "internal failure"
        assert result.http_status == 500

    def test_http_error_non_json(self) -> None:
        fake = _FakeSyncResponse(502, json_data=None, text="Bad Gateway")
        with patch("scripts.bench.runners.requests.post", return_value=fake):
            result = _run_sync_request(
                base_url="http://localhost:8000",
                endpoint="generate",
                timeout_seconds=5.0,
                run_id="run-1",
                mode="closed",
                plan=_DUMMY_PLAN,
            )
        assert result.ok is False
        assert result.error_type == "http_error"
        assert result.error == "Bad Gateway"
        assert result.http_status == 502

    def test_timeout(self) -> None:
        with patch(
            "scripts.bench.runners.requests.post",
            side_effect=requests_lib.Timeout("timed out"),
        ):
            result = _run_sync_request(
                base_url="http://localhost:8000",
                endpoint="generate",
                timeout_seconds=0.001,
                run_id="run-1",
                mode="closed",
                plan=_DUMMY_PLAN,
            )
        assert result.ok is False
        assert result.error_type == "timeout"
        assert "timed out" in result.error  # type: ignore[operator]

    def test_request_exception(self) -> None:
        with patch(
            "scripts.bench.runners.requests.post",
            side_effect=requests_lib.ConnectionError("refused"),
        ):
            result = _run_sync_request(
                base_url="http://localhost:8000",
                endpoint="generate",
                timeout_seconds=5.0,
                run_id="run-1",
                mode="closed",
                plan=_DUMMY_PLAN,
            )
        assert result.ok is False
        assert result.error_type == "request_exception"
        assert "refused" in result.error  # type: ignore[operator]
