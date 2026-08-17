from __future__ import annotations

import hashlib
from typing import Any, Generator
from unittest.mock import patch

import pytest
import requests as requests_lib

import scripts.bench.runners as bench_runners
from scripts.bench.models import RequestPlan, RunClock
from scripts.bench.runners import (
    StreamAccumulator,
    StreamProtocolError,
    _deterministic_gate,
    _endpoint_path,
    _full_url,
    _make_result,
    _parse_sse_chunk,
    _request_runner,
    _run_stream_request,
    _run_sync_request,
    _tpot_ms,
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


def _scripted_clock(*offsets: float) -> RunClock:
    """A RunClock whose offset() walks a fixed script, then repeats the last."""
    remaining = list(offsets)

    def counter() -> float:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return RunClock(wall_epoch_s=1000.0, perf_epoch_s=0.0, counter=counter)


def _feed(acc: StreamAccumulator, *chunks: dict[str, Any]) -> StreamAccumulator:
    for chunk in chunks:
        acc.feed(chunk)
    return acc


def _token_event(index: int, token_str: str) -> dict[str, Any]:
    return {"type": "token", "token_str": token_str, "index": index}


def _done_event(**overrides: Any) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "type": "done",
        "finish_reason": "max_length",
        "prompt_tokens": 11,
        "output_tokens": 2,
        "ttft_ms": 80.0,
        "total_ms": 180.0,
        "tokens_per_s": 11.1,
        "queue_wait_ms": 25.0,
        "execution_ms": 155.0,
    }
    for key, value in overrides.items():
        if value is None:
            chunk.pop(key, None)
        else:
            chunk[key] = value
    return chunk


# ---------------------------------------------------------------------------
# _endpoint_path / _full_url
# ---------------------------------------------------------------------------


class TestEndpointPath:
    @pytest.mark.parametrize(
        "endpoint, expected",
        [
            ("generate", "/generate"),
            ("generate_v2", "/generate_v2"),
            ("generate_v3", "/generate_v3"),
            ("generate_v4", "/generate_v4"),
            ("stream", "/generate/stream"),
            ("stream_v2", "/generate/stream_v2"),
            ("stream_v3", "/generate/stream_v3"),
            ("stream_v4", "/generate/stream_v4"),
            ("generate/stream", "/generate/stream"),
            ("generate/stream_v2", "/generate/stream_v2"),
            ("generate/stream_v3", "/generate/stream_v3"),
            ("generate/stream_v4", "/generate/stream_v4"),
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
        'data: {"type":"token","token_str":"Hello","index":0}\n',
        'data: {"type":"token","token_str":" world","index":1}\n',
        (
            'data: {"type":"done","finish_reason":"max_length",'
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
        clock=_scripted_clock(0.0, 0.1, 0.4),
    )

    assert result.ok is True
    assert result.output_tokens == 2
    assert result.output_tokens_source == "server"
    assert result.prompt_tokens == 11
    assert result.tokens_per_s == 11.1
    # Server-reported components stay on the server clock.
    assert result.server_ttft_ms == 80.0
    assert result.server_total_ms == 180.0
    assert result.server_queue_wait_ms == 25.0
    assert result.server_execution_ms == 155.0
    # server_tpot = (155 - (80 - 25)) / (2 - 1)
    assert result.server_tpot_ms == pytest.approx(100.0)
    # Client components come from the scripted clock: start 0.0, first token
    # 0.1, completion 0.4.
    assert result.client_http_ms == pytest.approx(400.0)
    assert result.client_ttft_ms == pytest.approx(100.0)
    assert result.client_tpot_ms == pytest.approx(300.0)
    # Closed loop: no dispatch lag, so latency is the client HTTP duration.
    assert result.client_dispatch_lag_ms == 0.0
    assert result.latency_ms == pytest.approx(400.0)
    # Legacy aliases preserved for existing analysis scripts.
    assert result.ttft_ms == result.client_ttft_ms
    assert result.tpot_ms == result.client_tpot_ms
    assert result.queue_wait_ms == result.server_queue_wait_ms
    assert result.execution_ms == result.server_execution_ms
    # temperature 0.0 with no seed in the payload.
    assert result.deterministic_gate is False
    assert result.output_sha256 == hashlib.sha256(b"Hello world").hexdigest()


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
                clock=_scripted_clock(0.0, 0.2),
            )
        assert result.ok is True
        assert result.http_status == 200
        assert result.output_tokens == 5
        assert result.output_tokens_source == "server"
        assert result.prompt_tokens == 10
        assert result.tokens_per_s == 25.0
        assert result.server_ttft_ms == 50.0
        assert result.server_total_ms == 200.0
        assert result.server_queue_wait_ms == 15.0
        assert result.server_execution_ms == 185.0
        # server_tpot = (185 - (50 - 15)) / (5 - 1)
        assert result.server_tpot_ms == pytest.approx(37.5)
        # The sync endpoint streams nothing, so there is no client-observable
        # first token. The server's figure stays in server_ttft_ms.
        assert result.client_ttft_ms is None
        assert result.client_tpot_ms is None
        assert result.first_token_ts is None
        # Clock scripted 0.0 -> 0.2, so the client observed 200ms end to end.
        assert result.client_http_ms == pytest.approx(200.0)
        assert result.latency_ms == pytest.approx(200.0)
        assert result.response_text_chars == len("hello world")
        assert result.output_sha256 == hashlib.sha256(b"hello world").hexdigest()

    def test_success_computes_server_tpot(self) -> None:
        fake = _FakeSyncResponse(
            200,
            json_data={
                "text": "abc",
                "ttft_ms": 40.0,
                "total_ms": 200.0,
                "output_tokens": 5,
                "prompt_tokens": 8,
                "tokens_per_s": 25.0,
                "queue_wait_ms": 20.0,
                "execution_ms": 180.0,
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
                clock=_scripted_clock(0.0, 0.2),
            )
        # Entirely on the server clock:
        # (execution_ms - (ttft_ms - queue_wait_ms)) / (output_tokens - 1)
        # = (180 - (40 - 20)) / 4 = 40.0
        assert result.server_tpot_ms == pytest.approx(40.0)
        # No client-side TPOT for a non-streaming endpoint.
        assert result.client_tpot_ms is None

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
                clock=_scripted_clock(0.0, 0.2),
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
                clock=_scripted_clock(0.0, 0.2),
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
                clock=_scripted_clock(0.0, 0.2),
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
                clock=_scripted_clock(0.0, 0.2),
            )
        assert result.ok is False
        assert result.error_type == "request_exception"
        assert "refused" in result.error  # type: ignore[operator]


# ---------------------------------------------------------------------------
# RunClock
# ---------------------------------------------------------------------------


class TestRunClock:
    def test_offset_is_relative_to_epoch(self) -> None:
        clock = RunClock(wall_epoch_s=1000.0, perf_epoch_s=5.0, counter=lambda: 7.25)
        assert clock.offset() == pytest.approx(2.25)

    def test_scripted_clock_walks_then_holds(self) -> None:
        clock = _scripted_clock(0.0, 1.0, 4.0)
        assert [clock.offset(), clock.offset(), clock.offset(), clock.offset()] == [
            0.0,
            1.0,
            4.0,
            4.0,
        ]


# ---------------------------------------------------------------------------
# _tpot_ms
# ---------------------------------------------------------------------------


class TestTpotMs:
    def test_averages_over_inter_token_gaps(self) -> None:
        # 5 tokens: 200ms span, 40ms to the first token, 4 gaps left.
        assert _tpot_ms(200.0, 40.0, 5) == pytest.approx(40.0)

    def test_two_tokens_has_one_gap(self) -> None:
        assert _tpot_ms(180.0, 80.0, 2) == pytest.approx(100.0)

    @pytest.mark.parametrize("num_tokens", [0, 1])
    def test_below_two_tokens_is_none(self, num_tokens: int) -> None:
        assert _tpot_ms(200.0, 40.0, num_tokens) is None

    @pytest.mark.parametrize(
        "span, first, count",
        [(None, 40.0, 5), (200.0, None, 5), (200.0, 40.0, None)],
    )
    def test_missing_input_is_none(
        self, span: float | None, first: float | None, count: int | None
    ) -> None:
        assert _tpot_ms(span, first, count) is None

    def test_negative_span_clamps_to_zero(self) -> None:
        # Can happen when the first-token stamp lands after the span end
        # because the two came from different measurement points.
        assert _tpot_ms(40.0, 200.0, 5) == 0.0


# ---------------------------------------------------------------------------
# _deterministic_gate
# ---------------------------------------------------------------------------


class TestDeterministicGate:
    def test_greedy_and_seeded_is_gated(self) -> None:
        assert _deterministic_gate({"temperature": 0.0, "seed": 7}) is True

    def test_greedy_without_seed_is_not_gated(self) -> None:
        assert _deterministic_gate({"temperature": 0.0}) is False

    def test_sampling_with_seed_is_not_gated(self) -> None:
        # At temperature > 0 the output varies with batch shape, so the hash
        # is not comparable across server versions.
        assert _deterministic_gate({"temperature": 0.8, "seed": 7}) is False

    def test_empty_payload_is_not_gated(self) -> None:
        assert _deterministic_gate({}) is False


# ---------------------------------------------------------------------------
# StreamAccumulator
# ---------------------------------------------------------------------------


class TestStreamAccumulator:
    def test_counts_empty_string_tokens(self) -> None:
        acc = _feed(
            StreamAccumulator(now=_scripted_clock(0.0).offset),
            _token_event(0, "a"),
            _token_event(1, ""),
            _done_event(output_tokens=2),
        )
        assert acc.client_token_count == 2
        assert acc.response_text_chars == 1
        assert acc.resolve_output_tokens() == (2, "server")

    def test_done_supplies_authoritative_count(self) -> None:
        acc = _feed(
            StreamAccumulator(now=_scripted_clock(0.0).offset),
            _token_event(0, "a"),
            _done_event(output_tokens=1),
        )
        assert acc.resolve_output_tokens() == (1, "server")

    def test_null_ttft_for_zero_token_eos(self) -> None:
        acc = _feed(
            StreamAccumulator(now=_scripted_clock(0.0, 0.5).offset),
            _done_event(
                output_tokens=0,
                finish_reason="eos",
                ttft_ms=None,
            ),
        )
        assert acc.server_ttft_ms is None
        assert acc.server_decode_start_ms() is None
        assert acc.first_token_offset_s is None
        assert acc.done is True

    def test_server_decode_start_rebases_ttft_onto_execution_clock(self) -> None:
        acc = _feed(
            StreamAccumulator(now=_scripted_clock(0.0).offset),
            _token_event(0, "a"),
            _token_event(1, "b"),
            _done_event(ttft_ms=80.0, queue_wait_ms=25.0),
        )
        assert acc.server_decode_start_ms() == pytest.approx(55.0)

    def test_first_token_offset_stamped_once(self) -> None:
        acc = _feed(
            StreamAccumulator(now=_scripted_clock(0.25, 0.75, 1.5).offset),
            _token_event(0, "a"),
            _token_event(1, "b"),
            _done_event(),
        )
        assert acc.first_token_offset_s == pytest.approx(0.25)

    def test_error_event_stops_stream_and_keeps_partial_counts(self) -> None:
        acc = _feed(
            StreamAccumulator(now=_scripted_clock(0.0).offset),
            _token_event(0, "a"),
            {"type": "error", "error": "engine exploded"},
        )
        assert acc.error == "engine exploded"
        assert acc.done is True
        assert acc.client_token_count == 1
        assert acc.server_output_tokens is None

    def test_identical_streams_hash_identically(self) -> None:
        def build() -> StreamAccumulator:
            return _feed(
                StreamAccumulator(now=_scripted_clock(0.0).offset),
                _token_event(0, "Hello"),
                _token_event(1, ""),
                _done_event(),
            )

        assert build().output_sha256 == build().output_sha256

    def test_differing_token_changes_hash(self) -> None:
        base = _feed(
            StreamAccumulator(now=_scripted_clock(0.0).offset),
            _token_event(0, "Hello"),
            _token_event(1, " world"),
            _done_event(),
        )
        other = _feed(
            StreamAccumulator(now=_scripted_clock(0.0).offset),
            _token_event(0, "Hello"),
            _token_event(1, " there"),
            _done_event(),
        )
        assert base.output_sha256 != other.output_sha256

    def test_hash_is_order_sensitive(self) -> None:
        forward = _feed(
            StreamAccumulator(now=_scripted_clock(0.0).offset),
            _token_event(0, "ab"),
            _token_event(1, "c"),
            _done_event(),
        )
        reordered = _feed(
            StreamAccumulator(now=_scripted_clock(0.0).offset),
            _token_event(0, "c"),
            _token_event(1, "ab"),
            _done_event(),
        )
        assert forward.output_sha256 != reordered.output_sha256

    def test_rejects_non_contiguous_index(self) -> None:
        acc = StreamAccumulator(now=_scripted_clock(0.0).offset)
        with pytest.raises(StreamProtocolError, match="expected token index 0"):
            acc.feed(_token_event(1, "a"))

    def test_rejects_done_count_mismatch(self) -> None:
        acc = _feed(
            StreamAccumulator(now=_scripted_clock(0.0).offset),
            _token_event(0, "a"),
        )
        with pytest.raises(StreamProtocolError, match="does not match"):
            acc.feed(_done_event(output_tokens=2))

    def test_rejects_unknown_or_missing_type(self) -> None:
        acc = StreamAccumulator(now=_scripted_clock(0.0).offset)
        with pytest.raises(StreamProtocolError, match="unknown or missing"):
            acc.feed({"token_str": "legacy"})

    def test_rejects_event_after_terminal(self) -> None:
        acc = _feed(
            StreamAccumulator(now=_scripted_clock(0.0).offset),
            _done_event(output_tokens=0, finish_reason="eos"),
        )
        with pytest.raises(StreamProtocolError, match="after the terminal"):
            acc.feed(_token_event(0, "late"))
