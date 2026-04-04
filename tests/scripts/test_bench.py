from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import pytest
import requests as requests_lib

import scripts.bench.runners as bench_runners
from scripts.bench.cli import _validate_args
from scripts.bench.metrics import _percentiles, _rate, _summarize_results
from scripts.bench.models import RequestPlan, RequestResult, Scenario, ScenarioRequest
from scripts.bench.output import _resolve_output_dir, _write_json, _write_jsonl
from scripts.bench.planning import _build_request_plans
from scripts.bench.runners import (
    _endpoint_path,
    _full_url,
    _make_result,
    _parse_sse_chunk,
    _request_runner,
    _run_stream_request,
    _run_sync_request,
)
from scripts.bench.scenarios import (
    _default_scenarios,
    _load_prompt_from_request_spec,
    _load_scenarios,
)


def test_load_scenarios_merges_json_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("hello from file")
    scenario_file = tmp_path / "scenarios.json"
    scenario_file.write_text(
        json.dumps(
            {
                "custom": {
                    "description": "custom scenario",
                    "requests": [
                        {
                            "prompt_file": "prompt.txt",
                            "max_new_tokens": 12,
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "seed": 123,
                            "metadata": {"class": "custom"},
                        }
                    ],
                }
            }
        )
    )

    scenarios = _load_scenarios(str(scenario_file))

    assert "short_short" in scenarios
    assert scenarios["custom"].description == "custom scenario"
    assert scenarios["custom"].requests[0].seed == 123
    assert scenarios["custom"].requests[0].prompt == "hello from file"


def test_summarize_results_reports_failures_and_percentiles() -> None:
    args = Namespace(
        base_url="http://127.0.0.1:8000",
        endpoint="stream_v2",
        mode="closed",
    )
    scenario = _default_scenarios()["short_short"]
    results = [
        RequestResult(
            request_id="1",
            run_id="run-1",
            ordinal=0,
            scenario_name="short_short",
            endpoint="stream_v2",
            mode="closed",
            prompt_source="short",
            start_ts=0.0,
            first_token_ts=0.1,
            end_ts=0.5,
            latency_ms=500.0,
            ttft_ms=100.0,
            tpot_ms=20.0,
            output_tokens=21,
            prompt_tokens=10,
            tokens_per_s=42.0,
            queue_wait_ms=30.0,
            execution_ms=470.0,
            http_status=200,
            ok=True,
            error_type=None,
            error=None,
            prompt_length_chars=12,
            response_text_chars=50,
        ),
        RequestResult(
            request_id="2",
            run_id="run-1",
            ordinal=1,
            scenario_name="short_short",
            endpoint="stream_v2",
            mode="closed",
            prompt_source="short",
            start_ts=0.0,
            first_token_ts=None,
            end_ts=0.2,
            latency_ms=200.0,
            ttft_ms=None,
            tpot_ms=None,
            output_tokens=None,
            prompt_tokens=None,
            tokens_per_s=None,
            queue_wait_ms=None,
            execution_ms=None,
            http_status=503,
            ok=False,
            error_type="http_error",
            error="busy",
            prompt_length_chars=12,
            response_text_chars=None,
        ),
    ]

    summary = _summarize_results(
        args=args,
        scenario=scenario,
        run_id="run-1",
        run_started_ts=0.0,
        run_ended_ts=1.0,
        results=results,
        warmup_results=[],
    )

    assert summary["completed_requests"] == 1
    assert summary["failed_requests"] == 1
    assert summary["rejected_requests"] == 1
    assert summary["latency_ms"]["p50"] == 500.0
    assert summary["ttft_ms"]["p50"] == 100.0
    assert summary["queue_wait_ms"]["p50"] == 30.0
    assert summary["execution_ms"]["p50"] == 470.0
    assert summary["output_token_throughput_tps"] == 21.0
    assert summary["error_counts"]["http_error"] == 1


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
# _validate_args
# ---------------------------------------------------------------------------


def _make_args(**overrides: Any) -> Namespace:
    """Build a valid closed-loop Namespace, then apply overrides."""
    defaults = {
        "mode": "closed",
        "concurrency": 4,
        "arrival_rate": None,
        "requests": 10,
        "duration_seconds": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class TestValidateArgs:
    def test_valid_closed_loop(self) -> None:
        _validate_args(_make_args())

    def test_valid_open_loop(self) -> None:
        _validate_args(_make_args(mode="open", arrival_rate=5.0, concurrency=4))

    def test_closed_loop_zero_concurrency(self) -> None:
        with pytest.raises(ValueError, match="concurrency"):
            _validate_args(_make_args(concurrency=0))

    def test_open_loop_missing_arrival_rate(self) -> None:
        with pytest.raises(ValueError, match="arrival-rate"):
            _validate_args(_make_args(mode="open", arrival_rate=None))

    def test_open_loop_zero_arrival_rate(self) -> None:
        with pytest.raises(ValueError, match="arrival-rate"):
            _validate_args(_make_args(mode="open", arrival_rate=0))

    def test_open_loop_zero_concurrency(self) -> None:
        with pytest.raises(ValueError, match="concurrency"):
            _validate_args(_make_args(mode="open", arrival_rate=5.0, concurrency=0))

    def test_neither_requests_nor_duration(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            _validate_args(_make_args(requests=None, duration_seconds=None))

    def test_both_requests_and_duration(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            _validate_args(_make_args(requests=10, duration_seconds=5.0))

    def test_requests_zero(self) -> None:
        with pytest.raises(ValueError, match="requests must be positive"):
            _validate_args(_make_args(requests=0))

    def test_duration_zero(self) -> None:
        with pytest.raises(ValueError, match="duration-seconds must be positive"):
            _validate_args(_make_args(requests=None, duration_seconds=0))

    def test_valid_duration_mode(self) -> None:
        _validate_args(_make_args(requests=None, duration_seconds=10.0))


# ---------------------------------------------------------------------------
# _build_request_plans
# ---------------------------------------------------------------------------


def _simple_scenario(
    name: str = "test",
    requests: list[ScenarioRequest] | None = None,
) -> Scenario:
    if requests is None:
        requests = [
            ScenarioRequest(
                prompt="hello",
                max_new_tokens=16,
                temperature=0.5,
                top_p=0.9,
                metadata={"class": "default"},
            )
        ]
    return Scenario(name=name, description="test scenario", requests=requests)


class TestBuildRequestPlans:
    def test_basic_plan_generation(self) -> None:
        scenario = _simple_scenario()
        plans = _build_request_plans(scenario, 3, None, None, None, None, None)
        assert len(plans) == 3
        assert [p.ordinal for p in plans] == [0, 1, 2]
        assert plans[0].payload["prompt"] == "hello"
        assert plans[0].payload["max_new_tokens"] == 16
        assert plans[0].payload["temperature"] == 0.5
        assert plans[0].prompt_source == "default"
        assert "seed" not in plans[0].payload

    def test_weighted_cycling(self) -> None:
        scenario = _simple_scenario(
            requests=[
                ScenarioRequest(prompt="A", weight=3, metadata={"class": "a"}),
                ScenarioRequest(prompt="B", weight=1, metadata={"class": "b"}),
            ]
        )
        plans = _build_request_plans(scenario, 8, None, None, None, None, None)
        sources = [p.prompt_source for p in plans]
        # weight=3 expands to [A, A, A, B], cycling: A,A,A,B,A,A,A,B
        assert sources == ["a", "a", "a", "b", "a", "a", "a", "b"]

    def test_cli_overrides(self) -> None:
        scenario = _simple_scenario()
        plans = _build_request_plans(
            scenario,
            1,
            None,
            max_new_tokens_override=99,
            temperature_override=0.1,
            top_p_override=0.5,
            seed_override=42,
        )
        p = plans[0].payload
        assert p["max_new_tokens"] == 99
        assert p["temperature"] == 0.1
        assert p["top_p"] == 0.5
        assert p["seed"] == 42

    def test_prompt_override(self) -> None:
        scenario = _simple_scenario()
        plans = _build_request_plans(scenario, 1, "overridden", None, None, None, None)
        assert plans[0].payload["prompt"] == "overridden"
        assert plans[0].prompt_length_chars == len("overridden")

    def test_empty_requests_raises(self) -> None:
        scenario = _simple_scenario(requests=[])
        with pytest.raises(ValueError, match="does not contain any request templates"):
            _build_request_plans(scenario, 1, None, None, None, None, None)

    def test_seed_from_scenario(self) -> None:
        scenario = _simple_scenario(requests=[ScenarioRequest(prompt="x", seed=7)])
        plans = _build_request_plans(scenario, 1, None, None, None, None, None)
        assert plans[0].payload["seed"] == 7

    def test_seed_override_beats_scenario(self) -> None:
        scenario = _simple_scenario(requests=[ScenarioRequest(prompt="x", seed=7)])
        plans = _build_request_plans(
            scenario, 1, None, None, None, None, seed_override=99
        )
        assert plans[0].payload["seed"] == 99


# ---------------------------------------------------------------------------
# _percentiles
# ---------------------------------------------------------------------------


class TestPercentiles:
    def test_empty(self) -> None:
        result = _percentiles([])
        assert all(v is None for v in result.values())

    def test_single_value(self) -> None:
        result = _percentiles([42.0])
        assert result["mean"] == 42.0
        assert result["p50"] == 42.0
        assert result["p99"] == 42.0

    def test_two_values(self) -> None:
        # NOTE: _percentiles uses method="linear" which is not a valid method
        # for statistics.quantiles in Python 3.11. This means it currently only
        # works for lists with fewer than 2 elements (hitting early returns).
        # This test documents the bug — it should be fixed to use "inclusive"
        # or "exclusive".
        with pytest.raises(ValueError, match="Unknown method"):
            _percentiles([1.0, 2.0])


# ---------------------------------------------------------------------------
# _rate
# ---------------------------------------------------------------------------


class TestRate:
    def test_positive(self) -> None:
        assert _rate(10.0, 2.0) == 5.0

    def test_zero_denominator(self) -> None:
        assert _rate(10.0, 0.0) is None

    def test_negative_denominator(self) -> None:
        assert _rate(10.0, -1.0) is None


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
        result = _parse_sse_chunk('data: {"token_str":"hi"}')
        assert result == {"token_str": "hi"}

    def test_non_sse(self) -> None:
        assert _parse_sse_chunk("event: message") is None

    def test_empty(self) -> None:
        assert _parse_sse_chunk("") is None


# ---------------------------------------------------------------------------
# _make_result
# ---------------------------------------------------------------------------

_DUMMY_PLAN = RequestPlan(
    ordinal=5,
    scenario_name="test",
    payload={"prompt": "x"},
    prompt_length_chars=1,
    prompt_source="short",
    metadata={"key": "value"},
)


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
# _resolve_output_dir
# ---------------------------------------------------------------------------


class TestResolveOutputDir:
    def test_closed_loop_includes_concurrency(self) -> None:
        args = Namespace(
            out="results",
            scenario="short_short",
            endpoint="stream_v2",
            mode="closed",
            concurrency=8,
        )
        path = _resolve_output_dir(args)
        parts = str(path)
        assert "concurrency=8" in parts
        assert "scenario=short_short" in parts
        assert "mode=closed" in parts

    def test_open_loop_includes_arrival_rate(self) -> None:
        args = Namespace(
            out="results",
            scenario="burst",
            endpoint="generate_v2",
            mode="open",
            arrival_rate=10.0,
        )
        path = _resolve_output_dir(args)
        parts = str(path)
        assert "arrival_rate=10.0" in parts
        assert "mode=open" in parts


# ---------------------------------------------------------------------------
# _write_json / _write_jsonl
# ---------------------------------------------------------------------------


class TestWriteOutput:
    def test_write_json(self, tmp_path: Path) -> None:
        p = tmp_path / "out.json"
        _write_json(p, {"b": 2, "a": 1})
        content = p.read_text()
        parsed = json.loads(content)
        assert parsed == {"a": 1, "b": 2}
        assert content.endswith("\n")

    def test_write_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "out.jsonl"
        _write_jsonl(p, [{"x": 1}, {"y": 2}])
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"x": 1}
        assert json.loads(lines[1]) == {"y": 2}


# ---------------------------------------------------------------------------
# _load_prompt_from_request_spec
# ---------------------------------------------------------------------------


class TestLoadPromptFromRequestSpec:
    def test_inline_prompt(self, tmp_path: Path) -> None:
        assert _load_prompt_from_request_spec({"prompt": "hello"}, tmp_path) == "hello"

    def test_missing_both(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="prompt"):
            _load_prompt_from_request_spec({}, tmp_path)

    def test_prompt_file(self, tmp_path: Path) -> None:
        f = tmp_path / "p.txt"
        f.write_text("from file")
        assert (
            _load_prompt_from_request_spec({"prompt_file": "p.txt"}, tmp_path)
            == "from file"
        )


# ---------------------------------------------------------------------------
# _default_scenarios
# ---------------------------------------------------------------------------


def test_default_scenarios_keys() -> None:
    scenarios = _default_scenarios()
    expected = {"short_short", "long_long", "mixed", "burst", "seeded_deterministic"}
    assert set(scenarios.keys()) == expected
    for name, scenario in scenarios.items():
        assert scenario.name == name
        assert len(scenario.requests) >= 1


# ---------------------------------------------------------------------------
# _summarize_results edge case: empty results
# ---------------------------------------------------------------------------


def test_summarize_results_empty() -> None:
    args = Namespace(
        base_url="http://127.0.0.1:8000",
        endpoint="generate",
        mode="closed",
    )
    scenario = _default_scenarios()["short_short"]
    summary = _summarize_results(
        args=args,
        scenario=scenario,
        run_id="run-empty",
        run_started_ts=0.0,
        run_ended_ts=0.0,
        results=[],
        warmup_results=[],
    )
    assert summary["completed_requests"] == 0
    assert summary["failed_requests"] == 0
    assert summary["success_rate"] is None
    assert summary["latency_ms"]["mean"] is None


# ---------------------------------------------------------------------------
# _FakeSyncResponse helper
# ---------------------------------------------------------------------------


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
