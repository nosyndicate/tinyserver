from __future__ import annotations

from argparse import Namespace

from scripts.bench.metrics import (
    _class_summaries,
    _percentiles,
    _rate,
    _summarize_results,
)
from scripts.bench.models import RequestResult
from scripts.bench.scenarios import _default_scenarios


class TestPercentiles:
    def test_empty(self) -> None:
        result = _percentiles([])
        assert result["count"] == 0
        assert all(value is None for key, value in result.items() if key != "count")

    def test_single_value(self) -> None:
        result = _percentiles([42.0])
        assert result["count"] == 1
        assert result["min"] == 42.0
        assert result["max"] == 42.0
        assert result["mean"] == 42.0
        assert result["p50"] == 42.0
        assert result["p99"] == 42.0

    def test_two_values(self) -> None:
        result = _percentiles([1.0, 2.0])
        assert result["count"] == 2
        assert result["min"] == 1.0
        assert result["max"] == 2.0
        assert result["mean"] == 1.5
        assert result["p50"] == 1.5
        assert result["p99"] is not None


class TestRate:
    def test_positive(self) -> None:
        assert _rate(10.0, 2.0) == 5.0

    def test_zero_denominator(self) -> None:
        assert _rate(10.0, 0.0) is None

    def test_negative_denominator(self) -> None:
        assert _rate(10.0, -1.0) is None


def _make_args(**overrides: object) -> Namespace:
    defaults = dict(
        base_url="http://127.0.0.1:8000",
        endpoint="stream_v2",
        mode="closed",
        concurrency=4,
        arrival_rate=None,
        client_max_in_flight=256,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_result(request_id: str, **overrides: object) -> RequestResult:
    defaults = dict(
        request_id=request_id,
        run_id="run-1",
        ordinal=0,
        scenario_name="short_short",
        endpoint="stream_v2",
        mode="closed",
        prompt_source="short",
        start_ts=0.0,
        first_token_ts=None,
        end_ts=0.5,
        latency_ms=500.0,
        ttft_ms=None,
        tpot_ms=None,
        output_tokens=None,
        prompt_tokens=None,
        tokens_per_s=None,
        queue_wait_ms=None,
        execution_ms=None,
        http_status=200,
        ok=True,
        error_type=None,
        error=None,
        prompt_length_chars=12,
        response_text_chars=None,
    )
    defaults.update(overrides)
    return RequestResult(**defaults)  # type: ignore[arg-type]


def test_summarize_results_reports_failures_and_percentiles() -> None:
    scenario = _default_scenarios()["short_short"]
    results = [
        _make_result(
            "1",
            ordinal=0,
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
            response_text_chars=50,
        ),
        _make_result(
            "2",
            ordinal=1,
            end_ts=0.2,
            latency_ms=200.0,
            http_status=503,
            ok=False,
            error_type="http_error",
            error="busy",
        ),
    ]

    summary = _summarize_results(
        args=_make_args(),
        scenario=scenario,
        run_id="run-1",
        window_start_s=0.0,
        window_end_s=1.0,
        results=results,
        warmup_results=[],
    )

    assert summary["completed_requests"] == 1
    assert summary["failed_requests"] == 1
    assert summary["rejected_requests"] == 1
    assert summary["latency_ms"]["p50"] == 500.0
    assert summary["latency_ms"]["count"] == 1
    assert summary["ttft_ms"]["p50"] == 100.0
    assert summary["queue_wait_ms"]["p50"] == 30.0
    assert summary["execution_ms"]["p50"] == 470.0
    assert summary["output_token_throughput_tps"] == 21.0
    assert summary["error_counts"]["http_error"] == 1


def test_summarize_results_empty() -> None:
    scenario = _default_scenarios()["short_short"]
    summary = _summarize_results(
        args=_make_args(),
        scenario=scenario,
        run_id="run-empty",
        window_start_s=0.0,
        window_end_s=0.0,
        results=[],
        warmup_results=[],
    )
    assert summary["completed_requests"] == 0
    assert summary["failed_requests"] == 0
    assert summary["success_rate"] is None
    assert summary["latency_ms"]["mean"] is None
    assert summary["by_prompt_source"] == {}


def test_summarize_results_reports_server_ttft_and_tpot() -> None:
    scenario = _default_scenarios()["short_short"]
    results = [
        _make_result(
            "1",
            ttft_ms=100.0,
            tpot_ms=12.0,
            server_ttft_ms=120.0,
            server_tpot_ms=15.0,
        ),
    ]

    summary = _summarize_results(
        args=_make_args(),
        scenario=scenario,
        run_id="run-server",
        window_start_s=0.0,
        window_end_s=1.0,
        results=results,
        warmup_results=[],
    )

    # The server-sourced pair is summarized alongside the client aliases; on a
    # sync endpoint only these blocks carry numbers.
    assert summary["server_ttft_ms"]["p50"] == 120.0
    assert summary["server_tpot_ms"]["p50"] == 15.0
    assert summary["ttft_ms"]["p50"] == 100.0
    assert summary["tpot_ms"]["p50"] == 12.0


def test_summarize_results_server_blocks_empty_when_absent() -> None:
    scenario = _default_scenarios()["short_short"]
    summary = _summarize_results(
        args=_make_args(),
        scenario=scenario,
        run_id="run-no-server",
        window_start_s=0.0,
        window_end_s=1.0,
        results=[_make_result("1")],
        warmup_results=[],
    )
    assert summary["server_ttft_ms"]["count"] == 0
    assert summary["server_ttft_ms"]["mean"] is None
    assert summary["server_tpot_ms"]["count"] == 0
    assert summary["server_tpot_ms"]["mean"] is None


def test_summarize_results_merges_open_loop_stats() -> None:
    scenario = _default_scenarios()["short_short"]
    summary = _summarize_results(
        args=_make_args(mode="open", arrival_rate=8.0, client_max_in_flight=64),
        scenario=scenario,
        run_id="run-open",
        window_start_s=0.0,
        window_end_s=2.0,
        results=[_make_result("1", output_tokens=4)],
        warmup_results=[],
        open_loop={
            "target_arrival_rate": 8.0,
            "achieved_arrival_rate": 7.9,
            "client_max_in_flight": 64,
            "client_saturated": True,
        },
    )

    assert summary["target_arrival_rate"] == 8.0
    assert summary["achieved_arrival_rate"] == 7.9
    # The dispatcher's measured verdict wins over the echoed default (False),
    # and the shape echo reads null for the knob open loop never uses.
    assert summary["client_saturated"] is True
    assert summary["arrival_rate"] == 8.0
    assert summary["client_max_in_flight"] == 64
    assert summary["concurrency"] is None
    # The ordinary summary keys survive the merge.
    assert summary["completed_requests"] == 1


def test_summarize_results_pins_closed_loop_run_shape() -> None:
    # A closed loop cannot fall behind an arrival schedule it does not have,
    # so client_saturated is False (present, not omitted) and arrival_rate is
    # null; the *measured* offer-window keys stay open-loop-only. A stray
    # --arrival-rate passed in closed mode must not be echoed as if it were in
    # effect, hence the 8.0 below.
    scenario = _default_scenarios()["short_short"]
    summary = _summarize_results(
        args=_make_args(mode="closed", arrival_rate=8.0),
        scenario=scenario,
        run_id="run-closed",
        window_start_s=0.0,
        window_end_s=1.0,
        results=[_make_result("1")],
        warmup_results=[],
    )
    assert summary["client_saturated"] is False
    assert summary["arrival_rate"] is None
    assert summary["concurrency"] == 4
    assert summary["client_max_in_flight"] is None
    assert "achieved_arrival_rate" not in summary
    assert "target_arrival_rate" not in summary


def test_class_summaries_mixed_classes() -> None:
    results = [
        _make_result(
            "1",
            prompt_source="short",
            ordinal=0,
            latency_ms=500.0,
            ttft_ms=100.0,
            tpot_ms=20.0,
            queue_wait_ms=30.0,
            execution_ms=470.0,
            tokens_per_s=42.0,
        ),
        _make_result(
            "2",
            prompt_source="short",
            ordinal=1,
            latency_ms=300.0,
            ttft_ms=60.0,
            tokens_per_s=40.0,
        ),
        _make_result(
            "3",
            prompt_source="long",
            ordinal=2,
            latency_ms=900.0,
            http_status=503,
            ok=False,
            error_type="http_error",
            error="busy",
        ),
    ]

    summaries = _class_summaries(results)

    assert set(summaries) == {"short", "long"}
    short = summaries["short"]
    assert short["requests"] == 2
    assert short["completed_requests"] == 2
    assert short["success_rate"] == 1.0
    assert short["rejected_requests"] == 0
    assert short["latency_ms"]["count"] == 2
    assert short["latency_ms"]["min"] == 300.0
    assert short["latency_ms"]["max"] == 500.0
    assert short["latency_ms"]["p50"] == 400.0
    assert short["tokens_per_s"]["mean"] == 41.0
    long_class = summaries["long"]
    assert long_class["requests"] == 1
    assert long_class["completed_requests"] == 0
    assert long_class["success_rate"] == 0.0
    assert long_class["rejected_requests"] == 1
    # An all-failed class keeps its block shapes with nothing to report.
    assert long_class["latency_ms"]["count"] == 0
    assert long_class["latency_ms"]["mean"] is None


def test_class_summaries_empty() -> None:
    assert _class_summaries([]) == {}


def test_summarize_results_by_prompt_source() -> None:
    scenario = _default_scenarios()["short_short"]
    results = [
        _make_result("1", prompt_source="short", latency_ms=500.0),
        _make_result(
            "2",
            prompt_source="long",
            latency_ms=900.0,
            http_status=503,
            ok=False,
            error_type="http_error",
            error="busy",
        ),
    ]

    summary = _summarize_results(
        args=_make_args(),
        scenario=scenario,
        run_id="run-mixed",
        window_start_s=0.0,
        window_end_s=1.0,
        results=results,
        warmup_results=[],
    )

    assert summary["scenario_mix_counts"] == {"short": 1, "long": 1}
    assert summary["by_prompt_source"]["short"]["latency_ms"]["p50"] == 500.0
    assert summary["by_prompt_source"]["long"]["rejected_requests"] == 1
