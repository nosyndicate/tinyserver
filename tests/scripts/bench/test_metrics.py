from __future__ import annotations

from argparse import Namespace

import pytest

from scripts.bench.metrics import _percentiles, _rate, _summarize_results
from scripts.bench.models import RequestResult
from scripts.bench.scenarios import _default_scenarios


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


class TestRate:
    def test_positive(self) -> None:
        assert _rate(10.0, 2.0) == 5.0

    def test_zero_denominator(self) -> None:
        assert _rate(10.0, 0.0) is None

    def test_negative_denominator(self) -> None:
        assert _rate(10.0, -1.0) is None


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
    args = Namespace(
        base_url="http://127.0.0.1:8000",
        endpoint="stream_v2",
        mode="closed",
    )
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
