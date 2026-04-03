from __future__ import annotations

import argparse
import statistics
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .models import RequestResult, Scenario


def _percentiles(values: list[float]) -> dict[str, float | None]:
    """Compute mean and p50/p90/p95/p99 using linear interpolation."""
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p95": None, "p99": None}
    mean = statistics.mean(values)
    if len(values) < 2:
        return {
            "mean": mean,
            "p50": values[0],
            "p90": values[0],
            "p95": values[0],
            "p99": values[0],
        }
    q = statistics.quantiles(values, n=100, method="linear")
    return {
        "mean": mean,
        "p50": q[49],
        "p90": q[89],
        "p95": q[94],
        "p99": q[98],
    }


def _rate(numerator: float, denominator_seconds: float) -> float | None:
    if denominator_seconds <= 0:
        return None
    return numerator / denominator_seconds


def _summarize_results(
    args: argparse.Namespace,
    scenario: Scenario,
    run_id: str,
    run_started_ts: float,
    run_ended_ts: float,
    results: list[RequestResult],
    warmup_results: list[RequestResult],
) -> dict[str, Any]:
    completed = [result for result in results if result.ok]
    ttfts = [result.ttft_ms for result in completed if result.ttft_ms is not None]
    totals = [result.latency_ms for result in completed]
    tpots = [result.tpot_ms for result in completed if result.tpot_ms is not None]
    queue_waits = [
        result.queue_wait_ms for result in completed if result.queue_wait_ms is not None
    ]
    executions = [
        result.execution_ms for result in completed if result.execution_ms is not None
    ]
    output_tokens = [result.output_tokens or 0 for result in completed]
    throughput_window_seconds = max(run_ended_ts - run_started_ts, 0.0)
    error_counts = Counter(result.error_type or "ok" for result in results)
    status_counts = Counter(
        str(result.http_status) if result.http_status is not None else "none"
        for result in results
    )

    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "mode": args.mode,
        "scenario": scenario.name,
        "scenario_description": scenario.description,
        "requested_requests": len(results),
        "warmup_requests": len(warmup_results),
        "measurement_window_seconds": throughput_window_seconds,
        "completed_requests": len(completed),
        "failed_requests": len(results) - len(completed),
        "rejected_requests": sum(1 for result in results if result.http_status == 503),
        "success_rate": (len(completed) / len(results)) if results else None,
        "request_throughput_rps": _rate(len(completed), throughput_window_seconds),
        "output_token_throughput_tps": _rate(
            sum(output_tokens), throughput_window_seconds
        ),
        "total_output_tokens": sum(output_tokens),
        "latency_ms": _percentiles(totals),
        "ttft_ms": _percentiles(ttfts),
        "tpot_ms": _percentiles(tpots),
        "queue_wait_ms": _percentiles(queue_waits),
        "execution_ms": _percentiles(executions),
        "error_counts": dict(error_counts),
        "http_status_counts": dict(status_counts),
        "scenario_mix_counts": dict(
            Counter(result.prompt_source for result in results)
        ),
    }
    return summary
