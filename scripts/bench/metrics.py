from __future__ import annotations

import argparse
import statistics
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .models import RequestResult, Scenario


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(pct * (len(ordered) - 1))
    return ordered[idx]


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
        "latency_ms": {
            "mean": statistics.mean(totals) if totals else None,
            "p50": _percentile(totals, 0.50),
            "p90": _percentile(totals, 0.90),
            "p95": _percentile(totals, 0.95),
            "p99": _percentile(totals, 0.99),
        },
        "ttft_ms": {
            "mean": statistics.mean(ttfts) if ttfts else None,
            "p50": _percentile(ttfts, 0.50),
            "p90": _percentile(ttfts, 0.90),
            "p95": _percentile(ttfts, 0.95),
            "p99": _percentile(ttfts, 0.99),
        },
        "tpot_ms": {
            "mean": statistics.mean(tpots) if tpots else None,
            "p50": _percentile(tpots, 0.50),
            "p90": _percentile(tpots, 0.90),
            "p95": _percentile(tpots, 0.95),
            "p99": _percentile(tpots, 0.99),
        },
        "queue_wait_ms": {
            "mean": statistics.mean(queue_waits) if queue_waits else None,
            "p50": _percentile(queue_waits, 0.50),
            "p90": _percentile(queue_waits, 0.90),
            "p95": _percentile(queue_waits, 0.95),
            "p99": _percentile(queue_waits, 0.99),
        },
        "execution_ms": {
            "mean": statistics.mean(executions) if executions else None,
            "p50": _percentile(executions, 0.50),
            "p90": _percentile(executions, 0.90),
            "p95": _percentile(executions, 0.95),
            "p99": _percentile(executions, 0.99),
        },
        "error_counts": dict(error_counts),
        "http_status_counts": dict(status_counts),
        "scenario_mix_counts": dict(
            Counter(result.prompt_source for result in results)
        ),
    }
    return summary
