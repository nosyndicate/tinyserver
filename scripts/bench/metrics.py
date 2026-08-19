from __future__ import annotations

import argparse
import statistics
from collections import Counter
from datetime import datetime, timezone
from operator import attrgetter
from typing import Any

from .models import RequestResult, Scenario


def _percentiles(values: list[float]) -> dict[str, float | int | None]:
    """
    Count, min, max, mean and p50/p90/p95/p99 using inclusive interpolation.

    ``count`` is 0 for empty input rather than None: the sample size is known
    even when nothing else is.
    """
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    if len(values) < 2:
        return {
            "count": len(values),
            "min": values[0],
            "max": values[0],
            "mean": statistics.mean(values),
            "p50": values[0],
            "p90": values[0],
            "p95": values[0],
            "p99": values[0],
        }
    q = statistics.quantiles(values, n=100, method="inclusive")
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "p50": q[49],
        "p90": q[89],
        "p95": q[94],
        "p99": q[98],
    }


def _rate(numerator: float, denominator_seconds: float) -> float | None:
    if denominator_seconds <= 0:
        return None
    return numerator / denominator_seconds


# Overall distributions in the summary. ``ttft_ms``/``tpot_ms`` are the
# client-side aliases (see RequestResult for the naming); the server-sourced
# pair is summarized alongside and is the only meaningful one for sync
# endpoints, where no token ever crosses the wire individually.
_OVERALL_METRICS = (
    "latency_ms",
    "ttft_ms",
    "tpot_ms",
    "queue_wait_ms",
    "execution_ms",
    "server_ttft_ms",
    "server_tpot_ms",
)

# The per-class block carries the per-request decode-rate distribution
# (``tokens_per_s``) instead of the server TTFT/TPOT detail.
_CLASS_METRICS = (
    "latency_ms",
    "ttft_ms",
    "tpot_ms",
    "queue_wait_ms",
    "execution_ms",
    "tokens_per_s",
)


def _distribution_blocks(
    completed: list[RequestResult], metrics: tuple[str, ...]
) -> dict[str, dict[str, float | int | None]]:
    """One percentile block per metric, over completed requests only."""
    blocks: dict[str, dict[str, float | int | None]] = {}
    for metric in metrics:
        getter = attrgetter(metric)
        values = [
            value for result in completed if (value := getter(result)) is not None
        ]
        blocks[metric] = _percentiles(values)
    return blocks


def _class_summaries(results: list[RequestResult]) -> dict[str, dict[str, Any]]:
    """
    Per-class reduction keyed by ``prompt_source``.

    Counts and rates cover every request in the class; distributions cover
    completed ones only, mirroring the overall block. Classes are whatever the
    results actually carry, so a run with no requests produces ``{}``.
    """
    grouped: dict[str, list[RequestResult]] = {}
    for result in results:
        grouped.setdefault(result.prompt_source, []).append(result)

    summaries: dict[str, dict[str, Any]] = {}
    for class_name, class_results in grouped.items():
        completed = [result for result in class_results if result.ok]
        summaries[class_name] = {
            "requests": len(class_results),
            "completed_requests": len(completed),
            "success_rate": len(completed) / len(class_results),
            "rejected_requests": sum(
                1 for result in class_results if result.http_status == 503
            ),
            **_distribution_blocks(completed, _CLASS_METRICS),
        }
    return summaries


def _summarize_results(
    args: argparse.Namespace,
    scenario: Scenario,
    run_id: str,
    window_start_s: float,
    window_end_s: float,
    results: list[RequestResult],
    warmup_results: list[RequestResult],
    open_loop: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Reduce per-request results to a run summary.

    ``window_start_s`` / ``window_end_s`` are monotonic offsets from the run's
    ``RunClock`` epoch, taken around the dispatch loop after warmup.

    ``open_loop`` carries the offer-window report from the open-loop dispatcher
    (target vs. achieved arrival rate, dispatch lag, ``client_saturated``). It
    is ``None`` for closed-loop runs, where arrival rate is not an independent
    variable. When present it is merged in flat and last, so the dispatcher's
    measured report takes precedence over the run-shape echoes below.
    """
    completed = [result for result in results if result.ok]
    output_tokens = [result.output_tokens or 0 for result in completed]
    throughput_window_seconds = max(window_end_s - window_start_s, 0.0)
    error_counts = Counter(result.error_type or "ok" for result in results)
    status_counts = Counter(
        str(result.http_status) if result.http_status is not None else "none"
        for result in results
    )

    summary: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "mode": args.mode,
        "scenario": scenario.name,
        "scenario_description": scenario.description,
        # Run shape, echoed so the summary is interpretable without config.json.
        # A knob the mode never reads is null rather than its default value — a
        # default would read as a load setting that was in effect. A closed loop
        # has no arrival schedule to fall behind, so client_saturated is False
        # there; the open-loop merge below replaces it with the measured
        # verdict.
        "concurrency": args.concurrency if args.mode == "closed" else None,
        "arrival_rate": args.arrival_rate if args.mode == "open" else None,
        "client_max_in_flight": (
            args.client_max_in_flight if args.mode == "open" else None
        ),
        "client_saturated": False,
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
        **_distribution_blocks(completed, _OVERALL_METRICS),
        "by_prompt_source": _class_summaries(results),
        "error_counts": dict(error_counts),
        "http_status_counts": dict(status_counts),
        "scenario_mix_counts": dict(
            Counter(result.prompt_source for result in results)
        ),
    }
    if open_loop is not None:
        # Merged last by design: the measured offer-window report (achieved
        # rate, saturation verdict, window) overrides the echoed defaults above.
        summary.update(open_loop)
    return summary
