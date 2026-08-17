from __future__ import annotations

import argparse
import math
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from .metrics import _percentiles
from .models import RequestPlan, RequestResult, RunClock, Scenario
from .planning import _build_request_plans
from .runners import RequestRunner, _request_runner

# Fraction of the target arrival rate the client must actually sustain before a
# run is considered a clean measurement of the *server*.
_SATURATION_RATE_FLOOR = 0.95


def _run_warmup(
    args: argparse.Namespace, plans: list[RequestPlan], clock: RunClock
) -> list[RequestResult]:
    if args.warmup_requests <= 0:
        return []

    warmup_plans = plans[: args.warmup_requests]
    runner = _request_runner(args.endpoint)
    results: list[RequestResult] = []
    for plan in warmup_plans:
        results.append(
            runner(
                args.base_url,
                args.endpoint,
                args.timeout_seconds,
                "warmup",
                args.mode,
                plan,
                clock,
            )
        )
    return results


def _run_closed_loop(
    args: argparse.Namespace,
    plans: list[RequestPlan],
    run_id: str,
    clock: RunClock,
) -> list[RequestResult]:
    runner = _request_runner(args.endpoint)
    results: list[RequestResult] = []
    next_index = 0
    next_index_lock = threading.Lock()

    def worker() -> list[RequestResult]:
        nonlocal next_index
        local_results: list[RequestResult] = []
        while True:
            with next_index_lock:
                if next_index >= len(plans):
                    return local_results
                plan = plans[next_index]
                next_index += 1
            local_results.append(
                runner(
                    args.base_url,
                    args.endpoint,
                    args.timeout_seconds,
                    run_id,
                    args.mode,
                    plan,
                    clock,
                )
            )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(args.concurrency)]
        for future in as_completed(futures):
            results.extend(future.result())
    return sorted(results, key=lambda item: item.ordinal)


def _run_closed_loop_for_duration(
    args: argparse.Namespace,
    scenario: Scenario,
    run_id: str,
    prompt_override: str | None,
    clock: RunClock,
) -> list[RequestResult]:
    # Over-provision plans so workers won't exhaust them before the deadline.
    # _build_request_plans cycles through weighted requests, so extra plans
    # are cheap and any unused ones are simply discarded.
    estimated_requests = max(1, int(args.duration_seconds * args.concurrency * 4))
    plans = _build_request_plans(
        scenario,
        estimated_requests,
        prompt_override,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.seed,
    )

    runner = _request_runner(args.endpoint)
    results: list[RequestResult] = []
    next_index_lock = threading.Lock()
    results_lock = threading.Lock()
    next_index = 0
    plans_exhausted = False
    stop = threading.Event()

    def worker() -> None:
        nonlocal next_index, plans_exhausted
        local_results: list[RequestResult] = []
        try:
            while not stop.is_set():
                with next_index_lock:
                    if next_index >= len(plans):
                        plans_exhausted = True
                        break
                    plan = plans[next_index]
                    next_index += 1
                local_results.append(
                    runner(
                        args.base_url,
                        args.endpoint,
                        args.timeout_seconds,
                        run_id,
                        args.mode,
                        plan,
                        clock,
                    )
                )
        finally:
            with results_lock:
                results.extend(local_results)

    deadline = clock.offset() + args.duration_seconds
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(args.concurrency)]
        clock.sleep_until(deadline)
        stop.set()
        for future in as_completed(futures):
            future.result()

    if plans_exhausted:
        print(
            "WARNING: _run_closed_loop_for_duration: plan list exhausted before"
            " deadline; some workers stopped early. Consider increasing"
            " --duration-seconds or reducing --concurrency.",
            file=sys.stderr,
        )

    return sorted(results, key=lambda item: item.ordinal)


# ---------------------------------------------------------------------------
# Open loop
#
# Arrival rate and client capacity are separate concerns. The dispatcher offers
# requests on a fixed schedule; `--client-max-in-flight` bounds how many may be
# outstanding at once. When the bound is reached the dispatcher blocks, which is
# what makes client saturation *measurable* rather than invisible: without it,
# `executor.submit` returns immediately and queues internally, so the loop keeps
# "arriving" on schedule while nothing actually starts.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenLoopOutcome:
    """Results plus the honest account of the offer window that produced them."""

    results: list[RequestResult]
    stats: dict[str, Any] = field(default_factory=dict)


def _arrival_schedule(rate: float, count: int, start_offset_s: float) -> list[float]:
    """
    Fixed-interval arrival offsets from the run epoch.

    Deterministic by design (recorded as ``arrival_process: "deterministic"``);
    a Poisson process would be a separate, opt-in addition.
    """
    if rate <= 0:
        raise ValueError("arrival rate must be positive")
    interval = 1.0 / rate
    return [start_offset_s + index * interval for index in range(count)]


def _saturation_verdict(
    target_rate: float,
    achieved_rate: float | None,
    dispatch_lags_ms: list[float],
) -> bool:
    """
    True when the *client*, not the server, limited the run.

    Either signal is disqualifying: the dispatcher fell behind the offered rate,
    or its p95 lag exceeded a full arrival interval (one whole request's worth of
    slack consumed before the request even started).
    """
    if target_rate <= 0:
        return False
    if achieved_rate is None or achieved_rate < _SATURATION_RATE_FLOOR * target_rate:
        return True
    if not dispatch_lags_ms:
        return False
    p95_lag_ms = _percentiles(dispatch_lags_ms)["p95"]
    if p95_lag_ms is None:
        return False
    return p95_lag_ms / 1000.0 > 1.0 / target_rate


def _window_partition(
    results: list[RequestResult], offer_end_s: float
) -> tuple[list[RequestResult], list[RequestResult]]:
    """
    Split completions into those finishing inside the offer window and those
    that drained after it closed.
    """
    in_window = [result for result in results if result.end_ts <= offer_end_s]
    drained = [result for result in results if result.end_ts > offer_end_s]
    return in_window, drained


def _open_loop_stats(
    target_rate: float,
    offer_start_s: float,
    offer_end_s: float,
    max_in_flight: int,
    results: list[RequestResult],
    requested_duration_seconds: float | None = None,
) -> dict[str, Any]:
    """
    Reduce a finished open-loop run to its offer-window report.

    ``requested_duration_seconds`` is what ``--duration-seconds`` asked for, and
    is ``None`` in request-count mode. It is reported next to
    ``offer_window_seconds`` rather than replacing it: the two differ by under
    one arrival interval when ``duration × rate`` is non-integral, and the
    schedule-relative window is the honest denominator for the achieved rate.
    """
    offer_seconds = max(offer_end_s - offer_start_s, 0.0)
    achieved_rate = len(results) / offer_seconds if offer_seconds > 0 else None
    dispatch_lags = [result.client_dispatch_lag_ms for result in results]
    in_window, drained = _window_partition(results, offer_end_s)
    drain_seconds = (
        max(max(result.end_ts for result in drained) - offer_end_s, 0.0)
        if drained
        else 0.0
    )
    return {
        "arrival_process": "deterministic",
        "target_arrival_rate": target_rate,
        "achieved_arrival_rate": achieved_rate,
        "offer_window_seconds": offer_seconds,
        "requested_duration_seconds": requested_duration_seconds,
        "client_max_in_flight": max_in_flight,
        "client_dispatch_lag_ms": _percentiles(dispatch_lags),
        "completions_in_offer_window": len(in_window),
        "drain_seconds": drain_seconds,
        "client_saturated": _saturation_verdict(
            target_rate, achieved_rate, dispatch_lags
        ),
    }


def _dispatch_open_loop(
    args: argparse.Namespace,
    plans: list[RequestPlan],
    run_id: str,
    clock: RunClock,
    deadline_offset_s: float | None = None,
    runner: RequestRunner | None = None,
) -> OpenLoopOutcome:
    """
    Offer ``plans`` on a fixed arrival schedule, bounded by in-flight capacity.

    ``deadline_offset_s`` closes the offer window early (duration mode); requests
    already in flight are still drained and counted. ``runner`` is injectable so
    the dispatcher can be exercised without a server.
    """
    if runner is None:
        runner = _request_runner(args.endpoint)

    max_in_flight = args.client_max_in_flight
    slots = threading.Semaphore(max_in_flight)
    results: list[RequestResult] = []
    futures: list[Future[RequestResult]] = []

    def run_and_release(plan: RequestPlan, arrival_offset_s: float) -> RequestResult:
        try:
            return runner(
                args.base_url,
                args.endpoint,
                args.timeout_seconds,
                run_id,
                args.mode,
                plan,
                clock,
                scheduled_arrival_offset_s=arrival_offset_s,
            )
        finally:
            slots.release()

    offer_start_s = clock.offset()
    interval_s = 1.0 / args.arrival_rate
    schedule = _arrival_schedule(args.arrival_rate, len(plans), offer_start_s)
    last_arrival_s: float | None = None

    with ThreadPoolExecutor(max_workers=max_in_flight) as executor:
        for plan, arrival_offset_s in zip(plans, schedule):
            if deadline_offset_s is not None and arrival_offset_s >= deadline_offset_s:
                break
            clock.sleep_until(arrival_offset_s)
            if deadline_offset_s is not None and clock.offset() >= deadline_offset_s:
                break
            # Blocking here is the point: it is where client saturation becomes
            # visible, as dispatch lag on every request offered afterwards.
            if deadline_offset_s is None:
                slots.acquire()
            else:
                timeout = deadline_offset_s - clock.offset()
                if timeout <= 0 or not slots.acquire(timeout=timeout):
                    break
            futures.append(executor.submit(run_and_release, plan, arrival_offset_s))
            last_arrival_s = arrival_offset_s

        # The window an unsaturated client *intends* to occupy runs one full
        # interval past the last arrival — N requests spaced 1/rate apart span
        # (N-1)/rate, so ending at the last arrival would report an achieved
        # rate above target. A client that fell behind overshoots this and is
        # scored on the wall time it actually took.
        #
        # Deliberately NOT capped at the deadline in duration mode, though that
        # lets the window exceed --duration-seconds by under one interval when
        # duration x rate is non-integral. Capping shortens the denominator
        # without changing the request count, so the achieved rate would read
        # *above* target — the overshoot this formula exists to remove. The
        # requested duration is reported alongside instead, and _validate_args
        # rejects durations too short to sample the rate at all.
        intended_end_s = (
            last_arrival_s + interval_s if last_arrival_s is not None else offer_start_s
        )
        offer_end_s = max(clock.offset(), intended_end_s)
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item.ordinal)
    stats = _open_loop_stats(
        args.arrival_rate,
        offer_start_s,
        offer_end_s,
        max_in_flight,
        results,
        requested_duration_seconds=(
            deadline_offset_s - offer_start_s if deadline_offset_s is not None else None
        ),
    )
    return OpenLoopOutcome(results=results, stats=stats)


def _run_open_loop(
    args: argparse.Namespace,
    plans: list[RequestPlan],
    run_id: str,
    clock: RunClock,
) -> OpenLoopOutcome:
    return _dispatch_open_loop(args, plans, run_id, clock)


def _run_open_loop_for_duration(
    args: argparse.Namespace,
    scenario: Scenario,
    run_id: str,
    prompt_override: str | None,
    clock: RunClock,
) -> OpenLoopOutcome:
    total_requests = max(1, math.ceil(args.duration_seconds * args.arrival_rate))
    plans = _build_request_plans(
        scenario,
        total_requests,
        prompt_override,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.seed,
    )
    deadline_offset_s = clock.offset() + args.duration_seconds
    return _dispatch_open_loop(args, plans, run_id, clock, deadline_offset_s)
