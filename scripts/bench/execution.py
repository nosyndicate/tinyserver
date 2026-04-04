from __future__ import annotations

import argparse
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from .models import RequestPlan, RequestResult, Scenario
from .planning import _build_request_plans
from .runners import _request_runner


def _run_warmup(
    args: argparse.Namespace, plans: list[RequestPlan]
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
            )
        )
    return results


def _run_closed_loop(
    args: argparse.Namespace, plans: list[RequestPlan], run_id: str
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
    stop = threading.Event()

    def worker() -> None:
        nonlocal next_index
        local_results: list[RequestResult] = []
        try:
            while not stop.is_set():
                with next_index_lock:
                    if next_index >= len(plans):
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
                    )
                )
        finally:
            with results_lock:
                results.extend(local_results)

    deadline = time.perf_counter() + args.duration_seconds
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(args.concurrency)]
        time.sleep(max(deadline - time.perf_counter(), 0.0))
        stop.set()
        for future in as_completed(futures):
            future.result()
    return sorted(results, key=lambda item: item.ordinal)


def _run_open_loop(
    args: argparse.Namespace, plans: list[RequestPlan], run_id: str
) -> list[RequestResult]:
    runner = _request_runner(args.endpoint)
    results: list[RequestResult] = []
    futures: list[Future[RequestResult]] = []
    max_workers = args.concurrency or 64
    interval_seconds = 1.0 / args.arrival_rate

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        next_dispatch = time.perf_counter()
        for plan in plans:
            now = time.perf_counter()
            sleep_for = next_dispatch - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            futures.append(
                executor.submit(
                    runner,
                    args.base_url,
                    args.endpoint,
                    args.timeout_seconds,
                    run_id,
                    args.mode,
                    plan,
                )
            )
            next_dispatch += interval_seconds

        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item.ordinal)


def _run_open_loop_for_duration(
    args: argparse.Namespace,
    scenario: Scenario,
    run_id: str,
    prompt_override: str | None,
) -> list[RequestResult]:
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
    deadline = time.perf_counter() + args.duration_seconds
    runner = _request_runner(args.endpoint)
    results: list[RequestResult] = []
    futures: list[Future[RequestResult]] = []
    max_workers = args.concurrency or 64
    interval_seconds = 1.0 / args.arrival_rate

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        next_dispatch = time.perf_counter()
        for plan in plans:
            now = time.perf_counter()
            if now >= deadline:
                break
            sleep_for = min(max(next_dispatch - now, 0.0), max(deadline - now, 0.0))
            if sleep_for > 0:
                time.sleep(sleep_for)
            if time.perf_counter() >= deadline:
                break
            futures.append(
                executor.submit(
                    runner,
                    args.base_url,
                    args.endpoint,
                    args.timeout_seconds,
                    run_id,
                    args.mode,
                    plan,
                )
            )
            next_dispatch += interval_seconds

        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item.ordinal)
