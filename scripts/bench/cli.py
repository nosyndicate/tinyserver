from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .execution import (
    _run_closed_loop,
    _run_closed_loop_for_duration,
    _run_open_loop,
    _run_open_loop_for_duration,
    _run_warmup,
)
from .metrics import _summarize_results
from .models import RunClock
from .output import _resolve_output_dir, _write_json, _write_jsonl
from .planning import _build_request_plans
from .scenarios import DEFAULT_SCENARIO_FILE, _load_scenarios

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 120.0

#  When continuous batching is implemented, the next checkpoint should be:
#   - add scheduler metadata to worker events and SSE final chunks
#   - benchmark short_short, long_long, and mixed before/after batching
#   - add batch-size and queue-wait distributions to the saved artifacts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the inference server.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--endpoint",
        default="stream_v2",
        choices=[
            "generate",
            "generate_v2",
            "generate_v3",
            "generate_v4",
            "stream",
            "stream_v2",
            "stream_v3",
            "stream_v4",
            "generate/stream",
            "generate/stream_v2",
            "generate/stream_v3",
            "generate/stream_v4",
        ],
    )
    parser.add_argument("--scenario", default="short_short")
    parser.add_argument("--scenario-file")
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=["closed", "open"], default="closed")
    # Closed-loop worker count. Open loop is bounded by --client-max-in-flight
    # instead, so that arrival rate and client capacity stay separate knobs.
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--arrival-rate", type=float)
    parser.add_argument("--client-max-in-flight", type=int, default=256)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument("--out", default="bench-results")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.mode == "closed" and args.concurrency <= 0:
        raise ValueError("--concurrency must be positive for closed-loop mode")
    if args.mode == "open":
        if args.arrival_rate is None or args.arrival_rate <= 0:
            raise ValueError("--arrival-rate must be positive for open-loop mode")
        if args.client_max_in_flight <= 0:
            raise ValueError(
                "--client-max-in-flight must be positive; it bounds client capacity"
                " in open-loop mode (--concurrency is closed-loop only)"
            )
    if (args.requests is None) == (args.duration_seconds is None):
        raise ValueError("Specify exactly one of --requests or --duration-seconds")
    if args.requests is not None and args.requests <= 0:
        raise ValueError("--requests must be positive")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive")
    # Checked last, once both values above are known good. A window shorter than
    # one arrival interval cannot sample the rate it claims to offer: the
    # schedule calls for under one request, so any measurement is an artifact of
    # rounding rather than of the server.
    if args.mode == "open" and args.duration_seconds is not None:
        interval_s = 1.0 / args.arrival_rate
        if args.duration_seconds < interval_s:
            raise ValueError(
                "--duration-seconds must cover at least one arrival interval"
                f" ({interval_s:.4g}s at --arrival-rate {args.arrival_rate})"
            )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)

    scenarios = _load_scenarios(args.scenario_file)
    if args.scenario not in scenarios:
        raise ValueError(
            f"Unknown scenario {args.scenario!r}. Available: {', '.join(sorted(scenarios))}"
        )
    scenario = scenarios[args.scenario]

    prompt_override = Path(args.prompt_file).read_text() if args.prompt_file else None
    all_plans = _build_request_plans(
        scenario,
        (
            args.warmup_requests
            if args.duration_seconds is not None
            else args.requests + args.warmup_requests
        ),
        prompt_override,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.seed,
    )

    # One clock for the whole run: every recorded timestamp is a monotonic
    # offset from this epoch, and the wall-clock half is persisted in
    # config.json so offsets can be mapped back to a real date afterwards.
    run_clock = RunClock.start()
    warmup_results = _run_warmup(args, all_plans, run_clock)
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S")
    window_start_s = run_clock.offset()
    open_loop_stats: dict[str, Any] | None = None
    if args.requests is not None:
        measurement_plans = all_plans[args.warmup_requests :]
        if args.mode == "closed":
            results = _run_closed_loop(args, measurement_plans, run_id, run_clock)
        else:
            outcome = _run_open_loop(args, measurement_plans, run_id, run_clock)
            results, open_loop_stats = outcome.results, outcome.stats
    else:
        if args.mode == "closed":
            results = _run_closed_loop_for_duration(
                args, scenario, run_id, prompt_override, run_clock
            )
        else:
            outcome = _run_open_loop_for_duration(
                args, scenario, run_id, prompt_override, run_clock
            )
            results, open_loop_stats = outcome.results, outcome.stats
    window_end_s = run_clock.offset()

    summary = _summarize_results(
        args,
        scenario,
        run_id,
        window_start_s,
        window_end_s,
        results,
        warmup_results,
        open_loop_stats,
    )

    if args.summary_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    out_dir = _resolve_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "summary.json", summary)
    _write_json(
        out_dir / "config.json",
        {
            "args": vars(args),
            "clock": {
                "wall_epoch_s": run_clock.wall_epoch_s,
                "wall_epoch_iso": datetime.fromtimestamp(
                    run_clock.wall_epoch_s, timezone.utc
                ).isoformat(),
                "perf_epoch_s": run_clock.perf_epoch_s,
            },
            # Fixed-interval, so a rerun offers requests at the same instants.
            "arrival_process": "deterministic" if args.mode == "open" else None,
            "scenario_file": str(
                Path(args.scenario_file)
                if args.scenario_file is not None
                else DEFAULT_SCENARIO_FILE
            ),
            "scenario": {
                "name": scenario.name,
                "description": scenario.description,
                "requests": [asdict(req) for req in scenario.requests],
            },
        },
    )
    _write_jsonl(out_dir / "requests.jsonl", [result.to_json() for result in results])

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote results to {out_dir}")
    return 0
