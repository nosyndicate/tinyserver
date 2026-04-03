from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .execution import (
    _run_closed_loop,
    _run_closed_loop_for_duration,
    _run_open_loop,
    _run_open_loop_for_duration,
    _run_warmup,
)
from .metrics import _summarize_results
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
            "stream",
            "stream_v2",
            "generate/stream",
            "generate/stream_v2",
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
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--arrival-rate", type=float)
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
        if args.concurrency <= 0:
            raise ValueError(
                "--concurrency must be positive; it sets client worker count in open-loop mode"
            )
    if (args.requests is None) == (args.duration_seconds is None):
        raise ValueError("Specify exactly one of --requests or --duration-seconds")
    if args.requests is not None and args.requests <= 0:
        raise ValueError("--requests must be positive")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be positive")


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
    if args.requests is not None:
        plans = _build_request_plans(
            scenario,
            args.requests + args.warmup_requests,
            prompt_override,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.seed,
        )
        warmup_plans = plans
    else:
        warmup_plans = _build_request_plans(
            scenario,
            args.warmup_requests,
            prompt_override,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.seed,
        )
        plans = warmup_plans

    warmup_results = _run_warmup(args, warmup_plans)
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S")
    run_started_ts = time.time()
    if args.requests is not None:
        measurement_plans = plans[args.warmup_requests :]
        if args.mode == "closed":
            results = _run_closed_loop(args, measurement_plans, run_id)
        else:
            results = _run_open_loop(args, measurement_plans, run_id)
    else:
        if args.mode == "closed":
            results = _run_closed_loop_for_duration(
                args, scenario, run_id, prompt_override
            )
        else:
            results = _run_open_loop_for_duration(
                args, scenario, run_id, prompt_override
            )
    run_ended_ts = time.time()

    summary = _summarize_results(
        args,
        scenario,
        run_id,
        run_started_ts,
        run_ended_ts,
        results,
        warmup_results,
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
