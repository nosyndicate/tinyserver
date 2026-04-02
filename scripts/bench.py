from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

#  When continuous batching is implemented, the next checkpoint should be:
#   - add scheduler metadata to worker events and SSE final chunks
#   - benchmark short_short, long_long, and mixed before/after batching
#   - add batch-size and queue-wait distributions to the saved artifacts

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_HEADERS = {"Accept": "application/json"}
STREAM_HEADERS = {"Accept": "text/event-stream"}
DEFAULT_SCENARIO_FILE = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "scenarios.json"
)


@dataclass(frozen=True)
class ScenarioRequest:
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int | None = None
    weight: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    requests: list[ScenarioRequest]


@dataclass(frozen=True)
class RequestPlan:
    ordinal: int
    scenario_name: str
    payload: dict[str, Any]
    prompt_length_chars: int
    prompt_source: str
    metadata: dict[str, Any]


@dataclass
class RequestResult:
    request_id: str
    run_id: str
    ordinal: int
    scenario_name: str
    endpoint: str
    mode: str
    prompt_source: str
    start_ts: float
    first_token_ts: float | None
    end_ts: float
    latency_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    output_tokens: int | None
    prompt_tokens: int | None
    tokens_per_s: float | None
    queue_wait_ms: float | None
    execution_ms: float | None
    http_status: int | None
    ok: bool
    error_type: str | None
    error: str | None
    prompt_length_chars: int
    response_text_chars: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _default_scenarios() -> dict[str, Scenario]:
    short_prompt = "Write a short haiku about GPUs and scheduling."
    long_prompt = (
        "You are helping profile an inference server. "
        "Explain the tradeoff between latency, throughput, batching, prefill cost, "
        "decode efficiency, queueing, and fairness for online LLM serving. "
        "Then provide a compact checklist for benchmarking and what signals to log. "
        "Keep the explanation dense and technical.\n\n"
        "Context:\n"
        "- The server currently supports sync and streaming endpoints.\n"
        "- Prefill and decode are already separated.\n"
        "- The next steps include a worker, batching, scheduling policy, and KV cache improvements.\n"
        "- The benchmark should surface saturation behavior and queue pressure.\n"
    )

    return {
        "short_short": Scenario(
            name="short_short",
            description="Short prompt and short output; primarily TTFT-focused.",
            requests=[
                ScenarioRequest(
                    prompt=short_prompt,
                    max_new_tokens=48,
                    temperature=0.8,
                    top_p=0.95,
                    metadata={"class": "short"},
                )
            ],
        ),
        "long_long": Scenario(
            name="long_long",
            description="Long prompt and longer output; throughput and prefill stress.",
            requests=[
                ScenarioRequest(
                    prompt=long_prompt,
                    max_new_tokens=192,
                    temperature=0.8,
                    top_p=0.95,
                    metadata={"class": "long"},
                )
            ],
        ),
        "mixed": Scenario(
            name="mixed",
            description="Weighted mix of short and long requests.",
            requests=[
                ScenarioRequest(
                    prompt=short_prompt,
                    max_new_tokens=48,
                    temperature=0.8,
                    top_p=0.95,
                    weight=3,
                    metadata={"class": "short"},
                ),
                ScenarioRequest(
                    prompt=long_prompt,
                    max_new_tokens=192,
                    temperature=0.8,
                    top_p=0.95,
                    weight=1,
                    metadata={"class": "long"},
                ),
            ],
        ),
        "burst": Scenario(
            name="burst",
            description="Short prompt with moderate output, intended for open-loop pressure tests.",
            requests=[
                ScenarioRequest(
                    prompt=(
                        "List five concrete reasons a queue builds up in an LLM inference server "
                        "under bursty traffic."
                    ),
                    max_new_tokens=96,
                    temperature=0.7,
                    top_p=0.9,
                    metadata={"class": "burst"},
                )
            ],
        ),
        "seeded_deterministic": Scenario(
            name="seeded_deterministic",
            description="Determinism and regression scenario with fixed seed.",
            requests=[
                ScenarioRequest(
                    prompt="Summarize how KV cache reuse changes decode cost in one paragraph.",
                    max_new_tokens=64,
                    temperature=0.0,
                    top_p=1.0,
                    seed=7,
                    metadata={"class": "deterministic"},
                )
            ],
        ),
    }


def _load_scenarios(path: str | None) -> dict[str, Scenario]:
    scenarios = _default_scenarios()
    scenario_path = Path(path) if path is not None else DEFAULT_SCENARIO_FILE
    if not scenario_path.exists():
        return scenarios

    raw = json.loads(scenario_path.read_text())
    loaded: dict[str, Scenario] = {}
    for name, spec in raw.items():
        requests_spec = spec.get("requests")
        if not requests_spec:
            raise ValueError(f"scenario {name!r} must define a non-empty requests list")
        loaded[name] = Scenario(
            name=name,
            description=spec.get("description", ""),
            requests=[
                ScenarioRequest(
                    prompt=_load_prompt_from_request_spec(req, scenario_path.parent),
                    max_new_tokens=int(req.get("max_new_tokens", 64)),
                    temperature=float(req.get("temperature", 0.8)),
                    top_p=float(req.get("top_p", 0.95)),
                    seed=req.get("seed"),
                    weight=int(req.get("weight", 1)),
                    metadata=dict(req.get("metadata", {})),
                )
                for req in requests_spec
            ],
        )
    scenarios.update(loaded)
    return scenarios


def _load_prompt_from_request_spec(
    request_spec: dict[str, Any], scenario_dir: Path
) -> str:
    if "prompt" in request_spec:
        return str(request_spec["prompt"])
    prompt_file = request_spec.get("prompt_file")
    if prompt_file is None:
        raise ValueError("request spec must provide either 'prompt' or 'prompt_file'")
    return (scenario_dir / str(prompt_file)).read_text()


def _build_request_plans(
    scenario: Scenario,
    total_requests: int,
    prompt_override: str | None,
    max_new_tokens_override: int | None,
    temperature_override: float | None,
    top_p_override: float | None,
    seed_override: int | None,
) -> list[RequestPlan]:
    weighted_requests: list[ScenarioRequest] = []
    for req in scenario.requests:
        weighted_requests.extend([req] * max(req.weight, 1))

    if not weighted_requests:
        raise ValueError(
            f"scenario {scenario.name!r} does not contain any request templates"
        )

    plans: list[RequestPlan] = []
    for ordinal in range(total_requests):
        req = weighted_requests[ordinal % len(weighted_requests)]
        prompt = prompt_override if prompt_override is not None else req.prompt
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens_override
            if max_new_tokens_override is not None
            else req.max_new_tokens,
            "temperature": temperature_override
            if temperature_override is not None
            else req.temperature,
            "top_p": top_p_override if top_p_override is not None else req.top_p,
        }
        seed = seed_override if seed_override is not None else req.seed
        if seed is not None:
            payload["seed"] = seed
        plans.append(
            RequestPlan(
                ordinal=ordinal,
                scenario_name=scenario.name,
                payload=payload,
                prompt_length_chars=len(prompt),
                prompt_source=req.metadata.get("class", "default"),
                metadata=dict(req.metadata),
            )
        )
    return plans


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def _endpoint_path(endpoint: str) -> str:
    if endpoint in {"generate", "generate_v2"}:
        return endpoint if endpoint.startswith("/") else f"/{endpoint}"
    if endpoint in {"stream_v2", "generate/stream", "generate/stream_v2"}:
        mapping = {
            "stream_v2": "/generate/stream_v2",
            "generate/stream": "/generate/stream",
            "generate/stream_v2": "/generate/stream_v2",
        }
        return mapping[endpoint]
    if endpoint == "stream":
        return "/generate/stream"
    if endpoint.startswith("/"):
        return endpoint
    return f"/{endpoint}"


def _full_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}{_endpoint_path(endpoint)}"


def _parse_sse_chunk(line: str) -> dict[str, Any] | None:
    if not line.startswith("data: "):
        return None
    return json.loads(line[len("data: ") :])


def _run_sync_request(
    base_url: str,
    endpoint: str,
    timeout_seconds: float,
    run_id: str,
    mode: str,
    plan: RequestPlan,
) -> RequestResult:
    request_id = str(uuid.uuid4())
    start_ts = time.time()
    first_token_ts = None
    try:
        response = requests.post(
            _full_url(base_url, endpoint),
            json=plan.payload,
            timeout=timeout_seconds,
            headers=DEFAULT_HEADERS,
        )
        end_ts = time.time()
        latency_ms = (end_ts - start_ts) * 1000.0
        response_data: dict[str, Any] | None = None
        error = None
        try:
            response_data = response.json()
        except ValueError:
            error = response.text[:500]

        if response.ok and response_data is not None:
            ttft_ms = response_data.get("ttft_ms")
            total_ms = response_data.get("total_ms", latency_ms)
            output_tokens = response_data.get("output_tokens")
            prompt_tokens = response_data.get("prompt_tokens")
            tokens_per_s = response_data.get("tokens_per_s")
            response_text = response_data.get("text", "")
            if ttft_ms is not None:
                first_token_ts = start_ts + (float(ttft_ms) / 1000.0)
            tpot_ms = None
            if output_tokens and ttft_ms is not None and output_tokens > 1:
                tpot_ms = max(float(total_ms) - float(ttft_ms), 0.0) / (
                    output_tokens - 1
                )
            return RequestResult(
                request_id=request_id,
                run_id=run_id,
                ordinal=plan.ordinal,
                scenario_name=plan.scenario_name,
                endpoint=endpoint,
                mode=mode,
                prompt_source=plan.prompt_source,
                start_ts=start_ts,
                first_token_ts=first_token_ts,
                end_ts=end_ts,
                latency_ms=float(total_ms),
                ttft_ms=float(ttft_ms) if ttft_ms is not None else None,
                tpot_ms=tpot_ms,
                output_tokens=output_tokens,
                prompt_tokens=prompt_tokens,
                tokens_per_s=float(tokens_per_s) if tokens_per_s is not None else None,
                queue_wait_ms=(
                    float(response_data["queue_wait_ms"])
                    if response_data.get("queue_wait_ms") is not None
                    else None
                ),
                execution_ms=(
                    float(response_data["execution_ms"])
                    if response_data.get("execution_ms") is not None
                    else None
                ),
                http_status=response.status_code,
                ok=True,
                error_type=None,
                error=None,
                prompt_length_chars=plan.prompt_length_chars,
                response_text_chars=len(response_text),
                metadata=dict(plan.metadata),
            )

        error_message = None
        if response_data is not None:
            error_message = response_data.get("error") or json.dumps(response_data)
        elif error is not None:
            error_message = error
        return RequestResult(
            request_id=request_id,
            run_id=run_id,
            ordinal=plan.ordinal,
            scenario_name=plan.scenario_name,
            endpoint=endpoint,
            mode=mode,
            prompt_source=plan.prompt_source,
            start_ts=start_ts,
            first_token_ts=first_token_ts,
            end_ts=end_ts,
            latency_ms=latency_ms,
            ttft_ms=None,
            tpot_ms=None,
            output_tokens=None,
            prompt_tokens=None,
            tokens_per_s=None,
            queue_wait_ms=None,
            execution_ms=None,
            http_status=response.status_code,
            ok=False,
            error_type="http_error",
            error=error_message,
            prompt_length_chars=plan.prompt_length_chars,
            response_text_chars=None,
            metadata=dict(plan.metadata),
        )
    except requests.Timeout as exc:
        end_ts = time.time()
        return RequestResult(
            request_id=request_id,
            run_id=run_id,
            ordinal=plan.ordinal,
            scenario_name=plan.scenario_name,
            endpoint=endpoint,
            mode=mode,
            prompt_source=plan.prompt_source,
            start_ts=start_ts,
            first_token_ts=first_token_ts,
            end_ts=end_ts,
            latency_ms=(end_ts - start_ts) * 1000.0,
            ttft_ms=None,
            tpot_ms=None,
            output_tokens=None,
            prompt_tokens=None,
            tokens_per_s=None,
            queue_wait_ms=None,
            execution_ms=None,
            http_status=None,
            ok=False,
            error_type="timeout",
            error=str(exc),
            prompt_length_chars=plan.prompt_length_chars,
            response_text_chars=None,
            metadata=dict(plan.metadata),
        )
    except requests.RequestException as exc:
        end_ts = time.time()
        return RequestResult(
            request_id=request_id,
            run_id=run_id,
            ordinal=plan.ordinal,
            scenario_name=plan.scenario_name,
            endpoint=endpoint,
            mode=mode,
            prompt_source=plan.prompt_source,
            start_ts=start_ts,
            first_token_ts=first_token_ts,
            end_ts=end_ts,
            latency_ms=(end_ts - start_ts) * 1000.0,
            ttft_ms=None,
            tpot_ms=None,
            output_tokens=None,
            prompt_tokens=None,
            tokens_per_s=None,
            queue_wait_ms=None,
            execution_ms=None,
            http_status=None,
            ok=False,
            error_type="request_exception",
            error=str(exc),
            prompt_length_chars=plan.prompt_length_chars,
            response_text_chars=None,
            metadata=dict(plan.metadata),
        )


def _run_stream_request(
    base_url: str,
    endpoint: str,
    timeout_seconds: float,
    run_id: str,
    mode: str,
    plan: RequestPlan,
) -> RequestResult:
    request_id = str(uuid.uuid4())
    start_ts = time.time()
    first_token_ts: float | None = None
    output_tokens = 0
    response_text_chars = 0
    prompt_tokens: int | None = None
    queue_wait_ms: float | None = None
    execution_ms: float | None = None
    total_ms_from_server: float | None = None
    ttft_ms_from_server: float | None = None
    tokens_per_s_from_server: float | None = None

    try:
        with requests.post(
            _full_url(base_url, endpoint),
            json=plan.payload,
            timeout=timeout_seconds,
            headers=STREAM_HEADERS,
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                chunk = _parse_sse_chunk(raw_line)
                if chunk is None:
                    continue
                if chunk.get("error"):
                    end_ts = time.time()
                    return RequestResult(
                        request_id=request_id,
                        run_id=run_id,
                        ordinal=plan.ordinal,
                        scenario_name=plan.scenario_name,
                        endpoint=endpoint,
                        mode=mode,
                        prompt_source=plan.prompt_source,
                        start_ts=start_ts,
                        first_token_ts=first_token_ts,
                        end_ts=end_ts,
                        latency_ms=(end_ts - start_ts) * 1000.0,
                        ttft_ms=(first_token_ts - start_ts) * 1000.0
                        if first_token_ts
                        else None,
                        tpot_ms=None,
                        output_tokens=output_tokens,
                        prompt_tokens=prompt_tokens,
                        tokens_per_s=None,
                        queue_wait_ms=queue_wait_ms,
                        execution_ms=execution_ms,
                        http_status=response.status_code,
                        ok=False,
                        error_type="server_error",
                        error=str(chunk.get("error")),
                        prompt_length_chars=plan.prompt_length_chars,
                        response_text_chars=response_text_chars,
                        metadata=dict(plan.metadata),
                    )

                token_str = chunk.get("token_str", "")
                if token_str:
                    output_tokens += 1
                    response_text_chars += len(token_str)
                    if first_token_ts is None:
                        first_token_ts = time.time()
                if chunk.get("is_first") and first_token_ts is None:
                    first_token_ts = time.time()
                if chunk.get("is_done"):
                    prompt_tokens = chunk.get("prompt_tokens")
                    queue_wait_ms = (
                        float(chunk["queue_wait_ms"])
                        if chunk.get("queue_wait_ms") is not None
                        else None
                    )
                    execution_ms = (
                        float(chunk["execution_ms"])
                        if chunk.get("execution_ms") is not None
                        else None
                    )
                    total_ms_from_server = (
                        float(chunk["total_ms"])
                        if chunk.get("total_ms") is not None
                        else None
                    )
                    ttft_ms_from_server = (
                        float(chunk["ttft_ms"])
                        if chunk.get("ttft_ms") is not None
                        else None
                    )
                    tokens_per_s_from_server = (
                        float(chunk["tokens_per_s"])
                        if chunk.get("tokens_per_s") is not None
                        else None
                    )
                    end_ts = time.time()
                    latency_ms = (end_ts - start_ts) * 1000.0
                    ttft_ms = (
                        ttft_ms_from_server
                        if ttft_ms_from_server is not None
                        else (
                            (first_token_ts - start_ts) * 1000.0
                            if first_token_ts is not None
                            else latency_ms
                        )
                    )
                    total_ms = (
                        total_ms_from_server
                        if total_ms_from_server is not None
                        else latency_ms
                    )
                    tpot_ms = None
                    tokens_per_s = tokens_per_s_from_server
                    if output_tokens > 0 and tokens_per_s is None:
                        tokens_per_s = _rate(output_tokens, total_ms / 1000.0)
                    if output_tokens > 1:
                        tpot_ms = max(total_ms - ttft_ms, 0.0) / (output_tokens - 1)
                    return RequestResult(
                        request_id=request_id,
                        run_id=run_id,
                        ordinal=plan.ordinal,
                        scenario_name=plan.scenario_name,
                        endpoint=endpoint,
                        mode=mode,
                        prompt_source=plan.prompt_source,
                        start_ts=start_ts,
                        first_token_ts=first_token_ts,
                        end_ts=end_ts,
                        latency_ms=total_ms,
                        ttft_ms=ttft_ms,
                        tpot_ms=tpot_ms,
                        output_tokens=output_tokens,
                        prompt_tokens=prompt_tokens,
                        tokens_per_s=tokens_per_s,
                        queue_wait_ms=queue_wait_ms,
                        execution_ms=execution_ms,
                        http_status=response.status_code,
                        ok=True,
                        error_type=None,
                        error=None,
                        prompt_length_chars=plan.prompt_length_chars,
                        response_text_chars=response_text_chars,
                        metadata=dict(plan.metadata),
                    )

        end_ts = time.time()
        latency_ms = (end_ts - start_ts) * 1000.0
        return RequestResult(
            request_id=request_id,
            run_id=run_id,
            ordinal=plan.ordinal,
            scenario_name=plan.scenario_name,
            endpoint=endpoint,
            mode=mode,
            prompt_source=plan.prompt_source,
            start_ts=start_ts,
            first_token_ts=first_token_ts,
            end_ts=end_ts,
            latency_ms=latency_ms,
            ttft_ms=(first_token_ts - start_ts) * 1000.0 if first_token_ts else None,
            tpot_ms=None,
            output_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
            tokens_per_s=_rate(output_tokens, latency_ms / 1000.0)
            if output_tokens
            else None,
            queue_wait_ms=queue_wait_ms,
            execution_ms=execution_ms,
            http_status=200,
            ok=False,
            error_type="stream_ended_without_done",
            error="stream ended before is_done event",
            prompt_length_chars=plan.prompt_length_chars,
            response_text_chars=response_text_chars,
            metadata=dict(plan.metadata),
        )
    except requests.Timeout as exc:
        end_ts = time.time()
        return RequestResult(
            request_id=request_id,
            run_id=run_id,
            ordinal=plan.ordinal,
            scenario_name=plan.scenario_name,
            endpoint=endpoint,
            mode=mode,
            prompt_source=plan.prompt_source,
            start_ts=start_ts,
            first_token_ts=first_token_ts,
            end_ts=end_ts,
            latency_ms=(end_ts - start_ts) * 1000.0,
            ttft_ms=(first_token_ts - start_ts) * 1000.0 if first_token_ts else None,
            tpot_ms=None,
            output_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
            tokens_per_s=None,
            queue_wait_ms=queue_wait_ms,
            execution_ms=execution_ms,
            http_status=None,
            ok=False,
            error_type="timeout",
            error=str(exc),
            prompt_length_chars=plan.prompt_length_chars,
            response_text_chars=response_text_chars,
            metadata=dict(plan.metadata),
        )
    except requests.RequestException as exc:
        end_ts = time.time()
        return RequestResult(
            request_id=request_id,
            run_id=run_id,
            ordinal=plan.ordinal,
            scenario_name=plan.scenario_name,
            endpoint=endpoint,
            mode=mode,
            prompt_source=plan.prompt_source,
            start_ts=start_ts,
            first_token_ts=first_token_ts,
            end_ts=end_ts,
            latency_ms=(end_ts - start_ts) * 1000.0,
            ttft_ms=(first_token_ts - start_ts) * 1000.0 if first_token_ts else None,
            tpot_ms=None,
            output_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
            tokens_per_s=None,
            queue_wait_ms=queue_wait_ms,
            execution_ms=execution_ms,
            http_status=None,
            ok=False,
            error_type="request_exception",
            error=str(exc),
            prompt_length_chars=plan.prompt_length_chars,
            response_text_chars=response_text_chars,
            metadata=dict(plan.metadata),
        )


def _request_runner(endpoint: str):
    if "stream" in endpoint:
        return _run_stream_request
    return _run_sync_request


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
    runner = _request_runner(args.endpoint)
    weighted_requests: list[ScenarioRequest] = []
    for req in scenario.requests:
        weighted_requests.extend([req] * max(req.weight, 1))
    if not weighted_requests:
        raise ValueError(
            f"scenario {scenario.name!r} does not contain any request templates"
        )

    results: list[RequestResult] = []
    ordinal_lock = threading.Lock()
    results_lock = threading.Lock()
    next_ordinal = 0
    deadline = time.perf_counter() + args.duration_seconds

    def make_plan(ordinal: int) -> RequestPlan:
        req = weighted_requests[ordinal % len(weighted_requests)]
        prompt = prompt_override if prompt_override is not None else req.prompt
        payload = {
            "prompt": prompt,
            "max_new_tokens": args.max_new_tokens
            if args.max_new_tokens is not None
            else req.max_new_tokens,
            "temperature": args.temperature
            if args.temperature is not None
            else req.temperature,
            "top_p": args.top_p if args.top_p is not None else req.top_p,
        }
        seed = args.seed if args.seed is not None else req.seed
        if seed is not None:
            payload["seed"] = seed
        return RequestPlan(
            ordinal=ordinal,
            scenario_name=scenario.name,
            payload=payload,
            prompt_length_chars=len(prompt),
            prompt_source=req.metadata.get("class", "default"),
            metadata=dict(req.metadata),
        )

    def worker() -> None:
        nonlocal next_ordinal
        local_results: list[RequestResult] = []
        while True:
            with ordinal_lock:
                if time.perf_counter() >= deadline:
                    break
                ordinal = next_ordinal
                next_ordinal += 1
                plan = make_plan(ordinal)
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
        with results_lock:
            results.extend(local_results)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(args.concurrency)]
        for future in as_completed(futures):
            future.result()
    return sorted(results, key=lambda item: item.ordinal)


def _run_open_loop(
    args: argparse.Namespace, plans: list[RequestPlan], run_id: str
) -> list[RequestResult]:
    runner = _request_runner(args.endpoint)
    results: list[RequestResult] = []
    futures: list[Future[RequestResult]] = []
    max_workers = max(args.concurrency or 64, 4)
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
    max_workers = max(args.concurrency or 64, 4)
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


def _build_duration_plans(
    scenario: Scenario,
    duration_seconds: float,
    arrival_rate: float | None,
    concurrency: int | None,
    prompt_override: str | None,
    max_new_tokens_override: int | None,
    temperature_override: float | None,
    top_p_override: float | None,
    seed_override: int | None,
) -> list[RequestPlan]:
    if arrival_rate is not None:
        total_requests = max(1, math.ceil(duration_seconds * arrival_rate))
    else:
        total_requests = max(1, duration_seconds * max(concurrency or 1, 1))
    return _build_request_plans(
        scenario,
        int(total_requests),
        prompt_override,
        max_new_tokens_override,
        temperature_override,
        top_p_override,
        seed_override,
    )


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


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    base = Path(args.out)
    suffix_parts = [
        f"scenario={args.scenario}",
        f"endpoint={args.endpoint}",
        f"mode={args.mode}",
    ]
    if args.mode == "closed":
        suffix_parts.append(f"concurrency={args.concurrency}")
    else:
        suffix_parts.append(f"arrival_rate={args.arrival_rate}")
    return base / ts / "/".join(suffix_parts)


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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
