from __future__ import annotations

import json
import time
import uuid
from typing import Any

import requests

from .metrics import _rate
from .models import RequestPlan, RequestResult

DEFAULT_HEADERS = {"Accept": "application/json"}
STREAM_HEADERS = {"Accept": "text/event-stream"}


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
