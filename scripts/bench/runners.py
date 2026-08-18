from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import requests

from .metrics import _rate
from .models import RequestPlan, RequestResult, RunClock

DEFAULT_HEADERS = {"Accept": "application/json"}
STREAM_HEADERS = {"Accept": "text/event-stream"}

OUTPUT_TOKENS_FROM_SERVER = "server"
OUTPUT_TOKENS_FROM_CLIENT_COUNT = "client_count"


class StreamProtocolError(ValueError):
    """The server emitted an invalid explicit streaming-event sequence."""


def _endpoint_path(endpoint: str) -> str:
    if endpoint in {"generate", "generate_v2", "generate_v3", "generate_v4"}:
        return endpoint if endpoint.startswith("/") else f"/{endpoint}"
    if endpoint in {
        "stream_v2",
        "stream_v3",
        "stream_v4",
        "generate/stream",
        "generate/stream_v2",
        "generate/stream_v3",
        "generate/stream_v4",
    }:
        mapping = {
            "stream_v2": "/generate/stream_v2",
            "stream_v3": "/generate/stream_v3",
            "stream_v4": "/generate/stream_v4",
            "generate/stream": "/generate/stream",
            "generate/stream_v2": "/generate/stream_v2",
            "generate/stream_v3": "/generate/stream_v3",
            "generate/stream_v4": "/generate/stream_v4",
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


def _float_or_none(value: Any) -> float | None:
    return float(value) if value is not None else None


def _tpot_ms(
    span_ms: float | None, first_token_ms: float | None, num_tokens: int | None
) -> float | None:
    """
    Time per output token: the decode span after the first token, divided by the
    number of inter-token gaps.

    ``span_ms`` and ``first_token_ms`` must be measured on the *same* clock and
    from the same origin — mixing a server numerator with a client token count
    (the behaviour this corrected) produces a number with no meaning. Returns
    ``None`` below two tokens, where there is no gap to average over.
    """
    if span_ms is None or first_token_ms is None or num_tokens is None:
        return None
    if num_tokens < 2:
        return None
    return max(span_ms - first_token_ms, 0.0) / (num_tokens - 1)


def _deterministic_gate(payload: dict[str, Any]) -> bool:
    """
    Whether this request's sampling settings make the output reproducible, and
    therefore whether its ``output_sha256`` is meaningful to compare across
    server versions. Sampling at temperature > 0 varies with batch shape, so
    only greedy seeded requests are gated.
    """
    return payload.get("temperature") == 0.0 and payload.get("seed") is not None


@dataclass
class StreamAccumulator:
    """
    Folds SSE chunks into client-side measurements.

    Pure apart from the injected ``now`` (monotonic offsets in seconds), so the
    whole streaming measurement path is testable by feeding dicts.

    Every payload has an explicit ``type`` discriminator. Token events always
    represent sampled non-EOS tokens, including tokens whose decoded text is
    empty. A separate done event carries authoritative counts and metrics.
    """

    now: Callable[[], float]
    first_token_offset_s: float | None = None
    client_token_count: int = 0
    response_text_chars: int = 0
    done: bool = False
    error: str | None = None
    server_prompt_tokens: int | None = None
    server_output_tokens: int | None = None
    server_ttft_ms: float | None = None
    server_total_ms: float | None = None
    server_queue_wait_ms: float | None = None
    server_execution_ms: float | None = None
    server_tokens_per_s: float | None = None
    _hasher: Any = field(default_factory=hashlib.sha256)

    def feed(self, chunk: dict[str, Any]) -> None:
        if self.done:
            raise StreamProtocolError("received an event after the terminal event")

        event_type = chunk.get("type")
        if event_type == "error":
            error = chunk.get("error")
            if not isinstance(error, str) or not error:
                raise StreamProtocolError("error event requires a non-empty error")
            self.error = error
            self.done = True
            return

        if event_type == "token":
            token_str = chunk.get("token_str")
            index = chunk.get("index")
            if not isinstance(token_str, str):
                raise StreamProtocolError("token event requires token_str")
            if index != self.client_token_count:
                raise StreamProtocolError(
                    f"expected token index {self.client_token_count}, got {index!r}"
                )
            if self.first_token_offset_s is None:
                self.first_token_offset_s = self.now()
            self.client_token_count += 1
            self.response_text_chars += len(token_str)
            self._hasher.update(token_str.encode("utf-8"))
            return

        if event_type == "done":
            output_tokens = chunk.get("output_tokens")
            if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
                raise StreamProtocolError("done event requires integer output_tokens")
            if output_tokens != self.client_token_count:
                raise StreamProtocolError(
                    "done output_tokens does not match observed token events: "
                    f"{output_tokens} != {self.client_token_count}"
                )
            finish_reason = chunk.get("finish_reason")
            if finish_reason not in {"eos", "max_length"}:
                raise StreamProtocolError("done event has invalid finish_reason")
            prompt_tokens = chunk.get("prompt_tokens")
            if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
                raise StreamProtocolError("done event requires integer prompt_tokens")

            self.server_prompt_tokens = prompt_tokens
            self.server_output_tokens = output_tokens
            self.server_ttft_ms = _float_or_none(chunk.get("ttft_ms"))
            self.server_total_ms = _float_or_none(chunk.get("total_ms"))
            self.server_queue_wait_ms = _float_or_none(chunk.get("queue_wait_ms"))
            self.server_execution_ms = _float_or_none(chunk.get("execution_ms"))
            self.server_tokens_per_s = _float_or_none(chunk.get("tokens_per_s"))
            self.done = True
            return

        raise StreamProtocolError(
            f"unknown or missing stream event type: {event_type!r}"
        )

    @property
    def output_sha256(self) -> str:
        return self._hasher.hexdigest()

    def resolve_output_tokens(self) -> tuple[int, str]:
        """The authoritative token count, and where it came from."""
        if self.server_output_tokens is not None:
            return int(self.server_output_tokens), OUTPUT_TOKENS_FROM_SERVER
        return self.client_token_count, OUTPUT_TOKENS_FROM_CLIENT_COUNT

    def server_decode_start_ms(self) -> float | None:
        """Server TTFT rebased onto the execution clock (both start at prefill)."""
        if self.server_ttft_ms is None or self.server_queue_wait_ms is None:
            return None
        return self.server_ttft_ms - self.server_queue_wait_ms


def _make_result(
    *,
    run_id: str,
    endpoint: str,
    mode: str,
    plan: RequestPlan,
    start_ts: float,
    end_ts: float,
    scheduled_arrival_offset_s: float | None = None,
    latency_ms: float | None = None,
    first_token_ts: float | None = None,
    ttft_ms: float | None = None,
    tpot_ms: float | None = None,
    output_tokens: int | None = None,
    prompt_tokens: int | None = None,
    tokens_per_s: float | None = None,
    server_total_ms: float | None = None,
    server_queue_wait_ms: float | None = None,
    server_execution_ms: float | None = None,
    server_ttft_ms: float | None = None,
    server_tpot_ms: float | None = None,
    output_sha256: str | None = None,
    output_tokens_source: str | None = None,
    http_status: int | None = None,
    ok: bool = False,
    error_type: str | None = None,
    error: str | None = None,
    response_text_chars: int | None = None,
) -> RequestResult:
    if scheduled_arrival_offset_s is None:
        scheduled_arrival_offset_s = start_ts
    client_http_ms = (end_ts - start_ts) * 1000.0
    client_dispatch_lag_ms = max((start_ts - scheduled_arrival_offset_s) * 1000.0, 0.0)
    if latency_ms is None:
        latency_ms = client_dispatch_lag_ms + client_http_ms
    return RequestResult(
        request_id=str(uuid.uuid4()),
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
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        output_tokens=output_tokens,
        prompt_tokens=prompt_tokens,
        tokens_per_s=tokens_per_s,
        queue_wait_ms=server_queue_wait_ms,
        execution_ms=server_execution_ms,
        http_status=http_status,
        ok=ok,
        error_type=error_type,
        error=error,
        prompt_length_chars=plan.prompt_length_chars,
        response_text_chars=response_text_chars,
        scheduled_arrival_offset_s=scheduled_arrival_offset_s,
        client_dispatch_lag_ms=client_dispatch_lag_ms,
        client_http_ms=client_http_ms,
        client_ttft_ms=ttft_ms,
        client_tpot_ms=tpot_ms,
        server_total_ms=server_total_ms,
        server_queue_wait_ms=server_queue_wait_ms,
        server_execution_ms=server_execution_ms,
        server_ttft_ms=server_ttft_ms,
        server_tpot_ms=server_tpot_ms,
        output_sha256=output_sha256,
        output_tokens_source=output_tokens_source,
        deterministic_gate=_deterministic_gate(plan.payload),
        metadata=dict(plan.metadata),
    )


def _run_sync_request(
    base_url: str,
    endpoint: str,
    timeout_seconds: float,
    run_id: str,
    mode: str,
    plan: RequestPlan,
    clock: RunClock,
    *,
    scheduled_arrival_offset_s: float | None = None,
) -> RequestResult:
    start_ts = clock.offset()
    try:
        response = requests.post(
            _full_url(base_url, endpoint),
            json=plan.payload,
            timeout=timeout_seconds,
            headers=DEFAULT_HEADERS,
        )
        end_ts = clock.offset()
        client_http_ms = (end_ts - start_ts) * 1000.0
        response_data: dict[str, Any] | None = None
        error = None
        try:
            response_data = response.json()
        except ValueError:
            error = response.text[:500]

        if response.ok and response_data is not None:
            server_ttft_ms = _float_or_none(response_data.get("ttft_ms"))
            server_total_ms = _float_or_none(response_data.get("total_ms"))
            server_queue_wait_ms = _float_or_none(response_data.get("queue_wait_ms"))
            server_execution_ms = _float_or_none(response_data.get("execution_ms"))
            output_tokens = response_data.get("output_tokens")
            response_text = response_data.get("text", "")

            # A sync endpoint returns the whole response at once, so there is
            # no client-observable first token: client_ttft_ms and
            # client_tpot_ms (which _tpot_ms derives from it) are null here by
            # definition. Do not "restore" them from server_ttft_ms — that is
            # measured from server enqueue, on the server's clock, while
            # client_http_ms starts before the request is even sent. Assigning
            # one to the other mislabels a server duration rather than rebasing
            # it. The server's view is preserved in the server_* fields.
            client_ttft_ms = None
            first_token_ts = None
            server_decode_start_ms = (
                server_ttft_ms - server_queue_wait_ms
                if server_ttft_ms is not None and server_queue_wait_ms is not None
                else None
            )
            tokens_per_s = _float_or_none(response_data.get("tokens_per_s"))
            if tokens_per_s is None and output_tokens:
                tokens_per_s = _rate(
                    output_tokens,
                    (
                        server_execution_ms
                        if server_execution_ms is not None
                        else client_http_ms
                    )
                    / 1000.0,
                )
            return _make_result(
                run_id=run_id,
                endpoint=endpoint,
                mode=mode,
                plan=plan,
                start_ts=start_ts,
                end_ts=end_ts,
                scheduled_arrival_offset_s=scheduled_arrival_offset_s,
                first_token_ts=first_token_ts,
                ttft_ms=client_ttft_ms,
                tpot_ms=_tpot_ms(client_http_ms, client_ttft_ms, output_tokens),
                output_tokens=output_tokens,
                prompt_tokens=response_data.get("prompt_tokens"),
                tokens_per_s=tokens_per_s,
                server_total_ms=server_total_ms,
                server_queue_wait_ms=server_queue_wait_ms,
                server_execution_ms=server_execution_ms,
                server_ttft_ms=server_ttft_ms,
                server_tpot_ms=_tpot_ms(
                    server_execution_ms, server_decode_start_ms, output_tokens
                ),
                output_sha256=hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
                output_tokens_source=(
                    OUTPUT_TOKENS_FROM_SERVER if output_tokens is not None else None
                ),
                http_status=response.status_code,
                ok=True,
                response_text_chars=len(response_text),
            )

        error_message = None
        if response_data is not None:
            error_message = response_data.get("error") or json.dumps(response_data)
        elif error is not None:
            error_message = error
        return _make_result(
            run_id=run_id,
            endpoint=endpoint,
            mode=mode,
            plan=plan,
            start_ts=start_ts,
            end_ts=end_ts,
            scheduled_arrival_offset_s=scheduled_arrival_offset_s,
            http_status=response.status_code,
            error_type="http_error",
            error=error_message,
        )
    except requests.Timeout as exc:
        return _make_result(
            run_id=run_id,
            endpoint=endpoint,
            mode=mode,
            plan=plan,
            start_ts=start_ts,
            end_ts=clock.offset(),
            scheduled_arrival_offset_s=scheduled_arrival_offset_s,
            error_type="timeout",
            error=str(exc),
        )
    except requests.RequestException as exc:
        return _make_result(
            run_id=run_id,
            endpoint=endpoint,
            mode=mode,
            plan=plan,
            start_ts=start_ts,
            end_ts=clock.offset(),
            scheduled_arrival_offset_s=scheduled_arrival_offset_s,
            error_type="request_exception",
            error=str(exc),
        )


def _run_stream_request(
    base_url: str,
    endpoint: str,
    timeout_seconds: float,
    run_id: str,
    mode: str,
    plan: RequestPlan,
    clock: RunClock,
    *,
    scheduled_arrival_offset_s: float | None = None,
) -> RequestResult:
    start_ts = clock.offset()
    acc = StreamAccumulator(now=clock.offset)

    def _client_ttft_ms() -> float | None:
        if acc.first_token_offset_s is None:
            return None
        return (acc.first_token_offset_s - start_ts) * 1000.0

    def _error_result(
        error_type: str,
        error: str | None,
        end_ts: float,
        http_status: int | None,
    ) -> RequestResult:
        output_tokens, source = acc.resolve_output_tokens()
        return _make_result(
            run_id=run_id,
            endpoint=endpoint,
            mode=mode,
            plan=plan,
            start_ts=start_ts,
            end_ts=end_ts,
            scheduled_arrival_offset_s=scheduled_arrival_offset_s,
            first_token_ts=acc.first_token_offset_s,
            ttft_ms=_client_ttft_ms(),
            output_tokens=output_tokens,
            prompt_tokens=acc.server_prompt_tokens,
            tokens_per_s=None,
            server_total_ms=acc.server_total_ms,
            server_queue_wait_ms=acc.server_queue_wait_ms,
            server_execution_ms=acc.server_execution_ms,
            server_ttft_ms=acc.server_ttft_ms,
            output_sha256=acc.output_sha256,
            output_tokens_source=source,
            http_status=http_status,
            error_type=error_type,
            error=error,
            response_text_chars=acc.response_text_chars,
        )

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
                acc.feed(chunk)
                if acc.error is not None:
                    return _error_result(
                        error_type="server_error",
                        error=acc.error,
                        end_ts=clock.offset(),
                        http_status=response.status_code,
                    )
                if acc.done:
                    end_ts = clock.offset()
                    client_http_ms = (end_ts - start_ts) * 1000.0
                    client_ttft_ms = _client_ttft_ms()
                    output_tokens, source = acc.resolve_output_tokens()
                    tokens_per_s = acc.server_tokens_per_s
                    if tokens_per_s is None and output_tokens > 0:
                        # Match the server's definition: a decode rate over the
                        # execution span, not one polluted by queue wait.
                        tokens_per_s = _rate(
                            output_tokens,
                            (
                                acc.server_execution_ms
                                if acc.server_execution_ms is not None
                                else client_http_ms
                            )
                            / 1000.0,
                        )
                    return _make_result(
                        run_id=run_id,
                        endpoint=endpoint,
                        mode=mode,
                        plan=plan,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        scheduled_arrival_offset_s=scheduled_arrival_offset_s,
                        first_token_ts=acc.first_token_offset_s,
                        ttft_ms=client_ttft_ms,
                        tpot_ms=_tpot_ms(client_http_ms, client_ttft_ms, output_tokens),
                        output_tokens=output_tokens,
                        prompt_tokens=acc.server_prompt_tokens,
                        tokens_per_s=tokens_per_s,
                        server_total_ms=acc.server_total_ms,
                        server_queue_wait_ms=acc.server_queue_wait_ms,
                        server_execution_ms=acc.server_execution_ms,
                        server_ttft_ms=acc.server_ttft_ms,
                        server_tpot_ms=_tpot_ms(
                            acc.server_execution_ms,
                            acc.server_decode_start_ms(),
                            output_tokens,
                        ),
                        output_sha256=acc.output_sha256,
                        output_tokens_source=source,
                        http_status=response.status_code,
                        ok=True,
                        response_text_chars=acc.response_text_chars,
                    )

        return _error_result(
            error_type="protocol_error",
            error="stream ended before terminal event",
            end_ts=clock.offset(),
            http_status=200,
        )
    except StreamProtocolError as exc:
        return _error_result(
            error_type="protocol_error",
            error=str(exc),
            end_ts=clock.offset(),
            http_status=None,
        )
    except requests.Timeout as exc:
        return _error_result(
            error_type="timeout",
            error=str(exc),
            end_ts=clock.offset(),
            http_status=None,
        )
    except requests.RequestException as exc:
        return _error_result(
            error_type="request_exception",
            error=str(exc),
            end_ts=clock.offset(),
            http_status=None,
        )


class RequestRunner(Protocol):
    def __call__(
        self,
        base_url: str,
        endpoint: str,
        timeout_seconds: float,
        run_id: str,
        mode: str,
        plan: RequestPlan,
        clock: RunClock,
        *,
        scheduled_arrival_offset_s: float | None = None,
    ) -> RequestResult: ...


def _request_runner(endpoint: str) -> RequestRunner:
    if "stream" in endpoint:
        return _run_stream_request
    return _run_sync_request
