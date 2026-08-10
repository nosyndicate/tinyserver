from dataclasses import dataclass

from server.executor.types import (
    DecodeResult,
    DoneEvent,
    ErrorEvent,
    FinishReason,
    GenerationRequestState,
    PrefillResult,
    RequestStatus,
    TokenEvent,
)
from server.metrics.timers import now_ns, ns_to_ms


@dataclass(frozen=True)
class RequestTimings:
    """The four timing metrics reported for a completed request."""

    total_ms: float
    queue_wait_ms: float
    execution_ms: float
    ttft_ms: float | None


def compute_timings(
    *,
    enqueued_ns: int,
    start_ns: int,
    first_token_ns: int | None,
    end_ns: int,
) -> RequestTimings:
    """Derive the reported timings from the four raw request timestamps.

    Each metric is measured directly between two timestamps rather than derived
    from the others, so ``total_ms == queue_wait_ms + execution_ms`` holds to
    floating-point precision::

        enqueued_ns ---- queue_wait ----> start_ns ---- execution ----> end_ns
        |<------------------------- total ---------------------------->|

    ``start_ns`` is the start of the *first* prefill. v4 deliberately does not
    reset it when a preempted sequence resumes (see
    ``ScheduleInferenceEngine._post_prefill``), so time spent preempted is
    counted as execution rather than as queue wait — preemption is a cost of
    running, not of waiting to be admitted.

    The ``max(..., 0.0)`` guards are defensive only: ``enqueued_ns`` is stamped
    in ``Worker.submit`` before the request is queued and ``start_ns`` on the
    engine thread after it is dequeued, so the timestamps are already ordered.
    """
    return RequestTimings(
        total_ms=max(ns_to_ms(end_ns - enqueued_ns), 0.0),
        queue_wait_ms=max(ns_to_ms(start_ns - enqueued_ns), 0.0),
        execution_ms=max(ns_to_ms(end_ns - start_ns), 0.0),
        ttft_ms=(
            max(ns_to_ms(first_token_ns - enqueued_ns), 0.0)
            if first_token_ns is not None
            else None
        ),
    )


class RequestEventEmitter:
    """Translates executor results into events emitted to each request's sink.

    The engine calls the ``on_*`` methods as a request progresses through
    prefill and decode phases.  This class updates the mutable
    ``GenerationRequestState`` (timestamps, tokens, cache state) and emits
    ``TokenEvent``, ``DoneEvent``, or ``ErrorEvent`` instances through the
    request's ``EventSink`` so that downstream consumers (e.g., the HTTP/SSE
    layer) can read them.
    """

    def on_prefill_started(
        self, request_state: GenerationRequestState, start_ns: int
    ) -> None:
        """Record that prefill began and transition the request to PREFILLING."""
        request_state.status = RequestStatus.PREFILLING
        request_state.start_ns = start_ns

    def on_prefill_succeeded(
        self, request_state: GenerationRequestState, result: PrefillResult
    ) -> None:
        """Store prefill outputs and transition the request to DECODING."""
        request_state.all_logits = result.all_logits
        request_state.past_key_values = result.past_key_values
        request_state.num_prompt_tokens = result.num_prompt_tokens
        request_state.status = RequestStatus.DECODING

    def on_token(
        self, request_state: GenerationRequestState, result: DecodeResult
    ) -> None:
        """Handle a decode result: push a TokenEvent and finish if done.

        On EOS the token text is emitted as an empty string (the model's EOS
        token itself is not part of the output).  For all other finished
        reasons (e.g. max length), the final token text is included.
        """
        is_first = request_state.num_output_tokens == 0
        if is_first:
            request_state.first_token_ns = now_ns()

        if result.finish_reason == FinishReason.EOS:
            request_state.sink.emit(
                TokenEvent(
                    request_id=request_state.request_id,
                    token="",
                    is_first=is_first,
                    is_last=True,
                    index=request_state.num_output_tokens,
                )
            )
            request_state.finished_reason = FinishReason.EOS
            self._finish(request_state)
            return

        request_state.output_tokens.append(result.token)
        request_state.sink.emit(
            TokenEvent(
                request_id=request_state.request_id,
                token=result.token,
                is_first=is_first,
                is_last=result.is_finished,
                index=request_state.num_output_tokens - 1,
            )
        )

        if result.is_finished:
            request_state.finished_reason = result.finish_reason
            self._finish(request_state)
            return

        request_state.all_logits = result.all_logits
        request_state.past_key_values = result.past_key_values
        request_state.status = RequestStatus.DECODING

    def on_failed(self, request_state: GenerationRequestState, error: str) -> None:
        """Mark the request as failed and push an ErrorEvent."""
        request_state.status = RequestStatus.FAILED
        request_state.error = error
        request_state.sink.emit(
            ErrorEvent(request_id=request_state.request_id, error=error)
        )

    def _finish(self, request_state: GenerationRequestState) -> None:
        """Compute timing metrics and push a DoneEvent to complete the request."""
        request_state.status = RequestStatus.DONE
        end_ns = now_ns()

        if request_state.start_ns is None:
            raise RuntimeError("start_ns must be set before _finish()")

        if request_state.enqueued_ns is None:
            raise RuntimeError("enqueued_ns must be set before _finish()")

        if request_state.num_prompt_tokens is None:
            raise RuntimeError("num_prompt_tokens is required to finish the request")

        timings = compute_timings(
            enqueued_ns=request_state.enqueued_ns,
            start_ns=request_state.start_ns,
            first_token_ns=request_state.first_token_ns,
            end_ns=end_ns,
        )

        request_state.sink.emit(
            DoneEvent(
                request_id=request_state.request_id,
                text="".join(request_state.output_tokens),
                num_prompt_tokens=request_state.num_prompt_tokens,
                num_output_tokens=request_state.num_output_tokens,
                ttft_ms=timings.ttft_ms,
                total_ms=timings.total_ms,
                queue_wait_ms=timings.queue_wait_ms,
                execution_ms=timings.execution_ms,
            )
        )
