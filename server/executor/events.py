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


class RequestEventEmitter:
    def on_prefill_started(
        self, request_state: GenerationRequestState, start_ns: int
    ) -> None:
        request_state.status = RequestStatus.PREFILLING
        request_state.start_ns = start_ns

    def on_prefill_succeeded(
        self, request_state: GenerationRequestState, result: PrefillResult
    ) -> None:
        request_state.all_logits = result.all_logits
        request_state.past_key_values = result.past_key_values
        request_state.num_prompt_tokens = result.num_prompt_tokens
        request_state.status = RequestStatus.DECODING

    def on_token(
        self, request_state: GenerationRequestState, result: DecodeResult
    ) -> None:
        is_first = request_state.num_output_tokens == 0
        if is_first:
            request_state.first_token_ns = now_ns()

        if result.finish_reason == FinishReason.EOS:
            request_state.output_queue.put(
                TokenEvent(
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
        request_state.output_queue.put(
            TokenEvent(
                token=result.token,
                is_first=is_first,
                is_last=result.is_last,
                index=request_state.num_output_tokens - 1,
            )
        )

        if result.finish_reason is not None:
            request_state.finished_reason = result.finish_reason
            self._finish(request_state)
            return

        request_state.all_logits = result.all_logits
        request_state.past_key_values = result.past_key_values
        request_state.status = RequestStatus.DECODING

    def on_failed(self, request_state: GenerationRequestState, error: str) -> None:
        request_state.status = RequestStatus.FAILED
        request_state.error = error
        request_state.output_queue.put(
            ErrorEvent(request_id=request_state.request_id, error=error)
        )

    def _finish(self, request_state: GenerationRequestState) -> None:
        request_state.status = RequestStatus.DONE
        end_ns = now_ns()

        if request_state.start_ns is None:
            raise RuntimeError("start_ns must be set before _finish()")

        if request_state.enqueued_ns is None:
            raise RuntimeError("enqueued_ns must be set before _finish()")

        total_ms = ns_to_ms(end_ns - request_state.start_ns)
        queue_wait_ms = max(
            ns_to_ms(request_state.start_ns - request_state.enqueued_ns),
            0.0,
        )
        ttft_ms = (
            ns_to_ms(request_state.first_token_ns - request_state.start_ns)
            if request_state.first_token_ns is not None
            else -1.0
        )
        execution_ms = max(total_ms - queue_wait_ms, 0.0)

        if request_state.num_prompt_tokens is None:
            raise RuntimeError("num_prompt_tokens is required to finish the request")

        request_state.output_queue.put(
            DoneEvent(
                text="".join(request_state.output_tokens),
                num_prompt_tokens=request_state.num_prompt_tokens,
                num_output_tokens=request_state.num_output_tokens,
                ttft=ttft_ms,
                total_ms=total_ms,
                queue_wait_ms=queue_wait_ms,
                execution_ms=execution_ms,
            )
        )
