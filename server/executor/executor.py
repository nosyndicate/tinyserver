from typing import TypeVar

import torch

from server.executor.types import (
    BaseBatchExecutor,
    BaseExecutor,
    DoneEvent,
    ErrorEvent,
    FinishReason,
    GenerationRequestState,
    RequestStatus,
    TokenEvent,
)
from server.metrics.timers import now_ns, ns_to_ms
from server.model.hf_runner import ModelRunner

T = TypeVar("T")


def assert_not_none(value: T | None) -> T:
    if value is None:
        raise ValueError("Expected value to be not None")
    return value


def _handle_error(request_state: GenerationRequestState, error: Exception) -> None:
    request_state.status = RequestStatus.FAILED
    request_state.error = str(error)
    request_state.output_queue.put(
        ErrorEvent(request_id=request_state.request_id, error=str(error))
    )


def _finish(request_state: GenerationRequestState) -> None:
    """
    Finalize the request and emit a DoneEvent.

    Precondition: start_ns must be set (in prefill before decode is called).
    """
    request_state.status = RequestStatus.DONE
    end_ns = now_ns()

    # start_ns is always set by prefill() before _finish() can be called.
    if request_state.start_ns is None:
        raise RuntimeError("start_ns must be set before _finish()")

    if request_state.enqueued_ns is None:
        raise RuntimeError("enqueued_ns must be set before _finish()")

    total_ms = ns_to_ms(end_ns - request_state.start_ns)
    # queue_wait_ms is the time from when the request was enqueued to when execution started.
    # We max with 0 to avoid negative queue wait time in cases where the clock is not perfectly
    # monotonic or if there are any timing anomalies.
    queue_wait_ms = max(
        ns_to_ms(request_state.start_ns - request_state.enqueued_ns),
        0.0,
    )
    ttft_ms = (
        ns_to_ms(request_state.first_token_ns - request_state.start_ns)
        if request_state.first_token_ns is not None
        else -1.0
    )
    # execution_ms is the time spent executing (total_ms - queue_wait_ms).
    # We max with 0 to avoid negative execution time in cases where the clock
    # is not perfectly monotonic or if there are any timing anomalies.
    execution_ms = max(total_ms - queue_wait_ms, 0.0)

    if request_state.num_prompt_tokens is None:
        raise RuntimeError("num_prompt_tokens is required to finish the request")

    done_event = DoneEvent(
        text="".join(request_state.output_tokens),
        num_prompt_tokens=request_state.num_prompt_tokens,
        num_output_tokens=request_state.num_output_tokens,
        ttft=ttft_ms,
        total_ms=total_ms,
        queue_wait_ms=queue_wait_ms,
        execution_ms=execution_ms,
    )
    request_state.output_queue.put(done_event)


def _sample_and_emit(
    runner: ModelRunner,
    request_state: GenerationRequestState,
) -> int | None:
    """Sample the next token and emit a TokenEvent.

    Returns the token ID if generation should continue, or None if the request is finished.
    """
    logits = request_state.all_logits[:, -1, :]
    next_token_id = runner.sample_token(
        logits, request_state.sampling_params, request_state.generator
    )

    is_first = request_state.num_output_tokens == 0
    if is_first:
        request_state.first_token_ns = now_ns()

    if next_token_id == runner.eos_token_id:
        request_state.output_queue.put(
            TokenEvent(
                token="",
                is_first=is_first,
                is_last=True,
                index=request_state.num_output_tokens,
            )
        )
        request_state.finished_reason = FinishReason.EOS
        _finish(request_state)
        return None

    next_token = runner.tokenizer.decode([next_token_id], skip_special_tokens=True)
    is_last = (
        request_state.num_output_tokens + 1
        >= request_state.sampling_params.max_new_tokens
    )

    request_state.output_tokens.append(next_token)
    request_state.output_queue.put(
        TokenEvent(
            token=next_token,
            is_first=is_first,
            is_last=is_last,
            index=request_state.num_output_tokens - 1,
        )
    )

    if is_last:
        request_state.finished_reason = FinishReason.MAX_LENGTH
        _finish(request_state)
        return None

    return next_token_id


class Executor(BaseExecutor):
    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner

    def prefill(self, request_state: GenerationRequestState) -> None:
        request_state.status = RequestStatus.PREFILLING
        request_state.start_ns = now_ns()
        try:
            all_logits, past_key_values, num_input_toks = self._runner.prefill(
                request_state.prompt
            )
            request_state.all_logits = all_logits
            request_state.past_key_values = past_key_values
            request_state.num_prompt_tokens = num_input_toks
            request_state.status = RequestStatus.DECODING
        except Exception as e:
            _handle_error(request_state, e)

    @torch.inference_mode()
    def decode(self, request_state: GenerationRequestState) -> None:
        try:
            if request_state.all_logits is None:
                raise ValueError("No logits available for decoding step")

            next_token_id = _sample_and_emit(self._runner, request_state)
            if next_token_id is None:
                return

            next_input_id = torch.tensor(
                [[next_token_id]], device=request_state.all_logits.device
            )
            output = self._runner.model(
                next_input_id,
                past_key_values=request_state.past_key_values,
                use_cache=True,
            )
            request_state.all_logits = output.logits
            request_state.past_key_values = output.past_key_values
        except Exception as e:
            _handle_error(request_state, e)


class BatchExecutor(BaseBatchExecutor):
    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner

    @torch.inference_mode()
    def batched_prefill(self, request_states: list[GenerationRequestState]) -> None:
        current_time_ns = now_ns()
        for request_state in request_states:
            request_state.status = RequestStatus.PREFILLING
            request_state.start_ns = current_time_ns
        try:
            prefill_batch_outputs = self._runner.prefill_batch(
                [request_state.prompt for request_state in request_states]
            )

            if len(prefill_batch_outputs) != len(request_states):
                raise ValueError(
                    f"Expected {len(request_states)} prefill outputs, but got {len(prefill_batch_outputs)}"
                )

            for request_state, prefill_output in zip(
                request_states, prefill_batch_outputs
            ):
                request_state.all_logits = prefill_output.logits
                request_state.past_key_values = prefill_output.past_key_values
                request_state.num_prompt_tokens = prefill_output.num_prompt_tokens
                request_state.status = RequestStatus.DECODING

        except Exception as e:
            for request_state in request_states:
                _handle_error(request_state, e)

    @torch.inference_mode()
    def batched_decode(self, request_states: list[GenerationRequestState]) -> None:
        unfinished_request_states: list[tuple[GenerationRequestState, int]] = []
        for request_state in request_states:
            try:
                if request_state.all_logits is None:
                    raise ValueError("No logits available for decoding step")
                if request_state.past_key_values is None:
                    raise ValueError("No past_key_values available for decoding step")

                next_token_id = _sample_and_emit(self._runner, request_state)
                if next_token_id is not None:
                    unfinished_request_states.append((request_state, next_token_id))
            except Exception as e:
                _handle_error(request_state, e)

        if not unfinished_request_states:
            return

        next_input_ids = [token_id for _, token_id in unfinished_request_states]
        past_key_values = [
            assert_not_none(request_state.past_key_values)
            for request_state, _ in unfinished_request_states
        ]

        try:
            decode_outputs = self._runner.decode_batch(
                next_input_ids,
                past_key_values,
            )

            if len(decode_outputs) != len(unfinished_request_states):
                raise ValueError(
                    f"Expected {len(unfinished_request_states)} decode outputs, but got {len(decode_outputs)}"
                )

            for (request_state, _), decode_output in zip(
                unfinished_request_states, decode_outputs
            ):
                request_state.all_logits = decode_output.logits
                request_state.past_key_values = decode_output.past_key_values

        except Exception as e:
            for request_state, _ in unfinished_request_states:
                _handle_error(request_state, e)
