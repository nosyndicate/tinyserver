from typing import TypeVar

import torch

from server.executor.types import (
    BaseBatchExecutor,
    BaseExecutor,
    DecodeResult,
    FinishReason,
    GenerationRequestState,
    PrefillResult,
    RequestFailure,
)
from server.metrics.timers import now_ns
from server.model.hf_runner import ModelRunner

T = TypeVar("T")


def assert_not_none(value: T | None) -> T:
    if value is None:
        raise ValueError("Expected value to be not None")
    return value


def _sample(
    runner: ModelRunner,
    request_state: GenerationRequestState,
) -> DecodeResult:
    logits = request_state.all_logits[:, -1, :]
    next_token_id = runner.sample_token(
        logits, request_state.sampling_params, request_state.generator
    )

    if next_token_id == runner.eos_token_id:
        return DecodeResult(
            token_id=None,
            token="",
            is_last=True,
            finish_reason=FinishReason.EOS,
        )

    next_token = runner.tokenizer.decode([next_token_id], skip_special_tokens=True)
    is_last = (
        request_state.num_output_tokens + 1
        >= request_state.sampling_params.max_new_tokens
    )

    return DecodeResult(
        token_id=None if is_last else next_token_id,
        token=next_token,
        is_last=is_last,
        finish_reason=FinishReason.MAX_LENGTH if is_last else None,
    )


class Executor(BaseExecutor):
    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner

    def prefill(
        self, request_state: GenerationRequestState
    ) -> PrefillResult | RequestFailure:
        start_ns = now_ns()
        try:
            all_logits, past_key_values, num_input_toks = self._runner.prefill(
                request_state.prompt
            )
            return PrefillResult(
                all_logits=all_logits,
                past_key_values=past_key_values,
                num_prompt_tokens=num_input_toks,
                start_ns=start_ns,
            )
        except Exception as e:
            return RequestFailure(error=str(e))

    @torch.inference_mode()
    def decode(
        self, request_state: GenerationRequestState
    ) -> DecodeResult | RequestFailure:
        try:
            if request_state.all_logits is None:
                raise ValueError("No logits available for decoding step")

            result = _sample(self._runner, request_state)
            if result.token_id is None:
                return result

            next_input_id = torch.tensor(
                [[result.token_id]], device=request_state.all_logits.device
            )
            output = self._runner.model(
                next_input_id,
                past_key_values=request_state.past_key_values,
                use_cache=True,
            )
            return DecodeResult(
                token_id=result.token_id,
                token=result.token,
                is_last=result.is_last,
                finish_reason=result.finish_reason,
                all_logits=output.logits,
                past_key_values=output.past_key_values,
            )
        except Exception as e:
            return RequestFailure(error=str(e))


class BatchExecutor(BaseBatchExecutor):
    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner

    @torch.inference_mode()
    def batched_prefill(
        self, request_states: list[GenerationRequestState]
    ) -> list[PrefillResult | RequestFailure]:
        current_time_ns = now_ns()
        try:
            prefill_batch_outputs = self._runner.prefill_batch(
                [request_state.prompt for request_state in request_states]
            )

            if len(prefill_batch_outputs) != len(request_states):
                raise ValueError(
                    f"Expected {len(request_states)} prefill outputs, but got {len(prefill_batch_outputs)}"
                )

            return [
                PrefillResult(
                    all_logits=prefill_output.logits,
                    past_key_values=prefill_output.past_key_values,
                    num_prompt_tokens=prefill_output.num_prompt_tokens,
                    start_ns=current_time_ns,
                )
                for prefill_output in prefill_batch_outputs
            ]

        except Exception as e:
            return [RequestFailure(error=str(e)) for _ in request_states]

    @torch.inference_mode()
    def batched_decode(
        self, request_states: list[GenerationRequestState]
    ) -> list[DecodeResult | RequestFailure]:
        results: list[DecodeResult | RequestFailure | None] = [None] * len(
            request_states
        )
        unfinished_request_states: list[tuple[int, GenerationRequestState, int]] = []
        for i, request_state in enumerate(request_states):
            try:
                if request_state.all_logits is None:
                    raise ValueError("No logits available for decoding step")
                if request_state.past_key_values is None:
                    raise ValueError("No past_key_values available for decoding step")

                result = _sample(self._runner, request_state)
                if result.token_id is None:
                    results[i] = result
                else:
                    unfinished_request_states.append(
                        (i, request_state, result.token_id)
                    )
                    results[i] = result
            except Exception as e:
                results[i] = RequestFailure(error=str(e))

        if not unfinished_request_states:
            return [assert_not_none(result) for result in results]

        next_input_ids = [token_id for _, _, token_id in unfinished_request_states]
        past_key_values = [
            assert_not_none(request_state.past_key_values)
            for _, request_state, _ in unfinished_request_states
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

            for (index, _, _), decode_output in zip(
                unfinished_request_states, decode_outputs
            ):
                result = assert_not_none(results[index])
                if isinstance(result, RequestFailure):
                    continue
                results[index] = DecodeResult(
                    token_id=result.token_id,
                    token=result.token,
                    is_last=result.is_last,
                    finish_reason=result.finish_reason,
                    all_logits=decode_output.logits,
                    past_key_values=decode_output.past_key_values,
                )

        except Exception as e:
            for index, _, _ in unfinished_request_states:
                results[index] = RequestFailure(error=str(e))

        return [assert_not_none(result) for result in results]
