import torch

from server.executor.types import (
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
            self._handle_error(request_state, e)

    @torch.inference_mode()
    def decode_one_step(self, request_state: GenerationRequestState) -> None:
        try:
            if request_state.all_logits is None:
                raise ValueError("No logits available for decoding step")

            logits = request_state.all_logits[:, -1, :]
            next_token_id = self._runner.sample_token(
                logits, request_state.sampling_params, request_state.generator
            )

            is_first = request_state.num_output_tokens == 0
            if is_first:
                request_state.first_token_ns = now_ns()

            if next_token_id == self._runner.eos_token_id:
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

            next_token = self._runner.tokenizer.decode(
                [next_token_id], skip_special_tokens=True
            )

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
                    index=request_state.num_output_tokens,
                )
            )

            if is_last:
                request_state.finished_reason = FinishReason.MAX_LENGTH
                self._finish(request_state)
                return

            next_input_id = torch.tensor([[next_token_id]], device=logits.device)
            output = self._runner.model(
                next_input_id,
                past_key_values=request_state.past_key_values,
                use_cache=True,
            )

            request_state.all_logits = output.logits
            request_state.past_key_values = output.past_key_values

        except Exception as e:
            self._handle_error(request_state, e)

    def _handle_error(
        self, request_state: GenerationRequestState, error: Exception
    ) -> None:
        request_state.status = RequestStatus.FAILED
        request_state.error = str(error)
        request_state.output_queue.put(
            ErrorEvent(request_id=request_state.request_id, error=str(error))
        )

    def _finish(self, request_state: GenerationRequestState) -> None:
        request_state.status = RequestStatus.DONE
        end_ns = now_ns()
        total_ms = (
            ns_to_ms(end_ns - request_state.start_ns)
            if request_state.start_ns is not None
            else -1.0
        )
        queue_wait_ms = max(
            ns_to_ms(request_state.start_ns - request_state.enqueued_ns)
            if request_state.start_ns is not None
            and request_state.enqueued_ns is not None
            else 0.0,
            0.0,
        )
        ttft_ms = (
            ns_to_ms(request_state.first_token_ns - request_state.start_ns)
            if request_state.first_token_ns is not None
            and request_state.start_ns is not None
            else -1.0
        )
        execution_ms = max(total_ms, 0.0)

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
