import torch

from server.executor.types import (
    DoneEvent,
    ErrorEvent,
    GenerationRequestState,
    RequestStatus,
    TokenEvent,
)
from server.metrics.timers import now_ns, ns_to_ms
from server.model.hf_runner import ModelRunner


class Executor:

    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner

    def prefill(self, request_state: GenerationRequestState) -> None:
        """
        Run the prefill step for the given request state.
        This will update the request state with the initial logits and past key values.
        """

        request_state.status = RequestStatus.PREFILLING
        try:
            all_logits, past_key_values, num_input_toks = self._runner.prefill(
                request_state.prompt
            )
            request_state.all_logits = all_logits
            request_state.past_key_values = past_key_values
            request_state.output_tokens = num_input_toks

            request_state.status = RequestStatus.DECODING
        except Exception as e:
            # if any error happens during prefill, we mark the request as failed
            # and notify the handler via ErrorEvent
            request_state.status = RequestStatus.FAILED
            request_state.error = str(e)
            request_state.output_queue.put(
                ErrorEvent(request_id=request_state.request_id, error=str(e))
            )

    def decode_one_step(self, request_state: GenerationRequestState) -> None:
        try:
            assert (
                request_state.all_logits is not None
            ), "all_logits should not be None during decoding"
            logits = request_state.all_logits[:, -1, :]
            next_token_id = self._runner.sample_token(
                logits, request_state.sampling_params, request_state.generator
            )

            if next_token_id == self._runner.eos_token_id:
                is_first = request_state.output_tokens == 0
                if is_first:
                    request_state.first_token_ns = now_ns()

                request_state.output_queue.put(
                    TokenEvent(
                        token="",  # no more token, but we can still send an empty event to indicate eos
                        is_first=is_first,
                        is_last=True,
                        index=request_state.output_tokens,
                    )
                )

                request_state.finished_reason = "eos"
                self._finish(request_state)
                return

            # decode the token
            next_token = self._runner.tokenizer.decode(
                [next_token_id], skip_special_tokens=True
            )

            is_first = request_state.output_tokens == 0
            if is_first:
                request_state.first_token_ns = now_ns()

            is_last = (
                request_state.output_tokens + 1
                >= request_state.sampling_params.max_new_tokens
            )

            request_state.output_queue.put(
                TokenEvent(
                    token=next_token,
                    is_first=is_first,
                    is_last=is_last,
                    index=request_state.output_tokens,
                )
            )
            request_state.output_tokens += 1

            if is_last:
                request_state.finished_reason = "max_length"
                self._finish(request_state)
                return

            # update the request state for the next decoding step
            next_input_id = torch.tensor([[next_token_id]], device=logits.device)
            output = self._runner.model(
                next_input_id,
                past_key_values=request_state.past_key_values,
                use_cache=True,
            )

            request_state.all_logits = output.logits
            request_state.past_key_values = output.past_key_values

        except Exception as e:
            # if any error happens during decoding, we mark the request as failed
            # and notify the handler via ErrorEvent
            request_state.status = RequestStatus.FAILED
            request_state.error = str(e)
            request_state.output_queue.put(
                ErrorEvent(request_id=request_state.request_id, error=str(e))
            )

    def _finish(self, request_state: GenerationRequestState) -> None:
        """Finalize a completed request: compute metrics and send Done Event"""

        request_state.status = RequestStatus.DONE
        end_ns = now_ns()
        total_ms = (
            ns_to_ms(end_ns - request_state.start_ns)
            if request_state.start_ns
            else -1.0
        )

        ttft_ms = (
            ns_to_ms(request_state.first_token_ns - request_state.start_ns)
            if request_state.first_token_ns and request_state.start_ns
            else -1.0
        )

        request_state.output_queue.put(
            DoneEvent(
                text="",  # TODO: add the text here
                num_output=request_state.output_tokens,
                ttft=ttft_ms,
                total_ms=total_ms,
            )
        )
