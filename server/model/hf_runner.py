from dataclasses import dataclass
from typing import Generator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from server.metrics.logging import log_event
from server.model.batch_ops import (
    DecodeBatchOutput,
    PrefillBatchOutput,
    batched_decode_forward,
    batched_prefill,
)
from server.model.determinism import make_generator
from server.model.sampling import LOWEST_TEMPERATURE, SamplingParams, sample_token
from server.model.types import FinishReason, ModelConfig


@dataclass(frozen=True)
class GenerationStep:
    """One decode result with token identity preserved independently of text."""

    token_str: str
    index: int | None
    output_tokens: int
    finish_reason: FinishReason | None = None
    prompt_tokens: int | None = None

    @property
    def is_token(self) -> bool:
        return self.index is not None

    @property
    def is_done(self) -> bool:
        return self.finish_reason is not None


class ModelRunner:
    def __init__(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, device: str
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def generate_text(
        self, prompt: str, sampling_params: SamplingParams
    ) -> tuple[str, int, int]:
        """Generate text using a two-stage approach with prefill and decode loop."""
        if (
            sampling_params.temperature is not None
            and sampling_params.temperature > LOWEST_TEMPERATURE
        ):
            generator = make_generator(sampling_params.seed, self.device)
        else:
            generator = None

        all_logits, past_key_values, prompt_tokens = self.prefill(prompt)

        output_tokens = 0
        tokens: list[str] = []
        for step in self.decode_loop(
            all_logits,
            past_key_values,
            sampling_params,
            generator=generator,
            prompt_tokens=prompt_tokens,
        ):
            if step.is_token:
                tokens.append(step.token_str)

            output_tokens = step.output_tokens
            if step.is_done:
                break

        out_text = "".join(tokens)
        return out_text, prompt_tokens, output_tokens

    def generate_stream(
        self, prompt: str, sampling_params: SamplingParams
    ) -> Generator[GenerationStep, None, None]:
        """Generate text as a stream of tokens using the two-stage approach."""
        if (
            sampling_params.temperature is not None
            and sampling_params.temperature > LOWEST_TEMPERATURE
        ):
            generator = make_generator(sampling_params.seed, self.device)
        else:
            generator = None

        all_logits, past_key_values, prompt_tokens = self.prefill(prompt)

        # We cannot use @torch.inference_mode() on this generator function, because a decorator
        # would only wrap creation of the generator object, not the subsequent iteration.
        # Using a context manager here keeps inference_mode active while we iterate and yield tokens.
        with torch.inference_mode():
            yield from self.decode_loop(
                all_logits,
                past_key_values,
                sampling_params,
                generator=generator,
                prompt_tokens=prompt_tokens,
            )

    @torch.inference_mode()
    def prefill(self, prompt: str) -> tuple[torch.Tensor, DynamicCache, int]:
        """Run the model on the prompt to get initial logits and past_key_values for decoding.

        This method processes the input prompt by applying the chat template and tokenizing it,
        then runs a forward pass through the model to obtain the initial logits and past_key_values
        needed for the decoding loop.

        Args:
            prompt: The input prompt string.

        Returns:
            all_logits: Tensor of shape [1, prompt_len, vocab_size] containing the logits from the model for the input prompt.
            past_key_values: The past key values returned by the model, used for efficient decoding.
            prompt_tokens: The number of tokens in the input prompt after tokenization.
        """
        message = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer([formatted], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, use_cache=True)

        past_key_values: DynamicCache = outputs.past_key_values
        all_logits: torch.Tensor = outputs.logits  # shape [1, prompt_len, vocab_size]
        prompt_tokens = int(inputs["input_ids"].shape[1])
        return all_logits, past_key_values, prompt_tokens

    def prefill_batch(self, prompts: list[str]) -> list[PrefillBatchOutput]:
        """Run prefill for a batch of prompts and return the outputs."""
        prefill_batch_outputs = batched_prefill(
            self.model, self.tokenizer, prompts, self.device
        )
        return prefill_batch_outputs

    def decode_batch(
        self,
        token_ids: list[int],
        past_key_values: list[DynamicCache],
    ) -> list[DecodeBatchOutput]:
        return batched_decode_forward(
            self.model, token_ids, past_key_values, device=self.device
        )

    def decode_loop(
        self,
        all_logits: torch.Tensor,
        past_key_values: DynamicCache,
        sampling_params: SamplingParams,
        generator: torch.Generator | None = None,
        prompt_tokens: int | None = None,
    ) -> Generator[GenerationStep, None, None]:
        """
        Yield structured decode steps that distinguish tokens from completion.

        Args:
            all_logits: Tensor of shape [1, seq_len, vocab_size] containing the logits for the current sequence.
            past_key_values: The past key values from the model, used for efficient decoding.
            sampling_params: SamplingParams object containing the parameters for sampling.
            generator: Optional torch.Generator for reproducible sampling. If None, sampling will be non-deterministic.

        A non-EOS sample always has an index, even if it decodes to an empty
        string. EOS has no index and does not advance ``output_tokens``.

        Returns:
            None
        """
        token_counter = 0

        last_logits = all_logits[:, -1, :]  # shape [1, vocab_size]

        for _ in range(sampling_params.max_new_tokens):
            # 1. sample the next token ID from the logits
            next_token_id = sample_token(
                last_logits, sampling_params, generator=generator
            )

            # 2. if the next token is EOS, we stop generation
            if next_token_id == self.eos_token_id:
                yield GenerationStep(
                    token_str="",
                    index=None,
                    output_tokens=token_counter,
                    finish_reason=FinishReason.EOS,
                    prompt_tokens=prompt_tokens,
                )
                return

            # 3. decode the next token ID to text
            next_token = self.tokenizer.decode(
                [next_token_id], skip_special_tokens=True
            )

            # 4. yield the next token and continue
            index = token_counter
            token_counter += 1
            is_last = token_counter >= sampling_params.max_new_tokens
            yield GenerationStep(
                token_str=next_token,
                index=index,
                output_tokens=token_counter,
                finish_reason=FinishReason.MAX_LENGTH if is_last else None,
                prompt_tokens=prompt_tokens,
            )

            if is_last:
                return

            next_input_ids = torch.tensor(
                [[next_token_id]], device=self.model.device
            )  # shape [1, 1]
            output = self.model(
                input_ids=next_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            last_logits = output.logits[:, -1, :]  # shape [1, vocab_size]
            past_key_values = output.past_key_values

        # This is reachable only for a direct caller using max_new_tokens=0;
        # the HTTP schema requires at least one output token.
        yield GenerationStep(
            token_str="",
            index=None,
            output_tokens=token_counter,
            finish_reason=FinishReason.MAX_LENGTH,
            prompt_tokens=prompt_tokens,
        )

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id


def load_hf_model(config: ModelConfig) -> ModelRunner:
    """Load HF model/tokenizer and return a ready ModelRunner."""
    log_event(
        "model_init_start",
        model=config.model_name_or_path,
        device=config.device,
        dtype=str(config.dtype),
    )

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        config.model_name_or_path, use_fast=True
    )
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        dtype=config.dtype,
        device_map="auto" if config.device == "cuda" else None,
    )

    model.eval()
    log_event("model_init_done", model=config.model_name_or_path)
    return ModelRunner(model=model, tokenizer=tokenizer, device=config.device)
