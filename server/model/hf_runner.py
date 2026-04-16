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
from server.model.sampling import SamplingParams
from server.model.types import ModelConfig

LOWEST_TEMPERATURE = 1e-5


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

        token_counter = 0
        tokens = []
        for next_token, _, is_done in self.decode_loop(
            all_logits, past_key_values, sampling_params, generator=generator
        ):
            if next_token:
                tokens.append(next_token)

            if is_done:
                break

        out_text = "".join(tokens)

        out_ids = self.tokenizer(
            [out_text], return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        token_counter = int(out_ids.shape[1])
        return out_text, prompt_tokens, token_counter

    def generate_stream(
        self, prompt: str, sampling_params: SamplingParams
    ) -> Generator[tuple[str, bool, bool], None, None]:
        """Generate text as a stream of tokens using the two-stage approach."""
        if (
            sampling_params.temperature is not None
            and sampling_params.temperature > LOWEST_TEMPERATURE
        ):
            generator = make_generator(sampling_params.seed, self.device)
        else:
            generator = None

        all_logits, past_key_values, _ = self.prefill(prompt)

        # We cannot use @torch.inference_mode() on this generator function, because a decorator
        # would only wrap creation of the generator object, not the subsequent iteration.
        # Using a context manager here keeps inference_mode active while we iterate and yield tokens.
        with torch.inference_mode():
            for token_str, is_first, is_done in self.decode_loop(
                all_logits, past_key_values, sampling_params, generator=generator
            ):
                yield token_str, is_first, is_done

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

    @torch.inference_mode()
    def sample_token(
        self,
        logits: torch.Tensor,
        sampling_params: SamplingParams,
        generator: torch.Generator | None = None,
    ) -> int:
        """Sample a token ID from the logits using the provided sampling parameters.

        Args:
            logits: Tensor of shape [1, vocab_size] containing the logits for the next token
            sampling_params: SamplingParams object containing temperature, top_p, etc.
            generator: Optional torch.Generator for reproducible sampling. If None, sampling will be non-deterministic.

        Returns:
            The sampled token ID as an integer.
        """
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(
                f"Expected logits shape [1, vocab_size], got {tuple(logits.shape)}"
            )

        if sampling_params.temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())

        if not (0.0 < sampling_params.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {sampling_params.top_p}")

        # Work in float32 for more stable sampling math.
        scaled_logits = logits.float() / max(
            sampling_params.temperature, LOWEST_TEMPERATURE
        )

        # Apply top-p (nucleus) filtering when top_p < 1.0.
        # The fast path (no top-p filtering) is the implicit else case when top_p == 1.0.
        if sampling_params.top_p < 1.0:
            # Sort logits descending so we can compute cumulative probability mass.
            sorted_logits, sorted_indices = torch.sort(
                scaled_logits, dim=-1, descending=True
            )

            # Convert sorted logits to sorted probabilities.
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens whose cumulative mass *before this token* already exceeds top_p.
            remove_mask_sorted = (
                cumulative_probs - sorted_probs
            ) > sampling_params.top_p

            # Always keep at least one token.
            remove_mask_sorted[..., 0] = False

            # Scatter the sorted removal mask back to original vocab order.
            remove_mask = torch.zeros_like(remove_mask_sorted, dtype=torch.bool)
            remove_mask.scatter_(dim=-1, index=sorted_indices, src=remove_mask_sorted)

            # Mask logits directly.
            scaled_logits = scaled_logits.masked_fill(remove_mask, float("-inf"))

        probs = torch.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        return int(next_token.item())

    def decode_loop(
        self,
        all_logits: torch.Tensor,
        past_key_values: DynamicCache,
        sampling_params: SamplingParams,
        generator: torch.Generator | None = None,
    ) -> Generator[tuple[str, bool, bool], None, None]:
        """
        A generator that yields the next token, whether it's the first token, and whether generation is done.

        Args:
            all_logits: Tensor of shape [1, seq_len, vocab_size] containing the logits for the current sequence.
            past_key_values: The past key values from the model, used for efficient decoding.
            sampling_params: SamplingParams object containing the parameters for sampling.
            generator: Optional torch.Generator for reproducible sampling. If None, sampling will be non-deterministic.

        Yields:
            A tuple of (next_token: str, is_first_token: bool, is_done: bool) where:
                - next_token is the decoded text of the next token ID. (Maybe empty string if the token
                  is a special token or if generation is done.)
                - is_first_token indicates if this is the first generated token (after the prompt).
                - is_done indicates if generation should stop (either due to EOS token or max tokens reached).

        Returns:
            None
        """
        token_counter = 0

        last_logits = all_logits[:, -1, :]  # shape [1, vocab_size]

        for _ in range(sampling_params.max_new_tokens):
            # 1. sample the next token ID from the logits
            next_token_id = self.sample_token(
                last_logits, sampling_params, generator=generator
            )

            # 2. if the next token is EOS, we stop generation
            if next_token_id == self.eos_token_id:
                yield "", token_counter == 0, True
                return

            # 3. decode the next token ID to text
            next_token = self.tokenizer.decode(
                [next_token_id], skip_special_tokens=True
            )

            # 4. yield the next token and continue
            is_last = token_counter == sampling_params.max_new_tokens - 1
            yield next_token, token_counter == 0, is_last  # type: ignore[misc]

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
            token_counter += 1

        # max tokens reached, we stop generation
        yield "", token_counter == 0, True

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
