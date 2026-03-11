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
from server.model.determinism import make_generator
from server.model.sampling import SamplingParams


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3-1.7B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32


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

        if sampling_params.seed is not None:
            torch.manual_seed(sampling_params.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sampling_params.seed)

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        do_sample = sampling_params.temperature > 0
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=sampling_params.max_new_tokens,
            temperature=max(sampling_params.temperature, 1e-5) if do_sample else None,
            top_p=sampling_params.top_p if do_sample else None,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_ids = outputs[0][prompt_tokens:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        out_text = generated_text
        if sampling_params.stops:
            out_text = _apply_stop_strings(generated_text, sampling_params.stops)  # type: ignore[arg-type]

        output_tokens = int(outputs.shape[1]) - prompt_tokens
        return out_text, prompt_tokens, output_tokens  # type: ignore[return-value]

    @torch.inference_mode()
    def generate_text_two_stage(
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
        if sampling_params.stops:
            out_text = _apply_stop_strings(out_text, sampling_params.stops)  # type: ignore[arg-type]

        out_ids = self.tokenizer(
            [out_text], return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        token_counter = int(out_ids.shape[1])
        return out_text, prompt_tokens, token_counter

    @torch.inference_mode()
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

        yield from self.decode_loop(
            all_logits, past_key_values, sampling_params, generator=generator
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
                - is_done indicates if generation should stop (either due to EOS token, stop strings, or max tokens reached).

        Returns:
            None
        """
        token_counter = 0
        stops = sampling_params.stops or []
        max_stop_len = max((len(stop) for stop in stops), default=0)
        tail = ""

        last_logits = all_logits[:, -1, :]  # shape [1, vocab_size]

        for _ in range(sampling_params.max_new_tokens):
            # 1. sample the next token ID from the logits
            next_token_id = self.sample_token(
                last_logits, sampling_params, generator=generator
            )

            # 2. if the next token is EOS, we stop generation
            if next_token_id == self.tokenizer.eos_token_id:
                yield "", token_counter == 0, True
                return

            # 3. decode the next token ID to text
            next_token = self.tokenizer.decode(
                [next_token_id], skip_special_tokens=True
            )

            # 4. stop detection only needs the previous suffix + current token.
            if stops:
                candidate = tail + next_token
                if any(stop in candidate for stop in stops):
                    yield next_token, token_counter == 0, True
                    return

                if max_stop_len > 1:
                    tail = candidate[-(max_stop_len - 1) :]
                else:
                    tail = ""

            # 5. if we don't stop then we yield the next token and continue
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
        torch_dtype=config.dtype,
        device_map="auto" if config.device == "cuda" else None,
    )

    model.eval()
    log_event("model_init_done", model=config.model_name_or_path)
    return ModelRunner(model=model, tokenizer=tokenizer, device=config.device)


def _apply_stop_strings(text: str, stop_strings: list[str]) -> str:
    """Apply stop strings to the generated text."""
    cut = None
    for s in stop_strings:
        idx = text.find(s)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)
    return text if cut is None else text[:cut]
