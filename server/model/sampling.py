from dataclasses import dataclass

import torch

LOWEST_TEMPERATURE = 1e-5


@dataclass(frozen=True)
class SamplingParams:
    """Sampling parameters for text generation."""

    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int | None = None


def build_sampling_params(
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None = None,
) -> SamplingParams:
    """Builds a SamplingParams object from the given parameters."""
    return SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )


@torch.inference_mode()
def sample_token(
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
        remove_mask_sorted = (cumulative_probs - sorted_probs) > sampling_params.top_p

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
