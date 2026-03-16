from dataclasses import dataclass


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
