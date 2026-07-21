from dataclasses import dataclass

import torch

LOWEST_TEMPERATURE = 1e-5


@dataclass(frozen=True)
class SamplingParams:
    """Sampling parameters for text generation."""

    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int = 0
    seed: int | None = None


def build_sampling_params(
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int = 0,
    seed: int | None = None,
) -> SamplingParams:
    """Builds a SamplingParams object from the given parameters."""
    return SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )


@torch.inference_mode()
def top_p_sampling(
    logits: torch.Tensor,
    sampling_params: SamplingParams,
    generator: torch.Generator | None = None,
) -> int:
    # Sort logits descending so we can compute cumulative probability mass.
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)

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
    scaled_logits = logits.masked_fill(remove_mask, float("-inf"))

    probs = torch.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(next_token.item())


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

    if sampling_params.top_p < 1.0:
        # Apply top-p (nucleus) filtering when top_p < 1.0.
        # The fast path (no top-p filtering) is the implicit else case when top_p == 1.0.
        return top_p_sampling(scaled_logits, sampling_params, generator=generator)

    return int(
        torch.multinomial(
            torch.softmax(scaled_logits, dim=-1), num_samples=1, generator=generator
        ).item()
    )


def _check_param(name: str, tensor: torch.Tensor, batch_size: int) -> None:
    if tensor.shape != (batch_size,):
        raise ValueError(
            f"Expected {name} shape [{batch_size}], got {tuple(tensor.shape)}"
        )


@torch.inference_mode()
def sample_tokens(
    logits: torch.Tensor,  # [B, V]
    temperature: torch.Tensor,  # [B]
    top_k: torch.Tensor,  # [B], 0 disables
    top_p: torch.Tensor,  # [B], 1.0 disables
    uniforms: torch.Tensor,  # [B] in [0, 1)
) -> torch.Tensor:
    """Sample one token per row, in one batched pass, with no GPU->CPU sync.

    Filtering order is top-k first, then top-p over the survivors, matching
    vLLM/SGLang/HF.  All randomness enters through `uniforms`, so the function is
    pure: identical arguments always produce identical outputs, and a row's token
    never depends on who else is in the batch, avoiding the batch variant issue
    identified from https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

    Args:
        logits: [B, vocab_size] raw logits for the next token of each sequence.
        temperature: per-row temperature; rows <= 0 are greedy (plain argmax).
        top_k: per-row top-k cutoff; 0 (or >= vocab_size) disables it.
        top_p: per-row nucleus threshold; 1.0 disables it.
        uniforms: per-row uniform noise in [0, 1), typically from
            `determinism.uniforms_from_seeds`.

    Returns:
        [B] int64 tensor of sampled token ids, on the same device as `logits`.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits shape [B, vocab_size], got {logits.ndim}D")

    batch_size, vocab_size = logits.shape
    _check_param("temperature", temperature, batch_size)
    _check_param("top_k", top_k, batch_size)
    _check_param("top_p", top_p, batch_size)
    _check_param("uniforms", uniforms, batch_size)

    # Per-row temperature scaling, in float32 for stable softmax/cumsum math.
    scaled_logits = logits.float() / temperature.float().clamp(
        min=LOWEST_TEMPERATURE
    ).unsqueeze(-1)

    # stable=True makes ties break by index, which is the one place this filter
    # math could otherwise be nondeterministic on CUDA.
    sorted_logits, sorted_indices = torch.sort(
        scaled_logits, dim=-1, descending=True, stable=True
    )

    # top-k: drop everything ranked at or beyond k. top_k == 0 and top_k >= V
    # both no-op through this same expression, so there is no per-row branching.
    ranks = torch.arange(vocab_size, device=logits.device).unsqueeze(0)
    effective_k = torch.where(top_k > 0, top_k, vocab_size).unsqueeze(-1)
    sorted_logits = sorted_logits.masked_fill(ranks >= effective_k, float("-inf"))

    sorted_probs = torch.softmax(sorted_logits, dim=-1)

    # top-p: drop tokens whose cumulative mass *before* them already exceeds
    # top_p. Column 0 is always kept, so no row can end up empty.
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = (cumulative - sorted_probs) > top_p.unsqueeze(-1)
    remove[:, 0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)

    # Inverse-CDF pick. Scaling the uniform by the total surviving mass
    # renormalizes implicitly, so no separate divide is needed.
    cdf = torch.cumsum(sorted_probs, dim=-1)
    targets = (uniforms.float().unsqueeze(-1) * cdf[:, -1:]).contiguous()
    picked_rank = torch.searchsorted(cdf.contiguous(), targets).clamp(
        max=vocab_size - 1
    )

    # Column 0 of the stably-sorted order is the argmax, with ties broken to the
    # lowest index -- exactly what torch.argmax returns. So greedy rows cost no
    # extra kernel.
    greedy_rank = torch.zeros_like(picked_rank)
    rank = torch.where(temperature.unsqueeze(-1) <= 0, greedy_rank, picked_rank)

    return sorted_indices.gather(-1, rank).squeeze(-1)


def top_p_sample_rejection(
    logits: torch.Tensor,  # [B, V]
    top_p: torch.Tensor,  # [B]
    seeds: torch.Tensor,  # [B]
    max_rounds: int = 100,
) -> torch.Tensor:
    """
    Sample from logits using top-p rejection sampling (sorting-free).

    Algorithm outline:
        1. Convert logits -> probabilities (softmax).
        2. Set pivot = 0 (all tokens initially valid).
        3. Run inverse transform sampling over tokens with prob >= pivot.
        4. After sampling token j, set pivot = prob[j].
        5. Compute q = sum of probs for tokens with prob >= pivot.
           - If q < top_p: accept the token (we've narrowed enough).
           - If q >= top_p: reject and repeat from step 3.
        6. Repeat until accepted.

    Args:
        logits: [B, V] raw logits from the model
        top_p:  nucleus threshold (per-row tensor)
        seeds: RNG seeds (per-row tensor)
        max_rounds: safety cap on rejection rounds

    Returns:
        [B] tensor of sampled token indices
    """
    from kernels.top_p_sampling import rejection_sample_round_kernel, softmax_kernel

    B, V = logits.shape
    device = logits.device

    # Softmax: logits -> probs
    probs = torch.empty(B, V, dtype=torch.float32, device=device)
    BLOCK_SIZE = 1024
    softmax_kernel[(B,)](logits, probs, V, BLOCK_SIZE=BLOCK_SIZE)

    # Prepare per-row state
    top_p_tensor = top_p.to(dtype=torch.float32, device=device)
    seeds_tensor = seeds.to(dtype=torch.int32, device=device)

    pivot = torch.zeros(B, dtype=torch.float32, device=device)
    output = torch.zeros(B, dtype=torch.int32, device=device)
    accepted = torch.zeros(B, dtype=torch.int32, device=device)

    # RNG seeds — use random initial seeds
    # Iteratively launch rounds until all rows are accepted
    for round_idx in range(max_rounds):
        rejection_sample_round_kernel[(B,)](
            probs,
            pivot,
            top_p_tensor,
            output,
            accepted,
            seeds_tensor,
            round_idx,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # This cause CUDA sync per round
        # TODO consider skip some rounds and only check accepted every N rounds to reduce sync overhead
        if accepted.all():
            break

    # For any rows that didn't converge, fall back to argmax
    not_accepted = accepted == 0
    if not_accepted.any():
        fallback = probs[not_accepted].argmax(dim=-1).to(torch.int32)
        output[not_accepted] = fallback

    return output.long()
