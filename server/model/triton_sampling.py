"""Triton-optimized token sampling for LLM inference.

This module provides Triton kernels for token sampling. The implementation
supports both a fast path (no top-p filtering) and a full path with top-p
(nucleus) filtering.

The implementation maintains determinism by using a seeded RNG that incorporates
both the user-provided seed and the sequence position.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def sample_token_triton(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    seed: int | None,
    seq_pos: int,
) -> int | None:
    """Sample a token using Triton kernel with fallback to None on error.

    Args:
        logits: Tensor of shape [vocab_size] containing logits for next token.
            Must be on CUDA device.
        temperature: Sampling temperature. Must be > 0.
        top_p: Top-p (nucleus) filtering threshold in (0, 1].
        seed: Random seed for deterministic sampling. If None, uses 0.
        seq_pos: Sequence position for RNG variation (each token gets different RNG state).

    Returns:
        Sampled token ID as int, or None if Triton kernel fails (signal to use PyTorch fallback).
    """
    if not TRITON_AVAILABLE:
        return None

    if not torch.cuda.is_available():
        return None

    if logits.device.type != "cuda":
        return None

    if logits.ndim != 1:
        return None

    vocab_size = logits.shape[0]

    # Validate parameters
    if temperature <= 0:
        return None  # Greedy case handled separately

    if not (0.0 < top_p <= 1.0):
        return None

    try:
        # Allocate output tensor
        output = torch.empty(1, dtype=torch.int32, device=logits.device)

        # Ensure logits is contiguous and in float32 for numerical stability
        logits_contig = logits.float().contiguous()

        # Compute block size (must be power of 2 >= vocab_size)
        block_size = 1
        while block_size < vocab_size:
            block_size *= 2

        # Launch kernel
        grid = (1,)
        if top_p >= 1.0 - 1e-6:
            # Fast path: no top-p filtering needed
            sample_token_kernel_fast[grid](
                logits_contig,
                output,
                vocab_size,
                temperature,
                seed if seed is not None else 0,
                seq_pos,
                BLOCK_SIZE=block_size,
            )
        else:
            # Full path with top-p filtering
            sample_token_kernel_full[grid](
                logits_contig,
                output,
                vocab_size,
                temperature,
                top_p,
                seed if seed is not None else 0,
                seq_pos,
                BLOCK_SIZE=block_size,
            )

        return int(output.item())

    except Exception:
        # Any exception signals fallback to PyTorch implementation
        return None


@triton.jit
def _xorshift_rand(state):
    """XORShift PRNG step."""
    state = state ^ (state << 13)
    state = state ^ (state >> 17)
    state = state ^ (state << 5)
    return state


@triton.jit
def _generate_rand_float(seed, seq_pos):
    """Generate a random float in [0, 1) using XORShift-based PRNG.

    Args:
        seed: Base seed for the RNG.
        seq_pos: Sequence position to vary the RNG state.

    Returns:
        Random float in [0, 1).
    """
    # Combine seed and seq_pos for unique state per token position
    # Use a prime number that fits in int32 to mix seed and seq_pos
    state = seed + seq_pos * 1664525  # A prime number for good mixing

    # Ensure state is positive using tl.abs
    state = tl.abs(state)

    # Apply XORShift operations (multiple rounds for better distribution)
    state = _xorshift_rand(state)
    state = _xorshift_rand(state)
    state = _xorshift_rand(state)

    # Convert to float in [0, 1)
    # Use lower 23 bits for mantissa to get uniform distribution
    # Return value in [0, 1) by dividing by 2^24
    state_uint = state.to(tl.uint32)
    return state_uint.to(tl.float32) / 4294967296.0  # 2^32


@triton.jit
def sample_token_kernel_fast(
    logits_ptr,  # [vocab_size] logits in float32
    output_ptr,  # [1] output token id
    vocab_size,
    temperature,
    seed,
    seq_pos,
    BLOCK_SIZE: tl.constexpr,
):
    """Fast sampling kernel without top-p filtering.

    This kernel performs:
    1. Temperature scaling
    2. Softmax normalization
    3. Multinomial sampling using inverse transform sampling

    Args:
        logits_ptr: Pointer to logits tensor [vocab_size].
        output_ptr: Pointer to output tensor [1] for sampled token ID.
        vocab_size: Size of vocabulary.
        temperature: Sampling temperature.
        seed: Random seed for deterministic sampling.
        seq_pos: Sequence position for RNG variation.
        BLOCK_SIZE: Block size for parallel processing (must be >= vocab_size).
    """
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size

    # Load logits
    logits = tl.load(logits_ptr + offs, mask=mask, other=float("-inf"))

    # Temperature scaling
    scaled_logits = logits / temperature

    # Softmax: subtract max for numerical stability
    max_logit = tl.max(scaled_logits, axis=0)
    exp_logits = tl.exp(scaled_logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp

    # Multinomial sampling using inverse transform sampling
    rand_val = _generate_rand_float(seed, seq_pos)

    # Compute cumulative probabilities
    cumsum_probs = tl.cumsum(probs)

    # Find the first index where cumsum >= rand_val
    sampled_mask = cumsum_probs >= rand_val

    # Find first true using a scan: count how many False values come before each True
    # The first True will have count = 0
    sampled_mask_int = sampled_mask.to(tl.int32)

    # For each position, compute: sum of (1 - mask) for all positions <= this position
    # This gives us the count of False values up to and including this position
    # The first True will have count = 0 (no False values before it)
    # But we need: sum of (1 - mask) for all positions < this position
    # So we use: (1 - mask) * index, then take min where mask is True

    # Simpler approach: multiply index by mask, then find min where mask is True
    # But Triton doesn't have min with condition, so we use:
    # For each position where mask is True, compute index
    # For each position where mask is False, compute infinity
    # Take minimum - this gives us the first True index

    idx_float = offs.to(tl.float32)
    idx_where_true = tl.where(sampled_mask, idx_float, vocab_size.to(tl.float32))
    sampled_idx = tl.min(idx_where_true, axis=0).to(tl.int32)

    tl.store(output_ptr, sampled_idx)


@triton.jit
def sample_token_kernel_full(
    logits_ptr,  # [vocab_size] logits in float32
    output_ptr,  # [1] output token id
    vocab_size,
    temperature,
    top_p,
    seed,
    seq_pos,
    BLOCK_SIZE: tl.constexpr,
):
    """Full sampling kernel with top-p (nucleus) filtering.

    This kernel performs:
    1. Temperature scaling
    2. Softmax to get probabilities
    3. Top-p filtering using binary search for threshold
    4. Renormalize
    5. Multinomial sampling

    The top-p filtering uses a binary search approach to find the probability
    threshold, which is more efficient than sorting for large vocabularies.
    """
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size

    # Load logits
    logits = tl.load(logits_ptr + offs, mask=mask, other=float("-inf"))

    # Step 1: Temperature scaling
    scaled_logits = logits / temperature

    # Step 2: Compute softmax probabilities
    max_logit = tl.max(scaled_logits, axis=0)
    exp_logits = tl.exp(scaled_logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp

    # Step 3: Top-p filtering using binary search for threshold
    # Find threshold such that sum(probs >= threshold) >= top_p
    # and sum(probs > threshold) <= top_p

    # Get min and max probability (excluding -inf values)
    valid_probs = tl.where(mask, probs, 0.0)
    prob_min = tl.min(valid_probs, axis=0)
    prob_max = tl.max(valid_probs, axis=0)

    # Binary search for threshold (10 iterations for good precision)
    lo = prob_min
    hi = prob_max

    for _ in range(10):
        mid = (lo + hi) / 2.0
        above_mask = probs >= mid
        above_sum = tl.sum(probs * above_mask.to(tl.float32), axis=0)

        if above_sum < top_p:
            hi = mid
        else:
            lo = mid

    threshold = lo

    # Apply threshold: keep tokens with prob >= threshold
    # But we need to ensure sum of kept probs >= top_p
    # Check if we need to include threshold tokens

    # Keep tokens strictly above threshold
    keep_mask = probs > threshold
    above_sum = tl.sum(probs * keep_mask.to(tl.float32), axis=0)

    # If sum above threshold is less than top_p, include threshold tokens
    # For simplicity, include all tokens at or above threshold
    # This may slightly exceed top_p but is correct in spirit
    if above_sum < top_p:
        keep_mask = probs >= threshold

    # Ensure at least one token is kept
    # (this handles edge cases where threshold filtering removes all tokens)
    max_prob_idx = tl.argmax(probs, axis=0)
    keep_mask = keep_mask | (offs == max_prob_idx)

    # Renormalize
    scaled_logits = tl.where(keep_mask, scaled_logits, float("-inf"))
    max_logit = tl.max(scaled_logits, axis=0)
    exp_logits = tl.exp(scaled_logits - max_logit)
    sum_exp = tl.sum(exp_logits, axis=0)
    probs = exp_logits / sum_exp

    # Step 4: Multinomial sampling
    rand_val = _generate_rand_float(seed, seq_pos)
    cumsum_probs = tl.cumsum(probs)
    sampled_mask = cumsum_probs >= rand_val

    # Find first True: use min of indices where mask is True
    idx_float = offs.to(tl.float32)
    idx_where_true = tl.where(sampled_mask, idx_float, vocab_size.to(tl.float32))
    sampled_idx = tl.min(idx_where_true, axis=0).to(tl.int32)

    tl.store(output_ptr, sampled_idx)


# ============================================================================
# Reference PyTorch implementation for comparison and testing
# ============================================================================

LOWEST_TEMPERATURE = 1e-5


def sample_token_torch(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
) -> int:
    """Reference PyTorch implementation for comparison and fallback.

    Args:
        logits: Tensor of shape [vocab_size] containing logits.
        temperature: Sampling temperature.
        top_p: Top-p filtering threshold.
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        Sampled token ID as int.
    """
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got {logits.ndim}D")

    # Temperature scaling
    scaled_logits = logits.float() / max(temperature, LOWEST_TEMPERATURE)

    # Top-p filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            scaled_logits, dim=-1, descending=True
        )
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens whose cumulative mass before this token exceeds top_p
        remove_mask_sorted = (cumulative_probs - sorted_probs) > top_p
        remove_mask_sorted[0] = False  # Always keep at least one token

        # Scatter mask back to original order
        remove_mask = torch.zeros_like(remove_mask_sorted, dtype=torch.bool)
        remove_mask.scatter_(dim=-1, index=sorted_indices, src=remove_mask_sorted)

        scaled_logits = scaled_logits.masked_fill(remove_mask, float("-inf"))

    # Softmax and sample
    probs = torch.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(next_token.item())