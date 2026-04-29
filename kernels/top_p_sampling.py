import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    logits_ptr,  # [B, V]
    probs_ptr,  # [B, V]
    V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-row softmax. One program per row."""
    row = tl.program_id(0)
    logits_row = logits_ptr + row * V
    probs_row = probs_ptr + row * V

    # Pass 1: find max for numerical stability
    m = float("-inf")
    for start in range(0, V, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        x = tl.load(logits_row + offs, mask=mask, other=float("-inf"))
        m = tl.maximum(m, tl.max(x, axis=0))

    # Pass 2: compute sum of exp(x - max)
    s = 0.0
    for start in range(0, V, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        x = tl.load(logits_row + offs, mask=mask, other=float("-inf"))
        s += tl.sum(tl.exp(x - m), axis=0)

    # Pass 3: write normalized probabilities
    for start in range(0, V, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        x = tl.load(logits_row + offs, mask=mask, other=float("-inf"))
        p = tl.exp(x - m) / s
        tl.store(probs_row + offs, p, mask=mask)


@triton.jit
def rejection_sample_round_kernel(
    probs_ptr,  # [B, V] — input probabilities
    pivot_ptr,  # [B]    — current pivot per row
    top_p_ptr,  # [B]    — top-p threshold per row
    output_ptr,  # [B]    — sampled token index (written on accept)
    accepted_ptr,  # [B]    — bool flag: 1 if accepted
    seed_ptr,  # [B]    — per-row RNG seed (updated each round)
    V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Rejection sampling round for top-p sampling. One program per row.

    This is the sorting-free top-p sampling algorithm described https://flashinfer.ai/2025/03/10/sampling.html
    """
    row = tl.program_id(0)

    # Skip already-accepted rows
    already_done = tl.load(accepted_ptr + row)
    if already_done == 1:
        return

    probs_row = probs_ptr + row * V
    pivot = tl.load(pivot_ptr + row)
    top_p = tl.load(top_p_ptr + row)
    seed = tl.load(seed_ptr + row)

    # ---- Step (a): compute filtered_sum = sum{ p_i : p_i >= pivot } ----
    filtered_sum = 0.0
    for start in range(0, V, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        p = tl.load(probs_row + offs, mask=mask, other=0.0)
        valid = p >= pivot
        filtered_sum += tl.sum(tl.where(valid, p, 0.0), axis=0)

    # ---- Step (b): draw u ~ Uniform(0, filtered_sum) ----
    # Use Philox-based RNG via triton
    rand = tl.rand(seed, row)  # uniform in [0, 1)
    u = rand * filtered_sum

    # Update seed for next round
    tl.store(seed_ptr + row, seed + 1000003)  # simple seed progression

    # ---- Step (c): inverse transform sampling over filtered tokens ----
    # Walk through vocab, accumulating CDF of filtered probs.
    # The token where running CDF first exceeds u is our sample.
    cumsum = 0.0
    sampled_idx: tl.int32 = 0
    sampled_prob = 0.0
    found = False

    for start in range(0, V, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        p = tl.load(probs_row + offs, mask=mask, other=0.0)
        valid = p >= pivot

        # Masked probs for this block
        mp = tl.where(valid, p, 0.0)

        # Block sum for early exit check
        block_sum = tl.sum(mp, axis=0)

        if not found:
            if cumsum + block_sum > u:
                # The sampled token is in this block — do prefix scan
                # We do a simple sequential scan within the block
                # (Triton doesn't have a clean block-level scan with early exit,
                #  so we use a vectorized approach)
                prev_cumsum = cumsum
                cumsum_vec = tl.cumsum(mp, axis=0) + prev_cumsum

                # Find first index where cumsum_vec > u
                crossed = (cumsum_vec > u) & valid & mask
                # Use argmin trick: set non-crossed to V (large), crossed to offs
                candidate = tl.where(crossed, offs, V)
                sampled_idx = tl.min(candidate, axis=0).to(tl.int32)
                sampled_prob = tl.load(probs_row + sampled_idx)
                found = True
                cumsum = cumsum + block_sum
            else:
                cumsum = cumsum + block_sum

    # Fallback: if numerical issues prevent finding a token, pick the max-prob token
    if not found:
        max_p = 0.0
        for start in range(0, V, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            mask = offs < V
            p = tl.load(probs_row + offs, mask=mask, other=0.0)
            block_max = tl.max(p, axis=0)
            if block_max > max_p:
                max_p = block_max
        # Find the index of the max
        for start in range(0, V, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            mask = offs < V
            p = tl.load(probs_row + offs, mask=mask, other=0.0)
            is_max = (p == max_p) & mask
            candidate = tl.where(is_max, offs, V)
            idx = tl.min(candidate, axis=0).to(tl.int32)
            if idx < V:
                sampled_idx = idx
                sampled_prob = max_p

    # ---- Step (d) & (e): update pivot and check acceptance ----
    new_pivot = sampled_prob

    # Compute q = sum{ p_i : p_i >= new_pivot }
    q = 0.0
    for start in range(0, V, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < V
        p = tl.load(probs_row + offs, mask=mask, other=0.0)
        valid = p >= new_pivot
        q += tl.sum(tl.where(valid, p, 0.0), axis=0)

    # ---- Step (f): accept or reject ----
    if q < top_p:
        # Accept: the filtered set is now within the nucleus
        tl.store(output_ptr + row, sampled_idx)
        tl.store(accepted_ptr + row, 1)
    else:
        # Reject: raise the pivot and try again
        tl.store(pivot_ptr + row, new_pivot)
