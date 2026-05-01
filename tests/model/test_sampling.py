import pytest
import torch

from server.model.sampling import (
    LOWEST_TEMPERATURE,
    SamplingParams,
    rejection_sampling_based_top_p_sample,
    sample_token,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def make_params(
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 10,
    seed: int | None = None,
) -> SamplingParams:
    return SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )


def one_hot_logits(token_id: int, vocab_size: int = 100) -> torch.Tensor:
    logits = torch.full((1, vocab_size), -1e9)
    logits[0, token_id] = 1e9
    return logits


def test_greedy_returns_argmax() -> None:
    logits = one_hot_logits(7)
    assert sample_token(logits, make_params(temperature=0.0)) == 7


def test_greedy_at_lowest_temperature_boundary() -> None:
    logits = one_hot_logits(3)
    assert sample_token(logits, make_params(temperature=LOWEST_TEMPERATURE)) == 3


def test_wrong_logits_shape_raises() -> None:
    with pytest.raises(ValueError, match="Expected logits shape"):
        sample_token(torch.randn(5), make_params())

    with pytest.raises(ValueError, match="Expected logits shape"):
        sample_token(torch.randn(2, 5), make_params())


def test_invalid_top_p_raises() -> None:
    logits = torch.randn(1, 100)
    with pytest.raises(ValueError, match="top_p"):
        sample_token(logits, make_params(top_p=0.0))

    with pytest.raises(ValueError, match="top_p"):
        sample_token(logits, make_params(top_p=1.5))


def test_top_p_excludes_low_probability_tokens() -> None:
    # Token 0 gets all the mass; tokens 1-99 get ~0.
    # With top_p=0.5, only token 0 should ever be sampled.
    logits = one_hot_logits(0, vocab_size=100)
    for _ in range(20):
        assert sample_token(logits, make_params(top_p=0.5)) == 0


def test_deterministic_with_same_seed() -> None:
    logits = torch.randn(1, 100)
    params = make_params(temperature=1.0, seed=42)
    gen1 = torch.Generator()
    gen1.manual_seed(42)
    gen2 = torch.Generator()
    gen2.manual_seed(42)
    assert sample_token(logits, params, generator=gen1) == sample_token(
        logits, params, generator=gen2
    )


def test_different_seeds_can_produce_different_tokens() -> None:
    # Use a uniform distribution so every token has equal probability.
    logits = torch.zeros(1, 100)
    results = set()
    for seed in range(50):
        gen = torch.Generator()
        gen.manual_seed(seed)
        results.add(sample_token(logits, make_params(temperature=1.0), generator=gen))
    assert len(results) > 1


# ---------------------------------------------------------------------------
# rejection_sampling_based_top_p_sample — requires CUDA
# ---------------------------------------------------------------------------


def _call(
    logit_list: list[list[float]],
    top_p_list: list[float],
    seed_list: list[int],
    **kwargs,
) -> torch.Tensor:
    """Construct fresh CUDA tensors and call the function. Re-callable for determinism tests."""
    logits = torch.tensor(logit_list, dtype=torch.float32, device="cuda")
    top_p = torch.tensor(top_p_list, dtype=torch.float32, device="cuda")
    seeds = torch.tensor(seed_list, dtype=torch.int32, device="cuda")
    return rejection_sampling_based_top_p_sample(logits, top_p, seeds, **kwargs)


@requires_cuda
def test_rejection_output_shape_and_dtype() -> None:
    result = _call([[0.0] * 10] * 3, [0.9, 0.9, 0.9], [1, 2, 3])
    assert result.shape == (3,)
    assert result.dtype == torch.long


@requires_cuda
def test_rejection_output_in_range() -> None:
    V = 100
    logits = torch.randn(8, V).tolist()
    result = _call(logits, [0.9] * 8, list(range(8)))
    assert (result >= 0).all()
    assert (result < V).all()


@requires_cuda
def test_rejection_deterministic_with_same_seeds() -> None:
    logits = torch.arange(200).reshape(4, 50).float().tolist()
    top_p = [0.9] * 4
    seeds = [42, 43, 44, 45]
    result1 = _call(logits, top_p, seeds)
    result2 = _call(logits, top_p, seeds)
    assert torch.equal(result1, result2)


@requires_cuda
def test_rejection_different_seeds_produce_different_outputs() -> None:
    # Strictly-decreasing logits give each token a unique probability, so multiple tokens
    # can satisfy q < top_p (tokens 0-7 each have sum(p_i >= p_j) < 0.9).
    # B=20 with distinct seeds via tl.rand(seed, row) guarantees RNG diversity.
    B = 20
    row_logits = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    logits = [row_logits] * B
    top_p = [0.9] * B
    result = _call(logits, top_p, list(range(B)))
    assert len(set(result.tolist())) > 1


@requires_cuda
def test_rejection_fallback_to_argmax() -> None:
    """When no row accepts within max_rounds, the host fallback picks each row's argmax.

    Earlier this test used `[[0.0] * 10] * 4` and asserted the result was `0`, which
    only passed because `torch.argmax` on a uniform tensor returns the first index —
    accidental, not a real test of the fallback. Here we set two tokens per row to
    the same (highest) prob, placing the lower of the tied indices at a deliberately
    distinct, non-zero position per row. The fallback then has to pick that position.
    """
    V = 16
    # Each row: two tied "max" tokens (probs ~0.5 each); q = 1.0 ≥ top_p every round
    # → kernel never accepts → fallback fires. argmax(probs) returns the *lower*
    # tied index, which we set to a distinct non-zero value per row.
    rows = [
        ([3, 11], 3),
        ([5, 14], 5),
        ([7, 9], 7),
        ([1, 8], 1),
    ]
    logits = []
    expected = []
    for tied_positions, expected_argmax in rows:
        row = [-1e9] * V
        for pos in tied_positions:
            row[pos] = 0.0
        logits.append(row)
        expected.append(expected_argmax)

    result = _call(
        logits, [0.9] * len(rows), list(range(1, len(rows) + 1)), max_rounds=1
    )
    assert result.tolist() == expected


def _strict_top_p_distribution(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """CPU sort-based reference for the *strict* top-p convention used by the
    Triton rejection kernel: token i is in the kept set iff
    ``sum{p_j : p_j >= p_i} < top_p``. Returns the renormalized distribution
    restricted to the kept set (zeros elsewhere).

    Note: this is *not* the same as ``server.model.sampling.top_p_sampling``,
    which keeps one extra token (the first whose cumulative mass crosses top_p).
    The kernel implements the strict convention by construction.
    """
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    keep_sorted = cumprobs < top_p
    keep_sorted[0] = True  # always keep the most probable token
    truncated = torch.zeros_like(probs)
    truncated[sorted_idx[keep_sorted]] = sorted_probs[keep_sorted]
    return truncated / truncated.sum()


@requires_cuda
def test_rejection_mixed_top_p_per_row() -> None:
    """Each row's top_p is honored independently (not accidentally treated as a shared scalar).

    Four rows share the same descending probability profile (0.4, 0.3, 0.2, 0.1)
    but place those probs at *different* vocab positions, with row-specific top_p
    values that select a different number of tokens via the strict convention:

      - top_p=0.3  → no token satisfies q < top_p → fallback to argmax
      - top_p=0.5  → only the prob=0.4 token is in S
      - top_p=0.8  → prob=0.4 and prob=0.3 tokens are in S
      - top_p=0.95 → prob=0.4, prob=0.3, and prob=0.2 tokens are in S
    """
    import math

    rows = [
        # (logits, top_p, expected token set)
        ([math.log(0.4), math.log(0.3), math.log(0.2), math.log(0.1)], 0.3, {0}),
        ([math.log(0.1), math.log(0.2), math.log(0.3), math.log(0.4)], 0.5, {3}),
        ([math.log(0.1), math.log(0.4), math.log(0.3), math.log(0.2)], 0.8, {1, 2}),
        ([math.log(0.2), math.log(0.3), math.log(0.1), math.log(0.4)], 0.95, {3, 1, 0}),
    ]
    logits = [r[0] for r in rows]
    top_p = [r[1] for r in rows]
    expected_sets = [r[2] for r in rows]

    seen: list[set[int]] = [set() for _ in rows]
    trials = 80
    for trial in range(trials):
        seeds = [1 + trial * len(rows) + i for i in range(len(rows))]
        result = _call(logits, top_p, seeds)
        for i, tok in enumerate(result.tolist()):
            seen[i].add(tok)

    for i, (got, want) in enumerate(zip(seen, expected_sets)):
        assert got <= want, (
            f"row {i} (top_p={top_p[i]}): sampled tokens {got - want} "
            f"outside expected set {want}"
        )

    # Rows with multi-token expected sets must actually explore more than one
    # token across {trials} runs, otherwise the per-row top_p value isn't being
    # used (the kernel would have collapsed to one row's top_p for the batch).
    assert len(seen[2]) > 1, (
        f"row 2 only sampled {seen[2]}; per-row top_p=0.8 not honored"
    )
    assert len(seen[3]) > 1, (
        f"row 3 only sampled {seen[3]}; per-row top_p=0.95 not honored"
    )


@requires_cuda
def test_rejection_distribution_matches_reference_chi_square() -> None:
    """Chi-square goodness-of-fit against the sort-based CPU reference.

    Replicates the same logits across N rows with distinct per-row seeds — each
    row gives one independent sample. Counts which token each row produces and
    compares against the analytical strict-top-p distribution. A correct kernel
    fails this test only ~1/1000 runs (critical value at p=0.999).
    """
    # Geometric-decay weights → smooth, monotonic, non-degenerate distribution.
    # With top_p=0.85, several tokens sit inside the strict kept set, so the
    # chi-square test has multiple bins (df ≥ 4) rather than collapsing to one.
    V = 20
    weights = torch.tensor([0.7**k for k in range(V)], dtype=torch.float32)
    probs_target = weights / weights.sum()
    top_p_value = 0.85
    N = 8000

    # logits = log(probs) so softmax recovers `probs_target` exactly.
    logit_row = torch.log(probs_target).tolist()

    expected_dist = _strict_top_p_distribution(probs_target, top_p_value)
    expected_counts = expected_dist * N

    samples = _call(
        [logit_row] * N,
        [top_p_value] * N,
        list(range(1, N + 1)),
    ).cpu()
    observed = torch.bincount(samples, minlength=V).float()

    # No samples should land outside the strict top-p kept set.
    excluded = expected_dist == 0
    n_outside = int(observed[excluded].sum().item())
    assert n_outside == 0, (
        f"{n_outside}/{N} triton samples fell outside the strict top-p kept set"
    )

    # Chi-square over bins where expected count ≥ 5 (rule of thumb for validity).
    valid = expected_counts >= 5
    chi_sq = float(
        ((observed[valid] - expected_counts[valid]) ** 2 / expected_counts[valid])
        .sum()
        .item()
    )
    df = int(valid.sum().item()) - 1

    # Critical chi-square at p=0.999 (one-sided), df = 1..15. False-positive
    # rate ~1/1000 for a correct sampler.
    critical_at_p999 = [
        10.83,
        13.82,
        16.27,
        18.47,
        20.52,
        22.46,
        24.32,
        26.12,
        27.88,
        29.59,
        31.26,
        32.91,
        34.53,
        36.12,
        37.70,
    ]
    assert 1 <= df <= len(critical_at_p999), f"unexpected df={df}"
    critical = critical_at_p999[df - 1]
    assert chi_sq < critical, (
        f"chi-square goodness-of-fit failed: chi_sq={chi_sq:.2f} >= "
        f"critical={critical:.2f} (df={df}, N={N})"
    )


@requires_cuda
def test_rejection_top_p_1_returns_valid_tokens() -> None:
    logits = torch.randn(2, 50).tolist()
    result = _call(logits, [1.0, 1.0], [7, 8])
    assert result.shape == (2,)
    assert result.dtype == torch.long
    assert (result >= 0).all() and (result < 50).all()


@requires_cuda
def test_rejection_top_p_near_zero_returns_dominant_token() -> None:
    # One dominant token per row; with top_p=1e-6 only the argmax can satisfy q < top_p.
    logits = [[-10.0] * 10 for _ in range(3)]
    logits[0][2] = 10.0
    logits[1][7] = 10.0
    logits[2][0] = 10.0
    result = _call(logits, [1e-6] * 3, [1, 2, 3])
    assert result[0] == 2
    assert result[1] == 7
    assert result[2] == 0


@requires_cuda
def test_rejection_large_vocab() -> None:
    # V=2048 forces the kernel to iterate over two BLOCK_SIZE=1024 blocks per row.
    V = 2048
    B = 3
    logits = torch.randn(B, V).tolist()
    result = _call(logits, [0.9] * B, list(range(B)))
    assert result.shape == (B,)
    assert result.dtype == torch.long
    assert (result >= 0).all() and (result < V).all()


@requires_cuda
def test_rejection_batch_size_1() -> None:
    # Single-row batch; token 2 dominates.
    result = _call([[0.0, 0.0, 5.0] + [0.0] * 7], [1e-5], [99])
    assert result.shape == (1,)
    assert result.dtype == torch.long
    assert result[0] == 2
