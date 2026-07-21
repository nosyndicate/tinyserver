import pytest
import torch

from server.model.determinism import mix64, uniform_from_hash, uniforms_from_seeds
from server.model.sampling import sample_tokens
from tests.model.utils import requires_cuda


def _call(
    logit_rows: list[list[float]],
    temperature: list[float],
    top_k: list[int],
    top_p: list[float],
    uniforms: list[float],
    device: str = "cpu",
) -> torch.Tensor:
    """Build fresh tensors and sample. Re-callable, so purity tests are meaningful."""
    return sample_tokens(
        torch.tensor(logit_rows, dtype=torch.float32, device=device),
        torch.tensor(temperature, dtype=torch.float32, device=device),
        torch.tensor(top_k, dtype=torch.int64, device=device),
        torch.tensor(top_p, dtype=torch.float32, device=device),
        torch.tensor(uniforms, dtype=torch.float32, device=device),
    )


def _allowed_set(
    row_logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> set[int]:
    """Trusted per-row reference: which token ids may legally be sampled.

    Deliberately written as a plain Python loop over the sorted order rather than
    sharing any code with sample_tokens.
    """
    scaled = row_logits.double() / max(temperature, 1e-5)
    order = sorted(range(len(scaled)), key=lambda i: (-scaled[i].item(), i))
    if top_k > 0:
        order = order[:top_k]

    kept_logits = torch.tensor([scaled[i] for i in order], dtype=torch.float64)
    probs = torch.softmax(kept_logits, dim=-1)

    allowed = set()
    mass_before = 0.0
    for rank, token in enumerate(order):
        # Column 0 is always kept, so no row is ever empty.
        if rank == 0 or mass_before <= top_p:
            allowed.add(token)
        mass_before += probs[rank].item()
    return allowed


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


def test_identical_inputs_produce_identical_outputs() -> None:
    logits = torch.randn(6, 64).tolist()
    args = ([1.0] * 6, [0] * 6, [0.9] * 6, [0.1, 0.3, 0.5, 0.7, 0.9, 0.05])
    assert torch.equal(_call(logits, *args), _call(logits, *args))


def test_uniform_distribution_picks_by_hand_computed_cdf() -> None:
    """All-equal logits over V=4 give probs 0.25 each and (stable sort) order
    [0, 1, 2, 3], so the CDF is [.25, .5, .75, 1.0] and each uniform lands in a
    bucket we can name without running the code."""
    logits = [[0.0, 0.0, 0.0, 0.0]] * 4
    result = _call(logits, [1.0] * 4, [0] * 4, [1.0] * 4, [0.1, 0.4, 0.6, 0.9])
    assert result.tolist() == [0, 1, 2, 3]


def test_returns_int64_of_batch_shape() -> None:
    result = _call(
        torch.randn(3, 10).tolist(), [1.0] * 3, [0] * 3, [1.0] * 3, [0.5] * 3
    )
    assert result.shape == (3,)
    assert result.dtype == torch.int64


# ---------------------------------------------------------------------------
# Batch invariance — the property the Triton rejection kernel fails
# ---------------------------------------------------------------------------


def test_row_token_is_independent_of_batch_position() -> None:
    """Same row, same noise, sampled alone vs embedded at index 7 of a 16-row
    batch -> same token. kernels/top_p_sampling.py mixes the row index into the
    Philox offset and cannot pass this."""
    torch.manual_seed(0)
    target_row = torch.randn(128).tolist()
    others = torch.randn(15, 128).tolist()
    uniform = uniform_from_hash(seed=1234, step=9)

    alone = _call([target_row], [0.8], [40], [0.9], [uniform])

    batched_rows = others[:7] + [target_row] + others[7:]
    batched = _call(
        batched_rows,
        [0.8] * 16,
        [40] * 16,
        [0.9] * 16,
        [uniform_from_hash(seed=i, step=9) for i in range(7)]
        + [uniform]
        + [uniform_from_hash(seed=i, step=9) for i in range(8, 16)],
    )

    assert batched[7].item() == alone[0].item()


def test_batch_size_one_matches_row_in_larger_batch_for_every_row() -> None:
    torch.manual_seed(1)
    rows = torch.randn(8, 50).tolist()
    uniforms = [uniform_from_hash(100 + i, step=3) for i in range(8)]
    batched = _call(rows, [1.0] * 8, [10] * 8, [0.95] * 8, uniforms)
    for i, row in enumerate(rows):
        single = _call([row], [1.0], [10], [0.95], [uniforms[i]])
        assert single[0].item() == batched[i].item()


# ---------------------------------------------------------------------------
# Greedy
# ---------------------------------------------------------------------------


def test_greedy_rows_match_argmax_when_mixed_with_sampled_rows() -> None:
    torch.manual_seed(2)
    logits = torch.randn(6, 40)
    temperature = [0.0, 1.0, 0.0, 0.7, 0.0, 1.2]
    result = _call(
        logits.tolist(), temperature, [0] * 6, [0.9] * 6, [0.5, 0.5, 0.9, 0.9, 0.1, 0.1]
    )
    expected_greedy = logits.argmax(dim=-1)
    for i, temp in enumerate(temperature):
        if temp == 0.0:
            assert result[i].item() == expected_greedy[i].item()


def test_greedy_breaks_ties_to_lowest_index_like_argmax() -> None:
    logits = [[0.0, 5.0, 5.0, 1.0]]
    assert _call(logits, [0.0], [0], [1.0], [0.99]).tolist() == [1]


def test_greedy_ignores_uniform_value() -> None:
    torch.manual_seed(3)
    logits = torch.randn(2, 32).tolist()
    for uniform in (0.0, 0.25, 0.999):
        result = _call(logits, [0.0, 0.0], [0, 0], [1.0, 1.0], [uniform, uniform])
        assert result.tolist() == torch.tensor(logits).argmax(-1).tolist()


# ---------------------------------------------------------------------------
# Filter support: the pick always lands inside the top-k ∩ top-p allowed set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "top_k,top_p",
    [(0, 1.0), (5, 1.0), (0, 0.6), (5, 0.6), (1, 1.0), (1000, 1.0), (3, 0.3)],
)
def test_sampled_token_is_always_inside_allowed_set(top_k: int, top_p: float) -> None:
    torch.manual_seed(4)
    logits = torch.randn(4, 60)
    references = [
        _allowed_set(logits[i], temperature=0.9, top_k=top_k, top_p=top_p)
        for i in range(4)
    ]

    for step in range(40):
        uniforms = [(step * 4 + i) / 160.0 for i in range(4)]
        result = _call(logits.tolist(), [0.9] * 4, [top_k] * 4, [top_p] * 4, uniforms)
        for i, token in enumerate(result.tolist()):
            assert token in references[i], (
                f"row {i} sampled {token} outside allowed set "
                f"(top_k={top_k}, top_p={top_p}, u={uniforms[i]})"
            )


def test_per_row_filters_are_honored_independently() -> None:
    """One shared logit profile, different per-row top_k -> different allowed sets.
    Catches a batch-wide scalar being used in place of the per-row tensor."""
    row = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    result = _call([row] * 3, [1.0] * 3, [1, 2, 3], [1.0] * 3, [0.999, 0.999, 0.999])
    # u -> 1 picks the last surviving rank, which is exactly rank k-1.
    assert result.tolist() == [0, 1, 2]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_uniform_zero_picks_top_ranked_survivor() -> None:
    logits = [[1.0, 9.0, 3.0, 7.0]]
    assert _call(logits, [1.0], [0], [1.0], [0.0]).tolist() == [1]


def test_top_k_one_is_argmax() -> None:
    torch.manual_seed(5)
    logits = torch.randn(4, 30)
    result = _call(
        logits.tolist(), [1.0] * 4, [1] * 4, [1.0] * 4, [0.0, 0.4, 0.8, 0.99]
    )
    assert result.tolist() == logits.argmax(-1).tolist()


def test_tiny_top_p_returns_dominant_token() -> None:
    logits = [[-10.0] * 10 for _ in range(3)]
    logits[0][2] = 10.0
    logits[1][7] = 10.0
    logits[2][0] = 10.0
    result = _call(logits, [1.0] * 3, [0] * 3, [1e-6] * 3, [0.9] * 3)
    assert result.tolist() == [2, 7, 0]


def test_top_k_larger_than_vocab_is_a_no_op() -> None:
    torch.manual_seed(6)
    logits = torch.randn(3, 20).tolist()
    uniforms = [0.15, 0.55, 0.85]
    disabled = _call(logits, [1.0] * 3, [0] * 3, [1.0] * 3, uniforms)
    oversized = _call(logits, [1.0] * 3, [999] * 3, [1.0] * 3, uniforms)
    assert torch.equal(disabled, oversized)


def test_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="Expected logits shape"):
        sample_tokens(
            torch.randn(10),
            torch.ones(1),
            torch.zeros(1, dtype=torch.int64),
            torch.ones(1),
            torch.ones(1) * 0.5,
        )

    with pytest.raises(ValueError, match="Expected top_k shape"):
        sample_tokens(
            torch.randn(2, 10),
            torch.ones(2),
            torch.zeros(3, dtype=torch.int64),
            torch.ones(2),
            torch.ones(2) * 0.5,
        )


# ---------------------------------------------------------------------------
# Noise source
# ---------------------------------------------------------------------------


def test_mix64_is_a_pure_function() -> None:
    assert mix64(42, 7) == mix64(42, 7)
    assert mix64(42, 7) != mix64(43, 7)
    assert mix64(42, 7) != mix64(42, 8)


def test_mix64_stays_in_64_bit_range() -> None:
    for seed in (0, 1, 2**63, 2**64 - 1):
        for step in (0, 1, 10_000):
            assert 0 <= mix64(seed, step) < 2**64


def test_uniform_from_hash_is_in_unit_interval() -> None:
    for seed in range(200):
        u = uniform_from_hash(seed, step=seed % 7)
        assert 0.0 <= u < 1.0


def test_uniform_stream_varies_across_steps() -> None:
    """Nothing may be constant across decode steps, or a request would draw the
    same uniform every step and produce correlated samples."""
    stream = [uniform_from_hash(1234, step) for step in range(64)]
    assert len(set(stream)) == len(stream)


def test_uniforms_from_seeds_is_row_local() -> None:
    seeds = [11, 22, 33, 44]
    full = uniforms_from_seeds(seeds, steps=[5, 5, 5, 5])
    for i, seed in enumerate(seeds):
        assert uniforms_from_seeds([seed], steps=[5])[0].item() == full[i].item()


def test_uniforms_from_seeds_shape_and_range() -> None:
    values = uniforms_from_seeds(list(range(32)), steps=[2] * 32)
    assert values.shape == (32,)
    assert values.dtype == torch.float32
    assert bool(((values >= 0.0) & (values < 1.0)).all())


def test_uniforms_from_seeds_supports_per_row_steps() -> None:
    """Rows in a real batch are admitted at different times, so their step
    counters generally differ -- a shared scalar step would be wrong."""
    seeds = [7, 7, 7, 7]
    steps = [0, 1, 10, 1000]
    result = uniforms_from_seeds(seeds, steps=steps)
    expected = [uniform_from_hash(seed, step) for seed, step in zip(seeds, steps)]
    assert result.tolist() == pytest.approx(expected, abs=1e-7)


def test_uniforms_from_seeds_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        uniforms_from_seeds([1, 2, 3], steps=[0, 1])


# ---------------------------------------------------------------------------
# Distribution sanity
# ---------------------------------------------------------------------------

# Critical chi-square at p=0.999 (one-sided), df = 1..15. A correct sampler
# fails this ~1/1000 runs. Same table as the rejection-kernel test.
_CRITICAL_AT_P999 = [
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


def test_distribution_matches_masked_reference_chi_square() -> None:
    """Replicate one row N times with distinct per-row noise; each row is one
    independent draw. Compare the counts against the analytically masked and
    renormalized distribution."""
    V = 20
    top_p_value = 0.85
    N = 8000

    weights = torch.tensor([0.7**k for k in range(V)], dtype=torch.float64)
    probs_target = weights / weights.sum()
    logit_row = torch.log(probs_target).float().tolist()

    # Reference: keep tokens whose exclusive cumulative mass is <= top_p, renormalize.
    allowed = _allowed_set(
        torch.tensor(logit_row), temperature=1.0, top_k=0, top_p=top_p_value
    )
    expected_dist = torch.zeros(V, dtype=torch.float64)
    for token in allowed:
        expected_dist[token] = probs_target[token]
    expected_dist /= expected_dist.sum()
    expected_counts = expected_dist * N

    samples = _call(
        [logit_row] * N,
        [1.0] * N,
        [0] * N,
        [top_p_value] * N,
        [uniform_from_hash(seed, step=1) for seed in range(N)],
    )
    observed = torch.bincount(samples, minlength=V).double()

    n_outside = int(observed[expected_dist == 0].sum().item())
    assert n_outside == 0, f"{n_outside}/{N} samples fell outside the top-p kept set"

    valid = expected_counts >= 5
    chi_sq = float(
        ((observed[valid] - expected_counts[valid]) ** 2 / expected_counts[valid])
        .sum()
        .item()
    )
    df = int(valid.sum().item()) - 1
    assert 1 <= df <= len(_CRITICAL_AT_P999), f"unexpected df={df}"
    critical = _CRITICAL_AT_P999[df - 1]
    assert chi_sq < critical, (
        f"chi-square goodness-of-fit failed: chi_sq={chi_sq:.2f} >= "
        f"critical={critical:.2f} (df={df}, N={N})"
    )


# ---------------------------------------------------------------------------
# CUDA parity
# ---------------------------------------------------------------------------


@requires_cuda
def test_cuda_matches_cpu() -> None:
    torch.manual_seed(7)
    logits = torch.randn(12, 256).tolist()
    args = (
        [0.0, 0.8, 1.0, 0.5] * 3,
        [0, 20, 50, 0] * 3,
        [1.0, 0.9, 0.5, 1.0] * 3,
        [uniform_from_hash(s, step=4) for s in range(12)],
    )
    cpu = _call(logits, *args, device="cpu")
    cuda = _call(logits, *args, device="cuda")
    assert cpu.tolist() == cuda.cpu().tolist()
