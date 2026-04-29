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
    # Uniform logits → q = 1.0 >= top_p = 0.9 on every round → never accepted.
    # With max_rounds=1 all rows fall back to argmax → index 0.
    result = _call([[0.0] * 10] * 4, [0.9] * 4, [1, 2, 3, 4], max_rounds=1)
    assert (result == 0).all()


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
