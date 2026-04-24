import pytest
import torch

from server.model.sampling import LOWEST_TEMPERATURE, SamplingParams, sample_token


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
