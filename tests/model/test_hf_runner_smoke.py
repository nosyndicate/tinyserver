import pytest

from server.model.hf_runner import ModelRunner
from server.model.sampling import SamplingParams


@pytest.mark.smoke
def test_smoke_prefill_shapes(gpt2_runner: ModelRunner) -> None:
    all_logits, past_key_values, prompt_tokens = gpt2_runner.prefill("hello")

    assert all_logits.shape == (1, prompt_tokens, 128)
    assert past_key_values is not None
    assert prompt_tokens > 0


@pytest.mark.smoke
def test_smoke_decode_loop_kv_cache_path(gpt2_runner: ModelRunner) -> None:
    all_logits, past_key_values, _ = gpt2_runner.prefill("hello")

    chunks = list(
        gpt2_runner.decode_loop(
            all_logits,
            past_key_values,
            SamplingParams(max_new_tokens=3, temperature=0.0, top_p=1.0),
        )
    )

    assert 1 <= len(chunks) <= 3
    assert chunks[0][1] is True   # is_first
    assert chunks[-1][2] is True  # is_done
    assert all(isinstance(token_str, str) for token_str, _, _ in chunks)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "temperature,seed",
    [(0.0, None), (1.0, 42)],
)
def test_smoke_generate_text_two_stage_end_to_end(
    gpt2_runner: ModelRunner, temperature: float, seed: int | None
) -> None:
    out_text, prompt_tokens, output_tokens = gpt2_runner.generate_text_two_stage(
        "hello",
        SamplingParams(max_new_tokens=5, temperature=temperature, top_p=0.9, seed=seed),
    )

    assert isinstance(out_text, str)
    assert isinstance(prompt_tokens, int)
    assert isinstance(output_tokens, int)
    assert prompt_tokens > 0
    assert 0 <= output_tokens <= 5
