import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from server.model.batch_ops import (
    DecodeBatchOutput,
    batched_decode_forward,
    batched_prefill,
    build_attention_mask,
    pad_and_stack_kv_caches,
)

pytestmark = pytest.mark.slow


def test_batched_prefill_single_prompt_shapes(
    qwen3_model: PreTrainedModel, qwen3_tokenizer: PreTrainedTokenizerFast
) -> None:
    results = batched_prefill(
        qwen3_model, qwen3_tokenizer, ["Hello, how are you?"], device="cpu"
    )

    assert len(results) == 1
    result = results[0]

    vocab_size = qwen3_model.config.vocab_size
    n = result.num_prompt_tokens

    assert n > 0
    assert result.logits.shape == (1, n, vocab_size)
    assert result.logits.dtype == torch.float32

    num_layers = qwen3_model.config.num_hidden_layers
    num_kv_heads = qwen3_model.config.num_key_value_heads
    head_dim = qwen3_model.config.head_dim

    assert len(result.past_key_values.layers) == num_layers
    for layer in result.past_key_values.layers:
        assert layer.keys.shape == (1, num_kv_heads, n, head_dim)
        assert layer.values.shape == (1, num_kv_heads, n, head_dim)


def test_batched_prefill_multiple_prompts_varied_lengths(
    qwen3_model: PreTrainedModel, qwen3_tokenizer: PreTrainedTokenizerFast
) -> None:
    prompts = [
        "Hi",
        "What is the capital of France?",
        "Explain quantum mechanics in one paragraph.",
    ]
    results = batched_prefill(qwen3_model, qwen3_tokenizer, prompts, device="cpu")

    assert len(results) == 3

    vocab_size = qwen3_model.config.vocab_size
    num_layers = qwen3_model.config.num_hidden_layers
    num_kv_heads = qwen3_model.config.num_key_value_heads
    head_dim = qwen3_model.config.head_dim

    for result in results:
        n = result.num_prompt_tokens
        assert n > 0
        assert result.logits.shape == (1, n, vocab_size)
        assert len(result.past_key_values.layers) == num_layers
        for layer in result.past_key_values.layers:
            assert layer.keys.shape == (1, num_kv_heads, n, head_dim)
            assert layer.values.shape == (1, num_kv_heads, n, head_dim)

    token_counts = [r.num_prompt_tokens for r in results]
    assert token_counts[0] < token_counts[1] < token_counts[2]


def test_batched_prefill_consistent_with_single_prefill(
    qwen3_model: PreTrainedModel, qwen3_tokenizer: PreTrainedTokenizerFast
) -> None:
    prompts = [
        "Hello",
        "What is 2+2?",
    ]
    batched_results = batched_prefill(
        qwen3_model, qwen3_tokenizer, prompts, device="cpu"
    )

    for i, prompt in enumerate(prompts):
        message = [{"role": "user", "content": prompt}]
        formatted = qwen3_tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        single_inputs = qwen3_tokenizer([formatted], return_tensors="pt")
        with torch.inference_mode():
            single_output = qwen3_model(**single_inputs, use_cache=True)

        single_logits = single_output.logits
        single_prompt_len = int(single_inputs["input_ids"].shape[1])

        batched = batched_results[i]

        assert batched.num_prompt_tokens == single_prompt_len
        assert torch.allclose(batched.logits, single_logits, atol=1e-4, rtol=1e-4)


def test_split_prefill_outputs_equal_length_prompts(
    qwen3_model: PreTrainedModel, qwen3_tokenizer: PreTrainedTokenizerFast
) -> None:
    prompts = ["Hello there", "Hello there"]
    results = batched_prefill(qwen3_model, qwen3_tokenizer, prompts, device="cpu")

    assert len(results) == 2
    assert results[0].num_prompt_tokens == results[1].num_prompt_tokens
    assert torch.allclose(results[0].logits, results[1].logits, atol=1e-5)

    for layer_idx in range(len(results[0].past_key_values.layers)):
        assert torch.allclose(
            results[0].past_key_values.layers[layer_idx].keys,
            results[1].past_key_values.layers[layer_idx].keys,
            atol=1e-5,
        )
        assert torch.allclose(
            results[0].past_key_values.layers[layer_idx].values,
            results[1].past_key_values.layers[layer_idx].values,
            atol=1e-5,
        )


def test_slice_dynamic_cache_batch_isolates_single_request(
    qwen3_model: PreTrainedModel, qwen3_tokenizer: PreTrainedTokenizerFast
) -> None:

    prompts = ["Short", "This is a much longer prompt than the first one"]
    results = batched_prefill(qwen3_model, qwen3_tokenizer, prompts, device="cpu")

    message = [{"role": "user", "content": prompts[0]}]
    formatted = qwen3_tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    single_inputs = qwen3_tokenizer([formatted], return_tensors="pt")
    with torch.inference_mode():
        single_output = qwen3_model(**single_inputs, use_cache=True)
    single_cache = single_output.past_key_values
    single_prompt_len = int(single_inputs["input_ids"].shape[1])

    assert results[0].num_prompt_tokens == single_prompt_len
    assert torch.allclose(results[0].logits, single_output.logits, atol=1e-4, rtol=1e-4)

    for layer_idx in range(len(single_cache.layers)):
        assert (
            results[0].past_key_values.layers[layer_idx].keys.shape
            == single_cache.layers[layer_idx].keys.shape
        )
        assert (
            results[0].past_key_values.layers[layer_idx].values.shape
            == single_cache.layers[layer_idx].values.shape
        )
        assert torch.allclose(
            results[0].past_key_values.layers[layer_idx].keys,
            single_cache.layers[layer_idx].keys,
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.allclose(
            results[0].past_key_values.layers[layer_idx].values,
            single_cache.layers[layer_idx].values,
            atol=1e-4,
            rtol=1e-4,
        )


def test_slice_dynamic_cache_batch_isolates_second_request(
    qwen3_model: PreTrainedModel, qwen3_tokenizer: PreTrainedTokenizerFast
) -> None:
    """Verify the *second* (longer) item in the batch also has correct cache values."""
    prompts = ["Short", "This is a much longer prompt than the first one"]
    results = batched_prefill(qwen3_model, qwen3_tokenizer, prompts, device="cpu")

    message = [{"role": "user", "content": prompts[1]}]
    formatted = qwen3_tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    single_inputs = qwen3_tokenizer([formatted], return_tensors="pt")
    with torch.inference_mode():
        single_output = qwen3_model(**single_inputs, use_cache=True)
    single_cache = single_output.past_key_values

    assert results[1].num_prompt_tokens == int(single_inputs["input_ids"].shape[1])
    assert torch.allclose(results[1].logits, single_output.logits, atol=1e-4, rtol=1e-4)
    for layer_idx in range(len(single_cache.layers)):
        assert torch.allclose(
            results[1].past_key_values.layers[layer_idx].keys,
            single_cache.layers[layer_idx].keys,
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.allclose(
            results[1].past_key_values.layers[layer_idx].values,
            single_cache.layers[layer_idx].values,
            atol=1e-4,
            rtol=1e-4,
        )


def test_batched_prefill_decode_non_first_item(
    qwen3_model: PreTrainedModel, qwen3_tokenizer: PreTrainedTokenizerFast
) -> None:
    """Verify decode works using a non-first batch item's cache."""
    prompts = ["Hello", "What is the meaning of life?"]
    results = batched_prefill(qwen3_model, qwen3_tokenizer, prompts, device="cpu")
    result = results[1]

    next_token_id = int(torch.argmax(result.logits[:, -1, :], dim=-1).item())
    next_input_ids = torch.tensor([[next_token_id]])

    with torch.inference_mode():
        decode_output = qwen3_model(
            input_ids=next_input_ids,
            past_key_values=result.past_key_values,
            use_cache=True,
        )

    assert decode_output.logits.shape == (1, 1, qwen3_model.config.vocab_size)

    expected_seq_len = result.num_prompt_tokens + 1
    for layer in decode_output.past_key_values.layers:
        assert layer.keys.shape[2] == expected_seq_len


def test_batched_prefill_empty_prompts_raises(
    qwen3_model: PreTrainedModel, qwen3_tokenizer: PreTrainedTokenizerFast
) -> None:
    with pytest.raises(ValueError, match="prompts must not be empty"):
        batched_prefill(qwen3_model, qwen3_tokenizer, [], device="cpu")


def test_batched_prefill_outputs_can_be_used_for_decode(
    qwen3_model: PreTrainedModel, qwen3_tokenizer: PreTrainedTokenizerFast
) -> None:
    results = batched_prefill(
        qwen3_model, qwen3_tokenizer, ["Tell me a joke"], device="cpu"
    )
    result = results[0]

    next_token_id = int(torch.argmax(result.logits[:, -1, :], dim=-1).item())
    next_input_ids = torch.tensor([[next_token_id]])

    with torch.inference_mode():
        decode_output = qwen3_model(
            input_ids=next_input_ids,
            past_key_values=result.past_key_values,
            use_cache=True,
        )

    assert decode_output.logits.shape == (1, 1, qwen3_model.config.vocab_size)

    expected_seq_len = result.num_prompt_tokens + 1
    for layer in decode_output.past_key_values.layers:
        assert layer.keys.shape[2] == expected_seq_len


# ------------------------------
# Tests for decode helpers
# ------------------------------


def test_pad_and_stack_kv_caches_shapes(
    qwen3_model: PreTrainedModel, qwen3_tokenizer: PreTrainedTokenizerFast
) -> None:
    prompts = ["Hi", "What is the capital of France?"]
    results = batched_prefill(qwen3_model, qwen3_tokenizer, prompts, device="cpu")

    caches = [r.past_key_values for r in results]
    batched_cache, seq_lengths = pad_and_stack_kv_caches(caches)

    assert seq_lengths == [r.num_prompt_tokens for r in results]

    num_kv_heads = qwen3_model.config.num_key_value_heads
    head_dim = qwen3_model.config.head_dim
    max_seq_len = max(seq_lengths)

    for layer in batched_cache.layers:
        assert layer.keys.shape == (2, num_kv_heads, max_seq_len, head_dim)
        assert layer.values.shape == (2, num_kv_heads, max_seq_len, head_dim)


def test_build_attention_mask_values() -> None:
    mask = build_attention_mask([3, 5], device="cpu")

    assert mask.shape == (2, 6)  # max_len + 1
    # Row 0: seq_len=3, rightmost 4 positions should be 1
    assert mask[0].tolist() == [0, 0, 1, 1, 1, 1]
    # Row 1: seq_len=5, rightmost 6 positions should be 1 (all ones)
    assert mask[1].tolist() == [1, 1, 1, 1, 1, 1]


def test_batched_decode_forward_shapes(
    qwen3_model: PreTrainedModel, qwen3_tokenizer: PreTrainedTokenizerFast
) -> None:
    prompts = ["Hello", "What is 2+2?"]
    results = batched_prefill(qwen3_model, qwen3_tokenizer, prompts, device="cpu")

    token_ids = [int(torch.argmax(r.logits[:, -1, :]).item()) for r in results]
    caches = [r.past_key_values for r in results]

    decode_results = batched_decode_forward(
        qwen3_model, token_ids, caches, device="cpu"
    )

    vocab_size = qwen3_model.config.vocab_size
    assert len(decode_results) == 2

    for i, dec in enumerate(decode_results):
        assert isinstance(dec, DecodeBatchOutput)
        assert dec.logits.shape == (1, 1, vocab_size)
        expected_seq_len = results[i].num_prompt_tokens + 1
        for layer in dec.past_key_values.layers:
            assert layer.keys.shape[2] == expected_seq_len

    # The decode cache should be the prefill cache plus exactly one new token.
    for i, dec in enumerate(decode_results):
        prefill_n = results[i].num_prompt_tokens
        for layer_idx in range(len(dec.past_key_values.layers)):
            prefill_keys = results[i].past_key_values.layers[layer_idx].keys
            prefill_values = results[i].past_key_values.layers[layer_idx].values
            decode_keys = dec.past_key_values.layers[layer_idx].keys
            decode_values = dec.past_key_values.layers[layer_idx].values

            assert torch.allclose(
                decode_keys[:, :, :prefill_n, :], prefill_keys, atol=1e-4, rtol=1e-4
            )
            assert torch.allclose(
                decode_values[:, :, :prefill_n, :], prefill_values, atol=1e-4, rtol=1e-4
            )


def test_batched_decode_forward_matches_single_decode(
    qwen3_model: PreTrainedModel, qwen3_tokenizer: PreTrainedTokenizerFast
) -> None:
    prompts = ["Hello", "Explain gravity briefly."]
    results = batched_prefill(qwen3_model, qwen3_tokenizer, prompts, device="cpu")

    token_ids = [int(torch.argmax(r.logits[:, -1, :]).item()) for r in results]
    caches = [r.past_key_values for r in results]

    decode_results = batched_decode_forward(
        qwen3_model, token_ids, caches, device="cpu"
    )

    # Compare each batched decode result against a single-request decode.
    for i in range(len(prompts)):
        single_input = torch.tensor([[token_ids[i]]])
        with torch.inference_mode():
            single_output = qwen3_model(
                input_ids=single_input,
                past_key_values=caches[i],
                use_cache=True,
            )

        assert torch.allclose(
            decode_results[i].logits,
            single_output.logits,
            atol=1e-4,
            rtol=1e-4,
        )


def test_batched_decode_forward_cache_can_be_reused(
    qwen3_model: PreTrainedModel, qwen3_tokenizer: PreTrainedTokenizerFast
) -> None:
    results = batched_prefill(
        qwen3_model, qwen3_tokenizer, ["Tell me a story"], device="cpu"
    )
    result = results[0]

    # First decode step
    token_id_1 = int(torch.argmax(result.logits[:, -1, :]).item())
    decode_1 = batched_decode_forward(
        qwen3_model, [token_id_1], [result.past_key_values], device="cpu"
    )
    dec1 = decode_1[0]

    assert dec1.logits.shape[1] == 1
    expected_len_1 = result.num_prompt_tokens + 1
    for layer in dec1.past_key_values.layers:
        assert layer.keys.shape[2] == expected_len_1

    # Decode cache should extend the prefill cache by one token.
    prefill_n = result.num_prompt_tokens
    for layer_idx in range(len(dec1.past_key_values.layers)):
        prefill_keys = result.past_key_values.layers[layer_idx].keys
        prefill_values = result.past_key_values.layers[layer_idx].values
        assert torch.allclose(
            dec1.past_key_values.layers[layer_idx].keys[:, :, :prefill_n, :],
            prefill_keys,
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.allclose(
            dec1.past_key_values.layers[layer_idx].values[:, :, :prefill_n, :],
            prefill_values,
            atol=1e-4,
            rtol=1e-4,
        )

    # Second decode step using the cache from the first
    token_id_2 = int(torch.argmax(dec1.logits[:, -1, :]).item())
    decode_2 = batched_decode_forward(
        qwen3_model, [token_id_2], [dec1.past_key_values], device="cpu"
    )
    dec2 = decode_2[0]

    expected_len_2 = expected_len_1 + 1
    for layer in dec2.past_key_values.layers:
        assert layer.keys.shape[2] == expected_len_2

    # Second decode cache should extend the first decode cache by one token.
    for layer_idx in range(len(dec2.past_key_values.layers)):
        prev_keys = dec1.past_key_values.layers[layer_idx].keys
        prev_values = dec1.past_key_values.layers[layer_idx].values
        assert torch.allclose(
            dec2.past_key_values.layers[layer_idx].keys[:, :, :expected_len_1, :],
            prev_keys,
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.allclose(
            dec2.past_key_values.layers[layer_idx].values[:, :, :expected_len_1, :],
            prev_values,
            atol=1e-4,
            rtol=1e-4,
        )


def test_split_decode_outputs_preserves_individual_caches(
    qwen3_model: PreTrainedModel, qwen3_tokenizer: PreTrainedTokenizerFast
) -> None:
    prompts = ["Short", "A slightly longer prompt here"]
    results = batched_prefill(qwen3_model, qwen3_tokenizer, prompts, device="cpu")

    token_ids = [int(torch.argmax(r.logits[:, -1, :]).item()) for r in results]
    caches = [r.past_key_values for r in results]

    decode_results = batched_decode_forward(
        qwen3_model, token_ids, caches, device="cpu"
    )

    # Each split cache should work independently for a subsequent decode step.
    for i, dec in enumerate(decode_results):
        next_token = int(torch.argmax(dec.logits[:, -1, :]).item())
        with torch.inference_mode():
            solo = qwen3_model(
                input_ids=torch.tensor([[next_token]]),
                past_key_values=dec.past_key_values,
                use_cache=True,
            )
        expected_seq_len = results[i].num_prompt_tokens + 2
        for layer in solo.past_key_values.layers:
            assert layer.keys.shape[2] == expected_seq_len
