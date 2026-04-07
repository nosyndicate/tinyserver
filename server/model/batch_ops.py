from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class PrefillBatchOutput:
    logits: Tensor
    past_key_values: DynamicCache
    num_prompt_tokens: int


@dataclass
class DecodeBatchOutput:
    logits: Tensor
    past_key_values: DynamicCache


@torch.inference_mode()
def batched_prefill(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    prompts: list[str],
    device: str,
) -> list[PrefillBatchOutput]:
    """
    Run prefill for a batch of requests and split the outputs into individual results.
    """
    if not prompts:
        raise ValueError("prompts must not be empty")

    all_formatted_prompts: list[str] = []
    for prompt in prompts:
        message = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        all_formatted_prompts.append(formatted)

    original_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            all_formatted_prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)
    finally:
        tokenizer.padding_side = original_padding_side

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # We use Qwen3 which uses RoPE position embeddings, so we need to provide
    # correct position ids that account for left padding to make sure we are
    # getting the right kv cache outputs.
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )

    logits: Tensor = outputs.logits
    batched_cache: DynamicCache = outputs.past_key_values

    return split_prefill_outputs(
        logits=logits,
        past_key_values=batched_cache,
        attention_mask=attention_mask,
    )


@torch.inference_mode()
def batched_decode_forward(
    model: PreTrainedModel,
    token_ids: list[int],
    past_key_values: list[DynamicCache],
    device: str,
) -> list[DecodeBatchOutput]:
    input_ids = torch.tensor(token_ids, device=device, dtype=torch.long).reshape(-1, 1)

    batched_cache, seq_lengths = pad_and_stack_kv_caches(past_key_values)

    attention_mask = build_attention_mask(
        seq_lengths=seq_lengths,
        device=device,
    )

    # position_ids are of size [batch_size, 1] and indicate the position of token generated from
    # last forward pass for each request.
    position_ids = torch.tensor(seq_lengths, device=device).reshape(-1, 1)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=batched_cache,
        use_cache=True,
    )

    return split_decode_outputs(outputs, seq_lengths)


def pad_and_stack_kv_caches(
    caches: list[DynamicCache],
) -> tuple[DynamicCache, list[int]]:
    """
    For a batch of requests, pad and stack their kv caches into a single cache that can be used
    for a batched decode forward pass.
    """
    num_layers = len(caches[0].layers)
    # Cache has the shape [b, num_kv_heads, seq_len, head_dim].
    # We need to pad seq_len to the max seq_len in the batch.
    seq_lengths = [cache.layers[0].keys.shape[2] for cache in caches]

    max_seq_len = max(seq_lengths)

    batched_cache = DynamicCache()
    for layer_idx in range(num_layers):
        padded_keys = []
        padded_values = []
        for cache in caches:
            keys = cache.layers[layer_idx].keys
            values = cache.layers[layer_idx].values

            # Pad the seq_len dimension to max_seq_len with zeros on the left (since we are using left padding).
            pad_len = max_seq_len - keys.shape[2]
            padded_key = torch.nn.functional.pad(keys, (0, 0, pad_len, 0))
            padded_value = torch.nn.functional.pad(values, (0, 0, pad_len, 0))

            padded_keys.append(padded_key)
            padded_values.append(padded_value)

        # Stack along the batch dimension
        batched_keys = torch.cat(padded_keys, dim=0)
        batched_values = torch.cat(padded_values, dim=0)

        batched_cache.update(
            key_states=batched_keys,
            value_states=batched_values,
            layer_idx=layer_idx,
        )

    return batched_cache, seq_lengths


def build_attention_mask(
    seq_lengths: list[int],
    device: str,
) -> Tensor:
    batch_size = len(seq_lengths)
    max_len = max(seq_lengths)
    attention_mask = torch.zeros(
        (batch_size, max_len + 1), device=device, dtype=torch.long
    )
    for i, seq_len in enumerate(seq_lengths):
        attention_mask[i, -(seq_len + 1) :] = 1
    return attention_mask


def split_prefill_outputs(
    logits: Tensor,
    past_key_values: DynamicCache,
    attention_mask: Tensor,
) -> list[PrefillBatchOutput]:
    """
    For a batch of prefill outputs, split the combined logits and cache into individual
    results for each request.
    """
    results: list[PrefillBatchOutput] = []

    batch_size = int(logits.shape[0])
    prompt_lengths = attention_mask.sum(dim=1).tolist()

    for batch_idx in range(batch_size):
        bs, be = batch_idx, batch_idx + 1
        prompt_len = int(prompt_lengths[batch_idx])

        # Inputs were left-padded, so the real prompt tokens are the rightmost prompt_len positions.
        request_logits = logits[bs:be, -prompt_len:, :].contiguous()

        request_cache = _slice_dynamic_cache_batch(
            past_key_values=past_key_values,
            batch_idx=batch_idx,
            seq_len=prompt_len,
        )

        results.append(
            PrefillBatchOutput(
                logits=request_logits,
                past_key_values=request_cache,
                num_prompt_tokens=prompt_len,
            )
        )

    return results


def split_decode_outputs(
    outputs: CausalLMOutputWithPast,
    seq_lengths: list[int],
) -> list[DecodeBatchOutput]:
    """
    For a batch of decode forward outputs, split the combined logits and cache into individual
    results for each request.
    """
    results: list[DecodeBatchOutput] = []

    batch_size = int(outputs.logits.shape[0])
    for batch_idx in range(batch_size):
        bs, be = batch_idx, batch_idx + 1
        seq_len = seq_lengths[batch_idx] + 1

        # logits should have the shape [batch_size, 1, vocab_size],
        # where the 1 corresponds to the single new token being generated.
        request_logits = outputs.logits[bs:be, :, :].contiguous()

        request_cache = DynamicCache()
        for layer_idx in range(len(outputs.past_key_values.layers)):
            keys = outputs.past_key_values.layers[layer_idx].keys
            values = outputs.past_key_values.layers[layer_idx].values
            request_cache.update(
                key_states=keys[bs:be, :, -seq_len:, :].contiguous(),
                value_states=values[bs:be, :, -seq_len:, :].contiguous(),
                layer_idx=layer_idx,
            )

        results.append(
            DecodeBatchOutput(
                logits=request_logits,
                past_key_values=request_cache,
            )
        )

    return results


def _slice_dynamic_cache_batch(
    past_key_values: DynamicCache,
    batch_idx: int,
    seq_len: int,
) -> DynamicCache:
    """
    For a single request in a prefill batch, slice the combined cache to get the relevant portion
    for that request.
    """
    request_cache = DynamicCache()

    bs, be = batch_idx, batch_idx + 1
    for layer_idx in range(len(past_key_values.layers)):
        keys = past_key_values.layers[layer_idx].keys
        values = past_key_values.layers[layer_idx].values
        # Expected shape is roughly:
        # [batch, num_kv_heads, padded_seq_len, head_dim]
        request_key = keys[bs:be, :, -seq_len:, :].contiguous()
        request_value = values[bs:be, :, -seq_len:, :].contiguous()

        request_cache.update(
            key_states=request_key,
            value_states=request_value,
            layer_idx=layer_idx,
        )

    return request_cache
