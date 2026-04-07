from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizerFast


@dataclass
class PrefillBatchOutput:
    logits: Tensor
    past_key_values: DynamicCache
    num_prompt_tokens: int


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

    inputs = tokenizer(
        all_formatted_prompts, return_tensors="pt", padding=True, padding_side="left"
    ).to(device)
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
