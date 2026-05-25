from typing import Callable

import torch
import torch.nn.functional as F
from transformers import (
    Cache,
    Qwen3Config,
    Qwen3ForCausalLM,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers.processing_utils import Unpack

from server.model.inference_context import get_inference_context
from server.model.utils import bytes_to_gb, get_available_memory


def qwen3_cache_allocator(
    model: Qwen3ForCausalLM,
    config: Qwen3Config,
    memory_utilization: float,
    block_size: int,
    dtype: torch.dtype,
    device: str,
) -> None:
    available_mem = get_available_memory(memory_utilization)

    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    # kv cache dtype should match the model dtype to avoid unnecessary conversions during attention computation
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    block_bytes = (
        2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
    )  # 2 for key and value
    num_available_kv_blocks = int(available_mem // block_bytes)

    if num_available_kv_blocks <= 0:
        raise MemoryError(
            f"Not enough memory for even one block of KV cache. Available memory: {bytes_to_gb(available_mem)}, "
            f"required memory for one block: {bytes_to_gb(block_bytes)}."
        )

    kv_cache = torch.zeros(
        2,
        num_layers,
        num_available_kv_blocks,
        num_kv_heads,
        block_size,
        head_dim,
        device=device,
        dtype=dtype,
    )

    for i in range(num_layers):
        model.model.layers[i].self_attn.k_cache = kv_cache[0, i]
        model.model.layers[i].self_attn.v_cache = kv_cache[1, i]


def _patch_single_layer(
    layer: torch.nn.Module,
    rotary_emb: Callable[
        [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ],
) -> None:
    """
    Patch the attention forward pass for a single layer of the model.
    For the Qwen3 1.7B model, it is using the full attention across all layers,
    so we don't need to worry about how to patch sliding window attention.

    After patching the model, we expected the input to the huggingface model
    have the following format:
    - input_ids: (1, num_tokens)
        This is flattened input ids for all the sequence in the batch.
        For example, if we have three sequences in the batch, their token ids are as follows:
        seq1: [101, 102, 103]
        seq2: [201, ]
        seq3: [301, 302]

        The input_ids will be:
        [[101, 102, 103, 201, 301, 302]]

        This is different from the standard input format for huggingface models, where the whole
        batch will be padded to the same length and the input_ids will be:
        [[101,      102,      103],
         [<pad_id>, <pad_id>, 201]
         [<pad_id>, 301,      302]]

        In this way, we can avoid the computation on the padding tokens and make the attention calculation more efficient.

        Note, for most of the components in the decoder-only model, including
        - embedding layer
        - feed forward network
        - projection layer for q, k, v, and output
        - layer norm or rms norm

        They are all token-wise operation, which means we can directly feed in the flattened input_ids and hidden states
        without any modification. The only component that requires special handling is the attention calculation,
        where we need to make sure the attention mask and position ids are correctly calculated based on the flattened input.

    - position_ids: (1, num_tokens)
        Similar to input_ids, this is also a flattened position ids for all the sequence in the batch.
        The position ids will be reset for each sequence in the batch.  For example, for the three sequences mentioned above,
        the position ids will be (suppose this is a prefill step):
        [[0, 1, 2, 0, 0, 1]]

        Even in this case, we can still apply the rotary embedding in the same way as the standard input format.
        In the rotary embedding, the cos, and sin is computed based on the position ids for each token

            freqs = inv_freq @ position_ids
            emb = concat(freqs, freqs)
            cos = cos(emb)
            sin = sin(emb)

        So when the sin and cos is applied to the query and key, it is applied based on the position ids for
        the token.
    """

    attn_module = layer.self_attn
    num_attention_heads = attn_module.config.num_attention_heads
    num_key_value_heads = attn_module.config.num_key_value_heads
    head_dim = attn_module.config.head_dim
    num_groups = num_attention_heads // num_key_value_heads

    def _page_attention_forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        For the purpose of continuous batching,
        both input_ids and position_ids will be a batch 1 tensor.


        Mask construction:
        When multiple sequence are in the same flatten 1D tensor, we want to make sure the tokens from different
        sequence will not attend to each other.

        In a standard upper-triangular causal mask over the 3 prefill tokens:
        [[0, -inf, -inf],
         [0, 0, -inf],
         [0, 0, 0]]

        If there were two independent prefill sequences (say length 2 and 3, total 5), the mask would be block-diagonal
        so sequence A can't attend to sequence B:

        [[  0,  -inf | -inf, -inf, -inf],
         [  0,    0  | -inf, -inf, -inf],
         [-----------+-----------------],
         [-inf, -inf |   0,  -inf, -inf],
         [-inf, -inf |   0,    0,  -inf],
         [-inf, -inf |   0,    0,    0 ]]

        Args:
            hidden_states: (1, seq_len, hidden_size)
            position_ids: (1, seq_len)
        """
        batch, seq_len, _ = hidden_states.shape

        q = attn_module.q_proj(
            hidden_states
        )  # (1, seq_len, num_attention_heads * head_dim)
        k = attn_module.k_proj(
            hidden_states
        )  # (1, seq_len, num_key_value_heads * head_dim)
        v = attn_module.v_proj(
            hidden_states
        )  # (1, seq_len, num_key_value_heads * head_dim)

        q = q.view(batch, seq_len, num_attention_heads, head_dim)
        k = k.view(batch, seq_len, num_key_value_heads, head_dim)
        v = v.view(batch, seq_len, num_key_value_heads, head_dim)

        if hasattr(attn_module, "q_norm"):
            q = attn_module.q_norm(q)
        if hasattr(attn_module, "k_norm"):
            k = attn_module.k_norm(k)

        q = q.transpose(1, 2)  # (1, num_attention_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (1, num_key_value_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (1, num_key_value_heads, seq_len, head_dim)

        if position_ids is None:
            raise ValueError(
                "Position ids should not be None for the patched attention forward"
            )
        cos, sin = rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        inference_context = get_inference_context()

        if inference_context.mode == "prefill":
            # the order of the sequence in sequences should match
            # the order of the sequence in input_ids and position_ids.
            # we here to make sure the q and k will be correctly stored in the kv cache for each sequence.

            token_offset = 0
            for seq in inference_context.sequences:
                num_tokens = seq["num_tokens"]
                block_table = seq["block_table"]

                t_s, t_e = token_offset, token_offset + num_tokens
                k_src = (
                    k[:, :, t_s:t_e, :].squeeze(0).contiguous()
                )  # shape (num_key_value_heads, num_tokens, head_dim).
                v_src = (
                    v[:, :, t_s:t_e, :].squeeze(0).contiguous()
                )  # shape (num_key_value_heads, num_tokens, head_dim).

                # Since this is prefill, so the start position is hardcode to 0 now
                # If later we want to implement prefix caching or chunked prefill
                # we might need to change the start position to the actual position of the token in the sequence.
                store_kv_cache(
                    0,
                    block_table,
                    k_src,
                    v_src,
                    attn_module.k_cache,
                    attn_module.v_cache,
                )
                token_offset += num_tokens

            # Construct the mask for prefill, where tokens from different sequences
            # should not attend to each other.
            # TODO this mask is same across all layers, we can compute it once and reuse across layers
            mask = torch.full(
                (seq_len, seq_len),
                float("-inf"),
                dtype=q.dtype,
                device=hidden_states.device,
            )
            block_start = 0
            for seq in inference_context.sequences:
                num_tokens = seq["num_tokens"]
                causal_mask = torch.triu(
                    torch.full(
                        (num_tokens, num_tokens),
                        float("-inf"),
                        dtype=q.dtype,
                        device=hidden_states.device,
                    ),
                    diagonal=1,
                )

                b_s, b_e = block_start, block_start + num_tokens
                mask[b_s:b_e, b_s:b_e] = causal_mask
                block_start += num_tokens

            # if pytorch >= 2.5, then scaled_dot_product_attention support enable_gpa,
            # which can avoid we doing the repeat_interleave for k and v
            k_expanded = k.repeat_interleave(num_groups, dim=1)
            v_expanded = v.repeat_interleave(num_groups, dim=1)
            output = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded, attn_mask=mask, scale=head_dim**-0.5
            )
            output = output.transpose(1, 2).reshape(batch, seq_len, -1)
        else:
            raise NotImplementedError(
                "Need to implement the attention calculation for decoding step"
            )

        return attn_module.o_proj(output), None

    layer.self_attn.forward = _page_attention_forward


def store_kv_cache(
    start_position: int,
    block_table: list[int],
    k_src: torch.Tensor,
    v_src: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> None:
    """
    Scatter freshly computed K/V projections into the paged KV cache.

    For each token i in k_src the absolute sequence position is
    (start_pos + i).  The function uses block_table to translate that
    position into a physical cache address and writes k_src[i] / v_src[i]
    there.

    Args:
        start_position: The position of the first token in the current sequence.
        block_table: A list of block indices that this sequence occupies in the kv cache.
        k_src: The key tensor for the sequence, of shape (num_key_value_heads, seq_len, head_dim).
        v_src: The value tensor for the sequence, of shape (num_key_value_heads, seq_len, head_dim).
        k_cache: The key cache tensor for current attention layer with shape (num_blocks, num_key_value_heads, block_size, head_dim).
        v_cache: The value cache tensor for current attention layer with shape (num_blocks, num_key_value_heads, block_size, head_dim).
    """
    block_size = k_cache.shape[2]
    seq_len = k_src.shape[1]

    # We iterate all the tokens in the sequence and write them to the corresponding position
    # in the kv cache according to the block table. This is slower than doing it in a block-wise
    # manner, but it is much simpler and later can be optimized easily using parallel.
    # TODO the block id and pos_in_block for each token is compute on the fly,
    # we probably can pre-compute and store in the context for more efficient writing to the cache.
    for i in range(seq_len):
        abs_pos = start_position + i
        logical_block_idx = abs_pos // block_size
        block_idx = block_table[logical_block_idx]
        pos_in_block = abs_pos % block_size
        k_cache[block_idx, :, pos_in_block, :] = k_src[:, i, :]
        v_cache[block_idx, :, pos_in_block, :] = v_src[:, i, :]


def qwen3_model_patcher(model: Qwen3ForCausalLM) -> Qwen3ForCausalLM:
    """
    Replace the attention forward pass in the huggingface model so it will
    use the paged attention for calculation. This allow us to get to get rid
    of the usage of dynamic cache and use our custom block manager instead.
    """

    for layer in model.model.layers:
        _patch_single_layer(layer, model.model.rotary_emb)

    return model


def qwen3_model_loader(
    model: Qwen3ForCausalLM,
    config: Qwen3Config,
    memory_utilization: float,
    block_size: int,
    dtype: torch.dtype,
    device: str,
) -> None:
    """
    Allocate the kv cache for the Qwen3 model and patch the model so the attention module
    will use the pre-allocated cache for computing attention scores.
    """
    qwen3_cache_allocator(model, config, memory_utilization, block_size, dtype, device)
    qwen3_model_patcher(model)
