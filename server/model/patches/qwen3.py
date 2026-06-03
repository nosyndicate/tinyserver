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
from server.model.kernels.kv_cache import store_kv_cache
from server.model.kernels.varlen_attention import flash_attn_varlen_func
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

    def _prefill_sdpa_with_block_mask(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sequences: list[dict],
        batch: int,
        seq_len: int,
    ) -> torch.Tensor:
        mask = torch.full(
            (seq_len, seq_len),
            float("-inf"),
            dtype=q.dtype,
            device=q.device,
        )
        block_start = 0
        for seq in sequences:
            num_tokens = seq["num_tokens"]
            causal_mask = torch.triu(
                torch.full(
                    (num_tokens, num_tokens),
                    float("-inf"),
                    dtype=q.dtype,
                    device=q.device,
                ),
                diagonal=1,
            )

            b_s, b_e = block_start, block_start + num_tokens
            mask[b_s:b_e, b_s:b_e] = causal_mask
            block_start += num_tokens

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            scale=head_dim**-0.5,
            enable_gqa=True,  # Using enable_gqa to avoid manually repeating k and v for each attention head
        )
        return output.transpose(1, 2).reshape(batch, seq_len, -1)

    def _prefill_varlen_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sequences: list[dict],
        batch: int,
        seq_len: int,
    ) -> torch.Tensor:
        seq_lengths = [int(seq["num_tokens"]) for seq in sequences]
        if sum(seq_lengths) != seq_len:
            raise ValueError(
                f"Prefill sequence lengths sum to {sum(seq_lengths)}, expected flattened seq_len {seq_len}"
            )
        if any(num_tokens <= 0 for num_tokens in seq_lengths):
            raise ValueError("Prefill sequence lengths must all be positive")

        cu_seqlens = torch.empty(
            len(seq_lengths) + 1, dtype=torch.int32, device=q.device
        )
        cu_seqlens[0] = 0
        cu_seqlens[1:] = torch.tensor(
            seq_lengths, dtype=torch.int32, device=q.device
        ).cumsum(dim=0)
        max_seqlen = max(seq_lengths)

        # Varlen kernel consumes packed token-major tensors.
        q_packed = q.squeeze(0).transpose(0, 1)
        k_packed = k.squeeze(0).transpose(0, 1)
        v_packed = v.squeeze(0).transpose(0, 1)

        output = flash_attn_varlen_func(
            q_packed,
            k_packed,
            v_packed,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=True,
            softmax_scale=head_dim**-0.5,
        )
        return output.reshape(batch, seq_len, -1)

    def _page_attention_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
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

        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            # fallback if this patched module is ever called outside normal Qwen3Model.forward
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
                # For prefill, num_tokens for a sequence is the total number of tokens in the prompt
                num_tokens = seq["num_tokens"]
                block_table = torch.tensor(
                    seq["block_table"], device=hidden_states.device, dtype=torch.long
                )

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

            if q.is_cuda:
                output = _prefill_varlen_attention(
                    q, k, v, inference_context.sequences, batch, seq_len
                )
            else:
                output = _prefill_sdpa_with_block_mask(
                    q, k, v, inference_context.sequences, batch, seq_len
                )
        else:
            output_tensors = []

            # TODO this implementation calls the SDPA multiple times, need to replaced with a
            # a more efficient implementation.
            for i, seq in enumerate(inference_context.sequences):
                start_position = int(position_ids[0, i].item())
                block_table = torch.tensor(
                    seq["block_table"], device=hidden_states.device, dtype=torch.long
                )

                # Since in decoding, we only need to generate 1 token at a time,
                # the net new k and v for each sequence should only be for one token.
                k_src = (
                    k[:, :, i : i + 1, :].squeeze(0).contiguous()
                )  # shape (num_key_value_heads, num_tokens, head_dim).
                v_src = (
                    v[:, :, i : i + 1, :].squeeze(0).contiguous()
                )  # shape (num_key_value_heads, num_tokens, head_dim).
                store_kv_cache(
                    start_position,
                    block_table,
                    k_src,
                    v_src,
                    attn_module.k_cache,
                    attn_module.v_cache,
                )

                # Now, we need to gather the k and v from cache to make them contiguous for the attention calculation.
                k_full, v_full = gather_kv_cache(
                    start_position,
                    block_table,
                    attn_module.k_cache,
                    attn_module.v_cache,
                )

                q_i = q[
                    :, :, i : i + 1, :
                ]  # shape (1, num_attention_heads, 1, head_dim)
                k_full = k_full.unsqueeze(0).repeat_interleave(
                    num_groups, dim=1
                )  # shape (1, num_attention_heads, seq_len, head_dim)
                v_full = v_full.unsqueeze(0).repeat_interleave(
                    num_groups, dim=1
                )  # shape (1, num_attention_heads, seq_len, head_dim)

                # Compute the attention output for the current sequence, since the query has length 1 per sequence,
                # so causality is trivially satisfied
                # We actually don't need to mask anything
                output_i = F.scaled_dot_product_attention(
                    q_i, k_full, v_full, attn_mask=None, scale=head_dim**-0.5
                )  # shape: (1, num_attention_heads, 1, head_dim)

                output_tensors.append(output_i)

            output = torch.cat(
                output_tensors, dim=2
            )  # shape (1, num_attention_heads, seq_len, head_dim)
            output = output.transpose(1, 2).reshape(
                batch, seq_len, -1
            )  # shape (1, seq_len, num_attention_heads * head_dim)

        return attn_module.o_proj(output), None

    layer.self_attn.forward = _page_attention_forward


def gather_kv_cache(
    next_position: int,
    block_table: list[int],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For one sequence, gather the K/V from the paged KV cache and make them contiguous for attention calculation.
    This is the inverse operation of store_kv_cache.

    Args:
        next_position: The next position to generate for the current sequence, we should gather
            all the kv from position 0 to next_position (inclusive) from the cache and make them contiguous.
        block_table: A list of block indices that this sequence occupies in the kv cache.
        k_cache: The key cache tensor for current attention layer with shape (num_blocks, num_key_value_heads, block_size, head_dim).
        v_cache: The value cache tensor for current attention layer with shape (num_blocks, num_key_value_heads, block_size, head_dim).

    Returns:
        k_full: The gathered key tensor for the sequence, of shape (num_key_value_heads, seq_len, head_dim).
        v_full: The gathered value tensor for the sequence, of shape (num_key_value_heads, seq_len, head_dim).
    """
    block_size = k_cache.shape[2]
    seq_len = next_position + 1

    k_full = torch.empty(
        (k_cache.shape[1], seq_len, k_cache.shape[3]),
        device=k_cache.device,
        dtype=k_cache.dtype,
    )
    v_full = torch.empty(
        (v_cache.shape[1], seq_len, v_cache.shape[3]),
        device=v_cache.device,
        dtype=v_cache.dtype,
    )

    for i in range(seq_len):
        abs_pos = i
        logical_block_idx = abs_pos // block_size
        block_idx = block_table[logical_block_idx]
        pos_in_block = abs_pos % block_size
        k_full[:, i, :] = k_cache[block_idx, :, pos_in_block, :]
        v_full[:, i, :] = v_cache[block_idx, :, pos_in_block, :]

    return k_full, v_full


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
