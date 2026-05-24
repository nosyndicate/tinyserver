import torch
import torch.nn.functional as F
from transformers import (
    Cache,
    Qwen3ForCausalLM,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from transformers.processing_utils import Unpack

from server.model.inference_context import _inference_context


def _patch_single_layer(layer: torch.nn.Module, layer_idx: int) -> None:
    """
    Patch the attention forward pass for a single layer of the model.
    For the Qwen3 1.7B model, it is using the full attetion across all layers,
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

        This is differnet from the standard input format for huggingface models, where the whole
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



    """

    attn_module = layer.self_attn
    num_attention_heads = attn_module.config.num_attention_heads
    num_key_value_heads = attn_module.config.num_key_value_heads
    head_dim = attn_module.config.head_dim
    num_groups = num_attention_heads // num_key_value_heads

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
         [------------+----------------],
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

        if position_ids is not None:
            # TODO need to figure out why this part works
            cos, sin = attn_module.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if _inference_context.mode == "prefill":
            # the order of the sequence in sequences should match
            # the order of the sequence in input_ids and position_ids.
            # we here to make sure the q and k will be correctly stored in the kv cache for each sequence.

            token_offset = 0
            for seq in _inference_context.sequences:
                num_tokens = seq["num_tokens"]
                block_table = seq["block_table"]

                k_cache = k[:, :, token_offset : token_offset + num_tokens, :]
                v_cache = v[:, :, token_offset : token_offset + num_tokens, :]

                store_kv_cache(layer_idx, block_table, k_cache, v_cache)

                token_offset += num_tokens

            # construct the mask for prefill, where tokens from different sequences should not attend to each other.
            mask = torch.full(
                (seq_len, seq_len),
                float("-inf"),
                dtype=q.dtype,
                device=hidden_states.device,
            )
            block_start = 0
            for seq in _inference_context.sequences:
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
                mask[
                    block_start : block_start + num_tokens,
                    block_start : block_start + num_tokens,
                ] = causal_mask
                block_start += num_tokens

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
    layer_idx: int, block_table: list[int], k_cache: torch.Tensor, v_cache: torch.Tensor
) -> None:
    raise NotImplementedError()


def qwen3_model_patcher(model: Qwen3ForCausalLM) -> Qwen3ForCausalLM:
    """
    Replace the attention forward pass in the huggingface model so it will
    use the paged attention for calculation. This allow us to get to get rid
    of the usage of dynamic cache and use our custom block manager instead.
    """

    for layer_idx, layer in enumerate(model.model.layers):
        _patch_single_layer(layer, layer_idx)

    return model
