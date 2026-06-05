import math

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_kernel(
    # --- pointers ---
    out_ptr,  # (num_seqs, num_heads, head_size)
    query_ptr,  # (num_seqs, num_heads, head_size)
    k_cache_ptr,  # (num_blocks, num_key_value_heads, block_size, head_dim)
    v_cache_ptr,  # (num_blocks, num_key_value_heads, block_size, head_dim)
    block_tables_ptr,
    seq_lens_ptr,  # (num_seqs,)
    # --- scalars ---
    softmax_scale,
    num_heads,
    num_kv_heads,
    head_dims,
    block_size,
    max_num_blocks_per_seq,
    # --- strides ---
    stride_q_seq,
    stride_q_head,
    stride_q_dim,
    stride_kv_blk,
    stride_kv_head,
    stride_kv_slot,
    stride_kv_dim,
    stride_o_seq,
    stride_o_head,
    stride_o_dim,
):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # GQA: map the query head to the corresponding kv head using modulo
    heads_per_group = num_heads // num_kv_heads
    kv_head_idx = head_idx // heads_per_group

    slots_offsets = tl.arange(0, block_size)
    headdim_offsets = tl.arange(0, head_dims)

    q_offset = (
        seq_idx * stride_q_seq
        + head_idx * stride_q_head
        + headdim_offsets * stride_q_dim
    )

    q = tl.load(query_ptr + q_offset).to(tl.float32)  # (head_dims,)

    # Load the block table for this sequence, which maps sequence positions to block indices in the cache
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_logical_blocks = triton.cdiv(seq_len, block_size)

    # each sequence has one running max and one running sum for the online softmax computation
    m_i = tl.full([1], float("-inf"), dtype=tl.float32)  # (1,)
    l_i = tl.zeros([1], dtype=tl.int32)  # (1,)
    acc = tl.zeros([head_dims], dtype=tl.float32)  # (head_dims,)

    for logical_blk_idx in range(num_logical_blocks):
        physical_blk_idx = tl.load(
            block_tables_ptr + seq_idx * max_num_blocks_per_seq + logical_blk_idx
        )

        kv_offset = (
            physical_blk_idx * stride_kv_blk
            + kv_head_idx * stride_kv_head
            + slots_offsets[:, None] * stride_kv_slot
            + headdim_offsets[None, :] * stride_kv_dim
        )
        kv_mask = slots_offsets[:, None] < seq_len - logical_blk_idx * block_size

        k = tl.load(k_cache_ptr + kv_offset, mask=kv_mask, other=0.0).to(
            tl.float32
        )  # (block_size, head_dims)
        v = tl.load(v_cache_ptr + kv_offset, mask=kv_mask, other=0.0).to(
            tl.float32
        )  # (block_size, head_dims)

        qk = tl.dot(q, k.T)  # (block_size,)
        qk *= softmax_scale

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=0))  # (1,)
        p_ij = tl.exp(qk - m_i_new)  # (block_size,)
        scale = tl.exp(m_i - m_i_new)  # (1,)
        acc = acc * scale[:, None]
        acc = acc + tl.sum(p_ij * v, axis=0)  # (head_dims,)
        l_i = l_i * scale + tl.sum(p_ij, axis=0)  # (1,)
        m_i = m_i_new

    out = acc / l_i[:, None]  # (head_dims,)
    out_offset = (
        seq_idx * stride_o_seq
        + head_idx * stride_o_head
        + headdim_offsets * stride_o_dim
    )
    tl.store(out_ptr + out_offset, out)


def paged_attention_forward(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    block_size: int,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """
    Decode-phase paged attention (one new token per sequence).

    For each sequence s and query head h:
        out[s:, h, :] = softmax(Q[s,h] @ K[s].T * scale) @ V[s]

    where K[s] and V[s] are gathered from the paged cache using block_tables[s]

    Args:
        query: The query tensor of shape (num_seqs, num_heads, head_size).
        k_cache: The key cache tensor for current attention layer with
            shape (num_blocks, num_key_value_heads, block_size, head_dim).
        v_cache: The value cache tensor for current attention layer with
            shape (num_blocks, num_key_value_heads, block_size, head_dim).
        block_tables: A tensor of shape (num_seqs, max_num_blocks_per_seq). We assume for each sequence, we can
            allocate at most max_num_blocks_per_seq blocks in the cache. Each entry is an index into the first
            dimension of k_cache / v_cache, indicating which block of the cache corresponds to that logical
            block of the sequence.

            For example, block_tables[s] = [5, 2, 8, ...] means that for sequence s, the first logical block
            of tokens is stored in physical block 5 of the cache, the second logical block of tokens is stored
            in physical block 2 of the cache, and so on.
        block_size: The number of tokens stored in each block of the cache.
        seq_lens: A tensor of shape (num_seqs,) indicating the actual sequence length for each sequence in
            the batch. This is needed to know how many blocks of the cache are actually used for each sequence.

    Returns:
        out: Output tensor of shape (num_seqs, num_heads, head_size).
    """
    if query.dim() != 3:
        raise ValueError(f"query must be a 3D tensor, but got shape {query.shape}")

    num_seqs, num_heads, head_dims = query.shape
    num_kv_heads = k_cache.shape[1]
    _, max_num_blocks_per_seq = block_tables.shape

    grid = (num_seqs, num_heads)

    softmax_scale = 1.0 / math.sqrt(head_dims)

    out = torch.empty_like(query)

    _paged_attention_kernel[grid](
        query,
        k_cache,
        v_cache,
        out,
        block_tables,
        seq_lens,
        softmax_scale,
        num_heads,
        num_kv_heads,
        head_dims,  # tl.constexpr
        block_size,  # tl.constexpr
        max_num_blocks_per_seq,  # tl.constexpr
        *query.stride(),  # stride_q_seq, stride_q_head, stride_q_dim
        *k_cache.stride(),  # stride_kv_blk, stride_kv_head, stride_kv_slot, stride_kv_dim
        *out.stride(),  # stride_o_seq, stride_o_head, stride_o_dim
    )

    return out
