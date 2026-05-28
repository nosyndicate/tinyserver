import torch
import triton
import triton.language as tl


@triton.jit
def _store_kv_kernel(
    # --- pointers ---
    k_cache_ptr,
    v_cache_ptr,  # (num_blocks, num_kv_heads, block_size, head_dim)
    k_src_ptr,
    v_src_ptr,  # (num_kv_heads, num_tokens, head_dim)
    block_table_ptr,  # (num_logical_blocks,)
    # --- scalars ---
    start_position,
    # constexpr shapes
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    # --- strides for k_cache, v_cache ---
    stride_cache_blk,  # stride along the physical block dimension
    stride_cache_head,  # stride along the kv-head dimension
    stride_cache_slot,  # stride along the within-block token (slot) dimension
    stride_cache_dim,  # stride along the head dimension
    # --- strides for k_src, v_src ---
    stride_src_head,  # stride along the kv-head dimension
    stride_src_tok,  # stride along the token dimension
    stride_src_dim,  # stride along the head dimension
):
    """
    One Triton program = one (token, kv_head) pair.

    Grid:  (num_tokens, num_kv_heads)
    Block: single thread group loading head_dim elements at once.
    """
    # Each block processes one (token, kv-head) pair
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Load the key and value vectors for this (head, token) pair from the source tensors
    offsets = tl.arange(0, head_dim)
    src_offset = (
        head_idx * stride_src_head
        + token_idx * stride_src_tok
        + offsets * stride_src_dim
    )
    k = tl.load(k_src_ptr + src_offset)
    v = tl.load(v_src_ptr + src_offset)

    # Compute the destination address in the cache

    # The absolute position of the token in the sequence is (start_position + token_idx).
    token_pos = start_position + token_idx

    # Use the block table to find the physical block index corresponding to this token position.
    logical_blk_idx = token_pos // block_size

    # Which physical block this corresponds to in the cache
    physical_blk_idx = tl.load(block_table_ptr + logical_blk_idx)

    # Within the block, the slot index is the position modulo the block size
    slot_idx = token_pos % block_size

    cache_offsets = (
        physical_blk_idx * stride_cache_blk
        + head_idx * stride_cache_head
        + slot_idx * stride_cache_slot
        + offsets * stride_cache_dim
    )
    tl.store(k_cache_ptr + cache_offsets, k)
    tl.store(v_cache_ptr + cache_offsets, v)


def store_kv_cache(
    start_position: int,
    block_table: torch.Tensor,
    k_src: torch.Tensor,
    v_src: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> None:
    """
    For one sequence, scatter freshly computed K/V projections into the paged KV cache.
    The quantity of the kv cache is determined by the number of tokens in the sequence,
    which can be calculated by looking at the shape of the k_src or v_src.

    For each token i in k_src the absolute sequence position is
    (start_pos + i).  The function uses block_table to translate that
    position into a physical cache address and writes k_src[i] / v_src[i]
    there.

    Args:
        start_position: The position of the first token in the current sequence.
        block_table: A tensor of shape (num_blocks,) that maps absolute sequence positions to block indices in the cache.
        k_src: The key tensor for the sequence, of shape (num_key_value_heads, seq_len, head_dim).
        v_src: The value tensor for the sequence, of shape (num_key_value_heads, seq_len, head_dim).
        k_cache: The key cache tensor for current attention layer with shape (num_blocks, num_key_value_heads, block_size, head_dim).
        v_cache: The value cache tensor for current attention layer with shape (num_blocks, num_key_value_heads, block_size, head_dim).
    """
    if k_src.dim() != 3 or v_src.dim() != 3:
        raise ValueError(
            f"k_src and v_src must be 3D tensors, but got shapes {k_src.shape} and {v_src.shape}"
        )

    num_kv_heads, num_tokens, head_dim = k_src.shape
    block_size = k_cache.shape[2]
    grid = (num_tokens, num_kv_heads)

    _store_kv_kernel[grid](
        k_cache,
        v_cache,
        k_src,
        v_src,
        block_table,
        start_position,
        head_dim,
        block_size,
        *k_cache.stride(),
        *k_src.stride(),
    )
