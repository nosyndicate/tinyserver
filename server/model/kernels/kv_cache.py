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


@triton.jit
def _store_kv_cache_batched_kernel(
    # --- pointers ---
    k_src_ptr,
    v_src_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_mapping_ptr,
    slot_mapping_ptr,
    # --- scalars ---
    num_tokens,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    # --- strides for k_src, v_src ---
    k_stride_head,
    k_stride_tok,
    k_stride_headdim,
    v_stride_head,
    v_stride_tok,
    v_stride_headdim,
    # --- strides for k_cache, v_cache ---
    cache_stride_blk,
    cache_stride_head,
    cache_stride_slot,
    cache_stride_dim,
):
    program_id = tl.program_id(0)
    head_idx = tl.program_id(1)

    offsets_h = tl.arange(0, head_dim)
    # Promote to int64 so the offset arithmetic below cannot overflow int32 for
    # large batches / large paged caches (which would wrap to a wrong address).
    offsets_t = (program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)

    k_src_offset = (
        head_idx * k_stride_head
        + offsets_t[:, None] * k_stride_tok
        + offsets_h[None, :] * k_stride_headdim
    )  # (BLOCK_SIZE, head_dim)
    v_src_offset = (
        head_idx * v_stride_head
        + offsets_t[:, None] * v_stride_tok
        + offsets_h[None, :] * v_stride_headdim
    )  # (BLOCK_SIZE, head_dim)

    mask = offsets_t[:, None] < num_tokens  # (BLOCK_SIZE, 1)

    k = tl.load(
        k_src_ptr + k_src_offset, mask=mask, other=0.0
    )  # (BLOCK_SIZE, head_dim)
    v = tl.load(
        v_src_ptr + v_src_offset, mask=mask, other=0.0
    )  # (BLOCK_SIZE, head_dim)

    block_indices = tl.load(
        block_mapping_ptr + offsets_t, mask=offsets_t < num_tokens, other=0
    ).to(tl.int64)  # (BLOCK_SIZE,)
    slot_indices = tl.load(
        slot_mapping_ptr + offsets_t, mask=offsets_t < num_tokens, other=0
    ).to(tl.int64)  # (BLOCK_SIZE,)

    cache_offsets = (
        block_indices[:, None] * cache_stride_blk
        + head_idx * cache_stride_head
        + slot_indices[:, None] * cache_stride_slot
        + offsets_h[None, :] * cache_stride_dim
    )

    tl.store(k_cache_ptr + cache_offsets, value=k, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value=v, mask=mask)


def store_kv_cache_batched(
    k_src: torch.Tensor,
    v_src: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_mapping: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    Batched version of store_kv_cache. Unlike store_kv_cache, this function handle a whole batch of sequences at once,
    saving us from launching one kernel per sequence.  Here, the destination cache address for each token is precomputed
    and provided in the block_mapping, and slot_mapping tensors.

    Args:
        k_src: The key tensor for the batch, it has the flatten shape of (num_key_value_heads, num_tokens, head_dim)
            where num_tokens is the combined length of all sequences in the batch (i.e., sum of individual sequence lengths).
        v_src: The value tensor for the batch, of shape (num_key_value_heads, num_tokens, head_dim). Similar to k_src,
            num_tokens is the combined length of all sequences in the batch.
        k_cache: The key cache tensor for current attention layer with shape (num_blocks, num_key_value_heads, block_size, head_dim).
        v_cache: The value cache tensor for current attention layer with shape (num_blocks, num_key_value_heads, block_size, head_dim).
        block_mapping: A tensor of shape (num_tokens,) that maps each token in the batch to a block index in the cache.
        slot_mapping: A tensor of shape (num_tokens,) that maps each token in the batch to a slot index of the physical block in the cache.
    """

    if k_src.dim() != 3 or v_src.dim() != 3:
        raise ValueError(
            f"k_src and v_src must be 3D tensors, but got shapes {k_src.shape} and {v_src.shape}"
        )
    if k_src.shape != v_src.shape:
        raise ValueError(
            f"k_src and v_src must have the same shape, but got {k_src.shape} and {v_src.shape}"
        )

    BLOCK_SIZE = 16
    num_key_value_heads, num_tokens, head_dim = k_src.shape

    if block_mapping.shape[0] != num_tokens or slot_mapping.shape[0] != num_tokens:
        raise ValueError(
            f"block_mapping and slot_mapping must have length num_tokens ({num_tokens}), "
            f"but got {block_mapping.shape[0]} and {slot_mapping.shape[0]}"
        )

    grid = (triton.cdiv(num_tokens, BLOCK_SIZE), num_key_value_heads)

    _store_kv_cache_batched_kernel[grid](
        k_src,
        v_src,
        k_cache,
        v_cache,
        block_mapping,
        slot_mapping,
        num_tokens,
        head_dim,
        BLOCK_SIZE,
        *k_src.stride(),
        *v_src.stride(),
        *k_cache.stride(),
    )
