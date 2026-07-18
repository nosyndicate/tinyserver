import pytest
import torch

pytest.importorskip("triton")

from server.model.kernels.kv_cache import (  # noqa: E402
    store_kv_cache,
    store_kv_cache_batched,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton requires CUDA"
)


def _build_block_slot_mappings(
    seqs: list[dict],
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build block/slot mapping tensors for store_kv_cache_batched.

    Each entry in seqs must have keys: start_pos, seq_len, block_table (list[int]).
    The mapping mirrors the per-token address computation inside _store_kv_kernel so
    both paths land on exactly the same cache locations.
    """
    block_mapping: list[int] = []
    slot_mapping: list[int] = []
    for s in seqs:
        for i in range(s["seq_len"]):
            abs_pos = s["start_pos"] + i
            block_mapping.append(s["block_table"][abs_pos // block_size])
            slot_mapping.append(abs_pos % block_size)
    return (
        torch.tensor(block_mapping, dtype=torch.int32, device=device),
        torch.tensor(slot_mapping, dtype=torch.int32, device=device),
    )


def _empty_cache(
    num_blocks: int, num_kv_heads: int, block_size: int, head_dim: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (num_blocks, num_kv_heads, block_size, head_dim)
    return torch.zeros(shape, device=device), torch.zeros(shape, device=device)


def _run_per_seq(seqs, k_srcs, v_srcs, k_cache, v_cache, block_size):
    for s, k_src, v_src in zip(seqs, k_srcs, v_srcs):
        store_kv_cache(
            s["start_pos"],
            torch.tensor(s["block_table"], dtype=torch.int32, device=k_cache.device),
            k_src,
            v_src,
            k_cache,
            v_cache,
        )


def _run_batched(seqs, k_srcs, v_srcs, k_cache, v_cache, block_size):
    # Flatten all sequences along the token dimension
    k_flat = torch.cat(k_srcs, dim=1)  # (num_kv_heads, total_tokens, head_dim)
    v_flat = torch.cat(v_srcs, dim=1)
    block_mapping, slot_mapping = _build_block_slot_mappings(
        seqs, block_size, k_cache.device.type
    )
    block_mapping = block_mapping.to(k_cache.device)
    slot_mapping = slot_mapping.to(k_cache.device)
    store_kv_cache_batched(
        k_flat, v_flat, k_cache, v_cache, block_mapping, slot_mapping
    )


@requires_cuda
def test_batched_matches_per_seq_single_seq():
    """Single sequence: batched and per-seq write to identical cache locations."""
    device = "cuda"
    num_kv_heads, block_size, head_dim = 2, 4, 8
    num_blocks = 2

    seqs = [{"start_pos": 0, "seq_len": 4, "block_table": [0]}]
    k_src = torch.randn(num_kv_heads, 4, head_dim, device=device)
    v_src = torch.randn(num_kv_heads, 4, head_dim, device=device)

    k_ref, v_ref = _empty_cache(num_blocks, num_kv_heads, block_size, head_dim, device)
    _run_per_seq(seqs, [k_src], [v_src], k_ref, v_ref, block_size)

    k_bat, v_bat = _empty_cache(num_blocks, num_kv_heads, block_size, head_dim, device)
    _run_batched(seqs, [k_src], [v_src], k_bat, v_bat, block_size)

    assert torch.equal(k_bat, k_ref)
    assert torch.equal(v_bat, v_ref)


@requires_cuda
def test_batched_matches_per_seq_multiple_seqs():
    """Three sequences with different lengths all written in one batched call."""
    device = "cuda"
    num_kv_heads, block_size, head_dim = 4, 4, 16
    # seq0: 3 tokens -> 1 block; seq1: 5 tokens -> 2 blocks; seq2: 4 tokens -> 1 block
    num_blocks = 4

    seqs = [
        {"start_pos": 0, "seq_len": 3, "block_table": [0]},
        {"start_pos": 0, "seq_len": 5, "block_table": [1, 2]},
        {"start_pos": 0, "seq_len": 4, "block_table": [3]},
    ]
    k_srcs = [
        torch.randn(num_kv_heads, s["seq_len"], head_dim, device=device) for s in seqs
    ]
    v_srcs = [
        torch.randn(num_kv_heads, s["seq_len"], head_dim, device=device) for s in seqs
    ]

    k_ref, v_ref = _empty_cache(num_blocks, num_kv_heads, block_size, head_dim, device)
    _run_per_seq(seqs, k_srcs, v_srcs, k_ref, v_ref, block_size)

    k_bat, v_bat = _empty_cache(num_blocks, num_kv_heads, block_size, head_dim, device)
    _run_batched(seqs, k_srcs, v_srcs, k_bat, v_bat, block_size)

    assert torch.equal(k_bat, k_ref)
    assert torch.equal(v_bat, v_ref)


@requires_cuda
def test_batched_matches_per_seq_multi_block():
    """Sequence long enough to span multiple blocks — exercises block boundary logic."""
    device = "cuda"
    num_kv_heads, block_size, head_dim = 2, 4, 32
    seq_len = 12  # 3 blocks worth
    num_blocks = 3

    seqs = [{"start_pos": 0, "seq_len": seq_len, "block_table": [0, 1, 2]}]
    k_src = torch.randn(num_kv_heads, seq_len, head_dim, device=device)
    v_src = torch.randn(num_kv_heads, seq_len, head_dim, device=device)

    k_ref, v_ref = _empty_cache(num_blocks, num_kv_heads, block_size, head_dim, device)
    _run_per_seq(seqs, [k_src], [v_src], k_ref, v_ref, block_size)

    k_bat, v_bat = _empty_cache(num_blocks, num_kv_heads, block_size, head_dim, device)
    _run_batched(seqs, [k_src], [v_src], k_bat, v_bat, block_size)

    assert torch.equal(k_bat, k_ref)
    assert torch.equal(v_bat, v_ref)


@requires_cuda
def test_batched_matches_per_seq_single_token():
    """Edge case: each sequence contributes exactly one token."""
    device = "cuda"
    num_kv_heads, block_size, head_dim = 2, 4, 8
    num_blocks = 3

    seqs = [
        {"start_pos": 0, "seq_len": 1, "block_table": [0]},
        {"start_pos": 0, "seq_len": 1, "block_table": [1]},
        {"start_pos": 0, "seq_len": 1, "block_table": [2]},
    ]
    k_srcs = [torch.randn(num_kv_heads, 1, head_dim, device=device) for _ in seqs]
    v_srcs = [torch.randn(num_kv_heads, 1, head_dim, device=device) for _ in seqs]

    k_ref, v_ref = _empty_cache(num_blocks, num_kv_heads, block_size, head_dim, device)
    _run_per_seq(seqs, k_srcs, v_srcs, k_ref, v_ref, block_size)

    k_bat, v_bat = _empty_cache(num_blocks, num_kv_heads, block_size, head_dim, device)
    _run_batched(seqs, k_srcs, v_srcs, k_bat, v_bat, block_size)

    assert torch.equal(k_bat, k_ref)
    assert torch.equal(v_bat, v_ref)


@requires_cuda
def test_batched_matches_per_seq_nonzero_start_pos():
    """Sequences with start_pos > 0 (decode / continuation steps).

    store_kv_cache indexes block_table by abs_pos // block_size, so the table
    must have enough entries to cover the highest logical block index.  For
    start_pos=4 with block_size=4 the first logical block touched is 1, so the
    table needs at least 2 entries.  For start_pos=8 it's logical block 2, so
    at least 3 entries.  Slots for earlier logical blocks are never read but
    must exist in the table.
    """
    device = "cuda"
    num_kv_heads, block_size, head_dim = 2, 4, 8
    # Physical blocks: seq0 uses block 1 (logical 1), seq1 uses block 2 (logical 2).
    num_blocks = 3

    seqs = [
        # Tokens at abs positions 4-6 → logical block 1 → physical block 1
        {"start_pos": 4, "seq_len": 3, "block_table": [0, 1]},
        # Tokens at abs positions 8-11 → logical block 2 → physical block 2
        {"start_pos": 8, "seq_len": 4, "block_table": [0, 0, 2]},
    ]
    k_srcs = [
        torch.randn(num_kv_heads, s["seq_len"], head_dim, device=device) for s in seqs
    ]
    v_srcs = [
        torch.randn(num_kv_heads, s["seq_len"], head_dim, device=device) for s in seqs
    ]

    k_ref, v_ref = _empty_cache(num_blocks, num_kv_heads, block_size, head_dim, device)
    _run_per_seq(seqs, k_srcs, v_srcs, k_ref, v_ref, block_size)

    k_bat, v_bat = _empty_cache(num_blocks, num_kv_heads, block_size, head_dim, device)
    _run_batched(seqs, k_srcs, v_srcs, k_bat, v_bat, block_size)

    assert torch.equal(k_bat, k_ref)
    assert torch.equal(v_bat, v_ref)


def test_batched_wrong_k_src_dim_raises():
    """2D k_src must raise ValueError."""
    num_kv_heads, head_dim = 2, 8
    k_src = torch.zeros(num_kv_heads, head_dim)  # wrong: 2D
    v_src = torch.zeros(num_kv_heads, 1, head_dim)
    k_cache = torch.zeros(1, num_kv_heads, 4, head_dim)
    v_cache = torch.zeros_like(k_cache)
    block_mapping = torch.zeros(1, dtype=torch.int32)
    slot_mapping = torch.zeros(1, dtype=torch.int32)

    with pytest.raises(ValueError, match="3D"):
        store_kv_cache_batched(
            k_src, v_src, k_cache, v_cache, block_mapping, slot_mapping
        )


def test_batched_kv_shape_mismatch_raises():
    """k_src and v_src with different shapes must raise ValueError."""
    num_kv_heads, head_dim = 2, 8
    k_src = torch.zeros(num_kv_heads, 3, head_dim)
    v_src = torch.zeros(num_kv_heads, 4, head_dim)  # different seq_len
    k_cache = torch.zeros(2, num_kv_heads, 4, head_dim)
    v_cache = torch.zeros_like(k_cache)
    block_mapping = torch.zeros(3, dtype=torch.int32)
    slot_mapping = torch.zeros(3, dtype=torch.int32)

    with pytest.raises(ValueError):
        store_kv_cache_batched(
            k_src, v_src, k_cache, v_cache, block_mapping, slot_mapping
        )


def test_batched_block_mapping_length_mismatch_raises():
    """block_mapping length != num_tokens must raise ValueError."""
    num_kv_heads, head_dim = 2, 8
    k_src = torch.zeros(num_kv_heads, 4, head_dim)
    v_src = torch.zeros_like(k_src)
    k_cache = torch.zeros(2, num_kv_heads, 4, head_dim)
    v_cache = torch.zeros_like(k_cache)
    block_mapping = torch.zeros(3, dtype=torch.int32)  # wrong: should be 4
    slot_mapping = torch.zeros(4, dtype=torch.int32)

    with pytest.raises(ValueError):
        store_kv_cache_batched(
            k_src, v_src, k_cache, v_cache, block_mapping, slot_mapping
        )
