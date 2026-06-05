"""Direct unit tests for the decode-phase paged-attention Triton kernel.

These drive ``paged_attention_forward`` directly and compare against an eager
torch reference, instead of going through a full patched model (which is what
``test_paged_attention_validation.py`` does, slowly). They pin the behaviors the
recent kernel fixes touched: the argument order at launch, the GQA head->kv-head
mapping, masking of out-of-sequence positions in a partially filled last block,
and the 1-D/2-D online-softmax accumulation.
"""

import math

import pytest
import torch

from server.model.kernels.paged_attention import paged_attention_forward
from tests.model.utils import requires_cuda

_DEVICE = "cuda"
_ATOL = 1e-3
_RTOL = 1e-3


def _reference_paged_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    block_size: int,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Eager torch baseline mirroring the kernel's math (computed in fp32).

    For each sequence and head, gather K/V from the paged cache by walking the
    block table for ceil(seq_len/block_size) logical blocks (only the valid rows
    of the last block), map each query head to its kv head, then compute
    softmax(q @ K.T * scale) @ V.
    """
    num_seqs, num_heads, head_dim = query.shape
    num_kv_heads = k_cache.shape[1]
    heads_per_group = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(query)
    for s in range(num_seqs):
        seq_len = int(seq_lens[s])
        num_blocks = math.ceil(seq_len / block_size)
        ks, vs = [], []
        for logical_blk in range(num_blocks):
            phys_blk = int(block_tables[s, logical_blk])
            n = min(block_size, seq_len - logical_blk * block_size)
            ks.append(k_cache[phys_blk, :, :n, :])  # (num_kv_heads, n, head_dim)
            vs.append(v_cache[phys_blk, :, :n, :])
        K = torch.cat(ks, dim=1).float()  # (num_kv_heads, seq_len, head_dim)
        V = torch.cat(vs, dim=1).float()
        for h in range(num_heads):
            kv_head = h // heads_per_group
            q = query[s, h].float()  # (head_dim,)
            attn = (K[kv_head] @ q) * scale  # (seq_len,)
            weights = torch.softmax(attn, dim=0)  # (seq_len,)
            out[s, h] = (weights @ V[kv_head]).to(out.dtype)
    return out


def _random_paged_inputs(
    seq_lens: list[int],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
):
    """Build query / paged K-V cache / scattered block_tables.

    Each sequence is assigned a disjoint, *shuffled* run of physical blocks so
    that a contiguous-block assumption in the kernel would fail. Unused trailing
    block-table entries are padded with 0 (never read by the kernel).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    num_seqs = len(seq_lens)
    blocks_per_seq = [max(1, math.ceil(L / block_size)) for L in seq_lens]
    max_blocks = max(blocks_per_seq)
    total_blocks = sum(blocks_per_seq)
    # A little headroom so physical block ids are genuinely scattered.
    num_blocks = total_blocks + 3

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=gen)

    query = randn(num_seqs, num_heads, head_dim)
    k_cache = randn(num_blocks, num_kv_heads, block_size, head_dim)
    v_cache = randn(num_blocks, num_kv_heads, block_size, head_dim)

    phys = torch.randperm(num_blocks, generator=gen, device=device)[:total_blocks]
    block_tables = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=device)
    cursor = 0
    for s, n in enumerate(blocks_per_seq):
        block_tables[s, :n] = phys[cursor : cursor + n].to(torch.int32)
        cursor += n

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return query, k_cache, v_cache, block_tables, seq_lens_t


_CONFIGS = [
    # MHA, single block.
    pytest.param(4, 4, 64, 16, [10], id="mha-single-block"),
    # GQA, multi-block, ragged lengths (partial last block + scattered blocks).
    pytest.param(8, 2, 64, 16, [7, 16, 40], id="gqa-ragged"),
    # GQA, larger head dim: seq_len==1, block-aligned (64), and partial (33).
    pytest.param(16, 4, 128, 32, [1, 33, 64], id="gqa-headdim128"),
]


@requires_cuda
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,block_size,seq_lens", _CONFIGS
)
def test_matches_reference(num_heads, num_kv_heads, head_dim, block_size, seq_lens):
    query, k_cache, v_cache, block_tables, seq_lens_t = _random_paged_inputs(
        seq_lens,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        _DEVICE,
        torch.float32,
        seed=0,
    )

    out = paged_attention_forward(
        query, k_cache, v_cache, block_tables, block_size, seq_lens_t
    )
    ref = _reference_paged_attention(
        query, k_cache, v_cache, block_tables, block_size, seq_lens_t
    )

    assert out.shape == query.shape
    torch.testing.assert_close(out, ref, atol=_ATOL, rtol=_RTOL)


@requires_cuda
def test_query_not_mutated():
    """The kernel must write into ``out``, never into the input query tensor.

    Regression for the launch-time argument-order swap, which aliased out_ptr to
    the caller's query.
    """
    query, k_cache, v_cache, block_tables, seq_lens_t = _random_paged_inputs(
        [12, 5],
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        block_size=16,
        device=_DEVICE,
        dtype=torch.float32,
        seed=1,
    )
    query_before = query.clone()

    paged_attention_forward(query, k_cache, v_cache, block_tables, 16, seq_lens_t)

    assert torch.equal(query, query_before)


@requires_cuda
def test_output_shape_and_dtype():
    query, k_cache, v_cache, block_tables, seq_lens_t = _random_paged_inputs(
        [20],
        num_heads=4,
        num_kv_heads=4,
        head_dim=64,
        block_size=16,
        device=_DEVICE,
        dtype=torch.float32,
        seed=2,
    )
    out = paged_attention_forward(query, k_cache, v_cache, block_tables, 16, seq_lens_t)
    assert out.shape == query.shape
    assert out.dtype == query.dtype


@requires_cuda
def test_partial_last_block_masked():
    """Out-of-sequence slots in the final block must not affect the output.

    The cache slots beyond ``seq_len`` in the last block are filled with large
    garbage values; if the kernel failed to mask them (the pre-fix behavior let
    padding leak ``exp(0)`` into the softmax) the result would diverge wildly
    from the reference, which only reads the valid rows.
    """
    block_size = 16
    seq_len = 5  # 11 unused slots in the (single) block
    query, k_cache, v_cache, block_tables, seq_lens_t = _random_paged_inputs(
        [seq_len],
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        block_size=block_size,
        device=_DEVICE,
        dtype=torch.float32,
        seed=3,
    )

    # Poison the out-of-sequence slots of the sequence's physical block.
    phys_blk = int(block_tables[0, 0])
    k_cache[phys_blk, :, seq_len:, :] = 1e4
    v_cache[phys_blk, :, seq_len:, :] = 1e4

    out = paged_attention_forward(
        query, k_cache, v_cache, block_tables, block_size, seq_lens_t
    )
    ref = _reference_paged_attention(
        query, k_cache, v_cache, block_tables, block_size, seq_lens_t
    )

    torch.testing.assert_close(out, ref, atol=_ATOL, rtol=_RTOL)
    assert torch.isfinite(out).all()


def test_non_3d_query_raises():
    """A 2-D query must raise ValueError (no CUDA needed for the guard)."""
    query = torch.zeros(4, 64)  # wrong: 2D
    k_cache = torch.zeros(2, 4, 16, 64)
    v_cache = torch.zeros_like(k_cache)
    block_tables = torch.zeros(1, 1, dtype=torch.int32)
    seq_lens = torch.ones(1, dtype=torch.int32)

    with pytest.raises(ValueError, match="3D"):
        paged_attention_forward(query, k_cache, v_cache, block_tables, 16, seq_lens)
