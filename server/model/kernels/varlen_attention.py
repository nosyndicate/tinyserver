import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_varlen_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    stride_qt,
    stride_q_num_heads,
    stride_q_headdim,
    stride_kt,
    stride_k_num_heads,
    stride_k_headdim,
    stride_vt,
    stride_v_num_heads,
    stride_v_headdim,
    stride_ot,
    stride_o_num_heads,
    stride_o_headdim,
    num_heads_q: tl.constexpr,
    num_heads_k: tl.constexpr,
    head_dim: tl.constexpr,
    causal: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_q_idx = tl.program_id(0)  # which block of queries
    batch_idx = tl.program_id(1)  # which sequence in the batch
    head_q_idx = tl.program_id(2)  # which attention head

    # GQA: map query heads to key/value heads
    heads_per_group = num_heads_q // num_heads_k
    head_k_idx = head_q_idx // heads_per_group

    q_start = tl.load(cu_seqlens_q_ptr + batch_idx).to(
        tl.int32
    )  # start index of queries for this sequence
    q_end = tl.load(cu_seqlens_q_ptr + batch_idx + 1).to(
        tl.int32
    )  # end index of queries for this sequence
    k_start = tl.load(cu_seqlens_k_ptr + batch_idx).to(
        tl.int32
    )  # start index of keys/values for this sequence
    k_end = tl.load(cu_seqlens_k_ptr + batch_idx + 1).to(
        tl.int32
    )  # end index of keys/values for this sequence

    q_len = q_end - q_start
    k_len = k_end - k_start

    # We launch max_seqlen_q / BLOCK_M blocks for each sequence,
    # but some of the sequence is much shorter tham max_seqlen_q, so there will be some blocks that are out of range for the short sequences.
    # For these blocks, we can skip the computation to save time.
    if block_q_idx * BLOCK_M >= q_len:
        return  # this block is out of range for the queries

    offset_headdim = tl.arange(0, BLOCK_D)  # (BLOCK_D,)
    offset_seq = block_q_idx * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)

    q_offsets = (
        (q_start + offset_seq[:, None]) * stride_qt
        + head_q_idx * stride_q_num_heads
        + offset_headdim[None, :] * stride_q_headdim
    )
    q_masks = offset_seq < q_len  # (BLOCK_M,)
    q = tl.load(
        q_ptr + q_offsets,
        mask=(q_masks[:, None] & (offset_headdim[None, :] < head_dim)),
        other=0.0,
    )  # (BLOCK_M, BLOCK_D)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # (BLOCK_M,)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # (BLOCK_M,)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # (BLOCK_M, BLOCK_D)

    if causal:
        # For causal, we only need blocks where k <= q
        end = min(k_len, block_q_idx * BLOCK_M + BLOCK_M)
    else:
        end = k_len

    for block_k_idx in range(0, end, BLOCK_N):
        offset_seq_k = block_k_idx + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
        k_offsets = (
            (k_start + offset_seq_k[:, None]) * stride_kt
            + head_k_idx * stride_k_num_heads
            + offset_headdim[None, :] * stride_k_headdim
        )
        k_masks = offset_seq_k < k_len  # (BLOCK_N,)
        k = tl.load(
            k_ptr + k_offsets,
            mask=(k_masks[:, None] & (offset_headdim[None, :] < head_dim)),
            other=0.0,
        )  # (BLOCK_N, BLOCK_D)
        v = tl.load(
            v_ptr + k_offsets,
            mask=(k_masks[:, None] & (offset_headdim[None, :] < head_dim)),
            other=0.0,
        )  # (BLOCK_N, BLOCK_D)

        qk = tl.dot(q, tl.trans(k), input_precision="ieee")  # (BLOCK_M, BLOCK_N)
        qk *= softmax_scale

        # mask out the keys that are out of range, for both causal and non-causal attention
        qk = tl.where(offset_seq_k[None, :] < k_len, qk, float("-inf"))
        if causal:
            qk = tl.where(
                offset_seq[:, None] >= offset_seq_k[None, :], qk, float("-inf")
            )

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))  # (BLOCK_M,)
        p_ij = tl.exp(qk - m_i_new[:, None])  # (BLOCK_M, BLOCK_N)
        scale = tl.exp(m_i - m_i_new)  # (BLOCK_M,)

        acc = acc * scale[:, None]
        acc += tl.dot(p_ij.to(v.dtype), v, input_precision="ieee")  # (BLOCK_M, BLOCK_D)

        l_i_curr = tl.sum(p_ij, axis=1)  # (BLOCK_M,)
        l_i = l_i * scale + l_i_curr  # (BLOCK_M,)
        m_i = m_i_new

    o = acc / l_i[:, None]  # (BLOCK_M, BLOCK_D)
    output_offsets = (
        (q_start + offset_seq[:, None]) * stride_ot
        + head_q_idx * stride_o_num_heads
        + offset_headdim[None, :] * stride_o_headdim
    )
    tl.store(
        out_ptr + output_offsets,
        o,
        mask=(q_masks[:, None] & (offset_headdim[None, :] < head_dim)),
    )


def flash_attn_varlen_func(
    q: torch.Tensor,  # (total_tokens, num_heads, head_dim)
    k: torch.Tensor,  # (total_tokens, num_heads, head_dim)
    v: torch.Tensor,  # (total_tokens, num_heads, head_dim)
    cu_seqlens_q: torch.Tensor,  # (batch_size + 1,)
    cu_seqlens_k: torch.Tensor,  # (batch_size + 1,)
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float | None = None,
):
    """Flash attention for variable-length sequences, using cumulative sequence lengths.
    Supports GQA where num_heads_q can be different from num_heads_k, as long as num_heads_q is divisible by num_heads_k.

    Suppose we have three sequence:
        Sequence 0:  tokens [t0, t1, t2, t3, t4]               length = 5
        Sequence 1:  tokens [t5, t6, t7]                       length = 3
        Sequence 2:  tokens [t8, t9, t10, t11, t12, t13, t14]  length = 7

    Then we would have a flatten batch of tokens:
        [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14]
    with total_tokens = 15, and cumulative sequence lengths:
        cu_seqlens_q = [0, 5, 8, 15]
        cu_seqlens_k = [0, 5, 8, 15]

    Max sequence length for queries is max_seqlen_q = 7, and for keys/values is max_seqlen_k = 7.
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q, k, v must be 3D tensors")

    total_q_tokens, num_heads_q, head_dim = q.shape
    total_k_tokens, num_heads_k, _ = k.shape

    if num_heads_q % num_heads_k != 0:
        raise ValueError("num_heads_q must be divisible by num_heads_k for GQA")

    batch = cu_seqlens_q.shape[0] - 1

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    BLOCK_M: int = 64
    BLOCK_N: int = 32
    BLOCK_D = triton.next_power_of_2(head_dim)

    out = torch.empty_like(q)
    # grid: (num_tokens_per_seq, batch, num_heads)
    num_tokens_per_seq = triton.cdiv(max_seqlen_q, BLOCK_M)
    grid = (num_tokens_per_seq, batch, num_heads_q)
    _flash_attn_varlen_kernel[grid](
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        num_heads_q,
        num_heads_k,
        head_dim,
        causal,
        softmax_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return out


# ---------------------------------------------------------------------------
# Correctness test against PyTorch eager attention
# ---------------------------------------------------------------------------
def _reference_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal, sm_scale):
    """Naive per-sequence attention for correctness checking."""
    batch = cu_seqlens_q.shape[0] - 1
    nheads_q = q.shape[1]
    nheads_k = k.shape[1]
    heads_per_group = nheads_q // nheads_k
    out = torch.zeros_like(q)

    for b in range(batch):
        qs = cu_seqlens_q[b].item()
        qe = cu_seqlens_q[b + 1].item()
        ks = cu_seqlens_k[b].item()
        ke = cu_seqlens_k[b + 1].item()
        sq = qe - qs
        sk = ke - ks

        for hq in range(nheads_q):
            hk = hq // heads_per_group
            qi = q[qs:qe, hq]  # (sq, d)
            ki = k[ks:ke, hk]  # (sk, d)
            vi = v[ks:ke, hk]  # (sk, d)
            s = (qi @ ki.T) * sm_scale  # (sq, sk)
            if causal:
                offset = sk - sq
                row_idx = torch.arange(sq, device=q.device)[:, None]
                col_idx = torch.arange(sk, device=q.device)[None, :]
                mask = (row_idx + offset) >= col_idx
                s = s.masked_fill(~mask, float("-inf"))
            p = torch.softmax(s.float(), dim=-1).to(q.dtype)
            out[qs:qe, hq] = p @ vi
    return out


def test_flash_attn_varlen():
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16

    batch = 4
    nheads_q, nheads_k, headdim = 16, 4, 64  # GQA: 4 groups

    # Random sequence lengths
    seqlens_q = torch.randint(32, 257, (batch,))
    seqlens_k = seqlens_q.clone()  # self-attention

    cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, 0)
    cu_seqlens_k[1:] = torch.cumsum(seqlens_k, 0)

    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()
    max_sq = seqlens_q.max().item()
    max_sk = seqlens_k.max().item()

    q = torch.randn(total_q, nheads_q, headdim, device=device, dtype=dtype)
    k = torch.randn(total_k, nheads_k, headdim, device=device, dtype=dtype)
    v = torch.randn(total_k, nheads_k, headdim, device=device, dtype=dtype)

    softmax_scale = 1.0 / math.sqrt(headdim)

    for causal in [False, True]:
        out_triton = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_sq,
            max_sk,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        out_ref = _reference_attn_varlen(
            q, k, v, cu_seqlens_q, cu_seqlens_k, causal, softmax_scale
        )

        max_diff = (out_triton - out_ref).abs().max().item()
        mean_diff = (out_triton - out_ref).abs().mean().item()
        print(f"causal={causal:5}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
        assert max_diff < 5e-2, f"Max diff too large: {max_diff}"  # fp16 tolerance
        assert mean_diff < 1e-3, f"Mean diff too large: {mean_diff}"

    print("All tests passed!")


if __name__ == "__main__":
    test_flash_attn_varlen()
