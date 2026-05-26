"""Correctness of the patched paged-attention Qwen3 model vs stock HuggingFace.

Two levels of checking:
- per-step logits ``allclose`` (prefill last-token + several decode steps), and
- exact greedy token-match over a generation loop (single sequence and a batch).
"""

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from tests.model.conftest import _PAGED_BLOCK_SIZE, requires_cuda
from tests.model.paged_helpers import (
    allocate_block_tables,
    original_greedy_single,
    original_last_logits,
    patched_decode_logits,
    patched_greedy_batch,
    patched_prefill_logits,
)

pytestmark = pytest.mark.slow


_DEVICE = "cuda"
_PREFILL_ATOL = 1e-4
_PREFILL_RTOL = 1e-4
_DECODE_ATOL = 1e-3
_DECODE_RTOL = 1e-3


def _encode(tokenizer: PreTrainedTokenizerFast, text: str) -> list[int]:
    return tokenizer(text, return_tensors="pt").input_ids[0].tolist()


@requires_cuda
def test_prefill_single_seq_matches_original(
    qwen3_patched_cuda: PreTrainedModel,
    qwen3_original_cuda: PreTrainedModel,
    qwen3_tokenizer: PreTrainedTokenizerFast,
) -> None:
    ids = _encode(qwen3_tokenizer, "The capital of France is")
    block_tables = allocate_block_tables([len(ids)], _PAGED_BLOCK_SIZE)

    patched = patched_prefill_logits(qwen3_patched_cuda, [ids], block_tables, _DEVICE)[
        0
    ]
    original = original_last_logits(qwen3_original_cuda, ids, _DEVICE)

    assert torch.allclose(patched, original, atol=_PREFILL_ATOL, rtol=_PREFILL_RTOL)
    assert int(patched.argmax()) == int(original.argmax())


@requires_cuda
def test_prefill_multi_seq_matches_original(
    qwen3_patched_cuda: PreTrainedModel,
    qwen3_original_cuda: PreTrainedModel,
    qwen3_tokenizer: PreTrainedTokenizerFast,
) -> None:
    prompts = [
        "Hello",
        "The quick brown fox jumps over",
        "In a galaxy far, far away there lived",
    ]
    seq_ids = [_encode(qwen3_tokenizer, p) for p in prompts]
    block_tables = allocate_block_tables(
        [len(ids) for ids in seq_ids], _PAGED_BLOCK_SIZE
    )

    patched = patched_prefill_logits(qwen3_patched_cuda, seq_ids, block_tables, _DEVICE)

    # Each flattened sequence must match the original model run on that prompt alone,
    # proving the block-diagonal mask isolates sequences from each other.
    for i, ids in enumerate(seq_ids):
        original = original_last_logits(qwen3_original_cuda, ids, _DEVICE)
        assert torch.allclose(
            patched[i], original, atol=_PREFILL_ATOL, rtol=_PREFILL_RTOL
        ), f"sequence {i} mismatch"
        assert int(patched[i].argmax()) == int(original.argmax())


@requires_cuda
def test_decode_steps_match_original(
    qwen3_patched_cuda: PreTrainedModel,
    qwen3_original_cuda: PreTrainedModel,
    qwen3_tokenizer: PreTrainedTokenizerFast,
) -> None:
    num_decode_steps = 4
    ids = _encode(qwen3_tokenizer, "The capital of France is")
    block_tables = allocate_block_tables(
        [len(ids) + num_decode_steps], _PAGED_BLOCK_SIZE
    )

    # Prefill, then greedily take the first token from the patched model.
    last = patched_prefill_logits(qwen3_patched_cuda, [ids], block_tables, _DEVICE)[0]
    grown = list(ids)
    next_token = int(last.argmax())

    for _ in range(num_decode_steps):
        position = len(grown)  # absolute position of next_token
        patched = patched_decode_logits(
            qwen3_patched_cuda, [next_token], [position], block_tables, _DEVICE
        )[0]
        grown.append(next_token)
        original = original_last_logits(qwen3_original_cuda, grown, _DEVICE)

        assert torch.allclose(patched, original, atol=_DECODE_ATOL, rtol=_DECODE_RTOL)
        assert int(patched.argmax()) == int(original.argmax())
        next_token = int(patched.argmax())


@requires_cuda
def test_greedy_generation_single_token_match(
    qwen3_patched_cuda: PreTrainedModel,
    qwen3_original_cuda: PreTrainedModel,
    qwen3_tokenizer: PreTrainedTokenizerFast,
) -> None:
    max_new_tokens = 20
    ids = _encode(qwen3_tokenizer, "The capital of France is")
    block_tables = allocate_block_tables([len(ids) + max_new_tokens], _PAGED_BLOCK_SIZE)

    patched = patched_greedy_batch(
        qwen3_patched_cuda, [ids], max_new_tokens, block_tables, _DEVICE
    )[0]
    original = original_greedy_single(qwen3_original_cuda, ids, max_new_tokens, _DEVICE)

    assert patched == original


@requires_cuda
def test_greedy_generation_batch_token_match(
    qwen3_patched_cuda: PreTrainedModel,
    qwen3_original_cuda: PreTrainedModel,
    qwen3_tokenizer: PreTrainedTokenizerFast,
) -> None:
    max_new_tokens = 16
    prompts = [
        "The capital of France is",
        "Once upon a time",
        "Two plus two equals",
    ]
    seq_ids = [_encode(qwen3_tokenizer, p) for p in prompts]
    block_tables = allocate_block_tables(
        [len(ids) + max_new_tokens for ids in seq_ids], _PAGED_BLOCK_SIZE
    )

    patched = patched_greedy_batch(
        qwen3_patched_cuda, seq_ids, max_new_tokens, block_tables, _DEVICE
    )
    # Batched decode must match each sequence generated independently.
    for i, ids in enumerate(seq_ids):
        original = original_greedy_single(
            qwen3_original_cuda, ids, max_new_tokens, _DEVICE
        )
        assert patched[i] == original, f"sequence {i} diverged"
