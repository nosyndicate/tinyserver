"""Helpers to drive the patched paged-attention Qwen3 model directly.

The engine does not yet populate the inference context, so tests build the
flattened ``input_ids`` / ``position_ids``, assign ``block_table``s, and wrap the
forward in ``inference_context(...)`` themselves. These helpers call the real
patched model (no mocking) so the tests exercise the actual paged-attention path.
"""

from __future__ import annotations

import math

import torch
from transformers import PreTrainedModel

from server.model.inference_context import InferenceContext, inference_context


def allocate_block_tables(
    num_tokens_in_seqs: list[int], block_size: int
) -> list[list[int]]:
    """Assign each sequence a disjoint run of physical block ids.

    ``num_tokens_in_seqs[i]`` is the maximum number of tokens sequence ``i`` will ever hold
    (prompt + generated), so decode steps never run past the allocated blocks.
    """
    block_tables: list[list[int]] = []
    next_block = 0
    for num_tokens in num_tokens_in_seqs:
        num_blocks = max(1, math.ceil(num_tokens / block_size))
        block_tables.append(list(range(next_block, next_block + num_blocks)))
        next_block += num_blocks
    return block_tables


def build_prefill_inputs(
    seq_token_lists: list[list[int]],
    block_tables: list[list[int]],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, InferenceContext]:
    """Flatten a batch of prompts into the patched model's prefill input format."""
    flat_input_ids: list[int] = []
    flat_position_ids: list[int] = []
    sequences = []
    for toks, block_table in zip(seq_token_lists, block_tables):
        flat_input_ids.extend(toks)
        flat_position_ids.extend(range(len(toks)))
        sequences.append({"num_tokens": len(toks), "block_table": block_table})

    input_ids = torch.tensor([flat_input_ids], dtype=torch.long, device=device)
    position_ids = torch.tensor([flat_position_ids], dtype=torch.long, device=device)
    ctx = InferenceContext(mode="prefill", sequences=sequences)
    return input_ids, position_ids, ctx


def patched_prefill_logits(
    model: PreTrainedModel,
    seq_token_lists: list[list[int]],
    block_tables: list[list[int]],
    device: str,
) -> torch.Tensor:
    """Prefill the batch; return last-token logits per sequence, shape (B, vocab)."""
    input_ids, position_ids, ctx = build_prefill_inputs(
        seq_token_lists, block_tables, device
    )
    with torch.no_grad(), inference_context(ctx):
        out = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    logits = out.logits  # (1, total_tokens, vocab)

    last_indices = []
    offset = 0
    for toks in seq_token_lists:
        offset += len(toks)
        last_indices.append(offset - 1)
    return logits[0, last_indices, :]


def patched_decode_logits(
    model: PreTrainedModel,
    next_tokens: list[int],
    positions: list[int],
    block_tables: list[list[int]],
    device: str,
) -> torch.Tensor:
    """One decode step for a batch; return next-token logits per sequence (B, vocab).

    ``positions[i]`` is the absolute position of ``next_tokens[i]`` (i.e. the
    sequence's length before this token is appended).
    """
    input_ids = torch.tensor([next_tokens], dtype=torch.long, device=device)
    position_ids = torch.tensor([positions], dtype=torch.long, device=device)
    sequences = [{"num_tokens": 1, "block_table": bt} for bt in block_tables]
    ctx = InferenceContext(mode="decode", sequences=sequences)
    with torch.no_grad(), inference_context(ctx):
        out = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    return out.logits[0, :, :]  # (B, vocab)


def original_last_logits(
    model: PreTrainedModel, token_ids: list[int], device: str
) -> torch.Tensor:
    """Full forward of the unpatched model on a prefix; last-token logits (vocab,)."""
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    return out.logits[0, -1, :]


def patched_greedy_batch(
    model: PreTrainedModel,
    prompts_ids: list[list[int]],
    max_new_tokens: int,
    block_tables: list[list[int]],
    device: str,
) -> list[list[int]]:
    """Greedy generate for a batch via patched prefill + per-step batched decode."""
    last_logits = patched_prefill_logits(model, prompts_ids, block_tables, device)
    cur_lens = [len(p) for p in prompts_ids]
    generated: list[list[int]] = [[] for _ in prompts_ids]

    next_tokens = [int(last_logits[i].argmax()) for i in range(len(prompts_ids))]
    for i, tok in enumerate(next_tokens):
        generated[i].append(tok)

    for _ in range(max_new_tokens - 1):
        logits = patched_decode_logits(
            model, next_tokens, cur_lens, block_tables, device
        )
        cur_lens = [n + 1 for n in cur_lens]
        next_tokens = [int(logits[i].argmax()) for i in range(len(prompts_ids))]
        for i, tok in enumerate(next_tokens):
            generated[i].append(tok)
    return generated


def original_greedy_single(
    model: PreTrainedModel,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: str,
) -> list[int]:
    """Greedy generate for one prompt via repeated full forwards (exact baseline)."""
    ids = list(prompt_ids)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits = original_last_logits(model, ids, device)
        tok = int(logits.argmax())
        generated.append(tok)
        ids.append(tok)
    return generated
