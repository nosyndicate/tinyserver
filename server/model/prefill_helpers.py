"""Shared helper for building the patched model's prefill inputs.

Used by the production engine (``server.executor.engine``) and by the
paged-attention test helpers / benchmark script, so the flattened
``input_ids`` / ``position_ids`` / ``InferenceContext`` format has one source
of truth.
"""

from __future__ import annotations

import torch

from server.model.inference_context import InferenceContext


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
