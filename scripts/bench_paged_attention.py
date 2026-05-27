"""Benchmark patched paged-attention Qwen3 against the stock HuggingFace model.

Measures prefill latency (ms), prefill throughput (tokens/s), decode throughput
(tokens/s), and end-to-end latency across four workload scenarios: uniform short
sequences, ragged serving mixes, block-boundary edge cases, and long-context decode.
Each scenario runs both implementations under identical inputs so results are directly
comparable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from prettytable import PrettyTable
from transformers import AutoModelForCausalLM, PreTrainedModel

from server.model.inference_context import InferenceContext, inference_context
from server.model.patches.qwen3 import qwen3_model_loader
from tests.model.paged_helpers import allocate_block_tables, build_prefill_inputs

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "float32"
DEFAULT_BLOCK_SIZE = 4
DEFAULT_MEMORY_UTILIZATION = 0.2
DEFAULT_WARMUP = 2
DEFAULT_ITERS = 5
DEFAULT_SEED = 1234


@dataclass(frozen=True)
class Workload:
    name: str
    batch_sizes: tuple[int, ...]
    prompt_lengths: tuple[int, ...]
    decode_steps: int


@dataclass(frozen=True)
class BenchResult:
    prefill_ms: float
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    e2e_ms: float
    e2e_tokens_per_s: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark patched Qwen3 paged attention against stock HuggingFace."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=DEFAULT_DTYPE,
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--memory-utilization", type=float, default=DEFAULT_MEMORY_UTILIZATION
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--scenario",
        choices=[
            "uniform_short",
            "ragged_serving",
            "block_boundary",
            "long_context_decode",
            "all",
        ],
        default="all",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        help="Skip workload batch sizes larger than this value.",
    )
    return parser


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _median_seconds(
    fn: Callable[[], None], device: str, warmup: int, iters: int
) -> float:
    """Run `warmup` un-timed calls then `iters` timed calls and return the median.

    Device sync before and after each timed call ensures all GPU work has completed
    before the timestamp is recorded.
    """
    for _ in range(warmup):
        fn()
    _sync(device)

    samples = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _median_seconds_with_setup(
    setup: Callable[[], Any],
    fn: Callable[[Any], None],
    device: str,
    warmup: int,
    iters: int,
) -> float:
    """Like `_median_seconds` but calls `setup()` before each timed call.

    The value returned by `setup()` is passed to `fn(state)`, so KV-cache state is
    freshly initialized each iteration and the setup cost is excluded from the
    measurement.
    """
    for _ in range(warmup):
        fn(setup())
    _sync(device)

    samples = []
    for _ in range(iters):
        state = setup()
        _sync(device)
        t0 = time.perf_counter()
        fn(state)
        _sync(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _stable_seed(seed: int, *parts: object) -> int:
    """Derive a deterministic seed from a base seed and identifying parts.

    Each (workload, batch) combination gets a reproducible but distinct seed without
    needing an explicit lookup table.
    """
    return (
        int.from_bytes(
            hashlib.blake2b(repr((seed,) + parts).encode(), digest_size=8).digest(),
            "big",
        )
        % 100_000
    )


def _build_padded_inputs(
    seq_token_lists: list[list[int]], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build right-padded inputs for the stock HuggingFace model.

    Returns `(input_ids, attention_mask)` with a uniform batch shape; the mask is 0
    for padding positions.
    """
    batch = len(seq_token_lists)
    max_len = max(len(seq) for seq in seq_token_lists)
    input_ids = torch.zeros((batch, max_len), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long, device=device)

    for i, seq in enumerate(seq_token_lists):
        seq_tensor = torch.tensor(seq, dtype=torch.long, device=device)
        input_ids[i, : len(seq)] = seq_tensor
        attention_mask[i, : len(seq)] = 1

    return input_ids, attention_mask


def _rand_sequences(
    model: PreTrainedModel,
    prompt_lengths: list[int],
    device: str,
    seed: int,
) -> list[list[int]]:
    """Generate random token-ID sequences of the requested lengths.

    Uses a seeded generator so benchmarks are reproducible; each sequence has the
    exact requested length with no padding.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    seqs = []
    for length in prompt_lengths:
        ids = torch.randint(
            0,
            model.config.vocab_size,
            (length,),
            generator=generator,
            device=device,
        )
        seqs.append(ids.tolist())
    return seqs


def _rand_decode_tokens(
    model: PreTrainedModel,
    batch: int,
    steps: int,
    device: str,
    seed: int,
) -> list[list[int]]:
    """Pre-generate all decode step tokens as a list of per-step token vectors.

    Pre-generating avoids perturbing decode timing with token-sampling overhead;
    each inner list has one token per sequence in the batch.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (steps, batch),
        generator=generator,
        device=device,
    )
    return [row.tolist() for row in tokens]


@torch.no_grad()
def _patched_prefill(
    model: PreTrainedModel,
    seq_token_lists: list[list[int]],
    block_tables: list[list[int]],
    device: str,
) -> None:
    input_ids, position_ids, ctx = build_prefill_inputs(
        seq_token_lists, block_tables, device
    )
    with inference_context(ctx):
        model(input_ids=input_ids, position_ids=position_ids, use_cache=False)


@torch.no_grad()
def _patched_decode_step(
    model: PreTrainedModel,
    next_tokens: list[int],
    positions: list[int],
    block_tables: list[list[int]],
    device: str,
) -> None:
    input_ids = torch.tensor([next_tokens], dtype=torch.long, device=device)
    position_ids = torch.tensor([positions], dtype=torch.long, device=device)
    sequences = [{"num_tokens": 1, "block_table": bt} for bt in block_tables]
    ctx = InferenceContext(mode="decode", sequences=sequences)
    with inference_context(ctx):
        model(input_ids=input_ids, position_ids=position_ids, use_cache=False)


@torch.no_grad()
def _original_prefill(
    model: PreTrainedModel,
    seq_token_lists: list[list[int]],
    device: str,
) -> None:
    input_ids, attention_mask = _build_padded_inputs(seq_token_lists, device)
    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)


@torch.no_grad()
def _original_decode_loop(
    model: PreTrainedModel,
    seq_token_lists: list[list[int]],
    decode_tokens: list[list[int]],
    device: str,
) -> None:
    input_ids, attention_mask = _build_padded_inputs(seq_token_lists, device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values

    full_attention_mask = attention_mask
    for next_tokens in decode_tokens:
        nxt = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)
        full_attention_mask = torch.cat(
            [
                full_attention_mask,
                torch.ones(
                    (len(seq_token_lists), 1),
                    dtype=full_attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        out = model(
            input_ids=nxt,
            attention_mask=full_attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values


@torch.no_grad()
def _original_decode_setup(
    model: PreTrainedModel,
    seq_token_lists: list[list[int]],
    device: str,
) -> tuple[object, torch.Tensor]:
    input_ids, attention_mask = _build_padded_inputs(seq_token_lists, device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    return out.past_key_values, attention_mask


@torch.no_grad()
def _original_decode_cached(
    model: PreTrainedModel,
    decode_tokens: list[list[int]],
    state: tuple[object, torch.Tensor],
    device: str,
) -> None:
    past, full_attention_mask = state
    for next_tokens in decode_tokens:
        nxt = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)
        full_attention_mask = torch.cat(
            [
                full_attention_mask,
                torch.ones(
                    (len(next_tokens), 1),
                    dtype=full_attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        out = model(
            input_ids=nxt,
            attention_mask=full_attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values


def _patched_decode_loop(
    model: PreTrainedModel,
    seq_token_lists: list[list[int]],
    block_tables: list[list[int]],
    decode_tokens: list[list[int]],
    device: str,
) -> None:
    _patched_prefill(model, seq_token_lists, block_tables, device)
    positions = [len(seq) for seq in seq_token_lists]
    for next_tokens in decode_tokens:
        _patched_decode_step(model, next_tokens, positions, block_tables, device)
        positions = [position + 1 for position in positions]


def _patched_decode_setup(
    model: PreTrainedModel,
    seq_token_lists: list[list[int]],
    block_tables: list[list[int]],
    device: str,
) -> list[int]:
    _patched_prefill(model, seq_token_lists, block_tables, device)
    return [len(seq) for seq in seq_token_lists]


def _patched_decode_cached(
    model: PreTrainedModel,
    block_tables: list[list[int]],
    decode_tokens: list[list[int]],
    positions: list[int],
    device: str,
) -> None:
    for next_tokens in decode_tokens:
        _patched_decode_step(model, next_tokens, positions, block_tables, device)
        positions = [position + 1 for position in positions]


def _bench_patched(
    model: PreTrainedModel,
    seq_token_lists: list[list[int]],
    decode_tokens: list[list[int]],
    block_size: int,
    device: str,
    warmup: int,
    iters: int,
) -> BenchResult:
    """Benchmark the paged-attention model and return per-phase metrics.

    Prefill and decode are timed independently (decode uses the setup/fn split so
    prefill cost is excluded from the decode measurement), then a combined e2e pass
    is timed. Returns a `BenchResult` with latency (ms) and throughput (tokens/s)
    for each phase.
    """
    prompt_tokens = sum(len(seq) for seq in seq_token_lists)
    decode_tokens_count = len(seq_token_lists) * len(decode_tokens)
    block_tables = allocate_block_tables(
        [len(seq) + len(decode_tokens) for seq in seq_token_lists], block_size
    )

    prefill_s = _median_seconds(
        lambda: _patched_prefill(model, seq_token_lists, block_tables, device),
        device,
        warmup,
        iters,
    )
    decode_s = _median_seconds_with_setup(
        lambda: _patched_decode_setup(model, seq_token_lists, block_tables, device),
        lambda positions: _patched_decode_cached(
            model, block_tables, decode_tokens, positions, device
        ),
        device,
        warmup,
        iters,
    )
    e2e_s = _median_seconds(
        lambda: _patched_decode_loop(
            model, seq_token_lists, block_tables, decode_tokens, device
        ),
        device,
        warmup,
        iters,
    )
    return BenchResult(
        prefill_ms=prefill_s * 1e3,
        prefill_tokens_per_s=prompt_tokens / prefill_s,
        decode_tokens_per_s=decode_tokens_count / decode_s,
        e2e_ms=e2e_s * 1e3,
        e2e_tokens_per_s=(prompt_tokens + decode_tokens_count) / e2e_s,
    )


def _bench_original(
    model: PreTrainedModel,
    seq_token_lists: list[list[int]],
    decode_tokens: list[list[int]],
    device: str,
    warmup: int,
    iters: int,
) -> BenchResult:
    """Benchmark the stock HuggingFace model using padded inputs and HF KV-cache.

    Same measurement structure as `_bench_patched` so results are directly comparable.
    """
    prompt_tokens = sum(len(seq) for seq in seq_token_lists)
    decode_tokens_count = len(seq_token_lists) * len(decode_tokens)

    prefill_s = _median_seconds(
        lambda: _original_prefill(model, seq_token_lists, device),
        device,
        warmup,
        iters,
    )
    decode_s = _median_seconds_with_setup(
        lambda: _original_decode_setup(model, seq_token_lists, device),
        lambda state: _original_decode_cached(model, decode_tokens, state, device),
        device,
        warmup,
        iters,
    )
    e2e_s = _median_seconds(
        lambda: _original_decode_loop(model, seq_token_lists, decode_tokens, device),
        device,
        warmup,
        iters,
    )
    return BenchResult(
        prefill_ms=prefill_s * 1e3,
        prefill_tokens_per_s=prompt_tokens / prefill_s,
        decode_tokens_per_s=decode_tokens_count / decode_s,
        e2e_ms=e2e_s * 1e3,
        e2e_tokens_per_s=(prompt_tokens + decode_tokens_count) / e2e_s,
    )


def _scenario_prompt_lengths(workload: Workload, batch: int) -> list[int]:
    """Cycle through `workload.prompt_lengths` to produce exactly `batch` lengths.

    The base set is repeated and truncated so length variety is distributed evenly
    across all sequences in the batch.
    """
    base = list(workload.prompt_lengths)
    repeats = math.ceil(batch / len(base))
    return (base * repeats)[:batch]


def _workloads(block_size: int) -> dict[str, Workload]:
    """Return the four benchmark workload definitions.

    `block_boundary` prompt lengths are derived from `block_size` at construction time
    so they straddle block edges — the interesting case for paged-attention correctness
    and performance.
    """
    return {
        "uniform_short": Workload("uniform_short", (1, 4, 16), (128,), 32),
        "ragged_serving": Workload(
            "ragged_serving", (4, 8, 16), (16, 64, 128, 512), 32
        ),
        "block_boundary": Workload(
            "block_boundary",
            (4, 16),
            (max(1, block_size - 1), block_size, block_size + 1, 2 * block_size + 1),
            16,
        ),
        "long_context_decode": Workload("long_context_decode", (1, 4), (512, 1024), 64),
    }


def _format_ratio(patched: float, original: float, higher_is_better: bool) -> str:
    """Format a performance ratio with a consistent "patched speedup" convention.

    Computes `patched / original` for higher-is-better metrics and `original / patched`
    for lower-is-better metrics so >1.0 always means patched is better and <1.0 always
    means patched is worse.
    """
    if higher_is_better:
        if original == 0:
            return "n/a"
        ratio = patched / original
    else:
        if patched == 0:
            return "n/a"
        ratio = original / patched
    return f"{ratio:6.2f}x"


def _prompt_summary(prompt_lengths: list[int]) -> str:
    if min(prompt_lengths) != max(prompt_lengths):
        return f"{min(prompt_lengths)}-{max(prompt_lengths)}"
    return str(prompt_lengths[0])


def _build_row(
    batch: int,
    prompt_lengths: list[int],
    patched: BenchResult,
    original: BenchResult,
) -> list[str | int]:
    return [
        batch,
        _prompt_summary(prompt_lengths),
        f"{patched.prefill_ms:.2f}",
        f"{original.prefill_ms:.2f}",
        _format_ratio(patched.prefill_ms, original.prefill_ms, False),
        f"{patched.prefill_tokens_per_s:.1f}",
        f"{original.prefill_tokens_per_s:.1f}",
        _format_ratio(
            patched.prefill_tokens_per_s, original.prefill_tokens_per_s, True
        ),
        f"{patched.decode_tokens_per_s:.1f}",
        f"{original.decode_tokens_per_s:.1f}",
        _format_ratio(patched.decode_tokens_per_s, original.decode_tokens_per_s, True),
        f"{patched.e2e_ms:.2f}",
        f"{original.e2e_ms:.2f}",
        _format_ratio(patched.e2e_ms, original.e2e_ms, True),
    ]


def _build_results_table() -> PrettyTable:
    table = PrettyTable()
    table.field_names = [
        "batch",
        "prompt_len",
        "patched_pre_ms",
        "original_pre_ms",
        "pre_ratio",
        "patched_ptok/s",
        "original_ptok/s",
        "pre_tps_ratio",
        "patched_dtok/s",
        "original_dtok/s",
        "decode_tps_ratio",
        "patched_e2e_ms",
        "original_e2e_ms",
        "e2e_ratio",
    ]
    table.align = "r"
    table.align["prompt_len"] = "l"
    return table


def _load_models(
    model_name: str,
    dtype: torch.dtype,
    device: str,
    memory_utilization: float,
    block_size: int,
) -> tuple[PreTrainedModel, PreTrainedModel]:
    """Load two independent model copies — original and patched — on the same device.

    Both copies use the same dtype so hardware conditions are identical; the patched
    copy has paged-attention applied via `qwen3_model_loader`. Returns `(patched,
    original)`.
    """
    print(f"loading original model: {model_name}")
    original = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    original.eval()

    print(f"loading patched model:  {model_name}")
    patched = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    patched.eval()
    qwen3_model_loader(
        patched,
        patched.config,
        memory_utilization=memory_utilization,
        block_size=block_size,
        dtype=dtype,
        device=device,
    )
    return patched, original


def _run_workload(
    workload: Workload,
    patched_model: PreTrainedModel,
    original_model: PreTrainedModel,
    args: argparse.Namespace,
) -> None:
    """Run both models for each batch size in the workload and print a comparison table."""
    print(
        f"\n=== {workload.name} | decode_steps={workload.decode_steps} "
        f"| block_size={args.block_size} ==="
    )
    table = _build_results_table()

    for batch in workload.batch_sizes:
        if args.max_batch_size is not None and batch > args.max_batch_size:
            continue

        prompt_lengths = _scenario_prompt_lengths(workload, batch)
        seqs = _rand_sequences(
            patched_model,
            prompt_lengths,
            args.device,
            _stable_seed(args.seed, workload.name, batch),
        )
        decode_tokens = _rand_decode_tokens(
            patched_model,
            batch,
            workload.decode_steps,
            args.device,
            _stable_seed(args.seed, workload.name, batch, "decode"),
        )

        patched = _bench_patched(
            patched_model,
            seqs,
            decode_tokens,
            args.block_size,
            args.device,
            args.warmup,
            args.iters,
        )
        original = _bench_original(
            original_model,
            seqs,
            decode_tokens,
            args.device,
            args.warmup,
            args.iters,
        )
        table.add_row(_build_row(batch, prompt_lengths, patched, original))

    print(table)


def main(argv: list[str] | None = None) -> int:
    """Parse args, load models once, run selected workload(s), and free GPU memory."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the default paged-attention benchmark")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("--warmup must be non-negative and --iters must be positive")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")

    dtype = _dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)

    workloads = _workloads(args.block_size)
    selected = (
        list(workloads.values())
        if args.scenario == "all"
        else [workloads[args.scenario]]
    )

    patched_model, original_model = _load_models(
        args.model_name,
        dtype,
        args.device,
        args.memory_utilization,
        args.block_size,
    )

    print(
        f"\nmodel={args.model_name} device={args.device} dtype={args.dtype} "
        f"warmup={args.warmup} iters={args.iters} seed={args.seed}"
    )
    try:
        for workload in selected:
            _run_workload(workload, patched_model, original_model, args)
    finally:
        del patched_model
        del original_model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
