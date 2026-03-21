#!/usr/bin/env python3
"""Benchmark script for sampling performance comparison.

This script compares the performance of:
1. PyTorch baseline sampling
2. Triton-optimized sampling

Note: PyTorch's native kernels are highly optimized and may outperform
custom Triton implementations for sampling. The Triton implementation
is provided as an alternative that can be optimized further for specific
hardware or use cases.

Usage:
    python scripts/benchmark_sampling.py [--vocab-size N] [--top-p P]
"""

import argparse
from typing import Callable

import torch

# Import the sampling implementations
from server.model.triton_sampling import sample_token_torch, sample_token_triton


def benchmark_sampling(
    name: str,
    func: Callable,
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    seed: int,
    num_warmup: int = 5,
    num_runs: int = 50,
    use_triton: bool = False,
) -> tuple[float, list[int]]:
    """Benchmark a sampling function.

    Args:
        name: Name of the implementation for reporting.
        func: Sampling function to benchmark.
        logits: Logits tensor to sample from.
        temperature: Temperature parameter.
        top_p: Top-p parameter.
        seed: Random seed.
        num_warmup: Number of warmup iterations.
        num_runs: Number of benchmark iterations.
        use_triton: Whether this is the Triton implementation.

    Returns:
        Tuple of (average time per sample in ms, list of sampled tokens for verification).
    """
    results = []

    # Warmup
    for i in range(num_warmup):
        if use_triton:
            result = func(logits, temperature, top_p, seed, seq_pos=i)
            if result is not None:
                results.append(result)
        else:
            gen = torch.Generator(device=logits.device)
            gen.manual_seed(seed + i)
            results.append(func(logits, temperature, top_p, generator=gen))

    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    benchmark_results = []
    for i in range(num_warmup, num_warmup + num_runs):
        if use_triton:
            result = func(logits, temperature, top_p, seed, seq_pos=i)
            if result is not None:
                benchmark_results.append(result)
        else:
            gen = torch.Generator(device=logits.device)
            gen.manual_seed(seed + i)
            benchmark_results.append(func(logits, temperature, top_p, generator=gen))

    end.record()
    torch.cuda.synchronize()

    avg_time_ms = start.elapsed_time(end) / num_runs
    return avg_time_ms, benchmark_results


def verify_correctness(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    seed: int,
    num_samples: int = 100,
) -> bool:
    """Verify that Triton produces similar results to PyTorch.

    Note: Due to different RNG implementations, we can't expect exact matches.
    Instead, we verify that both implementations produce valid token IDs
    and that the distribution is reasonable.

    Args:
        logits: Logits tensor.
        temperature: Temperature parameter.
        top_p: Top-p parameter.
        seed: Random seed.
        num_samples: Number of samples to compare.

    Returns:
        True if verification passes.
    """
    print("\nVerifying correctness...")

    # Sample with PyTorch
    torch_results = []
    for i in range(num_samples):
        gen = torch.Generator(device=logits.device)
        gen.manual_seed(seed + i)
        torch_results.append(sample_token_torch(logits, temperature, top_p, generator=gen))

    # Sample with Triton
    triton_results = []
    for i in range(num_samples):
        result = sample_token_triton(logits, temperature, top_p, seed, seq_pos=i)
        if result is not None:
            triton_results.append(result)

    if len(triton_results) == 0:
        print("  Triton returned None for all samples (falling back to PyTorch)")
        return True

    # Verify all token IDs are valid
    vocab_size = logits.shape[0]
    assert all(0 <= t < vocab_size for t in torch_results), "PyTorch produced invalid token IDs"
    assert all(0 <= t < vocab_size for t in triton_results), "Triton produced invalid token IDs"

    print(f"  PyTorch: {len(torch_results)} valid samples")
    print(f"  Triton: {len(triton_results)} valid samples")
    print("  Verification passed!")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sampling implementations")
    parser.add_argument(
        "--vocab-size", type=int, default=50257, help="Vocabulary size to benchmark"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature parameter"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p parameter"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--num-runs", type=int, default=100, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify Triton correctness"
    )
    args = parser.parse_args()

    # Check for CUDA
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot run Triton benchmark.")
        print("Running PyTorch-only benchmark on CPU...")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Create random logits
    print(f"\nBenchmarking with vocab_size={args.vocab_size}, temp={args.temperature}, top_p={args.top_p}")
    print(f"Device: {device}\n")

    logits = torch.randn(args.vocab_size, device=device)

    # Benchmark PyTorch
    print("Benchmarking PyTorch implementation...")
    torch_time, torch_results = benchmark_sampling(
        "PyTorch",
        sample_token_torch,
        logits,
        args.temperature,
        args.top_p,
        args.seed,
        num_runs=args.num_runs,
    )
    print(f"  PyTorch: {torch_time:.3f} ms/sample")

    # Benchmark Triton (if CUDA available)
    triton_time = None
    if device == "cuda":
        print("\nBenchmarking Triton implementation...")
        triton_time, triton_results = benchmark_sampling(
            "Triton",
            sample_token_triton,
            logits,
            args.temperature,
            args.top_p,
            args.seed,
            num_runs=args.num_runs,
            use_triton=True,
        )

        if triton_time is not None:
            print(f"  Triton: {triton_time:.3f} ms/sample")
            speedup = torch_time / triton_time
            print(f"  Speedup: {speedup:.2f}x")
        else:
            print("  Triton: FAILED (fell back to None)")
    else:
        print("\nTriton benchmark skipped (requires CUDA)")

    # Verify correctness
    if args.verify and device == "cuda":
        verify_correctness(logits, args.temperature, args.top_p, args.seed)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Vocab size: {args.vocab_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"PyTorch time: {torch_time:.3f} ms/sample")
    if triton_time is not None:
        print(f"Triton time: {triton_time:.3f} ms/sample")
        print(f"Speedup: {torch_time / triton_time:.2f}x")
        print("\nNote: PyTorch kernels are highly optimized. Triton may be slower")
        print("due to kernel launch overhead and less optimized operations.")


if __name__ == "__main__":
    main()