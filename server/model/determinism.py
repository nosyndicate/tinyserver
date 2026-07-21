import os
from collections.abc import Sequence

import torch
import torch.backends.cudnn

_MASK64 = (1 << 64) - 1

# splitmix64 constants: the golden-ratio increment and the two finalizer multipliers.
_GOLDEN = 0x9E3779B97F4A7C15
_MIX_A = 0xBF58476D1CE4E5B9
_MIX_B = 0x94D049BB133111EB

# Largest float32 strictly below 1.0, used to keep uniforms inside [0, 1).
_MAX_UNIFORM = 1.0 - 2.0**-24


def make_generator(seed: int | None, device: str) -> torch.Generator | None:
    """Create a request-scoped torch.Generator seeded with the given seed.

    Returns None if seed is None, allowing callers to skip generator usage.
    """
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def mix64(seed: int, step: int) -> int:
    """Hash (seed, step) into a uniformly distributed 64-bit integer (splitmix64).

    Counter-based: the result depends only on its two arguments, never on any
    hidden RNG state.  That makes it batch-invariant (a request's noise does not
    depend on its position in the batch) and preemption-safe (nothing to
    checkpoint -- `step` is just the request's output-token count).
    """
    x = ((seed & _MASK64) * _GOLDEN + (step & _MASK64)) & _MASK64
    x = ((x ^ (x >> 30)) * _MIX_A) & _MASK64
    x = ((x ^ (x >> 27)) * _MIX_B) & _MASK64
    return x ^ (x >> 31)


def uniform_from_hash(seed: int, step: int) -> float:
    """One uniform in [0, 1) derived from (seed, step).

    Uses the top 53 bits of the hash (the float64 mantissa width), then clamps
    below 1.0 so callers can index a CDF without running off the end.
    """
    u = (mix64(seed, step) >> 11) * 2.0**-53
    return min(u, _MAX_UNIFORM)


def uniforms_from_seeds(
    seeds: Sequence[int],
    step: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the per-row uniform noise tensor [B] for a batched sampling step.

    Pure function of (seeds, step): row i's value depends only on `seeds[i]` and
    `step`, so a request draws the same noise regardless of batch size or of
    which row it lands on.
    """
    values = [uniform_from_hash(seed, step) for seed in seeds]
    return torch.tensor(values, dtype=dtype, device=device)


def configure_deterministic_mode() -> None:
    """Enable PyTorch deterministic mode for reproducible results.

    Intended for test fixtures/setup only, not production code.

    Note: CUBLAS_WORKSPACE_CONFIG must be set before CUDA/cuBLAS is
    initialized to reliably take effect.  When using this in tests,
    call it from a session-scoped conftest fixture (or set the env var
    in the test runner) before any torch.cuda operations occur.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
