import os

import torch
import torch.backends.cudnn


def make_generator(seed: int | None, device: str) -> torch.Generator | None:
    """Create a request-scoped torch.Generator seeded with the given seed.

    Returns None if seed is None, allowing callers to skip generator usage.
    """
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


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
