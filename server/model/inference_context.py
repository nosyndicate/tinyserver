"""Process-global inference context for the patched paged-attention model.

The patched attention forward (``server/model/patches/qwen3.py``) reads the
active context via ``get_inference_context()``. The current context is a
module-level global swapped by the ``inference_context`` context manager.

THREAD SAFETY: this global is intentionally unsynchronized. It is correct only
because a single worker thread drives the engine loop (the model forward and
the context swap happen on the same thread). A second engine thread sharing
the model would clobber the global mid-forward.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass
class InferenceContext:
    mode: str = "prefill"
    sequences: list[dict[str, Any]] = field(default_factory=list)


_inference_context = InferenceContext()


def get_inference_context() -> InferenceContext:
    """Get the current inference context."""
    return _inference_context


@contextmanager
def inference_context(
    new_context: InferenceContext,
) -> Generator[InferenceContext, None, None]:
    """Context manager: set the active inference context for the block inside.

    Swaps the module-global ``_inference_context`` (see the module docstring
    for the single-thread assumption) and restores the previous one on exit.
    """
    global _inference_context
    old_context = _inference_context
    _inference_context = new_context
    try:
        yield _inference_context
    finally:
        _inference_context = old_context
