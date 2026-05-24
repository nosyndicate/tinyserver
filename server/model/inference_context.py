from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator


@dataclass
class InferenceContext:
    mode: str = "prefill"
    sequences: list[dict[str, Any]] = []


_inference_context = InferenceContext()


def get_inference_context() -> InferenceContext:
    """Get the current inference context."""
    return _inference_context


@contextmanager
def inference_context(
    new_context: InferenceContext,
) -> Generator[InferenceContext, None, None]:
    """Context manager for managing inference context."""
    global _inference_context
    old_context = _inference_context
    _inference_context = new_context
    try:
        yield _inference_context
    finally:
        _inference_context = old_context
