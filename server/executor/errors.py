import torch

from server.executor.types import FailureCode


def classify_exception(error: BaseException) -> FailureCode:
    """Map an exception to a stable public failure category."""
    oom_type = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
    if isinstance(error, oom_type):
        return FailureCode.CUDA_OUT_OF_MEMORY
    if isinstance(error, RuntimeError) and "cuda out of memory" in str(error).lower():
        return FailureCode.CUDA_OUT_OF_MEMORY
    return FailureCode.GENERATION_ERROR
