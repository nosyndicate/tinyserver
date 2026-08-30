import torch

from server.executor.errors import classify_exception
from server.executor.types import FailureCode


def test_classifies_torch_cuda_oom() -> None:
    oom_type = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
    assert classify_exception(oom_type("allocation failed")) == (
        FailureCode.CUDA_OUT_OF_MEMORY
    )


def test_classifies_legacy_cuda_oom_runtime_error() -> None:
    error = RuntimeError("CUDA out of memory. Tried to allocate 368.00 MiB")
    assert classify_exception(error) == FailureCode.CUDA_OUT_OF_MEMORY


def test_does_not_classify_host_oom_or_generic_runtime_error() -> None:
    assert classify_exception(MemoryError("host allocation failed")) == (
        FailureCode.GENERATION_ERROR
    )
    assert classify_exception(RuntimeError("allocator failed")) == (
        FailureCode.GENERATION_ERROR
    )
