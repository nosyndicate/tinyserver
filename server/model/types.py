from dataclasses import dataclass
from typing import Protocol

import torch

from server.executor.types import Sequence


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3-1.7B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32


class ModelBackend(Protocol):
    def prefill_batch(self, sequences: list[Sequence]) -> None: ...

    def decode_batch(self, sequences: list[Sequence]) -> None: ...

    def release(self) -> None: ...
