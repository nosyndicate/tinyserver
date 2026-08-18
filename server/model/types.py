from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch


class FinishReason(str, Enum):
    EOS = "eos"
    MAX_LENGTH = "max_length"


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3-1.7B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    block_size: int = 256
    memory_utilization: float = 0.2


class ModelBackend(Protocol):
    device: str

    def tokenize(self, prompt: str) -> list[int]: ...

    def release(self) -> None: ...
