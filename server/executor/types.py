from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from queue import Queue

import torch
from torch import Tensor
from transformers import DynamicCache

from server.model.sampling import SamplingParams


@dataclass(frozen=True)
class TokenEvent:
    """
    Indicates a new token is generated.

    Attributes:
        token: The decoded text for this token, might be an empty string for special tokens like EOS.
        is_first: Whether this token is the first token in the sequence.
        is_last: Whether this token is the last token in the sequence. This can be true when the generation is stopped by max_new_tokens limit.
        index: The index of the token in the sequence, starting from 0.
    """

    token: str
    is_first: bool
    is_last: bool
    index: int


@dataclass(frozen=True)
class DoneEvent:
    """
    Indicates the generation is done, and provides the final results.

    Attributes:
        text: The full decoded text for the sequence, including all tokens.
        num_output: The number of tokens in the output sequence.
        ttft: The time to first token in milliseconds.
        total_ms: The total time taken for the sequence generation in milliseconds.

    """

    text: str
    num_output: int

    ttft: float
    total_ms: float


@dataclass(frozen=True)
class ErrorEvent:
    """
    Indicates that the generation has failed.

    Attributes:
        request_id: The unique identifier for the request, used for logging and tracking.
        error: The error message.
    """

    request_id: str
    error: str


@dataclass(frozen=True)
class ExecutorConfig:
    """
    The configuration for the executor, controls the backpressure.

    Attributes:
        max_queue_size: The maximum number of requests that can wait in the queue. If the queue is full, new requests will get an HTTP 503 error.
        max_active_requests: The maximum number of requests that the worker can process concurrently.
            If there are already max_active_requests in processing, new requests will wait in the queue until there is an available slot.
    """

    max_queue_size: int = field(default=64)
    max_active_requests: int = field(default=10)


class RequestStatus(Enum):
    QUEUED = "queued"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FinishReason(Enum):
    EOS = "eos"
    MAX_LENGTH = "max_length"


@dataclass
class GenerationRequestState:
    """Mutable state for a single in-flight generation request."""

    request_id: str
    sampling_params: SamplingParams
    prompt: str

    status: RequestStatus = RequestStatus.QUEUED

    # Prefill state
    all_logits: Tensor | None = None
    past_key_values: DynamicCache | None = None
    num_prompt_tokens: int | None = None

    # Decoding state
    first_token_ns: int | None = None
    start_ns: int | None = None
    output_tokens: list[str] = field(default_factory=list)

    # Final results
    finished_reason: FinishReason | None = None
    error: str | None = None

    output_queue: Queue[TokenEvent | DoneEvent | ErrorEvent] = field(
        default_factory=Queue
    )

    generator: torch.Generator | None = None

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_tokens)
