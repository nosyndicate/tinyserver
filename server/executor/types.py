"""
Define the various dataclasses for the executor.
"""

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
        ttft: The total time taken for the sequence generation.
        total_ms: The total time taken for the sequence generation in milliseconds.

    """

    text: str
    num_output: int

    ttft: float
    total_ms: float


@dataclass(frozen=True)
class ErrorEvent:
    """
    Indicate the generation is failed.

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
        max_queue_size: Max number of request that can wait in the queue. If the queue is full, new requests will get HTTP 503 error.
        max_active_requests: Max number of requests that the worker can process concurrently.
            If there are already max_active_requests in processing, new requests will wait in the queue until there is an available slot.
    """

    max_queue_size: int = field(default=64)
    max_active_requests: int = field(default=10)


class RequestStatus(Enum):
    """
    The lifecycle status of a generation request.

    State machine:
    QUEUED -> PREFILLING -> DECODING -> DONE
                     |            |
                     v            v
                 FAILED        FAILED
                     |            |
                     v            v
                 CANCELLED     CANCELLED
    """

    QUEUED = "queued"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationRequestState:
    """
    Mutable state for a single in-flight generation request.

    This object is created by the HTTP handler, submitted to the worker queue.
    And then updated exclusively by the worker until the request is done, failed or cancelled.

    Attributes:
        request_id: A unique identifier for the request, used for logging and tracking.
        sampling_params: The sampling parameters for the request.
        status: The current lifecycle status of the request.

        all_logits: The tensor we will sample the next token id from. It is updated after prefill and each decoding step.
        past_key_values: The cached key values for the transformer model, updated after prefill and each decoding step.
        output_tokens: The number of tokens generated so far, used for enforcing max_new_tokens limit

        finished_reason: Why generation stopped: 'eos', 'max_length' or None (not finished yet).
        error: The error message if the request failed, None otherwise.

    """

    request_id: str
    sampling_params: SamplingParams
    prompt: str

    # LifeCycle status of the request, used for scheduling and monitoring
    status: RequestStatus = RequestStatus.QUEUED

    # Prefill related states
    all_logits: Tensor | None = None
    past_key_values: DynamicCache | None = None
    num_prompt_tokens: int | None = None

    # Decoding related states
    num_output_tokens: int = 0
    first_token_ns: int | None = None
    start_ns: int | None = None
    output_tokens: list[str] = field(default_factory=list)

    # Final results
    finished_reason: str | None = None
    error: str | None = None

    # Communication channel to send events back to the HTTP handler
    output_queue: Queue[TokenEvent | DoneEvent | ErrorEvent] = field(
        default_factory=Queue
    )

    generator: torch.Generator | None = None
