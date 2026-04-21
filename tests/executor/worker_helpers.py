"""Shared helpers for executor tests."""

import itertools
import queue
import time

import torch
from transformers import DynamicCache

from server.executor.types import (
    DecodeResult,
    ErrorEvent,
    FinishReason,
    GenerationRequestState,
    PrefillResult,
    RequestStatus,
)
from server.executor.worker import BatchWorker, SimpleWorker
from server.metrics.timers import NS_PER_S, now_ns
from server.model.sampling import SamplingParams

_counter = itertools.count()


def make_req(request_id: str | None = None) -> GenerationRequestState:
    rid = request_id if request_id is not None else f"req-{next(_counter)}"
    return GenerationRequestState(
        request_id=rid,
        sampling_params=SamplingParams(max_new_tokens=10, temperature=1.0, top_p=1.0),
        prompt="hello",
        enqueued_ns=0,
    )


def make_decoding_req(request_id: str) -> GenerationRequestState:
    req = make_req(request_id)
    req.status = RequestStatus.DECODING
    req.start_ns = 0
    req.enqueued_ns = 0
    req.num_prompt_tokens = 1
    req.all_logits = torch.empty(1, 1, 1)
    req.past_key_values = DynamicCache()
    return req


def drain_events(req: GenerationRequestState, timeout: float = 0.0) -> list:
    """Return all events currently in req.output_queue.

    If timeout > 0, block for up to that many seconds waiting for the first
    event, then drain the rest non-blocking.
    """
    events: list = []
    try:
        if timeout > 0:
            events.append(req.output_queue.get(timeout=timeout))
        else:
            events.append(req.output_queue.get_nowait())
    except queue.Empty:
        return events
    while True:
        try:
            events.append(req.output_queue.get_nowait())
        except queue.Empty:
            break
    return events


def wait_for_status(
    req: GenerationRequestState, status: RequestStatus, timeout: float = 2.0
) -> bool:
    deadline = now_ns() + int(timeout * NS_PER_S)
    while now_ns() < deadline:
        if req.status == status:
            return True
        time.sleep(0.001)
    return False


def prefill_result() -> PrefillResult:
    return PrefillResult(
        all_logits=torch.empty(1, 1, 1),
        past_key_values=DynamicCache(),
        num_prompt_tokens=1,
        start_ns=time.monotonic_ns(),
    )


def decode_result(done: bool = True, token: str = "x") -> DecodeResult:
    return DecodeResult(
        token_id=1,
        token=token,
        finish_reason=FinishReason.MAX_LENGTH if done else None,
        all_logits=None if done else torch.empty(1, 1, 1),
        past_key_values=None if done else DynamicCache(),
    )


def _wait_for_worker_to_die(
    worker: SimpleWorker | BatchWorker, timeout: float = 2.0
) -> None:
    assert worker._thread is not None
    worker._thread.join(timeout=timeout)
    assert not worker._thread.is_alive(), "Worker thread did not exit within timeout"


def _assert_error_event(req: GenerationRequestState) -> None:
    events = drain_events(req)
    assert len(events) == 1, f"{req.request_id}: expected 1 ErrorEvent, got {events}"
    assert isinstance(events[0], ErrorEvent)
    assert req.status == RequestStatus.FAILED
