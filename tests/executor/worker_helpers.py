"""Shared test helpers for SimpleWorker and BatchWorker tests."""

import itertools
import queue
import time
from typing import Callable

from server.executor.types import (
    ErrorEvent,
    GenerationRequestState,
    RequestStatus,
)
from server.executor.worker import BatchWorker, SimpleWorker
from server.model.sampling import SamplingParams

_counter = itertools.count()


def make_req(request_id: str | None = None) -> GenerationRequestState:
    rid = request_id if request_id is not None else f"req-{next(_counter)}"
    return GenerationRequestState(
        request_id=rid,
        sampling_params=SamplingParams(max_new_tokens=10, temperature=1.0, top_p=1.0),
        prompt="hello",
    )


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
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if req.status == status:
            return True
        time.sleep(0.001)
    return False


def _make_fail_prefill() -> Callable[[GenerationRequestState], None]:
    """Returns a prefill side-effect that marks the request as failed."""

    def fail_prefill(req: GenerationRequestState) -> None:
        req.status = RequestStatus.FAILED
        req.error = "model error during prefill"

    return fail_prefill


def _make_fail_decode() -> Callable[[GenerationRequestState], None]:
    """Returns a decode side-effect that marks the request as failed."""

    def fail_decode(req: GenerationRequestState) -> None:
        req.status = RequestStatus.FAILED
        req.error = "model error during decode"

    return fail_decode


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
