from queue import Queue
from typing import Callable, cast

import pytest
from fastapi import HTTPException, Request

from server.api.routes import _submit_or_fail
from server.api.schema import GenerateRequest
from server.executor.engine import EngineCallbacks, EngineControl
from server.executor.types import GenerationRequestState
from server.executor.worker import Worker
from tests.executor.worker_helpers import make_req


class _NoopEngine:
    """Minimal ``InferenceEngine`` stub.

    The engine is never run in these tests; only ``cancel_inflight`` is touched,
    by ``Worker.stop()``'s drain path.
    """

    def run(
        self,
        inbound: Queue[GenerationRequestState],
        control: EngineControl,
        callbacks: EngineCallbacks,
    ) -> None:
        raise AssertionError("engine.run must not be called by these tests")

    def cancel_inflight(
        self,
        message: str,
        cancel_request: Callable[[GenerationRequestState, str], None],
    ) -> None:
        return None


def make_worker(max_queue_size: int = 16) -> Worker:
    """Lightweight local factory; the engine is never started here."""
    return Worker(_NoopEngine(), max_queue_size=max_queue_size)


class _FakeState:
    def __init__(self, worker: Worker, device: str) -> None:
        self.worker = worker
        self.device = device


class _FakeApp:
    def __init__(self, worker: Worker, device: str) -> None:
        self.state = _FakeState(worker, device)


class _FakeRequest:
    """Stand-in for ``fastapi.Request`` exposing only ``app.state``."""

    def __init__(self, worker: Worker, device: str = "cpu") -> None:
        self.app = _FakeApp(worker, device)


def make_request(worker: Worker, device: str = "cpu") -> Request:
    """Build a minimal stand-in for ``fastapi.Request`` for unit testing."""
    return cast(Request, _FakeRequest(worker, device))


def make_generate_request(prompt: str = "hello") -> GenerateRequest:
    return GenerateRequest(
        prompt=prompt,
        max_new_tokens=1,
        temperature=1.0,
        top_p=0.95,
        seed=None,
    )


def test_submit_or_fail_maps_worker_shutting_down_to_503() -> None:
    # stop() sets the shutdown event without ever starting the worker thread,
    # so the next submit() raises WorkerShuttingDown for real.
    worker = make_worker()
    worker.stop()
    request = make_request(worker)

    with pytest.raises(HTTPException) as excinfo:
        _submit_or_fail(request, make_generate_request())

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "Worker is shutting down. Please try again later."


def test_submit_or_fail_maps_queue_full_to_503() -> None:
    worker = make_worker(max_queue_size=1)
    worker.submit(make_req("filler"))  # single-slot queue is now full
    request = make_request(worker)

    with pytest.raises(HTTPException) as excinfo:
        _submit_or_fail(request, make_generate_request())

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "Server at capacity. Please try again later."


def test_submit_or_fail_happy_path_returns_state() -> None:
    worker = make_worker()
    request = make_request(worker)

    state = _submit_or_fail(request, make_generate_request(prompt="hello"))

    assert isinstance(state, GenerationRequestState)
    assert state.prompt == "hello"
    assert worker._inbound.qsize() == 1
