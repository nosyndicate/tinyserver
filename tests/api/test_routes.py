from queue import Queue
from typing import Callable, cast

import pytest
from fastapi import HTTPException, Request

from server.api import routes
from server.api.routes import (
    _await_generation,
    _stream_generation,
    _submit_or_fail,
)
from server.api.schema import GenerateRequest
from server.executor.engine import EngineCallbacks, EngineControl
from server.executor.types import GenerationRequestState, TokenEvent
from server.executor.worker import Worker
from server.model.sampling import SamplingParams
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


def make_state(request_id: str = "req-1") -> GenerationRequestState:
    return GenerationRequestState(
        request_id=request_id,
        sampling_params=SamplingParams(
            max_new_tokens=8, temperature=0.0, top_p=1.0
        ),
        prompt="hello",
    )


def test_await_generation_cancels_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    # When the client's wait times out (nothing ever lands on output_queue), the
    # handler must tell the worker to cancel before raising the 504, so the
    # engine stops decoding and frees the request's KV blocks.
    monkeypatch.setattr(routes, "_GENERATION_TIMEOUT_S", 0.01)
    worker = make_worker()
    state = make_state()

    with pytest.raises(HTTPException) as excinfo:
        _await_generation(state, worker)

    assert excinfo.value.status_code == 504
    assert state.cancelled.is_set()


def test_stream_generation_cancels_on_client_disconnect() -> None:
    # Closing the generator (what Starlette does on client disconnect) must run
    # the finally and cancel the request.
    worker = make_worker()
    state = make_state()
    # Prime one non-terminal token so the first next() enters the try and yields.
    state.output_queue.put(
        TokenEvent(token="hi", is_first=True, is_last=False, index=0)
    )

    gen = _stream_generation(state, worker)
    first_chunk = next(gen)
    assert "hi" in first_chunk
    assert not state.cancelled.is_set()  # still streaming

    gen.close()  # simulate GeneratorExit from a disconnected client

    assert state.cancelled.is_set()
