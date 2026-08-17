from queue import Queue
from typing import Callable, cast

import pytest
from fastapi import HTTPException, Request
from pydantic import ValidationError

from server.api.collector import CollectorRegistry
from server.api.routes import (
    _await_generation,
    _build_request_state,
    _stream_generation,
    _submit_or_fail,
)
from server.api.schema import (
    GenerateRequest,
    StreamDoneEvent,
    StreamErrorEvent,
    StreamTokenEvent,
)
from server.executor.engine import EngineCallbacks, EngineControl
from server.executor.sinks import SharedQueueSink
from server.executor.types import (
    DoneEvent,
    ErrorEvent,
    FinishReason,
    GenerationRequestState,
    TokenEvent,
)
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
    def __init__(
        self,
        worker: Worker,
        device: str,
        registry: CollectorRegistry,
        sink: SharedQueueSink,
    ) -> None:
        self.worker = worker
        self.device = device
        self.registry = registry
        self.sink = sink


class _FakeApp:
    def __init__(
        self,
        worker: Worker,
        device: str,
        registry: CollectorRegistry,
        sink: SharedQueueSink,
    ) -> None:
        self.state = _FakeState(worker, device, registry, sink)


class _FakeRequest:
    """Stand-in for ``fastapi.Request`` exposing only ``app.state``."""

    def __init__(
        self,
        worker: Worker,
        device: str,
        registry: CollectorRegistry,
        sink: SharedQueueSink,
    ) -> None:
        self.app = _FakeApp(worker, device, registry, sink)


def make_request(
    worker: Worker,
    device: str = "cpu",
    registry: CollectorRegistry | None = None,
    sink: SharedQueueSink | None = None,
) -> Request:
    """Build a minimal stand-in for ``fastapi.Request`` for unit testing."""
    return cast(
        Request,
        _FakeRequest(
            worker,
            device,
            registry if registry is not None else CollectorRegistry(),
            sink if sink is not None else SharedQueueSink(),
        ),
    )


def make_generate_request(prompt: str = "hello") -> GenerateRequest:
    return GenerateRequest(
        prompt=prompt,
        max_new_tokens=1,
        temperature=1.0,
        top_p=0.95,
        seed=None,
    )


def test_top_k_defaults_to_disabled() -> None:
    assert GenerateRequest(prompt="hi").top_k == 0


def test_negative_top_k_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hi", top_k=-1)


def test_explicit_stream_event_schemas() -> None:
    token = StreamTokenEvent(token_str="", index=0)
    done = StreamDoneEvent(
        finish_reason="max_length",
        prompt_tokens=3,
        output_tokens=1,
        ttft_ms=1.0,
        total_ms=2.0,
        tokens_per_s=500.0,
        queue_wait_ms=0.0,
        execution_ms=2.0,
    )
    error = StreamErrorEvent(error="boom")

    assert token.model_dump() == {"type": "token", "token_str": "", "index": 0}
    assert done.type == "done"
    assert done.finish_reason == "max_length"
    assert error.model_dump() == {"type": "error", "error": "boom"}

    with pytest.raises(ValidationError):
        StreamTokenEvent(token_str="x", index=0, output_tokens=1)  # type: ignore[call-arg]


def test_top_k_reaches_sampling_params() -> None:
    req = GenerateRequest(prompt="hi", max_new_tokens=1, top_k=5)
    state = _build_request_state(req, device="cpu", sink=SharedQueueSink())
    assert state.sampling_params.top_k == 5


def test_build_request_state_uses_the_shared_sink() -> None:
    # Every request must emit onto the one server-wide queue the pump drains,
    # not onto a queue of its own that nobody is reading.
    sink = SharedQueueSink()
    state = _build_request_state(
        GenerateRequest(prompt="hi", max_new_tokens=1), device="cpu", sink=sink
    )

    event = TokenEvent(request_id=state.request_id, token="x", index=0)
    state.sink.emit(event)

    assert sink.queue.get_nowait() is event


def test_submit_or_fail_maps_worker_shutting_down_to_503() -> None:
    # stop() sets the shutdown event without ever starting the worker thread,
    # so the next submit() raises WorkerShuttingDown for real.
    worker = make_worker()
    worker.stop()
    registry = CollectorRegistry()
    request = make_request(worker, registry=registry)

    with pytest.raises(HTTPException) as excinfo:
        _submit_or_fail(request, make_generate_request())

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "Worker is shutting down. Please try again later."
    # A rejected request must not leave its collector behind.
    assert len(registry) == 0


def test_submit_or_fail_maps_queue_full_to_503() -> None:
    worker = make_worker(max_queue_size=1)
    worker.submit(make_req("filler"))  # single-slot queue is now full
    registry = CollectorRegistry()
    request = make_request(worker, registry=registry)

    with pytest.raises(HTTPException) as excinfo:
        _submit_or_fail(request, make_generate_request())

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "Server at capacity. Please try again later."
    assert len(registry) == 0


def test_submit_or_fail_happy_path_returns_state_and_collector() -> None:
    worker = make_worker()
    registry = CollectorRegistry()
    request = make_request(worker, registry=registry)

    state, collector = _submit_or_fail(request, make_generate_request(prompt="hello"))

    assert isinstance(state, GenerationRequestState)
    assert state.prompt == "hello"
    assert worker._inbound.qsize() == 1
    assert len(registry) == 1


def test_submit_or_fail_unregisters_on_unexpected_submit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = make_worker()
    registry = CollectorRegistry()
    request = make_request(worker, registry=registry)

    def fail_submit(_: GenerationRequestState) -> None:
        raise RuntimeError("submit failed unexpectedly")

    monkeypatch.setattr(worker, "submit", fail_submit)

    with pytest.raises(RuntimeError, match="submit failed unexpectedly"):
        _submit_or_fail(request, make_generate_request())

    assert len(registry) == 0


async def test_submit_or_fail_registers_before_submitting() -> None:
    # The engine may emit its first token the instant it accepts the request, so
    # the collector has to exist by the time submit() returns — otherwise the
    # event is dropped as an unknown request_id.
    worker = make_worker()
    registry = CollectorRegistry()
    request = make_request(worker, registry=registry)

    state, collector = _submit_or_fail(request, make_generate_request())

    event = TokenEvent(request_id=state.request_id, token="hi", index=0)
    registry.dispatch(event)

    assert await collector.get(timeout=0.01) is event


def make_state(request_id: str = "req-1") -> GenerationRequestState:
    return GenerationRequestState(
        request_id=request_id,
        sampling_params=SamplingParams(max_new_tokens=8, temperature=0.0, top_p=1.0),
        prompt="hello",
        sink=SharedQueueSink(),
    )


def make_done(request_id: str = "req-1") -> DoneEvent:
    return DoneEvent(
        request_id=request_id,
        text="hello world",
        num_prompt_tokens=3,
        num_output_tokens=2,
        finish_reason=FinishReason.MAX_LENGTH,
        ttft_ms=12.5,
        total_ms=50.0,
        queue_wait_ms=5.0,
        execution_ms=45.0,
    )


async def test_await_generation_cancels_on_timeout() -> None:
    # When the client's wait times out (nothing is ever dispatched), the handler
    # must tell the worker to cancel before raising the 504, so the engine stops
    # decoding and frees the request's KV blocks.
    worker = make_worker()
    state = make_state()
    registry = CollectorRegistry()
    collector = registry.register(state.request_id)

    with pytest.raises(HTTPException) as excinfo:
        await _await_generation(state, worker, collector, registry, timeout=0.01)

    assert excinfo.value.status_code == 504
    assert state.cancelled.is_set()
    assert len(registry) == 0


async def test_await_generation_returns_done_event_metrics() -> None:
    # scripts/bench reads every one of these fields out of the HTTP response, so
    # a dropped field silently corrupts summary.json.
    worker = make_worker()
    state = make_state()
    registry = CollectorRegistry()
    collector = registry.register(state.request_id)
    registry.dispatch(
        TokenEvent(
            request_id=state.request_id,
            token="hello",
            index=0,
        )
    )
    registry.dispatch(make_done(state.request_id))

    response = await _await_generation(state, worker, collector, registry, timeout=1.0)

    assert response.text == "hello world"
    assert response.prompt_tokens == 3
    assert response.output_tokens == 2
    assert response.ttft_ms == 12.5
    assert response.total_ms == 50.0
    assert response.queue_wait_ms == 5.0
    assert response.execution_ms == 45.0
    # Decode rate is measured over execution_ms (45 ms), not total_ms (50 ms),
    # so the 5 ms of queue wait does not drag the reported rate down.
    assert response.tokens_per_s == pytest.approx(2 / 0.045)
    # Terminal event consumed: the handler is gone, so is its mailbox.
    assert len(registry) == 0


async def test_await_generation_maps_error_event_to_500() -> None:
    worker = make_worker()
    state = make_state()
    registry = CollectorRegistry()
    collector = registry.register(state.request_id)
    registry.dispatch(ErrorEvent(request_id=state.request_id, error="boom"))

    with pytest.raises(HTTPException) as excinfo:
        await _await_generation(state, worker, collector, registry, timeout=1.0)

    assert excinfo.value.status_code == 500
    assert "boom" in excinfo.value.detail
    assert len(registry) == 0


async def test_stream_generation_cancels_on_client_disconnect() -> None:
    # Closing the generator (what Starlette does on client disconnect) must run
    # the finally, cancel the request, and release the collector.
    worker = make_worker()
    state = make_state()
    registry = CollectorRegistry()
    collector = registry.register(state.request_id)
    # Prime one non-terminal token so the first step enters the try and yields.
    registry.dispatch(
        TokenEvent(
            request_id=state.request_id,
            token="hi",
            index=0,
        )
    )

    gen = _stream_generation(state, worker, collector, registry, timeout=1.0)
    first_chunk = await gen.__anext__()
    assert "hi" in first_chunk
    assert not state.cancelled.is_set()  # still streaming
    assert len(registry) == 1

    await gen.aclose()  # simulate the disconnect Starlette throws in

    assert state.cancelled.is_set()
    assert len(registry) == 0


async def test_stream_generation_emits_token_then_done_metrics() -> None:
    worker = make_worker()
    state = make_state()
    registry = CollectorRegistry()
    collector = registry.register(state.request_id)
    registry.dispatch(
        TokenEvent(
            request_id=state.request_id,
            token=" world",
            index=1,
        )
    )
    registry.dispatch(make_done(state.request_id))

    chunks = [
        chunk
        async for chunk in _stream_generation(
            state, worker, collector, registry, timeout=1.0
        )
    ]

    assert len(chunks) == 2
    assert chunks[0] == 'data: {"type":"token","token_str":" world","index":1}\n\n'
    assert '"type":"done"' in chunks[1]
    assert '"finish_reason":"max_length"' in chunks[1]
    assert '"ttft_ms":12.5' in chunks[1]
    assert len(registry) == 0
