"""Shared test helpers for SimpleWorker and BatchWorker tests."""

import itertools
import queue
import time
from typing import Any, Callable

from server.executor.types import (
    BatchExecutorConfig,
    ErrorEvent,
    ExecutorConfig,
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
    """Returns a prefill side-effect that sets FAILED and emits an ErrorEvent."""

    def fail_prefill(req: GenerationRequestState) -> None:
        req.status = RequestStatus.FAILED
        req.error = "model error during prefill"
        req.output_queue.put(ErrorEvent(request_id=req.request_id, error=req.error))

    return fail_prefill


def _make_fail_decode() -> Callable[[GenerationRequestState], None]:
    """Returns a decode side-effect that sets FAILED and emits an ErrorEvent."""

    def fail_decode(req: GenerationRequestState) -> None:
        req.status = RequestStatus.FAILED
        req.error = "model error during decode"
        req.output_queue.put(ErrorEvent(request_id=req.request_id, error=req.error))

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


# ─── Test workers that inject per-request behavior at the prefill/decode seam ──


class _SentinelRunner:
    """Placeholder passed to the real worker constructor; tests override the
    prefill/decode methods, so the runner is never actually called."""


class SimpleWorkerInjector:
    """
    Hand-crafted scheduling injector for SimpleWorker tests.

    prefill_side_effects: dict[str, callable | BaseException | None]
        None (default) → sets req.status = DECODING
        BaseException instance → raised (escapes to _handle_fatal_error)
        callable(req) → called instead of default behaviour

    decode_steps: dict[str, int]
        Number of decode calls before setting status=DONE (default: 1).

    decode_side_effects: dict[str, callable | BaseException | None]
        Same structure as prefill_side_effects; applied per request_id.

    prefill_hook / decode_hook: threading.Event | None
        Set exactly once on the first call (useful for synchronisation).

    decode_gate: threading.Event | None
        decode() blocks on gate.wait() before executing.
    """

    def __init__(
        self,
        prefill_side_effects: dict[str, Any] | None = None,
        decode_steps: dict[str, int] | None = None,
        decode_side_effects: dict[str, Any] | None = None,
        prefill_hook: Any = None,
        decode_hook: Any = None,
        decode_gate: Any = None,
    ) -> None:
        self._prefill_fx = prefill_side_effects or {}
        self._decode_steps = decode_steps or {}
        self._decode_fx = decode_side_effects or {}
        self._prefill_hook = prefill_hook
        self._decode_hook = decode_hook
        self._decode_gate = decode_gate
        self._decode_counts: dict[str, int] = {}

    def prefill(self, request_state: GenerationRequestState) -> None:
        if self._prefill_hook is not None:
            self._prefill_hook.set()
            self._prefill_hook = None

        fx = self._prefill_fx.get(request_state.request_id)
        if isinstance(fx, BaseException):
            raise fx
        elif callable(fx):
            fx(request_state)
        else:
            request_state.status = RequestStatus.DECODING

    def decode(self, request_state: GenerationRequestState) -> None:
        if self._decode_hook is not None:
            self._decode_hook.set()
            self._decode_hook = None

        if self._decode_gate is not None:
            self._decode_gate.wait(timeout=5.0)

        fx = self._decode_fx.get(request_state.request_id)
        if isinstance(fx, BaseException):
            raise fx
        elif callable(fx):
            fx(request_state)
        else:
            n = self._decode_counts.get(request_state.request_id, 0) + 1
            self._decode_counts[request_state.request_id] = n
            steps = self._decode_steps.get(request_state.request_id, 1)
            if n >= steps:
                request_state.status = RequestStatus.DONE


class TestableSimpleWorker(SimpleWorker):
    """SimpleWorker whose prefill/decode steps are delegated to an injector.

    This keeps the scheduling-loop tests focused on scheduling (not on the real
    model calls) by letting tests control per-request prefill/decode behavior.
    """

    __test__ = False

    def __init__(self, injector: SimpleWorkerInjector, config: ExecutorConfig) -> None:
        super().__init__(runner=_SentinelRunner(), config=config)  # type: ignore[arg-type]
        self._injector = injector

    def _prefill_request(self, request_state: GenerationRequestState) -> None:
        self._injector.prefill(request_state)

    def _decode_request_step(self, request_state: GenerationRequestState) -> None:
        self._injector.decode(request_state)


class BatchWorkerInjector:
    """
    Hand-crafted scheduling injector for BatchWorker tests.

    prefill_side_effects / decode_side_effects follow the same structure as
    SimpleWorkerInjector. BaseException raised from a per-request side effect
    escapes the batched call (like a BatchExecutor itself malfunctioning) and
    triggers _handle_fatal_error in the worker.
    """

    def __init__(
        self,
        prefill_side_effects: dict[str, Any] | None = None,
        decode_steps: dict[str, int] | None = None,
        decode_side_effects: dict[str, Any] | None = None,
        prefill_hook: Any = None,
        decode_hook: Any = None,
        decode_gate: Any = None,
    ) -> None:
        self._prefill_fx = prefill_side_effects or {}
        self._decode_steps = decode_steps or {}
        self._decode_fx = decode_side_effects or {}
        self._prefill_hook = prefill_hook
        self._decode_hook = decode_hook
        self._decode_gate = decode_gate
        self._decode_counts: dict[str, int] = {}
        self.prefill_call_sizes: list[int] = []
        self.decode_call_sizes: list[int] = []

    def batched_prefill(self, request_states: list[GenerationRequestState]) -> None:
        self.prefill_call_sizes.append(len(request_states))

        if self._prefill_hook is not None:
            self._prefill_hook.set()
            self._prefill_hook = None

        for req in request_states:
            fx = self._prefill_fx.get(req.request_id)
            if isinstance(fx, BaseException):
                raise fx
            elif callable(fx):
                fx(req)
            else:
                req.status = RequestStatus.DECODING

    def batched_decode(self, request_states: list[GenerationRequestState]) -> None:
        self.decode_call_sizes.append(len(request_states))

        if self._decode_hook is not None:
            self._decode_hook.set()
            self._decode_hook = None

        if self._decode_gate is not None:
            self._decode_gate.wait(timeout=5.0)

        for req in request_states:
            fx = self._decode_fx.get(req.request_id)
            if isinstance(fx, BaseException):
                raise fx
            elif callable(fx):
                fx(req)
            else:
                n = self._decode_counts.get(req.request_id, 0) + 1
                self._decode_counts[req.request_id] = n
                steps = self._decode_steps.get(req.request_id, 1)
                if n >= steps:
                    req.status = RequestStatus.DONE


class TestableBatchWorker(BatchWorker):
    """BatchWorker whose batched prefill/decode are delegated to an injector."""

    __test__ = False

    def __init__(
        self, injector: BatchWorkerInjector, config: BatchExecutorConfig
    ) -> None:
        super().__init__(runner=_SentinelRunner(), config=config)  # type: ignore[arg-type]
        self._injector = injector

    def _prefill_batch(self, request_states: list[GenerationRequestState]) -> None:
        self._injector.batched_prefill(request_states)

    def _decode_batch_step(self, request_states: list[GenerationRequestState]) -> None:
        self._injector.batched_decode(request_states)
