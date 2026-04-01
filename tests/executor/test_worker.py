"""
Tests for server/executor/worker.py

Structure:
  Group 1:  Constructor validation
  Group 2:  Lifecycle (start / stop)
  Group 3:  submit() semantics
  Group 4:  Happy-path request lifecycle
  Group 5:  Graceful (non-exception) failure
  Group 6:  prefill exception cancels the entire new_requests batch
  Group 7:  Shutdown detected mid-prefill loop
  Group 8:  Shutdown detected mid-decode loop
  Group 9:  Fatal exception in decode_one_step
  Group 10: Capacity and batching
  Group 11: Edge cases
"""

import itertools
import queue
import threading
import time
from typing import Any, Callable

import pytest

from server.executor.types import (
    BaseExecutor,
    ErrorEvent,
    ExecutorConfig,
    GenerationRequestState,
    RequestStatus,
)
from server.executor.worker import Worker
from server.model.sampling import SamplingParams

# ─── Test infrastructure ──────────────────────────────────────────────────────


class FakeExecutor(BaseExecutor):
    """
    Hand-crafted fake executor with per-request behavioral control.

    prefill_side_effects: dict[str, callable | BaseException | None]
        None (default) → sets req.status = DECODING
        BaseException instance → raised (triggers _handle_fatal_error)
        callable(req) → called instead of default behaviour

    decode_steps: dict[str, int]
        Number of decode calls before setting status=DONE (default: 1).

    decode_side_effects: dict[str, callable | BaseException | None]
        Same structure as prefill_side_effects; applied per request_id.

    prefill_hook / decode_hook: threading.Event | None
        Set exactly once on the first call (useful for synchronisation).

    decode_gate: threading.Event | None
        decode_one_step() blocks on gate.wait() before executing.
    """

    def __init__(
        self,
        prefill_side_effects: dict[str, Any] | None = None,
        decode_steps: dict[str, int] | None = None,
        decode_side_effects: dict[str, Any] | None = None,
        prefill_hook: threading.Event | None = None,
        decode_hook: threading.Event | None = None,
        decode_gate: threading.Event | None = None,
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
            self._prefill_hook = None  # fire once

        fx = self._prefill_fx.get(request_state.request_id)
        if isinstance(fx, BaseException):
            raise fx
        elif callable(fx):
            fx(request_state)
        else:
            request_state.status = RequestStatus.DECODING

    def decode_one_step(self, request_state: GenerationRequestState) -> None:
        if self._decode_hook is not None:
            self._decode_hook.set()
            self._decode_hook = None  # fire once

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


def make_worker(
    executor: FakeExecutor | None = None,
    max_queue_size: int = 16,
    max_active_requests: int = 4,
) -> Worker:
    if executor is None:
        executor = FakeExecutor()
    return Worker(
        executor,
        ExecutorConfig(
            max_queue_size=max_queue_size, max_active_requests=max_active_requests
        ),
    )


# ─── Group 1: Constructor validation ─────────────────────────────────────────


def test_max_queue_size_zero_raises() -> None:
    with pytest.raises(ValueError, match="max_queue_size"):
        Worker(FakeExecutor(), ExecutorConfig(max_queue_size=0, max_active_requests=1))


def test_max_active_requests_zero_raises() -> None:
    with pytest.raises(ValueError, match="max_active_requests"):
        Worker(FakeExecutor(), ExecutorConfig(max_queue_size=1, max_active_requests=0))


@pytest.mark.parametrize(
    "qs,mar",
    [(-1, 1), (1, -1), (0, 0), (-99, -99)],
)
def test_invalid_config_parametrized(qs: int, mar: int) -> None:
    with pytest.raises(ValueError):
        Worker(
            FakeExecutor(), ExecutorConfig(max_queue_size=qs, max_active_requests=mar)
        )


def test_minimum_valid_config_constructs() -> None:
    worker = Worker(
        FakeExecutor(), ExecutorConfig(max_queue_size=1, max_active_requests=1)
    )
    assert worker._thread is None
    assert worker._active == []
    assert not worker._shutdown_event.is_set()


def test_inbound_queue_maxsize_wired_from_config() -> None:
    worker = make_worker(max_queue_size=7)
    assert worker._inbound.maxsize == 7


# ─── Group 2: Lifecycle ───────────────────────────────────────────────────────


def test_start_creates_named_daemon_thread() -> None:
    worker = make_worker()
    try:
        worker.start()
        assert worker._thread is not None
        assert worker._thread.is_alive()
        assert worker._thread.daemon is True
        assert worker._thread.name == "inference-worker"
    finally:
        worker.stop()


def test_stop_joins_thread() -> None:
    worker = make_worker()
    worker.start()
    worker.stop()
    assert worker._thread is not None
    assert not worker._thread.is_alive()


def test_double_start_is_noop() -> None:
    worker = make_worker()
    try:
        worker.start()
        tid_after_first = id(worker._thread)
        worker.start()  # should log a warning and return
        assert id(worker._thread) == tid_after_first
        assert worker._thread is not None
        assert worker._thread.is_alive()
    finally:
        worker.stop()


def test_stop_before_start_drains_inbound() -> None:
    """stop() without start() should drain the inbound queue and emit ErrorEvents."""
    worker = make_worker()
    req = make_req("r0")
    worker.submit(req)  # goes into inbound; worker never started
    worker.stop()
    assert req.status == RequestStatus.FAILED
    events = drain_events(req)
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert events[0].request_id == "r0"


# ─── Group 3: submit() semantics ─────────────────────────────────────────────


def test_submit_after_stop_raises_runtime_error() -> None:
    worker = make_worker()
    worker.start()
    worker.stop()
    with pytest.raises(RuntimeError, match="shutting down"):
        worker.submit(make_req())


def test_submit_without_start_enqueues() -> None:
    worker = make_worker()
    worker.submit(make_req())
    assert worker._inbound.qsize() == 1


def test_submit_full_queue_raises_queue_full() -> None:
    worker = make_worker(max_queue_size=2)
    worker.submit(make_req())
    worker.submit(make_req())
    with pytest.raises(queue.Full):
        worker.submit(make_req())


def test_submit_full_queue_raises_with_running_worker() -> None:
    """Inbound queue full while worker is busy; third submit raises queue.Full."""
    decode_gate = threading.Event()
    decode_hook = threading.Event()
    executor = FakeExecutor(decode_hook=decode_hook, decode_gate=decode_gate)
    worker = make_worker(executor, max_queue_size=1, max_active_requests=1)
    r0, r1, r2 = make_req("r0"), make_req("r1"), make_req("r2")
    try:
        worker.start()
        worker.submit(r0)
        assert decode_hook.wait(
            timeout=2.0
        )  # r0 is now being decoded (active slot taken)
        worker.submit(r1)  # fills the single inbound queue slot
        with pytest.raises(queue.Full):
            worker.submit(r2)
    finally:
        decode_gate.set()
        worker.stop()


# ─── Group 4: Happy-path lifecycle ───────────────────────────────────────────


def test_single_request_completes() -> None:
    worker = make_worker()
    req = make_req("r0")
    try:
        worker.start()
        worker.submit(req)
        assert wait_for_status(req, RequestStatus.DONE), "request did not reach DONE"
    finally:
        worker.stop()


def test_multistep_decode_completes() -> None:
    executor = FakeExecutor(decode_steps={"r0": 5})
    worker = make_worker(executor)
    req = make_req("r0")
    try:
        worker.start()
        worker.submit(req)
        assert wait_for_status(req, RequestStatus.DONE), "request did not reach DONE"
        assert executor._decode_counts["r0"] == 5
    finally:
        worker.stop()


def test_three_concurrent_requests_all_complete() -> None:
    worker = make_worker(max_active_requests=3)
    reqs = [make_req(f"r{i}") for i in range(3)]
    try:
        worker.start()
        for req in reqs:
            worker.submit(req)
        for req in reqs:
            assert wait_for_status(
                req, RequestStatus.DONE
            ), f"{req.request_id} did not complete"
    finally:
        worker.stop()


def test_more_requests_than_max_active_all_complete() -> None:
    """5 requests with max_active=2 must all complete (tests queue drain + promotion)."""
    worker = make_worker(max_queue_size=8, max_active_requests=2)
    reqs = [make_req(f"r{i}") for i in range(5)]
    try:
        worker.start()
        for req in reqs:
            worker.submit(req)
        for req in reqs:
            assert wait_for_status(
                req, RequestStatus.DONE
            ), f"{req.request_id} did not complete"
    finally:
        worker.stop()


# ─── Group 5: Graceful (non-exception) failure ───────────────────────────────


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


def test_graceful_prefill_failure_emits_error_event() -> None:
    executor = FakeExecutor(prefill_side_effects={"r0": _make_fail_prefill()})
    worker = make_worker(executor)
    req = make_req("r0")
    try:
        worker.start()
        worker.submit(req)
        assert wait_for_status(req, RequestStatus.FAILED)
        events = drain_events(req)
        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)
        assert events[0].request_id == "r0"
    finally:
        worker.stop()


def test_graceful_prefill_failure_does_not_block_other_requests() -> None:
    executor = FakeExecutor(prefill_side_effects={"r1": _make_fail_prefill()})
    worker = make_worker(executor, max_active_requests=2)
    r0, r1 = make_req("r0"), make_req("r1")
    try:
        worker.start()
        worker.submit(r0)
        worker.submit(r1)
        assert wait_for_status(r0, RequestStatus.DONE)
        assert wait_for_status(r1, RequestStatus.FAILED)
    finally:
        worker.stop()


def test_graceful_decode_failure_removes_request_from_active() -> None:
    executor = FakeExecutor(decode_side_effects={"r0": _make_fail_decode()})
    worker = make_worker(executor)
    req = make_req("r0")
    try:
        worker.start()
        worker.submit(req)
        assert wait_for_status(req, RequestStatus.FAILED)
        events = drain_events(req)
        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)
        # After the cleanup pass, the failed request must be gone from _active.
        # Give the worker one loop iteration to clean up.
        time.sleep(0.05)
        assert req not in worker._active
    finally:
        worker.stop()


# ─── Group 6: prefill exception cancels the entire new_requests batch ────────


def _wait_for_worker_to_die(worker: Worker, timeout: float = 2.0) -> None:
    assert worker._thread is not None
    worker._thread.join(timeout=timeout)
    assert not worker._thread.is_alive(), "Worker thread did not exit within timeout"


def _assert_error_event(req: GenerationRequestState) -> None:
    events = drain_events(req)
    assert len(events) == 1, f"{req.request_id}: expected 1 ErrorEvent, got {events}"
    assert isinstance(events[0], ErrorEvent)
    assert req.status == RequestStatus.FAILED


def test_prefill_exception_single_request_gets_error_event() -> None:
    """The failing request itself (i=0, active is empty) gets an ErrorEvent."""
    executor = FakeExecutor(prefill_side_effects={"r0": RuntimeError("boom")})
    worker = make_worker(executor, max_active_requests=4)
    r0 = make_req("r0")
    worker.submit(r0)
    worker.start()
    _wait_for_worker_to_die(worker)
    _assert_error_event(r0)
    worker.stop()


def test_prefill_exception_all_requests_in_batch_get_error_event() -> None:
    """r0 raises at i=0; r1 and r2 (not yet reached) are also cancelled."""
    executor = FakeExecutor(prefill_side_effects={"r0": RuntimeError("boom")})
    worker = make_worker(executor, max_active_requests=4)
    r0, r1, r2 = make_req("r0"), make_req("r1"), make_req("r2")
    for req in [r0, r1, r2]:
        worker.submit(req)
    worker.start()
    _wait_for_worker_to_die(worker)
    for req in [r0, r1, r2]:
        _assert_error_event(req)
    worker.stop()


def test_prefill_exception_mid_batch_all_requests_cancelled() -> None:
    """r1 raises at i=1; r0 (in active), r1, r2, r3 all get ErrorEvents."""
    executor = FakeExecutor(prefill_side_effects={"r1": RuntimeError("boom")})
    worker = make_worker(executor, max_active_requests=4)
    r0, r1, r2, r3 = (make_req(f"r{i}") for i in range(4))
    for req in [r0, r1, r2, r3]:
        worker.submit(req)
    worker.start()
    _wait_for_worker_to_die(worker)
    for req in [r0, r1, r2, r3]:
        _assert_error_event(req)
    worker.stop()


def test_prefill_exception_inbound_overflow_also_cancelled() -> None:
    """r0/r1 in new_requests and r2/r3 still in inbound all get ErrorEvents."""
    # max_active_requests=2 → batch is [r0, r1]; r2/r3 stay in inbound.
    executor = FakeExecutor(prefill_side_effects={"r0": RuntimeError("boom")})
    worker = make_worker(executor, max_queue_size=8, max_active_requests=2)
    r0, r1, r2, r3 = (make_req(f"r{i}") for i in range(4))
    for req in [r0, r1, r2, r3]:
        worker.submit(req)
    worker.start()
    _wait_for_worker_to_die(worker)
    for req in [r0, r1, r2, r3]:
        _assert_error_event(req)
    worker.stop()


def test_prefill_exception_no_request_silently_lost() -> None:
    """Sanity: none of 4 requests is silently dropped; every one gets an ErrorEvent."""
    executor = FakeExecutor(prefill_side_effects={"r1": RuntimeError("boom")})
    worker = make_worker(executor, max_active_requests=4)
    reqs = [make_req(f"r{i}") for i in range(4)]
    for req in reqs:
        worker.submit(req)
    worker.start()
    _wait_for_worker_to_die(worker)
    for req in reqs:
        _assert_error_event(req)
    worker.stop()


# ─── Group 7: Shutdown detected mid-prefill loop ─────────────────────────────


def test_shutdown_between_prefill_calls_cancels_remaining() -> None:
    """
    Worker detects shutdown between two prefill calls and cancels:
    - new_requests[i:] (not yet prefilled)
    - self._active (already prefilled)
    """
    first_prefill_done = threading.Event()
    test_can_proceed = threading.Event()

    def blocking_prefill(req: GenerationRequestState) -> None:
        req.status = RequestStatus.DECODING
        first_prefill_done.set()
        test_can_proceed.wait(timeout=5.0)

    executor = FakeExecutor(prefill_side_effects={"r0": blocking_prefill})
    worker = make_worker(executor, max_active_requests=4)
    r0, r1 = make_req("r0"), make_req("r1")

    # Submit both before start so they land in the same new_requests batch.
    worker.submit(r0)
    worker.submit(r1)
    worker.start()

    assert first_prefill_done.wait(timeout=2.0)
    # Worker has completed r0's prefill (r0 is now in active) and is blocked.
    worker._shutdown_event.set()
    test_can_proceed.set()
    assert worker._thread is not None
    worker._thread.join(timeout=2.0)

    # r0 was in self._active → cancelled by the prefill-loop shutdown handler
    r0_events = drain_events(r0)
    assert len(r0_events) == 1
    assert isinstance(r0_events[0], ErrorEvent)
    assert r0.status == RequestStatus.FAILED

    # r1 was in new_requests[1:] → cancelled by the same handler
    r1_events = drain_events(r1)
    assert len(r1_events) == 1
    assert isinstance(r1_events[0], ErrorEvent)
    assert r1.status == RequestStatus.FAILED

    worker.stop()


def test_shutdown_set_at_first_prefill_cancels_both() -> None:
    """
    r0's prefill sets _shutdown_event; the check fires at i=1 and cancels
    r1 (new_requests[1:]) and r0 (now in active).
    """

    def prefill_with_shutdown(req: GenerationRequestState) -> None:
        req.status = RequestStatus.DECODING
        worker._shutdown_event.set()

    executor = FakeExecutor(prefill_side_effects={"r0": prefill_with_shutdown})
    worker = make_worker(executor, max_active_requests=4)
    r0, r1 = make_req("r0"), make_req("r1")
    worker.submit(r0)
    worker.submit(r1)
    worker.start()
    assert worker._thread is not None
    worker._thread.join(timeout=2.0)

    for req in [r0, r1]:
        events = drain_events(req)
        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)
        assert req.status == RequestStatus.FAILED

    worker.stop()


def test_shutdown_during_idle_sleep_pending_request_drained() -> None:
    """
    A request submitted just before stop() is called should be cancelled by
    stop()'s inbound drain, not silently discarded.
    """
    worker = make_worker()
    worker.start()
    # Allow worker to enter the idle sleep path (no active requests).
    time.sleep(0.03)

    req = make_req("idle-req")
    worker.submit(req)
    worker.stop()

    # The request was either processed normally (DONE) or drained by stop() (FAILED).
    # Either outcome is acceptable — what must NOT happen is status staying QUEUED
    # with an empty output_queue (silent discard).
    assert req.status in (RequestStatus.DONE, RequestStatus.FAILED)
    if req.status == RequestStatus.FAILED:
        events = drain_events(req)
        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)


# ─── Group 8: Shutdown detected mid-decode loop ──────────────────────────────


def test_decode_completes_before_shutdown_check_request_is_done() -> None:
    """
    Shutdown event is set while the worker is inside decode_one_step().
    Once decode finishes (gate released), the cleanup pass removes the DONE
    request from active; the outer while-loop then sees the shutdown event
    and exits cleanly.  The request should be DONE, not FAILED.
    """
    decode_entered = threading.Event()
    decode_gate = threading.Event()

    def blocking_decode_then_done(req: GenerationRequestState) -> None:
        decode_entered.set()
        decode_gate.wait(timeout=5.0)
        req.status = RequestStatus.DONE

    executor = FakeExecutor(decode_side_effects={"r0": blocking_decode_then_done})
    worker = make_worker(executor, max_active_requests=1)
    r0 = make_req("r0")
    try:
        worker.start()
        worker.submit(r0)
        decode_entered.wait(timeout=2.0)  # worker is blocked inside decode
        worker._shutdown_event.set()
        decode_gate.set()
        assert worker._thread is not None
        worker._thread.join(timeout=2.0)
    finally:
        worker.stop()

    assert r0.status == RequestStatus.DONE
    assert drain_events(r0) == []


def test_shutdown_between_decode_calls_cancels_all_active() -> None:
    """
    r0's decode sets _shutdown_event; the loop checks shutdown before r1's
    decode and cancels both active requests.
    """

    def decode_and_set_shutdown(req: GenerationRequestState) -> None:
        worker._shutdown_event.set()
        # Leave status as DECODING — do NOT finish the request.

    executor = FakeExecutor(decode_side_effects={"r0": decode_and_set_shutdown})
    worker = make_worker(executor, max_active_requests=2)
    r0, r1 = make_req("r0"), make_req("r1")
    worker.submit(r0)
    worker.submit(r1)
    worker.start()
    assert worker._thread is not None
    worker._thread.join(timeout=2.0)

    for req in [r0, r1]:
        events = drain_events(req)
        assert len(events) == 1, f"{req.request_id} should have one ErrorEvent"
        assert isinstance(events[0], ErrorEvent)
        assert req.status == RequestStatus.FAILED

    worker.stop()


def test_stop_cancels_active_request_blocked_in_decode() -> None:
    """
    stop() is called while a request is blocked inside decode_one_step().
    After the gate is released the worker exits its loop; stop()'s post-join
    cancel pass picks up the still-DECODING request and emits an ErrorEvent.
    """
    decode_entered = threading.Event()
    decode_gate = threading.Event()

    def blocking_decode_stays_decoding(req: GenerationRequestState) -> None:
        decode_entered.set()
        decode_gate.wait(timeout=5.0)
        # Leave status DECODING — request should be cancelled by stop().

    executor = FakeExecutor(decode_side_effects={"r0": blocking_decode_stays_decoding})
    worker = make_worker(executor, max_active_requests=1)
    r0 = make_req("r0")
    worker.start()
    worker.submit(r0)
    decode_entered.wait(timeout=2.0)  # worker is blocked in decode

    # Call stop() in a background thread (it will block on _thread.join() until
    # we release the gate).
    stop_done = threading.Event()

    def do_stop() -> None:
        worker.stop()
        stop_done.set()

    threading.Thread(target=do_stop, daemon=True).start()

    # Wait until stop() has set _shutdown_event, then release the decode gate.
    deadline = time.monotonic() + 2.0
    while not worker._shutdown_event.is_set() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert worker._shutdown_event.is_set()
    decode_gate.set()

    stop_done.wait(timeout=5.0)

    assert r0.status == RequestStatus.FAILED
    events = drain_events(r0)
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)


# ─── Group 9: Fatal exception in decode_one_step ─────────────────────────────


def test_decode_exception_crashes_worker_and_cancels_active() -> None:
    """An unhandled exception in decode_one_step triggers _handle_fatal_error."""
    executor = FakeExecutor(decode_side_effects={"r0": RuntimeError("decode boom")})
    worker = make_worker(executor)
    r0 = make_req("r0")
    worker.submit(r0)
    worker.start()
    _wait_for_worker_to_die(worker)

    events = drain_events(r0)
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert r0.status == RequestStatus.FAILED
    worker.stop()


def test_decode_exception_on_second_step_crashes_worker() -> None:
    """Exception raised on the second decode step also triggers _handle_fatal_error."""
    call_count = [0]

    def raise_on_second(req: GenerationRequestState) -> None:
        call_count[0] += 1
        if call_count[0] == 1:
            pass  # first call: stay DECODING
        else:
            raise RuntimeError("second step boom")

    executor = FakeExecutor(decode_side_effects={"r0": raise_on_second})
    worker = make_worker(executor)
    r0 = make_req("r0")
    worker.submit(r0)
    worker.start()
    _wait_for_worker_to_die(worker)

    events = drain_events(r0)
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    worker.stop()


def test_stop_after_decode_crash_is_safe() -> None:
    """stop() must not raise even if the worker thread has already died."""
    executor = FakeExecutor(decode_side_effects={"r0": RuntimeError("boom")})
    worker = make_worker(executor)
    worker.submit(make_req("r0"))
    worker.start()
    _wait_for_worker_to_die(worker)
    worker.stop()  # must not raise
    assert worker._thread is not None
    assert not worker._thread.is_alive()


# ─── Group 10: Capacity and batching ─────────────────────────────────────────


def test_max_active_requests_never_exceeded() -> None:
    """At no point should len(_active) exceed max_active_requests."""
    decode_gate = threading.Event()
    decode_hook = threading.Event()
    executor = FakeExecutor(decode_hook=decode_hook, decode_gate=decode_gate)
    worker = make_worker(executor, max_queue_size=8, max_active_requests=2)
    try:
        worker.start()
        for i in range(5):
            worker.submit(make_req(f"r{i}"))
        assert decode_hook.wait(timeout=2.0)
        # The worker is now blocked inside decode_one_step; active list is frozen.
        assert len(worker._active) <= 2
    finally:
        decode_gate.set()
        worker.stop()


def test_requests_processed_in_fifo_order() -> None:
    """With max_active_requests=1 requests are prefilled in submission order."""
    prefill_order: list[str] = []

    def recording_prefill(req: GenerationRequestState) -> None:
        prefill_order.append(req.request_id)
        req.status = RequestStatus.DECODING

    executor = FakeExecutor(
        prefill_side_effects={f"r{i}": recording_prefill for i in range(3)}
    )
    worker = make_worker(executor, max_active_requests=1)
    reqs = [make_req(f"r{i}") for i in range(3)]
    # Pre-load to guarantee ordering.
    for req in reqs:
        worker.submit(req)
    try:
        worker.start()
        for req in reqs:
            assert wait_for_status(req, RequestStatus.DONE)
    finally:
        worker.stop()

    assert prefill_order == ["r0", "r1", "r2"]


def test_queue_drains_into_active_as_slots_free() -> None:
    """Requests waiting in the inbound queue are promoted as active slots open up."""
    worker = make_worker(max_queue_size=8, max_active_requests=1)
    reqs = [make_req(f"r{i}") for i in range(4)]
    try:
        worker.start()
        for req in reqs:
            worker.submit(req)
        for req in reqs:
            assert wait_for_status(
                req, RequestStatus.DONE
            ), f"{req.request_id} did not complete"
    finally:
        worker.stop()


def test_active_list_pruned_after_iteration() -> None:
    """DONE and FAILED requests are removed from _active; DECODING requests stay.

    The cleanup pass runs after the full decode for-loop, so we need r2 to
    complete a first decode call (staying DECODING) to let the loop finish and
    the cleanup run.  We then block on r2's second call to freeze the state
    while we inspect _active.
    """
    r2_gate = threading.Event()
    r2_second_decode_started = threading.Event()
    r2_decode_count = [0]

    def controlled_r2_decode(req: GenerationRequestState) -> None:
        r2_decode_count[0] += 1
        if r2_decode_count[0] == 1:
            return  # first call: stay DECODING so the loop can finish + cleanup runs
        # Second call: signal the test, then block — keeping r2 in _active.
        r2_second_decode_started.set()
        r2_gate.wait(timeout=5.0)
        req.status = RequestStatus.DONE

    executor = FakeExecutor(
        decode_side_effects={
            "r1": _make_fail_decode(),
            "r2": controlled_r2_decode,
        },
    )
    worker = make_worker(executor, max_active_requests=3)
    r0, r1, r2 = make_req("r0"), make_req("r1"), make_req("r2")
    for req in [r0, r1, r2]:
        worker.submit(req)
    try:
        worker.start()
        # After the first decode loop finishes and cleanup runs, the worker starts
        # the second loop with only r2 active.  r2's second decode call signals us
        # and then blocks, freezing _active so we can safely inspect it.
        assert r2_second_decode_started.wait(
            timeout=2.0
        ), "r2's second decode never started"
        active_ids = {req.request_id for req in worker._active}
        assert "r0" not in active_ids
        assert "r1" not in active_ids
        assert "r2" in active_ids
    finally:
        r2_gate.set()
        worker.stop()


# ─── Group 11: Edge cases ─────────────────────────────────────────────────────


def test_idle_worker_with_empty_queue_does_not_crash() -> None:
    worker = make_worker()
    try:
        worker.start()
        time.sleep(0.05)  # let the worker spin through several idle
        assert worker._thread is not None
        assert worker._thread.is_alive()
    finally:
        worker.stop()
    assert worker._thread is not None
    assert not worker._thread.is_alive()


def test_worker_restart_after_stop() -> None:
    """stop() then start() must work; the second cycle should clear the shutdown event."""
    executor = FakeExecutor()
    worker = make_worker(executor)

    r0 = make_req("r0")
    worker.start()
    worker.submit(r0)
    assert wait_for_status(r0, RequestStatus.DONE)
    worker.stop()
    thread_id_first = id(worker._thread)

    r1 = make_req("r1")
    worker.start()
    assert id(worker._thread) != thread_id_first  # new thread was created
    worker.submit(r1)
    assert wait_for_status(r1, RequestStatus.DONE)
    worker.stop()


def test_concurrent_submits_from_multiple_threads() -> None:
    """10 threads each submitting 5 requests must not corrupt worker state."""
    worker = make_worker(max_queue_size=64, max_active_requests=8)
    submitted: list[GenerationRequestState] = []
    lock = threading.Lock()
    full_errors = [0]

    def submit_batch(n: int) -> None:
        for _ in range(n):
            req = make_req()
            try:
                worker.submit(req)
                with lock:
                    submitted.append(req)
            except queue.Full:
                with lock:
                    full_errors[0] += 1

    try:
        worker.start()
        threads = [threading.Thread(target=submit_batch, args=(5,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for req in submitted:
            assert wait_for_status(
                req, RequestStatus.DONE
            ), f"{req.request_id} did not finish"
    finally:
        worker.stop()

    assert len(submitted) + full_errors[0] == 50


def test_max_active_one_serialises_requests() -> None:
    """With max_active_requests=1 r1's prefill only starts after r0 is DONE."""
    prefill_order: list[str] = []
    done_at_prefill: dict[str, RequestStatus] = {}

    def recording_prefill(req: GenerationRequestState) -> None:
        prefill_order.append(req.request_id)
        # Snapshot r0's status at the moment r1 is about to be prefilled.
        if req.request_id == "r1":
            done_at_prefill["r0_status"] = r0.status
        req.status = RequestStatus.DECODING

    executor = FakeExecutor(
        prefill_side_effects={
            "r0": recording_prefill,
            "r1": recording_prefill,
        }
    )
    worker = make_worker(executor, max_active_requests=1)
    r0, r1 = make_req("r0"), make_req("r1")
    worker.submit(r0)
    worker.submit(r1)
    try:
        worker.start()
        assert wait_for_status(r0, RequestStatus.DONE)
        assert wait_for_status(r1, RequestStatus.DONE)
    finally:
        worker.stop()

    assert prefill_order == ["r0", "r1"]
    assert done_at_prefill.get("r0_status") == RequestStatus.DONE
