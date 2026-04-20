"""
Tests for server/executor/worker.py — BatchWorker

Structure:
  Group 1:  Constructor validation
  Group 2:  Lifecycle (start / stop)
  Group 3:  submit() semantics
  Group 4:  Happy-path request lifecycle
  Group 5:  Graceful (non-exception) failure
  Group 6:  batched_prefill exception handling
  Group 7:  Shutdown during prefill
  Group 8:  Shutdown during decode
  Group 9:  Fatal exception in batched_decode
  Group 10: Batch sizing
  Group 11: Edge cases and pure method tests
"""

import queue
import threading
import time
from typing import Any

import pytest
import torch
from transformers import DynamicCache

from server.executor.types import (
    BaseBatchExecutor,
    BatchExecutorConfig,
    DecodeResult,
    ErrorEvent,
    FinishReason,
    GenerationRequestState,
    PrefillResult,
    RequestFailure,
    RequestStatus,
)
from server.executor.worker import BatchWorker

from .worker_helpers import (
    _assert_error_event,
    _make_fail_decode,
    _make_fail_prefill,
    _wait_for_worker_to_die,
    drain_events,
    make_req,
    wait_for_status,
)

# ─── Test infrastructure ──────────────────────────────────────────────────────


def _prefill_result() -> PrefillResult:
    return PrefillResult(
        all_logits=torch.empty(1, 1, 1),
        past_key_values=DynamicCache(),
        num_prompt_tokens=1,
        start_ns=time.monotonic_ns(),
    )


def _decode_result(done: bool) -> DecodeResult:
    return DecodeResult(
        token_id=0,
        token="",
        finish_reason=FinishReason.MAX_LENGTH if done else None,
        all_logits=None if done else torch.empty(1, 1, 1),
        past_key_values=None if done else DynamicCache(),
    )


def _result_from_status(
    req: GenerationRequestState,
) -> PrefillResult | DecodeResult | RequestFailure | None:
    if req.status == RequestStatus.FAILED:
        return RequestFailure(error=req.error or "request failed")
    if req.status == RequestStatus.DECODING:
        return _prefill_result()
    if req.status == RequestStatus.DONE:
        return _decode_result(done=True)
    return None


class FakeBatchExecutor(BaseBatchExecutor):
    """
    Hand-crafted fake batch executor with per-request behavioral control.

    prefill_side_effects: dict[str, callable | BaseException | None]
        None (default) -> sets req.status = DECODING for all requests in batch
        BaseException instance -> raised from batched_prefill (triggers _handle_fatal_error)
        callable(req) -> called per-request instead of default behaviour

    decode_steps: dict[str, int]
        Number of decode calls before setting status=DONE (default: 1).

    decode_side_effects: dict[str, callable | BaseException | None]
        Same structure as prefill_side_effects; applied per request_id.
        BaseException is raised from batched_decode (triggers _handle_fatal_error)

    prefill_hook / decode_hook: threading.Event | None
        Set exactly once on the first call (useful for synchronisation).

    decode_gate: threading.Event | None
        batched_decode() blocks on gate.wait() before executing.

    prefill_call_sizes: list[int]
        Records the len(batch) for each batched_prefill call.

    decode_call_sizes: list[int]
        Records the len(batch) for each batched_decode call.
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
        self.prefill_call_sizes: list[int] = []
        self.decode_call_sizes: list[int] = []

    def batched_prefill(
        self, request_states: list[GenerationRequestState]
    ) -> list[PrefillResult | RequestFailure | None]:
        self.prefill_call_sizes.append(len(request_states))

        if self._prefill_hook is not None:
            self._prefill_hook.set()
            self._prefill_hook = None  # fire once

        results: list[PrefillResult | RequestFailure | None] = []
        for req in request_states:
            fx = self._prefill_fx.get(req.request_id)
            if isinstance(fx, BaseException):
                raise fx
            elif callable(fx):
                result = fx(req)
                if isinstance(result, PrefillResult | RequestFailure):
                    results.append(result)
                    continue
                result = _result_from_status(req)
                results.append(
                    _prefill_result() if isinstance(result, DecodeResult) else result
                )
            else:
                results.append(_prefill_result())
        return results

    def batched_decode(
        self, request_states: list[GenerationRequestState]
    ) -> list[DecodeResult | RequestFailure | None]:
        self.decode_call_sizes.append(len(request_states))

        if self._decode_hook is not None:
            self._decode_hook.set()
            self._decode_hook = None  # fire once

        if self._decode_gate is not None:
            self._decode_gate.wait(timeout=5.0)

        results: list[DecodeResult | RequestFailure | None] = []
        for req in request_states:
            fx = self._decode_fx.get(req.request_id)
            if isinstance(fx, BaseException):
                raise fx
            elif callable(fx):
                result = fx(req)
                if isinstance(result, DecodeResult | RequestFailure):
                    results.append(result)
                    continue
                result = _result_from_status(req)
                results.append(None if isinstance(result, PrefillResult) else result)
            else:
                n = self._decode_counts.get(req.request_id, 0) + 1
                self._decode_counts[req.request_id] = n
                steps = self._decode_steps.get(req.request_id, 1)
                results.append(_decode_result(done=n >= steps))
        return results


def make_batch_worker(
    executor: FakeBatchExecutor | None = None,
    max_queue_size: int = 16,
    max_active_requests: int = 4,
    max_prefill_batch_size: int = 4,
    max_decode_batch_size: int = 4,
) -> BatchWorker:
    if executor is None:
        executor = FakeBatchExecutor()
    return BatchWorker(
        executor,
        BatchExecutorConfig(
            max_queue_size=max_queue_size,
            max_active_requests=max_active_requests,
            max_prefill_batch_size=max_prefill_batch_size,
            max_decode_batch_size=max_decode_batch_size,
        ),
    )


# ─── Group 1: Constructor validation ─────────────────────────────────────────


def test_max_queue_size_zero_raises() -> None:
    with pytest.raises(ValueError, match="max_queue_size"):
        BatchWorker(
            FakeBatchExecutor(),
            BatchExecutorConfig(
                max_queue_size=0,
                max_active_requests=1,
                max_prefill_batch_size=1,
                max_decode_batch_size=1,
            ),
        )


def test_max_active_requests_zero_raises() -> None:
    with pytest.raises(ValueError, match="max_active_requests"):
        BatchWorker(
            FakeBatchExecutor(),
            BatchExecutorConfig(
                max_queue_size=1,
                max_active_requests=0,
                max_prefill_batch_size=1,
                max_decode_batch_size=1,
            ),
        )


def test_max_prefill_batch_size_zero_raises() -> None:
    with pytest.raises(ValueError, match="max_prefill_batch_size"):
        BatchWorker(
            FakeBatchExecutor(),
            BatchExecutorConfig(
                max_queue_size=1,
                max_active_requests=1,
                max_prefill_batch_size=0,
                max_decode_batch_size=1,
            ),
        )


def test_max_decode_batch_size_zero_raises() -> None:
    with pytest.raises(ValueError, match="max_decode_batch_size"):
        BatchWorker(
            FakeBatchExecutor(),
            BatchExecutorConfig(
                max_queue_size=1,
                max_active_requests=1,
                max_prefill_batch_size=1,
                max_decode_batch_size=0,
            ),
        )


def test_prefill_batch_exceeds_active_raises() -> None:
    with pytest.raises(ValueError, match="max_prefill_batch_size cannot be greater"):
        BatchWorker(
            FakeBatchExecutor(),
            BatchExecutorConfig(
                max_queue_size=8,
                max_active_requests=2,
                max_prefill_batch_size=4,
                max_decode_batch_size=2,
            ),
        )


def test_decode_batch_exceeds_active_raises() -> None:
    with pytest.raises(ValueError, match="max_decode_batch_size cannot be greater"):
        BatchWorker(
            FakeBatchExecutor(),
            BatchExecutorConfig(
                max_queue_size=8,
                max_active_requests=2,
                max_prefill_batch_size=2,
                max_decode_batch_size=4,
            ),
        )


@pytest.mark.parametrize(
    "qs,mar,pbs,dbs",
    [
        (-1, 1, 1, 1),
        (1, -1, 1, 1),
        (1, 1, -1, 1),
        (1, 1, 1, -1),
        (0, 0, 0, 0),
        (-99, -99, -99, -99),
    ],
)
def test_invalid_config_parametrized(qs: int, mar: int, pbs: int, dbs: int) -> None:
    with pytest.raises(ValueError):
        BatchWorker(
            FakeBatchExecutor(),
            BatchExecutorConfig(
                max_queue_size=qs,
                max_active_requests=mar,
                max_prefill_batch_size=pbs,
                max_decode_batch_size=dbs,
            ),
        )


def test_minimum_valid_config_constructs() -> None:
    worker = BatchWorker(
        FakeBatchExecutor(),
        BatchExecutorConfig(
            max_queue_size=1,
            max_active_requests=1,
            max_prefill_batch_size=1,
            max_decode_batch_size=1,
        ),
    )
    assert worker._thread is None
    assert worker._waiting == []
    assert worker._active == []
    assert not worker._shutdown_event.is_set()


def test_inbound_queue_maxsize_wired_from_config() -> None:
    worker = make_batch_worker(max_queue_size=7)
    assert worker._inbound.maxsize == 7


# ─── Group 2: Lifecycle ───────────────────────────────────────────────────────


def test_start_creates_named_daemon_thread() -> None:
    worker = make_batch_worker()
    try:
        worker.start()
        assert worker._thread is not None
        assert worker._thread.is_alive()
        assert worker._thread.daemon is True
        assert worker._thread.name == "inference-worker"
    finally:
        worker.stop()


def test_stop_joins_thread() -> None:
    worker = make_batch_worker()
    worker.start()
    worker.stop()
    assert worker._thread is not None
    assert not worker._thread.is_alive()


def test_double_start_is_noop() -> None:
    worker = make_batch_worker()
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
    worker = make_batch_worker()
    req = make_req("r0")
    worker.submit(req)
    worker.stop()
    assert req.status == RequestStatus.FAILED
    events = drain_events(req)
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert events[0].request_id == "r0"


# ─── Group 3: submit() semantics ─────────────────────────────────────────────


def test_submit_after_stop_raises_runtime_error() -> None:
    worker = make_batch_worker()
    worker.start()
    worker.stop()
    with pytest.raises(RuntimeError, match="shutting down"):
        worker.submit(make_req())


def test_submit_without_start_enqueues() -> None:
    worker = make_batch_worker()
    worker.submit(make_req())
    assert worker._inbound.qsize() == 1


def test_submit_full_queue_raises_queue_full() -> None:
    worker = make_batch_worker(max_queue_size=2)
    worker.submit(make_req())
    worker.submit(make_req())
    with pytest.raises(queue.Full):
        worker.submit(make_req())


def test_submit_full_queue_raises_with_running_worker() -> None:
    """Inbound queue full while worker is busy; third submit raises queue.Full."""
    decode_gate = threading.Event()
    decode_hook = threading.Event()
    executor = FakeBatchExecutor(decode_hook=decode_hook, decode_gate=decode_gate)
    worker = make_batch_worker(
        executor,
        max_queue_size=1,
        max_active_requests=1,
        max_prefill_batch_size=1,
        max_decode_batch_size=1,
    )
    r0, r1, r2 = make_req("r0"), make_req("r1"), make_req("r2")
    try:
        worker.start()
        worker.submit(r0)
        assert decode_hook.wait(timeout=2.0)
        worker.submit(r1)  # fills the single inbound queue slot
        with pytest.raises(queue.Full):
            worker.submit(r2)
    finally:
        decode_gate.set()
        worker.stop()


# ─── Group 4: Happy-path request lifecycle ────────────────────────────────────


def test_single_request_completes() -> None:
    worker = make_batch_worker()
    req = make_req("r0")
    try:
        worker.start()
        worker.submit(req)
        assert wait_for_status(req, RequestStatus.DONE), "request did not reach DONE"
    finally:
        worker.stop()


def test_multistep_decode_completes() -> None:
    executor = FakeBatchExecutor(decode_steps={"r0": 5})
    worker = make_batch_worker(executor)
    req = make_req("r0")
    try:
        worker.start()
        worker.submit(req)
        assert wait_for_status(req, RequestStatus.DONE), "request did not reach DONE"
        assert executor._decode_counts["r0"] == 5
    finally:
        worker.stop()


def test_three_concurrent_requests_all_complete() -> None:
    worker = make_batch_worker(
        max_active_requests=3, max_prefill_batch_size=3, max_decode_batch_size=3
    )
    reqs = [make_req(f"r{i}") for i in range(3)]
    try:
        worker.start()
        for req in reqs:
            worker.submit(req)
        for req in reqs:
            assert wait_for_status(req, RequestStatus.DONE), (
                f"{req.request_id} did not complete"
            )
    finally:
        worker.stop()


def test_more_requests_than_max_active_all_complete() -> None:
    """5 requests with max_active=2 must all complete (tests queue drain + promotion)."""
    worker = make_batch_worker(
        max_queue_size=8,
        max_active_requests=2,
        max_prefill_batch_size=2,
        max_decode_batch_size=2,
    )
    reqs = [make_req(f"r{i}") for i in range(5)]
    try:
        worker.start()
        for req in reqs:
            worker.submit(req)
        for req in reqs:
            assert wait_for_status(req, RequestStatus.DONE), (
                f"{req.request_id} did not complete"
            )
    finally:
        worker.stop()


def test_prefill_batch_sizes_recorded() -> None:
    """With max_prefill_batch_size=2, 3 requests produce batches of [2, 1]."""
    executor = FakeBatchExecutor()
    worker = make_batch_worker(
        executor,
        max_active_requests=4,
        max_prefill_batch_size=2,
        max_decode_batch_size=4,
    )
    reqs = [make_req(f"r{i}") for i in range(3)]
    try:
        worker.start()
        for req in reqs:
            worker.submit(req)
        for req in reqs:
            assert wait_for_status(req, RequestStatus.DONE), (
                f"{req.request_id} did not complete"
            )
    finally:
        worker.stop()

    assert executor.prefill_call_sizes == [2, 1]


# ─── Group 5: Graceful (non-exception) failure ───────────────────────────────


def test_graceful_prefill_failure_emits_error_event() -> None:
    executor = FakeBatchExecutor(prefill_side_effects={"r0": _make_fail_prefill()})
    worker = make_batch_worker(executor)
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
    executor = FakeBatchExecutor(prefill_side_effects={"r1": _make_fail_prefill()})
    worker = make_batch_worker(
        executor,
        max_active_requests=2,
        max_prefill_batch_size=2,
        max_decode_batch_size=2,
    )
    r0, r1 = make_req("r0"), make_req("r1")
    try:
        worker.start()
        worker.submit(r0)
        worker.submit(r1)
        assert wait_for_status(r0, RequestStatus.DONE)
        assert wait_for_status(r1, RequestStatus.FAILED)
    finally:
        worker.stop()


def test_graceful_prefill_failure_not_added_to_active() -> None:
    """A request that fails during prefill (status != DECODING) must not appear in _active."""
    executor = FakeBatchExecutor(
        prefill_side_effects={"r1": _make_fail_prefill()},
        decode_steps={"r0": 2},
    )
    r0_gate = threading.Event()
    r0_second_started = threading.Event()
    r0_count = [0]

    def controlled_r0_decode(req: GenerationRequestState) -> None:
        r0_count[0] += 1
        if r0_count[0] == 1:
            return  # stay DECODING
        r0_second_started.set()
        r0_gate.wait(timeout=5.0)
        req.status = RequestStatus.DONE

    executor._decode_fx["r0"] = controlled_r0_decode
    worker = make_batch_worker(
        executor,
        max_active_requests=3,
        max_prefill_batch_size=2,
        max_decode_batch_size=3,
    )
    r0, r1, r2 = make_req("r0"), make_req("r1"), make_req("r2")
    for req in [r0, r1, r2]:
        worker.submit(req)
    try:
        worker.start()
        assert r0_second_started.wait(timeout=2.0), "r0 second decode never started"
        active_ids = {req.request_id for req in worker._active}
        assert "r1" not in active_ids
    finally:
        r0_gate.set()
        worker.stop()


def test_graceful_decode_failure_removes_request_from_active() -> None:
    executor = FakeBatchExecutor(decode_side_effects={"r0": _make_fail_decode()})
    worker = make_batch_worker(executor)
    req = make_req("r0")
    try:
        worker.start()
        worker.submit(req)
        assert wait_for_status(req, RequestStatus.FAILED)
        events = drain_events(req)
        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)
        time.sleep(0.05)
        assert req not in worker._active
    finally:
        worker.stop()


# ─── Group 6: batched_prefill exception handling ─────────────────────────────


def test_prefill_exception_single_request_gets_error_event() -> None:
    executor = FakeBatchExecutor(prefill_side_effects={"r0": RuntimeError("boom")})
    worker = make_batch_worker(executor, max_active_requests=4)
    r0 = make_req("r0")
    worker.submit(r0)
    worker.start()
    _wait_for_worker_to_die(worker)
    _assert_error_event(r0)
    worker.stop()


def test_prefill_exception_entire_batch_gets_error_event() -> None:
    """3 requests in one batch, prefill raises. All 3 get ErrorEvents."""
    executor = FakeBatchExecutor(prefill_side_effects={"r0": RuntimeError("boom")})
    worker = make_batch_worker(executor, max_active_requests=4)
    r0, r1, r2 = make_req("r0"), make_req("r1"), make_req("r2")
    for req in [r0, r1, r2]:
        worker.submit(req)
    worker.start()
    _wait_for_worker_to_die(worker)
    for req in [r0, r1, r2]:
        _assert_error_event(req)
    worker.stop()


def test_prefill_exception_inbound_overflow_also_cancelled() -> None:
    """max_active=2, submit 4. Prefill raises on first batch. Inbound drained too."""
    executor = FakeBatchExecutor(prefill_side_effects={"r0": RuntimeError("boom")})
    worker = make_batch_worker(
        executor,
        max_queue_size=8,
        max_active_requests=2,
        max_prefill_batch_size=2,
        max_decode_batch_size=2,
    )
    r0, r1, r2, r3 = (make_req(f"r{i}") for i in range(4))
    for req in [r0, r1, r2, r3]:
        worker.submit(req)
    worker.start()
    _wait_for_worker_to_die(worker)
    for req in [r0, r1, r2, r3]:
        _assert_error_event(req)
    worker.stop()


def test_prefill_exception_waiting_requests_also_cancelled() -> None:
    """Requests remaining in _waiting (not yet in the prefill batch) get ErrorEvents."""
    executor = FakeBatchExecutor(prefill_side_effects={"r0": RuntimeError("boom")})
    worker = make_batch_worker(
        executor,
        max_active_requests=4,
        max_prefill_batch_size=2,
        max_decode_batch_size=4,
    )
    reqs = [make_req(f"r{i}") for i in range(4)]
    for req in reqs:
        worker.submit(req)
    worker.start()
    _wait_for_worker_to_die(worker)
    for req in reqs:
        _assert_error_event(req)
    worker.stop()


def test_prefill_exception_no_request_silently_lost() -> None:
    """Sanity: every submitted request gets an ErrorEvent."""
    executor = FakeBatchExecutor(prefill_side_effects={"r1": RuntimeError("boom")})
    worker = make_batch_worker(executor, max_active_requests=4)
    reqs = [make_req(f"r{i}") for i in range(4)]
    for req in reqs:
        worker.submit(req)
    worker.start()
    _wait_for_worker_to_die(worker)
    for req in reqs:
        _assert_error_event(req)
    worker.stop()


# ─── Group 7: Shutdown during prefill ─────────────────────────────────────────


def test_shutdown_after_prefill_batch_cancels_remaining() -> None:
    """
    After the first prefill batch completes, _shutdown_event is set.
    The remaining _waiting requests and newly _active requests are cancelled
    by the graceful shutdown block at the end of _run_loop.
    """
    prefill_done = threading.Event()
    test_can_proceed = threading.Event()

    def blocking_prefill(req: GenerationRequestState) -> None:
        req.status = RequestStatus.DECODING
        prefill_done.set()
        test_can_proceed.wait(timeout=5.0)

    executor = FakeBatchExecutor(prefill_side_effects={"r0": blocking_prefill})
    worker = make_batch_worker(
        executor,
        max_active_requests=4,
        max_prefill_batch_size=1,
        max_decode_batch_size=4,
    )
    r0, r1 = make_req("r0"), make_req("r1")
    worker.submit(r0)
    worker.submit(r1)
    worker.start()

    assert prefill_done.wait(timeout=2.0)
    # r0 is now in _active, r1 is in _waiting
    worker._shutdown_event.set()
    test_can_proceed.set()
    assert worker._thread is not None
    worker._thread.join(timeout=2.0)
    worker.stop()

    for req in [r0, r1]:
        events = drain_events(req)
        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)
        assert req.status == RequestStatus.FAILED


def test_shutdown_set_during_prefill_cancels_batch() -> None:
    """r0's prefill sets _shutdown_event; after batch_prefill returns the loop exits."""

    def prefill_with_shutdown(req: GenerationRequestState) -> None:
        req.status = RequestStatus.DECODING
        worker._shutdown_event.set()

    executor = FakeBatchExecutor(prefill_side_effects={"r0": prefill_with_shutdown})
    worker = make_batch_worker(
        executor,
        max_active_requests=4,
        max_prefill_batch_size=1,
        max_decode_batch_size=4,
    )
    r0, r1 = make_req("r0"), make_req("r1")
    worker.submit(r0)
    worker.submit(r1)
    worker.start()
    assert worker._thread is not None
    worker._thread.join(timeout=2.0)
    worker.stop()

    for req in [r0, r1]:
        events = drain_events(req)
        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)
        assert req.status == RequestStatus.FAILED


def test_shutdown_during_idle_sleep_pending_request_drained() -> None:
    """
    A request submitted just before stop() is called should be cancelled by
    stop()'s inbound drain, not silently discarded.
    """
    worker = make_batch_worker()
    worker.start()
    time.sleep(0.03)

    req = make_req("idle-req")
    worker.submit(req)
    worker.stop()

    assert req.status in (RequestStatus.DONE, RequestStatus.FAILED)
    if req.status == RequestStatus.FAILED:
        events = drain_events(req)
        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)


# ─── Group 8: Shutdown during decode ──────────────────────────────────────────


def test_decode_completes_before_shutdown_check_request_is_done() -> None:
    """
    Shutdown event is set while batched_decode is in-flight (decode gate blocks).
    Once decode finishes, the cleanup pass removes DONE requests from _active;
    the outer while-loop then sees the shutdown event and exits cleanly.
    The request should be DONE, not FAILED.
    """
    decode_entered = threading.Event()
    decode_gate = threading.Event()

    def blocking_decode_then_done(req: GenerationRequestState) -> None:
        decode_entered.set()
        decode_gate.wait(timeout=5.0)
        req.status = RequestStatus.DONE

    executor = FakeBatchExecutor(decode_side_effects={"r0": blocking_decode_then_done})
    worker = make_batch_worker(
        executor,
        max_active_requests=1,
        max_prefill_batch_size=1,
        max_decode_batch_size=1,
    )
    r0 = make_req("r0")
    try:
        worker.start()
        worker.submit(r0)
        decode_entered.wait(timeout=2.0)
        worker._shutdown_event.set()
        decode_gate.set()
        assert worker._thread is not None
        worker._thread.join(timeout=2.0)
    finally:
        worker.stop()

    assert r0.status == RequestStatus.DONE
    assert not any(isinstance(event, ErrorEvent) for event in drain_events(r0))


def test_shutdown_between_decode_phases_cancels_all() -> None:
    """
    Both requests stay DECODING after batched_decode returns (via side effects
    that set _shutdown_event but don't change status). The graceful shutdown
    block cancels all _waiting + _active.
    """

    def decode_and_set_shutdown(req: GenerationRequestState) -> None:
        worker._shutdown_event.set()
        # Leave status as DECODING — do NOT finish the request.

    executor = FakeBatchExecutor(
        decode_side_effects={
            "r0": decode_and_set_shutdown,
            "r1": decode_and_set_shutdown,
        }
    )
    worker = make_batch_worker(
        executor,
        max_active_requests=2,
        max_prefill_batch_size=2,
        max_decode_batch_size=2,
    )
    r0, r1 = make_req("r0"), make_req("r1")
    worker.submit(r0)
    worker.submit(r1)
    worker.start()
    assert worker._thread is not None
    worker._thread.join(timeout=2.0)
    worker.stop()

    for req in [r0, r1]:
        events = drain_events(req)
        assert len(events) == 1, f"{req.request_id} should have one ErrorEvent"
        assert isinstance(events[0], ErrorEvent)
        assert req.status == RequestStatus.FAILED


def test_stop_cancels_active_request_blocked_in_decode() -> None:
    """
    stop() called while batched_decode is in-flight. After join, stop() drains
    _waiting + _active and emits ErrorEvents.
    """
    decode_entered = threading.Event()
    decode_gate = threading.Event()

    def blocking_decode_stays_decoding(req: GenerationRequestState) -> None:
        decode_entered.set()
        decode_gate.wait(timeout=5.0)
        # Leave status DECODING — request should be cancelled by stop().

    executor = FakeBatchExecutor(
        decode_side_effects={"r0": blocking_decode_stays_decoding}
    )
    worker = make_batch_worker(
        executor,
        max_active_requests=1,
        max_prefill_batch_size=1,
        max_decode_batch_size=1,
    )
    r0 = make_req("r0")
    worker.start()
    worker.submit(r0)
    decode_entered.wait(timeout=2.0)

    stop_done = threading.Event()

    def do_stop() -> None:
        worker.stop()
        stop_done.set()

    threading.Thread(target=do_stop, daemon=True).start()

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


# ─── Group 9: Fatal exception in batched_decode ───────────────────────────────


def test_decode_exception_crashes_worker_and_cancels_active() -> None:
    executor = FakeBatchExecutor(
        decode_side_effects={"r0": RuntimeError("decode boom")}
    )
    worker = make_batch_worker(executor)
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
    call_count = [0]

    def raise_on_second(req: GenerationRequestState) -> None:
        call_count[0] += 1
        if call_count[0] == 1:
            pass  # first call: stay DECODING
        else:
            raise RuntimeError("second step boom")

    executor = FakeBatchExecutor(decode_side_effects={"r0": raise_on_second})
    worker = make_batch_worker(executor)
    r0 = make_req("r0")
    worker.submit(r0)
    worker.start()
    _wait_for_worker_to_die(worker)

    events = drain_events(r0)
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    worker.stop()


def test_stop_after_decode_crash_is_safe() -> None:
    executor = FakeBatchExecutor(decode_side_effects={"r0": RuntimeError("boom")})
    worker = make_batch_worker(executor)
    worker.submit(make_req("r0"))
    worker.start()
    _wait_for_worker_to_die(worker)
    worker.stop()  # must not raise
    assert worker._thread is not None
    assert not worker._thread.is_alive()


def test_decode_exception_waiting_requests_also_cancelled() -> None:
    """Requests in _waiting (not yet prefilled) are cancelled when decode raises."""
    prefill_count = [0]

    def controlled_prefill(req: GenerationRequestState) -> None:
        prefill_count[0] += 1
        req.status = RequestStatus.DECODING

    executor = FakeBatchExecutor(
        prefill_side_effects={
            "r0": controlled_prefill,
            "r1": controlled_prefill,
        },
        decode_side_effects={"r0": RuntimeError("decode boom")},
    )
    worker = make_batch_worker(
        executor,
        max_active_requests=4,
        max_prefill_batch_size=1,
        max_decode_batch_size=4,
    )
    reqs = [make_req(f"r{i}") for i in range(4)]
    for req in reqs:
        worker.submit(req)
    worker.start()
    _wait_for_worker_to_die(worker)

    for req in reqs:
        _assert_error_event(req)
    worker.stop()


# ─── Group 10: Batch sizing ───────────────────────────────────────────────────


def test_prefill_batch_size_capped_by_config() -> None:
    """max_prefill_batch_size=2, 5 requests -> prefill batches of [2, 2, 1]."""
    executor = FakeBatchExecutor()
    worker = make_batch_worker(
        executor,
        max_queue_size=16,
        max_active_requests=8,
        max_prefill_batch_size=2,
        max_decode_batch_size=8,
    )
    reqs = [make_req(f"r{i}") for i in range(5)]
    try:
        worker.start()
        for req in reqs:
            worker.submit(req)
        for req in reqs:
            assert wait_for_status(req, RequestStatus.DONE), (
                f"{req.request_id} did not complete"
            )
    finally:
        worker.stop()

    assert executor.prefill_call_sizes == [2, 2, 1]


def test_decode_batch_size_capped_by_config() -> None:
    """max_decode_batch_size=2, 5 requests -> decode batches of at most 2."""
    executor = FakeBatchExecutor()
    worker = make_batch_worker(
        executor,
        max_queue_size=16,
        max_active_requests=8,
        max_prefill_batch_size=8,
        max_decode_batch_size=2,
    )
    reqs = [make_req(f"r{i}") for i in range(5)]
    try:
        worker.start()
        for req in reqs:
            worker.submit(req)
        for req in reqs:
            assert wait_for_status(req, RequestStatus.DONE), (
                f"{req.request_id} did not complete"
            )
    finally:
        worker.stop()

    assert all(size <= 2 for size in executor.decode_call_sizes), (
        f"Decode batches exceeded max_decode_batch_size=2: {executor.decode_call_sizes}"
    )


def test_prefill_batch_capped_by_remaining_active_slots() -> None:
    """
    max_active=3, max_prefill_batch_size=3. Submit 6 requests.
    First batch of 3 fills all active slots. The next prefill batch
    can only proceed after some active requests complete.
    """
    executor = FakeBatchExecutor()
    worker = make_batch_worker(
        executor,
        max_queue_size=16,
        max_active_requests=3,
        max_prefill_batch_size=3,
        max_decode_batch_size=3,
    )
    reqs = [make_req(f"r{i}") for i in range(6)]
    try:
        worker.start()
        for req in reqs:
            worker.submit(req)
        for req in reqs:
            assert wait_for_status(req, RequestStatus.DONE), (
                f"{req.request_id} did not complete"
            )
    finally:
        worker.stop()

    # First prefill should be 3 (filling all active slots).
    # Subsequent prefills happen as slots free up.
    assert executor.prefill_call_sizes[0] == 3
    assert sum(executor.prefill_call_sizes) == 6


def test_drain_inbound_respects_max_active() -> None:
    """max_active=3, submit 10. _drain_inbound moves at most 3 to _waiting."""
    decode_gate = threading.Event()
    decode_hook = threading.Event()
    executor = FakeBatchExecutor(decode_hook=decode_hook, decode_gate=decode_gate)
    worker = make_batch_worker(
        executor,
        max_queue_size=16,
        max_active_requests=3,
        max_prefill_batch_size=3,
        max_decode_batch_size=3,
    )
    reqs = [make_req(f"r{i}") for i in range(10)]
    try:
        worker.start()
        for req in reqs:
            worker.submit(req)

        assert decode_hook.wait(timeout=2.0)
        total_in_flight = len(worker._waiting) + len(worker._active)
        assert total_in_flight <= 3
    finally:
        decode_gate.set()
        worker.stop()


def test_all_batch_sizes_equal_max_active_no_splitting() -> None:
    """All batch sizes == max_active, submit 4 -> single prefill and decode call."""
    executor = FakeBatchExecutor()
    worker = make_batch_worker(
        executor,
        max_active_requests=4,
        max_prefill_batch_size=4,
        max_decode_batch_size=4,
    )
    reqs = [make_req(f"r{i}") for i in range(4)]
    try:
        worker.start()
        for req in reqs:
            worker.submit(req)
        for req in reqs:
            assert wait_for_status(req, RequestStatus.DONE)
    finally:
        worker.stop()

    assert executor.prefill_call_sizes == [4]
    assert executor.decode_call_sizes == [4]


def test_batch_size_one_processes_sequentially() -> None:
    """max_prefill_batch_size=1, max_decode_batch_size=1 -> individual calls."""
    executor = FakeBatchExecutor()
    worker = make_batch_worker(
        executor,
        max_active_requests=3,
        max_prefill_batch_size=1,
        max_decode_batch_size=1,
    )
    reqs = [make_req(f"r{i}") for i in range(3)]
    try:
        worker.start()
        for req in reqs:
            worker.submit(req)
        for req in reqs:
            assert wait_for_status(req, RequestStatus.DONE)
    finally:
        worker.stop()

    assert executor.prefill_call_sizes == [1, 1, 1]


def test_decode_batch_slicing_with_mixed_active() -> None:
    """
    max_decode_batch_size=2, max_active=4. 4 requests active but decode batch
    only gets 2 per call.
    """
    executor = FakeBatchExecutor(decode_steps={f"r{i}": 2 for i in range(4)})
    worker = make_batch_worker(
        executor,
        max_queue_size=16,
        max_active_requests=4,
        max_prefill_batch_size=4,
        max_decode_batch_size=2,
    )
    reqs = [make_req(f"r{i}") for i in range(4)]
    try:
        worker.start()
        for req in reqs:
            worker.submit(req)
        for req in reqs:
            assert wait_for_status(req, RequestStatus.DONE), (
                f"{req.request_id} did not complete"
            )
    finally:
        worker.stop()

    assert all(size <= 2 for size in executor.decode_call_sizes), (
        f"Decode batches exceeded max_decode_batch_size=2: {executor.decode_call_sizes}"
    )


# ─── Group 11: Edge cases and pure method tests ───────────────────────────────


def test_idle_worker_with_empty_queue_does_not_crash() -> None:
    worker = make_batch_worker()
    try:
        worker.start()
        time.sleep(0.05)
        assert worker._thread is not None
        assert worker._thread.is_alive()
    finally:
        worker.stop()
    assert worker._thread is not None
    assert not worker._thread.is_alive()


def test_worker_restart_after_stop() -> None:
    """stop() then start() must work; the second cycle should clear the shutdown event."""
    executor = FakeBatchExecutor()
    worker = make_batch_worker(executor)

    r0 = make_req("r0")
    worker.start()
    worker.submit(r0)
    assert wait_for_status(r0, RequestStatus.DONE)
    worker.stop()
    thread_id_first = id(worker._thread)

    r1 = make_req("r1")
    worker.start()
    assert id(worker._thread) != thread_id_first
    worker.submit(r1)
    assert wait_for_status(r1, RequestStatus.DONE)
    worker.stop()


def test_concurrent_submits_from_multiple_threads() -> None:
    """10 threads each submitting 5 requests must not corrupt worker state."""
    worker = make_batch_worker(
        max_queue_size=64,
        max_active_requests=8,
        max_prefill_batch_size=8,
        max_decode_batch_size=8,
    )
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
            assert wait_for_status(req, RequestStatus.DONE), (
                f"{req.request_id} did not finish"
            )
    finally:
        worker.stop()

    assert len(submitted) + full_errors[0] == 50


def test_max_active_one_serialises_requests() -> None:
    """With max_active_requests=1, r1's prefill only starts after r0 is DONE."""
    prefill_order: list[str] = []
    done_at_prefill: dict[str, RequestStatus] = {}

    def recording_prefill(req: GenerationRequestState) -> None:
        prefill_order.append(req.request_id)
        if req.request_id == "r1":
            done_at_prefill["r0_status"] = r0.status
        req.status = RequestStatus.DECODING

    executor = FakeBatchExecutor(
        prefill_side_effects={"r0": recording_prefill, "r1": recording_prefill}
    )
    worker = make_batch_worker(
        executor,
        max_active_requests=1,
        max_prefill_batch_size=1,
        max_decode_batch_size=1,
    )
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


# ─── Pure method tests (no thread needed) ─────────────────────────────────────


def test_select_prefill_batch_empty_waiting_returns_empty() -> None:
    """_select_prefill_batch with empty _waiting returns empty list."""
    worker = make_batch_worker()
    assert worker._select_prefill_batch() == []


def test_select_decode_batch_no_decoding_returns_empty() -> None:
    """_select_decode_batch with no DECODING requests returns empty list."""
    worker = make_batch_worker()
    req = make_req("r0")
    req.status = RequestStatus.DONE
    worker._active.append(req)
    assert worker._select_decode_batch() == []


def test_drain_inbound_empty_queue_no_op() -> None:
    """_drain_inbound with empty inbound queue is a no-op."""
    worker = make_batch_worker()
    worker._drain_inbound()
    assert worker._waiting == []
    assert worker._active == []


def test_select_prefill_batch_respects_max_prefill_size() -> None:
    """_select_prefill_batch caps batch at max_prefill_batch_size."""
    worker = make_batch_worker(
        max_active_requests=10,
        max_prefill_batch_size=2,
        max_decode_batch_size=10,
    )
    for i in range(5):
        worker._waiting.append(make_req(f"r{i}"))

    batch = worker._select_prefill_batch()
    assert len(batch) == 2
    assert batch[0].request_id == "r0"
    assert batch[1].request_id == "r1"
    assert len(worker._waiting) == 3


def test_select_decode_batch_respects_max_decode_size() -> None:
    """_select_decode_batch caps batch at max_decode_batch_size."""
    worker = make_batch_worker(
        max_active_requests=10,
        max_prefill_batch_size=10,
        max_decode_batch_size=2,
    )
    for i in range(5):
        req = make_req(f"r{i}")
        req.status = RequestStatus.DECODING
        worker._active.append(req)

    batch = worker._select_decode_batch()
    assert len(batch) == 2
    assert batch[0].request_id == "r0"
    assert batch[1].request_id == "r1"
