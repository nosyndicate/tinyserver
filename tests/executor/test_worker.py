import queue
import threading
import time

import pytest

from server.executor.types import (
    BaseExecutor,
    DecodeResult,
    ErrorEvent,
    ExecutorConfig,
    GenerationRequestState,
    PrefillResult,
    RequestFailure,
    RequestStatus,
    TokenEvent,
)
from server.executor.worker import SimpleWorker

from .worker_helpers import (
    _assert_error_event,
    _wait_for_worker_to_die,
    decode_result,
    drain_events,
    make_req,
    prefill_result,
    wait_for_status,
)


class FakeExecutor(BaseExecutor):
    def __init__(
        self,
        prefill_results: (
            dict[str, PrefillResult | RequestFailure | Exception] | None
        ) = None,
        decode_results: (
            dict[str, list[DecodeResult | RequestFailure | Exception]] | None
        ) = None,
        decode_gate: threading.Event | None = None,
        decode_entered: threading.Event | None = None,
    ) -> None:
        self.prefill_results = prefill_results or {}
        self.decode_results = {k: list(v) for k, v in (decode_results or {}).items()}
        self.decode_gate = decode_gate
        self.decode_entered = decode_entered

    def prefill(
        self, request_state: GenerationRequestState
    ) -> PrefillResult | RequestFailure:
        result = self.prefill_results.get(request_state.request_id, prefill_result())
        if isinstance(result, Exception):
            raise result
        return result

    def decode(
        self, request_state: GenerationRequestState
    ) -> DecodeResult | RequestFailure:
        if self.decode_entered is not None:
            self.decode_entered.set()
        if self.decode_gate is not None:
            self.decode_gate.wait(timeout=5.0)

        scripted = self.decode_results.get(request_state.request_id)
        if scripted:
            result = scripted.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        return decode_result(done=True)


def make_worker(
    executor: FakeExecutor | None = None,
    max_queue_size: int = 16,
    max_active_requests: int = 4,
) -> SimpleWorker:
    return SimpleWorker(
        executor or FakeExecutor(),
        ExecutorConfig(
            max_queue_size=max_queue_size,
            max_active_requests=max_active_requests,
        ),
    )


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="max_queue_size"):
        make_worker(max_queue_size=0)
    with pytest.raises(ValueError, match="max_active_requests"):
        make_worker(max_active_requests=0)


def test_minimum_valid_config_constructs() -> None:
    worker = make_worker(max_queue_size=1, max_active_requests=1)

    assert worker._thread is None
    assert worker._inbound.maxsize == 1
    assert not worker._shutdown_event.is_set()


def test_start_stop_and_double_start() -> None:
    worker = make_worker()
    try:
        worker.start()
        first_thread = worker._thread
        assert first_thread is not None
        assert first_thread.is_alive()
        assert first_thread.daemon is True
        assert first_thread.name == "inference-worker"

        worker.start()
        assert worker._thread is first_thread
        assert worker._thread.is_alive()
    finally:
        worker.stop()

    assert worker._thread is not None
    assert not worker._thread.is_alive()


def test_worker_restart_after_stop() -> None:
    worker = make_worker()

    r0 = make_req("r0")
    worker.start()
    worker.submit(r0)
    assert wait_for_status(r0, RequestStatus.DONE)
    worker.stop()
    first_thread = worker._thread

    r1 = make_req("r1")
    worker.start()
    assert worker._thread is not first_thread
    worker.submit(r1)
    assert wait_for_status(r1, RequestStatus.DONE)
    worker.stop()


def test_submit_without_start_enqueues_and_stop_drains() -> None:
    worker = make_worker()
    req = make_req("r0")

    worker.submit(req)
    assert worker._inbound.qsize() == 1

    worker.stop()

    assert req.status == RequestStatus.FAILED
    events = drain_events(req)
    assert len(events) == 1 and isinstance(events[0], ErrorEvent)


def test_submit_after_stop_is_rejected() -> None:
    worker = make_worker()
    worker.start()
    worker.stop()

    with pytest.raises(RuntimeError, match="shutting down"):
        worker.submit(make_req("r0"))


def test_submit_full_queue_raises_queue_full() -> None:
    worker = make_worker(max_queue_size=1)
    worker.submit(make_req("r0"))

    with pytest.raises(queue.Full):
        worker.submit(make_req("r1"))


def test_submit_full_queue_while_worker_is_busy() -> None:
    decode_gate = threading.Event()
    decode_entered = threading.Event()
    worker = make_worker(
        FakeExecutor(decode_gate=decode_gate, decode_entered=decode_entered),
        max_queue_size=1,
        max_active_requests=1,
    )
    try:
        worker.start()
        worker.submit(make_req("r0"))
        assert decode_entered.wait(timeout=2.0)
        worker.submit(make_req("r1"))
        with pytest.raises(queue.Full):
            worker.submit(make_req("r2"))
    finally:
        decode_gate.set()
        worker.stop()


def test_happy_path_worker_engine_executor_integration_completes_request() -> None:
    worker = make_worker()
    req = make_req("r0")
    try:
        worker.start()
        worker.submit(req)
        assert wait_for_status(req, RequestStatus.DONE)
    finally:
        worker.stop()


def test_request_failure_result_emits_error_event() -> None:
    worker = make_worker(
        FakeExecutor(prefill_results={"r0": RequestFailure("prefill failed")})
    )
    req = make_req("r0")
    try:
        worker.start()
        worker.submit(req)
        assert wait_for_status(req, RequestStatus.FAILED)
    finally:
        worker.stop()

    events = drain_events(req)
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert events[0].error == "prefill failed"


def test_fatal_engine_callback_cancels_inbound_requests() -> None:
    worker = make_worker(
        FakeExecutor(prefill_results={"r0": RuntimeError("prefill boom")}),
        max_queue_size=4,
        max_active_requests=1,
    )
    r0, r1, r2, r3 = make_req("r0"), make_req("r1"), make_req("r2"), make_req("r3")
    worker.submit(r0)
    worker.submit(r1)
    worker.submit(r2)
    worker.submit(r3)

    worker.start()
    _wait_for_worker_to_die(worker)
    worker.stop()

    _assert_error_event(r0)
    _assert_error_event(r1)
    _assert_error_event(r2)
    _assert_error_event(r3)


def test_stop_cancels_active_request_blocked_in_decode() -> None:
    decode_entered = threading.Event()
    decode_gate = threading.Event()
    worker = make_worker(
        FakeExecutor(
            decode_results={"r0": [decode_result(done=False)]},
            decode_gate=decode_gate,
            decode_entered=decode_entered,
        ),
        max_active_requests=1,
    )
    req = make_req("r0")
    worker.start()
    worker.submit(req)
    assert decode_entered.wait(timeout=2.0)

    stop_done = threading.Event()
    threading.Thread(
        target=lambda: (worker.stop(), stop_done.set()), daemon=True
    ).start()

    deadline = time.monotonic() + 2.0
    while not worker._shutdown_event.is_set() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert worker._shutdown_event.is_set()
    decode_gate.set()
    assert stop_done.wait(timeout=5.0)

    assert req.status == RequestStatus.FAILED
    events = drain_events(req)
    assert len(events) == 2
    assert isinstance(events[0], TokenEvent)
    assert isinstance(events[1], ErrorEvent)
