import queue
import threading
import time

import pytest

from server.executor.engine import BatchInferenceEngine, SimpleInferenceEngine
from server.executor.types import (
    BaseBatchExecutor,
    BaseExecutor,
    BatchExecutorConfig,
    DecodeResult,
    ErrorEvent,
    ExecutorConfig,
    GenerationRequestState,
    PrefillResult,
    RequestFailure,
    RequestStatus,
    TokenEvent,
)
from server.executor.worker import Worker
from server.metrics.timers import NS_PER_S, now_ns

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
) -> Worker:
    exec_impl = executor or FakeExecutor()
    config = ExecutorConfig(
        max_active_requests=max_active_requests,
    )
    return Worker(
        SimpleInferenceEngine(exec_impl, config), max_queue_size=max_queue_size
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"max_queue_size": 0}, "max_queue_size"),
        ({"max_active_requests": 0}, "max_active_requests"),
    ],
)
def test_config_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make_worker(**kwargs)


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

    deadline = now_ns() + 2 * NS_PER_S
    while not worker._shutdown_event.is_set() and now_ns() < deadline:
        time.sleep(0.001)
    assert worker._shutdown_event.is_set()
    decode_gate.set()
    assert stop_done.wait(timeout=5.0)

    assert req.status == RequestStatus.FAILED
    events = drain_events(req)
    assert len(events) == 2
    assert isinstance(events[0], TokenEvent)
    assert isinstance(events[1], ErrorEvent)


# --- Batch-specific tests ---


class FakeBatchExecutor(BaseBatchExecutor):
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
        self.prefill_call_sizes: list[int] = []
        self.decode_call_sizes: list[int] = []

    def batched_prefill(
        self, request_states: list[GenerationRequestState]
    ) -> list[PrefillResult | RequestFailure]:
        self.prefill_call_sizes.append(len(request_states))
        results: list[PrefillResult | RequestFailure] = []
        for req in request_states:
            result = self.prefill_results.get(req.request_id, prefill_result())
            if isinstance(result, Exception):
                raise result
            results.append(result)
        return results

    def batched_decode(
        self, request_states: list[GenerationRequestState]
    ) -> list[DecodeResult | RequestFailure]:
        self.decode_call_sizes.append(len(request_states))
        if self.decode_entered is not None:
            self.decode_entered.set()
        if self.decode_gate is not None:
            self.decode_gate.wait(timeout=5.0)

        results: list[DecodeResult | RequestFailure] = []
        for req in request_states:
            scripted = self.decode_results.get(req.request_id)
            if scripted:
                result = scripted.pop(0)
                if isinstance(result, Exception):
                    raise result
                results.append(result)
                continue

            results.append(decode_result(done=True))
        return results


def make_batch_worker(
    executor: FakeBatchExecutor | None = None,
    max_queue_size: int = 16,
    max_active_requests: int = 4,
    max_prefill_batch_size: int = 4,
    max_decode_batch_size: int = 4,
) -> Worker:
    exec_impl = executor or FakeBatchExecutor()
    config = BatchExecutorConfig(
        max_active_requests=max_active_requests,
        max_prefill_batch_size=max_prefill_batch_size,
        max_decode_batch_size=max_decode_batch_size,
    )
    return Worker(
        BatchInferenceEngine(exec_impl, config), max_queue_size=max_queue_size
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"max_prefill_batch_size": 0}, "max_prefill_batch_size"),
        ({"max_decode_batch_size": 0}, "max_decode_batch_size"),
        (
            {"max_active_requests": 2, "max_prefill_batch_size": 3},
            "max_prefill_batch_size cannot be greater",
        ),
        (
            {
                "max_active_requests": 2,
                "max_prefill_batch_size": 2,
                "max_decode_batch_size": 3,
            },
            "max_decode_batch_size cannot be greater",
        ),
    ],
)
def test_batch_config_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make_batch_worker(**kwargs)


def test_concurrent_submits_from_multiple_threads() -> None:
    worker = make_batch_worker(
        max_queue_size=50,
        max_active_requests=10,
        max_prefill_batch_size=5,
        max_decode_batch_size=5,
    )
    num_threads = 5
    reqs_per_thread = 5
    all_reqs = [
        [make_req(f"t{t}r{r}") for r in range(reqs_per_thread)]
        for t in range(num_threads)
    ]

    worker.start()
    try:
        threads = [
            threading.Thread(
                target=lambda reqs=thread_reqs: [worker.submit(req) for req in reqs]
            )
            for thread_reqs in all_reqs
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for thread_reqs in all_reqs:
            for req in thread_reqs:
                assert wait_for_status(req, RequestStatus.DONE)
    finally:
        worker.stop()


def test_concurrent_submits_exceeds_queue_size() -> None:
    max_queue_size = 5
    worker = make_batch_worker(
        max_queue_size=max_queue_size,
        max_active_requests=10,
        max_prefill_batch_size=5,
        max_decode_batch_size=5,
    )

    pre_filled = [make_req(f"pre{i}") for i in range(max_queue_size)]
    for req in pre_filled:
        worker.submit(req)

    num_threads = 5
    reqs_per_thread = 5
    overflow_reqs = [
        [make_req(f"ov_t{t}r{r}") for r in range(reqs_per_thread)]
        for t in range(num_threads)
    ]
    rejected_count = 0
    lock = threading.Lock()

    def submit_thread(thread_reqs: list[GenerationRequestState]) -> None:
        nonlocal rejected_count
        for req in thread_reqs:
            try:
                worker.submit(req)
            except queue.Full:
                with lock:
                    rejected_count += 1

    threads = [
        threading.Thread(target=submit_thread, args=(thread_reqs,))
        for thread_reqs in overflow_reqs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert rejected_count == num_threads * reqs_per_thread

    worker.start()
    try:
        for req in pre_filled:
            assert wait_for_status(req, RequestStatus.DONE)
    finally:
        worker.stop()


def test_max_active_requests_never_exceed() -> None:
    max_active = 3
    decode_gate = threading.Event()
    decode_entered = threading.Event()
    worker = make_batch_worker(
        FakeBatchExecutor(decode_gate=decode_gate, decode_entered=decode_entered),
        max_queue_size=10,
        max_active_requests=max_active,
        max_prefill_batch_size=max_active,
        max_decode_batch_size=max_active,
    )
    for i in range(6):
        worker.submit(make_req(f"r{i}"))

    worker.start()
    try:
        assert decode_entered.wait(timeout=2.0)
        engine = worker._engine
        assert len(engine._active) + len(engine._waiting) <= max_active
    finally:
        decode_gate.set()
        worker.stop()


def test_stop_after_decode_crash_is_safe() -> None:
    worker = make_batch_worker(
        FakeBatchExecutor(decode_results={"r0": [RuntimeError("decode crash")]}),
        max_active_requests=1,
        max_prefill_batch_size=1,
        max_decode_batch_size=1,
    )
    req = make_req("r0")
    worker.start()
    worker.submit(req)

    _wait_for_worker_to_die(worker)
    worker.stop()

    _assert_error_event(req)


def test_active_list_pruned_after_iteration() -> None:
    worker = make_batch_worker(
        max_active_requests=2,
        max_prefill_batch_size=2,
        max_decode_batch_size=2,
    )
    r0, r1 = make_req("r0"), make_req("r1")

    worker.start()
    try:
        worker.submit(r0)
        worker.submit(r1)
        assert wait_for_status(r0, RequestStatus.DONE)
        assert wait_for_status(r1, RequestStatus.DONE)

        deadline = now_ns() + 2 * NS_PER_S
        while worker._engine._active and now_ns() < deadline:
            time.sleep(0.001)
        assert worker._engine._active == []
    finally:
        worker.stop()


def test_prefill_batch_size_capped_by_config() -> None:
    executor = FakeBatchExecutor()
    worker = make_batch_worker(
        executor,
        max_queue_size=16,
        max_active_requests=5,
        max_prefill_batch_size=2,
        max_decode_batch_size=5,
    )
    reqs = [make_req(f"r{i}") for i in range(5)]
    for req in reqs:
        worker.submit(req)

    worker.start()
    try:
        for req in reqs:
            assert wait_for_status(req, RequestStatus.DONE)
    finally:
        worker.stop()

    assert executor.prefill_call_sizes
    assert max(executor.prefill_call_sizes) <= 2


def test_decode_batch_size_capped_by_config() -> None:
    executor = FakeBatchExecutor()
    worker = make_batch_worker(
        executor,
        max_queue_size=16,
        max_active_requests=5,
        max_prefill_batch_size=5,
        max_decode_batch_size=2,
    )
    reqs = [make_req(f"r{i}") for i in range(5)]
    for req in reqs:
        worker.submit(req)

    worker.start()
    try:
        for req in reqs:
            assert wait_for_status(req, RequestStatus.DONE)
    finally:
        worker.stop()

    assert executor.decode_call_sizes
    assert max(executor.decode_call_sizes) <= 2
