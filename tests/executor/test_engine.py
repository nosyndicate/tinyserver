from queue import Queue
from typing import Callable

import torch
from transformers import DynamicCache

from server.executor.engine import (
    BatchInferenceEngine,
    EngineCallbacks,
    EngineControl,
    SimpleInferenceEngine,
)
from server.executor.types import (
    BatchExecutorConfig,
    DecodeResult,
    ErrorEvent,
    ExecutorConfig,
    FinishReason,
    GenerationRequestState,
    PrefillResult,
    RequestFailure,
    RequestStatus,
)
from server.model.sampling import SamplingParams


def make_req(request_id: str) -> GenerationRequestState:
    return GenerationRequestState(
        request_id=request_id,
        sampling_params=SamplingParams(max_new_tokens=10, temperature=1.0, top_p=1.0),
        prompt=f"prompt-{request_id}",
        enqueued_ns=0,
    )


def make_decoding_req(request_id: str) -> GenerationRequestState:
    req = make_req(request_id)
    req.status = RequestStatus.DECODING
    req.start_ns = 0
    req.num_prompt_tokens = 1
    req.all_logits = torch.empty(1, 1, 1)
    req.past_key_values = DynamicCache()
    return req


def prefill_result() -> PrefillResult:
    return PrefillResult(
        all_logits=torch.empty(1, 1, 1),
        past_key_values=DynamicCache(),
        num_prompt_tokens=1,
        start_ns=0,
    )


def decode_result(done: bool = True, token: str = "x") -> DecodeResult:
    return DecodeResult(
        token_id=1,
        token=token,
        finish_reason=FinishReason.MAX_LENGTH if done else None,
        all_logits=None if done else torch.empty(1, 1, 1),
        past_key_values=None if done else DynamicCache(),
    )


class LoopControl:
    def __init__(self, should_stop_values: list[bool] | None = None) -> None:
        self._should_stop_values = list(should_stop_values or [])
        self.stopped = False
        self.waits: list[float] = []

    def should_stop(self) -> bool:
        if self._should_stop_values:
            return self._should_stop_values.pop(0)
        return self.stopped

    def wait_idle(self, timeout: float) -> bool:
        self.waits.append(timeout)
        self.stopped = True
        return True

    def engine_control(self) -> EngineControl:
        return EngineControl(should_stop=self.should_stop, wait_idle=self.wait_idle)


class CallbackRecorder:
    def __init__(self) -> None:
        self.cancelled: list[tuple[GenerationRequestState, str]] = []
        self.fatal: list[tuple[Exception, list[GenerationRequestState] | None]] = []

    def cancel_request(self, req: GenerationRequestState, message: str) -> None:
        req.status = RequestStatus.FAILED
        req.error = message
        req.output_queue.put(ErrorEvent(request_id=req.request_id, error=message))
        self.cancelled.append((req, message))

    def handle_fatal_error(
        self,
        error: Exception,
        extra_requests: list[GenerationRequestState] | None,
    ) -> None:
        self.fatal.append((error, extra_requests))

    def engine_callbacks(self) -> EngineCallbacks:
        return EngineCallbacks(
            cancel_request=self.cancel_request,
            handle_fatal_error=self.handle_fatal_error,
        )


class FakeSimpleExecutor:
    def __init__(
        self,
        prefill: list[PrefillResult | RequestFailure | Exception] | None = None,
        decode: list[DecodeResult | RequestFailure | Exception] | None = None,
    ) -> None:
        self.prefill_results = list(prefill or [])
        self.decode_results = list(decode or [])
        self.prefill_calls: list[GenerationRequestState] = []
        self.decode_calls: list[GenerationRequestState] = []

    def prefill(
        self, request_state: GenerationRequestState
    ) -> PrefillResult | RequestFailure:
        self.prefill_calls.append(request_state)
        result = self.prefill_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def decode(
        self, request_state: GenerationRequestState
    ) -> DecodeResult | RequestFailure:
        self.decode_calls.append(request_state)
        result = self.decode_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeBatchExecutor:
    def __init__(
        self,
        prefill: list[list[PrefillResult | RequestFailure] | Exception] | None = None,
        decode: list[list[DecodeResult | RequestFailure] | Exception] | None = None,
    ) -> None:
        self.prefill_results = list(prefill or [])
        self.decode_results = list(decode or [])
        self.prefill_calls: list[list[GenerationRequestState]] = []
        self.decode_calls: list[list[GenerationRequestState]] = []

    def batched_prefill(
        self, request_states: list[GenerationRequestState]
    ) -> list[PrefillResult | RequestFailure]:
        self.prefill_calls.append(list(request_states))
        result = self.prefill_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def batched_decode(
        self, request_states: list[GenerationRequestState]
    ) -> list[DecodeResult | RequestFailure]:
        self.decode_calls.append(list(request_states))
        result = self.decode_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def fill_queue(reqs: list[GenerationRequestState]) -> Queue[GenerationRequestState]:
    inbound: Queue[GenerationRequestState] = Queue()
    for req in reqs:
        inbound.put_nowait(req)
    return inbound


def simple_config(max_active_requests: int = 2) -> ExecutorConfig:
    return ExecutorConfig(max_queue_size=16, max_active_requests=max_active_requests)


def batch_config(
    max_active_requests: int = 4,
    max_prefill_batch_size: int = 2,
    max_decode_batch_size: int = 2,
) -> BatchExecutorConfig:
    return BatchExecutorConfig(
        max_queue_size=16,
        max_active_requests=max_active_requests,
        max_prefill_batch_size=max_prefill_batch_size,
        max_decode_batch_size=max_decode_batch_size,
    )


def run_engine(
    run: Callable[
        [Queue[GenerationRequestState], EngineControl, EngineCallbacks], None
    ],
    inbound: Queue[GenerationRequestState],
    control: LoopControl | None = None,
    callbacks: CallbackRecorder | None = None,
) -> tuple[LoopControl, CallbackRecorder]:
    control = control or LoopControl()
    callbacks = callbacks or CallbackRecorder()
    run(inbound, control.engine_control(), callbacks.engine_callbacks())
    return control, callbacks


def test_simple_engine_drains_at_most_max_active_requests() -> None:
    executor = FakeSimpleExecutor(
        prefill=[prefill_result(), prefill_result()],
        decode=[decode_result(), decode_result()],
    )
    engine = SimpleInferenceEngine(executor, simple_config(max_active_requests=2))
    reqs = [make_req(f"r{i}") for i in range(3)]
    inbound = fill_queue(reqs)

    run_engine(engine.run, inbound)

    assert executor.prefill_calls == reqs[:2]
    assert inbound.qsize() == 1


def test_simple_engine_applies_prefill_and_only_decodes_successes() -> None:
    executor = FakeSimpleExecutor(
        prefill=[prefill_result(), RequestFailure("prefill failed")],
        decode=[decode_result()],
    )
    engine = SimpleInferenceEngine(executor, simple_config(max_active_requests=2))
    r0, r1 = make_req("r0"), make_req("r1")

    run_engine(engine.run, fill_queue([r0, r1]))

    assert r0.status == RequestStatus.DONE
    assert r0.num_prompt_tokens == 1
    assert r1.status == RequestStatus.FAILED
    assert executor.decode_calls == [r0]
    assert engine._active == []


def test_simple_engine_applies_decode_result_and_prunes_completed_requests() -> None:
    executor = FakeSimpleExecutor(prefill=[], decode=[decode_result(done=True)])
    engine = SimpleInferenceEngine(executor, simple_config())
    req = make_decoding_req("r0")
    engine._active.append(req)

    run_engine(engine.run, Queue())

    assert req.status == RequestStatus.DONE
    assert engine._active == []


def test_simple_engine_prefill_exception_routes_remaining_new_requests() -> None:
    error = RuntimeError("prefill crash")
    executor = FakeSimpleExecutor(prefill=[error])
    engine = SimpleInferenceEngine(executor, simple_config(max_active_requests=2))
    r0, r1 = make_req("r0"), make_req("r1")

    _, callbacks = run_engine(engine.run, fill_queue([r0, r1]))

    assert callbacks.fatal == [(error, [r0, r1])]


def test_simple_engine_decode_exception_routes_fatal_without_extra_requests() -> None:
    error = RuntimeError("decode crash")
    executor = FakeSimpleExecutor(decode=[error])
    engine = SimpleInferenceEngine(executor, simple_config())
    engine._active.append(make_decoding_req("r0"))

    _, callbacks = run_engine(engine.run, Queue())

    assert callbacks.fatal == [(error, None)]


def test_simple_engine_shutdown_before_prefill_cancels_new_and_active_requests() -> (
    None
):
    executor = FakeSimpleExecutor()
    engine = SimpleInferenceEngine(executor, simple_config(max_active_requests=2))
    active = make_decoding_req("active")
    new = make_req("new")
    engine._active.append(active)

    _, callbacks = run_engine(
        engine.run,
        fill_queue([new]),
        control=LoopControl([False, True]),
    )

    assert [req for req, _ in callbacks.cancelled] == [new, active]
    assert engine._active == []


def test_simple_engine_shutdown_during_decode_cancels_active_requests() -> None:
    executor = FakeSimpleExecutor(decode=[decode_result(done=False)])
    engine = SimpleInferenceEngine(executor, simple_config())
    r0, r1 = make_decoding_req("r0"), make_decoding_req("r1")
    engine._active.extend([r0, r1])

    _, callbacks = run_engine(
        engine.run,
        Queue(),
        control=LoopControl([False, False, True]),
    )

    assert executor.decode_calls == [r0]
    assert [req for req, _ in callbacks.cancelled] == [r0, r1]
    assert engine._active == []


def test_batch_engine_drain_and_selection_methods_respect_limits() -> None:
    engine = BatchInferenceEngine(
        FakeBatchExecutor(),
        batch_config(
            max_active_requests=3,
            max_prefill_batch_size=2,
            max_decode_batch_size=1,
        ),
    )
    active = make_decoding_req("active")
    engine._active.append(active)
    waiting = [make_req("w0"), make_req("w1"), make_req("w2")]
    inbound = fill_queue(waiting)

    engine.drain_inbound(inbound)

    assert engine._waiting == waiting[:2]
    assert inbound.qsize() == 1

    prefill_batch = engine.select_prefill_batch()
    assert prefill_batch == waiting[:2]
    assert engine._waiting == []

    r0, r1 = make_decoding_req("r0"), make_decoding_req("r1")
    done = make_req("done")
    done.status = RequestStatus.DONE
    engine._active = [r0, done, r1]
    assert engine.select_decode_batch() == [r0]


def test_batch_engine_prefill_failure_does_not_enter_active() -> None:
    executor = FakeBatchExecutor(prefill=[[RequestFailure("prefill failed")]])
    engine = BatchInferenceEngine(executor, batch_config(max_prefill_batch_size=1))
    req = make_req("r0")

    run_engine(engine.run, fill_queue([req]))

    assert req.status == RequestStatus.FAILED
    assert engine._active == []


def test_batch_engine_decode_failure_is_pruned_from_active() -> None:
    executor = FakeBatchExecutor(decode=[[RequestFailure("decode failed")]])
    engine = BatchInferenceEngine(executor, batch_config())
    req = make_decoding_req("r0")
    engine._active.append(req)

    run_engine(engine.run, Queue())

    assert req.status == RequestStatus.FAILED
    assert engine._active == []


def test_batch_engine_prefill_result_length_mismatch_routes_fatal() -> None:
    executor = FakeBatchExecutor(prefill=[[]])
    engine = BatchInferenceEngine(executor, batch_config(max_prefill_batch_size=1))
    req = make_req("r0")

    _, callbacks = run_engine(engine.run, fill_queue([req]))

    assert len(callbacks.fatal) == 1
    error, extra = callbacks.fatal[0]
    assert isinstance(error, ValueError)
    assert "Expected 1 prefill results, but got 0" in str(error)
    assert extra == [req]


def test_batch_engine_decode_result_length_mismatch_routes_fatal() -> None:
    executor = FakeBatchExecutor(decode=[[]])
    engine = BatchInferenceEngine(executor, batch_config())
    req = make_decoding_req("r0")
    engine._active.append(req)

    _, callbacks = run_engine(engine.run, Queue())

    assert len(callbacks.fatal) == 1
    error, extra = callbacks.fatal[0]
    assert isinstance(error, ValueError)
    assert "Expected 1 decode results, but got 0" in str(error)
    assert extra == []


def test_batch_engine_fatal_prefill_exception_includes_current_prefill_batch() -> None:
    error = RuntimeError("prefill crash")
    executor = FakeBatchExecutor(prefill=[error])
    engine = BatchInferenceEngine(executor, batch_config(max_prefill_batch_size=2))
    r0, r1 = make_req("r0"), make_req("r1")

    _, callbacks = run_engine(engine.run, fill_queue([r0, r1]))

    assert callbacks.fatal == [(error, [r0, r1])]


def test_batch_engine_cancel_inflight_cancels_waiting_and_active_requests() -> None:
    engine = BatchInferenceEngine(FakeBatchExecutor(), batch_config())
    callbacks = CallbackRecorder()
    waiting = make_req("waiting")
    active = make_decoding_req("active")
    engine._waiting.append(waiting)
    engine._active.append(active)

    engine.cancel_inflight("cancelled", callbacks.cancel_request)

    assert [req for req, _ in callbacks.cancelled] == [waiting, active]
    assert engine._waiting == []
    assert engine._active == []


def test_batch_engine_idle_state_waits_briefly() -> None:
    engine = BatchInferenceEngine(FakeBatchExecutor(), batch_config())

    control, callbacks = run_engine(engine.run, Queue())

    assert callbacks.fatal == []
    assert control.waits == [0.01]
