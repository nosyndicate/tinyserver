from queue import Empty

import pytest
import torch
from transformers import DynamicCache

from server.executor.events import RequestEventEmitter, compute_timings
from server.executor.sinks import SharedQueueSink
from server.executor.types import (
    DecodeResult,
    DoneEvent,
    ErrorEvent,
    FinishReason,
    GenerationRequestState,
    PrefillResult,
    RequestStatus,
    TokenEvent,
)
from server.model.sampling import SamplingParams


def make_req() -> GenerationRequestState:
    return GenerationRequestState(
        request_id="r0",
        sampling_params=SamplingParams(max_new_tokens=2, temperature=1.0, top_p=1.0),
        prompt="hello",
        sink=SharedQueueSink(),
        enqueued_ns=0,
    )


def drain_events(req: GenerationRequestState) -> list:
    assert isinstance(req.sink, SharedQueueSink)
    events: list = []
    while True:
        try:
            events.append(req.sink.queue.get_nowait())
        except Empty:
            return events


def test_prefill_success_updates_request_state() -> None:
    req = make_req()
    logits = torch.randn(1, 3, 10)
    cache = DynamicCache()
    result = PrefillResult(
        all_logits=logits,
        past_key_values=cache,
        num_prompt_tokens=3,
        start_ns=10,
    )

    emitter = RequestEventEmitter()
    emitter.on_prefill_started(req, result.start_ns)
    emitter.on_prefill_succeeded(req, result)

    assert req.status == RequestStatus.DECODING
    assert req.start_ns == 10
    assert req.all_logits is logits
    assert req.past_key_values is cache
    assert req.num_prompt_tokens == 3


def test_first_token_emits_token_event_and_updates_decode_state() -> None:
    req = make_req()
    req.status = RequestStatus.DECODING
    req.start_ns = 0
    logits = torch.randn(1, 1, 10)
    cache = DynamicCache()

    RequestEventEmitter().on_token(
        req,
        DecodeResult(
            token_id=7,
            token="a",
            finish_reason=None,
            all_logits=logits,
            past_key_values=cache,
        ),
    )

    events = drain_events(req)
    assert req.first_token_ns is not None
    assert req.output_tokens == ["a"]
    assert req.all_logits is logits
    assert req.past_key_values is cache
    assert isinstance(events[0], TokenEvent)
    assert events[0].is_first is True
    assert events[0].is_last is False
    assert events[0].index == 0


def test_second_token_is_not_first() -> None:
    req = make_req()
    req.status = RequestStatus.DECODING
    req.start_ns = 0
    req.first_token_ns = 1
    req.output_tokens.append("a")

    RequestEventEmitter().on_token(
        req,
        DecodeResult(token_id=8, token="b", finish_reason=None),
    )

    event = drain_events(req)[0]
    assert isinstance(event, TokenEvent)
    assert event.is_first is False
    assert event.index == 1


def test_eos_emits_final_empty_token_and_done_event() -> None:
    req = make_req()
    req.status = RequestStatus.DECODING
    req.start_ns = 0
    req.num_prompt_tokens = 3
    req.output_tokens.append("a")

    RequestEventEmitter().on_token(
        req,
        DecodeResult(
            token_id=2,
            token="",
            finish_reason=FinishReason.EOS,
        ),
    )

    events = drain_events(req)
    assert req.status == RequestStatus.DONE
    assert req.finished_reason == FinishReason.EOS
    assert isinstance(events[0], TokenEvent)
    assert events[0].token == ""
    assert events[0].is_last is True
    assert events[0].request_id == req.request_id
    assert isinstance(events[1], DoneEvent)
    assert events[1].text == "a"
    assert events[1].num_output_tokens == 1
    assert events[1].request_id == req.request_id


def test_max_length_emits_final_token_and_done_event() -> None:
    req = make_req()
    req.status = RequestStatus.DECODING
    req.start_ns = 0
    req.num_prompt_tokens = 3

    RequestEventEmitter().on_token(
        req,
        DecodeResult(
            token_id=2,
            token="a",
            finish_reason=FinishReason.MAX_LENGTH,
        ),
    )

    events = drain_events(req)
    assert req.status == RequestStatus.DONE
    assert req.finished_reason == FinishReason.MAX_LENGTH
    assert req.output_tokens == ["a"]
    assert isinstance(events[0], TokenEvent)
    assert events[0].token == "a"
    assert events[0].is_last is True
    assert isinstance(events[1], DoneEvent)
    assert events[1].text == "a"


def test_failure_emits_error_event() -> None:
    req = make_req()

    RequestEventEmitter().on_failed(req, "model error")

    events = drain_events(req)
    assert req.status == RequestStatus.FAILED
    assert req.error == "model error"
    assert isinstance(events[0], ErrorEvent)
    assert events[0].request_id == "r0"
    assert events[0].error == "model error"


# --- timing arithmetic -------------------------------------------------------
#
# compute_timings is pure, so the timeline can be written out directly instead
# of faking a clock. All values are nanoseconds; 1 ms == 1_000_000 ns.


def test_timings_split_queue_and_execution() -> None:
    timings = compute_timings(
        enqueued_ns=0,
        start_ns=30_000_000,  # queued 30 ms
        first_token_ns=45_000_000,
        end_ns=130_000_000,  # executed 100 ms
    )

    assert timings.queue_wait_ms == pytest.approx(30.0)
    assert timings.execution_ms == pytest.approx(100.0)
    assert timings.total_ms == pytest.approx(130.0)
    # TTFT is measured from enqueue, so it includes the queue wait.
    assert timings.ttft_ms == pytest.approx(45.0)


@pytest.mark.parametrize(
    "enqueued_ns,start_ns,end_ns",
    [
        (0, 0, 1_000_000),  # no queue wait at all
        (0, 5_000_000, 5_000_000),  # queued, then finished instantly
        (1_000, 900_000_000, 4_000_000_000),  # queue dominates
        (7_777_777, 8_888_888, 999_999_999),  # non-round timestamps
    ],
)
def test_total_equals_queue_plus_execution(
    enqueued_ns: int, start_ns: int, end_ns: int
) -> None:
    timings = compute_timings(
        enqueued_ns=enqueued_ns,
        start_ns=start_ns,
        first_token_ns=None,
        end_ns=end_ns,
    )

    assert timings.total_ms == pytest.approx(
        timings.queue_wait_ms + timings.execution_ms
    )


def test_heavy_queue_wait_does_not_zero_out_execution() -> None:
    # The previous implementation computed execution_ms as
    # (end - start) - queue_wait, which clamped to 0.0 whenever the queue wait
    # exceeded the execution time. Under load that was the common case.
    timings = compute_timings(
        enqueued_ns=0,
        start_ns=4_000_000_000,  # 4 s in the queue
        first_token_ns=4_020_000_000,
        end_ns=5_000_000_000,  # 1 s executing
    )

    assert timings.queue_wait_ms == pytest.approx(4000.0)
    assert timings.execution_ms == pytest.approx(1000.0)
    assert timings.total_ms == pytest.approx(5000.0)


def test_ttft_is_none_without_a_first_token() -> None:
    timings = compute_timings(
        enqueued_ns=0,
        start_ns=1_000_000,
        first_token_ns=None,
        end_ns=2_000_000,
    )

    assert timings.ttft_ms is None


def test_timings_are_never_negative() -> None:
    # Defensive: out-of-order timestamps should clamp rather than produce
    # negative values that would fail the API schema's ge=0.0 constraint.
    timings = compute_timings(
        enqueued_ns=5_000_000,
        start_ns=1_000_000,
        first_token_ns=2_000_000,
        end_ns=0,
    )

    assert timings.total_ms == 0.0
    assert timings.queue_wait_ms == 0.0
    assert timings.execution_ms == 0.0
    assert timings.ttft_ms == 0.0


def test_done_event_carries_timings_measured_from_enqueue() -> None:
    req = make_req()
    req.status = RequestStatus.DECODING
    req.num_prompt_tokens = 3
    # enqueued_ns is 0 from make_req(); start_ns is set far in the past relative
    # to the real perf_counter clock read inside _finish, so total_ms and
    # execution_ms are both large and positive, and their difference is exactly
    # the fabricated queue wait.
    req.start_ns = 25_000_000

    RequestEventEmitter().on_token(
        req,
        DecodeResult(token_id=2, token="a", finish_reason=FinishReason.MAX_LENGTH),
    )

    done = drain_events(req)[1]
    assert isinstance(done, DoneEvent)
    assert done.queue_wait_ms == pytest.approx(25.0)
    assert done.total_ms == pytest.approx(done.queue_wait_ms + done.execution_ms)
    assert done.ttft_ms is not None
    assert done.ttft_ms >= done.queue_wait_ms


def test_finish_requires_start_ns() -> None:
    req = make_req()
    req.num_prompt_tokens = 3

    with pytest.raises(RuntimeError, match="start_ns"):
        RequestEventEmitter().on_token(
            req,
            DecodeResult(
                token_id=2,
                token="",
                finish_reason=FinishReason.EOS,
            ),
        )


def test_finish_requires_enqueued_ns() -> None:
    req = make_req()
    req.start_ns = 0
    req.enqueued_ns = None
    req.num_prompt_tokens = 3

    with pytest.raises(RuntimeError, match="enqueued_ns"):
        RequestEventEmitter().on_token(
            req,
            DecodeResult(
                token_id=2,
                token="",
                finish_reason=FinishReason.EOS,
            ),
        )


def test_finish_requires_prompt_tokens() -> None:
    req = make_req()
    req.start_ns = 0

    with pytest.raises(RuntimeError, match="num_prompt_tokens"):
        RequestEventEmitter().on_token(
            req,
            DecodeResult(
                token_id=2,
                token="",
                finish_reason=FinishReason.EOS,
            ),
        )
