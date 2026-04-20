import queue

import pytest
import torch
from transformers import DynamicCache

from server.executor.events import RequestEventEmitter
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
        enqueued_ns=0,
    )


def drain_events(req: GenerationRequestState) -> list:
    events: list = []
    while True:
        try:
            events.append(req.output_queue.get_nowait())
        except queue.Empty:
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
    assert isinstance(events[1], DoneEvent)
    assert events[1].text == "a"
    assert events[1].num_output_tokens == 1


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
