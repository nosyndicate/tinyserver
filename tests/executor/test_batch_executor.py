"""Tests for BatchWorker._prefill_batch and BatchWorker._decode_batch_step.

These tests exercise the real worker methods (which replaced the old
BatchExecutor class) with a hand-rolled FakeModelRunner that implements only
the small runner surface the worker consumes.
"""

import itertools
import queue

import torch
from transformers import DynamicCache

from server.executor.types import (
    BatchExecutorConfig,
    DoneEvent,
    ErrorEvent,
    FinishReason,
    GenerationRequestState,
    RequestStatus,
    TokenEvent,
)
from server.executor.worker import BatchWorker
from server.model.batch_ops import DecodeBatchOutput, PrefillBatchOutput
from server.model.sampling import SamplingParams

# ─── Test infrastructure ──────────────────────────────────────────────────────

VOCAB = 100
EOS = 2


class FakeTokenizer:
    """Minimal tokenizer fake."""

    def __init__(
        self,
        token_map: dict[int, str] | None = None,
        default: str = "x",
    ) -> None:
        self._map = token_map or {}
        self._default = default

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(self._map.get(tid, self._default) for tid in token_ids)


class FakeModelRunner:
    """Hand-crafted fake ModelRunner for BatchWorker tests.

    Exposes only the runner surface the worker actually calls:
        - prefill_batch(prompts) -> list[PrefillBatchOutput]
        - decode_batch(token_ids, pkvs) -> list[DecodeBatchOutput]
        - sample_token(logits, params, generator) -> int
        - eos_token_id, tokenizer
    """

    def __init__(
        self,
        sample_tokens: list[int] | None = None,
        prefill_batches: list[list[PrefillBatchOutput]] | None = None,
        decode_batches: list[list[DecodeBatchOutput]] | None = None,
        eos_token_id: int = EOS,
        token_map: dict[int, str] | None = None,
    ) -> None:
        self.tokenizer = FakeTokenizer(token_map)
        self._eos_token_id = eos_token_id
        self._sample_tokens = list(sample_tokens or [])
        self._prefill_batches = list(prefill_batches or [])
        self._decode_batches = list(decode_batches or [])

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    def sample_token(
        self,
        logits: torch.Tensor,
        sampling_params: SamplingParams,
        generator: torch.Generator | None = None,
    ) -> int:
        return self._sample_tokens.pop(0)

    def prefill_batch(self, prompts: list[str]) -> list[PrefillBatchOutput]:
        return self._prefill_batches.pop(0)

    def decode_batch(
        self,
        token_ids: list[int],
        past_key_values: list[DynamicCache],
    ) -> list[DecodeBatchOutput]:
        return self._decode_batches.pop(0)


_counter = itertools.count()


def make_req(request_id: str | None = None) -> GenerationRequestState:
    rid = request_id if request_id is not None else f"req-{next(_counter)}"
    return GenerationRequestState(
        request_id=rid,
        sampling_params=SamplingParams(max_new_tokens=10, temperature=1.0, top_p=1.0),
        prompt="hello",
    )


def drain_events(req: GenerationRequestState) -> list:
    events: list = []
    while True:
        try:
            events.append(req.output_queue.get_nowait())
        except queue.Empty:
            break
    return events


def _make_prefill_output(
    num_prompt_tokens: int = 5,
) -> PrefillBatchOutput:
    return PrefillBatchOutput(
        logits=torch.randn(1, num_prompt_tokens, VOCAB),
        past_key_values=DynamicCache(),
        num_prompt_tokens=num_prompt_tokens,
    )


def _make_decode_output() -> DecodeBatchOutput:
    return DecodeBatchOutput(
        logits=torch.randn(1, 1, VOCAB),
        past_key_values=DynamicCache(),
    )


def make_worker(runner: FakeModelRunner) -> BatchWorker:
    return BatchWorker(
        runner,  # type: ignore[arg-type]
        BatchExecutorConfig(
            max_queue_size=16,
            max_active_requests=4,
            max_prefill_batch_size=4,
            max_decode_batch_size=4,
        ),
    )


# ─── Group 1: _prefill_batch — happy path ────────────────────────────────────


def test_batched_prefill_transitions_to_decoding() -> None:
    runner = FakeModelRunner(
        prefill_batches=[
            [_make_prefill_output(5), _make_prefill_output(8)],
        ],
    )
    worker = make_worker(runner)
    r0, r1 = make_req("r0"), make_req("r1")

    worker._prefill_batch([r0, r1])

    assert r0.status == RequestStatus.DECODING
    assert r1.status == RequestStatus.DECODING


def test_batched_prefill_sets_shared_start_ns() -> None:
    runner = FakeModelRunner(
        prefill_batches=[
            [_make_prefill_output(), _make_prefill_output()],
        ],
    )
    worker = make_worker(runner)
    r0, r1 = make_req("r0"), make_req("r1")

    worker._prefill_batch([r0, r1])

    assert r0.start_ns is not None
    assert r1.start_ns is not None
    assert r0.start_ns == r1.start_ns


def test_batched_prefill_assigns_outputs_to_correct_requests() -> None:
    out0 = _make_prefill_output(5)
    out1 = _make_prefill_output(8)
    runner = FakeModelRunner(prefill_batches=[[out0, out1]])
    worker = make_worker(runner)
    r0, r1 = make_req("r0"), make_req("r1")

    worker._prefill_batch([r0, r1])

    assert r0.all_logits is out0.logits
    assert r0.past_key_values is out0.past_key_values
    assert r0.num_prompt_tokens == 5
    assert r1.all_logits is out1.logits
    assert r1.past_key_values is out1.past_key_values
    assert r1.num_prompt_tokens == 8


def test_batched_prefill_single_request() -> None:
    runner = FakeModelRunner(prefill_batches=[[_make_prefill_output(3)]])
    worker = make_worker(runner)
    req = make_req("r0")

    worker._prefill_batch([req])

    assert req.status == RequestStatus.DECODING
    assert req.num_prompt_tokens == 3


# ─── Group 2: _prefill_batch — error handling ────────────────────────────────


def test_batched_prefill_runner_exception_marks_all_failed() -> None:
    runner = FakeModelRunner(prefill_batches=[])
    # Override to raise
    runner.prefill_batch = lambda prompts: (_ for _ in ()).throw(
        RuntimeError("model crash")
    )
    worker = make_worker(runner)
    r0, r1 = make_req("r0"), make_req("r1")

    worker._prefill_batch([r0, r1])

    assert r0.status == RequestStatus.FAILED
    assert r1.status == RequestStatus.FAILED
    for req in [r0, r1]:
        events = drain_events(req)
        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)


def test_batched_prefill_output_count_mismatch_marks_all_failed() -> None:
    runner = FakeModelRunner(prefill_batches=[[_make_prefill_output()]])
    worker = make_worker(runner)
    r0, r1 = make_req("r0"), make_req("r1")

    worker._prefill_batch([r0, r1])

    assert r0.status == RequestStatus.FAILED
    assert r1.status == RequestStatus.FAILED
    for req in [r0, r1]:
        events = drain_events(req)
        assert len(events) == 1
        assert isinstance(events[0], ErrorEvent)


# ─── Group 3: _decode_batch_step — token generation and state ────────────────


def test_batched_decode_updates_logits_and_cache() -> None:
    req = make_req("r0")
    req.status = RequestStatus.DECODING
    req.start_ns = 0
    req.enqueued_ns = 0
    req.all_logits = torch.randn(1, 5, VOCAB)
    req.past_key_values = DynamicCache()

    new_logits = torch.randn(1, 1, VOCAB)
    new_cache = DynamicCache()
    runner = FakeModelRunner(
        sample_tokens=[42],
        decode_batches=[
            [DecodeBatchOutput(logits=new_logits, past_key_values=new_cache)],
        ],
        token_map={42: "a"},
    )
    worker = make_worker(runner)

    worker._decode_batch_step([req])

    assert req.all_logits is new_logits
    assert req.past_key_values is new_cache


def test_batched_decode_sets_first_token_ns_on_first_step() -> None:
    req = make_req("r0")
    req.status = RequestStatus.DECODING
    req.start_ns = 0
    req.enqueued_ns = 0
    req.all_logits = torch.randn(1, 5, VOCAB)
    req.past_key_values = DynamicCache()

    runner = FakeModelRunner(
        sample_tokens=[42],
        decode_batches=[[_make_decode_output()]],
        token_map={42: "a"},
    )
    worker = make_worker(runner)

    worker._decode_batch_step([req])

    assert req.first_token_ns is not None


def test_batched_decode_emits_token_event() -> None:
    req = make_req("r0")
    req.status = RequestStatus.DECODING
    req.start_ns = 0
    req.enqueued_ns = 0
    req.all_logits = torch.randn(1, 5, VOCAB)
    req.past_key_values = DynamicCache()

    runner = FakeModelRunner(
        sample_tokens=[42],
        decode_batches=[[_make_decode_output()]],
        token_map={42: "hello"},
    )
    worker = make_worker(runner)

    worker._decode_batch_step([req])

    events = drain_events(req)
    assert len(events) == 1
    tok_event = events[0]
    assert isinstance(tok_event, TokenEvent)
    assert tok_event.token == "hello"
    assert tok_event.is_first is True
    assert tok_event.is_last is False
    assert tok_event.index == 0


def test_batched_decode_second_step_not_first() -> None:
    req = make_req("r0")
    req.status = RequestStatus.DECODING
    req.start_ns = 0
    req.enqueued_ns = 0
    req.first_token_ns = 100
    req.all_logits = torch.randn(1, 5, VOCAB)
    req.past_key_values = DynamicCache()
    req.output_tokens.append("a")

    runner = FakeModelRunner(
        sample_tokens=[42],
        decode_batches=[[_make_decode_output()]],
        token_map={42: "b"},
    )
    worker = make_worker(runner)

    worker._decode_batch_step([req])

    events = drain_events(req)
    tok_event = events[0]
    assert isinstance(tok_event, TokenEvent)
    assert tok_event.is_first is False
    assert tok_event.index == 1


# ─── Group 4: _decode_batch_step — completion (EOS / max length) ─────────────


def test_batched_decode_eos_finishes_request() -> None:
    req = make_req("r0")
    req.status = RequestStatus.DECODING
    req.start_ns = 1
    req.enqueued_ns = 0
    req.num_prompt_tokens = 5
    req.all_logits = torch.randn(1, 5, VOCAB)
    req.past_key_values = DynamicCache()

    runner = FakeModelRunner(sample_tokens=[EOS])
    worker = make_worker(runner)

    worker._decode_batch_step([req])

    assert req.status == RequestStatus.DONE
    assert req.finished_reason == FinishReason.EOS
    events = drain_events(req)
    tok_event = events[0]
    assert isinstance(tok_event, TokenEvent)
    assert tok_event.is_last is True
    assert tok_event.token == ""


def test_batched_decode_eos_emits_done_event() -> None:
    req = make_req("r0")
    req.status = RequestStatus.DECODING
    req.start_ns = 1
    req.enqueued_ns = 0
    req.num_prompt_tokens = 5
    req.all_logits = torch.randn(1, 5, VOCAB)
    req.past_key_values = DynamicCache()

    runner = FakeModelRunner(sample_tokens=[EOS])
    worker = make_worker(runner)

    worker._decode_batch_step([req])

    events = drain_events(req)
    assert len(events) == 2
    assert isinstance(events[0], TokenEvent)
    assert isinstance(events[1], DoneEvent)
    assert events[1].ttft >= 0


def test_batched_decode_max_length_finishes_request() -> None:
    req = make_req("r0")
    req.status = RequestStatus.DECODING
    req.start_ns = 1
    req.enqueued_ns = 0
    req.num_prompt_tokens = 5
    req.all_logits = torch.randn(1, 5, VOCAB)
    req.past_key_values = DynamicCache()
    req.sampling_params = SamplingParams(max_new_tokens=1, temperature=1.0, top_p=1.0)

    runner = FakeModelRunner(
        sample_tokens=[42],
        token_map={42: "a"},
    )
    worker = make_worker(runner)

    worker._decode_batch_step([req])

    assert req.status == RequestStatus.DONE
    assert req.finished_reason == FinishReason.MAX_LENGTH
    events = drain_events(req)
    assert len(events) == 2
    assert isinstance(events[0], TokenEvent)
    assert events[0].is_last is True


def test_batched_decode_all_finish_no_batch_decode_called() -> None:
    r0, r1 = make_req("r0"), make_req("r1")
    for req in [r0, r1]:
        req.status = RequestStatus.DECODING
        req.start_ns = 1
        req.enqueued_ns = 0
        req.num_prompt_tokens = 5
        req.all_logits = torch.randn(1, 5, VOCAB)
        req.past_key_values = DynamicCache()

    runner = FakeModelRunner(sample_tokens=[EOS, EOS])
    worker = make_worker(runner)

    worker._decode_batch_step([r0, r1])

    assert r0.status == RequestStatus.DONE
    assert r1.status == RequestStatus.DONE


# ─── Group 5: _decode_batch_step — error handling ────────────────────────────


def test_batched_decode_none_logits_marks_failed() -> None:
    req = make_req("r0")
    req.status = RequestStatus.DECODING
    req.start_ns = 0
    req.enqueued_ns = 0
    req.all_logits = None
    req.past_key_values = DynamicCache()

    runner = FakeModelRunner(sample_tokens=[])
    worker = make_worker(runner)

    worker._decode_batch_step([req])

    assert req.status == RequestStatus.FAILED
    events = drain_events(req)
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)


def test_batched_decode_none_past_key_values_marks_failed() -> None:
    req = make_req("r0")
    req.status = RequestStatus.DECODING
    req.start_ns = 0
    req.enqueued_ns = 0
    req.all_logits = torch.randn(1, 5, VOCAB)
    req.past_key_values = None

    runner = FakeModelRunner(sample_tokens=[])
    worker = make_worker(runner)

    worker._decode_batch_step([req])

    assert req.status == RequestStatus.FAILED
    events = drain_events(req)
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)


def test_batched_decode_error_in_one_request_others_continue() -> None:
    r0 = make_req("r0")
    r0.status = RequestStatus.DECODING
    r0.start_ns = 0
    r0.enqueued_ns = 0
    r0.all_logits = None
    r0.past_key_values = DynamicCache()

    r1 = make_req("r1")
    r1.status = RequestStatus.DECODING
    r1.start_ns = 0
    r1.enqueued_ns = 0
    r1.all_logits = torch.randn(1, 5, VOCAB)
    r1.past_key_values = DynamicCache()

    runner = FakeModelRunner(
        sample_tokens=[42],
        decode_batches=[[_make_decode_output()]],
        token_map={42: "a"},
    )
    worker = make_worker(runner)

    worker._decode_batch_step([r0, r1])

    assert r0.status == RequestStatus.FAILED
    assert r1.status == RequestStatus.DECODING


def test_batched_decode_decode_batch_exception_marks_unfinished_failed() -> None:
    req = make_req("r0")
    req.status = RequestStatus.DECODING
    req.start_ns = 0
    req.enqueued_ns = 0
    req.all_logits = torch.randn(1, 5, VOCAB)
    req.past_key_values = DynamicCache()

    runner = FakeModelRunner(
        sample_tokens=[42],
        decode_batches=[],
        token_map={42: "a"},
    )
    runner.decode_batch = lambda tids, pvs: (_ for _ in ()).throw(
        RuntimeError("batch decode crash")
    )
    worker = make_worker(runner)

    worker._decode_batch_step([req])

    assert req.status == RequestStatus.FAILED
    events = drain_events(req)
    assert any(isinstance(e, ErrorEvent) for e in events)


def test_batched_decode_decode_batch_output_mismatch_marks_failed() -> None:
    req = make_req("r0")
    req.status = RequestStatus.DECODING
    req.start_ns = 0
    req.enqueued_ns = 0
    req.all_logits = torch.randn(1, 5, VOCAB)
    req.past_key_values = DynamicCache()

    runner = FakeModelRunner(
        sample_tokens=[42],
        decode_batches=[[]],
        token_map={42: "a"},
    )
    worker = make_worker(runner)

    worker._decode_batch_step([req])

    assert req.status == RequestStatus.FAILED
    events = drain_events(req)
    assert any(isinstance(e, ErrorEvent) for e in events)


# ─── Group 6: Integration — full prefill → decode cycle ──────────────────────


def test_full_prefill_decode_cycle() -> None:
    runner = FakeModelRunner(
        prefill_batches=[[_make_prefill_output(5)]],
        sample_tokens=[42, EOS],
        decode_batches=[[_make_decode_output()]],
        token_map={42: "hi"},
    )
    worker = make_worker(runner)
    req = make_req("r0")
    req.enqueued_ns = 0

    worker._prefill_batch([req])
    assert req.status == RequestStatus.DECODING
    assert req.num_prompt_tokens == 5

    worker._decode_batch_step([req])
    assert req.status == RequestStatus.DECODING
    assert req.output_tokens == ["hi"]

    worker._decode_batch_step([req])
    assert req.status == RequestStatus.DONE
    assert req.finished_reason == FinishReason.EOS

    events = drain_events(req)
    token_events = [e for e in events if isinstance(e, TokenEvent)]
    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert len(token_events) == 2
    assert len(done_events) == 1
    done = done_events[0]
    assert done.text == "hi"
    assert done.num_prompt_tokens == 5
    assert done.num_output_tokens == 1
    assert done.ttft >= 0


def test_full_cycle_multiple_requests() -> None:
    runner = FakeModelRunner(
        prefill_batches=[
            [_make_prefill_output(4), _make_prefill_output(6)],
        ],
        sample_tokens=[42, 43, EOS, EOS],
        decode_batches=[
            [_make_decode_output(), _make_decode_output()],
        ],
        token_map={42: "a", 43: "b"},
    )
    worker = make_worker(runner)
    r0, r1 = make_req("r0"), make_req("r1")
    r0.enqueued_ns = 0
    r1.enqueued_ns = 0

    worker._prefill_batch([r0, r1])
    assert r0.status == RequestStatus.DECODING
    assert r1.status == RequestStatus.DECODING

    worker._decode_batch_step([r0, r1])
    assert r0.status == RequestStatus.DECODING
    assert r1.status == RequestStatus.DECODING
    assert r0.output_tokens == ["a"]
    assert r1.output_tokens == ["b"]

    worker._decode_batch_step([r0, r1])
    assert r0.status == RequestStatus.DONE
    assert r1.status == RequestStatus.DONE

    for req in [r0, r1]:
        events = drain_events(req)
        done = [e for e in events if isinstance(e, DoneEvent)]
        assert len(done) == 1
        assert done[0].num_prompt_tokens in (4, 6)
