"""Tests for ScheduleInferenceEngine's prefill/decode execution path.

These use a fake backend whose model emits logits whose argmax is a fixed
function of the last input token (``next(t) = t + 1``). That makes the
generated token chain fully deterministic without a real model, so we can
assert the engine's position/``num_tokens`` handoff, event stream, and finish
conditions directly.
"""

from queue import Queue

import pytest
import torch

from server.executor.engine import (
    EngineCallbacks,
    ScheduleInferenceEngine,
)
from server.executor.scheduler import Scheduler
from server.executor.sinks import SharedQueueSink
from server.executor.types import (
    DoneEvent,
    ErrorEvent,
    GenerationRequestState,
    RequestStatus,
    Sequence,
    SequenceBatchTask,
    SequenceState,
    TokenEvent,
)
from server.model.block_manager import BlockManager
from server.model.inference_context import InferenceContext, inference_context
from server.model.sampling import SamplingParams
from tests.executor.worker_helpers import drain_events

VOCAB = 32
EOS = 31


class _FakeOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeModel:
    """Returns logits whose per-position argmax is ``input_token + step``.

    Records every call's (input_ids, position_ids) so tests can assert the
    exact tensors the engine feeds to the model.
    """

    def __init__(self, vocab: int, step: int = 1) -> None:
        self.vocab = vocab
        self.step = step
        self.calls: list[tuple[list[list[int]], list[list[int]]]] = []

    def __call__(self, input_ids, position_ids=None, use_cache=False):
        seq_len = input_ids.shape[1]
        logits = torch.full(
            (1, seq_len, self.vocab), -1.0, dtype=torch.float32, device=input_ids.device
        )
        for j in range(seq_len):
            tok = int(input_ids[0, j])
            logits[0, j, (tok + self.step) % self.vocab] = 1.0
        self.calls.append(
            (
                input_ids.cpu().tolist(),
                position_ids.cpu().tolist() if position_ids is not None else None,
            )
        )
        return _FakeOutput(logits)


class _FakeTokenizer:
    def __init__(self, eos_token_id: int) -> None:
        self.eos_token_id = eos_token_id

    def decode(self, token_ids, skip_special_tokens=True) -> str:
        return "".join(f"<{i}>" for i in token_ids)


class _FakeBackend:
    def __init__(self, prompt_tokens: list[int]) -> None:
        self.tokenizer = _FakeTokenizer(EOS)
        self.model = _FakeModel(VOCAB)
        self._prompt_tokens = list(prompt_tokens)
        self.device = "cpu"

    def tokenize(self, prompt: str) -> list[int]:
        return list(self._prompt_tokens)


def _make_engine(
    prompt_tokens: list[int],
) -> tuple[ScheduleInferenceEngine, Scheduler, BlockManager, _FakeBackend]:
    backend = _FakeBackend(prompt_tokens)
    block_manager = BlockManager(total_blocks=8, block_size=4)
    scheduler = Scheduler(
        block_manager=block_manager,
        max_num_sequences=4,
        max_num_tokens=64,
    )
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)
    return engine, scheduler, block_manager, backend


def _make_req(max_new_tokens: int) -> GenerationRequestState:
    return GenerationRequestState(
        request_id="req-1",
        sampling_params=SamplingParams(
            max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0
        ),
        prompt="hello",
        sink=SharedQueueSink(),
        enqueued_ns=0,
    )


class _StopWhenDone:
    """Stops the run loop once every tracked request reaches DONE."""

    def __init__(self, requests: list[GenerationRequestState], max_calls: int = 2000):
        self._requests = requests
        self._calls = 0
        self._max_calls = max_calls

    def should_stop(self) -> bool:
        self._calls += 1
        if self._calls > self._max_calls:
            return True
        return all(r.status == RequestStatus.DONE for r in self._requests)

    def wait_idle(self, _timeout: float) -> bool:
        return False


class _StopOnPreemption:
    """Stops the run loop as soon as the scheduler records a preemption,
    leaving the victim sitting in ``waiting`` with state PREEMPTED (not yet
    resumed) -- used to test cleanup paths mid-preemption."""

    def __init__(self, scheduler: Scheduler, max_calls: int = 2000):
        self._scheduler = scheduler
        self._calls = 0
        self._max_calls = max_calls

    def should_stop(self) -> bool:
        self._calls += 1
        if self._calls > self._max_calls:
            return True
        return self._scheduler.preemption_count > 0

    def wait_idle(self, _timeout: float) -> bool:
        return False


def _callbacks(recorder: dict) -> EngineCallbacks:
    def cancel_request(req: GenerationRequestState, message: str) -> None:
        recorder.setdefault("cancelled", []).append((req.request_id, message))

    def handle_fatal_error(error: Exception, extra):
        recorder["fatal"] = error

    return EngineCallbacks(
        cancel_request=cancel_request, handle_fatal_error=handle_fatal_error
    )


def _run_to_completion(engine, request) -> _FakeBackend:
    inbound: Queue = Queue()
    inbound.put(request)
    control = _StopWhenDone([request])
    recorder: dict = {}
    engine.run(
        inbound=inbound,
        control=control,
        callbacks=_callbacks(recorder),
    )
    assert "fatal" not in recorder, f"engine hit fatal error: {recorder.get('fatal')}"
    return engine._backend


def test_prefill_then_decode_finishes_at_max_length() -> None:
    engine, scheduler, block_manager, backend = _make_engine(prompt_tokens=[3, 4, 5])
    req = _make_req(max_new_tokens=4)

    _run_to_completion(engine, req)

    # Event stream: 4 explicit tokens then a separate DoneEvent.
    events = drain_events(req)
    token_events = [e for e in events if isinstance(e, TokenEvent)]
    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(token_events) == 4
    assert [t.token for t in token_events] == ["<6>", "<7>", "<8>", "<9>"]
    assert [event.index for event in token_events] == [0, 1, 2, 3]
    assert len(done) == 1
    assert done[0].num_output_tokens == 4
    assert done[0].num_prompt_tokens == 3
    assert req.status == RequestStatus.DONE


def test_decode_position_equals_num_tokens_and_increments() -> None:
    """The load-bearing invariant: decode position_id == num_tokens (pre-store),
    advancing by exactly one per decode step; input is the last generated token."""
    engine, *_ = _make_engine(prompt_tokens=[3, 4, 5])
    req = _make_req(max_new_tokens=4)
    backend = _run_to_completion(engine, req)

    prefill_calls = [c for c in backend.model.calls if len(c[0][0]) > 1]
    decode_calls = [c for c in backend.model.calls if len(c[0][0]) == 1]

    # One prefill over the flattened prompt [3,4,5] with positions [0,1,2].
    assert len(prefill_calls) == 1
    assert prefill_calls[0][0] == [[3, 4, 5]]
    assert prefill_calls[0][1] == [[0, 1, 2]]

    # Three decode steps: input is g1,g2,g3 == [6],[7],[8]; positions 3,4,5.
    assert [c[0] for c in decode_calls] == [[[6]], [[7]], [[8]]]
    assert [c[1] for c in decode_calls] == [[[3]], [[4]], [[5]]]


def test_finished_sequence_is_reaped_and_blocks_freed() -> None:
    engine, scheduler, block_manager, backend = _make_engine(prompt_tokens=[3, 4, 5])
    req = _make_req(max_new_tokens=4)
    _run_to_completion(engine, req)

    # Scheduler dropped the finished sequence; all blocks returned to the pool.
    assert scheduler.running == []
    assert len(scheduler.waiting) == 0
    assert sorted(block_manager.free_blocks) == list(range(block_manager.total_blocks))
    # Engine-side tracking cleared too.
    assert engine._all_requests == {}
    assert engine._seq_to_request == {}


def test_stops_on_eos_during_decode() -> None:
    # Prompt [28, 29]: prefill last token 29 -> g1=30 (not EOS); decode input
    # 30 -> g2=31 == EOS, so generation stops on the second token, mid-decode.
    engine, *_ = _make_engine(prompt_tokens=[28, 29])
    req = _make_req(max_new_tokens=10)
    backend = _run_to_completion(engine, req)

    events = drain_events(req)
    token_events = [e for e in events if isinstance(e, TokenEvent)]
    assert [t.token for t in token_events] == ["<30>"]
    assert req.num_output_tokens == 1  # EOS token is not counted as output
    assert req.status == RequestStatus.DONE
    assert req.finished_reason is not None

    # Exactly one decode step fed the non-EOS token g1=30 at position P=2.
    decode_calls = [c for c in backend.model.calls if len(c[0][0]) == 1]
    assert decode_calls == [([[30]], [[2]])]


def test_cancel_inflight_clears_state_and_invokes_callback() -> None:
    engine, scheduler, block_manager, backend = _make_engine(prompt_tokens=[3, 4, 5])
    req = _make_req(max_new_tokens=4)
    inbound: Queue = Queue()
    inbound.put(req)
    engine._drain_inbound(inbound)

    # Request is now tracked and the sequence is waiting in the scheduler.
    assert req.request_id in engine._all_requests
    assert len(scheduler.waiting) == 1

    cancelled: list = []

    def cancel_request(r: GenerationRequestState, message: str) -> None:
        cancelled.append((r.request_id, message))
        r.status = RequestStatus.FAILED

    engine.cancel_inflight("boom", cancel_request)

    assert cancelled == [(req.request_id, "boom")]
    assert engine._all_requests == {}
    assert engine._seq_to_request == {}
    assert len(scheduler.waiting) == 0
    assert scheduler.running == []
    assert sorted(block_manager.free_blocks) == list(range(block_manager.total_blocks))


def test_reap_cancelled_marks_admitted_sequence_finished_and_frees_blocks() -> None:
    # A tracked (admitted, running) request whose cancelled flag is set must be
    # marked finished and dropped from engine tracking; the scheduler's existing
    # reap path then frees its blocks on the next schedule().
    engine, scheduler, block_manager, backend = _make_engine(prompt_tokens=[3, 4, 5])
    req = _make_req(max_new_tokens=4)
    inbound: Queue = Queue()
    inbound.put(req)
    engine._drain_inbound(inbound)
    scheduler.schedule()  # prefill: sequence is now RUNNING and holds blocks

    seq = scheduler.running[0]
    assert block_manager.allocated_blocks  # blocks are held before cancellation

    req.cancelled.set()
    engine._reap_cancelled()

    assert seq.finished is True
    assert req.status == RequestStatus.CANCELLED
    assert engine._all_requests == {}
    assert engine._seq_to_request == {}

    scheduler.schedule()  # reaps the finished sequence, returning its blocks
    assert scheduler.running == []
    assert sorted(block_manager.free_blocks) == list(range(block_manager.total_blocks))


def test_prepare_decode_builds_one_token_per_sequence() -> None:
    """Unit-level check of _prepare_decode's tensor shapes and context."""

    engine, *_ = _make_engine(prompt_tokens=[1, 2, 3])
    seqs = [
        Sequence(
            sequence_id="a",
            prompt_token_ids=[10, 11],
            generated_token_ids=[21],
            num_prompt_tokens=2,
            num_tokens=2,
            max_new_tokens=4,
            block_table=[0],
            state=SequenceState.RUNNING,
        ),
        Sequence(
            sequence_id="b",
            prompt_token_ids=[20, 21, 22],
            generated_token_ids=[33],
            num_prompt_tokens=3,
            num_tokens=3,
            max_new_tokens=4,
            block_table=[1, 2],
            state=SequenceState.RUNNING,
        ),
    ]
    input_ids, position_ids, ctx = engine._prepare_decode(seqs)
    assert input_ids.shape == (1, 2)  # (1, B): one token per sequence
    assert input_ids.tolist() == [[21, 33]]
    assert position_ids.shape == (1, 2)
    assert position_ids.tolist() == [[2, 3]]  # == num_tokens of each seq
    assert ctx.mode == "decode"
    assert [s["block_table"] for s in ctx.sequences] == [[0], [1, 2]]


def test_inference_context_roundtrip_unused_for_decode_num_tokens() -> None:
    """Sanity: the decode context only needs block_table (num_tokens unused)."""
    ctx = InferenceContext(mode="decode", sequences=[{"block_table": [0, 1]}])
    with inference_context(ctx):
        from server.model.inference_context import get_inference_context

        assert get_inference_context().mode == "decode"
        assert get_inference_context().sequences[0]["block_table"] == [0, 1]


def test_oversized_prompt_is_failed_not_requeued() -> None:
    # Cache capacity = 8 blocks * 4 tokens = 32. A 40-token prompt needs
    # ceil(40/4) = 10 blocks and can never fit, so it must be failed once
    # rather than re-queued (and re-tokenized) on every loop forever.
    engine, scheduler, block_manager, backend = _make_engine(
        prompt_tokens=list(range(40))
    )
    req = _make_req(max_new_tokens=4)
    inbound: Queue = Queue()
    inbound.put(req)

    engine._drain_inbound(inbound)

    assert req.status == RequestStatus.FAILED
    assert len(scheduler.waiting) == 0
    assert engine._all_requests == {}
    assert inbound.empty()  # not re-queued back for retry
    events = drain_events(req)
    assert any(isinstance(e, ErrorEvent) for e in events)


def _counting_backend(prompt_tokens: list[int]) -> _FakeBackend:
    backend = _FakeBackend(prompt_tokens)
    backend.tokenize_calls = 0
    original = backend.tokenize

    def counting(prompt: str) -> list[int]:
        backend.tokenize_calls += 1
        return original(prompt)

    backend.tokenize = counting
    return backend


def test_drain_stops_at_scheduler_capacity_and_leaves_backlog_inbound() -> None:
    backend = _counting_backend(prompt_tokens=[3, 4, 5])
    block_manager = BlockManager(total_blocks=32, block_size=4)
    scheduler = Scheduler(
        block_manager=block_manager,
        max_num_sequences=2,
        max_num_tokens=64,
    )
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)

    inbound: Queue = Queue()
    requests = [_make_req(max_new_tokens=4) for _ in range(4)]
    for i, req in enumerate(requests):
        req.request_id = f"req-{i}"
        inbound.put(req)

    engine._drain_inbound(inbound)

    assert backend.tokenize_calls == 2
    assert len(scheduler.waiting) == 2
    assert not scheduler.running
    assert inbound.qsize() == 2
    assert len(engine._all_requests) == 2


def test_over_capacity_load_never_creates_prefilled_starved_tail() -> None:
    backend = _FakeBackend(prompt_tokens=[3, 4, 5])
    scheduler = Scheduler(
        block_manager=BlockManager(total_blocks=1024, block_size=4),
        max_num_sequences=16,
        max_num_tokens=64,
    )
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)
    inbound: Queue = Queue()
    for i in range(64):
        req = _make_req(max_new_tokens=4)
        req.request_id = f"req-{i}"
        inbound.put(req)

    engine._drain_inbound(inbound)

    assert len(scheduler.waiting) == 16
    assert not scheduler.running
    assert inbound.qsize() == 48
    assert all(not seq.block_table for seq in scheduler.waiting)

    prefill = scheduler.schedule()
    assert prefill is not None
    assert prefill.kind is SequenceBatchTask.PREFILL
    assert len(prefill.sequences) == 16
    assert len(scheduler.running) == 16
    assert not scheduler.waiting

    decode = scheduler.schedule()
    assert decode is not None
    assert decode.kind is SequenceBatchTask.DECODE
    assert decode.sequences == scheduler.running


def test_drain_never_reenters_inbound_and_returns() -> None:
    # The engine consumes only available logical slots and never puts a request
    # back into the shared queue.
    backend = _FakeBackend(prompt_tokens=[3, 4, 5])
    scheduler = Scheduler(
        block_manager=BlockManager(total_blocks=32, block_size=4),
        max_num_sequences=1,
        max_num_tokens=64,
    )
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)

    inbound: Queue = Queue(maxsize=2)
    inbound.put_nowait(_make_req(max_new_tokens=4))
    inbound.put_nowait(_make_req(max_new_tokens=4))

    engine._drain_inbound(inbound)

    assert len(scheduler.waiting) == 1
    assert inbound.qsize() == 1


def test_drain_drops_cancelled_inbound_without_emitting_or_consuming_slot() -> None:
    backend = _FakeBackend(prompt_tokens=[3, 4, 5])
    scheduler = Scheduler(
        block_manager=BlockManager(total_blocks=32, block_size=4),
        max_num_sequences=1,
        max_num_tokens=64,
    )
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)

    cancelled = _make_req(max_new_tokens=4)
    cancelled.cancelled.set()
    feasible = _make_req(max_new_tokens=4)
    inbound: Queue = Queue()
    inbound.put(cancelled)
    inbound.put(feasible)
    engine._drain_inbound(inbound)

    assert cancelled.status is RequestStatus.CANCELLED
    assert cancelled.sink.queue.empty()
    assert len(scheduler.waiting) == 1
    assert engine._all_requests[feasible.request_id].request is feasible


def test_cancelled_slot_is_reaped_before_next_inbound_drain() -> None:
    backend = _FakeBackend(prompt_tokens=[3, 4, 5])
    scheduler = Scheduler(
        block_manager=BlockManager(total_blocks=32, block_size=4),
        max_num_sequences=1,
        max_num_tokens=64,
    )
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)
    first = _make_req(max_new_tokens=4)
    first.request_id = "first"
    replacement = _make_req(max_new_tokens=4)
    replacement.request_id = "replacement"
    inbound: Queue = Queue()
    inbound.put(first)
    engine._drain_inbound(inbound)
    assert not scheduler.has_sequence_capacity()

    first.cancelled.set()
    inbound.put(replacement)
    # This is the ordering used at the head of each engine-loop iteration.
    engine._reap_cancelled()
    scheduler.reap_finished()
    engine._drain_inbound(inbound)

    assert first.status is RequestStatus.CANCELLED
    assert [engine._seq_to_request[s.sequence_id] for s in scheduler.waiting] == [
        replacement
    ]
    assert inbound.empty()


def test_infeasible_requests_do_not_consume_sequence_slots() -> None:
    backend = _FakeBackend(prompt_tokens=[])
    backend.tokenize = lambda prompt: list(range(len(prompt)))
    scheduler = Scheduler(
        block_manager=BlockManager(total_blocks=2, block_size=4),
        max_num_sequences=2,
        max_num_tokens=8,
    )
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)

    infeasible = [_make_req(max_new_tokens=4) for _ in range(3)]
    for i, req in enumerate(infeasible):
        req.request_id = f"infeasible-{i}"
        req.prompt = "0123456789"
    feasible = [_make_req(max_new_tokens=2) for _ in range(2)]
    for i, req in enumerate(feasible):
        req.request_id = f"feasible-{i}"
        req.prompt = "01"

    inbound: Queue = Queue()
    for req in infeasible + feasible:
        inbound.put(req)
    engine._drain_inbound(inbound)

    assert all(req.status is RequestStatus.FAILED for req in infeasible)
    assert len(scheduler.waiting) == 2
    assert set(engine._all_requests) == {req.request_id for req in feasible}
    assert inbound.empty()


def test_cancel_inflight_fails_scheduler_waiting_requests() -> None:
    engine, scheduler, *_ = _make_engine(prompt_tokens=[3, 4, 5])

    req = _make_req(max_new_tokens=4)
    inbound: Queue = Queue()
    inbound.put(req)
    engine._drain_inbound(inbound)
    assert len(scheduler.waiting) == 1

    cancelled: list = []

    def cancel_request(r: GenerationRequestState, message: str) -> None:
        cancelled.append((r.request_id, message))

    engine.cancel_inflight("boom", cancel_request)

    assert cancelled == [(req.request_id, "boom")]
    assert not scheduler.waiting
    assert engine._all_requests == {}


def test_post_decode_isolates_sampling_failure() -> None:
    """A sampling failure for one sequence fails only it; the batch continues."""

    engine, *_ = _make_engine(prompt_tokens=[1, 2, 3])

    req_a = _make_req(max_new_tokens=5)
    req_a.request_id = "a"
    req_b = _make_req(max_new_tokens=5)
    req_b.request_id = "b"

    seq_a = Sequence(
        sequence_id="sa",
        prompt_token_ids=[1, 2, 3],
        generated_token_ids=[4],
        num_prompt_tokens=3,
        num_tokens=3,
        max_new_tokens=5,
        block_table=[0],
        state=SequenceState.RUNNING,
    )
    seq_b = Sequence(
        sequence_id="sb",
        prompt_token_ids=[1, 2, 3],
        generated_token_ids=[4],
        num_prompt_tokens=3,
        num_tokens=3,
        max_new_tokens=5,
        block_table=[1],
        state=SequenceState.RUNNING,
    )
    engine._seq_to_request[seq_a.sequence_id] = req_a
    engine._seq_to_request[seq_b.sequence_id] = req_b

    # Tokenizer that blows up when decoding token 0 (the "bad" token).
    class _BadTokenizer(_FakeTokenizer):
        def decode(self, token_ids, skip_special_tokens=True) -> str:
            if 0 in token_ids:
                raise RuntimeError("boom")
            return super().decode(token_ids, skip_special_tokens=True)

    engine._backend.tokenizer = _BadTokenizer(EOS)

    # out.logits [1, 2, vocab]: seq a argmax -> 0 (decode raises), seq b -> 7 (ok).
    logits = torch.full((1, 2, VOCAB), -1.0, dtype=torch.float32)
    logits[0, 0, 0] = 1.0
    logits[0, 1, 7] = 1.0
    out = _FakeOutput(logits)

    engine._post_decode(out, [seq_a, seq_b])

    assert req_a.status == RequestStatus.FAILED
    assert seq_a.finished is True
    # seq b continued: a token was emitted and it is still decoding.
    assert req_b.status != RequestStatus.FAILED
    assert req_b.num_output_tokens == 1
    assert seq_b.finished is False
    assert any(isinstance(e, ErrorEvent) for e in drain_events(req_a))


def test_post_decode_mixed_greedy_sampled_topk_batch() -> None:
    """A batch mixing a greedy row, a sampled top-k=1 row, and a sampled
    top-k=2 row all decode correctly in one batched sample_tokens call."""
    engine, *_ = _make_engine(prompt_tokens=[1, 2, 3])

    req_greedy = GenerationRequestState(
        request_id="greedy",
        sampling_params=SamplingParams(max_new_tokens=5, temperature=0.0, top_p=1.0),
        prompt="hello",
        sink=SharedQueueSink(),
        enqueued_ns=0,
    )
    # top_k=1 collapses the nucleus to the single argmax, so its sampled token
    # is deterministic even though temperature > 0.
    req_k1 = GenerationRequestState(
        request_id="k1",
        sampling_params=SamplingParams(
            max_new_tokens=5, temperature=1.0, top_p=1.0, top_k=1, seed=123
        ),
        prompt="hello",
        sink=SharedQueueSink(),
        enqueued_ns=0,
    )
    # top_k=2 leaves two survivors; the picked token must be one of them.
    req_k2 = GenerationRequestState(
        request_id="k2",
        sampling_params=SamplingParams(
            max_new_tokens=5, temperature=1.0, top_p=1.0, top_k=2, seed=7
        ),
        prompt="hello",
        sink=SharedQueueSink(),
        enqueued_ns=0,
    )

    reqs = [req_greedy, req_k1, req_k2]
    seqs = []
    for i, req in enumerate(reqs):
        seq = Sequence(
            sequence_id=f"s{i}",
            prompt_token_ids=[1, 2, 3],
            generated_token_ids=[4],
            num_prompt_tokens=3,
            num_tokens=3,
            max_new_tokens=5,
            block_table=[i],
            state=SequenceState.RUNNING,
        )
        engine._seq_to_request[seq.sequence_id] = req
        seqs.append(seq)

    # Row 0 argmax -> 5; row 1 argmax -> 9; row 2 top-2 -> {12, 13}.
    logits = torch.full((1, 3, VOCAB), -10.0, dtype=torch.float32)
    logits[0, 0, 5] = 1.0
    logits[0, 1, 9] = 1.0
    logits[0, 2, 12] = 2.0
    logits[0, 2, 13] = 1.5
    out = _FakeOutput(logits)

    engine._post_decode(out, seqs)

    assert seqs[0].generated_token_ids == [4, 5]  # greedy -> exact argmax
    assert seqs[1].generated_token_ids == [4, 9]  # top_k=1 -> forced argmax
    assert seqs[2].generated_token_ids[-1] in {12, 13}  # top_k=2 -> allowed set

    for req in reqs:
        assert req.status != RequestStatus.FAILED
        assert req.num_output_tokens == 1
    for seq in seqs:
        assert seq.num_tokens == 4  # advanced by one
        assert seq.finished is False


def _make_resumable_request(
    request_id: str, max_new_tokens: int, start_ns: int, num_prompt_tokens: int
) -> GenerationRequestState:
    """A request that has already been through one prefill + some decoding,
    as if it were about to be resumed after a preemption."""
    req = GenerationRequestState(
        request_id=request_id,
        sampling_params=SamplingParams(
            max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0
        ),
        prompt="hello",
        sink=SharedQueueSink(),
        enqueued_ns=0,
        start_ns=start_ns,
        num_prompt_tokens=num_prompt_tokens,
    )
    req.status = RequestStatus.DECODING
    return req


def test_prepare_prefill_feeds_prompt_plus_generated_for_resumed_sequence() -> None:
    """_prepare_prefill must distinguish fresh vs. resumed via the
    scheduler-provided resumed_sequence_ids."""

    engine, *_ = _make_engine(prompt_tokens=[1, 2, 3])
    fresh = Sequence(
        sequence_id="fresh",
        prompt_token_ids=[10, 11],
        generated_token_ids=[],
        num_prompt_tokens=2,
        num_tokens=2,
        max_new_tokens=4,
        block_table=[0],
        state=SequenceState.RUNNING,
    )
    resumed = Sequence(
        sequence_id="resumed",
        prompt_token_ids=[20, 21, 22],
        generated_token_ids=[33, 34],
        num_prompt_tokens=3,
        num_tokens=5,
        max_new_tokens=4,
        block_table=[1, 2],
        state=SequenceState.RUNNING,
    )

    input_ids, position_ids, ctx = engine._prepare_prefill(
        [fresh, resumed], resumed_sequence_ids=frozenset({"resumed"})
    )

    assert input_ids.tolist() == [[10, 11, 20, 21, 22, 33, 34]]
    assert position_ids.tolist() == [[0, 1, 0, 1, 2, 3, 4]]
    assert [s["num_tokens"] for s in ctx.sequences] == [2, 5]
    assert [s["block_table"] for s in ctx.sequences] == [[0], [1, 2]]


def test_post_prefill_mixed_fresh_and_resumed_batch_slices_correct_logits() -> None:
    """The flattened logits offset must stride by each sequence's *fed*
    length (P fresh, P+G resumed), not uniformly by num_prompt_tokens --
    otherwise a mixed batch corrupts every sequence after a resumed one."""

    engine, *_ = _make_engine(prompt_tokens=[1, 2, 3])

    fresh = Sequence(
        sequence_id="fresh",
        prompt_token_ids=[10, 11],
        generated_token_ids=[],
        num_prompt_tokens=2,
        num_tokens=2,
        max_new_tokens=4,
        block_table=[0],
        state=SequenceState.RUNNING,
    )
    resumed = Sequence(
        sequence_id="resumed",
        prompt_token_ids=[20, 21, 22],
        generated_token_ids=[33, 34],
        num_prompt_tokens=3,
        num_tokens=5,
        max_new_tokens=4,
        block_table=[1, 2],
        state=SequenceState.RUNNING,
    )
    req_fresh = _make_req(max_new_tokens=4)
    req_fresh.request_id = "fresh"
    req_resumed = _make_resumable_request(
        "resumed", max_new_tokens=4, start_ns=500, num_prompt_tokens=3
    )
    engine._seq_to_request["fresh"] = req_fresh
    engine._seq_to_request["resumed"] = req_resumed

    # Fed lengths: fresh=2, resumed=3+2=5 -> flattened length 7. The
    # "correct" last-token row for fresh is index 1 (offset 0 + 2 - 1); for
    # resumed it's index 6 (offset 2 + 5 - 1), NOT index 4 (the pre-fix bug:
    # offset would advance by num_prompt_tokens=3 instead of fed_len=5).
    logits = torch.full((1, 7, VOCAB), -1.0, dtype=torch.float32)
    logits[0, 1, 7] = 1.0  # fresh's correct row -> argmax 7
    logits[0, 6, 15] = 1.0  # resumed's correct row -> argmax 15
    logits[0, 4, 9] = 1.0  # the WRONG row a stride-by-prompt-len bug would read
    out = _FakeOutput(logits)

    engine._post_prefill(
        out,  # type: ignore[arg-type]
        [fresh, resumed],
        start_ns=1000,
        resumed_sequence_ids=frozenset({"resumed"}),
    )

    assert fresh.generated_token_ids == [7]
    assert resumed.generated_token_ids == [33, 34, 15]


def test_resume_prefill_does_not_reset_start_ns_or_num_prompt_tokens() -> None:
    """A resume is a second prefill for an already-admitted request: it must
    not overwrite start_ns (would corrupt queue_wait_ms/ttft_ms/total_ms) or
    otherwise disturb num_prompt_tokens."""

    engine, *_ = _make_engine(prompt_tokens=[1, 2, 3])
    resumed = Sequence(
        sequence_id="resumed",
        prompt_token_ids=[20, 21, 22],
        generated_token_ids=[33],
        num_prompt_tokens=3,
        num_tokens=4,
        max_new_tokens=4,
        block_table=[1],
        state=SequenceState.RUNNING,
    )
    req = _make_resumable_request(
        "resumed", max_new_tokens=4, start_ns=500, num_prompt_tokens=3
    )
    engine._seq_to_request["resumed"] = req

    logits = torch.full((1, 4, VOCAB), -1.0, dtype=torch.float32)
    logits[0, 3, 9] = 1.0  # fed_len = 3 + 1 = 4 -> last row index 3
    out = _FakeOutput(logits)

    # start_ns passed in as "now" (this prefill's start), distinct from the
    # request's original start_ns set at first admission.
    engine._post_prefill(
        out,
        [resumed],
        start_ns=999_999,
        resumed_sequence_ids=frozenset({"resumed"}),  # type: ignore[arg-type]
    )

    assert req.start_ns == 500  # unchanged
    assert req.num_prompt_tokens == 3
    assert resumed.generated_token_ids == [33, 9]


def test_forced_preemption_matches_uninterrupted_solo_run() -> None:
    """End-to-end: two concurrent requests sharing a tiny block pool force a
    real preemption (via the scheduler's preempt-youngest policy) and
    resume-by-recompute (via the engine). The preempted request's output must
    be byte-identical to running the same prompt alone with no contention."""
    prompt_tokens = [3, 4]
    max_new_tokens = 4

    # Baseline: same prompt, uninterrupted, plenty of blocks.
    solo_engine, *_ = _make_engine(prompt_tokens=prompt_tokens)
    solo_req = _make_req(max_new_tokens=max_new_tokens)
    solo_req.request_id = "solo"
    _run_to_completion(solo_engine, solo_req)
    solo_tokens = [t.token for t in drain_events(solo_req) if isinstance(t, TokenEvent)]

    # Contended run: block pool sized so decoding both concurrently forces the
    # scheduler to preempt the younger request at least once.
    backend = _FakeBackend(prompt_tokens)
    block_manager = BlockManager(total_blocks=6, block_size=1)
    scheduler = Scheduler(
        block_manager=block_manager,
        max_num_sequences=4,
        max_num_tokens=1024,
    )
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)

    req_a = _make_req(max_new_tokens=max_new_tokens)
    req_a.request_id = "a"
    req_b = _make_req(max_new_tokens=max_new_tokens)
    req_b.request_id = "b"

    inbound: Queue = Queue()
    inbound.put(req_a)
    inbound.put(req_b)
    control = _StopWhenDone([req_a, req_b])
    recorder: dict = {}
    engine.run(inbound=inbound, control=control, callbacks=_callbacks(recorder))

    assert "fatal" not in recorder, f"engine hit fatal error: {recorder.get('fatal')}"
    assert scheduler.preemption_count > 0  # the scenario actually exercised resume

    for req in (req_a, req_b):
        assert req.status == RequestStatus.DONE
        tokens = [t.token for t in drain_events(req) if isinstance(t, TokenEvent)]
        assert tokens == solo_tokens

    # Blocks fully reclaimed, no tracking leaks (mirrors item D's concerns).
    assert scheduler.running == []
    assert len(scheduler.waiting) == 0
    assert sorted(block_manager.free_blocks) == list(range(block_manager.total_blocks))
    assert engine._all_requests == {}
    assert engine._seq_to_request == {}


def test_preemption_does_not_retokenize_resumed_request() -> None:
    """A preempted request must resume via scheduler.waiting, not
    through the inbound admission/_make_sequence path. Across a real
    preemption, each request is tokenized exactly once."""
    prompt_tokens = [3, 4]
    max_new_tokens = 4

    backend = _counting_backend(prompt_tokens)
    block_manager = BlockManager(total_blocks=6, block_size=1)
    scheduler = Scheduler(
        block_manager=block_manager,
        max_num_sequences=4,
        max_num_tokens=1024,
    )
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)

    req_a = _make_req(max_new_tokens=max_new_tokens)
    req_a.request_id = "a"
    req_b = _make_req(max_new_tokens=max_new_tokens)
    req_b.request_id = "b"

    inbound: Queue = Queue()
    inbound.put(req_a)
    inbound.put(req_b)
    control = _StopWhenDone([req_a, req_b])
    recorder: dict = {}
    engine.run(inbound=inbound, control=control, callbacks=_callbacks(recorder))

    assert "fatal" not in recorder, f"engine hit fatal error: {recorder.get('fatal')}"
    assert scheduler.preemption_count > 0  # a real preemption actually happened
    # Exactly once per request: the resumed request was NOT re-tokenized.
    assert backend.tokenize_calls == 2
    # Make sure both requests finished successfully and produced the same output.
    assert req_a.status == RequestStatus.DONE
    assert req_b.status == RequestStatus.DONE


def test_shutdown_mid_preemption_cancels_and_frees_preempted_request() -> None:
    """A PREEMPTED sequence sitting in scheduler.waiting must still be
    cancelled and untracked on graceful shutdown."""
    prompt_tokens = [3, 4]
    max_new_tokens = 4

    backend = _FakeBackend(prompt_tokens)
    block_manager = BlockManager(total_blocks=6, block_size=1)
    scheduler = Scheduler(
        block_manager=block_manager,
        max_num_sequences=4,
        max_num_tokens=1024,
    )
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)  # type: ignore[arg-type]

    req_a = _make_req(max_new_tokens=max_new_tokens)
    req_a.request_id = "a"
    req_b = _make_req(max_new_tokens=max_new_tokens)
    req_b.request_id = "b"

    inbound: Queue = Queue()
    inbound.put(req_a)
    inbound.put(req_b)
    control = _StopOnPreemption(scheduler)
    recorder: dict = {}
    engine.run(inbound=inbound, control=control, callbacks=_callbacks(recorder))  # type: ignore[arg-type]

    assert "fatal" not in recorder, f"engine hit fatal error: {recorder.get('fatal')}"
    assert scheduler.preemption_count > 0

    cancelled_ids = {req_id for req_id, _msg in recorder.get("cancelled", [])}
    assert cancelled_ids == {"a", "b"}

    assert scheduler.running == []
    assert len(scheduler.waiting) == 0
    assert sorted(block_manager.free_blocks) == list(range(block_manager.total_blocks))
    assert engine._all_requests == {}
    assert engine._seq_to_request == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
