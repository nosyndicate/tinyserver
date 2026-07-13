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
from server.executor.types import (
    DoneEvent,
    ErrorEvent,
    GenerationRequestState,
    RequestStatus,
    Sequence,
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
        max_waiting=4,
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

    # Event stream: 4 tokens (last is_last) then a DoneEvent.
    events = drain_events(req)
    token_events = [e for e in events if isinstance(e, TokenEvent)]
    done = [e for e in events if isinstance(e, DoneEvent)]
    assert len(token_events) == 4
    assert [t.token for t in token_events] == ["<6>", "<7>", "<8>", "<9>"]
    assert token_events[-1].is_last is True
    assert token_events[0].is_first is True
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
    assert [t.token for t in token_events] == ["<30>", ""]
    assert token_events[0].is_first is True and token_events[0].is_last is False
    assert token_events[1].is_last is True  # EOS token
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


class _RejectAllScheduler:
    """Scheduler stub that never admits a sequence.

    Lets us exercise `_drain_inbound`'s deferral path without a real block
    manager. `can_add_new_sequence` always False (nothing is admittable) and
    `admission_headroom` is a settable budget (default 8, so small tests can
    pull everything from inbound). The block manager stub accepts every
    prompt as feasible.
    """

    class _BlockManager:
        def can_ever_allocate(self, seq) -> bool:
            return True

    def __init__(self) -> None:
        self.block_manager = self._BlockManager()
        self.added: list = []
        self.headroom = 8

    def can_add_new_sequence(self, seq) -> bool:
        return False

    def waiting_queue_is_full(self) -> bool:
        return False

    def admission_headroom(self) -> int:
        return self.headroom

    def add(self, seq) -> None:
        self.added.append(seq)


def _counting_backend(prompt_tokens: list[int]) -> _FakeBackend:
    backend = _FakeBackend(prompt_tokens)
    backend.tokenize_calls = 0
    original = backend.tokenize

    def counting(prompt: str) -> list[int]:
        backend.tokenize_calls += 1
        return original(prompt)

    backend.tokenize = counting
    return backend


def test_retry_does_not_retokenize_and_holds_in_pending() -> None:
    # A request that can never be admitted is tokenized exactly once and then
    # lives in the engine's private pending buffer across many drain calls.
    backend = _counting_backend(prompt_tokens=[3, 4, 5])
    scheduler = _RejectAllScheduler()
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)

    inbound: Queue = Queue()
    inbound.put(_make_req(max_new_tokens=4))

    for _ in range(5):
        engine._drain_inbound(inbound)

    assert backend.tokenize_calls == 1
    assert len(engine._pending) == 1
    assert inbound.empty()  # never re-queued into the shared inbound queue
    assert scheduler.added == []  # never admitted


def test_drain_never_reenters_inbound_and_returns() -> None:
    # Regression for the deadlock: even with a bounded inbound queue kept full
    # by a concurrent producer, the engine must not block on a put-back. It
    # defers into its private buffer and returns promptly.
    backend = _FakeBackend(prompt_tokens=[3, 4, 5])
    scheduler = _RejectAllScheduler()
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)

    inbound: Queue = Queue(maxsize=2)
    inbound.put_nowait(_make_req(max_new_tokens=4))
    inbound.put_nowait(_make_req(max_new_tokens=4))

    engine._drain_inbound(inbound)

    # Both drained into the private buffer; the engine put nothing back, so the
    # (still-full-capacity) inbound is now empty and a producer could refill it.
    assert len(engine._pending) == 2
    assert inbound.empty()
    inbound.put_nowait(_make_req(max_new_tokens=4))  # would raise Full if not drained
    assert inbound.qsize() == 1


def test_pending_preserves_fifo_across_drains() -> None:
    # A big request A submitted before a small request B must be admitted first
    # once capacity frees, even though B arrives while A is still deferred.
    backend = _FakeBackend(prompt_tokens=[3, 4, 5])

    class _GatedScheduler(_RejectAllScheduler):
        admit = False

        def can_add_new_sequence(self, seq) -> bool:
            return self.admit

    scheduler = _GatedScheduler()
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)

    req_a = _make_req(max_new_tokens=4)
    req_a.request_id = "A"
    req_b = _make_req(max_new_tokens=4)
    req_b.request_id = "B"

    inbound: Queue = Queue()
    inbound.put(req_a)
    engine._drain_inbound(inbound)  # A deferred
    inbound.put(req_b)
    engine._drain_inbound(inbound)  # B deferred behind A
    assert [r.request_id for r, _ in engine._pending] == ["A", "B"]

    scheduler.admit = True
    engine._drain_inbound(inbound)

    assert [seq for seq in scheduler.added]  # something admitted
    assert engine._seq_to_request[scheduler.added[0].sequence_id].request_id == "A"
    assert scheduler.added[1] is engine._all_requests["B"].sequence


def test_new_small_request_cannot_overtake_deferred_large() -> None:
    # FIFO must hold ACROSS drain calls, not just within one: a large deferred
    # request at the head of pending must not be overtaken by a smaller new
    # arrival that would fit on its own. (Regression: the old two-phase drain
    # ran _try_admit separately on pending and on new arrivals, so the
    # "once deferred, defer all later" latch reset between the two calls and
    # a small newcomer could be admitted ahead of a starved large request.)
    class _SizeGatedScheduler(_RejectAllScheduler):
        def can_add_new_sequence(self, seq) -> bool:
            return seq.num_tokens <= 3

    backend = _FakeBackend(prompt_tokens=[])
    # Tokenize by prompt length so the two requests get different sizes.
    backend.tokenize = lambda prompt: list(range(len(prompt)))
    scheduler = _SizeGatedScheduler()
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)

    req_large = _make_req(max_new_tokens=4)
    req_large.request_id = "large"
    req_large.prompt = "0123456789"  # 10 tokens -> never admittable here
    req_small = _make_req(max_new_tokens=4)
    req_small.request_id = "small"
    req_small.prompt = "01"  # 2 tokens -> admittable in isolation

    inbound: Queue = Queue()
    inbound.put(req_large)
    engine._drain_inbound(inbound)  # large is deferred
    inbound.put(req_small)
    engine._drain_inbound(inbound)  # small must queue BEHIND large

    assert scheduler.added == []  # nothing admitted: large blocks the line
    assert [r.request_id for r, _ in engine._pending] == ["large", "small"]


def test_pending_is_bounded_by_admission_headroom() -> None:
    # Backpressure regression: when the block manager (not the waiting queue)
    # is the bottleneck, the engine must stop pulling from inbound once the
    # pending buffer fills the admission budget. The surplus stays in the
    # bounded inbound queue, so submit() eventually raises Full -> 503 instead
    # of _pending growing without bound.
    backend = _FakeBackend(prompt_tokens=[3, 4, 5])
    scheduler = _RejectAllScheduler()
    scheduler.headroom = 2
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)

    inbound: Queue = Queue(maxsize=8)
    for _ in range(5):
        inbound.put_nowait(_make_req(max_new_tokens=4))

    engine._drain_inbound(inbound)
    assert len(engine._pending) == 2  # capped at the admission budget
    assert inbound.qsize() == 3  # the rest stays put as backpressure

    engine._drain_inbound(inbound)  # nothing was admitted -> no new headroom
    assert len(engine._pending) == 2
    assert inbound.qsize() == 3


def test_oversized_prompt_does_not_consume_budget() -> None:
    # A never-fits prompt is failed immediately; it must not eat an admission
    # slot that a feasible request behind it could use in the same drain.
    class _TinyBlockManager:
        total_blocks = 2
        block_size = 4

        def can_ever_allocate(self, seq) -> bool:
            return seq.num_tokens <= 8

    backend = _FakeBackend(prompt_tokens=[])
    backend.tokenize = lambda prompt: list(range(len(prompt)))
    scheduler = _RejectAllScheduler()
    scheduler.block_manager = _TinyBlockManager()
    scheduler.headroom = 1
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)

    oversized = _make_req(max_new_tokens=4)
    oversized.request_id = "oversized"
    oversized.prompt = "0123456789"  # 10 tokens > 8-token capacity
    normal = _make_req(max_new_tokens=4)
    normal.request_id = "normal"
    normal.prompt = "012"

    inbound: Queue = Queue()
    inbound.put(oversized)
    inbound.put(normal)
    engine._drain_inbound(inbound)

    assert oversized.status == RequestStatus.FAILED
    assert any(isinstance(e, ErrorEvent) for e in drain_events(oversized))
    # The single admission slot went to the feasible request behind it.
    assert [r.request_id for r, _ in engine._pending] == ["normal"]
    assert inbound.empty()


def test_cancel_inflight_fails_pending_requests() -> None:
    backend = _FakeBackend(prompt_tokens=[3, 4, 5])
    scheduler = _RejectAllScheduler()
    engine = ScheduleInferenceEngine(scheduler=scheduler, backend=backend)

    req = _make_req(max_new_tokens=4)
    inbound: Queue = Queue()
    inbound.put(req)
    engine._drain_inbound(inbound)
    assert len(engine._pending) == 1

    cancelled: list = []

    def cancel_request(r: GenerationRequestState, message: str) -> None:
        cancelled.append((r.request_id, message))

    # Real scheduler.clear() isn't available on the stub; provide a no-op.
    scheduler.clear = lambda: None  # type: ignore[attr-defined]
    engine.cancel_inflight("boom", cancel_request)

    assert cancelled == [(req.request_id, "boom")]
    assert len(engine._pending) == 0


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
        max_waiting=4,
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
    through the engine's _pending/_make_sequence path. Across a real
    preemption, each request is tokenized exactly once."""
    prompt_tokens = [3, 4]
    max_new_tokens = 4

    backend = _counting_backend(prompt_tokens)
    block_manager = BlockManager(total_blocks=6, block_size=1)
    scheduler = Scheduler(
        block_manager=block_manager,
        max_waiting=4,
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
        max_waiting=4,
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
