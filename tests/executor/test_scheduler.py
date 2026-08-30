import pytest

from server.executor.scheduler import Scheduler
from server.executor.types import (
    Sequence,
    SequenceBatchTask,
    SequenceState,
)
from server.model.block_manager import BlockManager


def make_sequence(
    sequence_id: str = "seq-0",
    num_tokens: int = 1,
    block_table: list[int] | None = None,
    max_new_tokens: int = 1,
) -> Sequence:
    """Minimal Sequence factory mirroring tests/model/test_block_manager.py."""
    return Sequence(
        sequence_id=sequence_id,
        prompt_token_ids=list(range(num_tokens)),
        generated_token_ids=[],
        num_prompt_tokens=num_tokens,
        num_tokens=num_tokens,
        max_new_tokens=max_new_tokens,
        block_table=list(block_table) if block_table is not None else [],
    )


def make_scheduler(
    total_blocks: int = 16,
    block_size: int = 4,
    max_num_sequences: int = 8,
    max_num_tokens: int = 1024,
) -> Scheduler:
    bm = BlockManager(total_blocks=total_blocks, block_size=block_size)
    return Scheduler(
        block_manager=bm,
        max_num_sequences=max_num_sequences,
        max_num_tokens=max_num_tokens,
    )


# --- logical sequence capacity --------------------------------------------


def test_has_sequence_capacity_counts_waiting_and_running() -> None:
    sched = make_scheduler()
    assert sched.has_sequence_capacity() is True
    for i in range(sched.max_num_sequences):
        sched.add(make_sequence(sequence_id=str(i)))
    assert sched.has_sequence_capacity() is False


def test_add_at_capacity_is_an_invariant_violation() -> None:
    sched = make_scheduler(max_num_sequences=1)
    sched.add(make_sequence())
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        sched.add(make_sequence(sequence_id="overflow"))


@pytest.mark.parametrize(
    ("max_num_sequences", "max_num_tokens", "message"),
    [
        (0, 8, "max_num_sequences must be positive"),
        (4, 3, "max_num_tokens must be greater than or equal"),
    ],
)
def test_scheduler_rejects_invalid_capacity_configuration(
    max_num_sequences: int, max_num_tokens: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_scheduler(
            max_num_sequences=max_num_sequences,
            max_num_tokens=max_num_tokens,
        )


# --- schedule(): prefill ---------------------------------------------------


def test_schedule_prefill_moves_waiting_to_running() -> None:
    sched = make_scheduler()
    seq = make_sequence(sequence_id="a", num_tokens=4)
    sched.add(seq)

    batch = sched.schedule()

    assert batch is not None
    assert batch.kind is SequenceBatchTask.PREFILL
    assert batch.sequences == [seq]
    assert seq.state is SequenceState.RUNNING
    assert seq.block_table  # blocks were allocated
    assert not sched.waiting
    assert sched.running == [seq]


def test_schedule_prefill_respects_max_num_sequences() -> None:
    sched = make_scheduler(max_num_sequences=2)
    for i in range(2):
        sched.add(make_sequence(sequence_id=f"s{i}", num_tokens=4))

    batch = sched.schedule()

    assert batch is not None
    assert batch.kind is SequenceBatchTask.PREFILL
    assert len(batch.sequences) == 2
    assert len(sched.running) == 2
    assert not sched.waiting


def test_schedule_prefill_respects_token_budget() -> None:
    sched = make_scheduler(max_num_tokens=8)
    sched.add(make_sequence(sequence_id="a", num_tokens=8))
    sched.add(make_sequence(sequence_id="b", num_tokens=8))

    batch = sched.schedule()

    assert batch is not None
    assert [s.sequence_id for s in batch.sequences] == ["a"]
    assert len(sched.waiting) == 1
    assert sched.reservation_blocked_count == 0


def test_schedule_prefill_allows_single_oversized_sequence() -> None:
    # Regression for head-of-line blocking: a sequence larger than the per-batch
    # token budget must still be scheduled on its own rather than stalling.
    sched = make_scheduler(max_num_sequences=4, max_num_tokens=4)
    seq = make_sequence(sequence_id="big", num_tokens=8)
    sched.add(seq)

    batch = sched.schedule()

    assert batch is not None
    assert batch.sequences == [seq]


def test_terminal_prefill_may_use_the_entire_cache() -> None:
    sched = make_scheduler(
        block_size=4,
        total_blocks=2,
        max_num_sequences=1,
        max_num_tokens=8,
    )
    terminal = make_sequence(num_tokens=8, max_new_tokens=1)
    sched.add(terminal)

    batch = sched.schedule()

    assert batch is not None
    assert batch.sequences == [terminal]
    assert sched.reservation_blocked_count == 0
    assert sched.block_manager.num_free_blocks == 0


def test_prefill_requiring_decode_is_blocked_without_next_kv_position() -> None:
    sched = make_scheduler(
        block_size=4,
        total_blocks=2,
        max_num_sequences=1,
        max_num_tokens=8,
    )
    needs_decode = make_sequence(num_tokens=8, max_new_tokens=2)
    sched.add(needs_decode)

    batch = sched.schedule()

    assert batch is None
    assert list(sched.waiting) == [needs_decode]
    assert not sched.running
    assert sched.reservation_blocked_count == 1
    assert needs_decode.block_table == []


def test_prefill_preserves_running_population_next_decode() -> None:
    sched = make_scheduler(
        block_size=4,
        total_blocks=2,
        max_num_sequences=2,
        max_num_tokens=8,
    )
    running = make_sequence(sequence_id="running", num_tokens=4, max_new_tokens=2)
    running.generated_token_ids.append(9)
    running.state = SequenceState.RUNNING
    sched.block_manager.allocate(running)
    sched.running.append(running)
    candidate = make_sequence(sequence_id="candidate", num_tokens=4, max_new_tokens=1)
    sched.add(candidate)

    batch = sched.schedule()

    # Candidate promotion would consume the only free block needed by the
    # running sequence, so prefill is blocked and decode advances instead.
    assert batch is not None
    assert batch.kind is SequenceBatchTask.DECODE
    assert batch.sequences == [running]
    assert list(sched.waiting) == [candidate]
    assert sched.reservation_blocked_count == 1


def test_prefill_reservation_updates_after_each_selected_candidate() -> None:
    sched = make_scheduler(
        block_size=4,
        total_blocks=3,
        max_num_sequences=3,
        max_num_tokens=8,
    )
    running = make_sequence(sequence_id="running", num_tokens=4, max_new_tokens=3)
    running.generated_token_ids.append(9)
    running.state = SequenceState.RUNNING
    sched.block_manager.allocate(running)
    sched.running.append(running)
    first = make_sequence(sequence_id="first", num_tokens=4, max_new_tokens=1)
    second = make_sequence(sequence_id="second", num_tokens=4, max_new_tokens=1)
    sched.add(first)
    sched.add(second)

    batch = sched.schedule()

    assert batch is not None
    assert batch.kind is SequenceBatchTask.PREFILL
    assert batch.sequences == [first]
    assert list(sched.waiting) == [second]
    assert sched.reservation_blocked_count == 1


# --- schedule(): decode ----------------------------------------------------


def _prefill_running(sched: Scheduler, ids: list[str], num_tokens: int) -> None:
    for sid in ids:
        sched.add(make_sequence(sequence_id=sid, num_tokens=num_tokens))
    sched.schedule()  # consume the prefill batch, populating running


def test_schedule_decode_returns_distinct_sequences_and_terminates() -> None:
    # Regression for the infinite-loop / duplicate-batch bug.
    sched = make_scheduler(block_size=4, total_blocks=16)
    _prefill_running(sched, ["a", "b", "c"], num_tokens=2)

    batch = sched.schedule()

    assert batch is not None
    assert batch.kind is SequenceBatchTask.DECODE
    ids = [s.sequence_id for s in batch.sequences]
    assert ids == ["a", "b", "c"]
    assert len(set(ids)) == len(ids)  # no duplicates


def test_schedule_decode_reserves_block_on_boundary() -> None:
    # num_tokens=4 with block_size=4 fills the block exactly; decoding one more
    # token must reserve a fresh block. The scheduler reserves the block but
    # leaves num_tokens unchanged — the engine advances it after generating.
    sched = make_scheduler(block_size=4, total_blocks=16)
    _prefill_running(sched, ["a"], num_tokens=4)
    seq = sched.running[0]
    assert len(seq.block_table) == 1

    batch = sched.schedule()

    assert batch is not None
    assert batch.kind is SequenceBatchTask.DECODE
    assert seq.num_tokens == 4  # scheduler does not advance num_tokens
    assert len(seq.block_table) == 2  # new block reserved for the next token


def test_schedule_decode_preempts_youngest_to_make_room() -> None:
    # block_size=1 so every decoded token needs a new block. With only enough
    # blocks for prefill, the oldest sequence can't append until the youngest
    # is preempted, freeing its block.
    sched = make_scheduler(block_size=1, total_blocks=2)
    _prefill_running(sched, ["a", "b"], num_tokens=1)
    assert not sched.block_manager.free_blocks

    batch = sched.schedule()

    # Oldest ("a") makes progress; youngest ("b") is evicted.
    assert batch is not None
    assert batch.kind is SequenceBatchTask.DECODE
    assert [s.sequence_id for s in batch.sequences] == ["a"]

    assert [s.sequence_id for s in sched.running] == ["a"]
    assert sched.running[0].num_tokens == 1  # scheduler never mutates num_tokens
    assert len(sched.running[0].block_table) == 2  # block reserved for next token

    # "b" is preempted: back at the front of waiting, blocks freed, state set.
    assert [s.sequence_id for s in sched.waiting] == ["b"]
    victim = sched.waiting[0]
    assert victim.state is SequenceState.PREEMPTED
    assert victim.block_table == []
    assert sched.preemption_count == 1


def test_preempt_evicts_youngest_and_requeues_at_front() -> None:
    sched = make_scheduler(block_size=4, total_blocks=16)
    _prefill_running(sched, ["a", "b"], num_tokens=4)
    free_before = len(sched.block_manager.free_blocks)

    victim = sched._preempt()

    # Youngest (tail) is evicted and moved to the FRONT of waiting.
    assert victim.sequence_id == "b"
    assert victim.state is SequenceState.PREEMPTED
    assert victim.block_table == []
    assert victim.sequence_id not in sched.block_manager.allocated_blocks
    assert [s.sequence_id for s in sched.running] == ["a"]
    assert sched.waiting[0] is victim
    assert len(sched.block_manager.free_blocks) > free_before  # blocks returned
    assert sched.preemption_count == 1


def test_preempt_recomputes_num_tokens_from_prompt_and_generated_lengths() -> None:
    # After preemption, num_tokens must reflect ALL
    # fed tokens (prompt + generated) that resume-prefill will recompute KV
    # for
    sched = make_scheduler(block_size=4, total_blocks=16)
    seq = Sequence(
        sequence_id="a",
        prompt_token_ids=[1, 2, 3],  # P = 3
        generated_token_ids=[9, 10],  # G = 2
        num_prompt_tokens=3,
        num_tokens=4,  # stale: P + G - 1 = 3 + 2 - 1
        max_new_tokens=5,
        block_table=[0],
    )
    sched.running.append(seq)

    victim = sched._preempt()

    assert victim.num_tokens == 5  # corrected to P + G = 3 + 2


def test_schedule_decode_preempts_multiple_youngest_for_older_sequences() -> None:
    # block_size=1, all 4 blocks used by 4 running sequences. Decoding the two
    # oldest requires evicting the two youngest, one preemption each.
    sched = make_scheduler(block_size=1, total_blocks=4)
    _prefill_running(sched, ["a", "b", "c", "d"], num_tokens=1)
    assert not sched.block_manager.free_blocks

    batch = sched.schedule()

    assert batch is not None
    assert batch.kind is SequenceBatchTask.DECODE
    assert [s.sequence_id for s in batch.sequences] == ["a", "b"]
    assert [s.sequence_id for s in sched.running] == ["a", "b"]
    assert sched.preemption_count == 2
    # "d" was preempted first, then "c" via appendleft, so the older victim
    # ("c") sits in front and resumes before "d".
    assert [s.sequence_id for s in sched.waiting] == ["c", "d"]
    assert all(s.state is SequenceState.PREEMPTED for s in sched.waiting)


def test_schedule_decode_leaves_youngest_running_when_it_cannot_fit() -> None:
    # block_size=1, 3 blocks: 2 held by running "a"/"b", 1 free. "a" takes the
    # last free block; "b" is then the youngest not-yet-scheduled sequence and
    # cannot append. It must NOT be self-preempted — it keeps its KV and waits.
    sched = make_scheduler(block_size=1, total_blocks=3)
    _prefill_running(sched, ["a", "b"], num_tokens=1)
    assert len(sched.block_manager.free_blocks) == 1

    batch = sched.schedule()

    assert batch is not None
    assert batch.kind is SequenceBatchTask.DECODE
    assert [s.sequence_id for s in batch.sequences] == ["a"]
    # "b" is untouched: still running, never evicted.
    assert [s.sequence_id for s in sched.running] == ["a", "b"]
    assert sched.running[1].state is SequenceState.RUNNING
    assert not sched.waiting
    assert sched.preemption_count == 0


def test_preempted_sequence_resumes_before_newer_waiting() -> None:
    # After a preemption, the evicted sequence must resume (re-prefill) ahead of
    # any newer waiting request once blocks free up.
    sched = make_scheduler(block_size=1, total_blocks=2)
    _prefill_running(sched, ["a", "b"], num_tokens=1)
    sched.schedule()  # "a" decodes, "b" is preempted to the front of waiting
    assert [s.sequence_id for s in sched.waiting] == ["b"]

    # "a" completes and frees its blocks; a newer request "c" arrives.
    sched.running[0].finished = True
    sched.add(make_sequence(sequence_id="c", num_tokens=1))

    batch = sched.schedule()

    assert batch is not None
    assert batch.kind is SequenceBatchTask.PREFILL
    # Resumed "b" is prefilled before the newer "c".
    assert [s.sequence_id for s in batch.sequences][0] == "b"
    resumed = next(s for s in sched.running if s.sequence_id == "b")
    assert resumed.state is SequenceState.RUNNING
    # The engine needs to know "b" is a resume (to recompute prompt+generated)
    # and "c" is fresh (prompt only) -- captured before "b"'s state was
    # overwritten to RUNNING above, since that's the only place it's known.
    assert batch.resumed_sequence_ids == frozenset({"b"})


def test_schedule_decode_drops_finished_sequences() -> None:
    sched = make_scheduler(block_size=4, total_blocks=16)
    _prefill_running(sched, ["a", "b"], num_tokens=2)
    sched.running[0].finished = True

    batch = sched.schedule()

    assert batch is not None
    assert [s.sequence_id for s in batch.sequences] == ["b"]
    assert [s.sequence_id for s in sched.running] == ["b"]


def test_schedule_reaps_finished_sequences_from_waiting() -> None:
    # A cancelled sequence can be sitting in `waiting` (e.g. PREEMPTED, blocks
    # already freed, or admitted-but-not-yet-prefilled). Marking it finished
    # must drop it so the prefill pass never re-schedules/re-prefills it.
    sched = make_scheduler(block_size=4, total_blocks=16)
    keep = make_sequence(sequence_id="keep", num_tokens=4)
    cancelled = make_sequence(sequence_id="cancelled", num_tokens=4)
    sched.add(cancelled)
    sched.add(keep)
    cancelled.finished = True

    batch = sched.schedule()

    assert batch is not None
    assert batch.kind is SequenceBatchTask.PREFILL
    # Only the live sequence is scheduled; the cancelled one is gone from both
    # the batch and the waiting queue, and never allocated a block.
    assert [s.sequence_id for s in batch.sequences] == ["keep"]
    assert "cancelled" not in [s.sequence_id for s in sched.waiting]
    assert "cancelled" not in sched.block_manager.allocated_blocks


# --- reap + capacity contract ---------------------------------------------


def test_schedule_reaps_finished_before_prefill_so_blocks_are_reused() -> None:
    # Regression: finished sequences must be reaped at the top of schedule(),
    # even when prefill is productive. Previously prefill's early return
    # skipped cleanup, so a finished sequence's blocks stayed allocated and
    # blocked new prefills (schedule() returned None instead).
    sched = make_scheduler(block_size=4, total_blocks=4)
    # One running sequence holding all 4 blocks, now finished.
    done = make_sequence(sequence_id="done", num_tokens=16)
    sched.add(done)
    sched.schedule()  # prefill -> running, allocates all 4 blocks
    done.finished = True
    assert not sched.block_manager.free_blocks

    # A waiting sequence that can only prefill once `done`'s blocks free up.
    sched.add(make_sequence(sequence_id="next", num_tokens=4))

    batch = sched.schedule()

    assert batch is not None
    assert batch.kind is SequenceBatchTask.PREFILL
    assert [s.sequence_id for s in batch.sequences] == ["next"]
    # The finished sequence was reaped and its blocks reused, not leaked.
    assert all(not s.finished for s in sched.running)
    assert done.sequence_id not in sched.block_manager.allocated_blocks


def test_schedule_does_not_mutate_num_tokens_during_decode() -> None:
    # Contract guard: the scheduler reserves capacity but never advances
    # num_tokens — the engine owns that. Capture lengths before and after.
    sched = make_scheduler(block_size=4, total_blocks=16)
    _prefill_running(sched, ["a", "b", "c"], num_tokens=2)
    before = {s.sequence_id: s.num_tokens for s in sched.running}

    batch = sched.schedule()

    assert batch is not None
    assert batch.kind is SequenceBatchTask.DECODE
    after = {s.sequence_id: s.num_tokens for s in sched.running}
    assert before == after  # scheduler left num_tokens untouched


# --- clear -----------------------------------------------------------------


def test_clear_frees_running_blocks() -> None:
    sched = make_scheduler(block_size=4, total_blocks=16)
    _prefill_running(sched, ["a", "b"], num_tokens=4)
    free_before_clear = len(sched.block_manager.free_blocks)

    sched.clear()

    assert not sched.running
    assert not sched.waiting
    assert len(sched.block_manager.free_blocks) > free_before_clear
    assert len(sched.block_manager.free_blocks) == 16


def test_clear_handles_preempted_sequence_in_waiting() -> None:
    # A PREEMPTED sequence in `waiting` already holds no blocks (freed at
    # eviction) -- clear() must not double-free or otherwise choke on it.
    sched = make_scheduler(block_size=4, total_blocks=16)
    _prefill_running(sched, ["a"], num_tokens=4)
    preempted = make_sequence(sequence_id="b", num_tokens=4)
    preempted.state = SequenceState.PREEMPTED
    sched.waiting.append(preempted)

    sched.clear()

    assert not sched.running
    assert not sched.waiting
    assert len(sched.block_manager.free_blocks) == 16
