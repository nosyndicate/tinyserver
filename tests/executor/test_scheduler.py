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
        prompt_token_ids=[],
        generated_token_ids=[],
        num_prompt_tokens=num_tokens,
        num_tokens=num_tokens,
        max_new_tokens=max_new_tokens,
        block_table=list(block_table) if block_table is not None else [],
    )


def make_scheduler(
    total_blocks: int = 16,
    block_size: int = 4,
    max_waiting: int = 8,
    max_num_sequences: int = 8,
    max_num_tokens: int = 1024,
) -> Scheduler:
    bm = BlockManager(total_blocks=total_blocks, block_size=block_size)
    return Scheduler(
        block_manager=bm,
        max_waiting=max_waiting,
        max_num_sequences=max_num_sequences,
        max_num_tokens=max_num_tokens,
    )


# --- can_add_new_sequence --------------------------------------------------


def test_can_add_new_sequence_true_when_capacity_available() -> None:
    sched = make_scheduler()
    assert sched.can_add_new_sequence(make_sequence(num_tokens=1)) is True


def test_can_add_new_sequence_false_when_waiting_full() -> None:
    sched = make_scheduler(max_waiting=1)
    sched.add(make_sequence())
    assert sched.can_add_new_sequence(make_sequence(num_tokens=1)) is False


def test_can_add_new_sequence_false_when_blocks_exhausted() -> None:
    sched = make_scheduler(total_blocks=1, block_size=4)
    # Drain the single free block via a running sequence.
    seq = make_sequence(num_tokens=1)
    sched.block_manager.allocate(seq)
    assert sched.can_add_new_sequence(make_sequence(num_tokens=1)) is False


def test_can_add_new_sequence_false_when_prompt_exceeds_free_blocks() -> None:
    # Regression: previously used has_free_blocks_for(1), which admitted a
    # sequence as long as ONE token fit, ignoring that the full prompt needed
    # more blocks than were free. can_allocate must reject it.
    sched = make_scheduler(total_blocks=4, block_size=4)  # 4 blocks = 16 tokens
    holder = make_sequence(sequence_id="holder", num_tokens=12)  # 3 blocks
    sched.block_manager.allocate(holder)
    assert len(sched.block_manager.free_blocks) == 1  # one 4-token block free

    big = make_sequence(sequence_id="big", num_tokens=8)  # needs 2 blocks
    assert sched.can_add_new_sequence(big) is False


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
    for i in range(3):
        sched.add(make_sequence(sequence_id=f"s{i}", num_tokens=4))

    batch = sched.schedule()

    assert batch is not None
    assert batch.kind is SequenceBatchTask.PREFILL
    assert len(batch.sequences) == 2
    assert len(sched.waiting) == 1


def test_schedule_prefill_respects_token_budget() -> None:
    sched = make_scheduler(max_num_tokens=8)
    sched.add(make_sequence(sequence_id="a", num_tokens=8))
    sched.add(make_sequence(sequence_id="b", num_tokens=8))

    batch = sched.schedule()

    assert batch is not None
    assert [s.sequence_id for s in batch.sequences] == ["a"]
    assert len(sched.waiting) == 1


def test_schedule_prefill_allows_single_oversized_sequence() -> None:
    # Regression for head-of-line blocking: a sequence larger than the per-batch
    # token budget must still be scheduled on its own rather than stalling.
    sched = make_scheduler(max_num_tokens=4)
    seq = make_sequence(sequence_id="big", num_tokens=8)
    sched.add(seq)

    batch = sched.schedule()

    assert batch is not None
    assert batch.sequences == [seq]


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


def test_schedule_decode_skips_sequence_without_free_blocks() -> None:
    # block_size=1 so every decoded token needs a new block. With only enough
    # blocks for prefill, no sequence can append and decode yields nothing.
    sched = make_scheduler(block_size=1, total_blocks=2)
    _prefill_running(sched, ["a", "b"], num_tokens=1)
    assert not sched.block_manager.free_blocks

    batch = sched.schedule()

    assert batch is None
    # num_tokens left untouched (the scheduler never mutates it).
    assert all(seq.num_tokens == 1 for seq in sched.running)


def test_schedule_decode_drops_finished_sequences() -> None:
    sched = make_scheduler(block_size=4, total_blocks=16)
    _prefill_running(sched, ["a", "b"], num_tokens=2)
    sched.running[0].finished = True

    batch = sched.schedule()

    assert batch is not None
    assert [s.sequence_id for s in batch.sequences] == ["b"]
    assert [s.sequence_id for s in sched.running] == ["b"]


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
