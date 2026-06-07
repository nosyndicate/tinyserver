import heapq

import pytest

from server.executor.types import Sequence
from server.model.block_manager import BlockManager


def make_sequence(
    sequence_id: str = "seq-0",
    num_tokens: int = 0,
    block_table: list[int] | None = None,
) -> Sequence:
    """Minimal Sequence factory for BlockManager tests.

    BlockManager only reads sequence_id, num_tokens and block_table; the other
    fields are set to harmless defaults.
    """
    return Sequence(
        sequence_id=sequence_id,
        seq_len=num_tokens,
        prompt_token_ids=[],
        generated_token_ids=[],
        num_prompt_tokens=num_tokens,
        num_tokens=num_tokens,
        block_table=list(block_table) if block_table is not None else [],
    )


# --- _num_blocks_needed ----------------------------------------------------


def test_num_blocks_needed_ceiling() -> None:
    bm = BlockManager(total_blocks=8, block_size=4)
    assert bm._num_blocks_needed(0) == 0
    assert bm._num_blocks_needed(1) == 1
    assert bm._num_blocks_needed(4) == 1
    assert bm._num_blocks_needed(5) == 2
    assert bm._num_blocks_needed(8) == 2


# --- can_allocate / allocate ----------------------------------------------


def test_can_allocate_true_and_false() -> None:
    bm = BlockManager(total_blocks=2, block_size=4)
    assert bm.can_allocate(make_sequence(num_tokens=8)) is True  # needs 2
    assert bm.can_allocate(make_sequence(num_tokens=9)) is False  # needs 3


def test_allocate_updates_state() -> None:
    bm = BlockManager(total_blocks=8, block_size=4)
    seq = make_sequence(num_tokens=6)  # needs 2 blocks

    bm.allocate(seq)

    assert len(seq.block_table) == 2
    assert bm.allocated_blocks[seq.sequence_id] == set(seq.block_table)
    assert len(bm.free_blocks) == 6
    # No overlap between allocated and free.
    assert set(seq.block_table).isdisjoint(bm.free_blocks)


def test_allocate_raises_when_insufficient() -> None:
    bm = BlockManager(total_blocks=1, block_size=4)
    with pytest.raises(MemoryError):
        bm.allocate(make_sequence(num_tokens=8))  # needs 2, only 1 free


def test_allocate_zero_tokens() -> None:
    bm = BlockManager(total_blocks=4, block_size=4)
    seq = make_sequence(num_tokens=0)

    bm.allocate(seq)

    assert seq.block_table == []
    assert bm.allocated_blocks[seq.sequence_id] == set()
    assert len(bm.free_blocks) == 4


def test_allocate_twice_raises_and_does_not_leak() -> None:
    bm = BlockManager(total_blocks=8, block_size=4)
    seq = make_sequence(num_tokens=6)
    bm.allocate(seq)
    free_after_first = len(bm.free_blocks)

    with pytest.raises(ValueError):
        bm.allocate(seq)

    # No blocks were leaked by the rejected second allocation.
    assert len(bm.free_blocks) == free_after_first


def test_allocate_preserves_block_order() -> None:
    # Drain the heap down to free blocks {7, 8}, whose set iteration order is NOT
    # ascending in CPython. block_table must still be the heap pop order [7, 8].
    bm = BlockManager(total_blocks=9, block_size=4)
    for block in range(7):
        heapq.heappop(bm.free_blocks)
    assert sorted(bm.free_blocks) == [7, 8]

    seq = make_sequence(num_tokens=8)  # needs 2 blocks
    bm.allocate(seq)

    assert seq.block_table == [7, 8]


# --- can_append / append (idempotent ensure-capacity) ----------------------


def test_append_within_block_is_noop() -> None:
    bm = BlockManager(total_blocks=8, block_size=4)
    seq = make_sequence(num_tokens=2)  # 1 block
    bm.allocate(seq)
    free_before = len(bm.free_blocks)
    table_before = list(seq.block_table)

    seq.num_tokens = 3  # still fits in the first block
    bm.append(seq)

    assert seq.block_table == table_before
    assert len(bm.free_blocks) == free_before


def test_append_crosses_block_boundary() -> None:
    bm = BlockManager(total_blocks=8, block_size=4)
    seq = make_sequence(num_tokens=4)  # exactly 1 full block
    bm.allocate(seq)
    free_before = len(bm.free_blocks)

    seq.num_tokens = 5  # needs a 2nd block
    bm.append(seq)

    assert len(seq.block_table) == 2
    assert len(bm.free_blocks) == free_before - 1
    assert bm.allocated_blocks[seq.sequence_id] == set(seq.block_table)


def test_append_is_idempotent() -> None:
    bm = BlockManager(total_blocks=8, block_size=4)
    seq = make_sequence(num_tokens=5)
    bm.allocate(seq)  # allocate already covers 5 tokens -> 2 blocks
    free_before = len(bm.free_blocks)
    table_before = list(seq.block_table)

    bm.append(seq)  # nothing additional needed
    bm.append(seq)

    assert seq.block_table == table_before
    assert len(bm.free_blocks) == free_before


def test_append_tops_up_multiple_blocks() -> None:
    bm = BlockManager(total_blocks=8, block_size=4)
    seq = make_sequence(num_tokens=2)  # 1 block
    bm.allocate(seq)

    seq.num_tokens = 9  # needs 3 blocks total -> 2 more
    bm.append(seq)

    assert len(seq.block_table) == 3
    assert bm.allocated_blocks[seq.sequence_id] == set(seq.block_table)


def test_can_append_reflects_availability() -> None:
    bm = BlockManager(total_blocks=2, block_size=4)
    seq = make_sequence(num_tokens=4)  # 1 block
    bm.allocate(seq)  # 1 free block remains

    seq.num_tokens = 5  # needs 1 more
    assert bm.can_append(seq) is True

    seq.num_tokens = 9  # needs 2 more, only 1 free
    assert bm.can_append(seq) is False


def test_append_raises_when_exhausted() -> None:
    bm = BlockManager(total_blocks=1, block_size=4)
    seq = make_sequence(num_tokens=4)  # uses the only block
    bm.allocate(seq)

    seq.num_tokens = 5  # needs another block, none free
    with pytest.raises(MemoryError):
        bm.append(seq)


def test_append_unallocated_sequence_raises() -> None:
    bm = BlockManager(total_blocks=8, block_size=4)
    seq = make_sequence(num_tokens=5)  # never allocated
    with pytest.raises(ValueError):
        bm.append(seq)


# --- free ------------------------------------------------------------------


def test_free_returns_blocks_and_clears_table() -> None:
    bm = BlockManager(total_blocks=8, block_size=4)
    seq = make_sequence(num_tokens=6)
    bm.allocate(seq)

    bm.free(seq)

    assert seq.block_table == []
    assert seq.sequence_id not in bm.allocated_blocks
    assert len(bm.free_blocks) == 8


def test_free_is_idempotent() -> None:
    bm = BlockManager(total_blocks=8, block_size=4)
    seq = make_sequence(num_tokens=6)
    bm.allocate(seq)

    bm.free(seq)
    bm.free(seq)  # double free is a safe no-op

    assert len(bm.free_blocks) == 8


def test_allocate_append_free_reallocate_round_trip() -> None:
    bm = BlockManager(total_blocks=4, block_size=4)
    seq = make_sequence(num_tokens=4)
    bm.allocate(seq)
    seq.num_tokens = 5
    bm.append(seq)  # 2 blocks held

    bm.free(seq)
    assert len(bm.free_blocks) == 4

    # All blocks are available again for a fresh allocation.
    seq2 = make_sequence(sequence_id="seq-1", num_tokens=16)  # needs all 4 blocks
    bm.allocate(seq2)
    assert len(seq2.block_table) == 4
    assert len(bm.free_blocks) == 0
