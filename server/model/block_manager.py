import heapq

from server.executor.types import Sequence


class BlockManager:
    """
    Manages the allocation and deallocation of blocks for sequences.

    Args:
        total_blocks: Total number of blocks available in the system.
        block_size: Number of tokens each block can hold.
    """

    def __init__(self, total_blocks: int, block_size: int):
        self.total_blocks = total_blocks
        self.block_size = block_size

        # free_blocks is a min-heap of available block IDs. We use a heap to efficiently allocate and free blocks.
        self.free_blocks = list(range(total_blocks))
        heapq.heapify(self.free_blocks)
        self.allocated_blocks: dict[
            str, set[int]
        ] = {}  # sequence_id -> set of block_ids

    def _num_blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size  # Ceiling division

    def has_free_blocks_for(self, num_tokens: int) -> bool:
        """Checks if there are enough free blocks to hold num_tokens tokens."""
        return len(self.free_blocks) >= self._num_blocks_needed(num_tokens)

    def can_allocate(self, sequence: Sequence) -> bool:
        """Checks if the requested sequence can be allocated given the current free blocks."""
        return self.has_free_blocks_for(sequence.num_tokens)

    def _additional_blocks_needed(
        self, sequence: Sequence, extra_tokens: int = 0
    ) -> int:
        """Number of extra blocks needed to hold num_tokens + extra_tokens tokens."""
        needed = self._num_blocks_needed(sequence.num_tokens + extra_tokens)
        return needed - len(sequence.block_table)

    def can_append(self, sequence: Sequence, extra_tokens: int = 0) -> bool:
        """Checks if the sequence has (or can obtain) enough blocks for num_tokens.

        Pass extra_tokens > 0 to check capacity for tokens beyond num_tokens,
        e.g. reserving a block for a token the engine is about to generate
        without having advanced num_tokens yet.
        """
        additional = self._additional_blocks_needed(sequence, extra_tokens)
        return len(self.free_blocks) >= max(0, additional)

    def allocate(self, sequence: Sequence) -> None:
        if sequence.sequence_id in self.allocated_blocks:
            raise ValueError(
                f"Sequence {sequence.sequence_id} is already allocated; "
                "free it before re-allocating"
            )
        if not self.can_allocate(sequence):
            raise MemoryError("Not enough free blocks to allocate")

        num_blocks_needed = self._num_blocks_needed(sequence.num_tokens)
        # Pop into an ordered list: logical block order in block_table is
        # significant for paged attention. The set is only for free bookkeeping.
        block_ids = [heapq.heappop(self.free_blocks) for _ in range(num_blocks_needed)]

        self.allocated_blocks[sequence.sequence_id] = set(block_ids)
        sequence.block_table = list(block_ids)

    def append(self, sequence: Sequence, extra_tokens: int = 0) -> None:
        """
        Idempotently ensures the sequence has enough blocks for num_tokens +
        extra_tokens, allocating any additional blocks needed. The scheduler
        passes extra_tokens=1 during decode to reserve the slot for the token
        the engine is about to generate; num_tokens is advanced by the engine
        afterwards, not here.
        """
        if sequence.sequence_id not in self.allocated_blocks:
            raise ValueError(
                f"Sequence {sequence.sequence_id} is not allocated; "
                "call allocate() first"
            )

        additional = self._additional_blocks_needed(sequence, extra_tokens)
        if additional <= 0:
            # Already has enough capacity for num_tokens.
            return
        if len(self.free_blocks) < additional:
            raise MemoryError("Not enough free blocks to append")

        for _ in range(additional):
            block_id = heapq.heappop(self.free_blocks)
            self.allocated_blocks[sequence.sequence_id].add(block_id)
            sequence.block_table.append(block_id)

    def free(self, sequence: Sequence) -> None:
        """
        Clears the blocks allocated to the sequence and returns them to the free pool.
        """
        allocated = self.allocated_blocks.pop(sequence.sequence_id, set())
        for block_id in allocated:
            heapq.heappush(self.free_blocks, block_id)

        sequence.block_table = []
