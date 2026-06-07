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

    def can_allocate(self, sequence: Sequence) -> bool:
        """Checks if the requested sequence can be allocated given the current free blocks."""
        num_blocks_needed = self._num_blocks_needed(sequence.num_tokens)
        return len(self.free_blocks) >= num_blocks_needed

    def can_append(self, sequence: Sequence) -> bool:
        """Checks if we can append one more token to the sequence."""
        if sequence.num_tokens % self.block_size == 0:
            # Need a new block for the next token
            return len(self.free_blocks) >= 1
        else:
            # Current block has space for the next token
            return True

    def allocate(self, sequence: Sequence) -> None:
        if not self.can_allocate(sequence):
            raise MemoryError("Not enough free blocks to allocate")

        num_blocks_needed = self._num_blocks_needed(sequence.num_tokens)
        allocated = set()
        for _ in range(num_blocks_needed):
            block_id = heapq.heappop(self.free_blocks)
            allocated.add(block_id)

        self.allocated_blocks[sequence.sequence_id] = allocated
        sequence.block_table = list(allocated)

    def append(self, sequence: Sequence) -> None:
        if not self.can_append(sequence):
            raise MemoryError("Not enough free blocks to append")

        if sequence.num_tokens % self.block_size == 0:
            # Need to allocate a new block
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
