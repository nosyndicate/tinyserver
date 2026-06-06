import math

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
        self.free_blocks = set(range(total_blocks))
        self.allocated_blocks: dict[
            str, set[int]
        ] = {}  # sequence_id -> set of block_ids

    def _num_blocks_needed(self, num_tokens: int) -> int:
        return int(math.ceil(num_tokens / self.block_size))

    def can_allocate(self, sequence: Sequence) -> bool:
        """Checks if the requested sequence can be allocated given the current free blocks."""
        num_blocks = self._num_blocks_needed(sequence.num_tokens)
        return len(self.free_blocks) >= num_blocks

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

        num_blocks = self._num_blocks_needed(sequence.num_tokens)
        allocated = set()
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop()
            allocated.add(block_id)

        self.allocated_blocks[sequence.sequence_id] = allocated
        sequence.block_table = list(allocated)

    def append(self, sequence: Sequence) -> None:
        if not self.can_append(sequence):
            raise MemoryError("Not enough free blocks to append")

        if sequence.num_tokens % self.block_size == 0:
            # Need to allocate a new block
            block_id = self.free_blocks.pop()
            self.allocated_blocks[sequence.sequence_id].add(block_id)
            sequence.block_table.append(block_id)

    def free(self, sequence: Sequence) -> None:
        allocated = self.allocated_blocks.pop(sequence.sequence_id, set())
        for block_id in allocated:
            self.free_blocks.add(block_id)

        sequence.block_table = []
