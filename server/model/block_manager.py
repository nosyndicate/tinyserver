import heapq

from server.executor.types import Sequence


class BlockManager:
    """
    Manages the allocation and deallocation of blocks for sequences.

    Args:
        total_blocks: Total number of blocks available in the system.
        block_size: Number of tokens each block can hold.

    RESERVATION INVARIANT: admission uses a check-then-reserve protocol —
    ``can_admit`` checks the *effective* free pool (physical free blocks minus
    every outstanding reservation) and ``reserve`` records the sequence's
    worst-case block count. As long as every tracked sequence went through
    that pair, ``sum(reserved.values()) <= len(free_blocks)`` always holds, so
    ``allocate``/``append`` for a reserved sequence can never run out of
    physical blocks. This is what prevents two concurrently admitted requests
    from jointly over-committing the pool and deadlocking mid-decode.

    THREAD SAFETY: ``free_blocks``, ``allocated_blocks`` and the reservation
    ledger are not locked. This is correct only because a single worker thread
    drives the engine loop; sharing a manager across engine threads would need
    external synchronization.
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
        # Reservation ledger: sequence_id -> worst-case blocks the sequence may
        # still need beyond what is already in its block_table. Recorded at
        # admission (reserve), drawn down as blocks materialize (allocate /
        # append), released on free. _total_reserved caches the sum so the
        # admission check stays O(1).
        self.reserved: dict[str, int] = {}
        self._total_reserved = 0

    def _num_blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size  # Ceiling division

    def has_free_blocks_for(self, num_tokens: int) -> bool:
        """Checks if there are enough free blocks to hold num_tokens tokens."""
        return len(self.free_blocks) >= self._num_blocks_needed(num_tokens)

    def can_allocate(self, sequence: Sequence) -> bool:
        """Checks if the sequence's prompt can be physically allocated now.

        Prompt-only on purpose: ``allocate`` only pops blocks for the prompt, so
        this guard must match what is actually allocated. Admission decisions use
        ``can_admit`` + ``reserve`` instead, which account for the generation
        budget.
        """
        return self.has_free_blocks_for(sequence.num_tokens)

    def worst_case_blocks(self, sequence: Sequence) -> int:
        """Blocks needed if the sequence generates its full ``max_new_tokens``.

        The budget is prompt + max_new_tokens - 1, not + max_new_tokens: prefill
        stores KV for positions 0..P-1, and each decode step stores KV for its
        *input* token before sampling the next one. The final sampled token ends
        generation immediately, so its KV is never written to the cache. The
        max(0, ...) keeps max_new_tokens=0 sequences prompt-only.
        """
        return self._num_blocks_needed(
            sequence.num_tokens + max(0, sequence.max_new_tokens - 1)
        )

    def can_admit(self, sequence: Sequence) -> bool:
        """Whether the sequence's worst-case footprint fits the effective pool.

        The effective pool is the physical free blocks minus every outstanding
        reservation, so already-admitted sequences' unmaterialized budgets are
        counted even though their blocks haven't been popped yet. This is a
        check only — the caller must follow up with ``reserve`` to claim the
        capacity. Only call this for sequences that are not yet reserved: a
        reserved sequence's own budget sits in ``_total_reserved`` and would be
        double-counted against it.

        Conservative: most generations stop early at EOS, so reserving the full
        budget trades throughput for a liveness guarantee.
        """
        effective_free = len(self.free_blocks) - self._total_reserved
        return effective_free >= self.worst_case_blocks(sequence)

    def reserve(self, sequence: Sequence) -> None:
        """Record the sequence's worst-case block budget in the ledger.

        Called at admission, after ``can_admit`` passed. The reservation is
        drawn down as blocks materialize (``allocate``/``append``) and any
        remainder is released by ``free``.
        """
        if sequence.sequence_id in self.reserved:
            raise ValueError(
                f"Sequence {sequence.sequence_id} already holds a reservation; "
                "free it before re-reserving"
            )
        blocks = self.worst_case_blocks(sequence)
        self.reserved[sequence.sequence_id] = blocks
        self._total_reserved += blocks

    def _consume_reservation(self, sequence_id: str, num_blocks: int) -> None:
        """Draw down a reservation as physical blocks materialize.

        A reserved block that has been popped into the block_table is no longer
        "promised future use" — it is held — so it leaves the ledger. The min()
        floor makes this a no-op for unreserved sequences (e.g. tests that call
        allocate directly without going through admission).
        """
        used = min(self.reserved.get(sequence_id, 0), num_blocks)
        if used:
            self.reserved[sequence_id] -= used
            self._total_reserved -= used

    def can_ever_allocate(self, sequence: Sequence) -> bool:
        """Whether the request could ever fit in the cache, ignoring current usage.

        Unlike ``can_admit``, this is checked against the cache's total capacity,
        so a request whose worst-case footprint can never be scheduled (no
        matter how many blocks free up) is detected up front.
        """
        return self.worst_case_blocks(sequence) <= self.total_blocks

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
        self._consume_reservation(sequence.sequence_id, num_blocks_needed)

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
        self._consume_reservation(sequence.sequence_id, additional)

    def free(self, sequence: Sequence) -> None:
        """
        Clears the blocks allocated to the sequence and returns them to the free pool.

        Freeing a sequence that was already freed (or never allocated) is a
        safe no-op: ``allocated_blocks.pop`` returns an empty set the second
        time, so nothing is pushed back. A ``RuntimeError`` is raised if any
        of the sequence's blocks are already in the free pool — that can only
        happen if a physical block was allocated to two sequences at once,
        which would alias the KV cache.
        """
        allocated = self.allocated_blocks.pop(sequence.sequence_id, set())
        if allocated:
            already_free = set(self.free_blocks) & allocated
            if already_free:
                raise RuntimeError(
                    f"Block(s) {sorted(already_free)} freed for sequence "
                    f"{sequence.sequence_id} are already in the free pool; this "
                    "indicates a double-allocation that would alias the KV cache"
                )
            for block_id in allocated:
                heapq.heappush(self.free_blocks, block_id)

        # Release whatever budget the sequence never materialized (e.g. it hit
        # EOS early, or was evicted while still waiting). Popping keeps free()
        # idempotent for the ledger too.
        self._total_reserved -= self.reserved.pop(sequence.sequence_id, 0)

        sequence.block_table = []
