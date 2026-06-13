from collections import deque

from server.executor.types import (
    ScheduledBatch,
    Sequence,
    SequenceBatchTask,
    SequenceState,
)
from server.model.block_manager import BlockManager


class Scheduler:
    def __init__(
        self,
        block_manager: BlockManager,
        max_waiting: int,
        max_num_sequences: int,
        max_num_tokens: int,
    ) -> None:
        self.block_manager = block_manager
        self.max_waiting = max_waiting
        self.max_num_sequences = max_num_sequences
        self.max_num_tokens = max_num_tokens

        # The waiting queue holds sequences that are waiting to be scheduled.
        self.waiting: deque[Sequence] = deque()
        # The running list holds sequences that are currently running.
        self.running: list[Sequence] = []

    def can_add_new_sequence(self) -> bool:
        """
        Check if we can add a new sequence to the scheduler.
        This is true if we have enough capacity in the waiting list
        and the block manager can allocate a new sequence.
        """
        if len(self.waiting) >= self.max_waiting:
            return False

        # We check the block manager here to avoid adding too many sequences
        # to the waiting list and then find out later that we can't even start them.
        # This is a simple heuristic, in the future we might want to consider
        # more sophisticated strategies to handle this situation, such as preemption
        # or dynamic adjustment of max_waiting.
        return self.block_manager.has_free_blocks_for(1)

    def add(self, sequence: Sequence) -> None:
        self.waiting.append(sequence)

    def clear(self) -> None:
        """
        Clear all sequences from the scheduler, freeing their allocated blocks.
        """
        for seq in self.running:
            self.block_manager.free(seq)

        self.running.clear()
        self.waiting.clear()

    def schedule(self) -> ScheduledBatch | None:
        """
        Decide which sequences to run next.
        Simple policy: prioritize prefill so new requests get TTFT.

        TODO: consider adding more sophisticated policies that take into account other factors,
        also, consider preemption and do decoding first
        """
        scheduled: list[Sequence] = []
        total_tokens = 0
        while self.waiting and len(scheduled) < self.max_num_sequences:
            seq_to_add = self.waiting[0]
            enough_memory = self.block_manager.can_allocate(seq_to_add)
            # Allow a single oversized sequence through when the batch is still
            # empty; otherwise it would block the whole queue forever.
            enough_budget = (
                not scheduled
                or seq_to_add.num_tokens + total_tokens <= self.max_num_tokens
            )

            if enough_memory and enough_budget:
                seq = self.waiting.popleft()
                self.block_manager.allocate(seq)
                seq.state = SequenceState.RUNNING
                self.running.append(seq)
                scheduled.append(seq)
                total_tokens += seq.num_tokens
            else:
                break

        if scheduled:
            return ScheduledBatch(kind=SequenceBatchTask.PREFILL, sequences=scheduled)

        # Before pick a batch of sequences for decoding, check if any of the sequence is already finished.
        # If so, free the blocks and remove it from running.
        # This part of logic might need to be move into a separate method and called at the end of processing
        # one batch of sequences.
        remain_running = []
        for seq in self.running:
            if seq.finished:
                self.block_manager.free(seq)
            else:
                remain_running.append(seq)

        self.running = remain_running

        scheduled = []
        total_tokens = 0
        for seq_to_add in self.running:
            if len(scheduled) >= self.max_num_sequences:
                break

            # Reserve a slot for the token about to be generated. The executor
            # fills this slot and must NOT increment num_tokens again for it.
            seq_to_add.num_tokens += 1

            # If we don't have enough memory to decode, skip for now and
            # wait for other sequences to finish and free up memory. In the future, we can consider
            # preemption or other strategies to handle this situation.
            if not self.block_manager.can_append(seq_to_add):
                seq_to_add.num_tokens -= 1
                continue

            # See if we can at least decode one more token for this sequence.
            if 1 + total_tokens > self.max_num_tokens:
                # no budget this round, simply break
                seq_to_add.num_tokens -= 1
                break

            self.block_manager.append(seq_to_add)
            scheduled.append(seq_to_add)
            total_tokens += 1

        if scheduled:
            return ScheduledBatch(kind=SequenceBatchTask.DECODE, sequences=scheduled)

        return None
