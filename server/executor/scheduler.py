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

    def can_add_new_sequence(self, sequence: Sequence) -> bool:
        """
        Check if we can add ``sequence`` to the scheduler: room in the waiting
        queue AND the block manager can reserve the sequence's worst-case
        footprint (prompt + max_new_tokens) right now.

        Checked against the full generation budget rather than the prompt alone
        so we don't admit a request that prefills fine but wedges on a later
        decode step when no free block remains.
        """
        if len(self.waiting) >= self.max_waiting:
            return False
        return self.block_manager.can_admit(sequence)

    def admission_headroom(self) -> int:
        """Number of free slots in the waiting queue.

        Used by the engine to bound how many not-yet-admitted requests it
        holds in total (its private pending buffer plus this waiting queue),
        so that overload backs up into the bounded inbound queue and surfaces
        as 503s instead of growing an unbounded buffer.
        """
        return max(0, self.max_waiting - len(self.waiting))

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

    def abort_youngest_running(self) -> Sequence | None:
        """Evict the most recently started running sequence, freeing its blocks.

        A liveness backstop for the residual multi-sequence race: two requests
        can each pass ``can_admit`` against the free pool without reserving the
        other's budget, both start, both grow, and collide with no free blocks
        left. The engine detects that wedge and calls this to break it. The
        youngest sequence is picked because it is furthest from finishing (the
        older ones are closer to freeing their own blocks). Returns the evicted
        sequence, or ``None`` if nothing is running. Surfacing an error for the
        victim is the caller's job — this only reclaims the blocks.
        """
        if not self.running:
            return None
        victim = self.running.pop()
        self.block_manager.free(victim)
        return victim

    def _reap_finished(self) -> None:
        """Free blocks of finished sequences and drop them from running.

        Finished sequences are detected by the engine (EOS / max-len), which
        sets ``seq.finished = True``. Freeing their blocks is the scheduler's
        job, so it lives here. Called at the top of every ``schedule()`` so
        blocks are reclaimed promptly regardless of which phase runs next.
        """
        remain_running = []
        for seq in self.running:
            if seq.finished:
                self.block_manager.free(seq)
            else:
                remain_running.append(seq)
        self.running = remain_running

    def schedule(self) -> ScheduledBatch | None:
        """
        Decide which sequences to run next.

        Reaps finished sequences first, then applies a simple policy:
        prioritize prefill so new requests get TTFT, falling back to decode.

        Capacity contract: the scheduler reserves block capacity for the token
        the engine is about to generate (``extra_tokens=1``) but does NOT
        advance ``num_tokens`` — the engine does that after producing each
        token, and sets ``finished`` when generation ends.

        TODO: consider more sophisticated policies (preemption, decoding first).
        """
        self._reap_finished()

        scheduled: list[Sequence] = []
        total_tokens = 0
        while self.waiting and len(scheduled) < self.max_num_sequences:
            seq_to_add = self.waiting[0]
            # Worst-case check: free blocks shrink as each prompt is allocated in
            # this loop, so re-checking prompt + max_new_tokens here stops a
            # sequence from being started when it could not decode past prefill.
            enough_memory = self.block_manager.can_admit(seq_to_add)
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

        scheduled = []
        total_tokens = 0
        for seq_to_add in self.running:
            if len(scheduled) >= self.max_num_sequences:
                break

            # Reserve a block for the token the engine is about to generate.
            # The engine advances num_tokens after producing it; the scheduler
            # only ensures the capacity exists here, so it never mutates
            # num_tokens and there is no rollback to undo.
            if not self.block_manager.can_append(seq_to_add, extra_tokens=1):
                # Not enough memory to decode this one; skip it for now and
                # wait for other sequences to finish and free up blocks.
                # TODO: consider preemption or other strategies here.
                continue

            # See if we can at least decode one more token for this sequence.
            if 1 + total_tokens > self.max_num_tokens:
                # No budget this round, simply break.
                break

            self.block_manager.append(seq_to_add, extra_tokens=1)
            scheduled.append(seq_to_add)
            total_tokens += 1

        if scheduled:
            return ScheduledBatch(kind=SequenceBatchTask.DECODE, sequences=scheduled)

        return None
