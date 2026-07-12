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

        # Number of preemptions performed over the scheduler's lifetime. A high
        # rate signals the watermark is too small or the pool too tight; full
        # observability (logging/metrics) is a separate piece of work.
        self.preemption_count = 0

    def can_add_new_sequence(self, sequence: Sequence) -> bool:
        """
        Check if we can add ``sequence`` to the scheduler: room in the waiting
        queue AND the block manager can allocate the sequence's prompt while
        leaving a small free-block headroom. The headroom is a thrash guard.

        Feasibility (worst-case prompt + max_new_tokens fitting the cache at all)
        is rejected earlier, in the engine's fail-fast on ``can_ever_allocate``.
        """
        if len(self.waiting) >= self.max_waiting:
            return False

        headroom = max(1, int(0.01 * self.block_manager.total_blocks))
        return self.block_manager.can_allocate_with_headroom(sequence, headroom)

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

    def _preempt(self) -> Sequence:
        """Evict the youngest running sequence to free blocks for a
        higher-priority one, returning the evicted sequence.

        Blocks are freed immediately (recompute-based preemption: there is no
        KV swap to CPU). The victim keeps its ``generated_token_ids`` and
        re-enters at the FRONT of ``waiting``, so it resumes before any newer
        waiting request. Preempt-youngest + requeue-at-front keeps priority
        order stable, so the oldest sequence always makes progress and the
        eviction loop terminates.

        Only the tail of ``running`` is ever evicted (``pop()``), so callers
        must ensure the youngest sequence is a valid victim (not one already
        scheduled this round).
        """
        victim = self.running.pop()  # youngest == last appended
        self.block_manager.free(victim)
        victim.state = SequenceState.PREEMPTED
        victim.num_tokens = len(victim.prompt_token_ids) + len(
            victim.generated_token_ids
        )
        self.waiting.appendleft(victim)
        self.preemption_count += 1
        return victim

    def schedule(self) -> ScheduledBatch | None:
        """
        Decide which sequences to run next.

        Reaps finished sequences first, then applies a simple policy:
        prioritize prefill so new requests get TTFT, falling back to decode.
        Under memory pressure the decode phase preempts the youngest running
        sequence (see ``_preempt``) so the oldest always makes progress.

        Capacity contract: the scheduler reserves block capacity for the token
        the engine is about to generate (``extra_tokens=1``) but does NOT
        advance ``num_tokens`` — the engine does that after producing each
        token, and sets ``finished`` when generation ends.
        """
        self._reap_finished()

        scheduled: list[Sequence] = []
        resumed_ids: set[str] = set()
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
                # If this is a resumed sequence, mark it
                if seq.state == SequenceState.PREEMPTED:
                    resumed_ids.add(seq.sequence_id)
                self.block_manager.allocate(seq)
                seq.state = SequenceState.RUNNING
                self.running.append(seq)
                scheduled.append(seq)
                total_tokens += seq.num_tokens
            else:
                break

        if scheduled:
            return ScheduledBatch(
                kind=SequenceBatchTask.PREFILL,
                sequences=scheduled,
                resumed_sequence_ids=frozenset(resumed_ids),
            )

        scheduled = []
        total_tokens = 0
        # Index-based walk because we mutate ``running`` while iterating: making
        # room for a sequence may preempt (pop) younger sequences off the tail.
        # We only ever pop the tail, and ``seq_to_add = running[i]`` is never
        # the tail while a younger sequence remains, so the current sequence is
        # never popped and ``i`` stays valid as the tail shrinks.
        idx = 0
        while idx < len(self.running):
            if len(scheduled) >= self.max_num_sequences:
                break

            seq_to_add = self.running[idx]

            # See if we can at least decode one more token for this sequence.
            # If no, don't bother try to do preemption of other sequences.
            if 1 + total_tokens > self.max_num_tokens:
                # No budget this round, simply break.
                break

            # Reserve a block for the token the engine is about to generate.
            # The engine advances num_tokens after producing it; the scheduler
            # only ensures the capacity exists here, so it never mutates
            # num_tokens and there is no rollback to undo.
            #
            # If there aren't enough blocks, preempt the youngest not-yet-
            # scheduled sequence (the tail) and retry, until seq_to_add fits or
            # it IS the youngest (nothing younger left to evict). The
            # already-scheduled sequences occupy running[0:i] and have had
            # append() called, so they must never be evicted — we only pop the
            # tail, which is always in the not-yet-scheduled range running[i:].
            while not self.block_manager.can_append(seq_to_add, extra_tokens=1):
                if self.running[-1] is seq_to_add:
                    break
                self._preempt()

            if not self.block_manager.can_append(seq_to_add, extra_tokens=1):
                # seq_to_add is the youngest remaining and still doesn't fit; it
                # keeps its KV and waits for other sequences to free blocks.
                # Every later sequence is younger, so none of them fit either.
                break

            self.block_manager.append(seq_to_add, extra_tokens=1)
            scheduled.append(seq_to_add)
            total_tokens += 1
            idx += 1

        if scheduled:
            return ScheduledBatch(kind=SequenceBatchTask.DECODE, sequences=scheduled)

        return None
