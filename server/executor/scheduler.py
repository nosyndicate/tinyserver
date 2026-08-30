import logging
from collections import deque

from server.executor.types import (
    ScheduledBatch,
    Sequence,
    SequenceBatchTask,
    SequenceState,
)
from server.model.block_manager import BlockManager

logger = logging.getLogger(__name__)


def validate_scheduler_config(max_num_sequences: int, max_num_tokens: int) -> None:
    """Validate the coupled sequence and token budgets."""
    if max_num_sequences <= 0:
        raise ValueError("max_num_sequences must be positive")
    if max_num_tokens < max_num_sequences:
        raise ValueError(
            "max_num_tokens must be greater than or equal to max_num_sequences"
        )


class Scheduler:
    def __init__(
        self,
        block_manager: BlockManager,
        max_num_sequences: int,
        max_num_tokens: int,
    ) -> None:
        validate_scheduler_config(max_num_sequences, max_num_tokens)

        self.block_manager = block_manager
        self.max_num_sequences = max_num_sequences
        self.max_num_tokens = max_num_tokens

        # the scheduler's population consists of sequences in either the waiting
        # queue or the running list. The total number of sequences in both should
        # never exceed max_num_sequences.
        # The waiting queue holds sequences that are waiting to be scheduled.
        self.waiting: deque[Sequence] = deque()
        # The running list holds sequences that are currently running.
        self.running: list[Sequence] = []

        # Scheduling-decision counters used to distinguish memory pressure from
        # logical-slot exhaustion under load.
        self.preemption_count = 0
        self.reservation_blocked_count = 0

    def has_sequence_capacity(self) -> bool:
        """Whether a new request can enter the scheduler-owned population."""
        return len(self.waiting) + len(self.running) < self.max_num_sequences

    def add(self, sequence: Sequence) -> None:
        """Admit a new sequence from the worker queue."""
        if not self.has_sequence_capacity():
            raise RuntimeError("scheduler sequence capacity exceeded")
        self.waiting.append(sequence)

    def _assert_population_invariant(self) -> None:
        population = len(self.waiting) + len(self.running)
        if population > self.max_num_sequences:
            raise RuntimeError(
                "scheduler population exceeds max_num_sequences: "
                f"{population} > {self.max_num_sequences}"
            )

    def clear(self) -> None:
        """
        Clear all sequences from the scheduler, freeing their allocated blocks.
        """
        for seq in self.running:
            self.block_manager.free(seq)

        self.running.clear()
        self.waiting.clear()

    def reap_finished(self) -> None:
        """Free blocks of finished sequences and drop them from running.

        Finished sequences are detected by the engine (EOS / max-len), which
        sets ``seq.finished = True``. Freeing their blocks is the scheduler's
        job, so it lives here. Public and idempotent so the engine can expose
        released slots before draining inbound work; ``schedule()`` also calls
        it defensively for direct scheduler users.
        """
        remain_running = []
        for seq in self.running:
            if seq.finished:
                self.block_manager.free(seq)
            else:
                remain_running.append(seq)
        self.running = remain_running

        # A cancelled sequence may also be sitting in `waiting` — either
        # PREEMPTED (blocks already freed by `_preempt`) or admitted-but-not-yet
        # prefilled (never allocated). Drop them so the prefill pass doesn't
        # re-schedule and re-prefill a cancelled request. No free() needed: they
        # hold no blocks in either case.
        if any(seq.finished for seq in self.waiting):
            self.waiting = deque(seq for seq in self.waiting if not seq.finished)

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
        population_before = len(self.waiting) + len(self.running)
        victim = self.running.pop()  # youngest == last appended
        self.block_manager.free(victim)
        victim.state = SequenceState.PREEMPTED
        victim.num_tokens = len(victim.prompt_token_ids) + len(
            victim.generated_token_ids
        )
        self.waiting.appendleft(victim)
        population_after = len(self.waiting) + len(self.running)
        if population_after != population_before:
            raise RuntimeError("preemption changed scheduler population")
        self.preemption_count += 1
        logger.debug(
            "preempted %s (count=%d)", victim.sequence_id, self.preemption_count
        )
        return victim

    @staticmethod
    def _needs_decode_after_prefill(sequence: Sequence) -> bool:
        """Whether a full prefill/recompute can be followed by decode."""
        outputs_after_prefill = len(sequence.generated_token_ids) + 1
        return outputs_after_prefill < sequence.max_new_tokens

    @staticmethod
    def _needs_next_decode(sequence: Sequence) -> bool:
        """Whether an already-running sequence still needs a decode call."""
        return (
            not sequence.finished
            and len(sequence.generated_token_ids) < sequence.max_new_tokens
        )

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
        self.reap_finished()
        self._assert_population_invariant()

        scheduled: list[Sequence] = []
        resumed_ids: set[str] = set()
        total_tokens = 0
        decode_reserve = sum(
            self.block_manager.additional_blocks_required(seq, extra_tokens=1)
            for seq in self.running
            if self._needs_next_decode(seq)
        )
        # No separate batch-width bound is needed: every scheduled sequence
        # moves from ``waiting`` into ``running``, so the population cap below
        # already bounds how wide this batch can get.
        while self.waiting and len(self.running) < self.max_num_sequences:
            seq_to_add = self.waiting[0]
            # Allow a single oversized sequence through when the batch is still
            # empty; otherwise it would block the whole queue forever.
            enough_budget = (
                not scheduled
                or seq_to_add.num_tokens + total_tokens <= self.max_num_tokens
            )

            if enough_budget:
                candidate_extra = (
                    1 if self._needs_decode_after_prefill(seq_to_add) else 0
                )
                candidate_total = self.block_manager.blocks_required_to_allocate(
                    seq_to_add, extra_tokens=candidate_extra
                )
                required = candidate_total + decode_reserve
                enough_memory = required <= self.block_manager.num_free_blocks
            else:
                enough_memory = False

            if enough_memory:
                seq = self.waiting.popleft()
                # If this is a resumed sequence, mark it
                if seq.state == SequenceState.PREEMPTED:
                    resumed_ids.add(seq.sequence_id)
                self.block_manager.allocate(seq)
                seq.state = SequenceState.RUNNING
                self.running.append(seq)
                scheduled.append(seq)
                total_tokens += seq.num_tokens
                if candidate_extra:
                    decode_reserve += self.block_manager.additional_blocks_required(
                        seq, extra_tokens=1
                    )
            else:
                if enough_budget:
                    self.reservation_blocked_count += 1
                    logger.debug(
                        "prefill reservation blocked %s "
                        "(required=%d free=%d decode_reserve=%d count=%d)",
                        seq_to_add.sequence_id,
                        required,
                        self.block_manager.num_free_blocks,
                        decode_reserve,
                        self.reservation_blocked_count,
                    )
                break

        if scheduled:
            self._assert_population_invariant()
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
        # As in the prefill pass, batch width needs no bound of its own: the
        # population invariant caps ``running``, and this walks it at most once.
        while idx < len(self.running):
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
            self._assert_population_invariant()
            return ScheduledBatch(kind=SequenceBatchTask.DECODE, sequences=scheduled)

        self._assert_population_invariant()
        return None
