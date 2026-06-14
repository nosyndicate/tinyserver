import logging
import uuid
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable, Protocol, runtime_checkable

import torch

from server.executor.events import RequestEventEmitter
from server.executor.scheduler import Scheduler
from server.executor.types import (
    BaseBatchExecutor,
    BaseExecutor,
    BatchEngineConfig,
    DecodeResult,
    EngineConfig,
    FinishReason,
    GenerationRequestState,
    PrefillResult,
    RequestContext,
    RequestFailure,
    RequestStatus,
    Sequence,
    SequenceBatchTask,
)
from server.metrics.timers import now_ns
from server.model.inference_context import InferenceContext, inference_context
from server.model.prefill_helpers import build_prefill_inputs
from server.model.sampling import sample_token
from server.model.types import ModelBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineControl:
    """Handle for the runner loop to check shutdown and idle conditions.

    Attributes:
        should_stop: Returns True when the engine should terminate.
        wait_idle: Called when no work is available; returns True if shutdown was requested during the wait.
    """

    should_stop: Callable[[], bool]
    wait_idle: Callable[[float], bool]


@dataclass(frozen=True)
class EngineCallbacks:
    """Callbacks from the engine to the surrounding worker/runtime.

    Attributes:
        cancel_request: Marks a single request as failed with an error message.
        handle_fatal_error: Propagates an irrecoverable error; optionally receives the batch of requests that were being processed.
    """

    cancel_request: Callable[[GenerationRequestState, str], None]
    handle_fatal_error: Callable[[Exception, list[GenerationRequestState] | None], None]


@runtime_checkable
class InferenceEngine(Protocol):
    """Protocol that any inference engine must implement.

    An engine drives the request lifecycle: draining an inbound queue,
    running prefill/decode phases on an executor, and emitting events.
    """

    def run(
        self,
        inbound: Queue[GenerationRequestState],
        control: EngineControl,
        callbacks: EngineCallbacks,
    ) -> None:
        """Main event loop: process requests until shutdown."""

    def cancel_inflight(
        self,
        message: str,
        cancel_request: Callable[[GenerationRequestState, str], None],
    ) -> None:
        """Cancel all in-flight requests with the given error message."""


class SimpleInferenceEngine:
    """Process requests one at a time through prefill, then decode one-by-one.

    Drains the inbound queue up to `max_active_requests`, runs prefill
    sequentially, then iterates over all active decode requests individually.
    Suitable for workloads where batching provides little benefit.
    """

    def __init__(self, executor: BaseExecutor, config: EngineConfig) -> None:
        if config.max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive")
        self._executor = executor
        self._config = config
        self._active: list[GenerationRequestState] = []
        self._emitter = RequestEventEmitter()

    def cancel_inflight(
        self,
        message: str,
        cancel_request: Callable[[GenerationRequestState, str], None],
    ) -> None:
        for pending in self._active:
            try:
                cancel_request(pending, message)
            except Exception:
                logger.exception(
                    "Failed to emit error event for active request %s",
                    pending.request_id,
                )
        self._active.clear()

    def _cancel_requests(
        self,
        requests: list[GenerationRequestState],
        message: str,
        cancel_request: Callable[[GenerationRequestState, str], None],
        phase: str,
    ) -> None:
        """Cancel a list of requests, logging failures with phase context."""
        for pending in requests:
            try:
                cancel_request(pending, message)
            except Exception:
                logger.exception(
                    "Failed to clean up request %s during %s",
                    pending.request_id,
                    phase,
                )

    def run(
        self,
        inbound: Queue[GenerationRequestState],
        control: EngineControl,
        callbacks: EngineCallbacks,
    ) -> None:
        try:
            while not control.should_stop():
                new_requests: list[GenerationRequestState] = []
                while (
                    len(self._active) + len(new_requests)
                    < self._config.max_active_requests
                ):
                    try:
                        request_state = inbound.get_nowait()
                        new_requests.append(request_state)
                    except Empty:
                        break

                for i, req in enumerate(new_requests):
                    if control.should_stop():
                        self._cancel_requests(
                            new_requests[i:] + self._active,
                            "Worker is shutting down, request cancelled",
                            callbacks.cancel_request,
                            "prefill",
                        )
                        self._active.clear()
                        return

                    try:
                        result = self._executor.prefill(req)
                    except Exception as e:
                        callbacks.handle_fatal_error(e, new_requests[i:])
                        return
                    if isinstance(result, RequestFailure):
                        self._emitter.on_failed(req, result.error)
                    elif isinstance(result, PrefillResult):
                        self._emitter.on_prefill_started(req, result.start_ns)
                        self._emitter.on_prefill_succeeded(req, result)
                    if req.status == RequestStatus.DECODING:
                        self._active.append(req)

                for req in self._active:
                    if control.should_stop():
                        self._cancel_requests(
                            self._active,
                            "Worker is shutting down, request cancelled",
                            callbacks.cancel_request,
                            "decoding",
                        )
                        self._active.clear()
                        return

                    if req.status == RequestStatus.DECODING:
                        result = self._executor.decode(req)
                        if isinstance(result, RequestFailure):
                            self._emitter.on_failed(req, result.error)
                        elif isinstance(result, DecodeResult):
                            self._emitter.on_token(req, result)

                self._active = [
                    req for req in self._active if req.status == RequestStatus.DECODING
                ]

                if not self._active:
                    control.wait_idle(0.01)
        except Exception as e:
            callbacks.handle_fatal_error(e, None)


class BatchInferenceEngine:
    """Process requests in batches for both prefill and decode phases.

    Drains the inbound queue into a waiting list, then groups waiting
    requests into prefill batches (up to `max_prefill_batch_size`) and
    active decode requests into decode batches (up to `max_decode_batch_size`).
    Reduces per-request overhead when many requests share the same model.
    """

    def __init__(self, executor: BaseBatchExecutor, config: BatchEngineConfig) -> None:
        if config.max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive")
        if config.max_prefill_batch_size <= 0:
            raise ValueError("max_prefill_batch_size must be positive")
        if config.max_decode_batch_size <= 0:
            raise ValueError("max_decode_batch_size must be positive")
        if config.max_prefill_batch_size > config.max_active_requests:
            raise ValueError(
                "max_prefill_batch_size cannot be greater than max_active_requests"
            )
        if config.max_decode_batch_size > config.max_active_requests:
            raise ValueError(
                "max_decode_batch_size cannot be greater than max_active_requests"
            )
        self._executor = executor
        self._config = config
        self._waiting: list[GenerationRequestState] = []
        self._active: list[GenerationRequestState] = []
        self._emitter = RequestEventEmitter()

    def cancel_inflight(
        self,
        message: str,
        cancel_request: Callable[[GenerationRequestState, str], None],
    ) -> None:
        for pending in self._waiting + self._active:
            try:
                cancel_request(pending, message)
            except Exception:
                logger.exception(
                    "Failed to emit error event for request %s",
                    pending.request_id,
                )
        self._waiting.clear()
        self._active.clear()

    def drain_inbound(self, inbound: Queue[GenerationRequestState]) -> None:
        max_num_reqs = self._config.max_active_requests
        while len(self._waiting) + len(self._active) < max_num_reqs:
            try:
                req = inbound.get_nowait()
                self._waiting.append(req)
            except Empty:
                break

    def select_prefill_batch(self) -> list[GenerationRequestState]:
        batch_size = min(
            self._config.max_prefill_batch_size,
            self._config.max_active_requests - len(self._active),
            len(self._waiting),
        )
        batch = self._waiting[:batch_size]
        self._waiting = self._waiting[batch_size:]
        return batch

    def select_decode_batch(self) -> list[GenerationRequestState]:
        decoding = [req for req in self._active if req.status == RequestStatus.DECODING]
        return decoding[: self._config.max_decode_batch_size]

    def run(
        self,
        inbound: Queue[GenerationRequestState],
        control: EngineControl,
        callbacks: EngineCallbacks,
    ) -> None:
        prefill_batch: list[GenerationRequestState] = []
        try:
            while not control.should_stop():
                self.drain_inbound(inbound)

                if not control.should_stop():
                    prefill_batch = self.select_prefill_batch()
                    if prefill_batch:
                        results = self._executor.batched_prefill(prefill_batch)
                        if len(results) != len(prefill_batch):
                            raise ValueError(
                                f"Expected {len(prefill_batch)} prefill results, but got {len(results)}"
                            )
                        for req, result in zip(prefill_batch, results):
                            if isinstance(result, RequestFailure):
                                self._emitter.on_failed(req, result.error)
                            elif isinstance(result, PrefillResult):
                                self._emitter.on_prefill_started(req, result.start_ns)
                                self._emitter.on_prefill_succeeded(req, result)
                        self._active.extend(
                            req
                            for req in prefill_batch
                            if req.status == RequestStatus.DECODING
                        )
                        prefill_batch = []

                if not control.should_stop():
                    decoding_batch = self.select_decode_batch()
                    if decoding_batch:
                        results = self._executor.batched_decode(decoding_batch)
                        if len(results) != len(decoding_batch):
                            raise ValueError(
                                f"Expected {len(decoding_batch)} decode results, but got {len(results)}"
                            )
                        for req, result in zip(decoding_batch, results):
                            if isinstance(result, RequestFailure):
                                self._emitter.on_failed(req, result.error)
                            elif isinstance(result, DecodeResult):
                                self._emitter.on_token(req, result)

                self._active = [
                    req for req in self._active if req.status == RequestStatus.DECODING
                ]

                if not self._waiting and not self._active:
                    control.wait_idle(0.01)
        except Exception as e:
            callbacks.handle_fatal_error(e, prefill_batch)


class ScheduleInferenceEngine:
    """
    A ScheduledWorker implementation that uses a scheduler to decide which requests to process next.
    The worker interacts with the scheduler to get the next batch of requests to process, and then
    calls the executor to process them.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        backend: ModelBackend,
    ) -> None:
        self._backend = backend
        self._scheduler = scheduler
        self._device = backend.device
        self._all_requests: dict[str, RequestContext] = {}
        # Map sequence_id -> request state, since the scheduler hands back
        # ``Sequence`` objects and post-processing needs the request (for
        # sampling params, generator, and event emission).
        self._seq_to_request: dict[str, GenerationRequestState] = {}
        self._emitter = RequestEventEmitter()

    def _drain_inbound(self, inbound: Queue[GenerationRequestState]) -> None:
        """
        Move requests from the inbound queue to the scheduler's waiting list.
        Convert the GenerationRequestState to Sequence and add to the scheduler.
        """
        need_to_wait = []

        # TODO The current implementation might require us to tokenize the prompt
        # several times if GPU memory is not enough to schedule the request.
        # Consider a better solution.
        while True:
            try:
                req = inbound.get_nowait()
                seq = self._make_sequence(req)
                if not self._scheduler.block_manager.can_ever_allocate(seq):
                    # The prompt can never fit in the KV cache, no matter how
                    # many blocks free up. Fail it once instead of re-queuing
                    # (and re-tokenizing) it on every loop iteration forever.
                    self._emitter.on_failed(
                        req,
                        f"prompt of {seq.num_tokens} tokens exceeds cache capacity of "
                        f"{self._scheduler.block_manager.total_blocks * self._scheduler.block_manager.block_size} tokens",
                    )
                    continue
                if self._scheduler.can_add_new_sequence(seq):
                    self._all_requests[req.request_id] = RequestContext(
                        request=req, sequence=seq
                    )
                    self._seq_to_request[seq.sequence_id] = req
                    self._scheduler.add(seq)
                else:
                    # Collect requests that can't be scheduled right now to put them back to the inbound queue later,
                    # so they can be retried in the next loop iteration.
                    need_to_wait.append(req)
                    # if the scheduler's waiting queue is full, it's unlikely that the next requests can be
                    # scheduled, so break early to avoid unnecessary retries.
                    # otherwise, if the scheduler can accept new sequences but the current one just can't be
                    # scheduled due to block manager capacity, we can continue to try the next requests
                    # which might be schedulable.
                    if self._scheduler.waiting_queue_is_full():
                        break
                    else:
                        continue
            except Empty:
                break

        # Put back the requests that can't be scheduled right now to the inbound queue for later processing
        for req in need_to_wait:
            inbound.put(req)

    def _make_sequence(self, req: GenerationRequestState) -> Sequence:
        token_ids = self._backend.tokenize(req.prompt)
        sequence_id = str(uuid.uuid4())
        return Sequence(
            sequence_id=sequence_id,
            prompt_token_ids=token_ids,
            generated_token_ids=[],
            num_prompt_tokens=len(token_ids),
            num_tokens=len(token_ids),
            block_table=[],
        )

    def cancel_inflight(
        self,
        message: str,
        cancel_request: Callable[[GenerationRequestState, str], None],
    ) -> None:
        """Cancel all in-flight requests with the given error message.

        Asks the scheduler to clear its waiting/running sequences (freeing
        their blocks) and emits an error event for each tracked request via
        the supplied ``cancel_request`` callback.
        """
        self._scheduler.clear()
        for context in self._all_requests.values():
            try:
                cancel_request(context.request, message)
            except Exception:
                logger.exception(
                    "Failed to emit error event for request %s",
                    context.request.request_id,
                )
        self._all_requests.clear()
        self._seq_to_request.clear()

    def _prepare_prefill(
        self, sequences: list[Sequence]
    ) -> tuple[torch.Tensor, torch.Tensor, InferenceContext]:
        """Prepare the inputs and context for prefill given a list of sequences."""
        seq_token_lists: list[list[int]] = []
        block_tables: list[list[int]] = []
        for seq in sequences:
            seq_token_lists.append(seq.prompt_token_ids)
            block_tables.append(seq.block_table)

        input_ids, position_ids, ctx = build_prefill_inputs(
            seq_token_lists, block_tables, device=self._device
        )
        return input_ids, position_ids, ctx

    def _prepare_decode(
        self, sequences: list[Sequence]
    ) -> tuple[torch.Tensor, torch.Tensor, InferenceContext]:
        """Prepare the inputs and context for a decode step.

        Each sequence contributes exactly one token (the last generated one),
        which the patched attention stores into the paged KV cache at its
        current position (``seq.num_tokens``) before attending over the full
        sequence. See ``build_prefill_inputs`` for the prefill counterpart.
        """
        input_ids = torch.tensor(
            [[seq.generated_token_ids[-1] for seq in sequences]],
            dtype=torch.long,
            device=self._device,
        )
        # position_id is the absolute position of the token being stored, which
        # equals the number of tokens already committed to the cache.
        position_ids = torch.tensor(
            [[seq.num_tokens for seq in sequences]],
            dtype=torch.long,
            device=self._device,
        )
        # The decode attention branch only reads ``block_table`` from each
        # sequence entry (positions come from ``position_ids``), so num_tokens
        # is not needed here, unlike the prefill context.
        ctx = InferenceContext(
            mode="decode",
            sequences=[{"block_table": seq.block_table} for seq in sequences],
        )
        return input_ids, position_ids, ctx

    def run(
        self,
        inbound: Queue[GenerationRequestState],
        control: EngineControl,
        callbacks: EngineCallbacks,
    ) -> None:
        try:
            while not control.should_stop():
                # Move new requests from the inbound queue to the waiting list of scheduler
                # and let scheduler decide which sequences to run next.
                self._drain_inbound(inbound)

                if not control.should_stop():
                    batch = self._scheduler.schedule()

                    if batch is None:
                        # No sequences are scheduled, sleep for a short while to avoid busy loop. Use the sleep on _shutdown_event so when shutdown
                        # signal is set, it can break the sleep immediately and exit the loop.
                        control.wait_idle(0.01)
                        continue

                    start_ns = now_ns()
                    if batch.kind == SequenceBatchTask.PREFILL:
                        input_ids, position_ids, ctx = self._prepare_prefill(
                            batch.sequences
                        )
                    elif batch.kind == SequenceBatchTask.DECODE:
                        input_ids, position_ids, ctx = self._prepare_decode(
                            batch.sequences
                        )
                    else:
                        raise ValueError(f"Unknown batch kind: {batch.kind}")

                    with torch.inference_mode(), inference_context(ctx):
                        out = self._backend.model(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            use_cache=False,
                        )

                    if batch.kind == SequenceBatchTask.PREFILL:
                        self._post_prefill(out, batch.sequences, start_ns)
                    else:
                        self._post_decode(out, batch.sequences)

            # Graceful shutdown: cancel any remaining waiting and active requests
            self.cancel_inflight(
                "Worker is shutting down, request cancelled",
                callbacks.cancel_request,
            )
        except Exception as e:
            callbacks.handle_fatal_error(e, None)

    def _sample_one(
        self, logits: torch.Tensor, request_state: GenerationRequestState
    ) -> DecodeResult:
        """Sample a single token from a ``[1, vocab]`` logit row.

        Mirrors ``Executor._sample`` but operates on a pre-sliced logit row,
        since this engine samples directly from the batched model output
        rather than from per-request ``all_logits`` (the KV cache lives in the
        paged blocks, not on the request state).
        """
        token_id = sample_token(
            logits, request_state.sampling_params, request_state.generator
        )
        if token_id == self._backend.tokenizer.eos_token_id:
            return DecodeResult(
                token_id=token_id, token="", finish_reason=FinishReason.EOS
            )
        token = self._backend.tokenizer.decode([token_id], skip_special_tokens=True)
        is_last = (
            request_state.num_output_tokens + 1
            >= request_state.sampling_params.max_new_tokens
        )
        return DecodeResult(
            token_id=token_id,
            token=token,
            finish_reason=FinishReason.MAX_LENGTH if is_last else None,
        )

    def _post_prefill(
        self,
        out,
        sequences: list[Sequence],
        start_ns: int,
    ) -> None:
        """Sample the first generated token for each prefilled sequence.

        Prefill committed the prompt's k/v to the cache (positions ``0..P-1``)
        during the forward, so ``num_tokens`` already equals ``P`` and is NOT
        advanced here. The first token's k/v is stored later, during the first
        decode step. The flattened ``out.logits`` is sliced per sequence using
        each prompt's length.
        """
        offset = 0
        for seq in sequences:
            request_state = self._seq_to_request[seq.sequence_id]
            prompt_len = seq.num_prompt_tokens
            last_logit = out.logits[0, offset + prompt_len - 1, :].unsqueeze(0)
            offset += prompt_len

            try:
                # num_prompt_tokens is required by the emitter's _finish path,
                # and start_ns/DECODING transition are driven via
                # on_prefill_started + on_token. We skip on_prefill_succeeded:
                # it expects a per-request PrefillResult with tensors this
                # engine does not carry.
                request_state.num_prompt_tokens = prompt_len
                self._emitter.on_prefill_started(request_state, start_ns)
                result = self._sample_one(last_logit, request_state)
                seq.generated_token_ids.append(result.token_id)
                self._emitter.on_token(
                    request_state,
                    DecodeResult(
                        token_id=result.token_id,
                        token=result.token,
                        finish_reason=result.finish_reason,
                    ),
                )

                if result.is_finished:
                    seq.finished = True
                    self._cleanup_request(seq.sequence_id, request_state.request_id)
            except Exception:
                # Fail just this request instead of letting the exception
                # propagate to run()'s fatal handler and tear down the worker
                # along with every other in-flight request.
                logger.exception(
                    "Failed to sample first token for request %s",
                    request_state.request_id,
                )
                seq.finished = True
                self._emitter.on_failed(request_state, "sampling failed during prefill")
                self._cleanup_request(seq.sequence_id, request_state.request_id)

    def _post_decode(
        self,
        out,
        sequences: list[Sequence],
    ) -> None:
        """Advance each decoded sequence and sample its next token.

        The forward stored the input token's k/v at position ``num_tokens``
        (the value passed as ``position_id`` in ``_prepare_decode``), so we
        advance ``num_tokens`` by one to reflect the new cached length before
        the next decode step. ``out.logits`` has one row per sequence.
        """
        for i, seq in enumerate(sequences):
            request_state = self._seq_to_request[seq.sequence_id]
            # The input token was committed to the cache during the forward.
            seq.num_tokens += 1

            logit = out.logits[0, i, :].unsqueeze(0)
            try:
                result = self._sample_one(logit, request_state)
                seq.generated_token_ids.append(result.token_id)
                self._emitter.on_token(
                    request_state,
                    DecodeResult(
                        token_id=result.token_id,
                        token=result.token,
                        finish_reason=result.finish_reason,
                    ),
                )

                if result.is_finished:
                    seq.finished = True
                    self._cleanup_request(seq.sequence_id, request_state.request_id)
            except Exception:
                # Fail just this request instead of tearing down the worker.
                logger.exception(
                    "Failed to sample next token for request %s",
                    request_state.request_id,
                )
                seq.finished = True
                self._emitter.on_failed(request_state, "sampling failed during decode")
                self._cleanup_request(seq.sequence_id, request_state.request_id)

    def _cleanup_request(self, sequence_id: str, request_id: str) -> None:
        """Drop a finished request from the engine's tracking.

        Does not free blocks: the scheduler's ``_reap_finished`` reclaims them
        for any sequence with ``finished=True`` at the top of the next
        ``schedule()``.
        """
        self._seq_to_request.pop(sequence_id, None)
        self._all_requests.pop(request_id, None)
