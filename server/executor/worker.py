import logging
import threading
from abc import ABC, abstractmethod
from queue import Empty, Queue
from typing import TypeVar

import torch

from server.executor.types import (
    BatchExecutorConfig,
    DoneEvent,
    ErrorEvent,
    ExecutorConfig,
    FinishReason,
    GenerationRequestState,
    RequestStatus,
    TokenEvent,
)
from server.metrics.timers import now_ns, ns_to_ms
from server.model.hf_runner import ModelRunner

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _assert_not_none(value: T | None) -> T:
    if value is None:
        raise ValueError("Expected value to be not None")
    return value


def _handle_error(request_state: GenerationRequestState, error: Exception) -> None:
    request_state.status = RequestStatus.FAILED
    request_state.error = str(error)
    request_state.output_queue.put(
        ErrorEvent(request_id=request_state.request_id, error=str(error))
    )


def _finish(request_state: GenerationRequestState) -> None:
    """Finalize the request and emit a DoneEvent.

    Precondition: start_ns and enqueued_ns must be set before this is called.
    """
    request_state.status = RequestStatus.DONE
    end_ns = now_ns()

    if request_state.start_ns is None:
        raise RuntimeError("start_ns must be set before _finish()")
    if request_state.enqueued_ns is None:
        raise RuntimeError("enqueued_ns must be set before _finish()")

    total_ms = ns_to_ms(end_ns - request_state.start_ns)
    # queue_wait_ms is the time from when the request was enqueued to when execution started.
    # We max with 0 to avoid negative queue wait time from clock jitter.
    queue_wait_ms = max(
        ns_to_ms(request_state.start_ns - request_state.enqueued_ns), 0.0
    )
    ttft_ms = (
        ns_to_ms(request_state.first_token_ns - request_state.start_ns)
        if request_state.first_token_ns is not None
        else -1.0
    )
    execution_ms = max(total_ms - queue_wait_ms, 0.0)

    if request_state.num_prompt_tokens is None:
        raise RuntimeError("num_prompt_tokens is required to finish the request")

    request_state.output_queue.put(
        DoneEvent(
            text="".join(request_state.output_tokens),
            num_prompt_tokens=request_state.num_prompt_tokens,
            num_output_tokens=request_state.num_output_tokens,
            ttft=ttft_ms,
            total_ms=total_ms,
            queue_wait_ms=queue_wait_ms,
            execution_ms=execution_ms,
        )
    )


def _sample_and_emit(
    runner: ModelRunner,
    request_state: GenerationRequestState,
) -> int | None:
    """Sample the next token and emit a TokenEvent.

    Returns the token ID if generation should continue, or None if the request is finished.
    """
    logits = request_state.all_logits[:, -1, :]
    next_token_id = runner.sample_token(
        logits, request_state.sampling_params, request_state.generator
    )

    is_first = request_state.num_output_tokens == 0
    if is_first:
        request_state.first_token_ns = now_ns()

    if next_token_id == runner.eos_token_id:
        request_state.output_queue.put(
            TokenEvent(
                token="",
                is_first=is_first,
                is_last=True,
                index=request_state.num_output_tokens,
            )
        )
        request_state.finished_reason = FinishReason.EOS
        _finish(request_state)
        return None

    next_token = runner.tokenizer.decode([next_token_id], skip_special_tokens=True)
    is_last = (
        request_state.num_output_tokens + 1
        >= request_state.sampling_params.max_new_tokens
    )

    request_state.output_tokens.append(next_token)
    request_state.output_queue.put(
        TokenEvent(
            token=next_token,
            is_first=is_first,
            is_last=is_last,
            index=request_state.num_output_tokens - 1,
        )
    )

    if is_last:
        request_state.finished_reason = FinishReason.MAX_LENGTH
        _finish(request_state)
        return None

    return next_token_id


class Worker(ABC):
    """Abstract base class for worker implementations that manage the lifecycle of generation requests."""

    def __init__(self, max_queue_size: int) -> None:
        self._inbound: Queue[GenerationRequestState] = Queue(maxsize=max_queue_size)
        self._shutdown_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _cancel_request(
        self, request_state: GenerationRequestState, error_message: str
    ) -> None:
        request_state.status = RequestStatus.FAILED
        request_state.error = error_message
        request_state.output_queue.put(
            ErrorEvent(
                request_id=request_state.request_id,
                error=request_state.error,
            )
        )

    @abstractmethod
    def _cancel_inflight(self, message: str) -> None:
        """Cancel all in-flight (active/waiting) requests and clear them."""
        raise NotImplementedError

    def _handle_fatal_error(
        self,
        error: Exception,
        extra_requests: list[GenerationRequestState] | None = None,
    ) -> None:
        """Cancel all active and pending requests after an unrecoverable error.

        Args:
            error: The exception that caused the fatal error.
            extra_requests: Any additional requests that are not yet tracked by the
                worker (e.g. new_requests[i:] when a prefill call raises).
        """
        logger.exception("Worker thread crashed with unexpected exception: %s", error)
        error_message = f"Worker encountered an unexpected error: {error}"
        for pending in extra_requests or []:
            try:
                self._cancel_request(pending, error_message)
            except Exception:
                logger.exception(
                    "Failed to emit error event for extra request %s",
                    pending.request_id,
                )
        self._cancel_inflight(error_message)
        while True:
            try:
                pending = self._inbound.get_nowait()
            except Empty:
                break
            try:
                self._cancel_request(pending, error_message)
            except Exception:
                logger.exception(
                    "Failed to emit error event for pending request %s",
                    pending.request_id,
                )

    @abstractmethod
    def _run_loop(self) -> None:
        raise NotImplementedError

    def start(self) -> None:
        """Create the thread and start the main loop."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Worker thread is already running")
            return
        self._shutdown_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="inference-worker", daemon=True
        )
        self._thread.start()
        logger.info("Worker thread started")

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._shutdown_event.set()

        if self._thread is not None:
            self._thread.join()
            logger.info("Worker thread stopped")

        # Thread has stopped — drain any remaining inbound requests without races
        while True:
            try:
                req = self._inbound.get_nowait()
                self._cancel_request(req, "Worker is shutting down, request rejected")
            except Empty:
                break

        self._cancel_inflight("Worker is shutting down, request cancelled")

    def submit(self, request_state: GenerationRequestState) -> None:
        """Submit a new request to the worker.

        Raises:
            queue.Full: If the worker's inbound queue is full, indicating that the worker is overloaded.
        """
        if self._shutdown_event.is_set():
            raise RuntimeError("Cannot submit new request, worker is shutting down")

        request_state.enqueued_ns = now_ns()
        self._inbound.put_nowait(request_state)


class SimpleWorker(Worker):
    """
    Usage:
        worker = SimpleWorker(runner, config)
        worker.start()
        worker.submit(request_state)
        ...
        worker.stop()
    """

    def __init__(self, runner: ModelRunner, config: ExecutorConfig) -> None:
        if config.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if config.max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive")

        super().__init__(config.max_queue_size)
        self._runner = runner
        self._config = config
        self._active: list[GenerationRequestState] = []

    def _cancel_inflight(self, message: str) -> None:
        for pending in self._active:
            try:
                self._cancel_request(pending, message)
            except Exception:
                logger.exception(
                    "Failed to emit error event for active request %s",
                    pending.request_id,
                )
        self._active.clear()

    def _prefill_request(self, request_state: GenerationRequestState) -> None:
        request_state.status = RequestStatus.PREFILLING
        request_state.start_ns = now_ns()
        try:
            all_logits, past_key_values, num_input_toks = self._runner.prefill(
                request_state.prompt
            )
            request_state.all_logits = all_logits
            request_state.past_key_values = past_key_values
            request_state.num_prompt_tokens = num_input_toks
            request_state.status = RequestStatus.DECODING
        except Exception as e:
            _handle_error(request_state, e)

    @torch.inference_mode()
    def _decode_request_step(self, request_state: GenerationRequestState) -> None:
        try:
            if request_state.all_logits is None:
                raise ValueError("No logits available for decoding step")

            next_token_id = _sample_and_emit(self._runner, request_state)
            if next_token_id is None:
                return

            if request_state.past_key_values is None:
                raise ValueError("No past_key_values available for decoding step")

            logits, past_key_values = self._runner.decode_step(
                next_token_id, request_state.past_key_values
            )
            request_state.all_logits = logits
            request_state.past_key_values = past_key_values
        except Exception as e:
            _handle_error(request_state, e)

    def _run_loop(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                # Fetch all the new requests from the inbound queue and add them to the active list
                new_requests: list[GenerationRequestState] = []
                while (
                    len(self._active) + len(new_requests)
                    < self._config.max_active_requests
                ):
                    try:
                        request_state = self._inbound.get_nowait()
                        new_requests.append(request_state)
                    except Empty:
                        break

                # Process the new requests: run prefill and move to active if successful
                for i, req in enumerate(new_requests):
                    # If the worker is shutting down, reject the new request with an error event
                    if self._shutdown_event.is_set():
                        # Cancel this request and all remaining new + active requests
                        for pending in new_requests[i:] + self._active:
                            try:
                                self._cancel_request(
                                    pending,
                                    "Worker is shutting down, request cancelled",
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to clean up request %s during prefill",
                                    pending.request_id,
                                )
                        self._active.clear()
                        return

                    try:
                        self._prefill_request(req)
                    except Exception as e:
                        self._handle_fatal_error(e, extra_requests=new_requests[i:])
                        return
                    if req.status == RequestStatus.DECODING:
                        self._active.append(req)

                # Decode one step for each active request
                for req in self._active:
                    if self._shutdown_event.is_set():
                        # Cancel all active requests before exiting
                        for pending in self._active:
                            try:
                                self._cancel_request(
                                    pending,
                                    "Worker is shutting down, request cancelled",
                                )
                            except Exception:
                                logger.exception(
                                    "Failed to clean up request %s during decoding",
                                    pending.request_id,
                                )
                        self._active.clear()
                        return

                    if req.status == RequestStatus.DECODING:
                        self._decode_request_step(req)

                # After the decoding, some requests may have finished or failed, remove them from the active list
                self._active = [
                    req for req in self._active if req.status == RequestStatus.DECODING
                ]

                # If there are still active, we directly continue to the next loop iteration without
                # sleeping, to maximize throughput and minimize latency. Otherwise, we sleep for a
                # short while to avoid busy loop. Use the sleep on _shutdown_event so when shutdown
                # signal is set, it can break the sleep immediately and exit the loop.
                if not self._active:
                    self._shutdown_event.wait(timeout=0.01)
        except Exception as e:
            self._handle_fatal_error(e)


class BatchWorker(Worker):
    """
    A BatchWorker implementation that batches requests together for more efficient processing.
    The implementation is similar to SimpleWorker, but it collects new requests into batches before processing.
    The batch size is determined by the BatchExecutorConfig (e.g., max_prefill_batch_size and max_decode_batch_size).
    """

    def _check_config(self, config: BatchExecutorConfig) -> None:
        if config.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
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

    def __init__(self, runner: ModelRunner, config: BatchExecutorConfig) -> None:
        self._check_config(config)
        super().__init__(config.max_queue_size)
        self._runner = runner
        self._config = config
        self._waiting: list[GenerationRequestState] = []
        self._active: list[GenerationRequestState] = []

    def _cancel_inflight(self, message: str) -> None:
        for pending in self._waiting + self._active:
            try:
                self._cancel_request(pending, message)
            except Exception:
                logger.exception(
                    "Failed to emit error event for request %s",
                    pending.request_id,
                )
        self._waiting.clear()
        self._active.clear()

    def _drain_inbound(self) -> None:
        """Move all requests from the inbound queue to the waiting list, up to max_active_requests."""
        max_num_reqs = self._config.max_active_requests
        while len(self._waiting) + len(self._active) < max_num_reqs:
            try:
                req = self._inbound.get_nowait()
                self._waiting.append(req)
            except Empty:
                break

    def _select_prefill_batch(self) -> list[GenerationRequestState]:
        """Select a batch of requests from the waiting list for prefill, up to max_prefill_batch_size."""
        batch_size = min(
            self._config.max_prefill_batch_size,
            self._config.max_active_requests - len(self._active),
            len(self._waiting),
        )
        batch = self._waiting[:batch_size]
        self._waiting = self._waiting[batch_size:]
        return batch

    def _select_decode_batch(self) -> list[GenerationRequestState]:
        """Select a batch of requests from the active list for decode, up to max_decode_batch_size."""
        # Defensive filter; in the current implementation all active requests should be DECODING.
        decoding = [req for req in self._active if req.status == RequestStatus.DECODING]
        return decoding[: self._config.max_decode_batch_size]

    @torch.inference_mode()
    def _prefill_batch(self, request_states: list[GenerationRequestState]) -> None:
        current_time_ns = now_ns()
        for request_state in request_states:
            request_state.status = RequestStatus.PREFILLING
            request_state.start_ns = current_time_ns
        try:
            prefill_batch_outputs = self._runner.prefill_batch(
                [request_state.prompt for request_state in request_states]
            )

            if len(prefill_batch_outputs) != len(request_states):
                raise ValueError(
                    f"Expected {len(request_states)} prefill outputs, but got {len(prefill_batch_outputs)}"
                )

            for request_state, prefill_output in zip(
                request_states, prefill_batch_outputs
            ):
                request_state.all_logits = prefill_output.logits
                request_state.past_key_values = prefill_output.past_key_values
                request_state.num_prompt_tokens = prefill_output.num_prompt_tokens
                request_state.status = RequestStatus.DECODING
        except Exception as e:
            for request_state in request_states:
                _handle_error(request_state, e)

    @torch.inference_mode()
    def _decode_batch_step(self, request_states: list[GenerationRequestState]) -> None:
        unfinished_request_states: list[tuple[GenerationRequestState, int]] = []
        for request_state in request_states:
            try:
                if request_state.all_logits is None:
                    raise ValueError("No logits available for decoding step")
                if request_state.past_key_values is None:
                    raise ValueError("No past_key_values available for decoding step")

                next_token_id = _sample_and_emit(self._runner, request_state)
                if next_token_id is not None:
                    unfinished_request_states.append((request_state, next_token_id))
            except Exception as e:
                _handle_error(request_state, e)

        if not unfinished_request_states:
            return

        next_input_ids = [token_id for _, token_id in unfinished_request_states]
        past_key_values = [
            _assert_not_none(request_state.past_key_values)
            for request_state, _ in unfinished_request_states
        ]

        try:
            decode_outputs = self._runner.decode_batch(
                next_input_ids,
                past_key_values,
            )

            if len(decode_outputs) != len(unfinished_request_states):
                raise ValueError(
                    f"Expected {len(unfinished_request_states)} decode outputs, but got {len(decode_outputs)}"
                )

            for (request_state, _), decode_output in zip(
                unfinished_request_states, decode_outputs
            ):
                request_state.all_logits = decode_output.logits
                request_state.past_key_values = decode_output.past_key_values

        except Exception as e:
            for request_state, _ in unfinished_request_states:
                _handle_error(request_state, e)

    def _run_loop(self) -> None:
        prefill_batch: list[GenerationRequestState] = []
        try:
            while not self._shutdown_event.is_set():
                # Move new requests from the inbound queue to the waiting list
                self._drain_inbound()

                if not self._shutdown_event.is_set():
                    prefill_batch = self._select_prefill_batch()
                    if prefill_batch:
                        self._prefill_batch(prefill_batch)
                        self._active.extend(
                            req
                            for req in prefill_batch
                            if req.status == RequestStatus.DECODING
                        )
                        prefill_batch = []

                if not self._shutdown_event.is_set():
                    # If there are active requests, select a batch and decode one step
                    decoding_batch = self._select_decode_batch()
                    if decoding_batch:
                        self._decode_batch_step(decoding_batch)

                # After the decoding, some requests may have finished or failed, remove them from the active list
                self._active = [
                    req for req in self._active if req.status == RequestStatus.DECODING
                ]

                # If there are still waiting or active requests, we directly continue to the next loop iteration without
                # sleeping, to maximize throughput and minimize latency. Otherwise, we sleep for a
                # short while to avoid busy loop. Use the sleep on _shutdown_event so when shutdown
                # signal is set, it can break the sleep immediately and exit the loop.
                if not self._waiting and not self._active:
                    self._shutdown_event.wait(timeout=0.01)

            # Graceful shutdown: cancel any remaining waiting and active requests
            self._cancel_inflight("Worker is shutting down, request cancelled")
        except Exception as e:
            self._handle_fatal_error(e, extra_requests=prefill_batch)
