import logging
import threading
from abc import ABC, abstractmethod
from queue import Empty, Queue

from server.executor.types import (
    BaseBatchExecutor,
    BaseExecutor,
    BatchExecutorConfig,
    ErrorEvent,
    ExecutorConfig,
    GenerationRequestState,
    RequestStatus,
)
from server.metrics.timers import now_ns

logger = logging.getLogger(__name__)


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
        worker = SimpleWorker(executor, config)
        worker.start()
        worker.submit(request_state)
        ...
        worker.stop()
    """

    def __init__(self, executor: BaseExecutor, config: ExecutorConfig) -> None:
        if config.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if config.max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive")

        super().__init__(config.max_queue_size)
        self._executor = executor
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
                        self._executor.prefill(req)
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
                        self._executor.decode(req)

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

    def __init__(
        self, executor: BaseBatchExecutor, config: BatchExecutorConfig
    ) -> None:
        self._check_config(config)
        super().__init__(config.max_queue_size)
        self._executor = executor
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
        # This is just being defensive; in the current implementation all active requests should be in DECODING status
        decoding = [req for req in self._active if req.status == RequestStatus.DECODING]
        return decoding[: self._config.max_decode_batch_size]

    def _run_loop(self) -> None:
        prefill_batch: list[GenerationRequestState] = []
        try:
            while not self._shutdown_event.is_set():
                # Move new requests from the inbound queue to the waiting list
                self._drain_inbound()

                if not self._shutdown_event.is_set():
                    prefill_batch = self._select_prefill_batch()
                    if prefill_batch:
                        self._executor.batched_prefill(prefill_batch)
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
                        self._executor.batched_decode(decoding_batch)

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
