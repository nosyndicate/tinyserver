import logging
import threading
from abc import ABC, abstractmethod
from queue import Empty, Queue

from server.executor.engine import (
    BatchInferenceEngine,
    EngineCallbacks,
    EngineControl,
    SimpleInferenceEngine,
)
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
        self._engine = SimpleInferenceEngine(executor, config)

    def _cancel_inflight(self, message: str) -> None:
        self._engine.cancel_inflight(message, self._cancel_request)

    def _run_loop(self) -> None:
        self._engine.run(
            inbound=self._inbound,
            control=EngineControl(
                should_stop=self._shutdown_event.is_set,
                wait_idle=self._shutdown_event.wait,
            ),
            callbacks=EngineCallbacks(
                cancel_request=self._cancel_request,
                handle_fatal_error=self._handle_fatal_error,
            ),
        )


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
        self._engine = BatchInferenceEngine(executor, config)

    def _cancel_inflight(self, message: str) -> None:
        self._engine.cancel_inflight(message, self._cancel_request)

    def _run_loop(self) -> None:
        self._engine.run(
            inbound=self._inbound,
            control=EngineControl(
                should_stop=self._shutdown_event.is_set,
                wait_idle=self._shutdown_event.wait,
            ),
            callbacks=EngineCallbacks(
                cancel_request=self._cancel_request,
                handle_fatal_error=self._handle_fatal_error,
            ),
        )
