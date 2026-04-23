import logging
import threading
from queue import Empty, Queue

from server.executor.engine import (
    EngineCallbacks,
    EngineControl,
    InferenceEngine,
)
from server.executor.types import (
    ErrorEvent,
    GenerationRequestState,
    RequestStatus,
)
from server.metrics.timers import now_ns

logger = logging.getLogger(__name__)


class Worker:
    """Orchestrates generation requests by bridging a queue and an inference engine.

    Lifecycle:
        1. **Submission**: Callers invoke ``submit()`` to enqueue a ``GenerationRequestState``
           into a bounded inbound queue.  The worker records the enqueue timestamp so that
           queue‑wait latency can be measured later by the engine.
        2. **Execution**: A daemon thread runs the engine's ``run()`` loop (started via
           ``start()``).  The engine drains the inbound queue, runs prefill and decode
           phases on the underlying executor, and emits events back to each request's
           output queue.
        3. **Shutdown**: ``stop()`` sets a shutdown event, joins the thread, then drains
           any remaining queued requests and cancels all in‑flight requests by pushing
           ``ErrorEvent`` messages to their output queues.

    Error handling:
        - **Per‑request failures** are reported by the engine via ``cancel_request``,
          which marks the request as ``FAILED`` and pushes an ``ErrorEvent``.
        - **Fatal (unrecoverable) errors** are handled by ``_handle_fatal_error``: it
          logs the exception, cancels any explicitly‑passed requests, cancels all
          in‑flight requests via the engine, and drains the inbound queue so no request
          is silently dropped.

    Threading:
        The worker runs its engine loop in a single daemon thread.  All communication
        with callers happens through thread‑safe queues (``Queue``), so no explicit
        locks are needed.  ``submit()`` blocks if the queue is full (raises
        ``queue.Full``).
    """

    def __init__(self, engine: InferenceEngine, max_queue_size: int) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        self._inbound: Queue[GenerationRequestState] = Queue(maxsize=max_queue_size)
        self._shutdown_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._engine = engine

    def _cancel_request(
        self, request_state: GenerationRequestState, error_message: str
    ) -> None:
        """Mark a request as failed and push an ErrorEvent to its output queue."""
        request_state.status = RequestStatus.FAILED
        request_state.error = error_message
        request_state.output_queue.put(
            ErrorEvent(
                request_id=request_state.request_id,
                error=request_state.error,
            )
        )

    def _cancel_inflight(self, message: str) -> None:
        """Ask the engine to cancel all in-flight requests."""
        self._engine.cancel_inflight(message, self._cancel_request)

    def _handle_fatal_error(
        self,
        error: Exception,
        extra_requests: list[GenerationRequestState] | None = None,
    ) -> None:
        """Handle an irrecoverable engine error: cancel everything and drain the queue."""
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

    def _run_loop(self) -> None:
        """Thread target: wire up control/callbacks and delegate to the engine."""
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
        """Enqueue a request for the engine to process.

        Raises:
            RuntimeError: If the worker is shutting down.
            queue.Full: If the inbound queue is at capacity.
        """
        if self._shutdown_event.is_set():
            raise RuntimeError("Cannot submit new request, worker is shutting down")

        request_state.enqueued_ns = now_ns()
        self._inbound.put_nowait(request_state)
