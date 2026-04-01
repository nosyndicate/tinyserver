import logging
import threading
from queue import Empty, Queue

from server.executor.types import (
    ErrorEvent,
    ExecutorConfig,
    GenerationRequestState,
    RequestStatus,
    BaseExecutor
)

logger = logging.getLogger(__name__)


class Worker:
    """
    Usage:
        worker = Worker(executor, config)
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

        self._executor = executor
        self._config = config

        # Inbound queue, HTTP handlers put the new request here.
        # If the queue is full, submit() raises queue.Full; callers (e.g., HTTP
        # handlers) should translate that condition into an HTTP 503 response.
        self._inbound: Queue[GenerationRequestState] = Queue(
            maxsize=self._config.max_queue_size
        )

        # The list of active requests being processed by the worker
        self._active: list[GenerationRequestState] = []

        # Shutdown event to signal the worker thread to stop
        self._shutdown_event = threading.Event()

        # The worker thread that runs the main loop
        self._thread: threading.Thread | None = None

    def _cancel_request(
        self, request_state: GenerationRequestState, error_message: str
    ) -> None:
        """
        Helper method to cancel a request with a given error message.
        It updates the request state and emits an error event.
        """
        request_state.status = RequestStatus.FAILED
        request_state.error = error_message
        request_state.output_queue.put(
            ErrorEvent(
                request_id=request_state.request_id,
                error=request_state.error,
            )
        )

    def _handle_fatal_error(
        self,
        error: Exception,
        extra_requests: list[GenerationRequestState] | None = None,
    ) -> None:
        """Cancel all active and pending requests after an unrecoverable error.

        Args:
            error: The exception that caused the fatal error.
            extra_requests: Any additional requests that are not yet in
                self._active or self._inbound (e.g. new_requests[i:] when a
                prefill call raises).
        """
        logger.exception("Worker thread crashed with unexpected exception: %s", error)
        error_message = f"Worker encountered an unexpected error: {error}"
        # Cancel any batch-local requests that are not yet in self._active
        for pending in extra_requests or []:
            try:
                self._cancel_request(pending, error_message)
            except Exception:
                logger.exception(
                    "Failed to emit error event for extra request %s",
                    pending.request_id,
                )
        # Cancel all active requests
        for pending in self._active:
            try:
                self._cancel_request(pending, error_message)
            except Exception:
                logger.exception(
                    "Failed to emit error event for active request %s",
                    pending.request_id,
                )
        self._active.clear()
        # Drain the inbound queue
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
                        self._executor.decode_one_step(req)

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
            # This is a background thread, no other thread is expected to catch this exception, so we log it here.
            # We also attempt to cancel all active and pending requests with an error event.
            self._handle_fatal_error(e)

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

        # Drain the inbound queue and emit error events for pending requests
        while True:
            try:
                req = self._inbound.get_nowait()
                self._cancel_request(req, "Worker is shutting down, request rejected")
            except Empty:
                break

        if self._thread is not None:
            self._thread.join()
            logger.info("Worker thread stopped")

        # Since thread stopped, cancel any active requests that were still in-flight
        for pending in self._active:
            try:
                self._cancel_request(
                    pending, "Worker is shutting down, request cancelled"
                )
            except Exception:
                logger.exception(
                    "Failed to emit error event for active request %s",
                    pending.request_id,
                )

        self._active.clear()

    def submit(self, request_state: GenerationRequestState) -> None:
        """Submit a new request to the worker.

        Raises:
            queue.Full: If the worker's inbound queue is full, indicating that the worker is overloaded.
        """
        if self._shutdown_event.is_set():
            raise RuntimeError("Cannot submit new request, worker is shutting down")

        self._inbound.put_nowait(request_state)
