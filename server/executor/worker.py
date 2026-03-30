import logging
import threading
from queue import Empty, Queue
import time

from server.executor.executor import Executor
from server.executor.types import (
    ErrorEvent,
    ExecutorConfig,
    GenerationRequestState,
    RequestStatus,
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

    def __init__(self, executor: Executor, config: ExecutorConfig) -> None:
        if config.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if config.max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive")

        self._executor = executor
        self._config = config

        # Inbound queue, HTTP handlers put the new request here
        # If the queue is full, new requests will be rejected with a 503 error
        self._inbound: Queue[GenerationRequestState] = Queue(
            maxsize=self._config.max_queue_size
        )

        # The list of active requests being processed by the worker
        self._active: list[GenerationRequestState] = []

        # Shutdown event to signal the worker thread to stop
        self._shutdown_event = threading.Event()

        # The worker thread that runs the main loop
        self._thread: threading.Thread | None = None

    def _run_loop(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                # Fetch all the new requests from the inbound queue and add them to the active list
                new_requests: list[GenerationRequestState] = []
                while (
                    len(self._active) + len(new_requests) < self._config.max_active_requests
                ):
                    try:
                        request_state = self._inbound.get_nowait()
                        new_requests.append(request_state)
                    except Empty:
                        break

                # Process the new requests: run prefill and move to active if successful
                for req in new_requests:
                    # If the worker is shutting down, reject the new request with an error event
                    if self._shutdown_event.is_set():
                        req.output_queue.put(
                            ErrorEvent(
                                request_id=req.request_id,
                                error="Worker is shutting down, request rejected",
                            )
                        )
                        return

                    self._executor.prefill(req)
                    if req.status == RequestStatus.DECODING:
                        self._active.append(req)

                # Decode one step for each active request
                for req in self._active:
                    if self._shutdown_event.is_set():
                        return

                    if req.status == RequestStatus.DECODING:
                        self._executor.decode_one_step(req)

                # After the decoding, some requests may have finished or failed, remove them from the active list
                self._active = [
                    req for req in self._active if req.status == RequestStatus.DECODING
                ]

                # If there are still active, we directly continue to the next loop iteration without
                # sleeping, to maximize throughput and minimize latency. Otherwise, we sleep for a
                # short while to avoid busy loop.
                if not self._active:
                    time.sleep(0.01)
        except Exception as e:
            logger.exception("Worker thread crashed with unexpected exception: %s", e)
            raise

    def start(self) -> None:
        """Create the thread and start the main loop."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Worker thread is already running")
            return
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
                req.output_queue.put(
                    ErrorEvent(
                        request_id=req.request_id,
                        error="Worker is shutting down, request rejected",
                    )
                )
            except Empty:
                break

        if self._thread is not None:
            self._thread.join()
            logger.info("Worker thread stopped")

    def submit(self, request_state: GenerationRequestState) -> None:
        """Submit a new request to the worker.

        Raises:
            queue.Full: If the worker's inbound queue is full, indicating that the worker is overloaded.
        """
        self._inbound.put_nowait(request_state)
