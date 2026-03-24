import logging
import threading
from queue import Empty, Full, Queue

from server.executor.types import (
    ErrorEvent,
    ExecutorConfig,
    GenerationRequestState,
    RequestStatus,
)

logger = logging.getLogger(__name__)


class Worker:

    def __init__(self, config: ExecutorConfig) -> None:
        self._active_queue = Queue(maxsize=config.max_active_requests)
        self._waiting_queue = Queue(maxsize=config.max_queue_size)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """
        Start the background worker thread.
        """
        self._thread = threading.Thread(
            target=self._run, name="InferenceWorker", daemon=True
        )
        self._thread.start()
        logger.info("Worker thread started.")


    def stop(self) -> None:
        
        self._thread.join()

    def submit(self, request_state: GenerationRequestState) -> None:
        """
        Submit a new generation request to the worker by HTTP handler. This method is non-blocking.

        Args:
            request_state: The mutable state object for the generation request, created by the HTTP handler.

        Raises:
            Full: If the waiting queue is full, indicating that the worker is overloaded. The HTTP handler
                should catch this exception and return HTTP 503 error to the client.
        """
        # put_nowait will raise Full exception if the queue is full
        self._waiting_queue.put_nowait(request_state)

    def _run(self) -> None:
        pass
