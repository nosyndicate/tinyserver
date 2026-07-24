"""The Stage-1 output pump: a bridge thread from a shared queue to the loop.

Producers (engine/worker) run on their own threads and emit events onto one
server-wide ``queue.Queue`` (see
:class:`~server.executor.sinks.SharedQueueSink`). Consumers live on the asyncio
event loop in :mod:`server.api.collector`. Something has to move events across
that thread boundary — that is this pump.

The pump does the *minimum* thread work and nothing else: it blocks on the
shared ``queue.Queue`` and, for each event, hops onto the loop thread via
``loop.call_soon_threadsafe`` to run :meth:`CollectorRegistry.dispatch`. The
actual ``request_id -> collector`` routing lives in ``dispatch``, not here, on
purpose: **this whole file is deliberately throwaway.** In Stage 2 the engine
moves to a separate process and events arrive over a socket read directly on
the loop, so the pump (and only the pump) is deleted while ``dispatch`` stays.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading

from server.api.collector import CollectorRegistry
from server.executor.types import Event

logger = logging.getLogger(__name__)

# How long the pump thread blocks on the queue before looping back to re-check
# the stop flag. Small enough that shutdown is snappy, large enough to avoid
# a busy spin.
_POLL_TIMEOUT_S = 0.1


class OutputPump:
    """Drains a shared event queue onto the loop, dispatching via the registry.

    Owns the shared queue, a reference to the event loop, and a daemon thread.
    """

    def __init__(
        self, shared_queue: queue.Queue[Event], registry: CollectorRegistry
    ) -> None:
        self._shared_queue = shared_queue
        self._registry = registry
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the loop and start the pump thread.

        Called from ``lifespan`` where ``asyncio.get_running_loop()`` gives the
        loop all the request coroutines run on — captured once, not per request.
        """
        self._loop = loop
        self._thread = threading.Thread(
            target=self._run, name="output-pump", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        """Thread body: block on the queue, forward each event to the loop."""
        while not self._stop.is_set():
            try:
                event = self._shared_queue.get(timeout=_POLL_TIMEOUT_S)
            except queue.Empty:
                # No event this interval; loop back to re-check the stop flag.
                continue
            self._forward(event)

    def _forward(self, event: Event) -> None:
        """Schedule ``registry.dispatch(event)`` to run on the loop thread.

        ``call_soon_threadsafe`` raises ``RuntimeError`` if the loop is already
        closed (can happen during shutdown). This is the one genuine
        concurrency hazard called out in the design doc: catch it, log, drop.
        """
        assert self._loop is not None  # start() runs before any _forward
        try:
            self._loop.call_soon_threadsafe(self._registry.dispatch, event)
        except RuntimeError:
            logger.warning(
                "loop closed; dropping event for request_id %s (%s)",
                event.request_id,
                type(event).__name__,
            )

    def stop(self) -> None:
        """Stop the thread, then drain any remaining events synchronously.

        Ordering matters: we join the thread first so nothing else is pulling
        from the queue, then drain whatever is left straight into the registry
        so no event queued right before shutdown is silently lost.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

        # Drain the remainder. The loop may already be closing, so _forward's
        # RuntimeError guard still applies to each of these.
        while True:
            try:
                event = self._shared_queue.get_nowait()
            except queue.Empty:
                break
            self._forward(event)
