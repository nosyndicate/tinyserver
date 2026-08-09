"""The thread → event-loop bridge.

Temporary code that will be deleted when we start having real multiprocessing.

The engine runs on its own thread and knows nothing about asyncio; the HTTP
handlers are coroutines on the event loop. Those two worlds cannot touch each
other's data structures directly, and asyncio offers exactly one thread-safe
door between them: ``loop.call_soon_threadsafe``. This module is that door,
walked by a single background thread.

Note the shape: **one** thread for the whole server, not one per request. That
is the entire point of the exercise — a waiting request is now a suspended
coroutine costing a few hundred bytes, instead of a parked OS thread from
FastAPI's bounded worker pool.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from queue import Empty, Queue

from server.api.collector import CollectorRegistry
from server.executor.types import Event

logger = logging.getLogger(__name__)

# How long the pump thread blocks on the shared queue before looking up to check
# whether it has been asked to stop. Purely a shutdown-latency knob: it costs one
# wake-up per interval while idle, and nothing at all while events are flowing
# (a non-empty queue returns immediately).
_POLL_TIMEOUT_S = 0.1


class OutputPump:
    """Drains the engine's shared event queue onto the event loop.

    Args:
        queue: The one server-wide queue that every request emits into.
        registry: Where to route each event, by `request_id`.
        poll_timeout_s: See `_POLL_TIMEOUT_S`.
    """

    def __init__(
        self,
        queue: Queue[Event],
        registry: CollectorRegistry,
        poll_timeout_s: float = _POLL_TIMEOUT_S,
    ) -> None:
        self._queue = queue
        self._registry = registry
        self._poll_timeout_s = poll_timeout_s
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the loop and spawn the pump thread."""
        if self._thread is not None:
            raise RuntimeError("OutputPump.start() called twice")

        self._loop = loop
        self._thread = threading.Thread(
            target=self._run,
            name="output-pump",
            # Daemon so a botched shutdown can never wedge process exit; the
            # explicit stop() below is what we actually rely on.
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread, then drain whatever is left on the queue."""
        self._stopping.set()

        if self._thread is not None:
            # The worker is stopped before the pump, so no producer remains.
            # `_run` polls at most once more before exiting; joining guarantees
            # it cannot schedule an event after the shutdown flush barrier.
            self._thread.join()
            self._thread = None

        # Drain any events that arrived after the stop signal but before the
        # thread actually exited. This is the only way to guarantee that every
        # ErrorEvent from the worker is delivered to its handler, even if the
        # pump thread was still mid-`get` when the stop signal arrived.
        while True:
            try:
                event = self._queue.get_nowait()
            except Empty:
                break
            self._dispatch(event)

    async def stop_and_flush(self) -> None:
        """Stop the pump and run all queued dispatches before returning.

        ``stop`` schedules its final dispatches onto the event loop. The
        sentinel is queued after them, so awaiting it establishes that every
        terminal event drained during shutdown reached its collector before a
        caller invokes ``CollectorRegistry.fail_all``.
        """
        self.stop()
        assert self._loop is not None, "OutputPump.start() must be called first"
        if self._loop.is_closed():
            return

        flushed = self._loop.create_future()
        self._loop.call_soon(flushed.set_result, None)
        await flushed

    def _run(self) -> None:
        """The pump thread's whole life: block on the queue, hand off, repeat."""
        while not self._stopping.is_set():
            try:
                event = self._queue.get(timeout=self._poll_timeout_s)
            except Empty:
                # Idle tick. Loop back so `stop()` is noticed promptly rather
                # than only when the next event happens to arrive.
                continue
            self._dispatch(event)

    def _dispatch(self, event: Event) -> None:
        """Schedule `registry.dispatch(event)` to run on the loop thread.

        Using call_soon_threadsafe is the only way to safely cross the thread boundary.
        """
        assert self._loop is not None, "OutputPump.start() must be called first"

        try:
            self._loop.call_soon_threadsafe(self._registry.dispatch, event)
        except RuntimeError:
            # The loop was closed between our check and this call — a genuine
            # race during shutdown, not a bug we can prevent from this thread.
            # There is nothing left to deliver to: a closed loop means every
            # handler coroutine is already gone. Log and drop.
            logger.warning(
                "event loop closed; dropping event for request_id %s",
                event.request_id,
            )
