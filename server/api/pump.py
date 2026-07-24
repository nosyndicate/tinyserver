"""The thread → event-loop bridge. **This file is scaffolding, by design.**

The engine runs on its own thread and knows nothing about asyncio; the HTTP
handlers are coroutines on the event loop. Those two worlds cannot touch each
other's data structures directly, and asyncio offers exactly one thread-safe
door between them: ``loop.call_soon_threadsafe``. This module is that door,
walked by a single background thread.

Note the shape: **one** thread for the whole server, not one per request. That
is the entire point of the exercise — a waiting request is now a suspended
coroutine costing a few hundred bytes, instead of a parked OS thread from
FastAPI's bounded worker pool.

When the engine later moves into its own process, this file and its test are
deleted outright: a ``recv`` coroutine on a socket replaces the thread and calls
``registry.dispatch`` itself, with no ``call_soon_threadsafe`` in sight. That is
why the routing logic lives in :class:`~server.api.collector.CollectorRegistry`
and only the thread plumbing lives here — the throwaway part is kept small and
quarantined.
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
        queue: The one server-wide queue that every request's
            :class:`~server.executor.sinks.SharedQueueSink` emits into.
        registry: Where to route each event, by ``request_id``.
        poll_timeout_s: See :data:`_POLL_TIMEOUT_S`. Tests lower it to keep
            ``stop()`` snappy.
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
        """Capture the loop and spawn the pump thread.

        The loop is passed in rather than discovered because the pump thread
        cannot call ``asyncio.get_running_loop()`` — that only works *on* the
        loop's own thread. The caller (the app's lifespan handler) does run on
        the loop, so it captures the reference once here for the pump's lifetime.
        """
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
        """Stop the thread, then drain whatever is left on the queue.

        The drain matters: ``Worker.stop()`` emits a final ``ErrorEvent`` per
        in-flight request as it cancels them, and those land on the shared queue.
        Killing the thread without draining would silently swallow them and leave
        the corresponding handlers waiting out their full timeout. Hence the
        shutdown order in lifespan: stop the worker first, then the pump.

        Blocks the caller (and therefore the loop, since lifespan shutdown runs
        on it) for up to roughly ``poll_timeout_s``. That is acceptable at
        shutdown, when there is no traffic left to serve.
        """
        self._stopping.set()

        if self._thread is not None:
            # The thread can be mid-`get`, so allow one full poll interval plus
            # slack for the dispatch it may be finishing.
            self._thread.join(timeout=self._poll_timeout_s + 1.0)
            if self._thread.is_alive():
                logger.warning("output pump thread did not exit; abandoning it")
            self._thread = None

        while True:
            try:
                event = self._queue.get_nowait()
            except Empty:
                break
            self._dispatch(event)

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
        """Schedule ``registry.dispatch(event)`` to run on the loop thread.

        ``call_soon_threadsafe`` is the only asyncio call that is legal from
        another thread. It does not run the callback; it queues it for the loop,
        which is why the actual routing (and every ``OutputCollector`` mutation)
        still happens single-threaded on the loop.
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
