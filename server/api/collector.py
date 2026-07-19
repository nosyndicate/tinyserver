"""Async consumer primitives: per-request collectors + a routing registry.

This is the *consumer* side of the event pipeline. Producers (the engine,
the worker) emit :class:`~server.executor.types.Event` objects through an
``EventSink``; on the API side those events must reach the coroutine that is
serving the matching HTTP request. Two objects do that:

- :class:`OutputCollector` — one per in-flight request. It buffers events and
  lets a coroutine ``await`` the next one. It is the async analogue of the old
  per-request ``queue.Queue`` that HTTP handlers used to block on, except
  ``await``-ing frees the event loop to serve other requests (e.g. ``/health``)
  instead of parking a thread.
- :class:`CollectorRegistry` — a ``request_id -> OutputCollector`` map. It takes
  an incoming event and ``dispatch``es it to the right collector.

**Design rule (keep this true — Stage 2 reuses this file verbatim):** this
module must not import ``threading`` or ``queue``. All the thread/queue
machinery lives in ``pump.py`` (Stage 1) or the socket transport (Stage 2).
Everything here runs on the asyncio event loop thread.
"""

from __future__ import annotations

import asyncio
import logging

from server.executor.types import ErrorEvent, Event

logger = logging.getLogger(__name__)


class OutputCollector:
    """Buffers events for one request and hands them to an awaiting coroutine.

    Shape borrowed from vLLM/SGLang: a plain ``list`` buffer plus a single
    ``asyncio.Event`` flag, rather than an ``asyncio.Queue``. The reason is
    *coalescing* — if the producer emits many tokens in a burst, they all land
    in the list and a single ``get`` / wake-up drains them one at a time, so we
    never schedule one event-loop task-switch per token.

    Threading contract:
    - :meth:`put` is called *on the loop thread only*. In Stage 1 the pump
      thread never calls it directly; it hops onto the loop via
      ``loop.call_soon_threadsafe(registry.dispatch, ...)`` first. In Stage 2
      the socket recv coroutine calls it directly. Either way it runs on the
      loop, so no lock is needed.
    - :meth:`get` is awaited by the request-serving coroutine (also the loop).
    """

    def __init__(self) -> None:
        # Set whenever the buffer is non-empty; cleared when it drains empty.
        self._event = asyncio.Event()
        self._buffer: list[Event] = []

    def put(self, event: Event) -> None:
        """Append an event and wake any coroutine waiting in :meth:`get`.

        Loop-thread only (see class docstring). Not a coroutine — the caller
        does not await; it just enqueues and returns.
        """
        self._buffer.append(event)
        self._event.set()

    async def get(self, timeout: float) -> Event:
        """Return the next event, waiting up to ``timeout`` seconds.

        Raises :class:`asyncio.TimeoutError` if no event arrives in time
        (HTTP handlers map that to a 504 / a timeout stream chunk).
        """
        # Fast path: something is already buffered, so return it without
        # awaiting at all. This is vLLM's "get_nowait" trick — it avoids an
        # unnecessary task switch when a burst of tokens is already queued.
        if not self._buffer:
            # Slow path: nothing buffered yet, so wait for a producer to
            # `set()` the flag. `wait_for` raises TimeoutError if it doesn't.
            await asyncio.wait_for(self._event.wait(), timeout)

        event = self._buffer.pop(0)
        # If we just drained the last buffered event, clear the flag so the
        # next `get` takes the slow path and waits again.
        if not self._buffer:
            self._event.clear()
        return event


class CollectorRegistry:
    """Routes events to the right :class:`OutputCollector` by ``request_id``.

    A plain ``dict`` with no lock: every method runs on the loop thread
    (``register``/``unregister`` from HTTP handlers, ``dispatch`` from the pump
    via ``call_soon_threadsafe`` in Stage 1 / the recv coroutine in Stage 2).
    """

    def __init__(self) -> None:
        self._collectors: dict[str, OutputCollector] = {}

    def register(self, request_id: str) -> OutputCollector:
        """Create and store a collector for ``request_id``, returning it.

        Handlers call this *before* submitting the request, so no event can
        arrive before there is a collector to route it to.
        """
        collector = OutputCollector()
        self._collectors[request_id] = collector
        return collector

    def unregister(self, request_id: str) -> None:
        """Drop the collector for ``request_id``; tolerate an unknown id."""
        self._collectors.pop(request_id, None)

    def dispatch(self, event: Event) -> None:
        """Route one event to its request's collector.

        This is the *dispatch body* Stage 2 keeps unchanged — Stage 2 calls it
        directly from the socket recv coroutine instead of through the pump.

        An unknown ``request_id`` means the request already finished and was
        unregistered (a late/straggler event); we log at debug and drop it.
        """
        collector = self._collectors.get(event.request_id)
        if collector is None:
            logger.debug(
                "dropping event for unknown request_id %s (%s)",
                event.request_id,
                type(event).__name__,
            )
            return
        collector.put(event)

    def fail_all(self, error_message: str) -> None:
        """Push an :class:`ErrorEvent` into every live collector.

        Used at shutdown (Stage 1) and on engine-process crash (Stage 2) to
        unblock every request still waiting in :meth:`OutputCollector.get`.
        Iterate a snapshot since collectors may unregister as they wake.
        """
        for request_id, collector in list(self._collectors.items()):
            collector.put(ErrorEvent(request_id=request_id, error=error_message))
