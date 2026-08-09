from __future__ import annotations

import asyncio
import logging
from collections import deque

from server.executor.types import ErrorEvent, Event

logger = logging.getLogger(__name__)


class OutputCollector:
    """One request's mailbox: an ``asyncio.Event`` plus a buffer of events.

    Why not an ``asyncio.Queue``? During decode the engine emits a token every
    few milliseconds. A queue wakes its waiter once per item, so a burst of ten
    tokens costs ten wake-ups and ten task switches. An Event + buffer lets a
    burst **coalesce**: the producer appends ten times and sets the flag, and the
    single waiter drains all ten with one wake-up.

    Threading contract: every method here runs on the event loop thread.
    ``put`` is never called from the engine thread directly — the pump routes it
    through ``loop.call_soon_threadsafe``, and in the process-split world the
    recv coroutine is already on the loop. That single-threaded discipline is
    what lets this class hold no lock at all.

    Only one coroutine may call ``get`` at a time for each collector. If two
    coroutines call it concurrently, they may both try to clear the event, so
    events could be handled incorrectly.
    """

    def __init__(self) -> None:
        # `put` and `get` keep these states in sync: when the buffer has events,
        # the event is set; when the buffer is empty, the event is clear. Both
        # methods run on the same event-loop thread, so no code can see them
        # partway through an update.
        self._event = asyncio.Event()
        self._buffer: deque[Event] = deque()

    def put(self, event: Event) -> None:
        """Deliver one event to whoever is awaiting this request.

        Cheap and non-blocking: the buffer is unbounded, so a producer is never
        throttled by a slow client here — backpressure is the engine's admission
        control, not this queue's.
        """
        self._buffer.append(event)
        self._event.set()

    async def get(self, timeout: float | None = None) -> Event:
        """Return the next event, waiting at most ``timeout`` seconds for one.
        If _buffer already has an event, this function immediately removes and
        returns it. It does not await, so the coroutine keeps running without
        yielding control back to the asyncio event loop.

        Raises:
            TimeoutError: if no event arrives in time. Callers map this to the
                504 response / timeout stream chunk the sync handlers produce
                today. (On Python 3.11+ ``asyncio.TimeoutError`` *is* the
                builtin ``TimeoutError``, so a plain ``except TimeoutError``
                catches it.)
        """
        if not self._buffer:
            # Wait only when there is no buffered event to return. During
            # streaming, the pump often buffers the next token first, so
            # `get` can return it immediately without waiting. Only when
            # the buffer is empty does it wait on _event, which suspends
            # the coroutine until a new event arrives.
            await asyncio.wait_for(self._event.wait(), timeout)

        # Waking from `_event.wait()` implies a `put` happened,
        # and nothing else can have consumed it (single waiter).
        event = self._buffer.popleft()
        if not self._buffer:
            # Drained — re-arm so the next `get` actually waits instead of
            # spinning on a stale flag.
            self._event.clear()
        return event


class CollectorRegistry:
    """`request_id` → `OutputCollector` routing table.

    The producer side no longer knows which consumer an event belongs to: events
    from every in-flight request travel one shared path, tagged with their
    `request_id`. This is the lookup that puts them back on the right coroutine.

    Like `OutputCollector`, all access happens on the event loop
    thread — `register`/`unregister` from the request handlers, `dispatch`
    from the pump's `call_soon_threadsafe` callback (which the loop runs on the
    loop thread) — so a plain `dict` with no lock is correct.
    """

    def __init__(self) -> None:
        self._collectors: dict[str, OutputCollector] = {}

    def __len__(self) -> int:
        """Number of requests still waiting for events."""
        return len(self._collectors)

    def register(self, request_id: str) -> OutputCollector:
        """Create this request's mailbox.

        Callers must register **before** submitting to the engine, otherwise a
        fast first token could arrive with nowhere to go.

        A live request ID must never be replaced: doing so would orphan the
        original handler's collector and route subsequent events to a mailbox
        nobody is awaiting.
        """
        if request_id in self._collectors:
            raise ValueError(
                f"collector already registered for request_id {request_id!r}"
            )
        collector = OutputCollector()
        self._collectors[request_id] = collector
        return collector

    def unregister(self, request_id: str) -> None:
        """Forget this request's mailbox. Idempotent, so it is safe in a
        `finally:` block that may run after an earlier cleanup."""
        self._collectors.pop(request_id, None)

    def dispatch(self, event: Event) -> None:
        """Route one event to its request's collector."""
        collector = self._collectors.get(event.request_id)
        if collector is None:
            # Expected, not an error. The handler already unregistered — it timed
            # out, the client disconnected, or it returned on the DoneEvent —
            # while the engine was still finishing up. Nobody is listening, so
            # dropping is the correct behavior.
            logger.debug(
                "dropping event for unknown request_id %s (handler already gone)",
                event.request_id,
            )
            return
        collector.put(event)

    def fail_all(self, error_message: str) -> None:
        """Push an `ErrorEvent` to every live collector.

        Used when no further events can ever arrive — server shutdown now, and a
        dead engine process later. Without this, handlers would sit awaiting
        their full timeout for events that will never come.

        Collectors are left registered; each handler still unregisters itself in
        its own `finally:`. Iterating a snapshot keeps that harmless even if a
        woken handler unregisters while we are still looping.
        """
        for request_id, collector in list(self._collectors.items()):
            collector.put(ErrorEvent(request_id=request_id, error=error_message))
