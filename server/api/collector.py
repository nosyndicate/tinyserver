"""Consumer-side async primitives: per-request mailboxes and their routing table.

This is the *other half* of :class:`~server.executor.types.EventSink`. The engine
side emits events into a sink; this side hands them to the coroutine that is
awaiting them for a particular request.

**This file is transport-agnostic on purpose.** Right now events reach it from an
engine thread via a shared ``queue.Queue`` plus the pump thread
(``server/api/pump.py``). Later the engine moves into its own process and events
arrive over a socket instead — at which point the pump is deleted and a recv
coroutine calls :meth:`CollectorRegistry.dispatch` directly. Nothing in this
module changes. That is why there are deliberately **no ``queue`` or ``threading``
imports here**: if you find yourself needing one, it belongs in ``pump.py``.
"""

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
    single waiter drains all ten with one wake-up. This is the shape both vLLM
    (``RequestOutputCollector``) and SGLang converged on.

    **Threading contract: every method here runs on the event loop thread.**
    ``put`` is never called from the engine thread directly — the pump routes it
    through ``loop.call_soon_threadsafe``, and in the process-split world the
    recv coroutine is already on the loop. That single-threaded discipline is
    what lets this class hold no lock at all.

    It also assumes a **single waiter**: exactly one handler coroutine awaits a
    given request's collector. Two concurrent ``get`` calls on the same instance
    would race on ``clear()`` and is not a supported use.
    """

    def __init__(self) -> None:
        # Invariant maintained by put/get below:
        #     buffer non-empty  <=>  event set
        # It holds because both methods only ever run on the one loop thread, so
        # no one can observe the buffer and the flag mid-update.
        self._event = asyncio.Event()
        self._buffer: deque[Event] = deque()

    def put(self, event: Event) -> None:
        """Deliver one event to whoever is awaiting this request.

        Loop-thread only (see the class docstring). Cheap and non-blocking: the
        buffer is unbounded, so a producer is never throttled by a slow client
        here — backpressure is the engine's admission control, not this queue's.
        """
        self._buffer.append(event)
        self._event.set()

    async def get(self, timeout: float | None = None) -> Event:
        """Return the next event, waiting at most ``timeout`` seconds for one.

        Raises:
            TimeoutError: if no event arrives in time. Callers map this to the
                504 response / timeout stream chunk the sync handlers produce
                today. (On Python 3.11+ ``asyncio.TimeoutError`` *is* the
                builtin ``TimeoutError``, so a plain ``except TimeoutError``
                catches it.)
        """
        if not self._buffer:
            # Only pay for a suspension when there is genuinely nothing to
            # return. Mid-stream this branch is usually skipped: the pump has
            # already buffered the next token, so we hand it back without ever
            # yielding to the loop. (vLLM's `get_nowait` fast path.)
            await asyncio.wait_for(self._event.wait(), timeout)

        # Safe by the invariant: waking from `_event.wait()` implies a `put`
        # happened, and nothing else can have consumed it (single waiter).
        event = self._buffer.popleft()
        if not self._buffer:
            # Drained — re-arm so the next `get` actually waits instead of
            # spinning on a stale flag.
            self._event.clear()
        return event


class CollectorRegistry:
    """``request_id`` → :class:`OutputCollector` routing table.

    The producer side no longer knows which consumer an event belongs to: events
    from every in-flight request travel one shared path, tagged with their
    ``request_id``. This is the lookup that puts them back on the right coroutine.

    Like :class:`OutputCollector`, **all access happens on the event loop
    thread** — ``register``/``unregister`` from the request handlers, ``dispatch``
    from the pump's ``call_soon_threadsafe`` callback (which the loop runs on the
    loop thread) — so a plain ``dict`` with no lock is correct.
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
        """
        collector = OutputCollector()
        self._collectors[request_id] = collector
        return collector

    def unregister(self, request_id: str) -> None:
        """Forget this request's mailbox. Idempotent, so it is safe in a
        ``finally:`` block that may run after an earlier cleanup."""
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
        """Push an ``ErrorEvent`` to every live collector.

        Used when no further events can ever arrive — server shutdown now, and a
        dead engine process later. Without this, handlers would sit awaiting
        their full timeout for events that will never come.

        Collectors are left registered; each handler still unregisters itself in
        its own ``finally:``. Iterating a snapshot keeps that harmless even if a
        woken handler unregisters while we are still looping.
        """
        for request_id, collector in list(self._collectors.items()):
            collector.put(ErrorEvent(request_id=request_id, error=error_message))
