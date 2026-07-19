"""Producer-side event sinks.

An :class:`~server.executor.types.EventSink` is *where the engine hands off*
client-facing events. Isolating this behind a tiny protocol lets the same
producer call sites (``events.py``, ``worker.py``) work unchanged whether the
consumer is:

- Stage 1 (B2): an in-process ``queue.Queue`` drained by a pump thread, or
- Stage 2 (C): a socket to a separate engine process.

Only ``queue``-backed sinks live here; the socket sink arrives with Stage 2.
"""

from __future__ import annotations

from queue import Queue

from server.executor.types import Event


class DirectQueueSink:
    """Emit events onto a single, per-request ``queue.Queue``.

    This is the Stage-1 default: it wraps a request's own ``output_queue`` so
    that emitting through the sink is byte-for-byte equivalent to the old
    ``output_queue.put(...)`` calls. Synchronous tests can build a state, run
    the engine, and drain the same queue with no event loop involved.
    """

    def __init__(self, queue: Queue[Event]) -> None:
        self._queue = queue

    def emit(self, event: Event) -> None:
        self._queue.put(event)


class SharedQueueSink:
    """Emit events onto one server-wide ``queue.Queue`` drained by the pump.

    Every request state shares a single instance, so the pump thread has just
    one queue to drain and route by ``request_id``. Introduced now for the
    Stage-2 seam; it is wired into the app in a later PR.
    """

    def __init__(self) -> None:
        self._queue: Queue[Event] = Queue()

    @property
    def queue(self) -> Queue[Event]:
        return self._queue

    def emit(self, event: Event) -> None:
        self._queue.put(event)
