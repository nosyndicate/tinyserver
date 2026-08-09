"""Producer-side event sinks.

An :class:`~server.executor.types.EventSink` is *where the engine hands off*
client-facing events. Isolating this behind a tiny protocol lets the same
producer call sites (``events.py``, ``worker.py``) work unchanged whether the
consumer is:

- an in-process ``queue.Queue`` drained by a pump thread, or
- a socket to a separate engine process.

Only ``queue``-backed sinks live here; the socket sink arrives later.
"""

from __future__ import annotations

from queue import Queue

from server.executor.types import Event


class SharedQueueSink:
    """Emit events onto one server-wide ``queue.Queue`` drained by the pump.

    Every request state shares a single instance, so the pump thread has just
    one queue to drain and route by ``request_id``.
    """

    def __init__(self) -> None:
        self._queue: Queue[Event] = Queue()

    @property
    def queue(self) -> Queue[Event]:
        return self._queue

    def emit(self, event: Event) -> None:
        self._queue.put(event)
