"""Tests for the thread → loop bridge.

Deleted along with ``server/api/pump.py`` once the engine moves into its own
process. Everything worth keeping is covered by ``test_collector.py``.
"""

import asyncio
from queue import Queue

import pytest

from server.api.collector import CollectorRegistry
from server.api.pump import OutputPump
from server.executor.types import ErrorEvent, Event, TokenEvent
from tests.api.test_collector import make_done, make_token

# Short enough that stop()'s join is quick, long enough not to spin.
_POLL_S = 0.01


async def test_pump_delivers_events_across_the_thread_boundary() -> None:
    """The whole reason this class exists: a put from another thread wakes a
    coroutine that was suspended on the loop."""
    registry = CollectorRegistry()
    collector = registry.register("r1")
    queue: Queue[Event] = Queue()
    pump = OutputPump(queue, registry, poll_timeout_s=_POLL_S)
    pump.start(asyncio.get_running_loop())

    try:
        event = make_token("r1")
        # This test function is the stand-in for the engine thread: it puts on
        # the shared queue without touching any asyncio object.
        queue.put(event)

        assert await collector.get(timeout=2.0) is event
    finally:
        await pump.stop_and_flush()


async def test_pump_routes_each_request_to_its_own_collector() -> None:
    registry = CollectorRegistry()
    first = registry.register("r1")
    second = registry.register("r2")
    queue: Queue[Event] = Queue()
    pump = OutputPump(queue, registry, poll_timeout_s=_POLL_S)
    pump.start(asyncio.get_running_loop())

    try:
        queue.put(make_token("r2", token="second"))
        queue.put(make_token("r1", token="first"))

        from_second = await second.get(timeout=2.0)
        from_first = await first.get(timeout=2.0)
        assert isinstance(from_first, TokenEvent) and from_first.token == "first"
        assert isinstance(from_second, TokenEvent) and from_second.token == "second"
    finally:
        await pump.stop_and_flush()


async def test_stop_drains_queued_events() -> None:
    """Events sitting on the queue at shutdown still reach their handler.

    This is what keeps ``Worker.stop()``'s final cancel ErrorEvents from being
    swallowed: whether the pump thread or ``stop()``'s synchronous drain picks
    each one up, it gets dispatched either way.
    """
    registry = CollectorRegistry()
    collector = registry.register("r1")
    queue: Queue[Event] = Queue()
    pump = OutputPump(queue, registry, poll_timeout_s=_POLL_S)
    pump.start(asyncio.get_running_loop())

    for index in range(5):
        queue.put(make_token("r1", index=index))

    await pump.stop_and_flush()

    events = [await collector.get(timeout=0) for _ in range(5)]
    assert [e.index for e in events if isinstance(e, TokenEvent)] == [0, 1, 2, 3, 4]


async def test_stop_and_flush_preserves_done_before_shutdown_error() -> None:
    registry = CollectorRegistry()
    collector = registry.register("r1")
    queue: Queue[Event] = Queue()
    pump = OutputPump(queue, registry, poll_timeout_s=_POLL_S)
    pump.start(asyncio.get_running_loop())

    done = make_done("r1")
    queue.put(done)

    await pump.stop_and_flush()
    registry.fail_all("Server is shutting down")

    assert await collector.get(timeout=0) is done
    error = await collector.get(timeout=0)
    assert isinstance(error, ErrorEvent)
    assert error.error == "Server is shutting down"


def test_pump_drops_events_when_the_loop_is_closed() -> None:
    """The shutdown race: ``call_soon_threadsafe`` on a closed loop raises
    ``RuntimeError``, which must be swallowed rather than killing the thread.

    Deliberately a sync test — it needs a loop it owns and has already closed.
    """
    loop = asyncio.new_event_loop()
    loop.close()

    registry = CollectorRegistry()
    collector = registry.register("r1")
    queue: Queue[Event] = Queue()
    queue.put(make_token("r1"))

    pump = OutputPump(queue, registry, poll_timeout_s=_POLL_S)
    pump.start(loop)
    pump.stop()  # must not raise, and must not leave the thread wedged

    assert queue.empty()  # consumed, then dropped

    # Nothing was delivered. Constructing the collector off-loop above is fine:
    # asyncio.Event binds to no loop until it is first awaited, which happens here.
    with pytest.raises(TimeoutError):
        asyncio.run(collector.get(timeout=0.01))


def test_start_twice_is_rejected() -> None:
    registry = CollectorRegistry()
    queue: Queue[Event] = Queue()
    loop = asyncio.new_event_loop()
    pump = OutputPump(queue, registry, poll_timeout_s=_POLL_S)

    try:
        pump.start(loop)
        with pytest.raises(RuntimeError):
            pump.start(loop)
    finally:
        pump.stop()
        loop.close()
