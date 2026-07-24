"""Tests for the Stage-1 output pump in ``server.api.pump``.

The pump is the thread that bridges a shared ``queue.Queue`` (producer side) to
the asyncio loop (consumer side). These tests exercise the cross-thread hop,
the shutdown drain, and the loop-closed guard.
"""

from __future__ import annotations

import asyncio

from server.api.collector import CollectorRegistry
from server.api.pump import OutputPump
from server.executor.sinks import SharedQueueSink
from server.executor.types import TokenEvent


def _token(request_id: str, token: str) -> TokenEvent:
    return TokenEvent(
        request_id=request_id, token=token, is_first=True, is_last=False, index=0
    )


def test_pump_dispatches_event_across_the_thread_boundary() -> None:
    loop = asyncio.new_event_loop()
    try:
        sink = SharedQueueSink()
        registry = CollectorRegistry()
        collector = registry.register("r1")
        pump = OutputPump(sink.queue, registry)
        pump.start(loop)

        async def scenario() -> str:
            # Producer emits onto the shared queue from the loop thread; the
            # pump thread picks it up and forwards it back onto the loop.
            sink.emit(_token("r1", "hi"))
            event = await collector.get(timeout=2.0)
            assert isinstance(event, TokenEvent)
            return event.token

        assert loop.run_until_complete(scenario()) == "hi"
        pump.stop()
    finally:
        loop.close()


def test_stop_drains_events_left_on_the_queue() -> None:
    # An event queued right before shutdown must not be lost: stop() drains the
    # queue synchronously after joining the thread. We drive the drain path
    # directly (no thread) by handing the pump the loop reference, so the event
    # can only reach the collector via the drain in stop().
    loop = asyncio.new_event_loop()
    try:
        sink = SharedQueueSink()
        registry = CollectorRegistry()
        collector = registry.register("r1")
        pump = OutputPump(sink.queue, registry)
        pump._loop = loop  # loop captured without starting the pump thread

        sink.emit(_token("r1", "bye"))
        pump.stop()  # thread is None -> skip join, then drain the queue

        # stop() scheduled dispatch via call_soon_threadsafe; running the loop
        # to await the event executes that scheduled callback.
        event = loop.run_until_complete(collector.get(timeout=1.0))
        assert isinstance(event, TokenEvent)
        assert event.token == "bye"
    finally:
        loop.close()


def test_forward_drops_event_when_loop_is_closed() -> None:
    # If the loop is already closed, call_soon_threadsafe raises RuntimeError;
    # the pump must catch it and drop the event rather than crash.
    loop = asyncio.new_event_loop()
    loop.close()

    sink = SharedQueueSink()
    registry = CollectorRegistry()
    registry.register("r1")
    pump = OutputPump(sink.queue, registry)
    pump._loop = loop

    sink.emit(_token("r1", "lost"))
    pump.stop()  # drains into the closed loop; must not raise
