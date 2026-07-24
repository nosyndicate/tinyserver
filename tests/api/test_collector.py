"""Tests for the async consumer primitives in ``server.api.collector``.

The repo has no ``pytest-asyncio``, so async behaviour is driven with plain
``asyncio.run(...)`` inside ordinary ``def test_...`` functions.
"""

from __future__ import annotations

import asyncio

import pytest

from server.api.collector import CollectorRegistry, OutputCollector
from server.executor.types import ErrorEvent, TokenEvent


def _token(request_id: str, token: str, index: int) -> TokenEvent:
    return TokenEvent(
        request_id=request_id,
        token=token,
        is_first=(index == 0),
        is_last=False,
        index=index,
    )


def test_get_returns_events_in_fifo_order() -> None:
    async def scenario() -> tuple[str, str]:
        collector = OutputCollector()
        collector.put(_token("r1", "a", 0))
        collector.put(_token("r1", "b", 1))
        first = await collector.get(timeout=1.0)
        second = await collector.get(timeout=1.0)
        assert isinstance(first, TokenEvent)
        assert isinstance(second, TokenEvent)
        return first.token, second.token

    assert asyncio.run(scenario()) == ("a", "b")


def test_buffered_events_coalesce_and_take_the_fast_path() -> None:
    # Many puts before any get: all buffered, drained in order. A timeout of
    # 0.0 proves the fast path never awaits — a buffered event returns without
    # touching the timer.
    async def scenario() -> list[str]:
        collector = OutputCollector()
        for i in range(5):
            collector.put(_token("r1", str(i), i))
        drained: list[str] = []
        for _ in range(5):
            event = await collector.get(timeout=0.0)
            assert isinstance(event, TokenEvent)
            drained.append(event.token)
        return drained

    assert asyncio.run(scenario()) == ["0", "1", "2", "3", "4"]


def test_get_times_out_when_no_event_arrives() -> None:
    async def scenario() -> None:
        collector = OutputCollector()
        with pytest.raises(asyncio.TimeoutError):
            await collector.get(timeout=0.01)

    asyncio.run(scenario())


def test_get_wakes_on_a_later_put() -> None:
    # Slow path: buffer empty when get is called, event scheduled to arrive
    # afterwards on the same loop.
    async def scenario() -> str:
        collector = OutputCollector()
        loop = asyncio.get_running_loop()
        loop.call_soon(collector.put, _token("r1", "late", 0))
        event = await collector.get(timeout=1.0)
        assert isinstance(event, TokenEvent)
        return event.token

    assert asyncio.run(scenario()) == "late"


def test_registry_dispatch_routes_to_matching_collector() -> None:
    async def scenario() -> str:
        registry = CollectorRegistry()
        collector = registry.register("r1")
        registry.dispatch(_token("r1", "hi", 0))
        event = await collector.get(timeout=1.0)
        assert isinstance(event, TokenEvent)
        return event.token

    assert asyncio.run(scenario()) == "hi"


def test_registry_drops_events_for_unknown_request_id() -> None:
    # After unregister a late event has nowhere to go; dispatch must not raise.
    registry = CollectorRegistry()
    registry.register("r1")
    registry.unregister("r1")
    registry.dispatch(_token("r1", "straggler", 0))  # no exception = pass


def test_registry_unregister_tolerates_missing_id() -> None:
    registry = CollectorRegistry()
    registry.unregister("never-registered")  # no exception = pass


def test_fail_all_pushes_error_into_every_live_collector() -> None:
    async def scenario() -> list[ErrorEvent]:
        registry = CollectorRegistry()
        c1 = registry.register("r1")
        c2 = registry.register("r2")
        registry.fail_all("engine gone")
        e1 = await c1.get(timeout=1.0)
        e2 = await c2.get(timeout=1.0)
        assert isinstance(e1, ErrorEvent)
        assert isinstance(e2, ErrorEvent)
        return [e1, e2]

    errors = asyncio.run(scenario())
    assert {e.request_id for e in errors} == {"r1", "r2"}
    assert all(e.error == "engine gone" for e in errors)
