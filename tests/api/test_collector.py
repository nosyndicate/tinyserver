"""Tests for the async consumer primitives.

`asyncio_mode = "auto"` in pyproject means a bare `async def test_...` runs on
a fresh event loop; no decorator needed.
"""

import asyncio

import pytest

from server.api.collector import CollectorRegistry, OutputCollector
from server.executor.types import DoneEvent, ErrorEvent, TokenEvent


def make_token(request_id: str, index: int = 0, token: str = "hi") -> TokenEvent:
    return TokenEvent(
        request_id=request_id,
        token=token,
        is_first=index == 0,
        is_last=False,
        index=index,
    )


def make_done(request_id: str) -> DoneEvent:
    return DoneEvent(
        request_id=request_id,
        text="hi",
        num_prompt_tokens=1,
        num_output_tokens=1,
        ttft=1.0,
        total_ms=2.0,
        queue_wait_ms=0.5,
        execution_ms=1.5,
    )


# --------------------------------------------------------------------------- #
# OutputCollector
# --------------------------------------------------------------------------- #


async def test_get_returns_already_buffered_event() -> None:
    collector = OutputCollector()
    event = make_token("r1")
    collector.put(event)

    # A zero timeout proves the fast path: if `get` awaited at all, `wait_for`
    # would time out immediately instead of returning the buffered event.
    assert await collector.get(timeout=0) is event


async def test_get_preserves_put_order() -> None:
    collector = OutputCollector()
    for index in range(3):
        collector.put(make_token("r1", index=index))

    events = [await collector.get(timeout=0) for _ in range(3)]
    assert [e.index for e in events if isinstance(e, TokenEvent)] == [0, 1, 2]


async def test_burst_coalesces_then_collector_rearms() -> None:
    """A burst of puts is drained without re-waiting, and the flag resets after."""
    collector = OutputCollector()
    for index in range(5):
        collector.put(make_token("r1", index=index))

    for index in range(5):
        event = await collector.get(timeout=0)
        assert isinstance(event, TokenEvent)
        assert event.index == index

    # The buffer emptied, so `clear()` must have run: the next get has to wait
    # (and therefore time out) rather than seeing a stale set flag.
    with pytest.raises(TimeoutError):
        await collector.get(timeout=0.01)


async def test_get_times_out_when_no_event_arrives() -> None:
    collector = OutputCollector()
    with pytest.raises(TimeoutError):
        await collector.get(timeout=0.01)


async def test_get_wakes_on_a_later_put() -> None:
    """The suspend-then-resume path: `get` is awaiting before the event exists."""
    collector = OutputCollector()
    event = make_done("r1")
    asyncio.get_running_loop().call_later(0.01, collector.put, event)

    assert await collector.get(timeout=2.0) is event


async def test_get_recovers_after_a_timeout() -> None:
    """A timed-out get must not leave the collector wedged."""
    collector = OutputCollector()
    with pytest.raises(TimeoutError):
        await collector.get(timeout=0.01)

    event = make_token("r1")
    collector.put(event)
    assert await collector.get(timeout=0) is event


# --------------------------------------------------------------------------- #
# CollectorRegistry
# --------------------------------------------------------------------------- #


async def test_dispatch_routes_by_request_id() -> None:
    registry = CollectorRegistry()
    first = registry.register("r1")
    second = registry.register("r2")

    event = make_token("r2")
    registry.dispatch(event)

    assert await second.get(timeout=0) is event
    # r1's mailbox is untouched — no cross-talk between requests.
    with pytest.raises(TimeoutError):
        await first.get(timeout=0.01)


def test_dispatch_drops_events_for_unknown_request_id() -> None:
    """A late event after unregister is normal; it must not raise."""
    registry = CollectorRegistry()
    registry.register("r1")
    registry.unregister("r1")

    registry.dispatch(make_token("r1"))
    registry.dispatch(make_token("never-registered"))


def test_unregister_is_idempotent() -> None:
    registry = CollectorRegistry()
    registry.register("r1")

    registry.unregister("r1")
    registry.unregister("r1")

    assert len(registry) == 0


async def test_register_rejects_duplicate_request_id_without_replacing_collector() -> (
    None
):
    registry = CollectorRegistry()
    original = registry.register("r1")

    with pytest.raises(ValueError, match="collector already registered"):
        registry.register("r1")

    event = make_token("r1")
    registry.dispatch(event)
    assert await original.get(timeout=0) is event
    assert len(registry) == 1


def test_len_tracks_live_requests() -> None:
    registry = CollectorRegistry()
    assert len(registry) == 0

    registry.register("r1")
    registry.register("r2")
    assert len(registry) == 2

    registry.unregister("r1")
    assert len(registry) == 1


async def test_fail_all_delivers_a_tagged_error_to_every_collector() -> None:
    registry = CollectorRegistry()
    collectors = {rid: registry.register(rid) for rid in ("r1", "r2", "r3")}

    registry.fail_all("server shutting down")

    for request_id, collector in collectors.items():
        event = await collector.get(timeout=0)
        assert isinstance(event, ErrorEvent)
        # Each collector gets an error carrying its *own* request_id, so the
        # handler can log/report it like any other engine-side failure.
        assert event.request_id == request_id
        assert event.error == "server shutting down"


async def test_fail_all_preserves_events_already_buffered() -> None:
    """A shutdown error queues behind real output rather than replacing it."""
    registry = CollectorRegistry()
    collector = registry.register("r1")
    token = make_token("r1")
    collector.put(token)

    registry.fail_all("server shutting down")

    assert await collector.get(timeout=0) is token
    assert isinstance(await collector.get(timeout=0), ErrorEvent)


def test_fail_all_with_no_live_requests_is_a_noop() -> None:
    CollectorRegistry().fail_all("server shutting down")
