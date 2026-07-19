"""Tests for the Stage-1 queue-backed event sinks."""

from queue import Queue

from server.executor.sinks import DirectQueueSink, SharedQueueSink
from server.executor.types import Event, TokenEvent
from tests.executor.worker_helpers import make_req


def _token(rid: str, index: int) -> TokenEvent:
    return TokenEvent(
        request_id=rid, token=f"t{index}", is_first=index == 0, is_last=False, index=index
    )


def test_direct_queue_sink_emits_in_order() -> None:
    q: Queue[Event] = Queue()
    sink = DirectQueueSink(q)

    sink.emit(_token("r0", 0))
    sink.emit(_token("r0", 1))

    assert q.get_nowait().index == 0
    assert q.get_nowait().index == 1
    assert q.empty()


def test_shared_queue_sink_emits_to_its_queue() -> None:
    sink = SharedQueueSink()

    sink.emit(_token("r0", 0))
    sink.emit(_token("r1", 0))

    assert sink.queue.get_nowait().request_id == "r0"
    assert sink.queue.get_nowait().request_id == "r1"
    assert sink.queue.empty()


def test_default_sink_wraps_output_queue() -> None:
    # A request built without an explicit sink gets a DirectQueueSink that
    # emits onto its own output_queue — the behavior-preserving default.
    req = make_req()
    assert isinstance(req.sink, DirectQueueSink)

    event = _token(req.request_id, 0)
    req.sink.emit(event)

    assert req.output_queue.get_nowait() is event
    assert req.output_queue.empty()
