from server.executor.sinks import SharedQueueSink
from server.executor.types import TokenEvent
from tests.executor.worker_helpers import make_req


def _token(rid: str, index: int) -> TokenEvent:
    return TokenEvent(
        request_id=rid,
        token=f"t{index}",
        is_first=index == 0,
        is_last=False,
        index=index,
    )


def test_shared_queue_sink_emits_to_its_queue() -> None:
    sink = SharedQueueSink()

    sink.emit(_token("r0", 0))
    sink.emit(_token("r1", 0))

    assert sink.queue.get_nowait().request_id == "r0"
    assert sink.queue.get_nowait().request_id == "r1"
    assert sink.queue.empty()


def test_test_helper_provides_an_explicit_shared_queue_sink() -> None:
    # Test request factories explicitly provide their sink; production request
    # states must always receive their transport sink from the caller.
    req = make_req()
    assert isinstance(req.sink, SharedQueueSink)

    event = _token(req.request_id, 0)
    req.sink.emit(event)

    assert req.sink.queue.get_nowait() is event
    assert req.sink.queue.empty()
