import uuid
from queue import Full
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from server.api.collector import CollectorRegistry, OutputCollector
from server.api.schema import GenerateRequest, GenerateResponse, StreamChunk
from server.executor.types import (
    DoneEvent,
    ErrorEvent,
    Event,
    EventSink,
    GenerationRequestState,
    TokenEvent,
)
from server.executor.worker import Worker, WorkerShuttingDown
from server.metrics.logging import log_event
from server.model.determinism import make_generator
from server.model.sampling import build_sampling_params

_GENERATION_TIMEOUT_S = 300  # 5 minutes


def _compute_tokens_per_s(num_output_tokens: int, total_ms: float) -> float:
    """Compute tokens per second from output token count and total time."""
    return (num_output_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0


health_router = APIRouter()
v2_router = APIRouter()
v3_router = APIRouter()
v4_router = APIRouter()


def _get_worker(request: Request) -> Worker:
    """
    Retrieve the worker instance from the request's app state.
    """
    worker = request.app.state.worker
    if worker is None:
        raise RuntimeError("Worker not found in app state")
    return worker


def _get_registry(request: Request) -> CollectorRegistry:
    """
    Retrieve the collector registry from the request's app state.
    """
    registry = request.app.state.registry
    if registry is None:
        raise RuntimeError("Collector registry not found in app state")
    return registry


def _get_sink(request: Request) -> EventSink:
    """
    Retrieve the server-wide event sink from the request's app state.
    """
    sink = request.app.state.sink
    if sink is None:
        raise RuntimeError("Event sink not found in app state")
    return sink


def _build_request_state(
    req: GenerateRequest, device: str, sink: EventSink
) -> GenerationRequestState:
    """
    Build a GenerationRequestState from the incoming request and device.
    """
    sampling_params = build_sampling_params(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        seed=req.seed,
    )

    generator = make_generator(req.seed, device)

    return GenerationRequestState(
        request_id=str(uuid.uuid4()),
        sampling_params=sampling_params,
        prompt=req.prompt,
        generator=generator,
        sink=sink,
    )


async def _await_generation(
    state: GenerationRequestState,
    worker: Worker,
    collector: OutputCollector,
    registry: CollectorRegistry,
    timeout: float = _GENERATION_TIMEOUT_S,
) -> GenerateResponse:
    """
    Await this request's collector for a terminal event and build the
    non-streaming response. Intermediate TokenEvents are discarded.
    """
    try:
        while True:
            try:
                event: Event = await collector.get(timeout=timeout)
            except TimeoutError:
                # Client is giving up: tell the worker to stop decoding this
                # request and free its KV blocks instead of running to
                # max_new_tokens.
                worker.cancel(state)
                raise HTTPException(
                    status_code=504,
                    detail="Generation timed out.",
                )

            if isinstance(event, TokenEvent):
                continue
            elif isinstance(event, DoneEvent):
                tokens_per_s = _compute_tokens_per_s(
                    event.num_output_tokens, event.total_ms
                )

                return GenerateResponse(
                    text=event.text,
                    prompt_tokens=event.num_prompt_tokens,
                    output_tokens=event.num_output_tokens,
                    ttft_ms=event.ttft,
                    total_ms=event.total_ms,
                    tokens_per_s=tokens_per_s,
                    queue_wait_ms=event.queue_wait_ms,
                    execution_ms=event.execution_ms,
                )
            elif isinstance(event, ErrorEvent):
                raise HTTPException(
                    status_code=500,
                    detail=f"Generation failed: {event.error}",
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Unexpected event type received from worker",
                )
    finally:
        # Cancel on every exit path and stop routing to a collector nobody is
        # reading.
        worker.cancel(state)
        registry.unregister(state.request_id)


async def _stream_generation(
    state: GenerationRequestState,
    worker: Worker,
    collector: OutputCollector,
    registry: CollectorRegistry,
    timeout: float = _GENERATION_TIMEOUT_S,
) -> AsyncGenerator[str, None]:
    """
    Consume this request's collector and yield SSE chunks. The final token
    is held until the terminal DoneEvent so the last chunk carries the same
    timing metadata as the non-streaming endpoint.

    The ``finally`` cancels the request on every exit path — timeout return,
    the ``GeneratorExit``/``CancelledError`` Starlette throws in on a client
    disconnect, and normal completion. On normal completion the request is
    already reaped, so cancel is a harmless no-op; on abandonment it frees the
    KV blocks instead of decoding to max_new_tokens.
    """
    try:
        while True:
            try:
                event: Event = await collector.get(timeout=timeout)
            except TimeoutError:
                error_chunk = StreamChunk(
                    token_str="",
                    is_first=False,
                    is_done=True,
                    error="Generation timed out.",
                )
                yield f"data: {error_chunk.model_dump_json(exclude_none=True)}\n\n"
                return

            if isinstance(event, TokenEvent):
                if event.is_last:
                    try:
                        done_event = await collector.get(timeout=timeout)
                    except TimeoutError:
                        error_chunk = StreamChunk(
                            token_str="",
                            is_first=False,
                            is_done=True,
                            error="Generation timed out waiting for final event.",
                        )
                        yield f"data: {error_chunk.model_dump_json(exclude_none=True)}\n\n"
                        return
                    if isinstance(done_event, DoneEvent):
                        tokens_per_s = _compute_tokens_per_s(
                            done_event.num_output_tokens, done_event.total_ms
                        )
                        chunk = StreamChunk(
                            token_str=event.token,
                            is_first=event.is_first,
                            is_done=True,
                            prompt_tokens=done_event.num_prompt_tokens,
                            output_tokens=done_event.num_output_tokens,
                            ttft_ms=done_event.ttft,
                            total_ms=done_event.total_ms,
                            tokens_per_s=tokens_per_s,
                            queue_wait_ms=done_event.queue_wait_ms,
                            execution_ms=done_event.execution_ms,
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                        log_event(
                            "stream_done",
                            output_tokens=done_event.num_output_tokens,
                            ttft_ms=done_event.ttft,
                        )
                    elif isinstance(done_event, ErrorEvent):
                        error_chunk = StreamChunk(
                            token_str="",
                            is_first=False,
                            is_done=True,
                            error=done_event.error,
                        )
                        yield f"data: {error_chunk.model_dump_json(exclude_none=True)}\n\n"
                    else:
                        # Unexpected event type after last token, yield a generic error
                        error_chunk = StreamChunk(
                            token_str="",
                            is_first=False,
                            is_done=True,
                            error="Unexpected event type after last token",
                        )
                        yield f"data: {error_chunk.model_dump_json(exclude_none=True)}\n\n"

                    return
                chunk = StreamChunk(
                    token_str=event.token,
                    is_first=event.is_first,
                    is_done=False,
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

            elif isinstance(event, ErrorEvent):
                chunk = StreamChunk(
                    token_str="",
                    is_first=False,
                    is_done=True,
                    error=event.error,
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                return
    finally:
        # No matter how we exit the generator:
        # - normal finish (DoneEvent and return)
        # - timeout awaiting the collector
        # - client disconnect (Starlette throws GeneratorExit/CancelledError in
        #   at the yield)
        # cancel the request to free KV blocks, and stop routing to a collector
        # nobody is reading.
        worker.cancel(state)
        registry.unregister(state.request_id)


def _submit_or_fail(
    request: Request, req: GenerateRequest
) -> tuple[GenerationRequestState, OutputCollector]:
    """
    Register this request's collector and submit it to the worker, raising an
    HTTPException if the worker is at capacity.

    Registration happens *before* submit: the engine can emit a first token as
    soon as it accepts the request, and an event that arrives before its
    collector exists would be dropped as an unknown rid.

    Stays a plain ``def`` because ``Worker.submit`` is non-blocking
    (``put_nowait``), so it is safe to call from a coroutine.
    """
    worker = _get_worker(request)
    registry = _get_registry(request)
    state = _build_request_state(
        req, device=request.app.state.device, sink=_get_sink(request)
    )
    collector = registry.register(state.request_id)
    try:
        worker.submit(state)
    except Full:
        registry.unregister(state.request_id)
        raise HTTPException(
            status_code=503,
            detail="Server at capacity. Please try again later.",
        )
    except WorkerShuttingDown:
        registry.unregister(state.request_id)
        raise HTTPException(
            status_code=503,
            detail="Worker is shutting down. Please try again later.",
        )
    return state, collector


@health_router.get("/health")
async def health() -> dict[str, bool]:
    """Liveness probe."""
    return {"ok": True}


@v2_router.post("/generate_v2", response_model=GenerateResponse)
async def generate_v2(req: GenerateRequest, request: Request) -> GenerateResponse:
    state, collector = _submit_or_fail(request, req)

    return await _await_generation(
        state, _get_worker(request), collector, _get_registry(request)
    )


@v3_router.post("/generate_v3", response_model=GenerateResponse)
async def generate_v3(req: GenerateRequest, request: Request) -> GenerateResponse:
    state, collector = _submit_or_fail(request, req)

    return await _await_generation(
        state, _get_worker(request), collector, _get_registry(request)
    )


@v4_router.post("/generate_v4", response_model=GenerateResponse)
async def generate_v4(req: GenerateRequest, request: Request) -> GenerateResponse:
    """Generate text via the paged-attention scheduled engine."""
    state, collector = _submit_or_fail(request, req)

    return await _await_generation(
        state, _get_worker(request), collector, _get_registry(request)
    )


@v2_router.post("/generate/stream_v2", response_model=None)
async def generate_stream_v2(
    req: GenerateRequest, request: Request
) -> StreamingResponse:
    state, collector = _submit_or_fail(request, req)

    return StreamingResponse(
        _stream_generation(
            state, _get_worker(request), collector, _get_registry(request)
        ),
        media_type="text/event-stream",
    )


@v3_router.post("/generate/stream_v3", response_model=None)
async def generate_stream_v3(
    req: GenerateRequest, request: Request
) -> StreamingResponse:
    state, collector = _submit_or_fail(request, req)

    return StreamingResponse(
        _stream_generation(
            state, _get_worker(request), collector, _get_registry(request)
        ),
        media_type="text/event-stream",
    )


@v4_router.post("/generate/stream_v4", response_model=None)
async def generate_stream_v4(
    req: GenerateRequest, request: Request
) -> StreamingResponse:
    """Stream generated tokens via the paged-attention scheduled engine."""
    state, collector = _submit_or_fail(request, req)

    return StreamingResponse(
        _stream_generation(
            state, _get_worker(request), collector, _get_registry(request)
        ),
        media_type="text/event-stream",
    )
