import uuid
from queue import Empty, Full
from typing import Generator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from server.api.schema import GenerateRequest, GenerateResponse, StreamChunk
from server.executor.types import (
    DoneEvent,
    ErrorEvent,
    Event,
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


def _build_request_state(req: GenerateRequest, device: str) -> GenerationRequestState:
    """
    Build a GenerationRequestState from the incoming request and device.
    """
    sampling_params = build_sampling_params(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
    )

    generator = make_generator(req.seed, device)

    return GenerationRequestState(
        request_id=str(uuid.uuid4()),
        sampling_params=sampling_params,
        prompt=req.prompt,
        generator=generator,
    )


def _await_generation(
    state: GenerationRequestState, worker: Worker
) -> GenerateResponse:
    """
    Wait on the request's output queue for a terminal event and build the
    non-streaming response. Intermediate TokenEvents are discarded.
    """
    while True:
        try:
            event: Event = state.output_queue.get(timeout=_GENERATION_TIMEOUT_S)
        except Empty:
            # Client is giving up: tell the worker to stop decoding this request
            # and free its KV blocks instead of running to max_new_tokens.
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


def _stream_generation(
    state: GenerationRequestState, worker: Worker
) -> Generator[str, None, None]:
    """
    Consume the request's output queue and yield SSE chunks. The final token
    is held until the terminal DoneEvent so the last chunk carries the same
    timing metadata as the non-streaming endpoint.

    The ``finally`` cancels the request on every exit path — timeout return,
    ``GeneratorExit`` from a client disconnect, and normal completion. On
    normal completion the request is already reaped, so cancel is a harmless
    no-op; on abandonment it frees the KV blocks instead of decoding to
    max_new_tokens.
    """
    try:
        while True:
            try:
                event: Event = state.output_queue.get(timeout=_GENERATION_TIMEOUT_S)
            except Empty:
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
                        done_event = state.output_queue.get(
                            timeout=_GENERATION_TIMEOUT_S
                        )
                    except Empty:
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
        # - timeout on output_queue.get
        # - client disconnect (on yield, FastAPI raises GeneratorExit)
        # cancel the request to free KV blocks
        worker.cancel(state)


def _submit_or_fail(request: Request, req: GenerateRequest) -> GenerationRequestState:
    """
    Submit the request state to the worker, raising an HTTPException if the
    worker is at capacity.
    """
    worker = _get_worker(request)
    state = _build_request_state(req, device=request.app.state.device)
    try:
        worker.submit(state)
    except Full:
        raise HTTPException(
            status_code=503,
            detail="Server at capacity. Please try again later.",
        )
    except WorkerShuttingDown:
        raise HTTPException(
            status_code=503,
            detail="Worker is shutting down. Please try again later.",
        )
    return state


@health_router.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@v2_router.post("/generate_v2", response_model=GenerateResponse)
def generate_v2(req: GenerateRequest, request: Request) -> GenerateResponse:
    state = _submit_or_fail(request, req)

    return _await_generation(state, _get_worker(request))


@v3_router.post("/generate_v3", response_model=GenerateResponse)
def generate_v3(req: GenerateRequest, request: Request) -> GenerateResponse:
    state = _submit_or_fail(request, req)

    return _await_generation(state, _get_worker(request))


@v4_router.post("/generate_v4", response_model=GenerateResponse)
def generate_v4(req: GenerateRequest, request: Request) -> GenerateResponse:
    """Generate text via the paged-attention scheduled engine."""
    state = _submit_or_fail(request, req)

    return _await_generation(state, _get_worker(request))


@v2_router.post("/generate/stream_v2", response_model=None)
def generate_stream_v2(req: GenerateRequest, request: Request) -> StreamingResponse:
    state = _submit_or_fail(request, req)

    return StreamingResponse(
        _stream_generation(state, _get_worker(request)),
        media_type="text/event-stream",
    )


@v3_router.post("/generate/stream_v3", response_model=None)
def generate_stream_v3(req: GenerateRequest, request: Request) -> StreamingResponse:
    state = _submit_or_fail(request, req)

    return StreamingResponse(
        _stream_generation(state, _get_worker(request)),
        media_type="text/event-stream",
    )


@v4_router.post("/generate/stream_v4", response_model=None)
def generate_stream_v4(req: GenerateRequest, request: Request) -> StreamingResponse:
    """Stream generated tokens via the paged-attention scheduled engine."""
    state = _submit_or_fail(request, req)

    return StreamingResponse(
        _stream_generation(state, _get_worker(request)),
        media_type="text/event-stream",
    )
