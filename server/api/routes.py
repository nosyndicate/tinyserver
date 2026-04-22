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
from server.executor.worker import Worker
from server.metrics.logging import log_event
from server.metrics.timers import now_ns, ns_to_ms, timed
from server.model.determinism import make_generator
from server.model.hf_runner import ModelRunner
from server.model.sampling import build_sampling_params

_GENERATION_TIMEOUT_S = 300  # 5 minutes


def _compute_tokens_per_s(num_output_tokens: int, total_ms: float) -> float:
    """Compute tokens per second from output token count and total time."""
    return (num_output_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0


health_router = APIRouter()
v1_router = APIRouter()
v2_router = APIRouter()
v3_router = APIRouter()


def _get_runner(request: Request) -> ModelRunner:
    """
    Retrieve the model runner instance from the request's app state.
    """
    runner = request.app.state.runner
    if runner is None:
        raise RuntimeError("Model runner not found in app state")
    return runner


def _get_worker(request: Request) -> Worker:
    """
    Retrieve the worker instance from the request's app state.
    """
    worker = request.app.state.worker
    if worker is not None:
        return worker
    worker = request.app.state.batch_worker
    if worker is not None:
        return worker
    raise RuntimeError("Worker not found in app state")


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


@health_router.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@v1_router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    """Generate text based on the input prompt."""

    runner = _get_runner(request)
    sampling_params = build_sampling_params(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
    )

    log_event(
        "request_start",
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    with timed() as total_timer:
        text, prompt_tokens, output_tokens = runner.generate_text(
            req.prompt, sampling_params
        )

    total_ms = ns_to_ms(total_timer["end_ns"] - total_timer["start_ns"])

    # We don't have streaming right now, so ttft_ms is the same as total_ms.
    ttft_ms = total_ms
    tokens_per_s = _compute_tokens_per_s(output_tokens, total_ms)

    log_event(
        "request_done",
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_ms=total_ms,
        tokens_per_s=tokens_per_s,
    )

    return GenerateResponse(
        text=text,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        tokens_per_s=tokens_per_s,
        queue_wait_ms=0.0,
        execution_ms=total_ms,  # V1 doesn't queue, so execution_ms == total_ms
    )


@v2_router.post("/generate_v2", response_model=GenerateResponse)
def generate_v2(req: GenerateRequest, request: Request) -> GenerateResponse:
    worker = _get_worker(request)
    state = _build_request_state(req, device=request.app.state.device)

    try:
        worker.submit(state)
    except Full:
        raise HTTPException(
            status_code=503,
            detail="Server at capacity. Please try again later.",
        )

    while True:
        try:
            event: Event = state.output_queue.get(timeout=_GENERATION_TIMEOUT_S)
        except Empty:
            raise HTTPException(
                status_code=504,
                detail="Generation timed out.",
            )

        # Since this is the non-streaming endpoint, discard intermediate tokens
        # and wait for the terminal DoneEvent or ErrorEvent.
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


@v3_router.post("/generate_v3", response_model=GenerateResponse)
def generate_v3(req: GenerateRequest, request: Request) -> GenerateResponse:
    worker = _get_worker(request)
    state = _build_request_state(req, device=request.app.state.device)

    try:
        worker.submit(state)
    except Full:
        raise HTTPException(
            status_code=503,
            detail="Server at capacity. Please try again later.",
        )

    while True:
        try:
            event: Event = state.output_queue.get(timeout=_GENERATION_TIMEOUT_S)
        except Empty:
            raise HTTPException(
                status_code=504,
                detail="Generation timed out.",
            )

        # Since this is the non-streaming endpoint, discard intermediate tokens
        # and wait for the terminal DoneEvent or ErrorEvent.
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
                detail="Unexpected event type received from batch worker",
            )


@v1_router.post("/generate/stream")
def generate_stream(req: GenerateRequest, request: Request) -> StreamingResponse:
    runner = _get_runner(request)
    sampling_params = build_sampling_params(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
    )

    def _event_stream() -> Generator[str, None, None]:
        start_ns = now_ns()
        ttft_ms = None
        index = 0

        for token_str, is_first_token, is_done in runner.generate_stream(
            req.prompt, sampling_params
        ):
            if is_first_token:
                ttft_ms = ns_to_ms(now_ns() - start_ns)

            if token_str:
                index += 1

            chunk = StreamChunk(
                token_str=token_str,
                is_first=is_first_token,
                is_done=is_done,
            )

            yield f"data: {chunk.model_dump_json()}\n\n"

            if is_done:
                log_event("stream_done", output_tokens=index, ttft_ms=ttft_ms)
                return

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@v2_router.post("/generate/stream_v2", response_model=None)
def generate_stream_v2(req: GenerateRequest, request: Request) -> StreamingResponse:
    worker = _get_worker(request)
    state = _build_request_state(req, device=request.app.state.device)

    try:
        worker.submit(state)
    except Full:
        raise HTTPException(
            status_code=503,
            detail="Server at capacity. Please try again later.",
        )

    def _event_stream() -> Generator[str, None, None]:
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
                # For the final token, wait for the terminal DoneEvent so the last SSE
                # payload can carry the same timing metadata as /generate_v2.
                # Note: this introduces a small latency for the final token as it's
                # held until the executor finishes cleanup and emits DoneEvent.
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

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@v3_router.post("/generate/stream_v3", response_model=None)
def generate_stream_v3(req: GenerateRequest, request: Request) -> StreamingResponse:
    worker = _get_worker(request)
    state = _build_request_state(req, device=request.app.state.device)

    try:
        worker.submit(state)
    except Full:
        raise HTTPException(
            status_code=503,
            detail="Server at capacity. Please try again later.",
        )

    def _event_stream() -> Generator[str, None, None]:
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
                # For the final token, wait for the terminal DoneEvent so the last SSE
                # payload can carry the same timing metadata as /generate_v2.
                # Note: this introduces a small latency for the final token as it's
                # held until the executor finishes cleanup and emits DoneEvent.
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

    return StreamingResponse(_event_stream(), media_type="text/event-stream")
