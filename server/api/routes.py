from queue import Full
from typing import Generator

import torch
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from server.api.schema import GenerateRequest, GenerateResponse, StreamChunk
from server.executor.types import DoneEvent, ErrorEvent, Event, GenerationRequestState
from server.executor.worker import Worker
from server.metrics.logging import log_event
from server.metrics.timers import now_ns, ns_to_ms, timed
from server.model.hf_runner import LOWEST_TEMPERATURE, ModelConfig, load_hf_model
from server.model.sampling import build_sampling_params

router = APIRouter()

_runner = load_hf_model(ModelConfig())


def _get_worker(request: Request) -> Worker:
    """
    Retrieve the worker instance from the request's app state.
    """
    worker = request.app.state.worker
    assert worker is not None, "Worker not found in app state"
    return worker


def _build_request_state(req: GenerateRequest, device: str) -> GenerationRequestState:
    sampling_params = build_sampling_params(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
    )

    # Create a per-request random generator if temperature > 0 and seed is provided
    if req.temperature is not None and req.temperature > LOWEST_TEMPERATURE:
        generator = torch.Generator(req.seed, device=device)
    else:
        generator = None

    return GenerationRequestState(
        request_id="TODO",  # Generate a unique ID for this request
        sampling_params=sampling_params,
        prompt=req.prompt,
        generator=generator,
    )


@router.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate text based on the input prompt."""

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
        text, prompt_tokens, output_tokens = _runner.generate_text(
            req.prompt, sampling_params
        )

    total_ms = ns_to_ms(total_timer["end_ns"] - total_timer["start_ns"])

    # We don't have streaming right now, so ttft_ms is the same as total_ms.
    ttft_ms = total_ms
    tokens_per_s = (output_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0

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
    )


@router.post("/generate_v2", response_model=GenerateResponse)
def generate_v2(
    req: GenerateRequest, request: Request
) -> GenerateResponse | JSONResponse:
    worker = _get_worker(request)
    state = _build_request_state(req, device=request.app.state.device)

    try:
        worker.submit(state)
    except Full:
        return JSONResponse(
            status_code=503,
            content={"error": "Server at capacity. Please try again later."},
        )

    while True:
        event: Event = state.output_queue.get()  # block until there is an event

        # Since this is the non-streaming endpoint, we expect either a
        # DoneEvent or an ErrorEvent as the final event.
        if isinstance(event, DoneEvent):
            tokens_per_s = (
                (event.num_output_tokens / (event.total_ms / 1000.0))
                if event.total_ms > 0
                else 0.0
            )

            return GenerateResponse(
                text=event.text,
                prompt_tokens=event.num_prompt_tokens,
                output_tokens=event.num_output_tokens,
                ttft_ms=event.ttft,
                total_ms=event.total_ms,
                tokens_per_s=tokens_per_s,
            )
        elif isinstance(event, ErrorEvent):
            return JSONResponse(
                status_code=500,
                content={"error": f"Generation failed: {event.error}"},
            )


@router.post("/generate/stream")
def generate_stream(req: GenerateRequest) -> StreamingResponse:
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

        for token_str, is_first_token, is_done in _runner.generate_stream(
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
