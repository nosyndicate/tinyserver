"""v1 endpoints — the preserved phase-1 baseline (learning artifact).

This module is kept deliberately as a trace of the project's learning path:
the direct, unqueued serving design from phase 1. Its characteristics are
part of the exhibit, not defects to fix:

- The handlers are sync ``def``, so each request runs on (and blocks) one
  AnyIO threadpool thread for the entire GPU generation.
- There is no worker, no queue, no cancellation, and no timeout — the model
  runs inline in the HTTP handler via ``ModelRunner``.
- ``queue_wait_ms`` is hardcoded to 0 and ``ttft_ms == total_ms`` for the
  non-streaming endpoint, because there is no queue and no incremental
  delivery to measure.

The later versions (v2+) exist precisely because of these limitations.

Served only when the server is started with ``--api-version v1``; the
queue-based versions never mount this router (see ``server/main.py``).
"""

from typing import Generator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from server.api.schema import GenerateRequest, GenerateResponse, StreamChunk
from server.metrics.logging import log_event
from server.metrics.timers import now_ns, ns_to_ms, timed
from server.model.hf_runner import ModelRunner
from server.model.sampling import build_sampling_params

v1_router = APIRouter()


def _compute_tokens_per_s(num_output_tokens: int, total_ms: float) -> float:
    """Compute tokens per second from output token count and total time.

    Duplicated from ``routes.py`` on purpose so this module stays
    self-contained and one import away from deletion.
    """
    return (num_output_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0


def _get_runner(request: Request) -> ModelRunner:
    """
    Retrieve the model runner instance from the request's app state.
    """
    runner = request.app.state.runner
    if runner is None:
        raise RuntimeError("Model runner not found in app state")
    return runner


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
