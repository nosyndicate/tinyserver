from typing import Generator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from server.api.schema import GenerateRequest, GenerateResponse, StreamChunk
from server.metrics.logging import log_event
from server.metrics.timers import now_ns, ns_to_ms, timed
from server.model.hf_runner import ModelConfig, load_hf_model
from server.model.sampling import build_sampling_params

router = APIRouter()

_runner = load_hf_model(ModelConfig())


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
