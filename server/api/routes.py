from fastapi import APIRouter

from server.api.schema import GenerateRequest, GenerateResponse
from server.metrics.logging import log_event
from server.metrics.timers import ns_to_ms, timed
from server.model.hf_runner import ModelConfig, ModelRunner
from server.model.sampling import build_sampling_params

router = APIRouter()

_runner = ModelRunner(ModelConfig())


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
        stops=req.stops,
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
