from fastapi import APIRouter

from server.api.schema import GenerateRequest, GenerateResponse
from server.metrics.logging import log_event
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

    text, prompt_tokens, output_tokens = _runner.generate_text(
        req.prompt, sampling_params
    )

    log_event(
        "request_done",
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        total_ms=0.1,
        tokens_per_s=20.0,
    )

    return GenerateResponse(
        text=text,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=0.05,
        total_ms=0.1,
        tokens_per_s=20.0,
    )
