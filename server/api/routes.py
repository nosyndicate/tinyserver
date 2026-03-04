from fastapi import APIRouter

from server.api.schema import GenerateRequest, GenerateResponse
from server.metrics.logging import log_event

router = APIRouter()


@router.get("/health")
def health():
    return {"ok": True}


@router.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate text based on the input prompt."""

    log_event(
        "request_done",
        prompt_tokens=1,
        output_tokens=1,
        total_ms=0.1,
        tokens_per_s=20.0,
    )

    return GenerateResponse(
        text="test",
        prompt_tokens=1,
        output_tokens=1,
        ttft_ms=0.05,
        total_ms=0.1,
        tokens_per_s=20.0,
    )
