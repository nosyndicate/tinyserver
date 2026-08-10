from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for text generation")
    max_new_tokens: int = Field(
        128, ge=1, le=4096, description="The maximum number of new tokens to generate"
    )
    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(
        0.95, ge=0.0, le=1.0, description="Nucleus sampling top-p value"
    )
    top_k: int = Field(0, ge=0, description="Top-k filtering; 0 disables it")
    seed: int | None = Field(
        default=None, ge=0, description="Optional random seed for reproducibility"
    )


class GenerateResponse(BaseModel):
    text: str = Field(..., description="The generated text output")
    prompt_tokens: int = Field(
        ..., ge=0, description="Number of tokens in the input prompt"
    )
    output_tokens: int = Field(
        ..., ge=0, description="Number of tokens generated in the output"
    )
    ttft_ms: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Time from enqueue to the first token, in milliseconds. Null when "
            "the request finished without emitting a token."
        ),
    )
    total_ms: float = Field(
        ...,
        ge=0.0,
        description=(
            "Time from enqueue to completion, in milliseconds. Equals "
            "queue_wait_ms + execution_ms."
        ),
    )
    tokens_per_s: float = Field(
        ...,
        ge=0.0,
        description=(
            "Decode rate: output tokens divided by execution_ms. Excludes queue "
            "wait so the value stays comparable across load levels."
        ),
    )
    queue_wait_ms: float = Field(
        0.0,
        ge=0.0,
        description="Time from enqueue to the start of the first prefill",
    )
    execution_ms: float = Field(
        0.0,
        ge=0.0,
        description="Time from the start of the first prefill to completion",
    )


class StreamChunk(BaseModel):
    """A chunk of generated text for streaming responses."""

    token_str: str = Field(..., description="The generated token string")
    is_first: bool = Field(
        ..., description="Whether this is the first token in the output"
    )
    is_done: bool = Field(
        ..., description="Whether this is the last token in the output"
    )
    error: str | None = Field(
        default=None, description="Error message if generation failed, otherwise None"
    )
    prompt_tokens: int | None = Field(
        default=None, ge=0, description="Prompt token count when known"
    )
    output_tokens: int | None = Field(
        default=None, ge=0, description="Output token count when known"
    )
    ttft_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time from enqueue to the first token, in milliseconds",
    )
    total_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time from enqueue to completion, in milliseconds",
    )
    tokens_per_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Decode rate: output tokens divided by execution_ms",
    )
    queue_wait_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time from enqueue to the start of the first prefill",
    )
    execution_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time from the start of the first prefill to completion",
    )
