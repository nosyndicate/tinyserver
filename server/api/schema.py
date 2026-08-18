from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class StreamTokenEvent(BaseModel):
    """One sampled non-EOS output token, even when its text is empty."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["token"] = "token"
    token_str: str = Field(..., description="The generated token string")
    index: int = Field(..., ge=0, description="Zero-based output-token index")


class StreamDoneEvent(BaseModel):
    """Successful terminal event with authoritative counts and timings."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["done"] = "done"
    finish_reason: Literal["eos", "max_length"]
    prompt_tokens: int = Field(..., ge=0, description="Prompt token count")
    output_tokens: int = Field(..., ge=0, description="Output token count")
    ttft_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Time from enqueue to the first token, in milliseconds",
    )
    total_ms: float = Field(
        ...,
        ge=0.0,
        description="Time from enqueue to completion, in milliseconds",
    )
    tokens_per_s: float = Field(
        ...,
        ge=0.0,
        description="Decode rate: output tokens divided by execution_ms",
    )
    queue_wait_ms: float = Field(
        ...,
        ge=0.0,
        description="Time from enqueue to the start of the first prefill",
    )
    execution_ms: float = Field(
        ...,
        ge=0.0,
        description="Time from the start of the first prefill to completion",
    )


class StreamErrorEvent(BaseModel):
    """Terminal stream failure."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["error"] = "error"
    error: str


StreamEvent = Annotated[
    StreamTokenEvent | StreamDoneEvent | StreamErrorEvent,
    Field(discriminator="type"),
]
