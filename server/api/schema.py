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
    ttft_ms: float = Field(
        ..., ge=0.0, description="Time to first token in milliseconds"
    )
    total_ms: float = Field(
        ..., ge=0.0, description="Total time for generation in milliseconds"
    )
    tokens_per_s: float = Field(
        ..., ge=0.0, description="Generation speed in tokens per second"
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
