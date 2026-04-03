from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScenarioRequest:
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int | None = None
    weight: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    requests: list[ScenarioRequest]


@dataclass(frozen=True)
class RequestPlan:
    ordinal: int
    scenario_name: str
    payload: dict[str, Any]
    prompt_length_chars: int
    prompt_source: str
    metadata: dict[str, Any]


@dataclass
class RequestResult:
    request_id: str
    run_id: str
    ordinal: int
    scenario_name: str
    endpoint: str
    mode: str
    prompt_source: str
    start_ts: float
    first_token_ts: float | None
    end_ts: float
    latency_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    output_tokens: int | None
    prompt_tokens: int | None
    tokens_per_s: float | None
    queue_wait_ms: float | None
    execution_ms: float | None
    http_status: int | None
    ok: bool
    error_type: str | None
    error: str | None
    prompt_length_chars: int
    response_text_chars: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
