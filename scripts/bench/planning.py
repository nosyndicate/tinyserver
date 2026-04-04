from __future__ import annotations

from typing import Any

from .models import RequestPlan, Scenario, ScenarioRequest


def _build_request_plans(
    scenario: Scenario,
    total_requests: int,
    prompt_override: str | None,
    max_new_tokens_override: int | None,
    temperature_override: float | None,
    top_p_override: float | None,
    seed_override: int | None,
) -> list[RequestPlan]:
    weighted_requests: list[ScenarioRequest] = []
    for req in scenario.requests:
        weighted_requests.extend([req] * max(req.weight, 1))

    if not weighted_requests:
        raise ValueError(
            f"scenario {scenario.name!r} does not contain any request templates"
        )

    plans: list[RequestPlan] = []
    for ordinal in range(total_requests):
        req = weighted_requests[ordinal % len(weighted_requests)]
        prompt = prompt_override if prompt_override is not None else req.prompt
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens_override
            if max_new_tokens_override is not None
            else req.max_new_tokens,
            "temperature": temperature_override
            if temperature_override is not None
            else req.temperature,
            "top_p": top_p_override if top_p_override is not None else req.top_p,
        }
        seed = seed_override if seed_override is not None else req.seed
        if seed is not None:
            payload["seed"] = seed
        plans.append(
            RequestPlan(
                ordinal=ordinal,
                scenario_name=scenario.name,
                payload=payload,
                prompt_length_chars=len(prompt),
                prompt_source=req.metadata.get("class", "default"),
                metadata=dict(req.metadata),
            )
        )
    return plans
