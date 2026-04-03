from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Scenario, ScenarioRequest

DEFAULT_SCENARIO_FILE = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "scenarios.json"
)


def _default_scenarios() -> dict[str, Scenario]:
    short_prompt = "Write a short haiku about GPUs and scheduling."
    long_prompt = (
        "You are helping profile an inference server. "
        "Explain the tradeoff between latency, throughput, batching, prefill cost, "
        "decode efficiency, queueing, and fairness for online LLM serving. "
        "Then provide a compact checklist for benchmarking and what signals to log. "
        "Keep the explanation dense and technical.\n\n"
        "Context:\n"
        "- The server currently supports sync and streaming endpoints.\n"
        "- Prefill and decode are already separated.\n"
        "- The next steps include a worker, batching, scheduling policy, and KV cache improvements.\n"
        "- The benchmark should surface saturation behavior and queue pressure.\n"
    )

    return {
        "short_short": Scenario(
            name="short_short",
            description="Short prompt and short output; primarily TTFT-focused.",
            requests=[
                ScenarioRequest(
                    prompt=short_prompt,
                    max_new_tokens=48,
                    temperature=0.8,
                    top_p=0.95,
                    metadata={"class": "short"},
                )
            ],
        ),
        "long_long": Scenario(
            name="long_long",
            description="Long prompt and longer output; throughput and prefill stress.",
            requests=[
                ScenarioRequest(
                    prompt=long_prompt,
                    max_new_tokens=192,
                    temperature=0.8,
                    top_p=0.95,
                    metadata={"class": "long"},
                )
            ],
        ),
        "mixed": Scenario(
            name="mixed",
            description="Weighted mix of short and long requests.",
            requests=[
                ScenarioRequest(
                    prompt=short_prompt,
                    max_new_tokens=48,
                    temperature=0.8,
                    top_p=0.95,
                    weight=3,
                    metadata={"class": "short"},
                ),
                ScenarioRequest(
                    prompt=long_prompt,
                    max_new_tokens=192,
                    temperature=0.8,
                    top_p=0.95,
                    weight=1,
                    metadata={"class": "long"},
                ),
            ],
        ),
        "burst": Scenario(
            name="burst",
            description="Short prompt with moderate output, intended for open-loop pressure tests.",
            requests=[
                ScenarioRequest(
                    prompt=(
                        "List five concrete reasons a queue builds up in an LLM inference server "
                        "under bursty traffic."
                    ),
                    max_new_tokens=96,
                    temperature=0.7,
                    top_p=0.9,
                    metadata={"class": "burst"},
                )
            ],
        ),
        "seeded_deterministic": Scenario(
            name="seeded_deterministic",
            description="Determinism and regression scenario with fixed seed.",
            requests=[
                ScenarioRequest(
                    prompt="Summarize how KV cache reuse changes decode cost in one paragraph.",
                    max_new_tokens=64,
                    temperature=0.0,
                    top_p=1.0,
                    seed=7,
                    metadata={"class": "deterministic"},
                )
            ],
        ),
    }


def _load_scenarios(path: str | None) -> dict[str, Scenario]:
    scenarios = _default_scenarios()
    scenario_path = Path(path) if path is not None else DEFAULT_SCENARIO_FILE
    if not scenario_path.exists():
        return scenarios

    raw = json.loads(scenario_path.read_text())
    loaded: dict[str, Scenario] = {}
    for name, spec in raw.items():
        requests_spec = spec.get("requests")
        if not requests_spec:
            raise ValueError(f"scenario {name!r} must define a non-empty requests list")
        loaded[name] = Scenario(
            name=name,
            description=spec.get("description", ""),
            requests=[
                ScenarioRequest(
                    prompt=_load_prompt_from_request_spec(req, scenario_path.parent),
                    max_new_tokens=int(req.get("max_new_tokens", 64)),
                    temperature=float(req.get("temperature", 0.8)),
                    top_p=float(req.get("top_p", 0.95)),
                    seed=req.get("seed"),
                    weight=int(req.get("weight", 1)),
                    metadata=dict(req.get("metadata", {})),
                )
                for req in requests_spec
            ],
        )
    scenarios.update(loaded)
    return scenarios


def _load_prompt_from_request_spec(
    request_spec: dict[str, Any], scenario_dir: Path
) -> str:
    if "prompt" in request_spec:
        return str(request_spec["prompt"])
    prompt_file = request_spec.get("prompt_file")
    if prompt_file is None:
        raise ValueError("request spec must provide either 'prompt' or 'prompt_file'")
    return (scenario_dir / str(prompt_file)).read_text()
