from __future__ import annotations

import pytest

from scripts.bench.models import Scenario, ScenarioRequest
from scripts.bench.planning import _build_request_plans


def _simple_scenario(
    name: str = "test",
    requests: list[ScenarioRequest] | None = None,
) -> Scenario:
    if requests is None:
        requests = [
            ScenarioRequest(
                prompt="hello",
                max_new_tokens=16,
                temperature=0.5,
                top_p=0.9,
                metadata={"class": "default"},
            )
        ]
    return Scenario(name=name, description="test scenario", requests=requests)


class TestBuildRequestPlans:
    def test_basic_plan_generation(self) -> None:
        scenario = _simple_scenario()
        plans = _build_request_plans(scenario, 3, None, None, None, None, None)
        assert len(plans) == 3
        assert [p.ordinal for p in plans] == [0, 1, 2]
        assert plans[0].payload["prompt"] == "hello"
        assert plans[0].payload["max_new_tokens"] == 16
        assert plans[0].payload["temperature"] == 0.5
        assert plans[0].prompt_source == "default"
        assert "seed" not in plans[0].payload

    def test_weighted_cycling(self) -> None:
        scenario = _simple_scenario(
            requests=[
                ScenarioRequest(prompt="A", weight=3, metadata={"class": "a"}),
                ScenarioRequest(prompt="B", weight=1, metadata={"class": "b"}),
            ]
        )
        plans = _build_request_plans(scenario, 8, None, None, None, None, None)
        sources = [p.prompt_source for p in plans]
        # weight=3 expands to [A, A, A, B], cycling: A,A,A,B,A,A,A,B
        assert sources == ["a", "a", "a", "b", "a", "a", "a", "b"]

    def test_cli_overrides(self) -> None:
        scenario = _simple_scenario()
        plans = _build_request_plans(
            scenario,
            1,
            None,
            max_new_tokens_override=99,
            temperature_override=0.1,
            top_p_override=0.5,
            seed_override=42,
        )
        p = plans[0].payload
        assert p["max_new_tokens"] == 99
        assert p["temperature"] == 0.1
        assert p["top_p"] == 0.5
        assert p["seed"] == 42

    def test_prompt_override(self) -> None:
        scenario = _simple_scenario()
        plans = _build_request_plans(scenario, 1, "overridden", None, None, None, None)
        assert plans[0].payload["prompt"] == "overridden"
        assert plans[0].prompt_length_chars == len("overridden")

    def test_empty_requests_raises(self) -> None:
        scenario = _simple_scenario(requests=[])
        with pytest.raises(ValueError, match="does not contain any request templates"):
            _build_request_plans(scenario, 1, None, None, None, None, None)

    def test_seed_from_scenario(self) -> None:
        scenario = _simple_scenario(requests=[ScenarioRequest(prompt="x", seed=7)])
        plans = _build_request_plans(scenario, 1, None, None, None, None, None)
        assert plans[0].payload["seed"] == 7

    def test_seed_override_beats_scenario(self) -> None:
        scenario = _simple_scenario(requests=[ScenarioRequest(prompt="x", seed=7)])
        plans = _build_request_plans(
            scenario, 1, None, None, None, None, seed_override=99
        )
        assert plans[0].payload["seed"] == 99
