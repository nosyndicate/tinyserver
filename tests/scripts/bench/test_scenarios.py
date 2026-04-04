from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.bench.scenarios import (
    _default_scenarios,
    _load_prompt_from_request_spec,
    _load_scenarios,
)


def test_default_scenarios_keys() -> None:
    scenarios = _default_scenarios()
    expected = {"short_short", "long_long", "mixed", "burst", "seeded_deterministic"}
    assert set(scenarios.keys()) == expected
    for name, scenario in scenarios.items():
        assert scenario.name == name
        assert len(scenario.requests) >= 1


def test_load_scenarios_merges_json_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("hello from file")
    scenario_file = tmp_path / "scenarios.json"
    scenario_file.write_text(
        json.dumps(
            {
                "custom": {
                    "description": "custom scenario",
                    "requests": [
                        {
                            "prompt_file": "prompt.txt",
                            "max_new_tokens": 12,
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "seed": 123,
                            "metadata": {"class": "custom"},
                        }
                    ],
                }
            }
        )
    )

    scenarios = _load_scenarios(str(scenario_file))

    assert "short_short" in scenarios
    assert scenarios["custom"].description == "custom scenario"
    assert scenarios["custom"].requests[0].seed == 123
    assert scenarios["custom"].requests[0].prompt == "hello from file"


class TestLoadPromptFromRequestSpec:
    def test_inline_prompt(self, tmp_path: Path) -> None:
        assert _load_prompt_from_request_spec({"prompt": "hello"}, tmp_path) == "hello"

    def test_missing_both(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="prompt"):
            _load_prompt_from_request_spec({}, tmp_path)

    def test_prompt_file(self, tmp_path: Path) -> None:
        f = tmp_path / "p.txt"
        f.write_text("from file")
        assert (
            _load_prompt_from_request_spec({"prompt_file": "p.txt"}, tmp_path)
            == "from file"
        )
