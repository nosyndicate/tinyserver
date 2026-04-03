from __future__ import annotations

import json
from argparse import Namespace
from typing import Any, Generator

import scripts.bench.runners as bench_runners
from scripts.bench.metrics import _summarize_results
from scripts.bench.models import RequestPlan, RequestResult
from scripts.bench.runners import _run_stream_request
from scripts.bench.scenarios import _default_scenarios, _load_scenarios


def test_load_scenarios_merges_json_file(tmp_path: str) -> None:
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


def test_summarize_results_reports_failures_and_percentiles() -> None:
    args = Namespace(
        base_url="http://127.0.0.1:8000",
        endpoint="stream_v2",
        mode="closed",
    )
    scenario = _default_scenarios()["short_short"]
    results = [
        RequestResult(
            request_id="1",
            run_id="run-1",
            ordinal=0,
            scenario_name="short_short",
            endpoint="stream_v2",
            mode="closed",
            prompt_source="short",
            start_ts=0.0,
            first_token_ts=0.1,
            end_ts=0.5,
            latency_ms=500.0,
            ttft_ms=100.0,
            tpot_ms=20.0,
            output_tokens=21,
            prompt_tokens=10,
            tokens_per_s=42.0,
            queue_wait_ms=30.0,
            execution_ms=470.0,
            http_status=200,
            ok=True,
            error_type=None,
            error=None,
            prompt_length_chars=12,
            response_text_chars=50,
        ),
        RequestResult(
            request_id="2",
            run_id="run-1",
            ordinal=1,
            scenario_name="short_short",
            endpoint="stream_v2",
            mode="closed",
            prompt_source="short",
            start_ts=0.0,
            first_token_ts=None,
            end_ts=0.2,
            latency_ms=200.0,
            ttft_ms=None,
            tpot_ms=None,
            output_tokens=None,
            prompt_tokens=None,
            tokens_per_s=None,
            queue_wait_ms=None,
            execution_ms=None,
            http_status=503,
            ok=False,
            error_type="http_error",
            error="busy",
            prompt_length_chars=12,
            response_text_chars=None,
        ),
    ]

    summary = _summarize_results(
        args=args,
        scenario=scenario,
        run_id="run-1",
        run_started_ts=0.0,
        run_ended_ts=1.0,
        results=results,
        warmup_results=[],
    )

    assert summary["completed_requests"] == 1
    assert summary["failed_requests"] == 1
    assert summary["rejected_requests"] == 1
    assert summary["latency_ms"]["p50"] == 500.0
    assert summary["ttft_ms"]["p50"] == 100.0
    assert summary["queue_wait_ms"]["p50"] == 30.0
    assert summary["execution_ms"]["p50"] == 470.0
    assert summary["output_token_throughput_tps"] == 21.0
    assert summary["error_counts"]["http_error"] == 1


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self.status_code = 200
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = False) -> Generator[str, None, None]:  # noqa: ANN202
        yield from self._lines

    def __enter__(self) -> "_FakeStreamResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


def test_run_stream_request_uses_done_chunk_metadata(monkeypatch: Any) -> None:
    chunks = [
        'data: {"token_str":"Hello","is_first":true,"is_done":false}\n',
        (
            'data: {"token_str":" world","is_first":false,"is_done":true,'
            '"prompt_tokens":11,"output_tokens":2,"ttft_ms":80.0,"total_ms":180.0,'
            '"tokens_per_s":11.1,"queue_wait_ms":25.0,"execution_ms":155.0}\n'
        ),
    ]

    def fake_post(*args: Any, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(chunks)

    monkeypatch.setattr(bench_runners.requests, "post", fake_post)

    plan = RequestPlan(
        ordinal=0,
        scenario_name="short_short",
        payload={
            "prompt": "hello",
            "max_new_tokens": 2,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        prompt_length_chars=5,
        prompt_source="short",
        metadata={},
    )
    result = _run_stream_request(
        base_url="http://127.0.0.1:8000",
        endpoint="stream_v2",
        timeout_seconds=5.0,
        run_id="run-1",
        mode="closed",
        plan=plan,
    )

    assert result.ok is True
    assert result.output_tokens == 2
    assert result.prompt_tokens == 11
    assert result.ttft_ms == 80.0
    assert result.latency_ms == 180.0
    assert result.queue_wait_ms == 25.0
    assert result.execution_ms == 155.0
    assert result.tokens_per_s == 11.1
