from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class RunClock:
    """
    The single time reference for a benchmark run.

    Every duration and per-request timestamp is derived from ``perf_counter``,
    which is monotonic; wall clock is captured once so recorded offsets can be
    mapped back to a real date after the fact. Mixing the two per-request (the
    behaviour this corrected) let NTP slew leak into latency measurements.

    ``counter`` and ``sleeper`` are injectable so scheduling logic is testable
    without patching: a fake pair can advance virtual time instantly.
    """

    wall_epoch_s: float
    perf_epoch_s: float
    counter: Callable[[], float] = time.perf_counter
    sleeper: Callable[[float], None] = time.sleep

    @classmethod
    def start(cls) -> "RunClock":
        return cls(wall_epoch_s=time.time(), perf_epoch_s=time.perf_counter())

    def offset(self) -> float:
        """Monotonic seconds elapsed since the run epoch."""
        return self.counter() - self.perf_epoch_s

    def sleep_until(self, offset_s: float) -> None:
        """Block until ``offset_s`` seconds after the run epoch; no-op if past."""
        remaining = offset_s - self.offset()
        if remaining > 0:
            self.sleeper(remaining)


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
    """
    One row of ``requests.jsonl``, under the corrected measurement semantics.

    Artifacts written before that correction are distinguishable by the absence
    of the ``client_*`` / ``server_*`` fields below; their numbers are not
    comparable with these and should not be mixed into one table.

    Timestamps (``*_ts``, ``*_offset_s``) are seconds relative to the run's
    ``RunClock`` epoch, not wall clock. Client- and server-sourced metrics are
    kept in separate fields so the two clocks are never mixed in one number:

    - ``client_*`` is measured by this process.
    - ``server_*`` is reported by the server's final ``is_done`` SSE chunk, and
      is ``None`` for endpoints that do not report it (the v1 stream).

    ``ttft_ms`` / ``tpot_ms`` / ``queue_wait_ms`` / ``execution_ms`` are retained
    as aliases of ``client_ttft_ms`` / ``client_tpot_ms`` /
    ``server_queue_wait_ms`` / ``server_execution_ms`` so pre-existing analysis
    scripts keep loading.
    """

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
    # Scheduled arrival, from the run epoch. Equals start_ts in closed loop.
    scheduled_arrival_offset_s: float = 0.0
    client_dispatch_lag_ms: float = 0.0
    client_http_ms: float = 0.0
    client_ttft_ms: float | None = None
    client_tpot_ms: float | None = None
    server_total_ms: float | None = None
    server_queue_wait_ms: float | None = None
    server_execution_ms: float | None = None
    server_ttft_ms: float | None = None
    server_tpot_ms: float | None = None
    output_sha256: str | None = None
    output_tokens_source: str | None = None
    deterministic_gate: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
