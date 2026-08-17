from __future__ import annotations

import threading
import time
from argparse import Namespace
from typing import Any

import pytest

from scripts.bench.execution import (
    _arrival_schedule,
    _dispatch_open_loop,
    _open_loop_stats,
    _saturation_verdict,
    _window_partition,
)
from scripts.bench.models import RequestPlan, RequestResult, RunClock

# ---------------------------------------------------------------------------
# Builders and fakes
#
# No monkeypatching: the dispatcher takes its clock and its runner as ordinary
# parameters, so a virtual clock and a stub runner are all the seam needed.
# ---------------------------------------------------------------------------


def _make_args(**overrides: Any) -> Namespace:
    defaults = {
        "base_url": "http://127.0.0.1:8000",
        "endpoint": "stream_v2",
        "timeout_seconds": 120.0,
        "mode": "open",
        "arrival_rate": 10.0,
        "client_max_in_flight": 8,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_plan(ordinal: int) -> RequestPlan:
    return RequestPlan(
        ordinal=ordinal,
        scenario_name="short_short",
        payload={"prompt": "x", "max_new_tokens": 4},
        prompt_length_chars=1,
        prompt_source="short",
        metadata={},
    )


def _make_result(ordinal: int, **overrides: Any) -> RequestResult:
    defaults = dict(
        request_id=f"req-{ordinal}",
        run_id="run-1",
        ordinal=ordinal,
        scenario_name="short_short",
        endpoint="stream_v2",
        mode="open",
        prompt_source="short",
        start_ts=0.0,
        first_token_ts=None,
        end_ts=0.5,
        latency_ms=500.0,
        ttft_ms=None,
        tpot_ms=None,
        output_tokens=None,
        prompt_tokens=None,
        tokens_per_s=None,
        queue_wait_ms=None,
        execution_ms=None,
        http_status=200,
        ok=True,
        error_type=None,
        error=None,
        prompt_length_chars=1,
        response_text_chars=None,
    )
    defaults.update(overrides)
    return RequestResult(**defaults)  # type: ignore[arg-type]


class _StubRunner:
    """
    Stands in for a server: records the arrival offset it was handed and holds
    its in-flight slot for ``service_seconds``.

    The hold is a real sleep. A zero-cost virtual clock cannot be used here:
    advancing time without blocking does not order the dispatcher against its
    worker pool, so measured dispatch lag would be a thread-scheduling artifact
    rather than the quantity under test. Durations are kept in the tens of
    milliseconds so the whole class still runs in about a second.
    """

    def __init__(self, service_seconds: float = 0.0) -> None:
        self._service_seconds = service_seconds
        self.seen_arrivals: list[float | None] = []
        self._lock = threading.Lock()

    def __call__(
        self,
        base_url: str,
        endpoint: str,
        timeout_seconds: float,
        run_id: str,
        mode: str,
        plan: RequestPlan,
        clock: RunClock,
        *,
        scheduled_arrival_offset_s: float | None = None,
    ) -> RequestResult:
        start_ts = clock.offset()
        with self._lock:
            self.seen_arrivals.append(scheduled_arrival_offset_s)
        if self._service_seconds:
            time.sleep(self._service_seconds)
        end_ts = clock.offset()
        arrival = (
            scheduled_arrival_offset_s
            if scheduled_arrival_offset_s is not None
            else start_ts
        )
        return _make_result(
            plan.ordinal,
            run_id=run_id,
            start_ts=start_ts,
            end_ts=end_ts,
            latency_ms=(end_ts - arrival) * 1000.0,
            client_dispatch_lag_ms=max((start_ts - arrival) * 1000.0, 0.0),
            client_http_ms=(end_ts - start_ts) * 1000.0,
            scheduled_arrival_offset_s=arrival,
        )


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


class TestArrivalSchedule:
    def test_fixed_interval_from_start_offset(self) -> None:
        assert _arrival_schedule(4.0, 3, 10.0) == [10.0, 10.25, 10.5]

    def test_empty_count(self) -> None:
        assert _arrival_schedule(4.0, 0, 0.0) == []

    def test_non_positive_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="arrival rate"):
            _arrival_schedule(0.0, 3, 0.0)


class TestSaturationVerdict:
    def test_healthy_run(self) -> None:
        assert _saturation_verdict(10.0, 9.9, [1.0, 2.0, 3.0]) is False

    def test_achieved_rate_below_floor(self) -> None:
        # 9.0 / 10.0 is under the 0.95 floor.
        assert _saturation_verdict(10.0, 9.0, [1.0]) is True

    def test_achieved_rate_exactly_at_floor_passes(self) -> None:
        assert _saturation_verdict(10.0, 9.5, [1.0]) is False

    def test_p95_lag_over_one_arrival_interval(self) -> None:
        # Interval is 100ms; nearly every request waited longer than that.
        assert _saturation_verdict(10.0, 10.0, [150.0] * 20) is True

    def test_missing_achieved_rate(self) -> None:
        assert _saturation_verdict(10.0, None, []) is True

    def test_no_dispatch_lags(self) -> None:
        assert _saturation_verdict(10.0, 10.0, []) is False


class TestWindowPartition:
    def test_splits_on_offer_window_end(self) -> None:
        results = [
            _make_result(0, end_ts=1.0),
            _make_result(1, end_ts=5.0),
            _make_result(2, end_ts=7.5),
        ]
        in_window, drained = _window_partition(results, 5.0)
        assert [r.ordinal for r in in_window] == [0, 1]
        assert [r.ordinal for r in drained] == [2]

    def test_all_in_window(self) -> None:
        results = [_make_result(0, end_ts=1.0)]
        in_window, drained = _window_partition(results, 5.0)
        assert len(in_window) == 1
        assert drained == []


class TestOpenLoopStats:
    def test_reports_offer_window(self) -> None:
        results = [
            _make_result(0, end_ts=1.0, client_dispatch_lag_ms=1.0),
            _make_result(1, end_ts=2.0, client_dispatch_lag_ms=2.0),
            _make_result(2, end_ts=6.0, client_dispatch_lag_ms=3.0),
            _make_result(3, end_ts=8.0, client_dispatch_lag_ms=4.0),
        ]
        stats = _open_loop_stats(1.0, 0.0, 4.0, 16, results)

        assert stats["arrival_process"] == "deterministic"
        assert stats["target_arrival_rate"] == 1.0
        assert stats["achieved_arrival_rate"] == 1.0
        assert stats["offer_window_seconds"] == 4.0
        assert stats["client_max_in_flight"] == 16
        assert stats["completions_in_offer_window"] == 2
        assert stats["drain_seconds"] == 4.0
        assert stats["client_dispatch_lag_ms"]["mean"] == 2.5
        assert stats["client_saturated"] is False

    def test_requested_duration_is_reported_alongside_the_window(self) -> None:
        results = [_make_result(0, end_ts=1.0)]
        stats = _open_loop_stats(
            1.0, 0.0, 4.0, 16, results, requested_duration_seconds=3.9
        )
        assert stats["requested_duration_seconds"] == 3.9
        assert stats["offer_window_seconds"] == 4.0

    def test_requested_duration_is_none_in_request_count_mode(self) -> None:
        stats = _open_loop_stats(1.0, 0.0, 4.0, 16, [_make_result(0)])
        assert stats["requested_duration_seconds"] is None

    def test_drain_seconds_zero_when_nothing_drains(self) -> None:
        results = [_make_result(0, end_ts=1.0)]
        stats = _open_loop_stats(1.0, 0.0, 4.0, 16, results)
        assert stats["drain_seconds"] == 0.0

    def test_zero_length_window_has_no_achieved_rate(self) -> None:
        stats = _open_loop_stats(1.0, 2.0, 2.0, 16, [])
        assert stats["achieved_arrival_rate"] is None
        assert stats["client_saturated"] is True


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.timing
class TestDispatchOpenLoop:
    def test_healthy_run_is_not_saturated(self) -> None:
        # 20/s offered, ample capacity, instant service: nothing should queue.
        runner = _StubRunner()
        plans = [_make_plan(i) for i in range(5)]

        outcome = _dispatch_open_loop(
            _make_args(arrival_rate=20.0, client_max_in_flight=8),
            plans,
            "run-1",
            RunClock.start(),
            runner=runner,
        )

        assert [r.ordinal for r in outcome.results] == [0, 1, 2, 3, 4]
        assert outcome.stats["client_saturated"] is False
        assert outcome.stats["target_arrival_rate"] == 20.0
        # Well inside one 50ms arrival interval.
        assert max(r.client_dispatch_lag_ms for r in outcome.results) < 25.0

    def test_scheduled_arrivals_reach_the_runner(self) -> None:
        runner = _StubRunner()
        plans = [_make_plan(i) for i in range(4)]

        outcome = _dispatch_open_loop(
            _make_args(arrival_rate=20.0, client_max_in_flight=8),
            plans,
            "run-1",
            RunClock.start(),
            runner=runner,
        )

        # The offsets the dispatcher computed are the ones the runner recorded,
        # exactly 1/rate apart. Before this rework they were discarded entirely.
        arrivals = [r.scheduled_arrival_offset_s for r in outcome.results]
        assert len(runner.seen_arrivals) == 4
        assert sorted(runner.seen_arrivals) == arrivals  # type: ignore[type-var]
        spacings = [b - a for a, b in zip(arrivals, arrivals[1:])]
        assert all(abs(gap - 0.05) < 1e-9 for gap in spacings)

    def test_capacity_bound_produces_dispatch_lag_and_saturation(self) -> None:
        # Two slots, service far longer than the 20ms arrival interval: the
        # dispatcher must block, and must say so.
        runner = _StubRunner(service_seconds=0.1)
        plans = [_make_plan(i) for i in range(6)]

        outcome = _dispatch_open_loop(
            _make_args(arrival_rate=50.0, client_max_in_flight=2),
            plans,
            "run-1",
            RunClock.start(),
            runner=runner,
        )

        assert len(outcome.results) == 6
        assert max(r.client_dispatch_lag_ms for r in outcome.results) > 20.0
        assert outcome.stats["client_saturated"] is True
        assert outcome.stats["achieved_arrival_rate"] is not None
        assert outcome.stats["achieved_arrival_rate"] < 50.0

    def test_deadline_closes_the_offer_window_early(self) -> None:
        runner = _StubRunner()
        plans = [_make_plan(i) for i in range(100)]
        clock = RunClock.start()

        outcome = _dispatch_open_loop(
            _make_args(arrival_rate=20.0, client_max_in_flight=8),
            plans,
            "run-1",
            clock,
            # 0.27s is 5.4 arrival intervals — deliberately not a whole
            # multiple of 50ms, so this test never rests on a deadline that
            # happens to land on an arrival boundary. Do not round it off.
            deadline_offset_s=clock.offset() + 0.27,
            runner=runner,
        )

        # 20/s offered for 270ms is ~6 requests, not the 100 that were planned.
        assert 3 <= len(outcome.results) <= 8
        assert all(r.client_http_ms >= 0.0 for r in outcome.results)

    def test_in_flight_requests_drain_after_the_deadline(self) -> None:
        runner = _StubRunner(service_seconds=0.2)
        plans = [_make_plan(i) for i in range(20)]
        clock = RunClock.start()

        outcome = _dispatch_open_loop(
            _make_args(arrival_rate=20.0, client_max_in_flight=4),
            plans,
            "run-1",
            clock,
            deadline_offset_s=clock.offset() + 0.1,
            runner=runner,
        )

        # Everything offered still completes and is counted, after the window.
        assert outcome.results
        assert outcome.stats["drain_seconds"] > 0.0
        assert outcome.stats["completions_in_offer_window"] == 0

    def test_duration_ending_between_arrivals_still_reports_target_rate(self) -> None:
        # 20/s for 170ms: arrivals at 0, 50, 100, 150ms all fall inside, but the
        # last one's interval ends at 200ms — past the deadline. The window is
        # deliberately not capped there, so the achieved rate stays at target
        # rather than overshooting it; the requested duration is reported too.
        runner = _StubRunner()
        plans = [_make_plan(i) for i in range(4)]
        clock = RunClock.start()

        outcome = _dispatch_open_loop(
            _make_args(arrival_rate=20.0, client_max_in_flight=8),
            plans,
            "run-1",
            clock,
            deadline_offset_s=clock.offset() + 0.17,
            runner=runner,
        )

        stats = outcome.stats
        assert len(outcome.results) == 4
        assert stats["achieved_arrival_rate"] == pytest.approx(20.0, rel=0.02)
        # Derived as deadline - offer_start, so a hair under the 0.17 requested.
        assert stats["requested_duration_seconds"] == pytest.approx(0.17, abs=1e-3)
        # The overshoot is real but bounded by one arrival interval.
        overshoot = stats["offer_window_seconds"] - stats["requested_duration_seconds"]
        assert 0.0 < overshoot < 1.0 / 20.0
        assert stats["client_saturated"] is False

    def test_saturated_duration_run_reports_rate_below_target(self) -> None:
        # The property that makes capping the window unnecessary: when the
        # client falls behind, elapsed wall time dominates the schedule and the
        # achieved rate collapses on its own.
        runner = _StubRunner(service_seconds=0.15)
        plans = [_make_plan(i) for i in range(40)]
        clock = RunClock.start()

        outcome = _dispatch_open_loop(
            _make_args(arrival_rate=20.0, client_max_in_flight=2),
            plans,
            "run-1",
            clock,
            deadline_offset_s=clock.offset() + 0.5,
            runner=runner,
        )

        stats = outcome.stats
        assert stats["achieved_arrival_rate"] is not None
        assert stats["achieved_arrival_rate"] < 0.95 * 20.0
        assert stats["client_saturated"] is True

    def test_empty_plan_list(self) -> None:
        runner = _StubRunner()

        outcome = _dispatch_open_loop(
            _make_args(), [], "run-1", RunClock.start(), runner=runner
        )

        assert outcome.results == []
        assert outcome.stats["completions_in_offer_window"] == 0
