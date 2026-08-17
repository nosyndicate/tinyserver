from __future__ import annotations

from argparse import Namespace
from typing import Any

import pytest

from scripts.bench.cli import _validate_args


def _make_args(**overrides: Any) -> Namespace:
    """Build a valid closed-loop Namespace, then apply overrides."""
    defaults = {
        "mode": "closed",
        "concurrency": 4,
        "arrival_rate": None,
        "requests": 10,
        "duration_seconds": None,
        "client_max_in_flight": 256,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class TestValidateArgs:
    def test_valid_closed_loop(self) -> None:
        _validate_args(_make_args())

    def test_valid_open_loop(self) -> None:
        _validate_args(_make_args(mode="open", arrival_rate=5.0, concurrency=4))

    def test_valid_duration_mode(self) -> None:
        _validate_args(_make_args(requests=None, duration_seconds=10.0))

    def test_closed_loop_zero_concurrency(self) -> None:
        with pytest.raises(ValueError, match="concurrency"):
            _validate_args(_make_args(concurrency=0))

    def test_open_loop_missing_arrival_rate(self) -> None:
        with pytest.raises(ValueError, match="arrival-rate"):
            _validate_args(_make_args(mode="open", arrival_rate=None))

    def test_open_loop_zero_arrival_rate(self) -> None:
        with pytest.raises(ValueError, match="arrival-rate"):
            _validate_args(_make_args(mode="open", arrival_rate=0))

    def test_open_loop_zero_client_max_in_flight(self) -> None:
        with pytest.raises(ValueError, match="client-max-in-flight"):
            _validate_args(
                _make_args(mode="open", arrival_rate=5.0, client_max_in_flight=0)
            )

    def test_open_loop_ignores_concurrency(self) -> None:
        # --concurrency is closed-loop only; open loop is bounded by
        # --client-max-in-flight, so a zero here is irrelevant rather than fatal.
        _validate_args(_make_args(mode="open", arrival_rate=5.0, concurrency=0))

    def test_open_loop_duration_shorter_than_one_arrival_interval(self) -> None:
        # 0.1s at 1/s schedules under one request; nothing measurable there.
        with pytest.raises(ValueError, match="at least one arrival interval"):
            _validate_args(
                _make_args(
                    mode="open",
                    arrival_rate=1.0,
                    requests=None,
                    duration_seconds=0.1,
                )
            )

    def test_open_loop_duration_of_exactly_one_interval(self) -> None:
        _validate_args(
            _make_args(
                mode="open", arrival_rate=2.0, requests=None, duration_seconds=0.5
            )
        )

    def test_closed_loop_short_duration_is_unaffected(self) -> None:
        # The interval guard is open-loop only; closed loop has no arrival rate.
        _validate_args(_make_args(requests=None, duration_seconds=0.01))

    def test_neither_requests_nor_duration(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            _validate_args(_make_args(requests=None, duration_seconds=None))

    def test_both_requests_and_duration(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            _validate_args(_make_args(requests=10, duration_seconds=5.0))

    def test_requests_zero(self) -> None:
        with pytest.raises(ValueError, match="requests must be positive"):
            _validate_args(_make_args(requests=0))

    def test_duration_zero(self) -> None:
        with pytest.raises(ValueError, match="duration-seconds must be positive"):
            _validate_args(_make_args(requests=None, duration_seconds=0))
