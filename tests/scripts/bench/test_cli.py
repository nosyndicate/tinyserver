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

    def test_open_loop_zero_concurrency(self) -> None:
        with pytest.raises(ValueError, match="concurrency"):
            _validate_args(_make_args(mode="open", arrival_rate=5.0, concurrency=0))

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
