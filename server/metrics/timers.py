import time
from contextlib import contextmanager
from typing import Any, Generator


def now_ns() -> int:
    return time.perf_counter_ns()


def ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0


@contextmanager
def timed() -> Generator[dict[str, Any], None, None]:
    """
    Context manager to measure elapsed time.

    Returns:
        dict with 'start_ns' and 'end_ns' filled after exit.
    """
    t = {"start_ns": now_ns(), "end_ns": None}
    try:
        yield t
    finally:
        t["end_ns"] = now_ns()
