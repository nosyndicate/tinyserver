from typing import TypeVar

LOWEST_TEMPERATURE = 1e-5


T = TypeVar("T")


def assert_not_none(value: T | None) -> T:
    if value is None:
        raise ValueError("Expected value to be not None")
    return value
