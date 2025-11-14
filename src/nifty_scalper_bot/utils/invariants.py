"""Lightweight invariant helpers for guarding internal state."""

from __future__ import annotations


def assert_non_negative(name: str, value: float) -> None:
    """Raise if *value* is negative."""

    if value < 0:
        raise ValueError(f"{name} must be >= 0; got {value}")


def require_tuple_prefix(name: str, tup: object, length: int) -> None:
    """Validate that *tup* is a tuple with at least *length* entries."""

    if not isinstance(tup, tuple) or len(tup) < length:
        raise ValueError(f"{name} must be tuple with at least {length} items")


__all__ = ["assert_non_negative", "require_tuple_prefix"]
