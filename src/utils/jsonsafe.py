from __future__ import annotations

"""Helpers for serializing mixed Python objects to JSON."""

from datetime import date, datetime
from decimal import Decimal
from typing import Any


def json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation of ``value``."""

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, set):
        return list(value)
    return str(value)


__all__ = ["json_safe"]
