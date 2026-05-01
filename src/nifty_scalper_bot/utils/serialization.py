"""JSON-safe serialization helpers. Args: value. Returns: safe value. Raises: None."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd


def to_json_safe(value: Any) -> Any:
    """Convert values into JSON-safe shapes. Args: value. Returns: serializable object. Raises: None."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, 'item') and callable(getattr(value, 'item')):
        try:
            return to_json_safe(value.item())
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(v) for v in value]
    return str(value)
