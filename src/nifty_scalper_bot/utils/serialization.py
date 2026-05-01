from __future__ import annotations

from datetime import date, datetime
from typing import Any


def to_json_safe(value: Any) -> Any:
    try:
        import pandas as pd
    except Exception:
        pd = None

    try:
        import numpy as np
    except Exception:
        np = None

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if pd is not None and isinstance(value, pd.Timestamp):
        return value.isoformat()

    if np is not None and isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(v) for v in value]

    return str(value)
