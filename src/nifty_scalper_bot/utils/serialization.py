from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


def to_json_safe(value: Any) -> Any:

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if pd is not None and isinstance(value, pd.Timestamp):
        return value.isoformat()

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [to_json_safe(v) for v in value.tolist()]

    if is_dataclass(value):
        return to_json_safe(asdict(value))

    if isinstance(value, dict):
        return {str(to_json_safe(k)): to_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json_safe(v) for v in value]

    return str(value)


def json_dumps_safe(value: Any) -> str:
    return json.dumps(to_json_safe(value), ensure_ascii=False, default=str)
