from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class HistStatus(Enum):
    OK = "OK"
    NO_DATA = "NO_DATA"
    ERROR = "ERROR"


@dataclass
class HistResult:
    status: HistStatus
    df: pd.DataFrame
    reason: str = ""

    def __bool__(self) -> bool:
        """Return ``True`` only when historical data is available."""
        return self.status is HistStatus.OK
