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
