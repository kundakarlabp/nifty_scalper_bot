from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.diagnostics import checks
from src.strategies.runner import StrategyRunner


class _DummyRunner:
    def __init__(self, now: datetime, frame: pd.DataFrame) -> None:
        self.now_ist = now
        self._frame = frame

    def ohlc_window(self) -> pd.DataFrame:
        return self._frame


def test_check_data_window_handles_naive_index(monkeypatch) -> None:
    tz = ZoneInfo("Asia/Kolkata")
    now = datetime(2025, 9, 18, 13, 16, tzinfo=tz)
    last_bar = datetime(2025, 9, 18, 13, 15)
    frame = pd.DataFrame({"close": [1.0]}, index=pd.Index([last_bar]))
    runner = _DummyRunner(now=now, frame=frame)

    monkeypatch.setattr(StrategyRunner, "_SINGLETON", runner)

    result = checks.check_data_window()

    assert result.ok is True
    assert result.details["lag_s"] == 60.0
    assert result.details["last_bar_ts"].endswith("+05:30")
