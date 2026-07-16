from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from nifty_scalper_bot.data.candle_engine import IST, sanitize


@dataclass(frozen=True)
class HistoryRequest:
    symbol: str
    interval: str
    limit: int | None
    kwargs: dict[str, Any]


class SimulatedHistoryProvider:
    def __init__(self, clock, recorder=None) -> None:
        self.clock = clock
        self.recorder = recorder
        self._history: dict[str, pd.DataFrame] = {}
        self._requests: list[HistoryRequest] = []
        self._hooks: dict[str, set[str]] = {}

    def set_history(self, symbol: str, candles) -> None:
        frame = sanitize(pd.DataFrame(candles))
        ts = pd.to_datetime(frame["timestamp"])
        if ts.duplicated().any() or not ts.is_monotonic_increasing:
            raise AssertionError(f"non-canonical history for {symbol}")
        if (ts >= pd.Timestamp(self.clock.now())).any():
            raise AssertionError(f"future historical candle for {symbol}")
        self._history[symbol] = frame

    def enable_hook(self, symbol: str, hook: str) -> None:
        self._hooks.setdefault(symbol, set()).add(hook)

    def fetch_history(
        self, symbol: str, interval: str = "1min", limit: int | None = None, **kwargs
    ):
        self._requests.append(HistoryRequest(symbol, interval, limit, dict(kwargs)))
        if self.recorder:
            self.recorder.record(
                "HISTORY_REQUESTED", symbol, interval=interval, limit=limit
            )
        frame = self._history.get(symbol, pd.DataFrame()).copy()
        if limit:
            frame = frame.tail(limit)
        hooks = self._hooks.get(symbol, set())
        if "partial_response" in hooks and limit:
            frame = frame.tail(max(limit // 2, 1))
        if "duplicate_bars" in hooks and not frame.empty:
            frame = pd.concat([frame, frame.tail(1)], ignore_index=True)
        if "out_of_order" in hooks:
            frame = frame.iloc[::-1].reset_index(drop=True)
        if self.recorder:
            self.recorder.record("HISTORY_HYDRATED", symbol, rows=len(frame))
        return frame

    def record_requests(self) -> list[HistoryRequest]:
        return list(self._requests)


def make_history(end_before, count: int, start_price: float) -> list[dict[str, Any]]:
    end = pd.Timestamp(end_before).tz_convert(IST).floor("min")
    return [
        {
            "timestamp": end - pd.Timedelta(minutes=count - idx),
            "open": start_price + idx * 0.05,
            "high": start_price + idx * 0.05 + 0.2,
            "low": start_price + idx * 0.05 - 0.2,
            "close": start_price + idx * 0.05 + 0.1,
            "volume": 1000 + idx,
        }
        for idx in range(count)
    ]
