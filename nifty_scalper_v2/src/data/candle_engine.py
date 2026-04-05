"""Real-time OHLCV candle aggregation from tick stream."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from .tick_hub import EventBus, Tick

IST = ZoneInfo("Asia/Kolkata")


@dataclass(slots=True)
class OHLCV:
    symbol: str
    timeframe: str      # "1m", "5m", "15m"
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: int
    timestamp: datetime  # candle open time UTC
    is_complete: bool = False


_TIMEFRAME_MINUTES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15}


@dataclass
class _Bar:
    """Mutable bar being built."""
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    oi: int
    bar_open_utc: datetime
    ticks: int = 0

    def update(self, price: float, volume: int, oi: int) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume
        self.oi = oi
        self.ticks += 1

    def to_ohlcv(self, complete: bool = False) -> OHLCV:
        return OHLCV(
            symbol=self.symbol,
            timeframe=self.timeframe,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            oi=self.oi,
            timestamp=self.bar_open_utc,
            is_complete=complete,
        )


def _bar_open_utc(tick_ts: datetime, tf_minutes: int) -> datetime:
    """Round tick timestamp down to the nearest bar boundary in UTC."""
    ist = tick_ts.astimezone(IST)
    total_minutes = ist.hour * 60 + ist.minute
    bar_minutes = (total_minutes // tf_minutes) * tf_minutes
    bar_ist = ist.replace(
        hour=bar_minutes // 60,
        minute=bar_minutes % 60,
        second=0,
        microsecond=0,
    )
    return bar_ist.astimezone(timezone.utc)


class CandleEngine:
    """Aggregates ticks into OHLCV bars for multiple timeframes.

    Subscribes to 'tick' events on the EventBus.
    Emits 'candle_complete' events when a bar closes.
    """

    def __init__(
        self,
        event_bus: "EventBus",
        timeframes: list[str] | None = None,
        maxlen: int = 500,
    ) -> None:
        self._bus = event_bus
        self._timeframes = timeframes or ["1m", "5m", "15m"]
        self._maxlen = maxlen

        # dict[symbol][timeframe] → deque[OHLCV]
        self._history: dict[str, dict[str, deque[OHLCV]]] = {}
        # dict[symbol][timeframe] → current _Bar
        self._current: dict[str, dict[str, _Bar | None]] = {}

        event_bus.subscribe("tick", self._on_tick)

    def _on_tick(self, tick: "Tick") -> None:
        sym = tick.symbol
        if sym not in self._history:
            self._history[sym] = {tf: deque(maxlen=self._maxlen) for tf in self._timeframes}
            self._current[sym] = {tf: None for tf in self._timeframes}

        for tf in self._timeframes:
            tf_min = _TIMEFRAME_MINUTES[tf]
            bar_open = _bar_open_utc(tick.timestamp, tf_min)
            current = self._current[sym][tf]

            if current is None:
                # Start first bar
                self._current[sym][tf] = _Bar(
                    symbol=sym, timeframe=tf,
                    open=tick.ltp, high=tick.ltp, low=tick.ltp, close=tick.ltp,
                    volume=0, oi=tick.oi, bar_open_utc=bar_open,
                )
            elif bar_open > current.bar_open_utc:
                # Bar closed — emit complete candle
                completed = current.to_ohlcv(complete=True)
                self._history[sym][tf].append(completed)
                self._bus.publish("candle_complete", completed)
                # Start new bar
                self._current[sym][tf] = _Bar(
                    symbol=sym, timeframe=tf,
                    open=tick.ltp, high=tick.ltp, low=tick.ltp, close=tick.ltp,
                    volume=0, oi=tick.oi, bar_open_utc=bar_open,
                )
            else:
                current.update(tick.ltp, 0, tick.oi)  # volume incremental not available

    def get_candles(self, symbol: str, timeframe: str, n: int = 50) -> list[OHLCV]:
        """Return last n completed candles for symbol+timeframe."""
        hist = self._history.get(symbol, {}).get(timeframe)
        if not hist:
            return []
        candles = list(hist)
        return candles[-n:] if len(candles) >= n else candles

    def get_current_bar(self, symbol: str, timeframe: str) -> OHLCV | None:
        """Return incomplete current bar (if any)."""
        bar = self._current.get(symbol, {}).get(timeframe)
        return bar.to_ohlcv(complete=False) if bar else None
