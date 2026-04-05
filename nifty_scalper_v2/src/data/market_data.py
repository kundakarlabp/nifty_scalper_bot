"""MarketDataManager facade — thin wrapper over TickHub + IndicatorEngine."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tick_hub import Tick, TickHub
    from .candle_engine import OHLCV, CandleEngine
    from ..indicators.engine import IndicatorEngine, IndicatorSnapshot


class MarketDataManager:
    """Facade over TickHub + CandleEngine + IndicatorEngine."""

    def __init__(
        self,
        tick_hub: "TickHub",
        candle_engine: "CandleEngine",
        indicator_engine: "IndicatorEngine",
    ) -> None:
        self._hub = tick_hub
        self._candles = candle_engine
        self._indicators = indicator_engine

    def get_latest_tick(self, symbol: str) -> "Tick | None":
        return self._hub.get_latest(symbol)

    def get_ohlcv(self, symbol: str, timeframe: str = "1m", n: int = 50) -> "list[OHLCV]":
        return self._candles.get_candles(symbol, timeframe, n)

    def get_indicators(self, symbol: str) -> "IndicatorSnapshot | None":
        return self._indicators.get_snapshot(symbol)

    def get_all_snapshots(self) -> "dict[str, IndicatorSnapshot]":
        return self._indicators.get_all_snapshots()

    @property
    def warmup_complete(self) -> bool:
        return self._hub.warmup_complete
