from __future__ import annotations

from typing import Any

from nifty_scalper_bot.core.strategy_manager import StrategyManager


class _IndicatorEngine:
    def __init__(self, bars: int) -> None:
        self._bars = bars

    def get_indicators(self, symbol: str, names: Any) -> dict[str, Any]:
        return {"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "vwap": 100.5, "volume": 10.0}

    def get_history(self, symbol: str) -> list[dict[str, Any]]:
        return [{"timestamp": i} for i in range(self._bars)]

    def get_ohlc_bars(self, symbol: str) -> list[dict[str, Any]]:
        return [{"timestamp": i} for i in range(self._bars)]


class _DataHub:
    def __init__(self, bars_by_symbol: dict[str, int]) -> None:
        self._bars_by_symbol = bars_by_symbol

    def get_ohlc_bars(self, symbol: str) -> list[dict[str, Any]]:
        size = self._bars_by_symbol.get(symbol, 0)
        return [{"timestamp": i} for i in range(size)]


class _PositionManager:
    def get_position(self, symbol: str) -> None:
        return None


class _CaptureSMC:
    name = "SMC"

    def __init__(self) -> None:
        self.last: dict[str, Any] | None = None
        self.last_no_vote_reason = "none"

    def generate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any) -> None:
        self.last = dict(indicators)
        return None


def test_strategy_manager_injects_smc_history_context_option_domain() -> None:
    smc = _CaptureSMC()
    symbol = "NFO:NIFTY26MAY23850PE"
    manager = StrategyManager([smc], _IndicatorEngine(35), _PositionManager(), data_hub=_DataHub({symbol: 35}))
    manager.generate_signal(symbol, 100.0)
    assert smc.last is not None
    assert smc.last.get("history_domain_used") == "options"
    assert smc.last.get("option_history_count") == 35
    assert smc.last.get("history_ready_for_smc") is True
    assert smc.last.get("history_source") in {"data_hub", "indicator_engine"}


def test_strategy_manager_option_history_not_overwritten_by_underlying() -> None:
    smc = _CaptureSMC()
    symbol = "NFO:NIFTY26MAY23850PE"
    manager = StrategyManager([smc], _IndicatorEngine(341), _PositionManager(), data_hub=_DataHub({symbol: 5}))
    manager.generate_signal(symbol, 100.0)
    assert smc.last is not None
    assert smc.last.get("history_domain_used") == "options"
    assert smc.last.get("option_history_count") == 5
    assert smc.last.get("history_ready_for_smc") is False
    assert smc.last.get("history_quality") == "cold"
