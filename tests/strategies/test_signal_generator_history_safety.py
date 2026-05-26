from __future__ import annotations

from typing import Any, Iterable, Mapping

from nifty_scalper_bot.strategies.signal_generator import StrategyManager


class _IndicatorEngineNone:
    def get_indicators(self, symbol: str, names: Iterable[str]) -> Mapping[str, Any] | None:
        return None


class _IndicatorEngineBars:
    def get_indicators(self, symbol: str, names: Iterable[str]) -> Mapping[str, Any] | None:
        return {"bar_count": 40, "vix": 15.0, "volume": 1.0, "avg_volume": 1.0, "minutes_since_open": 30.0, "minutes_until_close": 120.0}

    def get_ohlc_bars(self, symbol: str) -> list[dict[str, Any]]:
        return [{"timestamp": f"t{i}"} for i in range(35)]


class _PositionManager:
    def get_position(self, symbol: str) -> None:
        return None


def test_generate_signal_handles_none_indicators_without_crash() -> None:
    manager = StrategyManager(strategies=[], indicator_engine=_IndicatorEngineNone(), position_manager=_PositionManager())
    assert manager.generate_signal("NSE:NIFTY", 25000.0) is None


def test_generate_signal_injects_history_count_for_strategies() -> None:
    captured: dict[str, Any] = {}

    class _Strategy:
        name = "SMC"
        MIN_BARS_REQUIRED = 1

        def get_required_indicators(self) -> list[str]:
            return []

        def generate_signal(self, symbol: str, indicators: Mapping[str, Any], current_price: float, position: Any) -> None:
            captured.update(indicators)
            return None

    manager = StrategyManager(strategies=[_Strategy()], indicator_engine=_IndicatorEngineBars(), position_manager=_PositionManager())
    manager.generate_signal("NSE:NIFTY", 25000.0)
    assert captured.get("indicator_history_count") == 35
    assert captured.get("history_count") == 35
