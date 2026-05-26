from __future__ import annotations

from typing import Any, Iterable, Mapping

from nifty_scalper_bot.strategies.signal_generator import StrategyManager, build_strategy_history_context


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


class _DataHubBars:
    def get_ohlc_bars(self, symbol: str) -> list[dict[str, Any]]:
        return [{"timestamp": f"dh{i}"} for i in range(12)]


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


def test_build_strategy_history_context_uses_data_hub_source() -> None:
    ctx = build_strategy_history_context(
        symbol="NFO:NIFTY26MAY24350CE",
        indicator_engine=_IndicatorEngineBars(),
        data_hub=_DataHubBars(),
    )
    assert ctx["history_source"] == "data_hub"
    assert ctx["indicator_history_count"] == 12
    assert ctx["option_history_count"] == 12


def test_build_strategy_history_context_sets_underlying_for_non_option() -> None:
    ctx = build_strategy_history_context(
        symbol="NFO:NIFTY26MAY24FUT",
        indicator_engine=_IndicatorEngineBars(),
        data_hub=None,
    )
    assert ctx["history_domain_used"] == "underlying"
    assert ctx["underlying_history_count"] == 35
    assert ctx["spot_history_count"] == 0


def test_build_strategy_history_context_sets_spot_domain_for_nifty_spot() -> None:
    ctx = build_strategy_history_context(
        symbol="NSE:NIFTY",
        indicator_engine=_IndicatorEngineBars(),
        data_hub=None,
    )
    assert ctx["history_domain_used"] == "spot"
    assert ctx["spot_history_count"] == 35
    assert ctx["underlying_history_count"] == 0


def test_build_strategy_history_context_falls_back_to_indicator_engine_when_data_hub_empty() -> None:
    class _DataHubEmpty:
        def get_ohlc_bars(self, symbol: str) -> list[dict[str, Any]]:
            return []

    ctx = build_strategy_history_context(
        symbol="NFO:NIFTY26MAY24350PE",
        indicator_engine=_IndicatorEngineBars(),
        data_hub=_DataHubEmpty(),
    )
    assert ctx["history_source"] == "indicator_engine"
    assert ctx["history_domain_used"] == "options"
    assert ctx["history_resolved_count"] == 35
