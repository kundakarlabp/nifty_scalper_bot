from __future__ import annotations

import logging
from typing import Any

from nifty_scalper_bot.strategies.signal_generator import StrategyManager


class _IndicatorEngine:
    def get_indicators(self, symbol: str, names: list[str] | None = None) -> dict[str, Any]:
        del symbol, names
        return {"bar_count": 40, "vix": 15.0, "volume": 1.0, "avg_volume": 1.0, "minutes_since_open": 30.0, "minutes_until_close": 120.0}


class _PositionManager:
    def get_position(self, symbol: str) -> None:
        del symbol
        return None


class _VWAPProLike:
    name = "VWAPPro"
    MIN_BARS_REQUIRED = 1

    def get_required_indicators(self) -> list[str]:
        return []

    def generate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any) -> None:
        del symbol, indicators, current_price, position
        return None


def test_strategy_no_vote_summary_includes_domain_skip_reason(caplog) -> None:
    manager = StrategyManager(
        strategies=[_VWAPProLike()],
        indicator_engine=_IndicatorEngine(),
        position_manager=_PositionManager(),
    )
    caplog.set_level(logging.INFO)
    manager.generate_signal("NSE:NIFTY", 25000.0)
    rec = next((r for r in caplog.records if "STRATEGY_NO_VOTE_SUMMARY" in r.message), None)
    assert rec is not None
    reasons = getattr(rec, "strategy_reasons", None)
    assert isinstance(reasons, dict)
    assert reasons.get("VWAPPro") == "context_symbol_skipped_for_option_strategy"
