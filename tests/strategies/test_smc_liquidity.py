from __future__ import annotations

import logging

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    SMCStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy


def test_smc_direction_context_not_ready_log_is_throttled(monkeypatch, caplog):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("SMC_DIRECTION_CONTEXT_NO_VOTE_LOG_THROTTLE_SECONDS", "60")
    strategy = SMCStrategy(SMCStrategyConfig(), indicator_engine=None)
    indicators = {"high": 101, "low": 99, "close": 100, "open": 100, "atr": 1}

    with caplog.at_level(logging.WARNING):
        assert (
            strategy._evaluate_signal("NFO:NIFTY26JUN25000CE", dict(indicators), 100.0)
            is None
        )
        assert (
            strategy._evaluate_signal("NFO:NIFTY26JUN25000CE", dict(indicators), 100.0)
            is None
        )

    records = [
        rec
        for rec in caplog.records
        if "direction_context_not_ready" in rec.getMessage()
    ]
    assert len(records) == 1


def test_smc_with_cold_history_is_skipped_before_invocation(caplog):
    from nifty_scalper_bot.strategies.signal_generator import StrategyManager

    class ColdEngine:
        def get_indicators(self, symbol, required):
            return {
                "bar_count": 6,
                "vix": 15,
                "volume": 1,
                "avg_volume": 1,
                "minutes_since_open": 10,
                "minutes_until_close": 300,
            }

        def get_ohlc_bars(self, symbol):
            return [
                {"timestamp": i, "open": 1, "high": 1, "low": 1, "close": 1}
                for i in range(6)
            ]

    class PositionManager:
        def get_position(self, symbol):
            return None

    class ExplodingSMC:
        name = "SMC"
        MIN_BARS_REQUIRED = 30
        config = {}

        def get_required_indicators(self):
            return []

        def generate_signal(self, *args, **kwargs):
            raise AssertionError("SMC should not be invoked while history is cold")

    manager = StrategyManager([ExplodingSMC()], ColdEngine(), PositionManager())

    with caplog.at_level(logging.INFO):
        assert manager.generate_signal("NFO:NIFTY26JUN25000CE", 100.0) is None

    assert any(
        "STRATEGY_SKIPPED_HISTORY_COLD" in record.getMessage()
        for record in caplog.records
    )
