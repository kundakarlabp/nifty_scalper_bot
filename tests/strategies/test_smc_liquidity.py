from __future__ import annotations

import logging

from nifty_scalper_bot.strategies.elite_strategies.config_models import SMCStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy


def test_smc_direction_context_not_ready_log_is_throttled(monkeypatch, caplog):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("SMC_DIRECTION_CONTEXT_NO_VOTE_LOG_THROTTLE_SECONDS", "60")
    strategy = SMCStrategy(SMCStrategyConfig(), indicator_engine=None)
    indicators = {"high": 101, "low": 99, "close": 100, "open": 100, "atr": 1}

    with caplog.at_level(logging.WARNING):
        assert strategy._evaluate_signal("NFO:NIFTY26JUN25000CE", dict(indicators), 100.0) is None
        assert strategy._evaluate_signal("NFO:NIFTY26JUN25000CE", dict(indicators), 100.0) is None

    records = [rec for rec in caplog.records if "direction_context_not_ready" in rec.getMessage()]
    assert len(records) == 1
