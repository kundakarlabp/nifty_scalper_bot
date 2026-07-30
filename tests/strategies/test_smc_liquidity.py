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


def test_real_smc_generate_signal_skips_cold_history_before_evaluate(
    monkeypatch, caplog
):
    strategy = SMCStrategy(SMCStrategyConfig(), indicator_engine=None)
    called = []

    def _spy(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError(
            "SMC _evaluate_signal should not be called below MIN_BARS_REQUIRED"
        )

    monkeypatch.setattr(strategy, "_evaluate_signal", _spy)
    indicators = {
        "high": 101,
        "low": 99,
        "close": 100,
        "open": 100,
        "atr": 1,
        "history_count": 6,
        "history_resolved_count": 6,
        "option_history_count": 6,
    }

    with caplog.at_level(logging.INFO):
        assert (
            strategy.generate_signal("NFO:NIFTY26JUN25000CE", indicators, 100.0) is None
        )

    assert called == []
    assert any(
        "STRATEGY_SKIPPED_HISTORY_COLD" in record.getMessage()
        for record in caplog.records
    )


def _ready_smc_indicators(**overrides):
    indicators = {
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "open": 99.5,
        "atr": 1.0,
        "direction_bias": "CE",
        "underlying_direction_bias": "CE",
        "premium_reclaim": True,
        "bullish_reversal": True,
        "choch_confirmed": True,
        "bos_confirmed": True,
        "retest_confirmed": True,
    }
    indicators.update(overrides)
    return indicators


def test_green_candle_without_swing_is_not_a_liquidity_sweep(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    strategy = SMCStrategy(SMCStrategyConfig(), indicator_engine=None)

    assert (
        strategy._evaluate_signal(
            "NFO:NIFTY26JUN25000CE", _ready_smc_indicators(), 100.0
        )
        is None
    )
    assert strategy.last_no_vote_reason == "no_liquidity_sweep"


def test_red_candle_without_swing_is_not_a_liquidity_sweep(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    strategy = SMCStrategy(SMCStrategyConfig(), indicator_engine=None)
    indicators = _ready_smc_indicators(
        open=100.5,
        close=99.5,
        direction_bias="PE",
        underlying_direction_bias="PE",
    )

    assert strategy._evaluate_signal("NFO:NIFTY26JUN25000PE", indicators, 100.0) is None
    assert strategy.last_no_vote_reason == "no_liquidity_sweep"


def test_bullish_swing_breach_without_reclaim_is_not_a_sweep(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    strategy = SMCStrategy(SMCStrategyConfig(), indicator_engine=None)
    indicators = _ready_smc_indicators(
        high=99.0,
        low=97.0,
        open=98.5,
        close=97.5,
        prior_swing_low=98.0,
        prior_swing_high=101.0,
    )

    assert strategy._evaluate_signal("NFO:NIFTY26JUN25000CE", indicators, 100.0) is None
    assert strategy.last_no_vote_reason == "no_liquidity_sweep"


def test_bullish_swing_breach_and_reclaim_is_a_valid_sweep(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    strategy = SMCStrategy(SMCStrategyConfig(), indicator_engine=None)
    indicators = _ready_smc_indicators(
        low=98.0,
        prior_swing_low=99.0,
        prior_swing_high=102.0,
    )

    signal = strategy._evaluate_signal("NFO:NIFTY26JUN25000CE", indicators, 100.0)

    assert signal is not None
    assert signal.signal == "BUY"
    assert signal.metadata["trade_side"] == "CE"


def test_bearish_option_premium_sweep_is_rejected_before_buy(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    strategy = SMCStrategy(SMCStrategyConfig(), indicator_engine=None)
    indicators = _ready_smc_indicators(
        high=102.5,
        low=100.0,
        open=101.5,
        close=100.5,
        prior_swing_low=99.0,
        prior_swing_high=102.0,
        direction_bias="PE",
        underlying_direction_bias="PE",
    )

    assert strategy._evaluate_signal("NFO:NIFTY26JUN25000PE", indicators, 100.0) is None
    assert strategy.last_no_vote_reason == "premium_not_reversing_up"


def test_bearish_underlying_sweep_can_still_select_pe(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    strategy = SMCStrategy(SMCStrategyConfig(), indicator_engine=None)
    indicators = _ready_smc_indicators(
        high=102.5,
        low=100.0,
        open=101.5,
        close=100.5,
        prior_swing_low=99.0,
        prior_swing_high=102.0,
        direction_bias="PE",
        underlying_direction_bias="PE",
        source_symbol="NSE:NIFTY 50",
    )

    signal = strategy._evaluate_signal("NFO:NIFTY26JUN25000PE", indicators, 100.0)

    assert signal is not None
    assert signal.signal == "BUY"
    assert signal.metadata["trade_side"] == "PE"
    assert signal.metadata["source_domain"] == "underlying_price"


def test_explicit_liquidity_sweep_confirmation_remains_supported(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    strategy = SMCStrategy(SMCStrategyConfig(), indicator_engine=None)
    indicators = _ready_smc_indicators(liquidity_sweep_confirmed=True)

    signal = strategy._evaluate_signal("NFO:NIFTY26JUN25000CE", indicators, 100.0)

    assert signal is not None
    assert signal.signal == "BUY"
    assert signal.metadata["trade_side"] == "CE"


def test_manager_swing_low_alias_is_accepted(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    strategy = SMCStrategy(SMCStrategyConfig(), indicator_engine=None)
    indicators = _ready_smc_indicators(
        low=98.0,
        swing_low=99.0,
        swing_high=102.0,
    )

    signal = strategy._evaluate_signal("NFO:NIFTY26JUN25000CE", indicators, 100.0)

    assert signal is not None
    assert signal.signal == "BUY"
    assert signal.metadata["trade_side"] == "CE"
