from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


class _Engine:
    def get_history(self, _symbol: str):
        return [{}] * 100

    def get_indicators(self, _symbol: str, _names=None):
        return {
            "vwap": 100.0,
            "avg_volume": 1000.0,
            "volume": 1200.0,
            "atr": 2.0,
            "direction_bias": "CE",
            "underlying_direction_bias": "CE",
            "underlying_direction_confidence": 0.85,
            "context_age_seconds": 0.1,
            "selected_ce": "NFO:NIFTY2662324050CE",
            "selected_pe": "NFO:NIFTY2662324050PE",
            "is_selected_option": True,
            "strike_distance_from_atm": 0.0,
            "bid": 99.5,
            "ask": 100.0,
            "spread_pct": 0.5,
            "quote_depth_valid": True,
            "tradable_quote": True,
            "tick_age_ms": 100.0,
            "quote_update_version": 10,
            "depth": {"buy": [{"quantity": 100}], "sell": [{"quantity": 100}]},
        }


class _Positions:
    def get_position(self, _symbol: str):
        return None


class _Strategy:
    name = "SMC"
    last_no_vote_reason = "none"

    def generate_signal(self, symbol, indicators, current_price, position):  # noqa: ANN001
        return Signal(
            action="BUY",
            symbol=symbol,
            quantity=3,
            confidence=0.88,
            reason="test",
            stop_loss=current_price - 5,
            take_profit=current_price + 10,
            metadata={
                "strategy": "SMC",
                "strategy_name": "SMC",
                "side": "CE",
                "trade_side": "CE",
                "contract_side": "CE",
                "raw_vote_score": 9.0,
                "raw_setup_score": 9.0,
                "strategy_score": 9.0,
                "is_selected_option": True,
                "selected_ce": indicators["selected_ce"],
                "selected_pe": indicators["selected_pe"],
                "strike_distance_from_atm": 0.0,
                "quote_depth_valid": True,
                "tradable_quote": True,
                "spread_pct": 0.5,
                "tick_age_ms": 100.0,
                "quote_update_version": 10,
            },
        )


def test_strategy_manager_exit_reports_trade_score_not_quantity(monkeypatch, caplog) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setenv("STRATEGY_TRIGGER_MIN_SCORE", "4.5")
    monkeypatch.setenv("LIVE_MAX_SPREAD_PCT", "0.75")

    manager = StrategyManager([_Strategy()], _Engine(), _Positions())
    manager._data_hub = SimpleNamespace(indicators_ready=True)
    manager._required_candles = 30
    manager._use_regime_adaptive = False
    manager._regime_manager = None
    manager._orchestrator = None
    manager._filter_signal = lambda _signal: True
    manager._compute_trade_quality_score = lambda *args, **kwargs: (8.5, {"quality_block_reason": "ok"})
    manager._combine_strategy_votes = lambda **kwargs: Signal(
        action="BUY",
        symbol="NFO:NIFTY2662324050CE",
        quantity=3,
        confidence=0.88,
        reason="approved",
        stop_loss=95.0,
        take_profit=110.0,
        metadata={
            "is_approved": True,
            "final_trade_score": 8.75,
            "setup_score": 9.0,
            "raw_setup_score": 9.0,
            "consensus_score": 8.75,
            "approval_path": "aligned_two_trigger_consensus",
        },
    )

    with caplog.at_level(logging.DEBUG, logger="nifty_scalper_bot.core.strategy_manager"):
        signal = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0, trace_id="score-diag")

    assert signal is not None
    exit_records = [record for record in caplog.records if getattr(record, "event", "") == "STRATEGY_MANAGER_EXIT"]
    assert exit_records
    assert exit_records[-1].signal_score == pytest.approx(8.75)
    assert exit_records[-1].signal_quantity == 3
