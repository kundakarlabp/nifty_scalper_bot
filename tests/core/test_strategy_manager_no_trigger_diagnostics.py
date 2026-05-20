from __future__ import annotations

import logging

from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.core.strategy_manager import StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


class _IE:
    def get_history(self, symbol: str):
        return [1] * 30

    def get_indicators(self, symbol: str, names):
        return {}


class _PM:
    def get_position(self, symbol: str):
        return None


def test_no_trigger_vote_logs_context_trigger_details(caplog):
    m = StrategyManager([], _IE(), _PM())
    signal = Signal(action='BUY', symbol='NFO:NIFTY26MAY23750PE', quantity=8.0, confidence=0.8, metadata={'role': 'context', 'trigger_block_reason': 'negative_premium_flow'})
    vote = StrategyVote(strategy='OrderFlow', side='PE', confidence=0.8, score=8.0, metadata={'role': 'context', 'trigger_block_reason': 'negative_premium_flow'})
    with caplog.at_level(logging.INFO):
        out = m._combine_strategy_votes(symbol='NFO:NIFTY26MAY23750PE', signals=[(signal, vote)], indicators={})
    assert out is None
    assert 'blocked_at=no_trigger_vote' in caplog.text
    assert 'context_trigger_details' in caplog.text
    assert 'trigger_block_reason' in caplog.text


def test_context_promotion_live_default_false(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.delenv('STRATEGY_CONTEXT_PROMOTION_LIVE_ALLOWED', raising=False)
    m = StrategyManager([], _IE(), _PM())
    profile = m.get_strategy_mode_profile()
    assert profile['allow_context_promotion'] is False


def test_context_promotion_live_can_be_enabled_explicitly(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('STRATEGY_CONTEXT_PROMOTION_LIVE_ALLOWED', 'true')
    m = StrategyManager([], _IE(), _PM())
    profile = m.get_strategy_mode_profile()
    assert profile['allow_context_promotion'] is True
