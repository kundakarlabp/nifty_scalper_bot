from __future__ import annotations

from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


def _make_signal() -> Signal:
    return Signal(
        action='BUY',
        symbol='NFO:NIFTY25000CE',
        quantity=1,
        confidence=0.6,
        reason='test',
        stop_loss=10.0,
        take_profit=20.0,
        metadata={'strategy_score': 0.0},
    )


def _manager_stub() -> StrategyManager:
    manager = object.__new__(StrategyManager)
    manager._combine_signals = lambda signals: signals[0] if signals else None
    return manager


def test_single_low_score_vote_returns_none() -> None:
    manager = _manager_stub()
    signal = _make_signal()
    vote = StrategyVote(
        strategy='SMC', side='CE', score=8.0, confidence=0.8, reasons=[], metadata={}
    )
    combined = manager._combine_strategy_votes(
        symbol='NIFTY',
        signals=[(signal, vote)],
        indicators={'direction_score': 9.0, 'data_score': 9.0, 'option_score': 9.0},
    )
    assert combined is None


def test_single_high_score_vote_returns_preliminary_consensus() -> None:
    manager = _manager_stub()
    signal = _make_signal()
    vote = StrategyVote(
        strategy='SMC', side='CE', score=9.1, confidence=0.9, reasons=[], metadata={}
    )
    combined = manager._combine_strategy_votes(
        symbol='NIFTY',
        signals=[(signal, vote)],
        indicators={'direction_score': 6.0, 'data_score': 6.0, 'option_score': 6.0},
    )
    assert combined is not None
    assert combined.metadata is not None
    assert combined.metadata.get('consensus_stage') == 'preliminary_single_high_conviction'


def test_strategy_manager_single_vote_uses_indicator_selected_context(monkeypatch) -> None:
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'true')
    manager = _manager_stub()
    signal = _make_signal()
    vote = StrategyVote(strategy='SMC', side='CE', score=6.8, confidence=0.8, reasons=[], metadata={})
    combined = manager._combine_strategy_votes(
        symbol='NIFTY',
        signals=[(signal, vote)],
        indicators={'selected_ce': 'NFO:NIFTY25000CE', 'selected_pe': 'NFO:NIFTY25000PE', 'atm_strike': 25000, 'strike_distance_from_atm': 0},
    )
    assert combined is not None
    assert combined.metadata is not None
    assert combined.metadata.get('consensus_stage') == 'single_vote_scalp_controlled'


def test_single_vote_scalp_enabled_allows_valid_near_atm_vote(monkeypatch) -> None:
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'true')
    manager = _manager_stub()
    signal = _make_signal()
    vote = StrategyVote(strategy='VWAPPro', side='CE', score=6.6, confidence=0.7, reasons=[], metadata={})
    combined = manager._combine_strategy_votes(
        symbol='NIFTY',
        signals=[(signal, vote)],
        indicators={'atm_strike': 25000, 'strike_distance_from_atm': 50, 'selected_ce': None, 'selected_pe': None},
    )
    assert combined is not None
    assert combined.metadata is not None
    assert combined.metadata.get('consensus_stage') == 'single_vote_scalp_controlled'


def test_single_vote_scalp_disabled_rejects_single_vote(monkeypatch) -> None:
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'false')
    manager = _manager_stub()
    signal = _make_signal()
    vote = StrategyVote(strategy='VWAPPro', side='CE', score=7.5, confidence=0.8, reasons=[], metadata={})
    combined = manager._combine_strategy_votes(
        symbol='NIFTY',
        signals=[(signal, vote)],
        indicators={'selected_ce': 'NFO:NIFTY25000CE', 'selected_pe': 'NFO:NIFTY25000PE', 'atm_strike': 25000},
    )
    assert combined is None



def test_context_only_vote_returns_none_and_logs(caplog) -> None:
    manager = _manager_stub()
    signal = _make_signal()
    vote = StrategyVote(
        strategy='OrderFlow',
        side='CE',
        score=6.0,
        confidence=0.7,
        reasons=[],
        metadata={'role': 'context'},
    )
    caplog.set_level('INFO')
    combined = manager._combine_strategy_votes(
        symbol='NFO:NIFTY25000CE',
        signals=[(signal, vote)],
        indicators={},
    )
    assert combined is None
    assert any('STRATEGY_NO_TRIGGER_VOTE' in rec.message for rec in caplog.records)


def test_live_context_only_vote_is_rejected_with_diagnostics(monkeypatch, caplog) -> None:
    from nifty_scalper_bot.core.strategy_manager import StrategyVote
    monkeypatch.setenv('EXECUTION_MODE','LIVE')
    monkeypatch.setenv('STRATEGY_CONTEXT_PROMOTION_LIVE_ALLOWED','false')
    manager = _manager_stub()
    signal = _make_signal()
    vote = StrategyVote(strategy='OrderFlow', side='CE', score=9.0, confidence=0.9, reasons=[], metadata={'role':'context'})
    caplog.set_level('INFO')
    out = manager._combine_strategy_votes(symbol='NFO:NIFTY25000CE', signals=[(signal,vote)], indicators={'context_age_seconds':10})
    assert out is None
    assert any('no_trigger_vote' in r.message for r in caplog.records)


def test_live_context_promotion_rejects_without_depth_even_when_env_enabled(monkeypatch, caplog) -> None:
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('STRATEGY_CONTEXT_PROMOTION_LIVE_ALLOWED', 'true')
    manager = _manager_stub()
    signal = _make_signal()
    signal.metadata = {'is_selected_option': True, 'quote_depth_valid': True, 'spread_pct': 1, 'context_age_seconds': 10}
    vote = StrategyVote(strategy='OrderFlow', side='CE', score=9.0, confidence=0.8, reasons=[], metadata={'role': 'context'})
    caplog.set_level('INFO')
    out = manager._try_context_promotion('NFO:NIFTY25000CE', [(signal, vote)], {'context_age_seconds': 10}, manager.get_strategy_mode_profile())
    assert out is None
    assert any('CONTEXT_PROMOTION_REJECTED_LIVE' in r.message and 'live_context_quote_depth_missing' in r.message for r in caplog.records)


def test_live_context_promotion_passes_only_with_strict_quality(monkeypatch) -> None:
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('STRATEGY_CONTEXT_PROMOTION_LIVE_ALLOWED', 'true')
    monkeypatch.setenv('STRATEGY_CONTEXT_PROMOTION_LIVE_MIN_SCORE', '8.5')
    monkeypatch.setenv('STRATEGY_CONTEXT_PROMOTION_LIVE_MIN_CONFIDENCE', '0.75')
    manager = _manager_stub()
    signal = _make_signal()
    signal.metadata = {
        'role': 'context',
        'strategy': 'OrderFlow',
        'side': 'CE',
        'strategy_score': 9,
        'quote_depth_valid': True,
        'bid': 100,
        'ask': 101,
        'spread_pct': 1,
        'context_age_seconds': 10,
        'is_selected_option': True,
    }
    vote = StrategyVote(strategy='OrderFlow', side='CE', score=9.0, confidence=0.8, reasons=[], metadata={'role': 'context'})
    out = manager._try_context_promotion('NFO:NIFTY25000CE', [(signal, vote)], {'context_age_seconds': 10}, manager.get_strategy_mode_profile())
    assert out is not None
