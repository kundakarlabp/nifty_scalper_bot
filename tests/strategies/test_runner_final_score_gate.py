from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.strategies.runner import StrategyRunner
from nifty_scalper_bot.strategies.signal_generator import Signal


class _Snapshot:
    def __init__(self, *, ltp: float = 431.05, bid: float = 0.0, ask: float = 0.0, tradable_quote: bool = False) -> None:
        self.ltp = ltp
        self.bid = bid
        self.ask = ask
        self.tradable_quote = tradable_quote
        self.source = 'ws'
        self.tick_age_s = 0.1
        self.real_ticks_last_60s = 5


def _build_runner() -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = MagicMock()
    runner._runtime_data_hard_ready = True
    runner._runtime_evaluation_ready = True
    runner._runtime_live_orders_armed = True
    runner._runtime_readiness_reason = None
    runner._order_manager = MagicMock()
    runner._order_manager.is_kill_switch_active = MagicMock(return_value=False)
    runner._order_manager.submit_trade_plan = MagicMock(return_value='order-1')
    runner._reset_execution_state = MagicMock()
    runner._trade_candidate_selector = MagicMock()
    runner._trade_candidate_selector.select_best_candidate = MagicMock(
        return_value=SimpleNamespace(
            symbol='NFO:NIFTY26MAY23350CE',
            score=6.0,
            data_quality_score=8.0,
            rr=1.8,
            side='CE',
            spread_pct=0.35,
            entry_price=431.05,
            stop_loss=420.27,
            target=452.60,
        )
    )
    return runner


def _build_signal(metadata: dict) -> Signal:
    return Signal(
        action='BUY',
        symbol='NFO:NIFTY26MAY23350CE',
        quantity=1,
        confidence=0.65,
        reason='SMC',
        stop_loss=420.27,
        take_profit=452.60,
        metadata=metadata,
    )


def test_final_score_gate_passes_precheck_and_not_generic_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner = _build_runner()
    runner._market_data = SimpleNamespace(get_symbol_snapshot=lambda _symbol: _Snapshot(bid=431.15, ask=432.65, tradable_quote=True))
    metadata = {
        'preliminary_only': True,
        'requires_runner_final_score': True,
        'direction_score': 6.0,
        'strategy_score': 6.0,
        'data_score': 8.0,
        'candidate_selected': True,
        'candidate_snapshots': [
            {'symbol': 'NFO:NIFTY26MAY23350CE', 'bid': 431.15, 'ask': 432.65, 'tradable_quote': True, 'ltp': 431.05, 'atm_strike': 23350}
        ],
        'atm_strike': 23350,
    }
    result = runner._handle_entry_signal_inner(_build_signal(metadata), 'NFO:NIFTY26MAY23350CE', 'NFO:NIFTY26MAY23350CE', 431.05, datetime.now(timezone.utc), trace_id='t1')
    assert result.reason != 'final_score_precheck_failed_unknown'


def test_quote_usable_from_mdm_revalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner = _build_runner()
    runner._market_data = SimpleNamespace(get_symbol_snapshot=lambda _symbol: _Snapshot(bid=431.15, ask=432.65, tradable_quote=True))
    metadata = {
        'preliminary_only': True,
        'requires_runner_final_score': True,
        'direction_score': 6.0,
        'strategy_score': 6.0,
        'data_score': 8.0,
        'candidate_selected': True,
        'candidate_snapshots': [{'symbol': 'NFO:NIFTY26MAY23350CE', 'bid': 0.0, 'ask': 0.0, 'tradable_quote': False, 'ltp': 431.05, 'atm_strike': 23350}],
        'atm_strike': 23350,
    }
    result = runner._handle_entry_signal_inner(_build_signal(metadata), 'NFO:NIFTY26MAY23350CE', 'NFO:NIFTY26MAY23350CE', 431.05, datetime.now(timezone.utc), trace_id='t2')
    assert result.reason != 'final_score_precheck_failed_unknown'


def test_final_score_precheck_failed_unknown_when_snapshot_and_mdm_not_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner = _build_runner()
    runner._market_data = SimpleNamespace(get_symbol_snapshot=lambda _symbol: _Snapshot(bid=0.0, ask=0.0, tradable_quote=False))
    metadata = {
        'preliminary_only': True,
        'requires_runner_final_score': True,
        'direction_score': 6.0,
        'strategy_score': 6.0,
        'data_score': 8.0,
        'candidate_selected': True,
        'candidate_snapshots': [{'symbol': 'NFO:NIFTY26MAY23350CE', 'bid': 0.0, 'ask': 0.0, 'tradable_quote': False, 'ltp': 431.05, 'atm_strike': 23350}],
        'atm_strike': 23350,
    }
    result = runner._handle_entry_signal_inner(_build_signal(metadata), 'NFO:NIFTY26MAY23350CE', 'NFO:NIFTY26MAY23350CE', 431.05, datetime.now(timezone.utc), trace_id='t3')
    assert result.reason == 'final_score_precheck_failed_unknown'


def test_build_single_candidate_uses_data_score_fallback() -> None:
    runner = _build_runner()
    runner._market_data = SimpleNamespace(get_symbol_snapshot=lambda _symbol: _Snapshot(bid=431.15, ask=432.65, tradable_quote=True))
    signal = _build_signal({'data_score': 8.0, 'atm_strike': 23350})
    candidate = runner._build_single_candidate_from_signal(signal=signal, metadata=signal.metadata or {}, option_side='CE')
    assert candidate is not None
    assert candidate['data_quality_score'] == 8.0
