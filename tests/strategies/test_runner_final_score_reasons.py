from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.strategies.runner import StrategyRunner
from nifty_scalper_bot.strategies.signal_generator import Signal


class _Snapshot:
    def __init__(self, bid=0.0, ask=0.0, tradable_quote=False):
        self.ltp = 431.05
        self.bid = bid
        self.ask = ask
        self.tradable_quote = tradable_quote
        self.source = 'ws'
        self.tick_age_s = 0.1
        self.real_ticks_last_60s = 5


def _runner() -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = MagicMock()
    runner._runtime_data_hard_ready = True
    runner._runtime_evaluation_ready = True
    runner._runtime_live_orders_armed = True
    runner._runtime_readiness_reason = None
    runner._order_manager = MagicMock()
    runner._order_manager.is_kill_switch_active = MagicMock(return_value=False)
    runner._reset_execution_state = MagicMock()
    runner._materialize_option_trade_plan = MagicMock(side_effect=lambda s, **_: s)
    return runner


def _signal(metadata: dict) -> Signal:
    return Signal(action='BUY', symbol='NFO:NIFTY26MAY23350CE', quantity=1, confidence=0.65, reason='VWAPPro', stop_loss=420.27, take_profit=452.60, metadata=metadata)


@pytest.mark.parametrize(
    'metadata,expected',
    [
        ({'preliminary_only': True, 'requires_runner_final_score': True, 'direction_score': 6.0, 'strategy_score': 6.0, 'option_score': 6.0, 'data_score': 8.0, 'candidate_selected': True, 'quote_usable_for_order_plan': True}, 'missing_final_score_components'),
        ({'preliminary_only': True, 'requires_runner_final_score': True, 'direction_score': 6.0, 'strategy_score': 6.0, 'option_score': 6.0, 'data_score': 8.0, 'rr_score': 7.0, 'candidate_selected': False, 'quote_usable_for_order_plan': True}, 'candidate_not_selected'),
        ({'preliminary_only': True, 'requires_runner_final_score': True, 'direction_score': 6.0, 'strategy_score': 6.0, 'option_score': 6.0, 'data_score': 8.0, 'rr_score': 7.0, 'candidate_selected': True, 'quote_usable_for_order_plan': False}, 'quote_not_usable_for_order_plan'),
    ],
)
def test_precheck_reasons(monkeypatch, metadata, expected):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner = _runner()
    runner._market_data = SimpleNamespace(get_symbol_snapshot=lambda _symbol: _Snapshot())
    result = runner._handle_entry_signal_inner(_signal(metadata), 'NFO:NIFTY26MAY23350CE', 'NFO:NIFTY26MAY23350CE', 431.05, datetime.now(timezone.utc), trace_id='r1')
    assert result.reason == expected
