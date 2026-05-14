from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from nifty_scalper_bot.strategies.runner import StrategyRunner
from nifty_scalper_bot.strategies.signal_generator import Signal


class _Snapshot:
    def __init__(self) -> None:
        self.ltp = 382.55
        self.bid = 382.4
        self.ask = 382.7
        self.tradable_quote = True
        self.source = 'ws'
        self.tick_age_s = 0.1
        self.real_ticks_last_60s = 20


def test_runner_preserves_candidate_metadata_after_materialize(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = MagicMock()
    runner._runtime_data_hard_ready = True
    runner._runtime_evaluation_ready = True
    runner._runtime_live_orders_armed = True
    runner._runtime_readiness_reason = None
    runner._order_manager = MagicMock()
    runner._order_manager.is_kill_switch_active = MagicMock(return_value=False)
    runner._reset_execution_state = MagicMock()
    runner._trade_candidate_selector = MagicMock()
    runner._market_data = SimpleNamespace(get_symbol_snapshot=lambda _symbol: _Snapshot())
    runner._extract_underlying = MagicMock(return_value='NIFTY')
    runner._signal_reject_cooldown_ts = {}
    runner._premium_squeeze_last_signal_ts = {}
    runner._underlying_last_signal_ts = {}
    runner._reason_last_signal_ts = {}
    runner._order_attempt_window = []
    runner._max_order_attempts_per_minute = 100
    runner._underlying_signal_cooldown_seconds = 0.0
    runner._reason_signal_cooldown_seconds = 0.0
    runner._cooldown_log_throttle_seconds = 1.0
    runner._is_option_symbol_tick_fresh = MagicMock(return_value=True)
    runner._materialize_option_trade_plan = MagicMock(side_effect=lambda s, **_: s)
    runner._compute_regime_snapshot = MagicMock(return_value=SimpleNamespace(value='normal'))
    runner._last_regime_inputs_by_symbol = {}
    runner._strategy_regime_decision = MagicMock(return_value=(True, 'ok'))
    runner._reject_signal_execution = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(accepted=False, reason=kwargs['reason']))
    runner._validate_entry_signal_strength = MagicMock(return_value=(True, 'ok'))
    runner._validate_execution_price = MagicMock(return_value=(True, 'ok'))
    runner._place_order_via_manager = MagicMock(return_value=SimpleNamespace(accepted=True, reason='ok'))

    candidate = SimpleNamespace(
        symbol='NFO:NIFTY26MAY23400CE', score=7.2, data_quality_score=8.3, rr=1.8,
        spread_pct=0.1, stop_loss=372.98, target=401.67, entry_price=382.55, side='CE'
    )
    runner._trade_candidate_selector.select_best_candidate.return_value = candidate

    signal = Signal(
        action='BUY', symbol='NFO:NIFTY26MAY23400CE', quantity=1, confidence=0.65,
        reason='SMC', stop_loss=372.98, take_profit=401.67,
        metadata={'preliminary_only': True, 'requires_runner_final_score': True, 'direction_score': 6.0, 'strategy_score': 6.0,
                  'candidate_snapshots': [{'symbol': 'NFO:NIFTY26MAY23400CE', 'atm_strike': 23400, 'bid': 382.4, 'ask': 382.7, 'tradable_quote': True}]}
    )
    result = runner._handle_entry_signal_inner(signal, signal.symbol, signal.symbol, 382.55, datetime.now(timezone.utc), trace_id='t1')
    assert result.reason != 'missing_final_score_components'
    passed_signal = runner._materialize_option_trade_plan.call_args.args[0]
    md = passed_signal.metadata
    assert md['candidate_selected'] is True
    assert md['candidate_symbol'] == candidate.symbol
    assert md['option_score'] > 0
    assert md['data_score'] > 0
    assert md['rr_score'] > 0
    assert md['quote_usable_for_order_plan'] is True
