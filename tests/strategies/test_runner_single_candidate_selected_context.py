from types import SimpleNamespace
from unittest.mock import MagicMock

from nifty_scalper_bot.strategies.runner import StrategyRunner
from nifty_scalper_bot.strategies.signal_generator import Signal


class _Snapshot:
    ltp = 100.0
    bid = 99.9
    ask = 100.1
    tradable_quote = True
    source = 'ws'
    tick_age_s = 0.1
    real_ticks_last_60s = 5


def test_single_candidate_uses_active_selected_context():
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = MagicMock()
    runner._market_data = SimpleNamespace(get_symbol_snapshot=lambda _symbol: _Snapshot())
    runner._active_selected_ce = 'NFO:NIFTY26MAY23400CE'
    runner._active_selected_pe = None
    signal = Signal(action='BUY', symbol='NFO:NIFTY26MAY23400CE', quantity=1, confidence=0.5, reason='SMC', stop_loss=95.0, take_profit=110.0, metadata={})
    candidate = runner._build_single_candidate_from_signal(signal=signal, metadata={}, option_side='CE')
    assert candidate is not None
    assert candidate['candidate_selected'] is True
    assert candidate['is_selected_option'] is True
    assert candidate['selected_ce'] == 'NFO:NIFTY26MAY23400CE'
