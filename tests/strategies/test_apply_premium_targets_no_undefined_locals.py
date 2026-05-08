from types import SimpleNamespace

from nifty_scalper_bot.strategies.runner import Signal, StrategyRunner


def test_apply_premium_targets_no_undefined_locals():
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None)
    signal = Signal(action='BUY', symbol='X', quantity=1, confidence=0.5, reason='r', stop_loss=None, take_profit=None, metadata={'premium_stop_pct': 0.1, 'premium_target_rr': 2.0})
    out = runner._apply_premium_targets(signal, premium=100.0, entry_side='BUY')
    assert out.stop_loss is not None and out.take_profit is not None
