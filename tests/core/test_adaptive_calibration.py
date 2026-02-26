from __future__ import annotations

from nifty_scalper_bot.core.adaptive_calibration import AdaptiveParameterStore, WalkForwardOptimizer


def test_adaptive_recalibration_trigger() -> None:
    opt = WalkForwardOptimizer(recalibrate_every=3)
    assert opt.should_recalibrate('s1') is False
    assert opt.should_recalibrate('s1') is False
    assert opt.should_recalibrate('s1') is True


def test_regime_change_blend() -> None:
    store = AdaptiveParameterStore(window_trades=10)
    stats = store.record_trade('s1', 10.0)
    opt = WalkForwardOptimizer(recalibrate_every=1)
    current = {'momentum_z_threshold': 1.0, 'microvol_percentile': 60.0, 'spread_threshold_pct': 0.2}
    tuned = opt.optimize('s1', 'trend', stats, current)
    shifted = opt.on_regime_change('s1', 'trend', current)
    assert shifted.keys() == tuned.keys()
