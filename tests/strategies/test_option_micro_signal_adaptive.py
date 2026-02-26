from __future__ import annotations

from nifty_scalper_bot.strategies.option_micro_signal import OptionMicroSignal


def test_percentile_microvol_filter_blocks_when_below_threshold() -> None:
    signal = OptionMicroSignal(
        spread_limit=5.0,
        min_depth=1,
        snmom_threshold=0.01,
        microvol_threshold=0.0,
        adverse_ticks=2,
        microvol_percentile=90.0,
    )
    signal.on_tick(bid=100.0, ask=100.1, last_price=100.1, last_size=1, depth=10, ts_ns=0)
    ts = 1
    for _ in range(220):
        signal.on_tick(bid=100.0, ask=100.5, last_price=100.5, last_size=1, depth=10, ts_ns=ts)
        ts += 1
    decision = signal.on_tick(bid=100.5001, ask=100.55, last_price=100.55, last_size=1, depth=10, ts_ns=ts)
    assert decision.enter is False


def test_logistic_confidence_probability_in_range() -> None:
    signal = OptionMicroSignal(
        spread_limit=5.0,
        min_depth=1,
        snmom_threshold=0.01,
        microvol_threshold=0.0,
        adverse_ticks=2,
    )
    signal.on_tick(bid=100.0, ask=100.1, last_price=100.1, last_size=1, depth=20, ts_ns=0)
    decision = signal.on_tick(bid=100.1, ask=100.3, last_price=100.3, last_size=2, depth=50, ts_ns=1)
    assert 0.0 <= decision.probability <= 1.0
    assert decision.features['probability'] == decision.probability
