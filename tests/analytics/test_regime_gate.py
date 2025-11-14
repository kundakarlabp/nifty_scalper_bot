from datetime import datetime, timedelta

import pandas as pd

from nifty_scalper_bot.analytics.regime_gate import (
    RegimeGate,
    RegimeGateConfig,
    RegimeKind,
)


def _trend_frame(start: datetime, bars: int) -> pd.DataFrame:
    rows = []
    for idx in range(bars):
        price = 100 + idx * 2
        rows.append(
            {
                "timestamp": start + timedelta(minutes=idx),
                "open": price - 1,
                "high": price + 1,
                "low": price - 2,
                "close": price,
                "volume": 1000 + idx * 10,
            }
        )
    frame = pd.DataFrame(rows).set_index("timestamp")
    return frame


def test_regime_gate_identifies_trend_and_bias() -> None:
    start = datetime(2024, 1, 1, 9, 30)
    config = RegimeGateConfig(
        warmup_bars=6,
        adx_period=5,
        atr_period=5,
        atr_smooth_period=5,
        adx_min_trend=0.5,
        open_buffer_minutes=0,
        preclose_buffer_minutes=0,
        vwap_max_trend_distance_pct=0.1,
    )
    gate = RegimeGate(config)
    frame = _trend_frame(start, 8)
    snapshot = None
    for timestamp, row in frame.iterrows():
        snapshot = gate.observe(timestamp.to_pydatetime(), row)
    assert snapshot is not None
    assert snapshot.regime is RegimeKind.TREND
    bias_allowed = gate.allow_signal(start + timedelta(minutes=10), bias="trend")
    bias_blocked = gate.allow_signal(start + timedelta(minutes=10), bias="mean_revert")
    assert bias_allowed is True
    assert bias_blocked is False


def test_regime_gate_blocks_no_trade_windows() -> None:
    start = datetime(2024, 1, 1, 9, 15)
    config = RegimeGateConfig(
        warmup_bars=4, open_buffer_minutes=5, preclose_buffer_minutes=10
    )
    gate = RegimeGate(config)
    frame = _trend_frame(start, 6)
    for timestamp, row in frame.iterrows():
        gate.observe(timestamp.to_pydatetime(), row)
    assert gate.snapshot is not None
    open_time = datetime(2024, 1, 1, 9, 17)
    lunch_time = datetime(2024, 1, 1, 12, 20)
    close_time = datetime(2024, 1, 1, 15, 25)
    assert gate.allow_signal(open_time, bias="trend") is False
    assert gate.allow_signal(lunch_time, bias="trend") is False
    assert gate.allow_signal(close_time, bias="trend") is False
