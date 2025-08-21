from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.config import settings


@dataclass
class Signal:
    side: str                 # "BUY" or "SELL"
    confidence: float
    score: int
    entry_price: float
    sl_points: float
    tp_points: float
    reasons: Sequence[str]


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def make_signal(spot_df: pd.DataFrame, opt_df: pd.DataFrame, *, regime: str = "auto") -> Optional[Signal]:
    """
    Simple but consistent rule-set:
    - EMA(9) vs EMA(21)
    - RSI(14) > 50 for BUY, < 50 for SELL
    - ATR-based SL/TP with regime adjustments and confidence tilt
    """
    if len(spot_df) < settings.strategy.min_bars_for_signal:
        return None

    close = spot_df["close"].astype(float)
    ema_fast = _ema(close, settings.strategy.ema_fast)
    ema_slow = _ema(close, settings.strategy.ema_slow)

    rsi = _rsi(close, settings.strategy.rsi_period)
    atr = _atr(spot_df, settings.strategy.atr_period)

    reasons = []

    side = None
    score = 0
    if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
        score += 1
        if rsi.iloc[-1] > 50:
            side = "BUY"
            score += 1
            reasons += ["EMA fast above slow", "RSI>50"]
    elif ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        score += 1
        if rsi.iloc[-1] < 50:
            side = "SELL"
            score += 1
            reasons += ["EMA fast below slow", "RSI<50"]

    if side is None or score < settings.strategy.min_signal_score:
        return None

    # Base SL/TP
    sl_mult = settings.strategy.atr_sl_multiplier
    tp_mult = settings.strategy.atr_tp_multiplier

    # Confidence tilt via RSI distance from 50
    conf = min(5.0, abs(float(rsi.iloc[-1]) - 50.0) / 5.0)  # 0..5
    sl_mult += settings.strategy.sl_confidence_adj * (conf / 5.0)
    tp_mult += settings.strategy.tp_confidence_adj * (conf / 5.0)

    # Regime tilt
    if regime == "trend":
        tp_mult += settings.strategy.trend_tp_boost
        sl_mult += settings.strategy.trend_sl_relax
        reasons.append("Regime: trend")
    elif regime == "range":
        tp_mult += settings.strategy.range_tp_tighten
        sl_mult += settings.strategy.range_sl_tighten
        reasons.append("Regime: range")

    sl_points = max(atr.iloc[-1] * sl_mult, 5.0)
    tp_points = max(atr.iloc[-1] * tp_mult, sl_points * 1.2)

    entry_price = float(opt_df["close"].astype(float).iloc[-1])

    return Signal(
        side=side,
        confidence=conf,
        score=score,
        entry_price=entry_price,
        sl_points=float(sl_points),
        tp_points=float(tp_points),
        reasons=reasons,
    )


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()