"""Directional bias scoring engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DirectionalBias:
    side: str
    score: float
    reasons: list[str]
    context: dict[str, Any]


class DirectionalBiasEngine:
    def evaluate(self, snapshot: dict[str, Any]) -> DirectionalBias:
        reasons: list[str] = []
        if bool(snapshot.get('spot_stale')) or bool(snapshot.get('futures_stale')):
            return DirectionalBias('NONE', 0.0, ['stale_data'], snapshot)
        price = float(snapshot.get('price') or 0.0)
        vwap = float(snapshot.get('spot_vwap') or snapshot.get('futures_vwap') or 0.0)
        ema9 = float(snapshot.get('ema9') or 0.0)
        ema21 = float(snapshot.get('ema21') or 0.0)
        rsi = float(snapshot.get('rsi') or 50.0)
        macd = float(snapshot.get('macd_hist') or 0.0)
        adx = float(snapshot.get('adx') or 0.0)
        vol_breakout = bool(snapshot.get('volume_breakout'))

        ce = 0
        pe = 0
        if price > vwap > 0:
            ce += 1; reasons.append('price_above_vwap')
        if price < vwap and vwap > 0:
            pe += 1; reasons.append('price_below_vwap')
        if ema9 > ema21:
            ce += 1
        if ema9 < ema21:
            pe += 1
        if 50 <= rsi <= 70:
            ce += 1
        if 30 <= rsi <= 50:
            pe += 1
        if macd > 0:
            ce += 1
        if macd < 0:
            pe += 1
        if adx >= 18 or vol_breakout:
            ce += 1 if ce >= pe else 0
            pe += 1 if pe > ce else 0
        if ce >= 4 and ce > pe:
            return DirectionalBias('CE', min(10.0, ce * 1.8), ['bullish_alignment'], snapshot)
        if pe >= 4 and pe > ce:
            return DirectionalBias('PE', min(10.0, pe * 1.8), ['bearish_alignment'], snapshot)
        return DirectionalBias('NONE', 0.0, ['mixed_market'], snapshot)
