"""Option trade candidate selection and quality gates."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass(slots=True)
class DataQualityResult:
    allowed: bool
    score: float
    reasons: list[str]


@dataclass(slots=True)
class TradeCandidate:
    symbol: str
    side: str
    score: float
    reasons: list[str]
    spread_pct: float | None
    tick_age_s: float | None
    premium: float | None
    atm_distance: int | None
    data_quality_score: float | None
    entry_price: float | None = None
    stop_loss: float | None = None
    target: float | None = None
    rr: float | None = None
    liquidity_score: float | None = None
    microstructure_score: float | None = None
    final_score: float | None = None


class TradeCandidateSelector:
    def __init__(self, *, quality_mode: str = 'normal', option_strike_window_each_side: int = 2, min_option_premium: float = 80.0, max_option_premium: float = float(os.getenv("MAX_OPTION_PREMIUM", "650"))) -> None:
        self.quality_mode = quality_mode
        self.option_strike_window_each_side = option_strike_window_each_side
        self.min_option_premium = min_option_premium
        self.max_option_premium = max_option_premium

    def _limits(self) -> tuple[float, float, int]:
        if self.quality_mode == 'strict':
            return 0.03, 5.0, 2
        if self.quality_mode == 'loose':
            return 0.07, 15.0, 1
        return 0.05, 10.0, 1

    def select_ranked_candidates(self, *, direction_bias: str, atm_strike: int, snapshots: list[dict[str, Any]]) -> list[TradeCandidate]:
        max_spread, max_age, min_ticks = self._limits()
        ranked: list[TradeCandidate] = []
        for s in snapshots:
            side = str(s.get('side') or s.get('option_type') or '').upper()
            symbol = str(s.get('symbol') or '')
            if side != direction_bias:
                continue
            strike = int(s.get('strike') or 0)
            atm_distance = abs((strike - atm_strike) // 50) if strike and atm_strike else 999
            if atm_distance > self.option_strike_window_each_side:
                continue
            bid, ask, ltp = self._f(s.get('bid')), self._f(s.get('ask')), self._f(s.get('ltp'))
            if ltp is None or bid is None or ask is None or bid <= 0 or ask <= 0:
                continue
            premium = ltp
            if premium < self.min_option_premium or premium > self.max_option_premium:
                continue
            mid = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid if mid > 0 else 1.0
            if spread_pct > max_spread:
                continue
            tick_age_s = self._f(s.get('tick_age_s'))
            if tick_age_s is None or tick_age_s > max_age:
                continue
            real_ticks = int(s.get('real_ticks_last_60s') or 0)
            if real_ticks < min_ticks:
                continue
            entry = ask if ask > 0 else ltp
            atr = self._f(s.get('atr_option')) or max(entry * 0.012, (ask - bid) * 1.5, 1.0)
            risk = max(atr * 0.8, entry * 0.08, 5.0)
            risk = min(18.0, max(4.0, risk))
            sl = entry - risk
            if sl <= 0:
                continue
            target = entry + max(1.6 * risk, atr * 1.2)
            rr = (target - entry) / (entry - sl)
            if rr < 1.5:
                continue
            liquidity = max(0.0, 10.0 - spread_pct * 100.0)
            micro = min(10.0, real_ticks * 3.0)
            score = 6.0 + liquidity * 0.2 + micro * 0.2 - atm_distance * 0.5
            ranked.append(TradeCandidate(symbol=symbol, side=side, score=score, reasons=['candidate_valid'], spread_pct=spread_pct, tick_age_s=tick_age_s, premium=premium, atm_distance=atm_distance, data_quality_score=10.0, entry_price=entry, stop_loss=sl, target=target, rr=rr, liquidity_score=liquidity, microstructure_score=micro, final_score=score))
        return sorted(ranked, key=lambda c: c.final_score or 0.0, reverse=True)

    def select_best_candidate(self, *, underlying: str, direction_bias: str, atm_strike: int, snapshots: list[dict[str, Any]]) -> TradeCandidate | None:
        ranked = self.select_ranked_candidates(direction_bias=direction_bias, atm_strike=atm_strike, snapshots=snapshots)
        return ranked[0] if ranked else None

    @staticmethod
    def _f(v: Any) -> float | None:
        try:
            return None if v is None else float(v)
        except Exception:
            return None


__all__ = ['DataQualityResult', 'TradeCandidate', 'TradeCandidateSelector']
