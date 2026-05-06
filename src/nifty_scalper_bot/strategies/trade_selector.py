"""Option trade candidate selection and quality gates."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any

from nifty_scalper_bot.utils.logging import get_logger, log_once_or_throttled

LOGGER = get_logger(__name__)


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
    def __init__(self, *, quality_mode: str = 'normal', option_strike_window_each_side: int = 2, min_option_premium: float = 80.0, max_option_premium: float = float(os.getenv('MAX_OPTION_PREMIUM', '650'))) -> None:
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
        allow_ltp_only = os.getenv('ALLOW_LTP_ONLY_CANDIDATE', 'true').lower() in {'1', 'true', 'yes', 'on'}
        ranked: list[TradeCandidate] = []
        rejects = {'side_mismatch': 0, 'atm_distance': 0, 'missing_bid_ask': 0, 'premium_out_of_range': 0, 'spread_too_wide': 0, 'tick_stale': 0, 'insufficient_ticks': 0, 'invalid_rr': 0}
        ltp_only_used = 0
        for s in snapshots:
            side = str(s.get('side') or s.get('option_type') or '').upper()
            symbol = str(s.get('symbol') or '')
            if side != direction_bias:
                rejects['side_mismatch'] += 1
                continue
            strike = int(s.get('strike') or 0)
            atm_distance = abs((strike - atm_strike) // 50) if strike and atm_strike else 999
            if atm_distance > self.option_strike_window_each_side:
                rejects['atm_distance'] += 1
                continue
            bid, ask, ltp = self._f(s.get('bid')), self._f(s.get('ask')), self._f(s.get('ltp'))
            has_bid_ask = bool((bid or 0) > 0 and (ask or 0) > 0)
            if ltp is None or ltp <= 0:
                rejects['missing_bid_ask'] += 1
                continue
            premium = ltp
            if premium < self.min_option_premium or premium > self.max_option_premium:
                rejects['premium_out_of_range'] += 1
                continue
            tick_age_s = self._f(s.get('tick_age_s'))
            if tick_age_s is None or tick_age_s > max_age:
                rejects['tick_stale'] += 1
                continue
            real_ticks = int(s.get('real_ticks_last_60s') or 0)
            if real_ticks < min_ticks:
                rejects['insufficient_ticks'] += 1
                continue
            reasons = ['candidate_valid']
            spread_pct: float | None = None
            if has_bid_ask:
                mid = ((bid or 0.0) + (ask or 0.0)) / 2.0
                spread_pct = ((ask or 0.0) - (bid or 0.0)) / mid if mid > 0 else 1.0
                if spread_pct > max_spread:
                    rejects['spread_too_wide'] += 1
                    continue
                entry = ask if (ask or 0.0) > 0 else ltp
                score_penalty = 0.0
            else:
                ltp_only_flag = bool(s.get('ltp_only_fallback'))
                if not (allow_ltp_only and ltp_only_flag):
                    rejects['missing_bid_ask'] += 1
                    continue
                entry = ltp * 1.003
                score_penalty = 1.5
                ltp_only_used += 1
                reasons.append('ltp_only_fallback')
            atr = self._f(s.get('atr_option')) or max(entry * 0.012, max((ask or 0.0) - (bid or 0.0), 0.0) * 1.5, 1.0)
            risk = max(atr * 0.8, entry * 0.08, 5.0)
            risk = min(18.0, max(4.0, risk))
            sl = entry - risk
            if sl <= 0:
                continue
            target = entry + max(1.6 * risk, atr * 1.2)
            rr = (target - entry) / (entry - sl)
            if rr < 1.5:
                rejects['invalid_rr'] += 1
                continue
            liquidity = 5.0 if spread_pct is None else max(0.0, 10.0 - spread_pct * 100.0)
            micro = min(10.0, real_ticks * 3.0)
            score = 6.0 + liquidity * 0.2 + micro * 0.2 - atm_distance * 0.5 - score_penalty
            ranked.append(TradeCandidate(symbol=symbol, side=side, score=score, reasons=reasons, spread_pct=spread_pct, tick_age_s=tick_age_s, premium=premium, atm_distance=atm_distance, data_quality_score=10.0, entry_price=entry, stop_loss=sl, target=target, rr=rr, liquidity_score=liquidity, microstructure_score=micro, final_score=score))
        sorted_ranked = sorted(ranked, key=lambda c: c.final_score or 0.0, reverse=True)
        key = f'candidate_summary_empty:{direction_bias}:{atm_strike}'
        event_extra = {'event': 'CANDIDATE_SELECTION_SUMMARY', 'direction': direction_bias, 'atm': atm_strike, 'total': len(snapshots), 'ranked': len(sorted_ranked), 'rejects': rejects, 'ltp_only_used': ltp_only_used}
        if sorted_ranked:
            LOGGER.debug('CANDIDATE_SELECTION_SUMMARY direction=%s atm=%s total=%s ranked=%s ltp_only_used=%s rejects=%s', direction_bias, atm_strike, len(snapshots), len(sorted_ranked), ltp_only_used, rejects, extra=event_extra)
        else:
            log_once_or_throttled(LOGGER, key, f'CANDIDATE_SELECTION_SUMMARY direction={direction_bias} atm={atm_strike} total={len(snapshots)} ranked=0 ltp_only_used={ltp_only_used} rejects={rejects}', interval_sec=30.0, level=logging.INFO, extra=event_extra)
        return sorted_ranked

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
