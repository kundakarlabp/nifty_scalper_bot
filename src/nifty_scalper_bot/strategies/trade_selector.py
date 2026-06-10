"""Option trade candidate selection and quality gates."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any

from nifty_scalper_bot.config.env_utils import parse_int_env
from nifty_scalper_bot.risk.cost_model import passes_cost_edge_gate
from nifty_scalper_bot.risk.expiry_gate import expiry_theta_block
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
    def __init__(self, *, quality_mode: str = 'normal', option_strike_window_each_side: int = 2, min_option_premium: float | None = None, max_option_premium: float | None = None, max_tick_age_s: float | None = None, max_option_spread_pct: float | None = None, require_real_ticks_last_60s: int | None = None) -> None:
        self.quality_mode = quality_mode
        self.option_strike_window_each_side = option_strike_window_each_side
        self.min_option_premium = float(min_option_premium if min_option_premium is not None else os.getenv('MIN_OPTION_PREMIUM', '40'))
        self.max_option_premium = float(max_option_premium if max_option_premium is not None else os.getenv('MAX_OPTION_PREMIUM', '650'))
        if self.min_option_premium <= 0:
            raise ValueError('min_option_premium must be > 0')
        if self.max_option_premium <= self.min_option_premium:
            raise ValueError('max_option_premium must be > min_option_premium')
        self.max_tick_age_s = max_tick_age_s
        self.max_option_spread_pct = max_option_spread_pct
        self.require_real_ticks_last_60s = require_real_ticks_last_60s
        self._last_rejects: dict[str, int] = {}
        self._candidate_reject_log_throttle_seconds = max(
            1.0, float(os.getenv("CANDIDATE_REJECT_LOG_THROTTLE_SECONDS", "15") or 15)
        )

    def _log_reject(self, reason: str, symbol: str, *, throttle_key_parts: tuple[Any, ...], **fields: Any) -> None:
        key = "CANDIDATE_REJECTED:" + ":".join(str(part) for part in throttle_key_parts)
        log_once_or_throttled(
            LOGGER,
            key,
            "CANDIDATE_REJECTED symbol=%s reason=%s fields=%s",
            symbol,
            reason,
            fields,
            interval_sec=self._candidate_reject_log_throttle_seconds,
            level=logging.INFO,
            extra={"event": "CANDIDATE_REJECTED", "symbol": symbol, "reason": reason, **fields},
        )

    def _limits(self) -> tuple[float, float, int]:
        if self.quality_mode == 'strict':
            return 3.0, 5.0, 2
        if self.quality_mode == 'loose':
            return 7.0, 15.0, 1
        return 5.0, 10.0, 1

    def select_ranked_candidates(self, *, direction_bias: str, atm_strike: int, snapshots: list[dict[str, Any]]) -> list[TradeCandidate]:
        blocked, gate_reason = expiry_theta_block()
        if blocked:
            self._last_rejects = {'expiry_theta_cutoff': len(snapshots)}
            log_once_or_throttled(LOGGER, f'expiry_theta_cutoff:{direction_bias}', f'CANDIDATE_SELECTION_BLOCKED reason=expiry_theta_cutoff detail={gate_reason} direction={direction_bias} total={len(snapshots)}', interval_sec=60.0, level=logging.INFO, extra={'event': 'CANDIDATE_SELECTION_BLOCKED', 'reason': 'expiry_theta_cutoff', 'detail': gate_reason, 'direction': direction_bias, 'total': len(snapshots)})
            return []
        max_spread, max_age, min_ticks = self._limits()
        if self.max_option_spread_pct is not None:
            max_spread = float(self.max_option_spread_pct)
        if self.max_tick_age_s is not None:
            max_age = float(self.max_tick_age_s)
        if self.require_real_ticks_last_60s is not None:
            min_ticks = int(self.require_real_ticks_last_60s)
        allow_ltp_only = os.getenv('ALLOW_LTP_ONLY_CANDIDATE', 'false').lower() in {'1', 'true', 'yes', 'on'}
        ranked: list[TradeCandidate] = []
        rejects = {'side_mismatch': 0, 'atm_distance': 0, 'missing_bid_ask': 0, 'premium_out_of_range': 0, 'spread_too_wide': 0, 'tick_stale': 0, 'insufficient_ticks': 0, 'invalid_rr': 0, 'cost_edge_insufficient': 0}
        ltp_only_used = 0
        for s in snapshots:
            side = str(s.get('side') or s.get('option_type') or '').upper()
            symbol = str(s.get('symbol') or '')
            if side != direction_bias:
                rejects['side_mismatch'] += 1
                self._log_reject("side_mismatch", symbol, throttle_key_parts=("side_mismatch", symbol, side, direction_bias), side=side, expected_direction=direction_bias)
                continue
            strike = int(s.get('strike') or 0)
            atm_distance = abs((strike - atm_strike) // 50) if strike and atm_strike else 999
            if atm_distance > self.option_strike_window_each_side:
                rejects['atm_distance'] += 1
                self._log_reject("atm_distance", symbol, throttle_key_parts=("atm_distance", symbol, strike, atm_strike), strike=strike, atm_strike=atm_strike, atm_distance=atm_distance, allowed_window=self.option_strike_window_each_side)
                continue
            bid, ask, ltp = self._f(s.get('bid')), self._f(s.get('ask')), self._f(s.get('ltp'))
            has_bid_ask = bool((bid or 0) > 0 and (ask or 0) > 0)
            if ltp is None or ltp <= 0:
                rejects['missing_bid_ask'] += 1
                self._log_reject("missing_bid_ask", symbol, throttle_key_parts=("missing_bid_ask", symbol, "invalid_ltp"), ltp=ltp, bid=bid, ask=ask, quote_quality=s.get("quote_quality"), ltp_only_fallback=bool(s.get("ltp_only_fallback")), allow_ltp_only=allow_ltp_only)
                continue
            premium = ltp
            if premium < self.min_option_premium or premium > self.max_option_premium:
                rejects['premium_out_of_range'] += 1
                LOGGER.info(
                    "CANDIDATE_REJECTED symbol=%s reason=premium_out_of_range premium=%s min=%s max=%s ltp=%s bid=%s ask=%s strike=%s atm=%s atm_distance=%s",
                    symbol,
                    premium,
                    self.min_option_premium,
                    self.max_option_premium,
                    ltp,
                    bid,
                    ask,
                    strike,
                    atm_strike,
                    atm_distance,
                    extra={
                        "event": "CANDIDATE_REJECTED",
                        "symbol": symbol,
                        "reason": "premium_out_of_range",
                        "premium": premium,
                        "min_option_premium": self.min_option_premium,
                        "max_option_premium": self.max_option_premium,
                        "ltp": ltp,
                        "bid": bid,
                        "ask": ask,
                        "strike": strike,
                        "atm": atm_strike,
                        "atm_distance": atm_distance,
                    },
                )
                continue
            tick_age_s = self._f(s.get('tick_age_s'))
            if tick_age_s is None or tick_age_s > max_age:
                rejects['tick_stale'] += 1
                self._log_reject("tick_stale", symbol, throttle_key_parts=("tick_stale", symbol, int(max_age)), tick_age_s=tick_age_s, max_age_s=max_age)
                continue
            real_ticks = int(s.get('real_ticks_last_60s') or 0)
            if real_ticks < min_ticks:
                rejects['insufficient_ticks'] += 1
                self._log_reject("insufficient_ticks", symbol, throttle_key_parts=("insufficient_ticks", symbol, min_ticks), real_ticks_last_60s=real_ticks, min_ticks=min_ticks)
                continue
            reasons = ['candidate_valid']
            spread_pct: float | None = None
            if has_bid_ask:
                mid = ((bid or 0.0) + (ask or 0.0)) / 2.0
                spread_pct = (((ask or 0.0) - (bid or 0.0)) / mid * 100.0) if mid > 0 else 100.0
                if spread_pct > max_spread:
                    rejects['spread_too_wide'] += 1
                    self._log_reject("spread_too_wide", symbol, throttle_key_parts=("spread_too_wide", symbol, int(max_spread * 100)), bid=bid, ask=ask, spread_pct=spread_pct, max_spread_pct=max_spread)
                    continue
                entry = ask if (ask or 0.0) > 0 else ltp
                score_penalty = 0.0
            else:
                ltp_only_flag = bool(s.get('ltp_only_fallback')) or (
                    str(s.get('quote_quality') or '').lower() == 'ltp_only'
                )
                if not (allow_ltp_only and ltp_only_flag):
                    rejects['missing_bid_ask'] += 1
                    self._log_reject("missing_bid_ask", symbol, throttle_key_parts=("missing_bid_ask", symbol, "no_bidask"), ltp=ltp, bid=bid, ask=ask, quote_quality=s.get("quote_quality"), ltp_only_fallback=ltp_only_flag, allow_ltp_only=allow_ltp_only)
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
                self._log_reject("invalid_rr", symbol, throttle_key_parts=("invalid_rr", symbol), entry=entry, stop_loss=sl, target=target, rr=rr, min_rr=1.5)
                continue
            half_spread = (((ask or 0.0) - (bid or 0.0)) / 2.0) if has_bid_ask else entry * 0.003
            lot_size = parse_int_env(os.getenv('NIFTY_LOT_SIZE'), 65)
            cost_ok, edge_multiple, cost = passes_cost_edge_gate(entry_price=entry, target_price=target, quantity=lot_size, half_spread=max(0.0, half_spread))
            if not cost_ok:
                rejects['cost_edge_insufficient'] += 1
                self._log_reject("cost_edge_insufficient", symbol, throttle_key_parts=("cost_edge_insufficient", symbol), entry=entry, target=target, edge_multiple=round(edge_multiple, 2), round_trip_cost=round(cost.total, 2), cost_per_unit=round(cost.cost_per_unit, 3))
                continue
            reasons.append(f'cost_edge_{edge_multiple:.1f}x')
            liquidity = 5.0 if spread_pct is None else max(0.0, 10.0 - spread_pct * 100.0)
            micro = min(10.0, real_ticks * 3.0)
            score = 6.0 + liquidity * 0.2 + micro * 0.2 - atm_distance * 0.5 - score_penalty
            dq = self.evaluate_data_quality(s)
            final = max(0.0, min(10.0, 0.7 * score + 0.3 * dq.score))
            ranked.append(TradeCandidate(symbol=symbol, side=side, score=final, reasons=reasons, spread_pct=spread_pct, tick_age_s=tick_age_s, premium=premium, atm_distance=atm_distance, data_quality_score=dq.score, entry_price=entry, stop_loss=sl, target=target, rr=rr, liquidity_score=liquidity, microstructure_score=micro, final_score=final))
        sorted_ranked = sorted(ranked, key=lambda c: c.final_score or 0.0, reverse=True)
        self._last_rejects = dict(rejects)
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


    def evaluate_data_quality(self, snapshot: dict[str, Any]) -> DataQualityResult:
        """Args: snapshot. Returns: data quality result. Raises: none."""
        max_spread, max_age, min_ticks = self._limits()
        reasons: list[str] = []
        score = 10.0
        tick_age = self._f(snapshot.get('tick_age_s'))
        if tick_age is None or tick_age > max_age:
            reasons.append('tick_stale')
            score -= 4.0
        bid, ask = self._f(snapshot.get('bid')), self._f(snapshot.get('ask'))
        if (bid or 0) <= 0 or (ask or 0) <= 0:
            reasons.append('missing_bid_ask')
            score -= 3.0
        else:
            mid = ((bid or 0.0) + (ask or 0.0)) / 2.0
            spread_pct = (((ask or 0.0) - (bid or 0.0)) / mid * 100.0) if mid > 0 else 100.0
            if spread_pct > max_spread:
                reasons.append('spread_too_wide')
                score -= 3.0
        real_ticks = int(snapshot.get('real_ticks_last_60s') or 0)
        if real_ticks < min_ticks:
            reasons.append('insufficient_ticks')
            score -= 2.0
        return DataQualityResult(allowed=not reasons, score=max(0.0, score), reasons=reasons or ['data_quality_ok'])

    @staticmethod
    def _f(v: Any) -> float | None:
        try:
            return None if v is None else float(v)
        except Exception:
            return None


__all__ = ['DataQualityResult', 'TradeCandidate', 'TradeCandidateSelector']
