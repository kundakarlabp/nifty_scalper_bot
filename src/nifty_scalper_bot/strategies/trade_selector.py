"""Option trade candidate selection and quality gates."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any

from nifty_scalper_bot.utils.logging import get_logger, log_throttled

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class DataQualityResult:
    """Args: quality attributes. Returns: quality verdict. Raises: none."""

    allowed: bool
    score: float
    reasons: list[str]


@dataclass(slots=True)
class TradeCandidate:
    """Args: candidate attributes. Returns: candidate descriptor. Raises: none."""

    symbol: str
    side: str
    score: float
    reasons: list[str]
    spread_pct: float | None
    tick_age_s: float | None
    premium: float | None
    atm_distance: int | None
    data_quality_score: float | None


class TradeCandidateSelector:
    """Args: optional threshold overrides. Returns: best option candidate. Raises: none."""

    def __init__(
        self,
        *,
        option_strike_window_each_side: int = 2,
        max_option_spread_pct: float = 0.05,
        min_option_premium: float = 20.0,
        max_option_premium: float = 400.0,
        max_tick_age_s: float | None = None,
        min_real_ticks_last_60s: int = 1,
        require_real_ticks_last_60s: bool | None = None,
    ) -> None:
        self.option_strike_window_each_side = max(1, int(option_strike_window_each_side))
        self.max_option_spread_pct = max(0.0, float(max_option_spread_pct))
        self.min_option_premium = max(0.0, float(min_option_premium))
        self.max_option_premium = max(self.min_option_premium, float(max_option_premium))
        configured_tick_age = (
            float(os.getenv('MAX_OPTION_TICK_AGE_SECONDS', '10'))
            if max_tick_age_s is None
            else float(max_tick_age_s)
        )
        self.max_tick_age_s = max(0.0, configured_tick_age)
        self.min_real_ticks_last_60s = max(0, int(min_real_ticks_last_60s))
        configured_require_real_ticks = (
            os.getenv('REQUIRE_REAL_TICKS_LAST_60S', 'false').strip().lower()
            in {'1', 'true', 'yes', 'on'}
        )
        if require_real_ticks_last_60s is None:
            self.require_real_ticks_last_60s = configured_require_real_ticks
        else:
            self.require_real_ticks_last_60s = bool(require_real_ticks_last_60s)

    def evaluate_data_quality(self, snapshot: dict[str, Any]) -> DataQualityResult:
        """Args: snapshot dict. Returns: DataQualityResult. Raises: none."""
        reasons: list[str] = []
        score = 10.0

        if bool(snapshot.get('latest_candle_provisional')):
            reasons.append('latest_candle_provisional')
        if bool(snapshot.get('latest_candle_synthetic')):
            reasons.append('latest_candle_synthetic')

        tick_age_s = self._to_float(snapshot.get('tick_age_s'))
        if tick_age_s is None or tick_age_s > self.max_tick_age_s:
            reasons.append('stale_tick')

        if not bool(snapshot.get('ohlc_valid', True)):
            reasons.append('invalid_ohlc')
        if bool(snapshot.get('ltp_jump_absurd', False)):
            reasons.append('absurd_ltp_jump')

        real_ticks_60s = int(snapshot.get('real_ticks_last_60s') or 0)
        if real_ticks_60s < self.min_real_ticks_last_60s:
            reasons.append('no_recent_real_tick')
            if self.require_real_ticks_last_60s:
                score -= 2.0
            else:
                score -= 0.75

        bid = self._to_float(snapshot.get('bid'))
        ask = self._to_float(snapshot.get('ask'))
        mid = self._to_float(snapshot.get('mid'))
        if bid is None or ask is None or bid <= 0 or ask <= 0 or mid is None or mid <= 0:
            reasons.append('invalid_bid_ask')
        else:
            spread_pct = max(0.0, (ask - bid) / mid)
            if spread_pct > self.max_option_spread_pct:
                reasons.append('spread_too_wide')

        volume = self._to_float(snapshot.get('latest_candle_volume'))
        if volume is not None and volume <= 0:
            score -= 1.0
            reasons.append('zero_candle_volume_soft')

        hard_blocks = {
            'latest_candle_provisional',
            'latest_candle_synthetic',
            'stale_tick',
            'invalid_ohlc',
            'absurd_ltp_jump',
            'invalid_bid_ask',
            'spread_too_wide',
        }
        if self.require_real_ticks_last_60s:
            hard_blocks.add('no_recent_real_tick')
        allowed = len(hard_blocks.intersection(reasons)) == 0
        score = max(0.0, min(10.0, score - 1.0 * len(hard_blocks.intersection(reasons))))
        LOGGER.debug(
            'DATA_QUALITY_CHECK symbol=%s allowed=%s score=%.2f reasons=%s',
            snapshot.get('symbol'),
            allowed,
            score,
            reasons,
            extra={'event': 'DATA_QUALITY_CHECK', 'allowed': allowed, 'score': score, 'reasons': reasons},
        )
        if not allowed:
            log_throttled(
                LOGGER,
                key=f"data_quality_blocked_{snapshot.get('symbol')}",
                msg=f"DATA_QUALITY_BLOCKED symbol={snapshot.get('symbol')} reasons={reasons}",
                level=logging.DEBUG,
                interval_sec=60.0,
                extra={'event': 'DATA_QUALITY_BLOCKED', 'reasons': reasons},
            )
        return DataQualityResult(allowed=allowed, score=score, reasons=reasons)

    def select_best_candidate(
        self,
        *,
        underlying: str,
        direction_bias: str,
        atm_strike: int,
        snapshots: list[dict[str, Any]],
    ) -> TradeCandidate | None:
        """Args: market snapshots. Returns: best candidate or None. Raises: none."""
        target_suffix = 'CE' if direction_bias == 'CE' else 'PE'
        best: TradeCandidate | None = None

        for snapshot in snapshots:
            symbol = str(snapshot.get('symbol') or '')
            if not symbol.endswith(target_suffix):
                self._log_rejection(symbol, 'direction_mismatch')
                continue

            strike = int(snapshot.get('strike') or 0)
            atm_distance = abs((strike - atm_strike) // 50) if strike and atm_strike else None
            if atm_distance is None or atm_distance > self.option_strike_window_each_side:
                self._log_rejection(symbol, 'far_otm')
                continue

            quality = self.evaluate_data_quality(snapshot)
            if not quality.allowed:
                self._log_rejection(symbol, ','.join(quality.reasons))
                continue

            premium = self._to_float(snapshot.get('ltp'))
            if premium is None or premium < self.min_option_premium or premium > self.max_option_premium:
                self._log_rejection(symbol, 'premium_out_of_range')
                continue

            bid = self._to_float(snapshot.get('bid'))
            ask = self._to_float(snapshot.get('ask'))
            mid = self._to_float(snapshot.get('mid'))
            if bid is None or ask is None or mid is None or bid <= 0 or ask <= 0 or mid <= 0:
                self._log_rejection(symbol, 'invalid_bid_ask')
                continue
            spread_pct = max(0.0, (ask - bid) / mid)
            if spread_pct > self.max_option_spread_pct:
                self._log_rejection(symbol, 'spread_too_wide')
                continue

            tick_age_s = self._to_float(snapshot.get('tick_age_s'))
            score = quality.score - spread_pct * 10.0 - (float(atm_distance or 0) * 0.4)
            candidate = TradeCandidate(
                symbol=symbol,
                side='BUY',
                score=round(score, 3),
                reasons=['candidate_valid'],
                spread_pct=spread_pct,
                tick_age_s=tick_age_s,
                premium=premium,
                atm_distance=atm_distance,
                data_quality_score=quality.score,
            )
            if best is None or candidate.score > best.score:
                best = candidate

        if best:
            LOGGER.info(
                'CANDIDATE_SELECTED symbol=%s score=%.2f spread_pct=%s tick_age_s=%s',
                best.symbol,
                best.score,
                best.spread_pct,
                best.tick_age_s,
                extra={'event': 'CANDIDATE_SELECTED', 'symbol': best.symbol, 'score': best.score},
            )
        return best

    def _log_rejection(self, symbol: str, reason: str) -> None:
        """Args: symbol and reason. Returns: None. Raises: none."""
        log_throttled(
            LOGGER,
            key=f'candidate_rejected_{symbol}_{reason}',
            msg=f'CANDIDATE_REJECTED symbol={symbol} reason={reason}',
            interval_sec=60.0,
            level=logging.DEBUG,
            extra={'event': 'CANDIDATE_REJECTED', 'symbol': symbol, 'reason': reason},
        )

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Args: arbitrary value. Returns: float value or None. Raises: none."""
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None


__all__ = ['DataQualityResult', 'TradeCandidate', 'TradeCandidateSelector']
