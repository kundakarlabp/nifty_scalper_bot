from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import ORBProStrategyConfig
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class ORBProStrategy(EliteStrategy):
    """Opening range breakout strategy returning scored vote metadata."""

    MIN_BARS_REQUIRED = 5

    def __init__(self, config: ORBProStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config
        # One breakout trade per direction per day: {(iso_date, 'CE'|'PE')}
        self._fired_today: set[tuple[str, str]] = set()

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: indicators set. Raises: Exception."""
        return {'orb_high', 'orb_low', 'orb_ready', 'close', 'open', 'volume', 'avg_volume', 'atr', 'direction_bias', 'regime'}

    def _evaluate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any | None = None) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        del position
        if str(os.getenv('ENABLE_ORB_STRATEGY', 'false')).strip().lower() not in {'1','true','yes','on'}:
            return None
        try:
            self._no_vote("stale_or_invalid_data")
            or_complete = bool(indicators.get('orb_ready'))
            orb_high = float(indicators.get('orb_high') or 0.0)
            orb_low = float(indicators.get('orb_low') or 0.0)
            close = float(indicators.get('close') or current_price)
            open_price = float(indicators.get('open') or current_price)
            atr = max(float(indicators.get('atr') or 0.0), current_price * 0.01, 1.0)
            vol = float(indicators.get('volume') or 0.0)
            avg_vol = float(indicators.get('avg_volume') or 0.0)
            direction = str(indicators.get('direction_bias') or '').upper()
            regime = str(indicators.get('regime') or '').upper()

            if not or_complete:
                self._no_vote('orb_not_ready')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=ORBPro reason=opening_range_incomplete')
                return None
            if orb_high <= orb_low:
                self._no_vote('invalid_orb_levels')
                return None
            if regime in {'CHOPPY'}:
                self._no_vote('choppy_regime')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=ORBPro reason=choppy_regime')
                return None

            breakout_side = 'CE' if close > orb_high else 'PE' if close < orb_low else 'UNKNOWN'
            if breakout_side == 'UNKNOWN':
                self._no_vote('no_breakout')
                return None

            day_key = (datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat(), breakout_side)
            if day_key in self._fired_today:
                self._no_vote('direction_already_traded_today')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=ORBPro reason=direction_already_traded_today side=%s', breakout_side)
                return None

            candle_range = max(abs(float(indicators.get('high') or close) - float(indicators.get('low') or close)), 1e-9)
            breakout_body_pct = abs(close - open_price) / candle_range
            if breakout_body_pct < 0.35:
                self._no_vote('wick_only_breakout')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=ORBPro reason=wick_only_breakout')
                return None

            retest_confirmed = bool(indicators.get('retest_confirmed'))
            if not retest_confirmed:
                retest_confirmed = abs(close - (orb_high if breakout_side == 'CE' else orb_low)) <= 0.35 * atr

            score = 1.0 + 2.0
            reasons = ['opening_range_complete', 'breakout_close_beyond_range']
            if retest_confirmed:
                score += 2.0
                reasons.append('retest_hold')
            if direction in {'CE', 'PE'} and direction == breakout_side:
                score += 2.0
                reasons.append('direction_alignment')
            vol_tick = avg_vol > 0 and vol >= avg_vol
            if vol_tick:
                score += 1.0
                reasons.append('volume_or_tick_confirmation')
            # Real RR from the actual stop (range invalidation) and target (2*ATR)
            stop_level = orb_low if breakout_side == 'CE' else orb_high
            risk = abs(close - stop_level)
            rr = (2.0 * atr) / risk if risk > 1e-9 else 0.0
            if rr >= 1.8:
                score += 2.0
                reasons.append(f'rr_strong_{rr:.1f}')
            elif rr >= 1.2:
                score += 1.0
                reasons.append(f'rr_acceptable_{rr:.1f}')
            else:
                reasons.append(f'rr_thin_{rr:.1f}')

            strategy_score = max(0.0, min(10.0, score))
            metadata = {
                'strategy': 'ORBPro',
                'side': breakout_side,
                'direction_bias': breakout_side,
                'strategy_score': strategy_score,
                'score_reasons': reasons,
                'setup_type': 'opening_range_breakout',
                'setup_quality': strategy_score,
                'required_data_present': True,
                'stale_data_used': bool(indicators.get('stale_data_used')),
                'candidate_symbol': symbol,
                'rejection_reasons': [],
                'opening_range_high': orb_high,
                'opening_range_low': orb_low,
                'opening_range_complete': or_complete,
                'breakout_side': breakout_side,
                'retest_confirmed': retest_confirmed,
                'breakout_body_pct': round(breakout_body_pct, 3),
                'volume_or_tick_confirmation': vol_tick,
                'invalidation_level': orb_low if breakout_side == 'CE' else orb_high,
            }
            LOGGER.info('STRATEGY_VOTE strategy=ORBPro side=%s score=%.2f', breakout_side, strategy_score)
            self._fired_today.add(day_key)
            if len(self._fired_today) > 8:
                today = day_key[0]
                self._fired_today = {k for k in self._fired_today if k[0] == today}
            return EliteSignal(
                symbol=symbol,
                signal='BUY',
                confidence=max(0.1, min(0.9, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=float(metadata['invalidation_level']),
                target=current_price + (2.0 * atr),
                quantity=self._cfg.quantity or 1,
                strategy_name='ORBPro',
                metadata=metadata,
            )
        except Exception as e:
            LOGGER.error('Failure in ORBProStrategy._evaluate_signal: %s', e, exc_info=e)
            return None


__all__ = ['ORBProStrategy']
