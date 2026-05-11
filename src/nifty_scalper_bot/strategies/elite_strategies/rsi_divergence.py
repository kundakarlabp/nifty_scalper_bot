from __future__ import annotations

from collections import deque
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import RSIDivergenceStrategyConfig
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class RSIDivergenceStrategy(EliteStrategy):
    """RSI divergence strategy with structure confirmation and regime-aware scoring."""

    MIN_BARS_REQUIRED = 20

    def __init__(self, config: RSIDivergenceStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config
        self._history: dict[str, deque[tuple[float, float]]] = {}

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: required indicators. Raises: Exception."""
        return {'rsi', 'close', 'atr', 'regime', 'direction_bias', 'confirmation_candle'}

    def _evaluate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any | None = None) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        del position
        try:
            self._no_vote("stale_or_invalid_data")
            rsi = float(indicators.get('rsi') or 50.0)
            close = float(indicators.get('close') or current_price)
            atr = max(float(indicators.get('atr') or 0.0), current_price * 0.01, 1.0)
            regime = str(indicators.get('regime') or 'UNKNOWN').upper()
            direction = str(indicators.get('direction_bias') or '').upper()
            hist = self._history.setdefault(symbol, deque(maxlen=10))
            hist.append((close, rsi))
            if len(hist) < 6:
                self._no_vote('insufficient_history')
                return None

            p1, r1 = hist[-4]
            p2, r2 = hist[-1]
            bullish_div = p2 < p1 and r2 > r1
            bearish_div = p2 > p1 and r2 < r1
            if not bullish_div and not bearish_div:
                self._no_vote('no_divergence')
                return None

            side = 'CE' if bullish_div else 'PE'
            confirmation = bool(indicators.get('confirmation_candle'))
            if not confirmation:
                confirmation = (bullish_div and close > hist[-2][0]) or (bearish_div and close < hist[-2][0])
            if not confirmation:
                self._no_vote('no_confirmation')
                LOGGER.debug('STRATEGY_NO_VOTE strategy=RSIDivergence reason=no_structure_confirmation')
                return None

            score = 2.0 + 2.0
            reasons = ['valid_divergence', 'structure_confirmation']
            if regime in {'RANGE', 'LOW_VOLATILITY'}:
                score += 2.0
                reasons.append('regime_support')
            elif regime in {'TREND_UP', 'TREND_DOWN'}:
                score -= 1.5
                reasons.append('strong_trend_penalty')
            if direction in {'CE', 'PE'} and direction == side:
                score += 1.0
                reasons.append('direction_context')
            score += 2.0
            reasons.append('rr_viable')

            strategy_score = max(0.0, min(10.0, score))
            if strategy_score < 3.5:
                self._no_vote('low_score')
                return None
            source_symbol = str(indicators.get('source_symbol') or '').strip()
            source_domain = 'underlying_price' if source_symbol else 'option_premium'
            metadata = {
                'strategy': 'RSIDivergence',
                'strategy_name': 'RSIDivergence',
                'role': 'trigger',
                'source_domain': source_domain,
                'signal_family': 'directional_trigger',
                'trade_side': side,
                'side': side,
                'direction_bias': side,
                'preliminary_only': True,
                'requires_runner_final_score': True,
                'direction_score': strategy_score,
                'strategy_score': strategy_score,
                'data_score': 8.0,
                'setup_quality': strategy_score,
                'setup_type': 'rsi_reversal',
                'required_data_present': True,
                'stale_data_used': bool(indicators.get('stale_data_used')),
                'candidate_symbol': symbol,
                'score_reasons': reasons,
                'rejection_reasons': [],
                'divergence_type': 'bullish' if bullish_div else 'bearish',
                'swing_points': [(p1, r1), (p2, r2)],
                'confirmation_candle': confirmation,
                'trend_regime': regime,
                'reversal_quality': round(strategy_score / 10.0, 3),
                'underlying_invalidation_level': p2 - atr if side == 'CE' else p2 + atr,
                'premium_stop_distance': max(0.9 * atr, current_price * 0.02, 1.0),
                'premium_target_rr': 2.0,
            }
            LOGGER.info('STRATEGY_VOTE strategy=RSIDivergence side=%s score=%.2f', side, strategy_score)
            return EliteSignal(
                symbol=symbol,
                signal='BUY',
                confidence=max(0.1, min(0.82, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=None,
                target=None,
                quantity=self._cfg.quantity or 1,
                strategy_name='RSIDivergence',
                metadata=metadata,
            )
        except Exception as e:
            LOGGER.error('Failure in RSIDivergenceStrategy._evaluate_signal: %s', e, exc_info=e)
            return None


__all__ = ['RSIDivergenceStrategy']
