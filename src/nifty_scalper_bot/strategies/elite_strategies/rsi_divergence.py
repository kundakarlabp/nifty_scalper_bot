from __future__ import annotations

from collections import deque
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import RSIDivergenceStrategyConfig
from nifty_scalper_bot.strategies.signal_quality import resolve_signal_domain
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
            # Noise filter: a divergence needs a real price displacement,
            # not a few-tick wiggle, or every flat patch becomes a "signal".
            if abs(p2 - p1) < 0.25 * atr:
                self._no_vote('price_move_too_small')
                return None
            bullish_div = p2 < p1 and r2 > r1
            bearish_div = p2 > p1 and r2 < r1
            if not bullish_div and not bearish_div:
                self._no_vote('no_divergence')
                return None

            contract_side, option_premium_domain, _ = resolve_signal_domain(symbol, indicators)
            if option_premium_domain:
                if not bullish_div:
                    self._no_vote('premium_no_bullish_divergence')
                    return None
                side = contract_side
            else:
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
            # Grade by RSI displacement: bigger divergence = stronger reversal evidence
            rsi_delta = abs(r2 - r1)
            if rsi_delta >= 5.0:
                score += 2.0
                reasons.append(f'divergence_strong_{rsi_delta:.1f}')
            elif rsi_delta >= 2.5:
                score += 1.0
                reasons.append(f'divergence_moderate_{rsi_delta:.1f}')
            else:
                reasons.append(f'divergence_weak_{rsi_delta:.1f}')

            strategy_score = max(0.0, min(10.0, score))
            if strategy_score < 3.5:
                self._no_vote('low_score')
                return None
            source_symbol = str(indicators.get('source_symbol') or '').strip()
            source_domain = 'underlying_price' if source_symbol else 'option_premium'
            metadata = {
                'strategy': 'RSIDivergence',
                'strategy_name': 'RSIDivergence',
                'role': 'context',
                'source_domain': 'option_premium' if option_premium_domain else source_domain,
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
                'context_score': strategy_score,
                'premium_stop_distance': max(atr * 1.0, current_price * 0.02, 1.0),
                'premium_target_rr': 1.8,
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
