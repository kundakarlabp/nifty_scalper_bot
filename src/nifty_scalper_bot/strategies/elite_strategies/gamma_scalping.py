from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import GammaScalpingStrategyConfig
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class GammaScalpingStrategy(EliteStrategy):
    """Expiry-gamma strategy gated by explicit mode flags."""

    MIN_BARS_REQUIRED = 3

    def __init__(self, config: GammaScalpingStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: indicators set. Raises: Exception."""
        return {'gamma', 'theta', 'atr', 'direction_bias', 'volatility_expansion_confirmed'}

    def _evaluate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any | None = None) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        del position
        try:
            strategy_mode = str(os.getenv('STRATEGY_MODE', 'directional_scalp')).lower()
            gamma_enabled = str(os.getenv('ALLOW_EXPIRY_GAMMA_STRATEGIES', 'false')).lower() in {'1', 'true', 'yes', 'on'}
            if not (strategy_mode == 'expiry_gamma' and gamma_enabled):
                LOGGER.debug('STRATEGY_NO_VOTE strategy=GammaScalping reason=gamma_mode_disabled')
                return None

            gamma = float(indicators.get('gamma') or 0.0)
            theta = float(indicators.get('theta') or 0.0)
            atr = max(float(indicators.get('atr') or 0.0), current_price * 0.01, 1.0)
            direction = str(indicators.get('direction_bias') or '').upper()
            if gamma <= float(getattr(self._cfg, 'min_gamma', 0.0005)):
                return None
            if theta >= 0:
                return None

            side = direction if direction in {'CE', 'PE'} else 'CE'
            premium_decay_risk = min(1.0, abs(theta) / max(current_price, 1.0))
            vol_exp = bool(indicators.get('volatility_expansion_confirmed') or gamma > 0.0015)

            score = 2.0 + (2.0 if vol_exp else 0.0)
            reasons = ['gamma_enabled_mode']
            if vol_exp:
                reasons.append('volatility_expansion_confirmed')
            if direction in {'CE', 'PE'}:
                score += 2.0
                reasons.append('direction_context')
            strategy_score = max(0.0, min(6.0, score))

            today = datetime.utcnow().date()
            metadata = {
                'strategy': 'GammaScalping',
                'side': side,
                'direction_bias': side,
                'strategy_score': strategy_score,
                'setup_quality': strategy_score,
                'setup_type': 'expiry_gamma',
                'required_data_present': True,
                'stale_data_used': bool(indicators.get('stale_data_used')),
                'candidate_symbol': symbol,
                'score_reasons': reasons,
                'rejection_reasons': [],
                'expiry_day': today.weekday() in {1, 3},
                'days_to_expiry': indicators.get('days_to_expiry'),
                'gamma_mode_enabled': True,
                'premium_decay_risk': round(premium_decay_risk, 4),
                'volatility_expansion_confirmed': vol_exp,
                'invalidation_level': current_price - atr if side == 'CE' else current_price + atr,
            }
            LOGGER.info('STRATEGY_VOTE strategy=GammaScalping side=%s score=%.2f', side, strategy_score)
            return EliteSignal(
                symbol=symbol,
                signal='BUY',
                confidence=max(0.1, min(0.75, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=float(metadata['invalidation_level']),
                target=current_price + (1.8 * atr),
                quantity=self._cfg.quantity or 1,
                strategy_name='GammaScalping',
                metadata=metadata,
            )
        except Exception as e:
            LOGGER.error('Failure in GammaScalpingStrategy._evaluate_signal: %s', e, exc_info=e)
            return None


__all__ = ['GammaScalpingStrategy']
