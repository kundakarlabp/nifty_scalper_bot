from __future__ import annotations

import os
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import StraddleThetaStrategyConfig
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class StraddleThetaStrategy(EliteStrategy):
    """Theta strategy emits vote only when STRATEGY_MODE=theta."""

    def __init__(self, config: StraddleThetaStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: required indicators. Raises: Exception."""
        return {'theta', 'iv', 'delta', 'atr'}

    def _evaluate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any | None = None) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        del position
        try:
            if str(os.getenv('STRATEGY_MODE', 'directional_scalp')).lower() != 'theta':
                LOGGER.debug('STRATEGY_NO_VOTE strategy=StraddleTheta reason=theta_mode_required')
                return None

            theta = float(indicators.get('theta') or 0.0)
            iv = float(indicators.get('iv') or 0.0)
            delta = abs(float(indicators.get('delta') or 0.0))
            atr = max(float(indicators.get('atr') or 0.0), current_price * 0.01, 1.0)
            if theta >= -1.0 or iv < float(getattr(self._cfg, 'min_iv', 12.0)) or delta > 0.35:
                return None

            strategy_score = 5.0
            metadata = {
                'strategy': 'StraddleTheta',
                'side': 'NO_TRADE',
                'direction_bias': 'UNKNOWN',
                'strategy_score': strategy_score,
                'setup_quality': strategy_score,
                'setup_type': 'theta_non_directional',
                'required_data_present': True,
                'stale_data_used': bool(indicators.get('stale_data_used')),
                'candidate_symbol': symbol,
                'score_reasons': ['theta_mode_enabled', 'theta_decay_setup'],
                'rejection_reasons': [],
                'invalidation_level': None,
            }
            LOGGER.info('STRATEGY_VOTE strategy=StraddleTheta side=NO_TRADE score=%.2f', strategy_score)
            return EliteSignal(
                symbol=symbol,
                signal='HOLD',
                confidence=0.35,
                entry_price=current_price,
                stop_loss=current_price - atr,
                target=current_price + atr,
                quantity=self._cfg.quantity or 1,
                strategy_name='StraddleTheta',
                metadata=metadata,
            )
        except Exception as e:
            LOGGER.error('Failure in StraddleThetaStrategy._evaluate_signal: %s', e, exc_info=e)
            return None


__all__ = ['StraddleThetaStrategy']
