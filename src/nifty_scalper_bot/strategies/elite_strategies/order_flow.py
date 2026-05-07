from __future__ import annotations

from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import OrderFlowStrategyConfig
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class OrderFlowStrategy(EliteStrategy):
    """Order-flow vote using spread, depth imbalance and tick direction."""

    MIN_BARS_REQUIRED = 5

    def __init__(self, config: OrderFlowStrategyConfig, indicator_engine: Any) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: indicator keys. Raises: Exception."""
        return {'bid', 'ask', 'depth', 'tick_direction', 'buy_qty', 'sell_qty', 'direction_bias', 'spread_pct', 'atr'}

    def _evaluate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any | None = None) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        del position
        try:
            self.last_no_vote_reason = "none"
            bid = float(indicators.get('bid') or 0.0)
            ask = float(indicators.get('ask') or 0.0)
            depth = indicators.get('depth') or {}
            tick_direction = str(indicators.get('tick_direction') or '').upper()
            direction = str(indicators.get('direction_bias') or '').upper()
            atr = max(float(indicators.get('atr') or 0.0), current_price * 0.01, 1.0)

            if bid <= 0 or ask <= 0 or ask <= bid:
                self.last_no_vote_reason = 'missing_depth'
                self.last_no_vote_reason = 'weak_imbalance'
                self.last_no_vote_reason = 'missing_depth'
                LOGGER.debug('STRATEGY_NO_VOTE strategy=OrderFlow reason=missing_bid_ask')
                return None
            spread_pct = float(indicators.get('spread_pct') or (((ask - bid) / ((ask + bid) / 2.0)) * 100.0))
            if spread_pct > 28.0:
                self.last_no_vote_reason = 'missing_depth'
                self.last_no_vote_reason = 'weak_imbalance'
                self.last_no_vote_reason = 'missing_depth'
                LOGGER.debug('STRATEGY_NO_VOTE strategy=OrderFlow reason=wide_spread')
                return None

            bids = depth.get('buy', []) if isinstance(depth, dict) else []
            asks = depth.get('sell', []) if isinstance(depth, dict) else []
            depth_available = bool(bids and asks)
            total_bid = sum(float(level.get('quantity', 0.0)) for level in bids[:5]) if depth_available else 0.0
            total_ask = sum(float(level.get('quantity', 0.0)) for level in asks[:5]) if depth_available else 0.0
            if total_bid + total_ask <= 0:
                self.last_no_vote_reason = 'missing_depth'
                self.last_no_vote_reason = 'weak_imbalance'
                self.last_no_vote_reason = 'missing_depth'
                LOGGER.debug('STRATEGY_NO_VOTE strategy=OrderFlow reason=depth_missing')
                return None
            depth_imbalance = (total_bid - total_ask) / max(total_bid + total_ask, 1.0)
            side = 'CE' if depth_imbalance > 0 else 'PE'

            score = 0.0
            reasons: list[str] = []
            if spread_pct <= 12.0:
                score += 2.0
                reasons.append('tight_spread')
            if abs(depth_imbalance) >= 0.15:
                score += 2.0
                reasons.append('depth_imbalance')
            if (side == 'CE' and tick_direction in {'UP', 'BUY'}) or (side == 'PE' and tick_direction in {'DOWN', 'SELL'}):
                score += 2.0
                reasons.append('tick_direction_alignment')
            if direction in {'CE', 'PE'} and direction == side:
                score += 2.0
                reasons.append('direction_context_alignment')
            if not bool(indicators.get('stale_data_used')):
                score += 1.0
            score += 1.0

            strategy_score = max(0.0, min(10.0, score))
            if strategy_score < 4.0:
                self.last_no_vote_reason = "low_score"
                return None
            metadata = {
                'strategy': 'OrderFlow',
                'side': side,
                'direction_bias': side,
                'strategy_score': strategy_score,
                'setup_quality': strategy_score,
                'setup_type': 'microstructure_imbalance',
                'required_data_present': depth_available,
                'stale_data_used': bool(indicators.get('stale_data_used')),
                'candidate_symbol': symbol,
                'score_reasons': reasons,
                'rejection_reasons': [] if depth_available else ['depth_missing'],
                'bid': bid,
                'ask': ask,
                'spread_pct': round(spread_pct, 3),
                'depth_imbalance': round(depth_imbalance, 4),
                'tick_direction': tick_direction,
                'liquidity_ok': spread_pct <= 12.0,
                'invalidation_level': bid - 0.5 * atr if side == 'CE' else ask + 0.5 * atr,
            }
            LOGGER.info('STRATEGY_VOTE strategy=OrderFlow side=%s score=%.2f', side, strategy_score)
            return EliteSignal(
                symbol=symbol,
                signal='BUY',
                confidence=max(0.1, min(0.85, strategy_score / 10.0)),
                entry_price=current_price,
                stop_loss=float(metadata['invalidation_level']),
                target=current_price + (1.8 * atr),
                quantity=self._cfg.quantity or 1,
                strategy_name='OrderFlow',
                metadata=metadata,
            )
        except Exception as e:
            LOGGER.error('Failure in OrderFlowStrategy._evaluate_signal: %s', e, exc_info=e)
            return None


__all__ = ['OrderFlowStrategy']
