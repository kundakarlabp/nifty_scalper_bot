"""Capital-aware Tuesday expiry gamma buyer elite strategy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os
from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    TuesdayGammaBuyerStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

BaseEliteStrategy = EliteStrategy
NIFTY_LOT_SIZE = int(os.getenv("NIFTY_LOT_SIZE", "75"))
MAX_RISK_PER_TRADE = 0.006
MAX_DAILY_RISK = 0.02
MAX_CAPITAL_DEPLOYMENT = 0.15


@dataclass(slots=True)
class _AccountState:
    """Args: none. Returns: account state container. Raises: never."""

    available_equity: float = 0.0


class EliteTuesdayGammaBuyer(BaseEliteStrategy):
    """Args: config, indicator_engine. Returns: initialized strategy. Raises: Exception."""

    def __init__(
        self,
        config: TuesdayGammaBuyerStrategyConfig,
        indicator_engine: Any,
    ) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._config = config
        self.account_state = _AccountState()
        self.daily_realized_pnl = 0.0
        self._trading_disabled_day: date | None = None

    def get_required_indicators(self) -> set[str]:
        """Args: none. Returns: required indicators. Raises: Exception."""
        return {
            'spot',
            'ltp',
            'vwap',
            'ema_fast',
            'ema_slow',
            'recent_high',
            'recent_low',
            'volume',
            'avg_volume',
            'atr',
            'atr_ma',
            'spot_return_3min',
        }

    def disable_trading_for_day(self) -> None:
        """Args: none. Returns: None. Raises: Exception."""
        try:
            self._trading_disabled_day = self._today_local()
            hook = getattr(self, '_on_daily_trading_disabled', None)
            if callable(hook):
                hook(self.name)
            LOGGER.info(
                'Condition met: tuesday_gamma_daily_halt',
                extra={'event': 'tuesday_gamma_daily_halt', 'strategy': self.name},
            )
        except Exception as e:
            LOGGER.error('Failure in disable_trading_for_day: %s', e, exc_info=e)

    def compute_position_size(self, entry: float, stop: float) -> int:
        """Args: entry, stop. Returns: lot-aligned quantity. Raises: Exception."""
        try:
            capital = float(self.account_state.available_equity)
            if capital <= 0:
                return 0

            risk_amount = capital * MAX_RISK_PER_TRADE
            risk_per_unit = abs(entry - stop)
            if risk_per_unit <= 0:
                return 0

            contracts = int(risk_amount / (risk_per_unit * NIFTY_LOT_SIZE))
            contracts = max(0, contracts)

            max_deploy = capital * MAX_CAPITAL_DEPLOYMENT
            while contracts > 0 and (entry * contracts * NIFTY_LOT_SIZE) > max_deploy:
                contracts -= 1

            quantity = contracts * NIFTY_LOT_SIZE
            if quantity < NIFTY_LOT_SIZE:
                return 0
            return quantity
        except Exception as e:
            LOGGER.error('Failure in compute_position_size: %s', e, exc_info=e)
            return 0

    def _today_local(self) -> date:
        """Args: none. Returns: local trading date. Raises: Exception."""
        clock = getattr(self, 'clock', None)
        if clock and callable(getattr(clock, 'now_local', None)):
            return clock.now_local().date()
        from datetime import datetime

        return datetime.now().date()

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        del position
        try:
            now_local = None
            clock = getattr(self, 'clock', None)
            if clock and callable(getattr(clock, 'now_local', None)):
                now_local = clock.now_local()

            strategy_mode = str(os.getenv('STRATEGY_MODE', 'directional_scalp')).lower()
            gamma_enabled = str(os.getenv('ALLOW_EXPIRY_GAMMA_STRATEGIES', 'false')).lower() in {'1', 'true', 'yes', 'on'}
            if not (strategy_mode == 'expiry_gamma' and gamma_enabled):
                LOGGER.debug('STRATEGY_NO_VOTE strategy=EliteTuesdayGammaBuyer reason=gamma_mode_disabled')
                return None

            if now_local is not None:
                if now_local.weekday() != 1:
                    return None
                if now_local.hour < 9 or now_local.hour > 14:
                    return None
                if now_local.hour == 9 and now_local.minute < 20:
                    return None
                if now_local.hour == 14 and now_local.minute >= 45:
                    return None

            if self._trading_disabled_day == self._today_local():
                return None

            spot = float(indicators.get('spot') or current_price or 0.0)
            vwap = float(indicators.get('vwap') or 0.0)
            ema_fast = float(indicators.get('ema_fast') or 0.0)
            ema_slow = float(indicators.get('ema_slow') or 0.0)
            recent_high = float(indicators.get('recent_high') or 0.0)
            recent_low = float(indicators.get('recent_low') or 0.0)
            volume = float(indicators.get('volume') or 0.0)
            avg_volume = float(indicators.get('avg_volume') or 0.0)
            atr = float(indicators.get('atr') or 0.0)
            atr_ma = float(indicators.get('atr_ma') or 0.0)
            spot_return_3min = float(indicators.get('spot_return_3min') or 0.0)

            if abs(spot_return_3min) > 0.006:
                return None

            if spot <= 0:
                return None

            score = 0
            if spot > vwap:
                score += 2
            else:
                score -= 2

            if ema_fast > ema_slow:
                score += 2
            else:
                score -= 2

            if current_price > recent_high:
                score += 2
            elif recent_low > 0 and current_price < recent_low:
                score -= 2

            if avg_volume > 0 and volume > 1.5 * avg_volume:
                score += 1
            elif avg_volume > 0 and volume < 0.8 * avg_volume:
                score -= 1

            if atr > atr_ma and atr_ma > 0:
                score += 1
            elif atr_ma > 0:
                score -= 1

            if score >= 5:
                option_type = 'CE'
            elif score <= -5:
                option_type = 'PE'
            else:
                return None

            position_manager = getattr(self, 'position_manager', None)
            if position_manager and callable(getattr(position_manager, 'has_open_position', None)):
                if bool(position_manager.has_open_position()):
                    return None

            capital = float(getattr(self.account_state, 'available_equity', 0.0) or 0.0)
            if capital <= 0:
                return None

            self.daily_realized_pnl = float(getattr(self, 'daily_realized_pnl', 0.0) or 0.0)
            if self.daily_realized_pnl <= -(capital * MAX_DAILY_RISK):
                self.disable_trading_for_day()
                return None

            stop_loss = current_price - (max(atr, current_price * 0.005) * 1.2)
            quantity = self.compute_position_size(entry=current_price, stop=stop_loss)
            if quantity < NIFTY_LOT_SIZE:
                return None

            collision_reduce = False
            orchestrator = getattr(self, '_orchestrator', None)
            if orchestrator and callable(getattr(orchestrator, 'active_strategies_for_symbol', None)):
                active = orchestrator.active_strategies_for_symbol(symbol)
                if active:
                    collision_reduce = True
            if collision_reduce:
                quantity = ((quantity // NIFTY_LOT_SIZE) // 2) * NIFTY_LOT_SIZE
                if quantity < NIFTY_LOT_SIZE:
                    return None

            if quantity % NIFTY_LOT_SIZE != 0:
                raise RuntimeError('Invalid lot size for NIFTY')

            return EliteSignal(
                symbol=symbol,
                signal='BUY',
                confidence=min(0.99, max(0.5, score / 10.0)),
                entry_price=current_price,
                stop_loss=stop_loss,
                target=current_price + (max(atr, current_price * 0.005) * 1.8),
                quantity=quantity,
                strategy_name='EliteTuesdayGammaBuyer',
                metadata={
                    'strategy': 'EliteTuesdayGammaBuyer',
                    'side': option_type,
                    'direction_bias': option_type,
                    'strategy_score': max(0.0, min(8.0, float(score))),
                    'setup_quality': max(0.0, min(8.0, float(score))),
                    'setup_type': 'expiry_gamma',
                    'required_data_present': True,
                    'stale_data_used': bool(indicators.get('stale_data_used')),
                    'candidate_symbol': symbol,
                    'score_reasons': ['expiry_gamma_mode', 'intraday_momentum_filter'],
                    'rejection_reasons': [],
                    'expiry_day': True,
                    'days_to_expiry': indicators.get('days_to_expiry'),
                    'gamma_mode_enabled': True,
                    'premium_decay_risk': round(abs(float(indicators.get('theta') or 0.0)) / max(current_price, 1.0), 4),
                    'volatility_expansion_confirmed': bool(atr > atr_ma and atr_ma > 0),
                    'option_type': option_type,
                    'bracket_type': 'VIRTUAL',
                    'sl_mode': 'ATR_TRAIL',
                    'enable_trailing': True,
                    'sl_atr_mult': 1.2,
                    'trailing_atr_mult': 1.2,
                    'tp2_atr_mult': 1.8,
                    'atr_multiplier': self._config.atr_multiplier,
                    'target_multiplier': self._config.target_multiplier,
                    'score': score,
                },
            )
        except Exception as e:
            LOGGER.error('Failure in EliteTuesdayGammaBuyer._evaluate_signal: %s', e, exc_info=e)
            return None
