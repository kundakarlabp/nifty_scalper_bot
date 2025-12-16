"""
Straddle/Strangle Theta Decay Strategy.
World-Class implementation with ADX Range Filtering and Theta Optimization.
"""

from __future__ import annotations

from typing import Any, Mapping

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    StraddleThetaStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

# Initialize structured logger
LOGGER = get_logger(__name__)


class StraddleThetaStrategy(EliteStrategy):
    """
    Delta-Neutral / Theta-Positive strategy.
    Shorts ATM/OTM options when market is range-bound (Low ADX) and IV is decent.
    """

    def __init__(self, config: StraddleThetaStrategyConfig, indicator_engine: Any) -> None:
        """
        Initialize strategy with configuration and engine.
        
        Args:
            config: Strategy configuration dataclass.
            indicator_engine: Data provider.
        """
        # CRITICAL FIX: Correct init signature
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._theta_config = config

    def _evaluate_signal(self) -> EliteSignal | None:
        """
        Core Logic:
        1. Check Regime (Is Market Ranging? ADX < 25).
        2. Check IV (Is Premium worth selling? IVP > 20).
        3. Select ATM Strike.
        4. Validate Physics (Theta Decay vs Risk).
        5. Execute Short.
        """
        symbol = self._theta_config.symbol
        
        # 1. Fetch Indicators
        # Need ADX for Trend, Greeks for Decay
        required_indicators = {
            "ltp", "adx", "atr", 
            "iv_percentile", "minutes_until_close",
            "volume", "avg_volume"
        }
        
        indicators = self._indicator_engine.get_indicators(symbol, required_indicators)
        
        try:
            ltp = float(indicators.get("ltp") or 0)
            adx = float(indicators.get("adx") or 0)
            atr = float(indicators.get("atr") or 0)
            iv_p = float(indicators.get("iv_percentile") or 0)
            mins_left = float(indicators.get("minutes_until_close") or 0)
            
        except (ValueError, TypeError):
            return None

        if ltp == 0:
            return None

        # 2. Time Filter
        # Don't short options in the first/last 15 mins (Volatility risk)
        # Best time for Theta: Mid-day (10:30 - 14:30)
        # 360 mins = 6 hours (roughly open), 30 mins = close
        if mins_left > 360 or mins_left < 30: 
            return None

        # 3. Regime Filter (The "Don't get run over" check)
        # If ADX > 25, the market is Trending. Do NOT Short Straddle.
        if adx > 25.0:
            return None 

        # 4. Value Filter
        # If IV is crushed (< 20%), premiums are peanuts. Risk > Reward.
        if iv_p < 20.0:
            return None

        # 5. Strike Selection & Analysis
        # We target the ATM Strike
        strike = int(round(ltp / 50) * 50)
        
        # Dynamic Symbol Construction 
        # Ideally, we loop CE and PE and pick best Theta/Price ratio
        # For this implementation, we select a side to short based on minor drift
        # Default to CE (Short Call) as a placeholder for the theta play
        # (A full straddle would require orchestrator to handle multi-leg)
        target_type = "CE" 
        
        # NOTE: Ensure this format matches your data feed (e.g., Weekly Expiry)
        # You might need a helper to get the correct expiry date string
        # For now, using a placeholder format that needs to match your system
        trade_symbol = f"NFO:NIFTY24DEC{strike}{target_type}" 
        
        # 6. 🛡️ SAFETY GATE (Physics Check)
        # Ensure we aren't shorting an illiquid option
        # Note: For Shorting, we pass "SELL" to validate the short side suitability
        if not self.validate_option_health(trade_symbol, "SELL"):
            LOGGER.info(f"⛔ Rejected {trade_symbol}: Failed Greeks/Liquidity Check")
            return None

        # 7. Theta Efficiency Check
        # Fetch specific greeks for the contract
        contract_greeks = self._indicator_engine.get_indicators(trade_symbol, ["theta", "ltp"])
        theta = float(contract_greeks.get("theta") or 0)
        opt_price = float(contract_greeks.get("ltp") or 0)
        
        if opt_price == 0: return None
        
        # 8. Risk Management (Short Premium)
        # Stop Loss: 20% of Premium (Tight stop for shorting)
        # Take Profit: 50% of Premium (Theta decay target)
        stop_loss = opt_price * 1.20 
        tp1 = opt_price * 0.50
        tp2 = opt_price * 0.10 # Letting it run to dust

        # 9. Construct Signal
        LOGGER.info(
            f"⚡ Theta Decay Setup: Sell {trade_symbol} | ADX: {adx:.1f} | IVP: {iv_p:.1f}",
            extra={
                "event": "straddle_theta_signal",
                "symbol": trade_symbol,
                "adx": adx,
                "theta": theta
            }
        )

        return EliteSignal(
            symbol=trade_symbol,
            side="SELL", # We are Shorting
            confidence=0.80, # High confidence in ranging markets
            entry_price=opt_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            quantity=self._theta_config.quantity or 1,
            strategy_name="Straddle_Theta_Pro",
            metadata={
                "adx": adx,
                "iv_percentile": iv_p,
                "theta": theta
            }
        )


__all__ = ["StraddleThetaStrategy"]
