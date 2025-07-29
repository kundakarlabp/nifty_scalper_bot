# src/risk/position_sizing.py
import logging
from typing import Dict, Optional
from config import (
    NIFTY_LOT_SIZE, 
    ACCOUNT_SIZE, 
    RISK_PER_TRADE, 
    MIN_LOTS, 
    MAX_LOTS
)

logger = logging.getLogger(__name__)

class PositionSizing:
    """
    Manages position sizing and risk control for the trading bot.
    Calculates appropriate position sizes based on account risk parameters,
    signal confidence, market volatility, and performance factors.
    """
    
    def __init__(self, 
                 account_size: float = None, 
                 max_positions: int = 1, 
                 risk_per_trade: float = None, 
                 max_drawdown: float = 0.05):
        """
        Initialize the PositionSizing manager.

        Args:
            account_size (float): Total trading capital.
            max_positions (int): Maximum concurrent positions allowed.
            risk_per_trade (float): Risk percentage per trade (0.01 = 1%).
            max_drawdown (float): Maximum allowed daily drawdown (0.05 = 5%).
        """
        # Configuration parameters
        self.account_size = account_size or ACCOUNT_SIZE
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade or RISK_PER_TRADE
        self.max_drawdown = max_drawdown
        
        # Instrument-specific parameters
        self.nifty_lot_size = NIFTY_LOT_SIZE
        self.min_lots = MIN_LOTS
        self.max_lots = MAX_LOTS
        
        # Runtime tracking variables
        self.current_positions = 0
        self.daily_pnl = 0.0
        self.max_daily_loss = self.account_size * max_drawdown
        self.current_daily_loss = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0

    def calculate_position_size(self, 
                                entry_price: float, 
                                stop_loss: float, 
                                signal_confidence: float = 1.0, 
                                market_volatility: float = 1.0) -> Dict[str, float]:
        """
        Calculate the appropriate position size based on risk parameters.

        Args:
            entry_price (float): Entry price for the trade.
            stop_loss (float): Stop loss price.
            signal_confidence (float): Confidence level of the signal (0.0 to 1.0).
            market_volatility (float): Market volatility factor.

        Returns:
            Dict[str, float]: Position details including quantity, lots, and risk metrics.
        """
        try:
            # Check if we can open another position
            if self.current_positions >= self.max_positions:
                logger.warning("Maximum positions reached")
                return self._empty_position_info()

            # Calculate risk per point
            risk_per_point = abs(entry_price - stop_loss)
            if risk_per_point <= 0:
                logger.error("Invalid stop loss calculation - zero risk per point")
                return self._empty_position_info()

            # Calculate base risk amount
            base_risk_amount = self.account_size * self.risk_per_trade
            
            # Apply multipliers
            confidence_multiplier = self._calculate_confidence_multiplier(signal_confidence)
            volatility_multiplier = self._calculate_volatility_multiplier(market_volatility)
            performance_multiplier = self._calculate_performance_multiplier()

            # Calculate adjusted risk amount
            adjusted_risk_amount = (base_risk_amount * 
                                    confidence_multiplier * 
                                    volatility_multiplier * 
                                    performance_multiplier)

            # Calculate raw quantity
            quantity = int(adjusted_risk_amount / risk_per_point)
            
            # Convert to lots
            lots = max(self.min_lots, quantity // self.nifty_lot_size)
            lots = min(self.max_lots, lots)
            final_quantity = lots * self.nifty_lot_size

            # Check against daily loss limit
            risk_amount = final_quantity * risk_per_point
            if self.current_daily_loss + risk_amount > self.max_daily_loss:
                remaining_risk_budget = self.max_daily_loss - self.current_daily_loss
                max_safe_quantity = int(remaining_risk_budget / risk_per_point)
                lots = max(0, max_safe_quantity // self.nifty_lot_size)
                final_quantity = lots * self.nifty_lot_size
                risk_amount = final_quantity * risk_per_point

            # Log calculation details
            logger.info(f"Position sizing: Lots={lots}, Quantity={final_quantity}, Risk=₹{risk_amount:.2f}")
            logger.debug(f"Multipliers - Confidence: {confidence_multiplier:.2f}, "
                        f"Volatility: {volatility_multiplier:.2f}, Performance: {performance_multiplier:.2f}")

            return {
                'quantity': final_quantity,
                'lots': lots,
                'risk_amount': risk_amount,
                'confidence_adjusted': signal_confidence * confidence_multiplier,
                'volatility_factor': volatility_multiplier,
                'performance_factor': performance_multiplier
            }

        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return self._empty_position_info()

    def _calculate_confidence_multiplier(self, signal_confidence: float) -> float:
        """
        Calculate confidence-based position size multiplier.

        Args:
            signal_confidence (float): Signal confidence level (0.0 to 1.0).

        Returns:
            float: Confidence multiplier.
        """
        # Base multiplier starts at 0.8
        # For confidences above 0.7, increase multiplier up to 2.0 at 1.0 confidence
        return 0.8 + (max(0, signal_confidence - 0.7) * 1.33)

    def _calculate_volatility_multiplier(self, market_volatility: float) -> float:
        """
        Calculate volatility-based position size multiplier.

        Args:
            market_volatility (float): Market volatility factor.

        Returns:
            float: Volatility multiplier (0.5 to 2.0).
        """
        # Inverse relationship - higher volatility = smaller positions
        # Clamp between 0.5 and 2.0
        return max(0.5, min(2.0, 1.0 / market_volatility))

    def _calculate_performance_multiplier(self) -> float:
        """
        Calculate performance-based position size multiplier.

        Returns:
            float: Performance multiplier.
        """
        try:
            # Increase position size after consecutive wins (max 1.5x)
            if self.consecutive_wins >= 3:
                return min(1.5, 1.0 + (self.consecutive_wins * 0.1))
            
            # Decrease position size after consecutive losses (min 0.5x)
            elif self.consecutive_losses >= 2:
                return max(0.5, 1.0 - (self.consecutive_losses * 0.15))
            
            # Decrease position size when approaching daily drawdown limit
            elif self.current_daily_loss >= (self.max_daily_loss * 0.5):
                return max(0.7, 1.0 - (self.current_daily_loss / self.max_daily_loss) * 0.5)
            
            # Neutral position size
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating performance multiplier: {e}", exc_info=True)
            return 1.0

    def _empty_position_info(self) -> Dict[str, float]:
        """
        Return an empty position information dictionary.

        Returns:
            Dict[str, float]: Empty position information.
        """
        return {
            'quantity': 0,
            'lots': 0,
            'risk_amount': 0,
            'confidence_adjusted': 0,
            'volatility_factor': 1.0,
            'performance_factor': 1.0
        }

    def update_position_status(self, is_open: bool = True) -> None:
        """
        Update the current position count.

        Args:
            is_open (bool): True if opening a position, False if closing.
        """
        if is_open:
            self.current_positions += 1
        else:
            self.current_positions = max(0, self.current_positions - 1)
        logger.info(f"Current positions: {self.current_positions}")

    def update_trade_result(self, pnl: float) -> None:
        """
        Update trading statistics after a trade is closed.

        Args:
            pnl (float): Profit or loss from the closed trade.
        """
        # Decrement position count
        self.current_positions = max(0, self.current_positions - 1)
        
        # Update P&L tracking
        self.daily_pnl += pnl
        
        # Update win/loss streaks and drawdown
        if pnl < 0:
            self.current_daily_loss += abs(pnl)
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        else:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            
        logger.info(f"Trade result updated. P&L: ₹{pnl:.2f}, Daily P&L: ₹{self.daily_pnl:.2f}")

    def reset_daily_stats(self) -> None:
        """Reset all daily trading statistics."""
        self.daily_pnl = 0.0
        self.current_daily_loss = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        logger.info("Daily statistics reset")

    def can_trade(self) -> bool:
        """
        Check if trading is allowed based on position and drawdown limits.

        Returns:
            bool: True if trading is allowed, False otherwise.
        """
        if self.current_positions >= self.max_positions:
            logger.warning("Maximum positions limit reached")
            return False
        if self.current_daily_loss >= self.max_daily_loss:
            logger.warning("Maximum daily drawdown reached")
            return False
        return True

    def get_risk_status(self) -> Dict[str, float]:
        """
        Get current risk management status.

        Returns:
            Dict[str, float]: Current risk metrics.
        """
        return {
            'account_size': self.account_size,
            'current_positions': self.current_positions,
            'max_positions': self.max_positions,
            'daily_pnl': self.daily_pnl,
            'current_daily_loss': self.current_daily_loss,
            'max_daily_loss': self.max_daily_loss,
            'drawdown_percentage': (self.current_daily_loss / self.account_size) * 100 if self.account_size > 0 else 0,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'available_risk_budget': max(0, self.max_daily_loss - self.current_daily_loss)
        }

# Example usage (if run directly)
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    risk_manager = PositionSizing(account_size=100000, risk_per_trade=0.01)
    position_info = risk_manager.calculate_position_size(
        entry_price=18000,
        stop_loss=17980,
        signal_confidence=0.85,
        market_volatility=1.2
    )
    print(f"Position Info: {position_info}")
    print(f"Risk Status: {risk_manager.get_risk_status()}")