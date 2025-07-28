import logging
from typing import Dict, Optional
from config import NIFTY_LOT_SIZE, ACCOUNT_SIZE, RISK_PER_TRADE, MIN_LOTS, MAX_LOTS

logger = logging.getLogger(__name__)

class PositionSizing:
    def __init__(self, account_size: float = None, max_positions: int = 1, 
                 risk_per_trade: float = None, max_drawdown: float = 0.05):
        self.account_size = account_size or ACCOUNT_SIZE
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade or RISK_PER_TRADE
        self.max_drawdown = max_drawdown
        self.nifty_lot_size = NIFTY_LOT_SIZE
        self.min_lots = MIN_LOTS
        self.max_lots = MAX_LOTS
        self.current_positions = 0
        self.daily_pnl = 0.0
        self.max_daily_loss = self.account_size * max_drawdown
        self.current_daily_loss = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              signal_confidence: float = 1.0, market_volatility: float = 1.0) -> Dict[str, float]:
        """Calculate dynamic position size based on risk management rules"""
        try:
            # Check if we can take more positions
            if self.current_positions >= self.max_positions:
                logger.warning("Maximum positions reached")
                return self._empty_position_info()
            
            # Calculate risk per point
            risk_per_point = abs(entry_price - stop_loss)
            if risk_per_point <= 0:
                logger.error("Invalid stop loss calculation")
                return self._empty_position_info()
            
            # Calculate maximum risk amount for this trade
            base_risk_amount = self.account_size * self.risk_per_trade
            
            # Adjust risk based on signal confidence (0.7-1.0 -> 0.8-1.2 multiplier)
            confidence_multiplier = 0.8 + (max(0, signal_confidence - 0.7) * 1.33)
            
            # Adjust risk based on market volatility (lower volatility = higher position)
            volatility_multiplier = max(0.5, min(2.0, 1.0 / market_volatility))
            
            # Adjust based on recent performance
            performance_multiplier = self._calculate_performance_multiplier()
            
            # Final risk amount calculation
            adjusted_risk_amount = (base_risk_amount * 
                                  confidence_multiplier * 
                                  volatility_multiplier * 
                                  performance_multiplier)
            
            # Calculate quantity based on risk
            quantity = int(adjusted_risk_amount / risk_per_point)
            
            # Convert to lot size (Nifty lot size = 75)
            lots = max(self.min_lots, quantity // self.nifty_lot_size)
            lots = min(self.max_lots, lots)  # Cap at maximum lots
            
            # Final quantity
            final_quantity = lots * self.nifty_lot_size
            
            # Ensure we stay within daily drawdown limits
            if self.current_daily_loss + (final_quantity * risk_per_point) > self.max_daily_loss:
                # Reduce position size to stay within drawdown limit
                remaining_risk_budget = self.max_daily_loss - self.current_daily_loss
                max_safe_quantity = int(remaining_risk_budget / risk_per_point)
                lots = max(0, max_safe_quantity // self.nifty_lot_size)
                final_quantity = lots * self.nifty_lot_size
            
            risk_amount = final_quantity * risk_per_point
            
            logger.info(f"Position sizing: Lots={lots}, Quantity={final_quantity}, Risk=₹{risk_amount:.2f}")
            logger.info(f"Multipliers - Confidence: {confidence_multiplier:.2f}, "
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
            logger.error(f"Error calculating position size: {e}")
            return self._empty_position_info()
    
    def _calculate_performance_multiplier(self) -> float:
        """Calculate multiplier based on recent trading performance"""
        try:
            # Winning streak bonus
            if self.consecutive_wins >= 3:
                return min(1.5, 1.0 + (self.consecutive_wins * 0.1))
            
            # Losing streak penalty
            elif self.consecutive_losses >= 2:
                return max(0.5, 1.0 - (self.consecutive_losses * 0.15))
            
            # Drawdown penalty
            elif self.current_daily_loss >= (self.max_daily_loss * 0.5):
                return max(0.7, 1.0 - (self.current_daily_loss / self.max_daily_loss) * 0.5)
            
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating performance multiplier: {e}")
            return 1.0
    
    def _empty_position_info(self) -> Dict[str, float]:
        """Return empty position info"""
        return {
            'quantity': 0,
            'lots': 0,
            'risk_amount': 0,
            'confidence_adjusted': 0,
            'volatility_factor': 1.0,
            'performance_factor': 1.0
        }
    
    def update_position_status(self, is_open: bool = True):
        """Update position tracking"""
        if is_open:
            self.current_positions += 1
        else:
            self.current_positions = max(0, self.current_positions - 1)
        
        logger.info(f"Current positions: {self.current_positions}")
    
    def update_trade_result(self, pnl: float):
        """Update after trade completion"""
        self.current_positions = max(0, self.current_positions - 1)
        self.daily_pnl += pnl
        if pnl < 0:
            self.current_daily_loss += abs(pnl)
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        else:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        
        logger.info(f"Trade result updated. P&L: ₹{pnl:.2f}, Daily P&L: ₹{self.daily_pnl:.2f}")
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0.0
        self.current_daily_loss = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        logger.info("Daily statistics reset")
    
    def can_trade(self) -> bool:
        """Check if we can place a new trade"""
        if self.current_positions >= self.max_positions:
            logger.warning("Maximum positions limit reached")
            return False
        
        if self.current_daily_loss >= self.max_daily_loss:
            logger.warning("Maximum daily drawdown reached")
            return False
        
        return True
    
    def get_risk_status(self) -> Dict[str, float]:
        """Get current risk management status"""
        return {
            'account_size': self.account_size,
            'current_positions': self.current_positions,
            'max_positions': self.max_positions,
            'daily_pnl': self.daily_pnl,
            'current_daily_loss': self.current_daily_loss,
            'max_daily_loss': self.max_daily_loss,
            'drawdown_percentage': (self.current_daily_loss / self.account_size) * 100,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'available_risk_budget': self.max_daily_loss - self.current_daily_loss
        }

# Example usage
if __name__ == "__main__":
    risk_manager = PositionSizing(account_size=100000, risk_per_trade=0.01)
    
    # Calculate position size for a trade
    position_info = risk_manager.calculate_position_size(
        entry_price=18000,
        stop_loss=17980,
        signal_confidence=0.85,
        market_volatility=1.2
    )
    
    print(f"Position Info: {position_info}")
    print(f"Risk Status: {risk_manager.get_risk_status()}")
