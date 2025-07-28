import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class BacktestAnalyzer:
    """Utility class for analyzing backtest results"""
    
    @staticmethod
    def plot_equity_curve(timestamps: List[datetime], equity_curve: List[float], 
                         title: str = "Equity Curve"):
        """Plot equity curve"""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, equity_curve)
            plt.title(title)
            plt.xlabel('Time')
            plt.ylabel('Equity (₹)')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            return plt
        except Exception as e:
            logger.error(f"❌ Error plotting equity curve: {e}")
            return None
    
    @staticmethod
    def plot_drawdown_curve(timestamps: List[datetime], equity_curve: List[float]):
        """Plot drawdown curve"""
        try:
            equity_array = np.array(equity_curve)
            peak = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - peak) / peak * 100
            
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, drawdown)
            plt.title('Drawdown Curve')
            plt.xlabel('Time')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            return plt
        except Exception as e:
            logger.error(f"❌ Error plotting drawdown curve: {e}")
            return None
    
    @staticmethod
    def calculate_trade_statistics(trades: List) -> Dict:
        """Calculate detailed trade statistics"""
        try:
            if not trades:
                return {}
            
            pnls = [t.pnl for t in trades]
            holding_periods = [t.holding_period for t in trades]
            confidence_scores = [t.confidence for t in trades]
            
            stats = {
                'total_pnl': sum(pnls),
                'average_pnl': np.mean(pnls),
                'pnl_std': np.std(pnls),
                'max_win': max(pnls) if pnls else 0,
                'max_loss': min(pnls) if pnls else 0,
                'average_holding_period': np.mean(holding_periods),
                'average_confidence': np.mean(confidence_scores),
                'winning_trades': len([p for p in pnls if p > 0]),
                'losing_trades': len([p for p in pnls if p < 0]),
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0
            }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Error calculating trade statistics: {e}")
            return {}

class StrategyOptimizer:
    """Utility class for strategy optimization"""
    
    @staticmethod
    def optimize_parameters(data: pd.DataFrame, param_ranges: Dict) -> Dict:
        """Optimize strategy parameters (simplified)"""
        try:
            logger.info("🔧 Starting parameter optimization...")
            
            # This is a simplified optimization example
            # In practice, you'd use more sophisticated methods
            
            best_params = {}
            best_performance = float('-inf')
            
            # Example: optimize stop loss and target points
            sl_range = param_ranges.get('stop_loss_points', [10, 20, 30])
            target_range = param_ranges.get('target_points', [20, 30, 40, 50])
            
            for sl in sl_range:
                for target in target_range:
                    # Run backtest with these parameters
                    # This is simplified - you'd need to modify the strategy to accept parameters
                    performance = target / sl  # Simple ratio as example metric
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_params = {
                            'stop_loss_points': sl,
                            'target_points': target
                        }
            
            logger.info(f"✅ Optimization complete. Best params: {best_params}")
            return {
                'best_params': best_params,
                'best_performance': best_performance
            }
            
        except Exception as e:
            logger.error(f"❌ Error in parameter optimization: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    print("Backtest utilities ready!")
    print("Available classes:")
    print("- BacktestAnalyzer: For analyzing results")
    print("- StrategyOptimizer: For parameter optimization")
