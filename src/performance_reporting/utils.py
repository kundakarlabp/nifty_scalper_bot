import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import List, Tuple

logger = logging.getLogger(__name__)

class PerformanceVisualizer:
    """Visualize performance data"""
    
    @staticmethod
    def plot_equity_curve(timestamps: List[datetime], equity_curve: List[float], 
                         title: str = "Equity Curve") -> plt.Figure:
        """Plot equity curve"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(timestamps, equity_curve, linewidth=2, color='#2E86AB')
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Equity (₹)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
            return None
    
    @staticmethod
    def plot_drawdown_curve(timestamps: List[datetime], equity_curve: List[float]) -> plt.Figure:
        """Plot drawdown curve"""
        try:
            equity_array = np.array(equity_curve)
            peak = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - peak) / peak * 100
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(timestamps, drawdown, linewidth=2, color='#A23B72')
            ax.set_title('Drawdown Curve', fontsize=16, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Drawdown (%)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Error plotting drawdown curve: {e}")
            return None
    
    @staticmethod
    def plot_pnl_distribution(pnls: List[float]) -> plt.Figure:
        """Plot P&L distribution histogram"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(pnls, bins=30, alpha=0.7, color='#F18F01', edgecolor='black')
            ax.set_title('P&L Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Profit/Loss (₹)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Error plotting P&L distribution: {e}")
            return None

class PerformanceExporter:
    """Export performance data to various formats"""
    
    @staticmethod
    def export_trades_to_csv(trade_records: List, filename: str) -> bool:
        """Export trade records to CSV"""
        try:
            # Convert trade records to DataFrame
            trades_data = []
            for trade in trade_records:
                trades_data.append({
                    'timestamp': trade.timestamp,
                    'symbol': trade.symbol,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'pnl_percentage': trade.pnl_percentage,
                    'holding_period': trade.holding_period,
                    'stop_loss': trade.stop_loss,
                    'target': trade.target,
                    'confidence': trade.confidence,
                    'status': trade.status
                })
            
            df = pd.DataFrame(trades_data)
            df.to_csv(filename, index=False)
            logger.info(f"✅ Trades exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting trades to CSV: {e}")
            return False
    
    @staticmethod
    def export_metrics_to_csv(metrics: dict, filename: str) -> bool:
        """Export performance metrics to CSV"""
        try:
            # Flatten metrics dictionary
            flat_metrics = {}
            for section, values in metrics.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        flat_metrics[f"{section}_{key}"] = value
                else:
                    flat_metrics[section] = values
            
            # Create DataFrame
            df = pd.DataFrame([flat_metrics])
            df.to_csv(filename, index=False)
            logger.info(f"✅ Metrics exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting metrics to CSV: {e}")
            return False

# Example usage
if __name__ == "__main__":
    print("Performance utilities ready!")
    print("Available classes:")
    print("- PerformanceVisualizer: For plotting performance charts")
    print("- PerformanceExporter: For exporting data to files")
