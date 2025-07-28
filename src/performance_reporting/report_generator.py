import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import json
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    average_pnl: float = 0.0
    max_pnl: float = 0.0
    min_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    average_holding_period: float = 0.0
    total_volume: float = 0.0
    average_position_size: float = 0.0
    risk_reward_ratio: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0

@dataclass
class TradeRecord:
    """Individual trade record"""
    timestamp: datetime
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_percentage: float
    holding_period: int
    stop_loss: float
    target: float
    confidence: float
    status: str

class PerformanceReportGenerator:
    """Generate comprehensive performance reports"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.trade_records = []
        self.equity_curve = []
        self.timestamps = []
    
    def add_trade_record(self, trade_record: TradeRecord):
        """Add a trade record to the performance tracker"""
        try:
            self.trade_records.append(trade_record)
            logger.debug(f"Added trade record: {trade_record.symbol} {trade_record.direction}")
        except Exception as e:
            logger.error(f"Error adding trade record: {e}")
    
    def add_equity_point(self, timestamp: datetime, equity: float):
        """Add equity curve point"""
        try:
            self.timestamps.append(timestamp)
            self.equity_curve.append(equity)
            logger.debug(f"Added equity point: {timestamp} - ₹{equity:,.2f}")
        except Exception as e:
            logger.error(f"Error adding equity point: {e}")
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.trade_records:
                logger.warning("No trade records available for metrics calculation")
                return PerformanceMetrics()
            
            # Convert trade records to lists for easier calculation
            pnls = [t.pnl for t in self.trade_records]
            holding_periods = [t.holding_period for t in self.trade_records]
            position_sizes = [t.quantity for t in self.trade_records]
            confidences = [t.confidence for t in self.trade_records]
            
            # Basic statistics
            total_trades = len(self.trade_records)
            winning_trades = len([p for p in pnls if p > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # P&L statistics
            total_pnl = sum(pnls)
            average_pnl = total_pnl / total_trades if total_trades > 0 else 0
            max_pnl = max(pnls) if pnls else 0
            min_pnl = min(pnls) if pnls else 0
            
            # Max drawdown
            max_drawdown = self._calculate_max_drawdown()
            
            # Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(pnls)
            
            # Sortino ratio
            sortino_ratio = self._calculate_sortino_ratio(pnls)
            
            # Profit factor
            gross_profits = sum([p for p in pnls if p > 0])
            gross_losses = abs(sum([p for p in pnls if p < 0]))
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
            
            # Consecutive wins/losses
            max_consecutive_wins = self._calculate_max_consecutive(pnls, positive=True)
            max_consecutive_losses = self._calculate_max_consecutive(pnls, positive=False)
            
            # Average holding period
            avg_holding_period = np.mean(holding_periods) if holding_periods else 0
            
            # Volume and position size
            total_volume = sum([abs(p) for p in pnls])
            avg_position_size = np.mean(position_sizes) if position_sizes else 0
            
            # Risk-reward ratio (simplified)
            risk_reward_ratio = self._calculate_avg_risk_reward()
            
            # Volatility
            volatility = np.std(pnls) if len(pnls) > 1 else 0
            
            # Calmar ratio
            calmar_ratio = self._calculate_calmar_ratio(total_pnl, max_drawdown)
            
            self.metrics = PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                average_pnl=average_pnl,
                max_pnl=max_pnl,
                min_pnl=min_pnl,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                profit_factor=profit_factor,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                average_holding_period=avg_holding_period,
                total_volume=total_volume,
                average_position_size=avg_position_size,
                risk_reward_ratio=risk_reward_ratio,
                volatility=volatility,
                calmar_ratio=calmar_ratio
            )
            
            logger.info(f"Performance metrics calculated: {total_trades} trades, P&L: ₹{total_pnl:,.2f}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics()
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        try:
            if len(self.equity_curve) < 2:
                return 0.0
            
            equity_array = np.array(self.equity_curve)
            peak = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - peak) / peak
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
            
            return abs(max_drawdown)
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = np.array(returns) - risk_free_rate
            std_dev = np.std(excess_returns)
            
            if std_dev == 0:
                return 0.0
            
            sharpe = np.mean(excess_returns) / std_dev
            # Annualize (assuming 252 trading days)
            annualized_sharpe = sharpe * np.sqrt(min(len(returns), 252))
            
            return annualized_sharpe
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (Sharpe ratio focusing on downside risk)"""
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = np.array(returns) - risk_free_rate
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0:
                return float('inf')
            
            downside_deviation = np.std(downside_returns)
            
            if downside_deviation == 0:
                return 0.0
            
            sortino = np.mean(excess_returns) / downside_deviation
            # Annualize
            annualized_sortino = sortino * np.sqrt(min(len(returns), 252))
            
            return annualized_sortino
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_max_consecutive(self, pnls: List[float], positive: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        try:
            max_count = 0
            current_count = 0
            
            for pnl in pnls:
                if (positive and pnl > 0) or (not positive and pnl < 0):
                    current_count += 1
                    max_count = max(max_count, current_count)
                else:
                    current_count = 0
            
            return max_count
        except Exception as e:
            logger.error(f"Error calculating consecutive trades: {e}")
            return 0
    
    def _calculate_avg_risk_reward(self) -> float:
        """Calculate average risk-reward ratio"""
        try:
            if not self.trade_records:
                return 0.0
            
            ratios = []
            for trade in self.trade_records:
                risk = abs(trade.entry_price - trade.stop_loss)
                reward = abs(trade.target - trade.entry_price)
                if risk > 0:
                    ratios.append(reward / risk)
            
            return np.mean(ratios) if ratios else 0.0
        except Exception as e:
            logger.error(f"Error calculating risk-reward ratio: {e}")
            return 0.0
    
    def _calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        try:
            if max_drawdown == 0:
                return float('inf') if total_return > 0 else 0.0
            return abs(total_return) / max_drawdown
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0
    
    def generate_performance_report(self, format_type: str = "dict") -> Dict:
        """Generate comprehensive performance report"""
        try:
            # Calculate metrics if not already done
            if self.metrics.total_trades == 0:
                self.calculate_performance_metrics()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_trades': self.metrics.total_trades,
                    'winning_trades': self.metrics.winning_trades,
                    'losing_trades': self.metrics.losing_trades,
                    'win_rate': round(self.metrics.win_rate * 100, 2),
                    'total_pnl': round(self.metrics.total_pnl, 2),
                    'average_pnl': round(self.metrics.average_pnl, 2),
                    'max_pnl': round(self.metrics.max_pnl, 2),
                    'min_pnl': round(self.metrics.min_pnl, 2)
                },
                'risk_metrics': {
                    'max_drawdown': round(self.metrics.max_drawdown * 100, 2),
                    'sharpe_ratio': round(self.metrics.sharpe_ratio, 2),
                    'sortino_ratio': round(self.metrics.sortino_ratio, 2),
                    'profit_factor': round(self.metrics.profit_factor, 2),
                    'calmar_ratio': round(self.metrics.calmar_ratio, 2),
                    'volatility': round(self.metrics.volatility, 2)
                },
                'trading_metrics': {
                    'max_consecutive_wins': self.metrics.max_consecutive_wins,
                    'max_consecutive_losses': self.metrics.max_consecutive_losses,
                    'average_holding_period': round(self.metrics.average_holding_period, 1),
                    'average_position_size': round(self.metrics.average_position_size, 2),
                    'risk_reward_ratio': round(self.metrics.risk_reward_ratio, 2),
                    'total_volume': round(self.metrics.total_volume, 2)
                },
                'recent_trades': self._get_recent_trades(5)
            }
            
            if format_type == "json":
                return json.dumps(report, indent=2, default=str)
            else:
                return report
                
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}
    
    def _get_recent_trades(self, count: int = 5) -> List[Dict]:
        """Get recent trade records for reporting"""
        try:
            recent_trades = self.trade_records[-count:] if len(self.trade_records) >= count else self.trade_records
            return [asdict(trade) for trade in recent_trades]
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    def get_equity_curve(self) -> tuple:
        """Get equity curve data"""
        return self.timestamps.copy(), self.equity_curve.copy()
    
    def reset(self):
        """Reset performance tracking"""
        self.metrics = PerformanceMetrics()
        self.trade_records = []
        self.equity_curve = []
        self.timestamps = []
        logger.info("Performance tracking reset")

# Example usage
if __name__ == "__main__":
    print("Performance Report Generator ready!")
    print("Import and use: from src.performance_reporting.report_generator import PerformanceReportGenerator")
