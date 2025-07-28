import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import time
from src.strategies.scalping_strategy import DynamicScalpingStrategy
from src.risk.position_sizing import PositionSizing

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: str  # 'BUY' or 'SELL'
    quantity: int
    pnl: float
    pnl_percentage: float
    holding_period: int  # in minutes
    stop_loss: float
    target: float
    confidence: float

@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    average_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    average_holding_period: float
    trades: List[Trade]

class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategy = DynamicScalpingStrategy()
        self.risk_manager = PositionSizing(account_size=initial_capital)
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
        
    def run_backtest(self, data: pd.DataFrame, start_date: datetime = None, 
                    end_date: datetime = None, show_progress: bool = True) -> BacktestResult:
        """Run backtest on historical data"""
        try:
            # Filter data by date range if specified
            if start_date or end_date:
                mask = True
                if start_date:
                    mask = mask & (data.index >= start_date)
                if end_date:
                    mask = mask & (data.index <= end_date)
                test_data = data[mask].copy()
            else:
                test_data = data.copy()
            
            if test_data.empty:
                logger.warning("⚠️  No data available for backtest")
                return self._create_empty_result()
            
            logger.info(f"🚀 Starting backtest on {len(test_data)} data points")
            
            # Reset tracking variables
            self.trades = []
            self.equity_curve = [self.initial_capital]
            self.timestamps = [test_data.index[0]]
            self.current_capital = self.initial_capital
            
            # Initialize position tracking
            current_position = None
            position_info = None
            
            # Process each bar
            total_bars = len(test_data)
            start_time = time.time()
            
            for i in range(50, len(test_data)):  # Start after 50 bars for indicator warmup
                current_bar = test_data.iloc[i]
                current_time = test_data.index[i]
                current_price = current_bar['close']
                
                # Show progress
                if show_progress and i % max(1, total_bars // 20) == 0:  # Show progress every 5%
                    elapsed_time = time.time() - start_time
                    progress_percent = (i - 50) / (total_bars - 50) * 100
                    eta = (elapsed_time / (i - 50)) * (total_bars - 50 - (i - 50)) if (i - 50) > 0 else 0
                    print(f"\r📊 Progress: {progress_percent:.1f}% | ETA: {eta:.1f}s | Trades: {len(self.trades)}", end='', flush=True)
                
                # If we have an open position, check for exit conditions
                if current_position:
                    should_exit, exit_reason = self._check_exit_conditions(
                        current_position, current_price, current_bar
                    )
                    
                    if should_exit:
                        # Close position
                        trade = self._close_position(
                            current_position, position_info, 
                            current_price, current_time, exit_reason
                        )
                        self.trades.append(trade)
                        current_position = None
                        position_info = None
                        continue
                
                # If no position, check for entry signals
                if current_position is None:
                    # Get historical data for signal generation (limit to last 100 bars for speed)
                    lookback_start = max(0, i-100)
                    lookback_data = test_data.iloc[lookback_start:i+1].copy()
                    
                    # Generate signal
                    signal = self.strategy.generate_signal(lookback_data, current_price)
                    
                    if signal and signal['confidence'] >= 0.7:  # Minimum confidence threshold
                        # Calculate position size
                        position_info = self.risk_manager.calculate_position_size(
                            entry_price=signal['entry_price'],
                            stop_loss=signal['stop_loss'],
                            signal_confidence=signal['confidence'],
                            market_volatility=signal['market_volatility']
                        )
                        
                        if position_info['quantity'] > 0:
                            # Open position
                            current_position = {
                                'entry_time': current_time,
                                'entry_price': signal['entry_price'],
                                'direction': signal['signal'],
                                'stop_loss': signal['stop_loss'],
                                'target': signal['target'],
                                'confidence': signal['confidence']
                            }
                            
                            logger.debug(f"�� Position opened: {signal['signal']} at {current_price}")
                
                # Update equity curve (less frequently for performance)
                if i % 10 == 0:  # Update every 10 bars
                    self.equity_curve.append(self.current_capital)
                    self.timestamps.append(current_time)
            
            # Close any remaining open positions at the end
            if current_position:
                final_price = test_data.iloc[-1]['close']
                trade = self._close_position(
                    current_position, position_info, 
                    final_price, test_data.index[-1], "End of period"
                )
                self.trades.append(trade)
            
            # Final equity curve update
            self.equity_curve.append(self.current_capital)
            self.timestamps.append(test_data.index[-1])
            
            # Calculate backtest results
            result = self._calculate_backtest_results()
            
            # Show completion
            if show_progress:
                elapsed_time = time.time() - start_time
                print(f"\n✅ Backtest completed in {elapsed_time:.1f}s. Total trades: {result.total_trades}")
            
            return result
            
        except KeyboardInterrupt:
            logger.info("⚠️  Backtest interrupted by user")
            # Return partial results
            result = self._calculate_backtest_results()
            if show_progress:
                print(f"\n⚠️  Backtest interrupted. Partial results: {result.total_trades} trades")
            return result
        except Exception as e:
            logger.error(f"❌ Error running backtest: {e}")
            if show_progress:
                print(f"\n❌ Backtest failed: {e}")
            return self._create_empty_result()
    
    def _check_exit_conditions(self, position: Dict, current_price: float, 
                              current_bar) -> Tuple[bool, str]:
        """Check if position should be exited"""
        try:
            # Stop loss check
            if position['direction'] == 'BUY' and current_price <= position['stop_loss']:
                return True, "Stop Loss (Long)"
            elif position['direction'] == 'SELL' and current_price >= position['stop_loss']:
                return True, "Stop Loss (Short)"
            
            # Target check
            if position['direction'] == 'BUY' and current_price >= position['target']:
                return True, "Target Achieved (Long)"
            elif position['direction'] == 'SELL' and current_price <= position['target']:
                return True, "Target Achieved (Short)"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"❌ Error checking exit conditions: {e}")
            return False, ""
    
    def _close_position(self, position: Dict, position_info: Dict, 
                       exit_price: float, exit_time: datetime, 
                       exit_reason: str) -> Trade:
        """Close position and create trade record"""
        try:
            # Calculate P&L
            if position['direction'] == 'BUY':
                pnl = (exit_price - position['entry_price']) * position_info['quantity']
            else:  # SELL
                pnl = (position['entry_price'] - exit_price) * position_info['quantity']
            
            pnl_percentage = (pnl / (position['entry_price'] * position_info['quantity'])) * 100
            
            # Calculate holding period
            holding_period = int((exit_time - position['entry_time']).total_seconds() / 60)
            
            # Update capital
            self.current_capital += pnl
            
            trade = Trade(
                entry_time=position['entry_time'],
                exit_time=exit_time,
                entry_price=position['entry_price'],
                exit_price=exit_price,
                direction=position['direction'],
                quantity=position_info['quantity'],
                pnl=pnl,
                pnl_percentage=pnl_percentage,
                holding_period=holding_period,
                stop_loss=position['stop_loss'],
                target=position['target'],
                confidence=position['confidence']
            )
            
            logger.debug(f"📉 Position closed: {exit_reason}. P&L: ₹{pnl:.2f}")
            return trade
            
        except Exception as e:
            logger.error(f"❌ Error closing position: {e}")
            # Return a minimal trade record
            return Trade(
                entry_time=position['entry_time'],
                exit_time=exit_time,
                entry_price=position['entry_price'],
                exit_price=exit_price,
                direction=position['direction'],
                quantity=position_info.get('quantity', 75) if position_info else 75,
                pnl=0,
                pnl_percentage=0,
                holding_period=1,
                stop_loss=position['stop_loss'],
                target=position['target'],
                confidence=position['confidence']
            )
    
    def _calculate_backtest_results(self) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        try:
            if not self.trades:
                return self._create_empty_result()
            
            # Basic statistics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.pnl > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # P&L statistics
            pnls = [t.pnl for t in self.trades]
            total_pnl = sum(pnls)
            average_pnl = total_pnl / total_trades if total_trades > 0 else 0
            
            # Max drawdown
            max_drawdown = self._calculate_max_drawdown()
            
            # Sharpe ratio (assuming risk-free rate of 0)
            returns = [t.pnl_percentage for t in self.trades]
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            
            # Profit factor
            gross_profits = sum([t.pnl for t in self.trades if t.pnl > 0])
            gross_losses = abs(sum([t.pnl for t in self.trades if t.pnl < 0]))
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
            
            # Consecutive wins/losses
            max_consecutive_wins = self._calculate_max_consecutive(pnls, positive=True)
            max_consecutive_losses = self._calculate_max_consecutive(pnls, positive=False)
            
            # Average holding period
            avg_holding_period = np.mean([t.holding_period for t in self.trades]) if self.trades else 0
            
            return BacktestResult(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                average_pnl=average_pnl,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                profit_factor=profit_factor,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                average_holding_period=avg_holding_period,
                trades=self.trades.copy()
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating backtest results: {e}")
            return self._create_empty_result()
    
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
            logger.error(f"❌ Error calculating max drawdown: {e}")
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
            # Annualize (assuming 252 trading days, but we're using trades, so approximate)
            annualized_sharpe = sharpe * np.sqrt(min(len(returns), 252))
            
            return annualized_sharpe
        except Exception as e:
            logger.error(f"❌ Error calculating Sharpe ratio: {e}")
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
            logger.error(f"❌ Error calculating consecutive trades: {e}")
            return 0
    
    def _create_empty_result(self) -> BacktestResult:
        """Create empty backtest result"""
        return BacktestResult(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            average_pnl=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            average_holding_period=0.0,
            trades=[]
        )
    
    def get_equity_curve(self) -> Tuple[List[datetime], List[float]]:
        """Get equity curve data for plotting"""
        return self.timestamps.copy(), self.equity_curve.copy()

# Example usage
if __name__ == "__main__":
    # This would be used in your backtesting application
    print("Backtest Engine ready for use!")
    print("Import and use: from src.backtesting.backtest_engine import BacktestEngine")
