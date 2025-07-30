# src/backtesting/backtest_engine.py
import pandas as pd
from src.strategies.scalping_strategy import DynamicScalpingStrategy
from src.risk.position_sizing import PositionSizing
import logging

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000.0, strategy=None, risk_manager=None):
        self.data = data
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []  # Store executed trades
        self.strategy = strategy or DynamicScalpingStrategy()
        self.risk_manager = risk_manager or PositionSizing(account_size=initial_capital)

    def run(self):
        try:
            logger.info("Starting backtest...")
            for timestamp, row in self.data.iterrows():
                signal = self.strategy.generate_signal(self.data.loc[:timestamp])  # Pass data up to current point

                if signal and signal != 0:
                    price = row['close']
                    size = self.risk_manager.calculate_position_size(price)

                    # Simulate execution
                    trade = {
                        'timestamp': timestamp,
                        'signal': signal,
                        'price': price,
                        'size': size,
                        'pnl': 0  # Will be updated on close
                    }

                    if signal == 1:  # Buy
                        cost = price * size
                        if self.current_capital >= cost:
                            self.current_capital -= cost
                            self.positions.append(trade)
                            logger.info(f"[{timestamp}] BUY {size} @ {price}")
                    elif signal == -1:  # Sell
                        if self.positions:
                            open_trade = self.positions[-1]
                            pnl = (price - open_trade['price']) * open_trade['size']
                            self.current_capital += price * open_trade['size'] + pnl
                            open_trade['exit_price'] = price
                            open_trade['pnl'] = pnl
                            logger.info(f"[{timestamp}] SELL {open_trade['size']} @ {price} | PnL: {pnl:.2f}")

            self._generate_report()
        except Exception as e:
            logger.error(f"Error during backtest run: {e}")

    def _generate_report(self):
        total_pnl = sum(trade['pnl'] for trade in self.positions if 'pnl' in trade)
        logger.info("--- Backtest Report ---")
        logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
        logger.info(f"Final Capital: ${self.current_capital:.2f}")
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        logger.info(f"Total Trades: {len(self.positions)}")
