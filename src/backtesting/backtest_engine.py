import pandas as pd
import logging
from src.strategies.scalping_strategy import DynamicScalpingStrategy
from src.risk.position_sizing import PositionSizing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BacktestEngine:
    """
    A simple bar-by-bar backtesting engine.
    """

    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000.0):
        self.data = data
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []  # List of dicts: [{'entry_price': ..., 'exit_price': ..., 'pnl': ...}]
        self.strategy = DynamicScalpingStrategy()
        self.risk_manager = PositionSizing(account_size=initial_capital)

    def run(self):
        try:
            logger.info("🔁 Starting Backtest...")

            buffer = []  # Rolling window of data for indicators
            for idx, (timestamp, row) in enumerate(self.data.iterrows()):
                buffer.append(row)
                df_buffer = pd.DataFrame(buffer)

                # Require minimum 50 bars to start generating signals
                if len(df_buffer) < 50:
                    continue

                # Generate signal
                signal = self.strategy.generate_signal(df_buffer)

                if signal:
                    entry_price = row['close']
                    stop_loss = entry_price - signal.get("sl", 5)
                    target_price = entry_price + signal.get("tp", 10)

                    position_size = self.risk_manager.calculate_position_size(entry_price, stop_loss)
                    if position_size == 0:
                        continue  # Skip if capital/risk constraint breached

                    # Simulate immediate exit on next bar for now (replace with real SL/TP logic)
                    if idx + 1 < len(self.data):
                        next_close = self.data.iloc[idx + 1]['close']
                        pnl = (next_close - entry_price) * position_size
                        self.current_capital += pnl

                        self.positions.append({
                            'timestamp': timestamp,
                            'entry_price': entry_price,
                            'exit_price': next_close,
                            'position_size': position_size,
                            'pnl': pnl
                        })

                        logger.info(f"📈 Trade Executed @ {timestamp} | Entry: {entry_price:.2f} | Exit: {next_close:.2f} | PnL: {pnl:.2f}")

            self._generate_report()

        except Exception as e:
            logger.error(f"❌ Error during backtest: {e}")

    def _generate_report(self):
        total_trades = len(self.positions)
        total_pnl = sum(p['pnl'] for p in self.positions)
        win_trades = len([p for p in self.positions if p['pnl'] > 0])
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0

        logger.info("✅ --- Backtest Report ---")
        logger.info(f"Initial Capital: ₹{self.initial_capital:.2f}")
        logger.info(f"Final Capital: ₹{self.current_capital:.2f}")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winning Trades: {win_trades} ({win_rate:.2f}%)")
        logger.info(f"Net PnL: ₹{total_pnl:.2f}")

# Example usage (you can move this to a CLI runner later):
# if __name__ == "__main__":
#     df = pd.read_csv("data/your_historical_data.csv", parse_dates=True, index_col="timestamp")
#     engine = BacktestEngine(df)
#     engine.run()
