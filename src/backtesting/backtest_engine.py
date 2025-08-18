import pandas as pd
import logging
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.risk.position_sizing import PositionSizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class BacktestEngine:
    """
    A simple bar-by-bar backtesting engine using EnhancedScalpingStrategy.
    """

    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000.0):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategy = EnhancedScalpingStrategy  # placeholder; set at run() time
        self.positions = []
        self.risk_manager = PositionSizer(account_size=self.initial_capital)

    def run(self) -> pd.DataFrame:
        """
        Executes a naive bar-by-bar backtest:
        - Calls strategy.generate_signal(df_window, current_price)
        - Enters at bar close; exits on next bar close (demo)
        """
        if self.data.empty:
            logger.error("No data to backtest.")
            return pd.DataFrame()

        self.data = self.data.sort_index()
        df_buffer = pd.DataFrame()

        for idx, (timestamp, row) in enumerate(self.data.iterrows()):
            df_buffer = pd.concat([df_buffer, row.to_frame().T]).tail(500)

            # Example: calculate signal using the latest close as current_price
            try:
                current_price = row['close']
            except Exception:
                logger.warning("Row without 'close' at %s; skipping", timestamp)
                continue

            try:
                # NOTE: you must construct the strategy instance with your StrategyConfig outside
                signal = self.strategy.generate_signal(df_buffer, current_price)  # type: ignore[attr-defined]
            except Exception as e:
                logger.error("Strategy error at %s: %s", timestamp, e, exc_info=True)
                continue

            if not signal:
                continue

            # Simplified sizing: assume stop loss distance is ATR or fixed points
            entry_price = current_price
            stop_loss = signal.get("stop_loss") or (entry_price - 20 if signal["direction"] == "BUY" else entry_price + 20)
            sl_points = abs(entry_price - stop_loss)
            position_size = self.risk_manager.calculate_position_size(entry_price, sl_points)

            if position_size <= 0:
                continue

            # Simplified exit logic: next bar close
            if idx + 1 < len(self.data):
                next_close = self.data.iloc[idx + 1]['close']
                pnl = (next_close - entry_price) * position_size if signal["direction"] == "BUY" else (entry_price - next_close) * position_size
                self.current_capital += pnl

                self.positions.append({
                    'timestamp': timestamp,
                    'direction': signal["direction"],
                    'entry_price': entry_price,
                    'exit_price': next_close,
                    'position_size': position_size,
                    'pnl': pnl
                })

                logger.info(f"📈 Trade @ {timestamp} | {signal['direction']} | Entry: {entry_price:.2f} | Exit: {next_close:.2f} | PnL: {pnl:.2f}")

        return pd.DataFrame(self.positions)
