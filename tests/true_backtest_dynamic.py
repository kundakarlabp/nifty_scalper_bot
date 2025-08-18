"""
High-fidelity backtesting engine for the Nifty Scalper Bot.

This script simulates the bot's behavior on historical data with a high
degree of parity to the live trading engine. It uses the same refactored
components as the live application, but with a `BacktestCsvSource` to
feed in historical data bar-by-bar.

Key Features:
- **Parity with Live Logic**: Uses the same Strategy, PositionSizer, and
  TradingSession classes as the live bot.
- **No Look-ahead Bias**: The `BacktestCsvSource` provides data one bar at a
  time, simulating real-time data flow.
- **Realistic Order Simulation**: Simulates Stop-Loss and Take-Profit orders
  on a bar-by-bar basis.
- **Comprehensive Reporting**: Generates detailed performance metrics and
  saves them to the /reports directory.

Usage:
    python3 tests/true_backtest_dynamic.py
"""

import logging
from pathlib import Path
import sys
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.backtesting.data_source import BacktestCsvSource
from src.execution.order_executor import OrderExecutor
from src.risk.position_sizing import PositionSizer
from src.risk.session import TradingSession, Trade
from src.strategies.scalping_strategy import EnhancedScalpingStrategy

# Configure logging for the backtest
logging.basicConfig(level=settings.log_level, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


class BacktestRunner:
    """Orchestrates the backtesting process."""
    def __init__(self, csv_filepath: Path):
        logger.info("Initializing backtest components...")
        self.data_source = BacktestCsvSource(csv_filepath, symbol="NIFTY_SPOT_PROXY")
        self.strategy = EnhancedScalpingStrategy(settings.strategy)
        self.sizer = PositionSizer(settings.risk)
        self.session = TradingSession(settings.risk, starting_equity=100_000.0)
        self.active_trade: Trade | None = None

    def run(self):
        logger.info("Starting backtest...")
        while self.data_source.tick():
            self.tick()
        logger.info("Backtest finished.")
        self.generate_report()

    def tick(self):
        current_dt = self.data_source.current_datetime
        current_bar = self.data_source._df.iloc[self.data_source._current_index]

        if self.active_trade:
            self._check_exit(current_bar)

        if self.active_trade is None:
            if self.session.check_risk_limits() is None:
                self._check_entry(current_dt, current_bar)

    def _check_exit(self, current_bar):
        trade = self.active_trade
        exit_price = None
        if trade.direction == "BUY":
            if current_bar["low"] <= trade.stop_loss:
                exit_price = trade.stop_loss
            elif current_bar["high"] >= trade.target:
                exit_price = trade.target
        elif trade.direction == "SELL":
             if current_bar["high"] >= trade.stop_loss:
                exit_price = trade.stop_loss
             elif current_bar["low"] <= trade.target:
                exit_price = trade.target

        if exit_price:
            # Simulate slippage
            slippage = settings.executor.tick_size * 2
            if trade.direction == "BUY":
                exit_price -= slippage
            else:
                exit_price += slippage

            self.session.finalize_trade(trade.order_id, exit_price)
            self.active_trade = None
            logger.info(f"{current_bar.name} - Exit trade at {exit_price:.2f}")

    def _check_entry(self, current_dt, current_bar):
        df_history = self.data_source.fetch_ohlc(0, datetime.min, current_dt, "minute")
        current_price = current_bar["close"]
        signal = self.strategy.generate_signal(df=df_history, current_price=current_price)

        if signal:
            print(f"INSIDE SIGNAL BLOCK: {signal}")
            logger.info(f"Signal received at {current_dt}: {signal}")
            quantity = self.sizer.calculate_quantity(
                session=self.session,
                entry_price=current_price,
                stop_loss_price=signal["stop_loss"],
                lot_size=settings.executor.nifty_lot_size,
            )
            if quantity > 0:
                trade = Trade(
                    symbol="BACKTEST_INSTRUMENT", direction=signal["signal"],
                    entry_price=current_price, quantity=quantity,
                    order_id=f"order_{current_dt.isoformat()}", atr=signal["market_volatility"]
                )
                trade.stop_loss = signal["stop_loss"]
                trade.target = signal["target"]
                self.session.add_trade(trade)
                self.active_trade = trade
                logger.info(f"{current_dt} - New Trade: {trade.direction} {trade.quantity} @ {trade.entry_price:.2f}")


    def generate_report(self):
        """Generates a detailed performance report using quantstats."""
        logger.info("Generating backtest report...")

        # Ensure the reports directory exists
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)

        if not self.session.trade_history:
            logger.warning("No trades were executed. Cannot generate report.")
            return

        # Create a returns series for quantstats
        equity_curve = pd.Series(
            [t.exit_price for t in self.session.trade_history],
            index=pd.to_datetime([t.exit_time for t in self.session.trade_history])
        ).pct_change().fillna(0)

        # Generate the report
        report_path = reports_dir / "backtest_report.html"
        try:
            import quantstats as qs
            qs.reports.html(equity_curve, output=str(report_path), title="Nifty Scalper Backtest")
            logger.info(f"Successfully generated backtest report: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate quantstats report: {e}", exc_info=True)
            # Fallback to simple text report
            self._generate_simple_report()

    def _generate_simple_report(self):
        """A basic text-based report as a fallback."""
        print("\n" + "="*50)
        print("BACKTEST SUMMARY REPORT (SIMPLE)")
        print("="*50)

        total_trades = len(self.session.trade_history)
        pnl = self.session.daily_pnl
        print(f"Total Trades: {total_trades}")
        print(f"Final Equity: {self.session.current_equity:,.2f}")
        print(f"Net P&L: {pnl:,.2f}")
        self._generate_csv_report()

    def _generate_csv_report(self):
        """Saves the trade history to a CSV file."""
        reports_dir = project_root / "reports"
        csv_path = reports_dir / "trade_history.csv"

        if not self.session.trade_history:
            return

        trade_data = [
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "symbol": t.symbol,
                "direction": t.direction,
                "quantity": t.quantity,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "fees": t.fees,
                "net_pnl": t.net_pnl,
            }
            for t in self.session.trade_history
        ]

        df = pd.DataFrame(trade_data)
        df.to_csv(csv_path, index=False)
        logger.info(f"Trade history saved to {csv_path}")

def main():
    """Entry point for the backtest script."""
    # Use the nifty_ohlc.csv file provided in the repo
    csv_file = project_root / "src" / "data" / "nifty_ohlc.csv"
    if not csv_file.exists():
        logger.error(f"Data file not found: {csv_file}")
        sys.exit(1)

    runner = BacktestRunner(csv_filepath=csv_file)
    runner.run()


if __name__ == "__main__":
    main()
