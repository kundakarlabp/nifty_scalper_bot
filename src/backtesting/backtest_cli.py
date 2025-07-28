import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import json
from datetime import datetime, timedelta
import pandas as pd
from src.backtesting.data_loader import HistoricalDataLoader, SampleDataGenerator
from src.backtesting.backtest_engine import BacktestEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class BacktestCLI:
    def __init__(self):
        self.data_loader = HistoricalDataLoader()
        self.sample_generator = SampleDataGenerator()
    
    def run_backtest(self, args):
        """Run backtest based on command line arguments"""
        try:
            logger.info("🚀 Starting backtest...")
            
            # Load data
            data = self._load_data(args)
            if data is None or data.empty:
                logger.error("❌ Failed to load data for backtesting")
                return
            
            # Initialize backtest engine
            engine = BacktestEngine(initial_capital=args.capital)
            
            # Parse date range
            start_date = None
            end_date = None
            if args.start_date:
                start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            if args.end_date:
                end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
            
            # Run backtest
            logger.info("🏃 Running backtest...")
            result = engine.run_backtest(data, start_date, end_date)
            
            # Display results
            self._display_results(result, args.format)
            
            # Save results if requested
            if args.output:
                self._save_results(result, args.output)
            
        except Exception as e:
            logger.error(f"❌ Error running backtest: {e}")
    
    def _load_data(self, args):
        """Load data based on arguments"""
        try:
            if args.instrument == "NIFTY":
                if self.data_loader.kite:
                    logger.info("📥 Loading Nifty 50 historical data from Zerodha...")
                    return self.data_loader.load_nifty_historical_data(
                        days=args.days, 
                        interval=args.interval
                    )
                else:
                    logger.info("📥 Generating sample Nifty 50 data...")
                    return self.sample_generator.generate_sample_nifty_data(
                        days=args.days, 
                        interval=args.interval
                    )
            else:
                logger.error(f"❌ Unsupported instrument: {args.instrument}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return None
    
    def _display_results(self, result, format_type="table"):
        """Display backtest results"""
        try:
            if format_type == "json":
                self._display_json_results(result)
            else:
                self._display_table_results(result)
        except Exception as e:
            logger.error(f"❌ Error displaying results: {e}")
    
    def _display_table_results(self, result):
        """Display results in table format"""
        print("\n" + "="*60)
        print("📊 BACKTEST RESULTS")
        print("="*60)
        
        print(f"📈 Total Trades:     {result.total_trades}")
        print(f"✅ Winning Trades:   {result.winning_trades}")
        print(f"❌ Losing Trades:    {result.losing_trades}")
        print(f"🎯 Win Rate:         {result.win_rate:.2%}")
        print(f"💰 Total P&L:        ₹{result.total_pnl:,.2f}")
        print(f"📊 Average P&L:      ₹{result.average_pnl:,.2f}")
        print(f"📉 Max Drawdown:     {result.max_drawdown:.2%}")
        print(f"⚡ Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        print(f"💰 Profit Factor:    {result.profit_factor:.2f}")
        print(f"🏆 Max Consecutive Wins:  {result.max_consecutive_wins}")
        print(f"😞 Max Consecutive Losses: {result.max_consecutive_losses}")
        print(f"⏱️  Avg Holding Period:   {result.average_holding_period:.1f} minutes")
        
        print("\n" + "="*60)
        
        if result.trades:
            print("\n📋 RECENT TRADES (last 5):")
            print("-" * 80)
            print(f"{'Date':<12} {'Dir':<4} {'Entry':<8} {'Exit':<8} {'P&L':<10} {'Holding':<8}")
            print("-" * 80)
            
            recent_trades = result.trades[-5:] if len(result.trades) >= 5 else result.trades
            for trade in recent_trades:
                print(f"{trade.entry_time.strftime('%m-%d %H:%M'):<12} "
                      f"{trade.direction:<4} "
                      f"{trade.entry_price:<8.2f} "
                      f"{trade.exit_price:<8.2f} "
                      f"₹{trade.pnl:<9.2f} "
                      f"{trade.holding_period:<8}min")
    
    def _display_json_results(self, result):
        """Display results in JSON format"""
        try:
            # Convert result to dictionary
            result_dict = {
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'win_rate': result.win_rate,
                'total_pnl': result.total_pnl,
                'average_pnl': result.average_pnl,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'profit_factor': result.profit_factor,
                'max_consecutive_wins': result.max_consecutive_wins,
                'max_consecutive_losses': result.max_consecutive_losses,
                'average_holding_period': result.average_holding_period,
                'trades_count': len(result.trades)
            }
            
            print(json.dumps(result_dict, indent=2))
        except Exception as e:
            logger.error(f"❌ Error displaying JSON results: {e}")
    
    def _save_results(self, result, output_file):
        """Save results to file"""
        try:
            # Save detailed results to CSV
            if result.trades:
                trades_df = pd.DataFrame([{
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'pnl_percentage': trade.pnl_percentage,
                    'holding_period': trade.holding_period,
                    'stop_loss': trade.stop_loss,
                    'target': trade.target,
                    'confidence': trade.confidence
                } for trade in result.trades])
                
                trades_df.to_csv(output_file, index=False)
                logger.info(f"✅ Results saved to {output_file}")
            else:
                logger.warning("⚠️  No trades to save")
                
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")

def main():
    """Main entry point for backtesting CLI"""
    parser = argparse.ArgumentParser(description='Nifty Scalper Backtesting Tool')
    parser.add_argument('--instrument', default='NIFTY', 
                       choices=['NIFTY'], help='Instrument to backtest')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days of historical data to use')
    parser.add_argument('--interval', default='minute',
                       choices=['minute', '5min', '15min', '1H', '1D'],
                       help='Data interval')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Initial capital for backtest')
    parser.add_argument('--format', default='table',
                       choices=['table', 'json'],
                       help='Output format')
    parser.add_argument('--output', help='Output file for trade details (CSV)')
    
    args = parser.parse_args()
    
    # Run backtest
    cli = BacktestCLI()
    cli.run_backtest(args)

if __name__ == "__main__":
    main()
