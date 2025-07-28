import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime, timedelta
from src.backtesting.data_loader import SampleDataGenerator
from src.backtesting.backtest_engine import BacktestEngine

# Set up logging
logging.basicConfig(level=logging.INFO)

print("🚀 Quick Backtest Demo")
print("=" * 40)

# Generate small sample dataset (faster)
print("📊 Generating sample data (2 days)...")
generator = SampleDataGenerator()
sample_data = generator.generate_sample_nifty_data(days=2, interval="5min")

print(f"✅ Generated {len(sample_data)} data points")

# Initialize backtest engine
print("🔧 Initializing backtest engine...")
engine = BacktestEngine(initial_capital=100000.0)

# Run quick backtest
print("🏃 Running quick backtest...")
print("💡 Press Ctrl+C to stop early if needed")
print()

try:
    result = engine.run_backtest(sample_data, show_progress=True)
    
    print("\n" + "="*50)
    print("📈 BACKTEST RESULTS")
    print("="*50)
    print(f"Total Trades:     {result.total_trades}")
    print(f"Win Rate:         {result.win_rate:.1%}")
    print(f"Total P&L:        ₹{result.total_pnl:,.2f}")
    print(f"Average P&L:      ₹{result.average_pnl:,.2f}")
    print(f"Max Drawdown:     {result.max_drawdown:.1%}")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    
    if result.trades:
        print(f"\n📋 Sample Trades:")
        for i, trade in enumerate(result.trades[:3]):  # Show first 3 trades
            print(f"  Trade {i+1}: {trade.direction} ₹{trade.entry_price:.2f} → ₹{trade.exit_price:.2f} (₹{trade.pnl:+.2f})")
    
    print("\n✅ Quick backtest completed!")
    
except KeyboardInterrupt:
    print("\n⚠️  Backtest interrupted by user")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n🔧 To run full backtests:")
print("   python -m src.backtesting.backtest_cli --days 7")
print("   python -m src.backtesting.backtest_cli --help  # for more options")
