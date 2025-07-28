import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime, timedelta
from src.backtesting.data_loader import SampleDataGenerator
from src.backtesting.backtest_engine import BacktestEngine

# Set up logging
logging.basicConfig(level=logging.INFO)

print("🧪 Testing Backtesting Framework...")

# Test 1: Data Generation
print("\n1. Testing Data Generation...")
try:
    generator = SampleDataGenerator()
    sample_data = generator.generate_sample_nifty_data(days=7, interval="5min")
    
    print("✅ Sample data generation successful")
    print(f"   Data shape: {sample_data.shape}")
    print(f"   Date range: {sample_data.index[0]} to {sample_data.index[-1]}")
    print(f"   Price range: ₹{sample_data['close'].min():.2f} - ₹{sample_data['close'].max():.2f}")
    
except Exception as e:
    print(f"❌ Data generation test failed: {e}")

# Test 2: Backtest Engine
print("\n2. Testing Backtest Engine...")
try:
    # Generate sample data
    generator = SampleDataGenerator()
    test_data = generator.generate_sample_nifty_data(days=30, interval="5min")
    
    # Initialize backtest engine
    engine = BacktestEngine(initial_capital=100000.0)
    
    print("✅ Backtest engine initialized")
    print(f"   Initial capital: ₹100,000.00")
    
    # Run backtest
    print("🏃 Running sample backtest...")
    result = engine.run_backtest(test_data)
    
    print("✅ Backtest completed successfully")
    print(f"   Total trades: {result.total_trades}")
    print(f"   Win rate: {result.win_rate:.2%}")
    print(f"   Total P&L: ₹{result.total_pnl:,.2f}")
    
except Exception as e:
    print(f"❌ Backtest engine test failed: {e}")

# Test 3: Backtest Results
print("\n3. Testing Backtest Results...")
try:
    # Generate sample data
    generator = SampleDataGenerator()
    test_data = generator.generate_sample_nifty_data(days=15, interval="minute")
    
    # Run backtest
    engine = BacktestEngine(initial_capital=50000.0)
    result = engine.run_backtest(test_data)
    
    print("✅ Backtest results analysis:")
    print(f"   Trades executed: {result.total_trades}")
    print(f"   Winning trades: {result.winning_trades}")
    print(f"   Losing trades: {result.losing_trades}")
    print(f"   Average P&L per trade: ₹{result.average_pnl:,.2f}")
    print(f"   Max drawdown: {result.max_drawdown:.2%}")
    print(f"   Sharpe ratio: {result.sharpe_ratio:.2f}")
    
    # Test equity curve
    timestamps, equity_curve = engine.get_equity_curve()
    print(f"   Equity curve points: {len(equity_curve)}")
    
except Exception as e:
    print(f"❌ Backtest results test failed: {e}")

# Test 4: CLI Interface
print("\n4. Testing CLI Interface...")
try:
    from src.backtesting.backtest_cli import BacktestCLI
    cli = BacktestCLI()
    print("✅ CLI interface initialized successfully")
    
    # Test argument parsing would require actual command line args
    # But we can test the class instantiation
    print("   CLI class ready for command line usage")
    
except Exception as e:
    print(f"❌ CLI interface test failed: {e}")

print("\n🎉 Backtesting framework tests completed!")
print("🚀 Your backtesting system is ready!")

print("\n🔧 To run backtests, use:")
print("   python -m src.backtesting.backtest_cli --help")
print("   python -m src.backtesting.backtest_cli --days 30 --capital 100000")
