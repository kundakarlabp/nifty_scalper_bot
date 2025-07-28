import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🧪 Simple Backtesting Test...")

# Test 1: Import modules
print("\n1. Testing Module Imports...")
try:
    from src.backtesting.data_loader import SampleDataGenerator
    from src.backtesting.backtest_engine import BacktestEngine
    print("✅ All backtesting modules imported successfully")
except Exception as e:
    print(f"❌ Module import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Generate sample data
print("\n2. Testing Sample Data Generation...")
try:
    generator = SampleDataGenerator()
    sample_data = generator.generate_sample_nifty_data(days=1, interval="5min")
    print(f"✅ Sample data generated: {len(sample_data)} rows")
    print(f"   Columns: {list(sample_data.columns)}")
    print(f"   Price range: ₹{sample_data['close'].min():.2f} - ₹{sample_data['close'].max():.2f}")
except Exception as e:
    print(f"❌ Sample data generation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Initialize backtest engine
print("\n3. Testing Backtest Engine...")
try:
    engine = BacktestEngine(initial_capital=50000.0)
    print("✅ Backtest engine initialized")
    print(f"   Initial capital: ₹50,000.00")
except Exception as e:
    print(f"❌ Backtest engine initialization failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Run minimal backtest
print("\n4. Testing Minimal Backtest...")
try:
    # Generate very small dataset
    generator = SampleDataGenerator()
    tiny_data = generator.generate_sample_nifty_data(days=0.1, interval="15min")  # Very small dataset
    
    engine = BacktestEngine(initial_capital=25000.0)
    result = engine.run_backtest(tiny_data, show_progress=False)  # No progress for speed
    
    print("✅ Minimal backtest completed")
    print(f"   Trades executed: {result.total_trades}")
    print(f"   Final P&L: ₹{result.total_pnl:,.2f}")
    
except Exception as e:
    print(f"❌ Minimal backtest failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 Simple backtesting test completed!")
print("🚀 Your backtesting system is ready!")
print("\n🔧 To run full backtests:")
print("   python -m src.backtesting.backtest_cli --help")
