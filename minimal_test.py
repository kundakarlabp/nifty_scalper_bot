import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🚀 Minimal Backtesting Test...")

try:
    # Test 1: Basic imports
    print("1. Testing imports...")
    from src.backtesting.data_loader import SampleDataGenerator
    from src.backtesting.backtest_engine import BacktestEngine
    print("   ✅ Imports successful")
    
    # Test 2: Generate tiny dataset
    print("2. Generating sample data...")
    generator = SampleDataGenerator()
    data = generator.generate_sample_nifty_data(days=0.05, interval="30min")  # Very small
    print(f"   ✅ Generated {len(data)} records")
    
    # Test 3: Run backtest
    print("3. Running backtest...")
    engine = BacktestEngine(initial_capital=10000.0)
    result = engine.run_backtest(data, show_progress=False)
    print(f"   ✅ Backtest completed: {result.total_trades} trades")
    
    print("\n🎉 All tests passed!")
    print("🚀 Backtesting system is working!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
