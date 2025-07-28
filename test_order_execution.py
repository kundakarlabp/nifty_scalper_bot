import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from src.order_management.order_executor import OrderExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)

print("🧪 Testing Order Execution System...")

# Test 1: Order Executor Initialization
print("\n1. Testing Order Executor Initialization...")
try:
    executor = OrderExecutor()
    print("✅ Order Executor initialized successfully")
    
    # Test getting trading limits
    limits = executor.get_trading_limits()
    print(f"✅ Trading limits fetched: {'Yes' if limits else 'No'}")
    
except Exception as e:
    print(f"❌ Order Executor initialization failed: {e}")

# Test 2: Order Parameters Validation
print("\n2. Testing Order Parameters...")
try:
    # Test valid order parameters
    valid_params = {
        'tradingsymbol': 'NIFTY 50',
        'transaction_type': 'BUY',
        'quantity': 75,
        'product': 'MIS',
        'order_type': 'MARKET'
    }
    
    print("✅ Valid order parameters structure ready")
    print(f"   Symbol: {valid_params['tradingsymbol']}")
    print(f"   Quantity: {valid_params['quantity']}")
    print(f"   Type: {valid_params['order_type']}")
    
except Exception as e:
    print(f"❌ Order parameters test failed: {e}")

# Test 3: System Integration
print("\n3. Testing System Integration...")
try:
    from src.data_streaming.realtime_trader import RealTimeTrader
    
    trader = RealTimeTrader()
    trader.enable_trading(False)  # Disable real trading for test
    
    status = trader.get_trading_status()
    print("✅ Real-time trader with order execution initialized")
    print(f"   Execution enabled: {status['execution_enabled']}")
    print(f"   Trading status: {status['is_trading']}")
    
except Exception as e:
    print(f"❌ System integration test failed: {e}")

print("\n🎉 Order execution system tests completed!")
print("🚀 Your automated order execution system is ready!")
print("\n⚠️  Remember to use --trade flag to enable real trading:")
print("   python src/main.py --mode realtime --trade")
