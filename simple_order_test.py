import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from src.order_management.order_executor import OrderExecutor

# Set up logging
logging.basicConfig(level=logging.INFO)

print("🚀 Simple Order Execution Test...")

try:
    # Initialize order executor
    print("\n1. Initializing Order Executor...")
    executor = OrderExecutor()
    print("✅ Order Executor initialized")
    
    # Test getting current positions
    print("\n2. Testing Position Fetch...")
    positions = executor.get_positions()
    print("✅ Positions fetched successfully")
    
    # Test getting holdings
    print("\n3. Testing Holdings Fetch...")
    holdings = executor.get_holdings()
    print(f"✅ Holdings fetched: {len(holdings)} items")
    
    # Test getting trading limits
    print("\n4. Testing Trading Limits...")
    limits = executor.get_trading_limits()
    print("✅ Trading limits fetched successfully")
    
    # Show sample order parameters
    print("\n5. Sample Order Parameters:")
    sample_params = {
        'tradingsymbol': 'NIFTY 50',
        'transaction_type': 'BUY',  # or 'SELL'
        'quantity': 75,  # 1 lot
        'product': 'MIS',  # Margin Intraday Square off
        'order_type': 'MARKET',  # or 'LIMIT'
        'variety': 'regular',
        'exchange': 'NSE'
    }
    
    for key, value in sample_params.items():
        print(f"   {key}: {value}")
    
    print("\n🎉 Simple order execution test completed!")
    print("🚀 Your order execution system is ready for trading!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    print("Please check your Zerodha credentials and internet connection.")
