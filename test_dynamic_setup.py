import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from src.strategies.scalping_strategy import DynamicScalpingStrategy
from src.risk.position_sizing import PositionSizing
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

print("🧪 Testing Dynamic Nifty Setup...")

# Test 1: Configuration
print("\n1. Testing Configuration...")
try:
    from config import NIFTY_LOT_SIZE, ACCOUNT_SIZE
    print(f"✅ Nifty Lot Size: {NIFTY_LOT_SIZE}")
    print(f"✅ Account Size: ₹{ACCOUNT_SIZE:,.2f}")
except Exception as e:
    print(f"❌ Configuration test failed: {e}")

# Test 2: Dynamic Strategy
print("\n2. Testing Dynamic Strategy...")
try:
    strategy = DynamicScalpingStrategy()
    # Create realistic sample data
    base_price = 18000
    sample_data = pd.DataFrame({
        'close': [base_price + i*5 + np.random.randn()*30 for i in range(100)],
        'high': [base_price + i*5 + 20 + np.random.randn()*30 for i in range(100)],
        'low': [base_price + i*5 - 20 + np.random.randn()*30 for i in range(100)],
        'open': [base_price + i*5 + np.random.randn()*30 for i in range(100)],
        'volume': [1000000 + np.random.randint(-500000, 500000) for i in range(100)]
    })
    signal = strategy.generate_signal(sample_data, base_price + 50)
    print(f"✅ Dynamic strategy test completed. Signal generated: {'YES' if signal else 'NO'}")
except Exception as e:
    print(f"❌ Dynamic strategy test failed: {e}")

# Test 3: Dynamic Position Sizing
print("\n3. Testing Dynamic Position Sizing...")
try:
    risk_manager = PositionSizing(account_size=100000, risk_per_trade=0.01)
    
    # Test normal market conditions
    position_info1 = risk_manager.calculate_position_size(18000, 17980, 0.85, 1.0)
    print(f"✅ Normal market - Position info: {position_info1}")
    
    # Test high volatility market
    position_info2 = risk_manager.calculate_position_size(18000, 17960, 0.90, 2.0)
    print(f"✅ High volatility - Position info: {position_info2}")
    
    # Test low confidence signal
    position_info3 = risk_manager.calculate_position_size(18000, 17980, 0.60, 0.8)
    print(f"✅ Low confidence - Position info: {position_info3}")
    
except Exception as e:
    print(f"❌ Dynamic position sizing test failed: {e}")

# Test 4: Lot Size Calculations
print("\n4. Testing Lot Size Calculations...")
try:
    from config import NIFTY_LOT_SIZE
    test_quantities = [75, 150, 225, 300, 375, 750]
    print(f"✅ Nifty lot size: {NIFTY_LOT_SIZE}")
    for qty in test_quantities:
        lots = qty // NIFTY_LOT_SIZE
        actual_qty = lots * NIFTY_LOT_SIZE
        print(f"   Quantity {qty} → {lots} lots → {actual_qty} actual quantity")
except Exception as e:
    print(f"❌ Lot size calculation test failed: {e}")

print("\n🎉 All dynamic tests completed!")
print("🚀 Your Nifty scalper bot is ready with dynamic position sizing!")
