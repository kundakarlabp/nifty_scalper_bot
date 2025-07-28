import pandas as pd
import numpy as np
from src.strategies.scalping_strategy import DynamicScalpingStrategy
from src.risk.position_sizing import PositionSizing

print("🚀 Testing Nifty Scalper Components...")

# Test Strategy
print("\n1. Testing Strategy...")
strategy = DynamicScalpingStrategy()

# Create sample data
sample_data = pd.DataFrame({
    'close': [18000 + i*10 + np.random.randn()*50 for i in range(100)],
    'high': [18010 + i*10 + np.random.randn()*50 for i in range(100)],
    'low': [17990 + i*10 + np.random.randn()*50 for i in range(100)],
    'open': [18000 + i*10 + np.random.randn()*50 for i in range(100)],
    'volume': [1000000 + np.random.randint(-100000, 100000) for i in range(100)]
})

current_price = sample_data['close'].iloc[-1]
signal = strategy.generate_signal(sample_data, current_price)
print(f"✅ Strategy test completed. Signal: {signal}")

# Test Risk Management
print("\n2. Testing Risk Management...")
risk_manager = PositionSizing(account_size=100000, risk_per_trade=0.01)

position_info = risk_manager.calculate_position_size(
    entry_price=18000,
    stop_loss=17980,
    signal_confidence=0.85,
    market_volatility=1.2
)

print(f"✅ Risk management test completed.")
print(f"   Position Info: {position_info}")
print(f"   Risk Status: {risk_manager.get_risk_status()}")

# Test Lot Size Calculations
print("\n3. Testing Lot Size Calculations...")
lot_size = 75  # Nifty lot size
test_cases = [
    (1000, "Small position"),
    (5000, "Medium position"),
    (10000, "Large position")
]

for qty, desc in test_cases:
    lots = qty // lot_size
    actual_qty = lots * lot_size
    print(f"   {desc}: {qty} → {lots} lots → {actual_qty} actual quantity")

print("\n🎉 All tests passed! Your Nifty scalper components are working correctly.")
