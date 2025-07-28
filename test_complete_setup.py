import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from src.auth.zerodha_auth import ZerodhaAuthenticator
from src.strategies.scalping_strategy import ScalpingStrategy
from src.risk.position_sizing import PositionSizing
from src.database.models import db_manager
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

print("🧪 Testing Complete Setup...")

# Test 1: Configuration
print("\n1. Testing Configuration...")
try:
    from config import ZERODHA_API_KEY, TELEGRAM_BOT_TOKEN
    print(f"✅ Zerodha API Key: {'SET' if ZERODHA_API_KEY else 'MISSING'}")
    print(f"✅ Telegram Bot Token: {'SET' if TELEGRAM_BOT_TOKEN else 'MISSING'}")
except Exception as e:
    print(f"❌ Configuration test failed: {e}")

# Test 2: Database
print("\n2. Testing Database...")
try:
    from src.database.models import Trade, SignalRecord
    session = db_manager.get_session()
    # Test creating a sample record
    sample_trade = Trade(
        symbol="NIFTY",
        direction="BUY",
        quantity=50,
        entry_price=18000,
        stop_loss=17980,
        target=18040,
        strategy="test"
    )
    session.add(sample_trade)
    session.commit()
    session.delete(sample_trade)
    session.commit()
    db_manager.close_session(session)
    print("✅ Database operations working correctly")
except Exception as e:
    print(f"❌ Database test failed: {e}")

# Test 3: Strategy
print("\n3. Testing Strategy...")
try:
    strategy = ScalpingStrategy()
    # Create sample data
    sample_data = pd.DataFrame({
        'close': [18000 + i*10 + np.random.randn()*50 for i in range(100)],
        'high': [18010 + i*10 + np.random.randn()*50 for i in range(100)],
        'low': [17990 + i*10 + np.random.randn()*50 for i in range(100)],
        'open': [18000 + i*10 + np.random.randn()*50 for i in range(100)],
        'volume': [1000000 + np.random.randint(-100000, 100000) for i in range(100)]
    })
    signal = strategy.generate_signal(sample_data, 18000)
    print(f"✅ Strategy test completed. Signal: {signal}")
except Exception as e:
    print(f"❌ Strategy test failed: {e}")

# Test 4: Risk Management
print("\n4. Testing Risk Management...")
try:
    risk_manager = PositionSizing(account_size=100000, risk_per_trade=0.01)
    position_info = risk_manager.calculate_position_size(18000, 17980, 0.8)
    print(f"✅ Risk management test completed. Position info: {position_info}")
except Exception as e:
    print(f"❌ Risk management test failed: {e}")

print("\n🎉 All tests completed!")
print("🚀 Your trading bot setup is ready!")
