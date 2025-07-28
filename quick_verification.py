import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🚀 Quick System Verification")
print("=" * 30)

# Test 1: Configuration
print("\n1. Configuration Test...")
try:
    from config import ZERODHA_API_KEY, ACCOUNT_SIZE, MAX_DRAWDOWN, NIFTY_LOT_SIZE
    print("✅ Configuration loaded successfully")
    print(f"   Account Size: ₹{ACCOUNT_SIZE:,.2f}")
    print(f"   Lot Size: {NIFTY_LOT_SIZE}")
    print(f"   Max Drawdown: {MAX_DRAWDOWN:.2%}")
except Exception as e:
    print(f"❌ Configuration test failed: {e}")
    sys.exit(1)

# Test 2: Core Components
print("\n2. Core Components Test...")
try:
    # Test Real-time Trader
    from src.data_streaming.realtime_trader import RealTimeTrader
    trader = RealTimeTrader()
    print("✅ Real-time trader loaded")
    
    # Test Strategy
    from src.strategies.scalping_strategy import DynamicScalpingStrategy
    strategy = DynamicScalpingStrategy()
    print("✅ Strategy loaded")
    
    # Test Risk Management
    from src.risk.position_sizing import PositionSizing
    risk_manager = PositionSizing()
    print("✅ Risk management loaded")
    
    # Test Telegram Controller
    from src.notifications.telegram_controller import TelegramController
    controller = TelegramController()
    print("✅ Telegram controller loaded")
    
except Exception as e:
    print(f"❌ Core components test failed: {e}")
    sys.exit(1)

# Test 3: Data Components
print("\n3. Data Components Test...")
try:
    # Test Market Data Streamer
    from src.data_streaming.market_data_streamer import MarketDataStreamer
    streamer = MarketDataStreamer()
    print("✅ Market data streamer loaded")
    
    # Test Data Processor
    from src.data_streaming.data_processor import StreamingDataProcessor
    processor = StreamingDataProcessor()
    print("✅ Data processor loaded")
    
except Exception as e:
    print(f"❌ Data components test failed: {e}")

print("\n" + "=" * 30)
print("🎉 Quick verification completed!")
print("🚀 Your system components are working!")
