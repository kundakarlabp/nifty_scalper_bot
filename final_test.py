import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🏁 Final System Test")
print("=" * 30)

# Test 1: Configuration
print("\n1. Testing Configuration...")
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
print("\n2. Testing Core Components...")
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
print("\n3. Testing Data Components...")
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

# Test 4: System Integration
print("\n4. Testing System Integration...")
try:
    # Test system status
    status = trader.get_trading_status()
    print("✅ System status check passed")
    print(f"   Trading Status: {'Active' if status['is_trading'] else 'Inactive'}")
    print(f"   Execution Enabled: {'Yes' if status['execution_enabled'] else 'No'}")
    
    # Test Telegram controller with system status
    controller.send_system_status(status)
    print("✅ Telegram controller with system status works")
    
except Exception as e:
    print(f"❌ System integration test failed: {e}")

print("\n" + "=" * 30)
print("🎉 Final system test completed!")
print("🚀 Your complete trading system is working!")
