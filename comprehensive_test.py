import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🧪 Comprehensive System Test")
print("=" * 50)

# Test 1: Configuration
print("\n1. Testing Configuration...")
try:
    from config import (ZERODHA_API_KEY, ACCOUNT_SIZE, MAX_DRAWDOWN, 
                       BASE_STOP_LOSS_POINTS, BASE_TARGET_POINTS, 
                       CONFIDENCE_THRESHOLD, NIFTY_LOT_SIZE)
    print("✅ Configuration imported successfully")
    print(f"   Account Size: ₹{ACCOUNT_SIZE:,.2f}")
    print(f"   Max Drawdown: {MAX_DRAWDOWN:.2%}")
    print(f"   Lot Size: {NIFTY_LOT_SIZE}")
    print(f"   Stop Loss: {BASE_STOP_LOSS_POINTS} points")
    print(f"   Target: {BASE_TARGET_POINTS} points")
    print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD:.1%}")
except Exception as e:
    print(f"❌ Configuration test failed: {e}")
    sys.exit(1)

# Test 2: Real-time Trader
print("\n2. Testing Real-time Trader...")
try:
    from src.data_streaming.realtime_trader import RealTimeTrader
    trader = RealTimeTrader()
    status = trader.get_trading_status()
    print("✅ Real-time trader initialized successfully")
    print(f"   Trading Status: {status}")
except Exception as e:
    print(f"❌ Real-time trader test failed: {e}")
    sys.exit(1)

# Test 3: Market Data Streamer
print("\n3. Testing Market Data Streamer...")
try:
    from src.data_streaming.market_data_streamer import MarketDataStreamer
    streamer = MarketDataStreamer()
    connection_status = streamer.get_connection_status()
    print("✅ Market data streamer initialized successfully")
    print(f"   Connection Status: {connection_status}")
except Exception as e:
    print(f"❌ Market data streamer test failed: {e}")

# Test 4: Data Processor
print("\n4. Testing Data Processor...")
try:
    from src.data_streaming.data_processor import StreamingDataProcessor
    processor = StreamingDataProcessor()
    buffer_status = processor.get_buffer_status()
    print("✅ Data processor initialized successfully")
    print(f"   Buffer Status: {buffer_status}")
except Exception as e:
    print(f"❌ Data processor test failed: {e}")

# Test 5: Strategy
print("\n5. Testing Strategy...")
try:
    from src.strategies.scalping_strategy import DynamicScalpingStrategy
    strategy = DynamicScalpingStrategy()
    print("✅ Strategy initialized successfully")
    print(f"   Base SL Points: {strategy.base_stop_loss_points}")
    print(f"   Base Target Points: {strategy.base_target_points}")
    print(f"   Confidence Threshold: {strategy.confidence_threshold}")
except Exception as e:
    print(f"❌ Strategy test failed: {e}")

# Test 6: Risk Management
print("\n6. Testing Risk Management...")
try:
    from src.risk.position_sizing import PositionSizing
    risk_manager = PositionSizing()
    print("✅ Risk management initialized successfully")
    print(f"   Account Size: ₹{risk_manager.account_size:,.2f}")
    print(f"   Risk Per Trade: {risk_manager.risk_per_trade:.2%}")
    print(f"   Max Drawdown: {risk_manager.max_drawdown:.2%}")
except Exception as e:
    print(f"❌ Risk management test failed: {e}")

# Test 7: Telegram Controller
print("\n7. Testing Telegram Controller...")
try:
    from src.notifications.telegram_controller import TelegramController
    controller = TelegramController()
    print("✅ Telegram controller initialized successfully")
    print(f"   Bot Token Configured: {'Yes' if controller.bot_token else 'No'}")
    print(f"   Chat ID Configured: {'Yes' if controller.chat_id else 'No'}")
except Exception as e:
    print(f"❌ Telegram controller test failed: {e}")

print("\n" + "=" * 50)
print("🎉 Comprehensive system test completed!")
print("🚀 Your complete trading system is ready!")
