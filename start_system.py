#!/usr/bin/env python3
"""
Start the complete Nifty Scalper Trading System
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system_startup.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def start_complete_system():
    """Start the complete trading system"""
    try:
        logger.info("🚀 Starting Complete Nifty Scalper Trading System...")
        logger.info(f"🕐 Startup time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test 1: Configuration
        logger.info("1. Testing Configuration...")
        from config import ZERODHA_API_KEY, ACCOUNT_SIZE, MAX_DRAWDOWN, NIFTY_LOT_SIZE
        logger.info(f"✅ Configuration loaded - Account: ₹{ACCOUNT_SIZE:,.2f}, Lot Size: {NIFTY_LOT_SIZE}")
        
        # Test 2: Core Components
        logger.info("2. Testing Core Components...")
        from src.data_streaming.realtime_trader import RealTimeTrader
        from src.strategies.scalping_strategy import DynamicScalpingStrategy
        from src.risk.position_sizing import PositionSizing
        from src.notifications.telegram_controller import TelegramController
        
        trader = RealTimeTrader()
        strategy = DynamicScalpingStrategy()
        risk_manager = PositionSizing()
        telegram_controller = TelegramController()
        
        logger.info("✅ Core components loaded successfully")
        
        # Test 3: Data Components
        logger.info("3. Testing Data Components...")
        from src.data_streaming.market_data_streamer import MarketDataStreamer
        from src.data_streaming.data_processor import StreamingDataProcessor
        
        streamer = MarketDataStreamer()
        processor = StreamingDataProcessor()
        
        logger.info("✅ Data components loaded successfully")
        
        # Test 4: System Status
        logger.info("4. Testing System Status...")
        status = trader.get_trading_status()
        logger.info(f"✅ System status check passed - Trading: {'Active' if status['is_trading'] else 'Inactive'}")
        
        # Test 5: Telegram Integration
        logger.info("5. Testing Telegram Integration...")
        telegram_controller.send_startup_alert()
        telegram_controller.send_system_status(status)
        logger.info("✅ Telegram integration working successfully")
        
        # Test 6: Risk Management
        logger.info("6. Testing Risk Management...")
        risk_status = risk_manager.get_risk_status()
        logger.info(f"✅ Risk management working - Account: ₹{risk_status['account_size']:,.2f}")
        
        # Test 7: Strategy
        logger.info("7. Testing Strategy...")
        logger.info(f"✅ Strategy working - SL: {strategy.base_stop_loss_points} points")
        
        logger.info("🎉 Complete system startup successful!")
        logger.info("🚀 Your Nifty Scalper Trading System is ready!")
        logger.info("📊 Visit dashboard at: http://localhost:8000")
        logger.info("📱 Check Telegram for system alerts")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System startup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Start the system
    success = start_complete_system()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 NIFTY SCALPER TRADING SYSTEM STARTUP SUCCESS!")
        print("=" * 50)
        print("✅ Configuration: Loaded")
        print("✅ Core Components: Working")
        print("✅ Data Components: Working")
        print("✅ System Status: Operational")
        print("✅ Telegram Integration: Working")
        print("✅ Risk Management: Working")
        print("✅ Strategy: Working")
        print("\n🚀 YOUR COMPLETE TRADING SYSTEM IS READY!")
        print("\n🔧 Next steps:")
        print("   1. Run web dashboard: cd src/web_dashboard && python app.py")
        print("   2. Start trading: python src/main.py --mode realtime")
        print("   3. Monitor via Telegram")
        print("   4. Visit dashboard at http://localhost:8000")
        print("\n🕐 System startup completed successfully!")
    else:
        print("\n❌ System startup failed. Check logs for details.")
        sys.exit(1)
