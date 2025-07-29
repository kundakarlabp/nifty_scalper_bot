"""
Standalone script to test TelegramController functionality.
Useful for verifying credentials and message formatting.
"""
import sys
import os
import logging
from datetime import datetime
import pytz

# Ensure correct path resolution for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def generate_sample_signal_message() -> str:
    """Generate a sample trading signal message for testing"""
    return f"""
🎯 *SIMULATED SIGNAL ALERT*
📈 Token: 256265 (NIFTY2351018000CE)
📊 Direction: BUY
💰 Entry: 180.50
🛑 SL: 178.00
🎯 Target: 184.00
🔥 Confidence: 92.5%
🌊 Volatility: 1.2
📦 Qty: 100 (2 lots)
🧠 Reason: EMA Bullish Crossover, RSI Oversold, BB Breakout
🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S')}
"""


def test_telegram():
    """Test the TelegramController by sending various types of messages."""
    try:
        from src.notifications.telegram_controller import TelegramController
        telegram = TelegramController()
        logger.info("✅ TelegramController initialized")

        # Test 1: Simple message
        logger.info("📤 Sending simple test message...")
        if telegram.send_message("*Test Message*\nThis is a test from `test_telegram.py`."):
            logger.info("✅ Simple message sent")
        else:
            logger.error("❌ Simple message failed")

        # Test 2: Startup alert
        logger.info("📤 Sending startup alert...")
        if telegram.send_startup_alert():
            logger.info("✅ Startup alert sent")
        else:
            logger.error("❌ Startup alert failed")

        # Test 3: System status message
        logger.info("📤 Sending system status...")
        sample_status = {
            'is_trading': True,
            'execution_enabled': True,
            'streaming_status': {'connected': True, 'tokens': 2},
            'active_signals': 1,
            'active_positions': 1,
            'trading_instruments': 2,
            'risk_status': {
                'account_size': 100000.0,
                'daily_pnl': 1500.0,
                'drawdown_percentage': 0.5,
                'current_positions': 1,
                'max_positions': 1
            }
        }
        if telegram.send_system_status(sample_status):
            logger.info("✅ System status sent")
        else:
            logger.error("❌ System status failed")

        # Test 4: Simulated signal alert
        logger.info("📤 Sending simulated signal alert...")
        if telegram.send_message(generate_sample_signal_message()):
            logger.info("✅ Signal alert sent")
        else:
            logger.error("❌ Signal alert failed")

        # Test 5: Shutdown alert
        logger.info("📤 Sending shutdown alert...")
        if telegram.send_shutdown_alert():
            logger.info("✅ Shutdown alert sent")
        else:
            logger.error("❌ Shutdown alert failed")

        logger.info("🏁 All Telegram tests completed")

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Make sure the project structure is correct.")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    test_telegram()
