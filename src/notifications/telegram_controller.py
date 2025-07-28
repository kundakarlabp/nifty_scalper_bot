import logging
import json
from datetime import datetime
import pytz
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

class TelegramController:
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.last_message_time = 0
        self.message_cooldown = 2  # 2 seconds between messages
        self.sent_messages = set()  # Track recently sent messages
        self.max_recent_messages = 100
        self.command_history = {}  # Track command timestamps
        self.command_cooldown = 5  # 5 seconds between same commands
        
    def send_message(self, message: str) -> bool:
        """Send message via Telegram with rate limiting"""
        try:
            if not self.bot_token or not self.chat_id:
                logger.warning("⚠️  Telegram credentials not configured")
                return False
            
            # Rate limiting - prevent spam
            import time
            current_time = time.time()
            
            # Prevent duplicate messages
            message_hash = hash(message[:100])  # Hash first 100 chars
            if message_hash in self.sent_messages:
                logger.debug("⚠️  Duplicate message prevented")
                return False
            
            # Add to recent messages
            self.sent_messages.add(message_hash)
            if len(self.sent_messages) > self.max_recent_messages:
                # Remove oldest messages
                oldest_messages = list(self.sent_messages)[:10]
                for msg in oldest_messages:
                    self.sent_messages.discard(msg)
            
            # Prevent too frequent messages
            if current_time - self.last_message_time < self.message_cooldown:
                logger.debug("⚠️  Rate limiting Telegram message")
                return False
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("✅ Telegram message sent successfully")
                self.last_message_time = current_time
                return True
            elif response.status_code == 429:
                logger.warning("⚠️  Telegram rate limit exceeded. Waiting...")
                # Wait and retry
                time.sleep(5)
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    logger.info("✅ Telegram message sent successfully (retry)")
                    self.last_message_time = current_time
                    return True
                else:
                    logger.error(f"❌ Failed to send Telegram message after retry: {response.status_code}")
                    return False
            else:
                logger.error(f"❌ Failed to send Telegram message: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending Telegram message: {e}")
            return False
    
    def send_startup_alert(self):
        """Send system startup alert"""
        message = """
🚀 **NIFTY SCALPER BOT STARTED**

✅ System is now operational and monitoring markets
📊 Real-time trading activated
🔔 Telegram alerts enabled
📈 Dashboard available at: http://localhost:8000

Use `/help` to see available commands
        """
        return self.send_message(message)
    
    def send_shutdown_alert(self):
        """Send system shutdown alert"""
        message = """
🛑 **NIFTY SCALPER BOT STOPPED**

⚠️ System has been shut down
📊 Trading operations suspended
📈 Dashboard no longer available

Use `/start` to restart the system
        """
        return self.send_message(message)
    
    def send_system_status(self, status: dict = None):
        """Send system status"""
        try:
            if not status:
                status = {
                    'is_trading': False,
                    'execution_enabled': False,
                    'streaming_status': {'connected': False, 'tokens': 0},
                    'active_signals': 0,
                    'active_positions': 0,
                    'trading_instruments': 0,
                    'risk_status': {
                        'account_size': 100000,
                        'daily_pnl': 0,
                        'drawdown_percentage': 0,
                        'current_positions': 0,
                        'max_positions': 1
                    }
                }
            
            message = f"""
📊 **SYSTEM STATUS**

✅ Trading Status: {'ACTIVE' if status.get('is_trading', False) else 'INACTIVE'}
⚡ Execution: {'ENABLED' if status.get('execution_enabled', False) else 'DISABLED'}
📡 WebSocket: {'CONNECTED' if status.get('streaming_status', {}).get('connected', False) else 'DISCONNECTED'}
🔔 Active Signals: {status.get('active_signals', 0)}
💼 Active Positions: {status.get('active_positions', 0)}
�� Trading Instruments: {status.get('trading_instruments', 0)}

💰 Risk Management:
- Account Size: ₹{status.get('risk_status', {}).get('account_size', 0):,.2f}
- Daily P&L: ₹{status.get('risk_status', {}).get('daily_pnl', 0):,.2f}
- Drawdown: {status.get('risk_status', {}).get('drawdown_percentage', 0):.2f}%
- Positions: {status.get('risk_status', {}).get('current_positions', 0)}/{status.get('risk_status', {}).get('max_positions', 0)}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S %Z')}
            """
            return self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error sending system status: {e}")
            return False

# Example usage
if __name__ == "__main__":
    print("Telegram Controller ready!")
    print("Import and use: from src.notifications.telegram_controller import TelegramController")
