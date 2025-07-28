import logging
import json
from datetime import datetime
import pytz
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

class TelegramCommandHandler:
    def __init__(self, trader=None):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.trader = trader
        self.timezone = pytz.timezone('Asia/Kolkata')
        self.is_polling = False
        
    def send_message(self, message: str) -> bool:
        """Send message via Telegram"""
        try:
            if not self.bot_token or not self.chat_id:
                logger.warning("⚠️  Telegram credentials not configured")
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
                return True
            else:
                logger.error(f"❌ Failed to send Telegram message: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending Telegram message: {e}")
            return False
    
    def send_startup_alert(self) -> bool:
        """Send system startup alert"""
        try:
            ist_time = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            message = f"""
🚀 **NIFTY SCALPER BOT STARTED**

✅ System is now operational and monitoring markets
📊 Real-time trading activated (EXECUTION ENABLED)
🔔 Telegram alerts enabled
📈 Dashboard available at: http://localhost:8000

Started at: {ist_time}
Use `/help` to see available commands
            """
            return self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error sending startup alert: {e}")
            return False
    
    def send_shutdown_alert(self) -> bool:
        """Send system shutdown alert"""
        try:
            ist_time = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            message = f"""
🛑 **NIFTY SCALPER BOT STOPPED**

⚠️ System has been shut down
📊 Trading operations suspended
📈 Dashboard no longer available

Stopped at: {ist_time}
Use `/start` to restart the system
            """
            return self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error sending shutdown alert: {e}")
            return False
    
    def send_system_status(self) -> bool:
        """Send current system status"""
        try:
            if self.trader:
                status = self.trader.get_trading_status()
                ist_time = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
                
                message = f"""
📊 **SYSTEM STATUS**

✅ Trading Status: {'ACTIVE' if status.get('is_trading', False) else 'INACTIVE'}
⚡ Execution: {'ENABLED' if status.get('execution_enabled', False) else 'DISABLED'}
📡 WebSocket: {'CONNECTED' if status.get('streaming_status', {}).get('connected', False) else 'DISCONNECTED'}
🔔 Active Signals: {status.get('active_signals', 0)}
💼 Active Positions: {status.get('active_positions', 0)}
📈 Trading Instruments: {status.get('trading_instruments', 0)}

💰 Risk Management:
- Account Size: ₹{status.get('risk_status', {}).get('account_size', 0):,.2f}
- Daily P&L: ₹{status.get('risk_status', {}).get('daily_pnl', 0):,.2f}
- Drawdown: {status.get('risk_status', {}).get('drawdown_percentage', 0):.2f}%
- Positions: {status.get('risk_status', {}).get('current_positions', 0)}/{status.get('risk_status', {}).get('max_positions', 0)}

🕐 Last Update: {ist_time}
                """
                return self.send_message(message)
            else:
                return self.send_message("📊 System status: Bot initialized but not connected to trader")
        except Exception as e:
            logger.error(f"❌ Error sending system status: {e}")
            return False
    
    def send_help_message(self) -> bool:
        """Send help message with available commands"""
        try:
            ist_time = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            message = f"""
🤖 **NIFTY SCALPER BOT COMMANDS**

🔧 **System Control:**
/start - Start trading system
/stop - Stop trading system
/status - Show system status
/help - Show this help message

�� **Trading Control:**
/enable - Enable trade execution
/disable - Disable trade execution
/trade - Toggle trading on/off
/pause - Pause trading temporarily

📊 **Information:**
/performance - Show performance summary
/signals - Show recent signals
/trades - Show recent trades
/metrics - Show trading metrics
/equity - Show equity curve

⚙️ **Configuration:**
/settings - Show current settings
/risk - Show risk management status
/limits - Show position limits

🔔 **Notifications:**
/alerts on - Enable alerts
/alerts off - Disable alerts
/daily on - Enable daily reports
/daily off - Disable daily reports

🕐 **Time-based:**
/time - Show current time
/uptime - Show system uptime
/ping - Test bot connectivity

🕐 Current Time: {ist_time}
            """
            return self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error sending help message: {e}")
            return False

# Example usage
if __name__ == "__main__":
    print("Telegram Command Handler ready!")
    print("Import and use: from src.notifications.telegram_commands import TelegramCommandHandler")
