# src/notifications/telegram_controller.py
import logging
import time
from datetime import datetime
import pytz
import requests
from collections import deque
from typing import Dict, Any, Optional

# Import configuration using the Config class for consistency
from config import Config

logger = logging.getLogger(__name__)

class TelegramController:
    """
    Handles Telegram messaging for the trading bot.
    Includes deduplication, rate limiting, and formatted alerts.
    """
    
    def __init__(self):
        """Initialize the Telegram controller with credentials and settings."""
        self.bot_token: str = Config.TELEGRAM_BOT_TOKEN
        self.chat_id: str = str(Config.TELEGRAM_USER_ID)
        self.last_message_time: float = 0
        self.message_cooldown: int = 2  # seconds
        self.sent_messages: deque = deque(maxlen=100)  # FIFO for duplicate prevention
        self.base_url: str = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(self, message: str) -> bool:
        """
        Send a message via Telegram with deduplication and rate-limiting.

        Args:
            message (str): The message to send.

        Returns:
            bool: True if message sent successfully, False otherwise.
        """
        try:
            # Validate credentials
            if not self.bot_token or not self.chat_id:
                logger.warning("⚠️ Telegram credentials missing")
                return False

            current_time = time.time()
            
            # Create a hash for duplicate detection
            message_hash = hash(message.lower().strip()[:100])

            # Duplicate check
            if message_hash in self.sent_messages:
                logger.debug("⚠️ Duplicate message skipped")
                return False

            # Rate limiting
            time_since_last = current_time - self.last_message_time
            if time_since_last < self.message_cooldown:
                sleep_time = self.message_cooldown - time_since_last
                logger.debug(f"⏳ Rate limiting - sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)

            # Construct the API URL (FIXED: removed extra space)
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }

            # Send the message
            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                # Success: record the message and timestamp
                self.sent_messages.append(message_hash)
                self.last_message_time = time.time()
                logger.info("✅ Telegram message sent")
                return True

            elif response.status_code == 429:
                # Rate limit hit: extract retry_after value or use default
                retry_after = response.json().get('parameters', {}).get('retry_after', 5)
                logger.warning(f"⚠️ Rate limit hit. Retrying in {retry_after}s...")
                time.sleep(retry_after)
                
                retry_response = requests.post(url, json=payload, timeout=10)
                if retry_response.status_code == 200:
                    self.sent_messages.append(message_hash)
                    self.last_message_time = time.time()
                    logger.info("✅ Telegram retry succeeded")
                    return True
                logger.error(f"❌ Retry failed: {retry_response.status_code}")
                return False

            else:
                # Other error
                logger.error(f"❌ Failed to send Telegram message: {response.status_code} - {response.text}")
                return False

        except requests.exceptions.Timeout:
            logger.error("❌ Telegram request timeout")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error in Telegram message: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Exception in Telegram message: {e}", exc_info=True)
            return False

    def send_startup_alert(self) -> bool:
        """Send a startup alert message."""
        message = """
🚀 *NIFTY SCALPER BOT STARTED*

✅ System is operational
📊 Trading Activated
🔔 Telegram Alerts On
📈 Dashboard: http://localhost:8000

Use `/help` for commands
        """
        return self.send_message(message)

    def send_shutdown_alert(self) -> bool:
        """Send a shutdown alert message."""
        message = """
🛑 *NIFTY SCALPER BOT STOPPED*

⚠️ Trading Halted
📈 Dashboard Offline

Use `/start` to resume
        """
        return self.send_message(message)

    def send_system_status(self, status: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send a formatted system status message.

        Args:
            status (Optional[Dict]): System status dictionary.

        Returns:
            bool: True if message sent successfully, False otherwise.
        """
        try:
            # Use default status if none provided
            if not status:
                status = self.default_status()

            # Format the message with status information
            message = f"""
📊 *SYSTEM STATUS*

✅ Trading: {'ACTIVE' if status.get('is_trading', False) else 'INACTIVE'}
⚡ Execution: {'ENABLED' if status.get('execution_enabled', False) else 'DISABLED'}
📡 WebSocket: {'CONNECTED' if status.get('streaming_status', {}).get('connected', False) else 'DISCONNECTED'}
🔔 Active Signals: {status.get('active_signals', 0)}
💼 Positions: {status.get('active_positions', 0)} / {status.get('trading_instruments', 0)}

💰 *RISK MGMT*
- Equity: ₹{status.get('risk_status', {}).get('account_size', 0):,.2f}
- P&L: ₹{status.get('risk_status', {}).get('daily_pnl', 0):,.2f}
- Drawdown: {status.get('risk_status', {}).get('drawdown_percentage', 0):.2f}%
- Open Trades: {status.get('risk_status', {}).get('current_positions', 0)}/{status.get('risk_status', {}).get('max_positions', 1)}

🕐 {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S %Z')}
"""
            return self.send_message(message)

        except Exception as e:
            logger.error(f"❌ Error sending system status: {e}", exc_info=True)
            return False

    @staticmethod
    def default_status() -> Dict[str, Any]:
        """Provide a default status dictionary."""
        return {
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

    def send_realtime_session_alert(self, session_type: str, timestamp: datetime) -> bool:
        """
        Send alert for real-time trading session start/stop.
        
        Args:
            session_type (str): Either 'START' or 'STOP'
            timestamp (datetime): When the session event occurred
            
        Returns:
            bool: True if message sent successfully
        """
        emoji = "🟢" if session_type.upper() == "START" else "🔴"
        action = "started" if session_type.upper() == "START" else "stopped"
        
        message = f"""
{emoji} *REAL-TIME TRADING SESSION {session_type.upper()}*

⏰ Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}
📊 Status: Session {action}
        """
        return self.send_message(message)

# Example usage (if run directly)
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    print("✅ Telegram Controller module ready!")
