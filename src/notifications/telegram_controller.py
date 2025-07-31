# src/notifications/telegram_controller.py
"""
Handles Telegram messaging for the trading bot.
Includes deduplication, rate limiting, and formatted alerts.
Also includes basic command handling for status checks.
"""
import logging
import time
from datetime import datetime
import pytz
import requests
from collections import deque
from typing import Dict, Any, Optional, Callable

# Import configuration using the Config class for consistency
from config import Config

logger = logging.getLogger(__name__)

class TelegramController:
    """
    Handles Telegram messaging for the trading bot.
    Includes deduplication, rate limiting, and formatted alerts.
    """
    
    def __init__(self, status_callback: Optional[Callable] = None):
        """
        Initialize the Telegram controller with credentials and settings.

        Args:
            status_callback (Optional[Callable]): A function that returns the current system status dict.
        """
        self.bot_token: str = Config.TELEGRAM_BOT_TOKEN
        self.chat_id: str = str(Config.TELEGRAM_USER_ID)
        self.last_message_time: float = 0
        self.message_cooldown: int = 2  # seconds
        self.sent_messages: deque = deque(maxlen=100)  # FIFO for duplicate prevention
        # FIXED: Removed extra spaces in the URL construction
        self.base_url: str = f"https://api.telegram.org/bot{self.bot_token}"
        self.status_callback = status_callback
        self.is_listening = False
        self.offset = 0 # For message polling

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
            
            # Create a hash for duplicate detection (using first 100 chars of lowercase message)
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
                time.sleep(max(sleep_time, 0)) # Ensure non-negative sleep

            # Construct the API URL
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
                try:
                    retry_after = response.json().get('parameters', {}).get('retry_after', 5)
                except:
                    retry_after = 5
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
            status (Optional[Dict]): System status dictionary. If None, uses callback or default.

        Returns:
            bool: True if message sent successfully, False otherwise.
        """
        try:
            # Use provided status, callback, or default
            if status is None:
                if self.status_callback:
                    status = self.status_callback()
                else:
                    status = self.default_status()

            # Format the message with status information
            message = f"""
📊 *SYSTEM STATUS*

✅ Trading: {'ACTIVE' if status.get('is_trading', False) else 'INACTIVE'}
⚡ Execution: {'ENABLED' if status.get('execution_enabled', False) else 'DISABLED'}
📡 WebSocket: {'CONNECTED' if status.get('streaming_status', {}).get('connected', False) else 'DISCONNECTED'}
🔔 Active Signals: {status.get('active_signals', 0)}
💼 Positions: {status.get('active_positions', 0)} / {status.get('trading_instruments_count', status.get('trading_instruments', 0))}

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
            # Try to send a simpler error message
            error_msg = f"❌ Error generating status report: {str(e)[:100]}..."
            return self.send_message(error_msg)

    @staticmethod
    def default_status() -> Dict[str, Any]:
        """Provide a default status dictionary."""
        return {
            'is_trading': False,
            'execution_enabled': False,
            'streaming_status': {'connected': False, 'tokens': 0},
            'active_signals': 0,
            'active_positions': 0,
            'trading_instruments_count': 0,
            'risk_status': {
                'account_size': 100000,
                'daily_pnl': 0,
                'drawdown_percentage': 0,
                'current_positions': 0,
                'max_positions': 1
            }
        }

    def send_realtime_session_alert(self, session_type: str, timestamp: Optional[datetime] = None) -> bool:
        """
        Send alert for real-time trading session start/stop.
        
        Args:
            session_type (str): Either 'START' or 'STOP'
            timestamp (Optional[datetime]): When the session event occurred. Uses current time if None.
            
        Returns:
            bool: True if message sent successfully
        """
        if timestamp is None:
            timestamp = datetime.now(pytz.timezone('Asia/Kolkata'))
            
        emoji = "🟢" if session_type.upper() == "START" else "🔴"
        action = "started" if session_type.upper() == "START" else "stopped"
        
        message = f"""
{emoji} *REAL-TIME TRADING SESSION {session_type.upper()}*

⏰ Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}
📊 Status: Session {action}
        """
        return self.send_message(message)

    def _handle_incoming_message(self, message_data: Dict[str, Any]):
        """Process an incoming message and respond to commands."""
        try:
            message = message_data.get('message', {})
            text = message.get('text', '').strip()
            chat_id = message.get('chat', {}).get('id')
            user_id = str(message.get('from', {}).get('id', ''))

            # Security: Only respond to messages from the configured user
            if user_id != self.chat_id:
                logger.debug(f" Ignoring message from unauthorized user {user_id}")
                return

            logger.info(f"Received Telegram command: '{text}' from user {user_id}")

            if text == '/start':
                self.send_message("🚀 *NIFTY SCALPER BOT*\n\nBot is operational. Use /help for commands.")

            elif text == '/help':
                help_text = """
*🤖 NIFTY SCALPER BOT - HELP*

/start - Check if bot is running
/status - Show current system status
/help - Display this help message
/ping - Simple connectivity test

*Note:* Commands are processed by the bot's main thread.
                """
                self.send_message(help_text)

            elif text == '/status':
                # Trigger status report
                self.send_system_status()

            elif text == '/ping':
                self.send_message("✅ Pong! Bot is responsive.")

            else:
                # Unknown command
                self.send_message("❓ Unknown command. Use /help for available commands.")

        except Exception as e:
            logger.error(f"❌ Error handling incoming Telegram message: {e}", exc_info=True)

    def start_polling(self):
        """
        Start polling for incoming messages. This should be run in a separate thread.
        This is a basic polling mechanism. For production, consider webhooks.
        """
        if not self.bot_token:
            logger.warning("⚠️ Telegram bot token missing. Cannot start polling.")
            return
            
        if self.is_listening:
            logger.info("⚠️ Telegram polling already started.")
            return
            
        self.is_listening = True
        logger.info("📡 Starting Telegram message polling...")
        
        while self.is_listening:
            try:
                url = f"{self.base_url}/getUpdates"
                params = {
                    'offset': self.offset,
                    'timeout': 30  # Long polling timeout
                }
                response = requests.get(url, params=params, timeout=35)
                
                if response.status_code == 200:
                    updates = response.json().get('result', [])
                    for update in updates:
                        self.offset = update['update_id'] + 1
                        self._handle_incoming_message(update)
                elif response.status_code == 401:
                    logger.error("❌ Telegram bot token is invalid. Stopping polling.")
                    self.is_listening = False
                elif response.status_code == 409:
                    logger.error("❌ Telegram bot conflict (maybe another instance is running). Stopping polling.")
                    self.is_listening = False
                else:
                    logger.warning(f"⚠️ Telegram getUpdates returned status {response.status_code}")
                    
                # Small delay to prevent excessive polling if there are errors
                if not updates:
                    time.sleep(1)
                    
            except requests.exceptions.Timeout:
                # Timeout is expected with long polling, continue
                continue
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Network error in Telegram polling: {e}")
                time.sleep(5) # Wait before retrying
            except Exception as e:
                logger.error(f"❌ Unexpected error in Telegram polling loop: {e}", exc_info=True)
                time.sleep(5) # Wait before retrying
                
        logger.info("🛑 Telegram message polling stopped.")

    def stop_polling(self):
        """Stop the polling loop."""
        self.is_listening = False
        logger.info("🛑 Stopping Telegram message polling...")

# Example usage (if run directly)
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("✅ Telegram Controller module ready!")
