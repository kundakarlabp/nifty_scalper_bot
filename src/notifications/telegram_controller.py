# src/notifications/telegram_controller.py
"""
Handles Telegram messaging for the trading bot.
Includes deduplication, rate limiting, formatted alerts, and command handling.
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
    Handles Telegram messaging and command processing for the trading bot.
    """
    
    def __init__(self, status_callback: Optional[Callable] = None, control_callback: Optional[Callable] = None):
        """
        Initialize the Telegram controller with credentials and settings.

        Args:
            status_callback (Optional[Callable]): A function that returns the current system status dict.
            control_callback (Optional[Callable]): A function to handle control commands (enable/disable).
        """
        self.bot_token: str = Config.TELEGRAM_BOT_TOKEN
        self.chat_id: str = str(Config.TELEGRAM_USER_ID)
        self.last_message_time: float = 0
        self.message_cooldown: int = 1  # seconds - faster for commands
        self.sent_messages: deque = deque(maxlen=100)  # FIFO for duplicate prevention
        # FIXED: Removed extra spaces in the URL construction
        self.base_url: str = f"https://api.telegram.org/bot{self.bot_token}"
        self.status_callback = status_callback
        self.control_callback = control_callback # For enable/disable commands
        self.is_listening = False
        self.offset = 0 # For message polling

    # --- Existing send_message and alert methods (mostly unchanged, minor improvements) ---
    def send_message(self, message: str) -> bool:
        # ... (Keep existing send_message logic) ...
        # (Implementation from previous optimized version is correct)
        # Ensure base_url is correct and rate limiting/deduplication works
        # ... 

    def send_startup_alert(self) -> bool:
        """Send a startup alert message."""
        message = """
🚀 *NIFTY SCALPER BOT STARTED*

✅ System is operational
📊 Trading Ready
🔔 Telegram Alerts & Commands Active
📈 Mode: Live Trading Default is *{}*

Use `/help` for commands
        """.format("ENABLED" if Config.ENABLE_LIVE_TRADING else "DISABLED")
        return self.send_message(message)

    def send_shutdown_alert(self) -> bool:
        """Send a shutdown alert message."""
        message = """
🛑 *NIFTY SCALPER BOT STOPPED*

⚠️ Trading Halted
📈 System Offline

Use `/start` to resume (requires restart)
        """
        return self.send_message(message)

    def send_system_status(self, status: Optional[Dict[str, Any]] = None) -> bool:
       # ... (Keep existing send_system_status logic, minor fixes if needed) ...
       # Use provided status, callback, or default
       # Format message with improved clarity
       # ...

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
                'account_size': Config.ACCOUNT_SIZE,
                'daily_pnl': 0,
                'drawdown_percentage': 0,
                'current_positions': 0,
                'max_positions': 1
            },
            'uptime_formatted': '0h 0m 0s'
        }

    def send_realtime_session_alert(self, session_type: str, timestamp: Optional[datetime] = None) -> bool:
        # ... (Keep existing logic) ...
        pass

    # --- NEW: Enhanced Signal Alert ---
    def send_signal_alert(self, token: int, signal_details: Dict[str, Any], position_details: Dict[str, Any]):
        """
        Send a detailed signal alert with TP, SL, and strength.
        """
        try:
            direction = signal_details.get('signal', 'UNKNOWN')
            entry = signal_details.get('entry_price', 0)
            sl = signal_details.get('stop_loss', 0)
            tp = signal_details.get('target', 0)
            confidence = signal_details.get('confidence', 0) * 100
            volatility = signal_details.get('market_volatility', 0)
            reasons = ', '.join(signal_details.get('reasons', [])[:3])
            qty = position_details.get('quantity', 0)
            lots = position_details.get('lots', 'N/A')

            # Calculate Risk & Reward
            risk_per_unit = abs(entry - sl)
            reward_per_unit = abs(tp - entry)
            rr_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0

            message = f"""
🎯 *TRADE SIGNAL GENERATED*

📈 *Instrument:* `{token}`
📊 *Direction:* `{direction}`
💰 *Entry:* `{entry:.2f}`
🛑 *Stop Loss:* `{sl:.2f}`
🎯 *Take Profit:* `{tp:.2f}`
📊 *Risk/Reward:* `{rr_ratio:.2f}:1`
🔥 *Strength:* `{confidence:.1f}%`
🌊 *Volatility:* `{volatility:.2f}`
📦 *Quantity:* `{qty}` ({lots} lots)
🧠 *Reasons:* `{reasons}`
            """
            return self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error sending signal alert: {e}", exc_info=True)
            return False

    # --- NEW: Command Handling Logic ---
    def _handle_incoming_message(self, message_data: Dict[str, Any]):
        """Process an incoming message and respond to commands."""
        try:
            message = message_data.get('message', {})
            text = message.get('text', '').strip()
            chat_id = str(message.get('chat', {}).get('id'))
            user_id = str(message.get('from', {}).get('id', ''))

            # Security: Only respond to messages from the configured user
            if user_id != self.chat_id:
                logger.debug(f"⚠️ Ignoring message from unauthorized user {user_id}")
                # Optionally send a security alert to the owner
                # if chat_id == self.chat_id: # Only if the message was in the correct chat
                #     self.send_message(f"⚠️ Security Alert: Unauthorized access attempt from user {user_id}")
                return

            logger.info(f"Received Telegram command: '{text}' from user {user_id}")

            # Parse command and arguments
            parts = text.split(' ', 1)
            command = parts[0].lower()
            # arguments = parts[1] if len(parts) > 1 else ""

            if command == '/start':
                self.send_message("🚀 *NIFTY SCALPER BOT*\n\nBot is operational. Use /help for commands.")

            elif command == '/help':
                help_text = f"""
*🤖 NIFTY SCALPER BOT - COMMANDS*

/start - Check if bot is running
/help - Display this help message
/status - Show detailed system status
/enable - *Enable* live trade execution
/disable - *Disable* live trade execution
/ping - Simple connectivity test
/signal - Show last generated signal (placeholder)

*Live Trading Default:* {'✅ ENABLED' if Config.ENABLE_LIVE_TRADING else '⚠️ DISABLED'}
*Note:* Commands are processed by the bot.
                """
                self.send_message(help_text)

            elif command == '/status':
                # Trigger status report
                self.send_system_status()

            elif command == '/ping':
                self.send_message("✅ Pong! Bot is responsive.")

            elif command == '/enable':
                if self.control_callback:
                    success = self.control_callback(True) # Call the control function to enable
                    if success:
                        self.send_message("✅ *Live Trade Execution ENABLED*")
                    else:
                        self.send_message("❌ *Failed to enable live trading*")
                else:
                    self.send_message("❌ Control callback not configured.")

            elif command == '/disable':
                if self.control_callback:
                    success = self.control_callback(False) # Call the control function to disable
                    if success:
                        self.send_message("✅ *Live Trade Execution DISABLED*")
                    else:
                        self.send_message("❌ *Failed to disable live trading*")
                else:
                    self.send_message("❌ Control callback not configured.")

            elif command == '/signal':
                 # Placeholder - you could fetch the last signal from trader state
                 # For now, show a message
                 self.send_message("ℹ️ *Last Signal:* (Feature: Fetch last signal details)")

            # elif command == '/force_entry':
            #     # Advanced: Force entry for a specific token
            #     # Requires parsing arguments and calling trader method
            #     # self.send_message("⚠️ /force_entry command not yet implemented.")
            #     pass

            # elif command == '/cancel':
            #     # Advanced: Cancel an order by ID
            #     # Requires parsing arguments and calling order executor
            #     # self.send_message("⚠️ /cancel command not yet implemented.")
            #     pass

            else:
                # Unknown command
                self.send_message("❓ Unknown command. Use /help for available commands.")

        except Exception as e:
            logger.error(f"❌ Error handling incoming Telegram message: {e}", exc_info=True)
            # Send error message back
            self.send_message("❌ An error occurred while processing your command.")

    # --- Polling Mechanism (Keep existing logic, minor fixes) ---
    def start_polling(self):
        # ... (Keep existing start_polling logic) ...
        pass

    def stop_polling(self):
        # ... (Keep existing stop_polling logic) ...
        pass

# Example usage (if run directly)
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print("✅ Telegram Controller module ready!")
