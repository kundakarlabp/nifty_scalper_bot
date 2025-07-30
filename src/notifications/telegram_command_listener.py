# src/notifications/telegram_command_listener.py
#!/usr/bin/env python3
"""
Telegram polling command listener for Nifty Scalper Bot.
This script polls Telegram for commands and dispatches them to a provided RealTimeTrader instance.
It should be run as a separate process/thread coordinated with the main trading application.
"""
import os
import sys
import time
import logging
import requests
from datetime import datetime

# Setup import paths - Ensure the root of the project is in the path
# This might be handled by your project structure or Docker setup
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import configuration (assuming Config class structure)
# from config import Config # Not directly used here, but confirms environment

# Configure logging for this specific module
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", # Include logger name
    handlers=[
        logging.FileHandler("logs/telegram_command_listener.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__) # Use __name__ for the specific module logger


class TelegramCommandListener:
    """
    Listens for commands via Telegram polling and interacts with a RealTimeTrader instance.
    This class should be initialized with a reference to the *main* RealTimeTrader
    that is running the trading logic.
    """
    def __init__(self, bot_token: str, chat_id: str, trader_instance):
        """
        Initializes the Telegram command listener.

        Args:
            bot_token (str): The Telegram bot token.
            chat_id (str): The authorized user's chat ID.
            trader_instance: An instance of RealTimeTrader to control.
                             This should be the main trader instance running the strategy.
        Raises:
            ValueError: If credentials are missing or trader_instance is None.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.trader = trader_instance # Store the provided trader instance
        self.last_update_id = 0
        self.is_listening = False

        if not self.bot_token or not self.chat_id:
            logger.error("❌ Telegram credentials (token or chat_id) not provided to listener.")
            raise ValueError("Missing Telegram credentials (token or chat_id).")
        
        if self.trader is None:
            logger.error("❌ No RealTimeTrader instance provided to TelegramCommandListener.")
            raise ValueError("A RealTimeTrader instance must be provided.")

        logger.info("✅ TelegramCommandListener initialized and linked to RealTimeTrader.")


    def send_message(self, message: str) -> bool:
        """
        Send a message to the authorized chat via Telegram Bot API.

        Args:
            message (str): The message text to send.

        Returns:
            bool: True if the message was sent successfully, False otherwise.
        """
        try:
            if not self.bot_token or not self.chat_id:
                 logger.warning("⚠️ Telegram credentials missing, cannot send message.")
                 return False

            # FIXED: Corrected URL construction (removed extra space)
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown" # Or "MarkdownV2" or "HTML"
            }
            # Set a timeout for the request
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.debug("📤 Telegram message sent successfully.")
                return True
            else:
                logger.error(f"❌ Failed to send Telegram message. Status: {response.status_code}, Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            # More specific handling for network-related errors
            logger.error(f"❌ Network error sending Telegram message: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error sending Telegram message: {e}", exc_info=True)
        return False


    def get_updates(self):
        """
        Poll the Telegram Bot API for new updates (messages).

        Returns:
            list: A list of update dictionaries, or an empty list on failure.
        """
        try:
            if not self.bot_token:
                logger.warning("⚠️ Telegram bot token missing, cannot get updates.")
                return []

            # FIXED: Corrected URL construction (removed extra space)
            url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
            params = {
                "offset": self.last_update_id + 1,
                "timeout": 30, # Timeout for long polling
                "allowed_updates": ["message"] # Only listen for message updates
            }
            # Set a timeout for the request
            resp = requests.get(url, params=params, timeout=35)
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    return data.get("result", [])
                else:
                    logger.error(f"❌ Telegram API error in getUpdates: {data.get('description', 'Unknown error')}")
            else:
                logger.error(f"❌ HTTP error getting Telegram updates: {resp.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Network error getting Telegram updates: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error getting Telegram updates: {e}", exc_info=True)
        return []


    def handle_command(self, cmd: str):
        """
        Dispatch received command to the appropriate action on the trader instance.

        Args:
            cmd (str): The command string received (e.g., '/start').
        """
        cmd = cmd.lower().strip()
        logger.info(f"📩 Received command: {cmd}")
        
        try:
            # Ensure trader instance is available
            if self.trader is None:
                logger.error("❌ No trader instance available to handle command.")
                self.send_message("❌ Bot internal error: Trader not available.")
                return

            if cmd == "/start":
                logger.info("🚀 Command: Start Trading")
                success = self.trader.start_trading()
                if success:
                    self.send_message("✅ Trading started.")
                else:
                    self.send_message("⚠️ Failed to start trading. Check bot logs.")

            elif cmd == "/stop":
                logger.info("🛑 Command: Stop Trading")
                self.trader.stop_trading() # Assume stop_trading always attempts to stop
                self.send_message("🛑 Trading stopped.")

            elif cmd == "/status":
                logger.info("📊 Command: Get Status")
                status = self.trader.get_trading_status()
                # Add checks for None status or missing keys
                if not status:
                     self.send_message("⚠️ Unable to fetch status.")
                     return

                try:
                    # Safely get nested status values
                    streaming_status = status.get("streaming_status", {})
                    connected = streaming_status.get("connected", False)
                    active = status.get("is_trading", False)
                    
                    msg = f"""
📊 *Bot Status*

Trading: {'✅ ON' if active else '❌ OFF'}
WebSocket: {'📡 CONNECTED' if connected else '❌ DISCONNECTED'}
Active Signals: {status.get('active_signals', 0)}
Active Positions: {status.get('active_positions', 0)}
Instruments Watched: {status.get('trading_instruments_count', 0)}
                    """
                    self.send_message(msg)
                except Exception as e:
                    logger.error(f"❌ Error formatting status message: {e}")
                    self.send_message("❌ Error formatting status.")

            elif cmd == "/risk":
                logger.info("🔐 Command: Get Risk Status")
                status = self.trader.get_trading_status()
                # Add checks for None status
                if not status:
                     self.send_message("⚠️ Unable to fetch risk status.")
                     return
                     
                try:
                    risk = status.get("risk_status", {})
                    if not risk:
                        self.send_message("⚠️ Risk status data unavailable.")
                        return
                        
                    msg = f"""
🔐 *Risk Settings*

Account: ₹{risk.get('account_size', 0):,.2f}
Drawdown: {risk.get('drawdown_percentage', 0):.2f}%
Current P&L: ₹{risk.get('daily_pnl', 0):,.2f}
Positions Open: {risk.get('current_positions', 0)}/{risk.get('max_positions', 1)}
                    """
                    self.send_message(msg)
                except Exception as e:
                    logger.error(f"❌ Error formatting risk message: {e}")
                    self.send_message("❌ Error formatting risk status.")

            elif cmd == "/help":
                logger.info("📘 Command: Show Help")
                help_text = """
📘 *Available Commands*
/start - Start trading
/stop - Stop trading
/status - Bot status
/risk - Risk settings
/ping - Check if bot is alive
/help - Show this help message
                """
                self.send_message(help_text)

            elif cmd == "/ping":
                logger.info("🏓 Command: Ping")
                self.send_message("🏓 Bot is active and listening!")

            else:
                logger.warning(f"❓ Unknown command received: {cmd}")
                self.send_message(f"❓ Unknown command: `{cmd}`\nUse /help for available commands.")

        except Exception as e:
            logger.error(f"❌ Unexpected error processing command '{cmd}': {e}", exc_info=True)
            # Send a generic error message to the user
            self.send_message(f"❌ An error occurred while processing your command: {cmd}")


    def start_listening(self):
        """
        Start the main polling loop to listen for Telegram commands.
        """
        if self.is_listening:
            logger.warning("⚠️ Telegram listener is already running.")
            return

        self.is_listening = True
        logger.info("📡 Telegram polling started...")
        
        # Send a startup message
        self.send_message("🤖 Nifty Scalper Bot command listener is live and listening!")

        try:
            while self.is_listening:
                try:
                    updates = self.get_updates()
                    if updates: # Only process if there are updates
                        for upd in updates:
                            # Safely get update_id
                            update_id = upd.get("update_id")
                            if update_id is not None:
                                self.last_update_id = max(self.last_update_id, update_id)
                            
                            # Extract message details
                            msg = upd.get("message", {})
                            # Ensure text exists and is a string
                            text = msg.get("text", "")
                            if not isinstance(text, str):
                                text = ""
                            text = text.strip()
                            
                            # Extract chat_id safely
                            chat = msg.get("chat", {})
                            chat_id = str(chat.get("id", "")) if chat.get("id") is not None else ""
                            
                            # Check if message is from the authorized user and is a command
                            if chat_id == self.chat_id and text.startswith("/"):
                                self.handle_command(text)
                    # Sleep briefly to avoid excessive polling
                    time.sleep(1) 
                    
                except Exception as e:
                    logger.error(f"❌ Error in main polling loop iteration: {e}", exc_info=True)
                    # Brief pause before retrying to avoid tight loop on persistent errors
                    time.sleep(5)

        except KeyboardInterrupt:
            logger.info("🛑 Telegram listener stopped by KeyboardInterrupt.")
        except Exception as e:
            logger.critical(f"💥 Telegram listener crashed unexpectedly: {e}", exc_info=True)
        finally:
            self.is_listening = False
            logger.info("🛑 Telegram command listener stopped.")


    def stop_listening(self):
        """
        Signal the polling loop to stop.
        """
        logger.info("🛑 Stop signal received for Telegram listener.")
        self.is_listening = False


def main():
    """
    Main function to run the Telegram command listener.
    This is an example of how to instantiate and run it.
    In practice, this might be started by your main application (src/main.py)
    or a process manager.
    """
    try:
        # Example: How to get credentials (replace with your method)
        # These would typically come from environment variables or a config object
        # passed in by the main application.
        # For this standalone script example, you'd need to define them.
        # In a real integration, main.py would create the trader and listener together.
        
        # Example credentials (replace with actual secure loading)
        # BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        # USER_CHAT_ID = os.getenv("TELEGRAM_USER_ID") # Should be a string
        # if not BOT_TOKEN or not USER_CHAT_ID:
        #     logger.error("❌ TELEGRAM_BOT_TOKEN and TELEGRAM_USER_ID must be set in environment.")
        #     sys.exit(1)
        # USER_CHAT_ID = str(USER_CHAT_ID)

        # Example: Creating a trader instance (this is just for standalone running)
        # In your real app, the *main* trader instance should be passed in.
        # from src.data_streaming.realtime_trader import RealTimeTrader
        # trader = RealTimeTrader() # This would be configured by main.py
        # logger.warning("⚠️ Standalone mode: Created a new RealTimeTrader instance. This is for testing only.")

        # listener = TelegramCommandListener(BOT_TOKEN, USER_CHAT_ID, trader)
        # listener.start_listening()
        
        logger.error("This script is intended to be run as part of the main application.")
        logger.error("Please start the bot using 'python src/main.py'")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("🛑 Telegram Command Listener stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logger.error(f"❌ Telegram Command Listener failed to start or crashed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
