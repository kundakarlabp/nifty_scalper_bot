#!/usr/bin/env python3
"""
Telegram polling command listener for Nifty Scalper Bot
"""
import os
import sys
import time
import logging
import requests
import pytz
from datetime import datetime

# Setup import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config import Config
from src.data_streaming.realtime_trader import RealTimeTrader

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/telegram_listener.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TelegramCommandListener:
    def __init__(self):
        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = str(Config.TELEGRAM_USER_ID)
        self.last_update_id = 0
        self.is_listening = False
        self.trader = RealTimeTrader()
        self.timezone = pytz.timezone("Asia/Kolkata")

        if not self.bot_token or not self.chat_id:
            logger.error("❌ Telegram credentials not set")
            raise ValueError("Missing Telegram credentials")

        # Add Nifty 50 by default (token hardcoded, modify later if needed)
        self.trader.add_trading_instrument(256265)
        logger.info("✅ TelegramCommandListener initialized")

    def send_message(self, message: str) -> bool:
        """Send a message to the bot's chat"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            return False

    def get_updates(self):
        """Poll Telegram for new updates"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
            params = {
                "offset": self.last_update_id + 1,
                "timeout": 30,
                "allowed_updates": ["message"]
            }
            resp = requests.get(url, params=params, timeout=35)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("result", [])
            return []
        except Exception as e:
            logger.warning(f"⚠️ Error getting updates: {e}")
            return []

    def handle_command(self, cmd: str):
        """Dispatch command to function"""
        cmd = cmd.lower().strip()
        logger.info(f"📩 Received command: {cmd}")
        try:
            if cmd == "/start":
                self.trader.start_trading()
                self.send_message("✅ Bot started")
            elif cmd == "/stop":
                self.trader.stop_trading()
                self.send_message("🛑 Bot stopped")
            elif cmd == "/status":
                status = self.trader.get_trading_status()
                connected = status.get("streaming_status", {}).get("connected", False)
                active = status.get("is_trading", False)
                msg = f"""
📊 *Bot Status*

Trading: {'✅ ON' if active else '❌ OFF'}
WebSocket: {'📡 CONNECTED' if connected else '❌ DISCONNECTED'}
Active Signals: {status.get('active_signals', 0)}
Active Positions: {status.get('active_positions', 0)}
                """
                self.send_message(msg)
            elif cmd == "/risk":
                risk = self.trader.get_trading_status().get("risk_status", {})
                msg = f"""
🔐 *Risk Settings*

Account: ₹{risk.get('account_size', 0):,.2f}
Drawdown: {risk.get('drawdown_percentage', 0):.2f}%
Current P&L: ₹{risk.get('daily_pnl', 0):,.2f}
                """
                self.send_message(msg)
            elif cmd == "/help":
                help_text = """
📘 *Available Commands*
/start - Start trading
/stop - Stop trading
/status - Bot status
/risk - Risk settings
/help - Show commands
                """
                self.send_message(help_text)
            elif cmd == "/ping":
                self.send_message("🏓 Bot is active!")
            else:
                self.send_message(f"❓ Unknown command: `{cmd}`\nUse /help")
        except Exception as e:
            logger.error(f"❌ Error processing '{cmd}': {e}")
            self.send_message(f"❌ Command error: {e}")

    def start_listening(self):
        """Start polling loop"""
        self.is_listening = True
        logger.info("📡 Telegram polling started...")
        self.send_message("🤖 Nifty Scalper Bot is live and listening!")

        while self.is_listening:
            try:
                updates = self.get_updates()
                for upd in updates:
                    self.last_update_id = max(self.last_update_id, upd["update_id"])
                    msg = upd.get("message", {})
                    text = msg.get("text", "").strip()
                    chat_id = str(msg.get("chat", {}).get("id"))
                    if chat_id == self.chat_id and text.startswith("/"):
                        self.handle_command(text)
            except Exception as e:
                logger.error(f"❌ Listener error: {e}")
            time.sleep(1)

    def stop_listening(self):
        """Stop polling"""
        self.is_listening = False
        logger.info("🛑 Telegram listener stopped")


def main():
    try:
        listener = TelegramCommandListener()
        listener.start_listening()
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user")
    except Exception as e:
        logger.error(f"❌ Listener crashed: {e}")


if __name__ == "__main__":
    main()