#!/usr/bin/env python3
"""
Dedicated Telegram Bot for Nifty Scalper - FIXED VERSION
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import time
import json
import requests
from datetime import datetime
import pytz
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from src.data_streaming.realtime_trader import RealTimeTrader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/telegram_bot_fix.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DedicatedTelegramBot:
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.last_update_id = 0
        self.trader = RealTimeTrader()
        self.timezone = pytz.timezone('Asia/Kolkata')
        self.is_running = False
        
        # Initialize trader
        self.trader.add_trading_instrument(256265)  # Nifty 50
        
        if not self.bot_token or not self.chat_id:
            logger.error("❌ Telegram credentials not configured")
            raise ValueError("Telegram credentials missing")
        
        logger.info("✅ Dedicated Telegram Bot initialized")
    
    def send_message(self, message: str) -> bool:
        """Send message via Telegram"""
        try:
            if not self.bot_token or not self.chat_id:
                return False
                
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"❌ Error sending Telegram message: {e}")
            return False
    
    def get_updates(self) -> list:
        """Get Telegram updates"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
            params = {
                'offset': self.last_update_id + 1,
                'timeout': 30
            }
            
            response = requests.get(url, params=params, timeout=35)
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    return data.get('result', [])
            return []
            
        except Exception as e:
            logger.error(f"❌ Error getting Telegram updates: {e}")
            return []
    
    def process_command(self, command: str, user_id: str = None):
        """Process Telegram command"""
        try:
            command = command.lower().strip()
            logger.info(f"📩 Processing command: {command}")
            
            if command in ['/start', '/begin']:
                self._handle_start()
            elif command in ['/stop', '/shutdown']:
                self._handle_stop()
            elif command in ['/status', '/stat']:
                self._handle_status()
            elif command in ['/help', '/h']:
                self._handle_help()
            elif command in ['/enable', '/enable_trading']:
                self._handle_enable()
            elif command in ['/disable', '/disable_trading']:
                self._handle_disable()
            elif command in ['/trade', '/toggle']:
                self._handle_toggle()
            elif command in ['/performance', '/perf']:
                self._handle_performance()
            else:
                self.send_message(f"❓ Unknown command: {command}\nUse /help for available commands")
                
        except Exception as e:
            logger.error(f"❌ Error processing command '{command}': {e}")
            self.send_message(f"❌ Error processing command: {e}")
    
    def _handle_start(self):
        """Handle start command"""
        try:
            if self.trader.start_trading():
                message = "✅ Trading system started successfully!"
                logger.info(message)
                self.send_message(message)
                self._handle_status()
            else:
                message = "❌ Failed to start trading system"
                logger.error(message)
                self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error in start command: {e}")
            self.send_message(f"❌ Error: {e}")
    
    def _handle_stop(self):
        """Handle stop command"""
        try:
            self.trader.stop_trading()
            message = "🛑 Trading system stopped"
            logger.info(message)
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error in stop command: {e}")
            self.send_message(f"❌ Error: {e}")
    
    def _handle_status(self):
        """Handle status command"""
        try:
            status = self.trader.get_trading_status()
            ist_time = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            
            message = f"""
📊 **SYSTEM STATUS**

✅ Trading: {'ACTIVE' if status.get('is_trading', False) else 'INACTIVE'}
⚡ Execution: {'ENABLED' if status.get('execution_enabled', False) else 'DISABLED'}
📡 WebSocket: {'CONNECTED' if status.get('streaming_status', {}).get('connected', False) else 'DISCONNECTED'}
�� Active Signals: {status.get('active_signals', 0)}
💼 Active Positions: {status.get('active_positions', 0)}
📈 Instruments: {status.get('trading_instruments', 0)}

💰 Risk Management:
- Account Size: ₹{status.get('risk_status', {}).get('account_size', 0):,.2f}
- Daily P&L: ₹{status.get('risk_status', {}).get('daily_pnl', 0):,.2f}
- Drawdown: {status.get('risk_status', {}).get('drawdown_percentage', 0):.2f}%

🕐 {ist_time}
            """
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error in status command: {e}")
            self.send_message(f"❌ Error: {e}")
    
    def _handle_help(self):
        """Handle help command"""
        message = """
🤖 **NIFTY SCALPER BOT COMMANDS**

🔧 **System Control:**
/start - Start trading system
/stop - Stop trading system
/status - Show system status
/help - Show this help message

📈 **Trading Control:**
/enable - Enable trade execution
/disable - Disable trade execution
/trade - Toggle trading on/off

📊 **Information:**
/performance - Show performance summary
        """
        self.send_message(message)
    
    def _handle_enable(self):
        """Handle enable command"""
        try:
            self.trader.enable_trading(True)
            message = "✅ Trade execution enabled"
            logger.info(message)
            self.send_message(message)
            self._handle_status()
        except Exception as e:
            logger.error(f"❌ Error in enable command: {e}")
            self.send_message(f"❌ Error: {e}")
    
    def _handle_disable(self):
        """Handle disable command"""
        try:
            self.trader.enable_trading(False)
            message = "🛑 Trade execution disabled"
            logger.info(message)
            self.send_message(message)
            self._handle_status()
        except Exception as e:
            logger.error(f"❌ Error in disable command: {e}")
            self.send_message(f"❌ Error: {e}")
    
    def _handle_toggle(self):
        """Handle toggle command"""
        try:
            current_status = self.trader.execution_enabled
            self.trader.enable_trading(not current_status)
            status_text = "enabled" if not current_status else "disabled"
            message = f"✅ Trade execution {status_text}"
            logger.info(message)
            self.send_message(message)
            self._handle_status()
        except Exception as e:
            logger.error(f"❌ Error in toggle command: {e}")
            self.send_message(f"❌ Error: {e}")
    
    def _handle_performance(self):
        """Handle performance command"""
        try:
            status = self.trader.get_trading_status()
            risk_status = status.get('risk_status', {})
            ist_time = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            
            message = f"""
📈 **PERFORMANCE SUMMARY**

💰 Account Performance:
- Account Size: ₹{risk_status.get('account_size', 0):,.2f}
- Daily P&L: ₹{risk_status.get('daily_pnl', 0):,.2f}
- Total Equity: ₹{risk_status.get('account_size', 0) + risk_status.get('daily_pnl', 0):,.2f}
- Drawdown: {risk_status.get('drawdown_percentage', 0):.2f}%

📊 Trading Metrics:
- Active Positions: {risk_status.get('current_positions', 0)}/{risk_status.get('max_positions', 0)}
- Win Rate: {risk_status.get('win_rate', 0):.1f}%
- Sharpe Ratio: {risk_status.get('sharpe_ratio', 0):.2f}
- Profit Factor: {risk_status.get('profit_factor', 0):.2f}

🕐 {ist_time}
            """
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error in performance command: {e}")
            self.send_message(f"❌ Error: {e}")
    
    def start_bot(self):
        """Start the Telegram bot"""
        try:
            self.is_running = True
            logger.info("🚀 Dedicated Telegram Bot started")
            
            # Send startup message
            self.send_message("🚀 Nifty Scalper Bot is now online and ready to receive commands!")
            self._handle_status()
            
            # Main polling loop
            while self.is_running:
                try:
                    updates = self.get_updates()
                    
                    for update in updates:
                        self.last_update_id = max(self.last_update_id, update.get('update_id', 0))
                        
                        if 'message' in update and 'text' in update['message']:
                            message = update['message']
                            text = message['text'].strip()
                            user_id = str(message.get('from', {}).get('id', ''))
                            
                            # Only process commands from authorized chat
                            if str(message.get('chat', {}).get('id', '')) == str(self.chat_id):
                                if text.startswith('/'):
                                    self.process_command(text, user_id)
                    
                    # Small delay to prevent excessive requests
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"❌ Error in polling loop: {e}")
                    time.sleep(5)
                    
        except KeyboardInterrupt:
            logger.info("�� Telegram bot stopped by user")
            self.is_running = False
        except Exception as e:
            logger.error(f"❌ Error in Telegram bot: {e}")
            self.is_running = False
    
    def stop_bot(self):
        """Stop the Telegram bot"""
        self.is_running = False
        logger.info("✅ Telegram bot stopped")

def main():
    """Main entry point"""
    try:
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Initialize and start bot
        bot = DedicatedTelegramBot()
        logger.info("✅ Dedicated Telegram Bot initialized successfully")
        
        # Start the bot
        bot.start_bot()
        
    except Exception as e:
        logger.error(f"❌ Failed to start Telegram bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
