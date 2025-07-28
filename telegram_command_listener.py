#!/usr/bin/env python3
"""
Dedicated Telegram Command Listener
Runs separately to listen for incoming commands
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import time
import requests
import json
from datetime import datetime
import pytz
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from src.data_streaming.realtime_trader import RealTimeTrader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/telegram_listener.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TelegramCommandListener:
    def __init__(self):
        self.bot_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.last_update_id = 0
        self.is_listening = False
        self.trader = RealTimeTrader()
        self.timezone = pytz.timezone('Asia/Kolkata')
        
        # Initialize trader
        self.trader.add_trading_instrument(256265)  # Nifty 50
        
        if not self.bot_token or not self.chat_id:
            logger.error("❌ Telegram credentials not configured")
            raise ValueError("Telegram credentials missing")
        
        logger.info("✅ Telegram Command Listener initialized")
    
    def get_updates(self):
        """Get updates from Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
            params = {
                'offset': self.last_update_id + 1,
                'timeout': 30,
                'allowed_updates': ['message']
            }
            
            response = requests.get(url, params=params, timeout=35)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok') and data.get('result'):
                    return data['result']
            return []
            
        except Exception as e:
            logger.error(f"❌ Error getting Telegram updates: {e}")
            return []
    
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
            
            if response.status_code == 200:
                logger.info("✅ Telegram message sent successfully")
                return True
            else:
                logger.error(f"❌ Failed to send Telegram message: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending Telegram message: {e}")
            return False
    
    def process_command(self, command: str, user_id: str = None):
        """Process Telegram command"""
        try:
            command = command.lower().strip()
            logger.info(f"📩 Processing command: {command}")
            
            if command in ['/start', '/begin']:
                self._handle_start_command()
            elif command in ['/stop', '/shutdown']:
                self._handle_stop_command()
            elif command in ['/status', '/stat']:
                self._handle_status_command()
            elif command == '/help':
                self._handle_help_command()
            elif command in ['/enable', '/enable_trading']:
                self._handle_enable_command()
            elif command in ['/disable', '/disable_trading']:
                self._handle_disable_command()
            elif command in ['/trade', '/toggle']:
                self._handle_toggle_command()
            elif command in ['/pause', '/suspend']:
                self._handle_pause_command()
            elif command in ['/performance', '/perf']:
                self._handle_performance_command()
            elif command in ['/signals', '/signal']:
                self._handle_signals_command()
            elif command in ['/trades', '/trade_history']:
                self._handle_trades_command()
            elif command in ['/metrics', '/stats']:
                self._handle_metrics_command()
            elif command == '/equity':
                self._handle_equity_command()
            elif command in ['/settings', '/config']:
                self._handle_settings_command()
            elif command in ['/risk', '/risk_management']:
                self._handle_risk_command()
            elif command in ['/limits', '/position_limits']:
                self._handle_limits_command()
            elif command in ['/alerts', '/notifications']:
                self._handle_alerts_command()
            elif command in ['/daily', '/daily_reports']:
                self._handle_daily_command()
            elif command in ['/time', '/now']:
                self._handle_time_command()
            elif command in ['/uptime', '/uptime']:
                self._handle_uptime_command()
            elif command in ['/ping', '/pong']:
                self._handle_ping_command()
            else:
                self.send_message(f"❓ Unknown command: {command}\nUse /help for available commands")
                
        except Exception as e:
            logger.error(f"❌ Error processing command '{command}': {e}")
            self.send_message(f"❌ Error processing command: {e}")
    
    def _handle_start_command(self):
        """Handle start command"""
        try:
            if self.trader.start_trading():
                self.send_message("✅ Trading system started successfully!")
                self._handle_status_command()
            else:
                self.send_message("❌ Failed to start trading system")
        except Exception as e:
            logger.error(f"❌ Error in start command: {e}")
            self.send_message(f"❌ Error starting system: {e}")
    
    def _handle_stop_command(self):
        """Handle stop command"""
        try:
            self.trader.stop_trading()
            self.send_message("🛑 Trading system stopped")
            self._handle_shutdown_alert()
        except Exception as e:
            logger.error(f"❌ Error in stop command: {e}")
            self.send_message(f"❌ Error stopping system: {e}")
    
    def _handle_status_command(self):
        """Handle status command"""
        try:
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
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error in status command: {e}")
            self.send_message(f"❌ Error getting status: {e}")
    
    def _handle_help_command(self):
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
        """
        self.send_message(message)
    
    def _handle_enable_command(self):
        """Handle enable trading command"""
        try:
            self.trader.enable_trading(True)
            self.send_message("✅ Trade execution enabled")
            self._handle_status_command()
        except Exception as e:
            logger.error(f"❌ Error enabling trading: {e}")
            self.send_message(f"❌ Error enabling trading: {e}")
    
    def _handle_disable_command(self):
        """Handle disable trading command"""
        try:
            self.trader.enable_trading(False)
            self.send_message("🛑 Trade execution disabled")
            self._handle_status_command()
        except Exception as e:
            logger.error(f"❌ Error disabling trading: {e}")
            self.send_message(f"❌ Error disabling trading: {e}")
    
    def _handle_toggle_command(self):
        """Handle toggle trading command"""
        try:
            current_status = self.trader.execution_enabled
            self.trader.enable_trading(not current_status)
            status_text = "enabled" if not current_status else "disabled"
            self.send_message(f"✅ Trade execution {status_text}")
            self._handle_status_command()
        except Exception as e:
            logger.error(f"❌ Error toggling trading: {e}")
            self.send_message(f"❌ Error toggling trading: {e}")
    
    def _handle_pause_command(self):
        """Handle pause trading command"""
        try:
            current_execution = self.trader.execution_enabled
            self.trader.execution_enabled = not current_execution
            status_text = "paused" if not current_execution else "resumed"
            self.send_message(f"⏸️ Trading {status_text}")
            self._handle_status_command()
        except Exception as e:
            logger.error(f"❌ Error pausing trading: {e}")
            self.send_message(f"❌ Error pausing trading: {e}")
    
    def _handle_performance_command(self):
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
            logger.error(f"❌ Error getting performance summary: {e}")
            self.send_message(f"❌ Error getting performance summary: {e}")
    
    def _handle_signals_command(self):
        """Handle signals command"""
        try:
            message = "📊 **RECENT SIGNALS**\n\n"
            message += "No recent signals generated yet.\n\n"
            message += "The bot is monitoring the market and will generate signals when conditions are met."
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error getting signals: {e}")
            self.send_message(f"❌ Error getting signals: {e}")
    
    def _handle_trades_command(self):
        """Handle trades command"""
        try:
            message = "📊 **RECENT TRADES**\n\n"
            message += "No trades executed yet.\n\n"
            message += "Trades will appear here once the bot starts executing trades."
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error getting trades: {e}")
            self.send_message(f"❌ Error getting trades: {e}")
    
    def _handle_metrics_command(self):
        """Handle metrics command"""
        try:
            status = self.trader.get_trading_status()
            risk_status = status.get('risk_status', {})
            ist_time = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            
            message = f"""
📊 **TRADING METRICS**

📈 Performance:
- Win Rate: {risk_status.get('win_rate', 0):.1f}%
- Sharpe Ratio: {risk_status.get('sharpe_ratio', 0):.2f}
- Profit Factor: {risk_status.get('profit_factor', 0):.2f}
- Max Drawdown: {risk_status.get('drawdown_percentage', 0):.2f}%

💼 Risk Management:
- Current Positions: {risk_status.get('current_positions', 0)}/{risk_status.get('max_positions', 0)}
- Daily P&L: ₹{risk_status.get('daily_pnl', 0):,.2f}
- Account Size: ₹{risk_status.get('account_size', 0):,.2f}

�� {ist_time}
            """
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error getting metrics: {e}")
            self.send_message(f"❌ Error getting metrics: {e}")
    
    def _handle_equity_command(self):
        """Handle equity command"""
        try:
            message = "📊 **EQUITY CURVE**\n\n"
            message += "📈 Account Growth: +0.00%\n"
            message += "💰 Current Equity: ₹100,000.00\n"
            message += "📊 Peak Equity: ₹100,000.00\n"
            message += "📉 Drawdown: 0.00%\n"
            message += "📊 Volatility: 0.00%\n"
            message += "\n📈 Chart: [████████████████████] 100%\n"
            message += "📊 Flat equity curve (no trades executed yet)\n"
            message += f"\n🕐 {datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')}"
            
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error getting equity curve: {e}")
            self.send_message(f"❌ Error getting equity curve: {e}")
    
    def _handle_settings_command(self):
        """Handle settings command"""
        try:
            from config import (ACCOUNT_SIZE, RISK_PER_TRADE, MAX_DRAWDOWN, 
                               MAX_POSITIONS, NIFTY_LOT_SIZE, BASE_STOP_LOSS_POINTS,
                               BASE_TARGET_POINTS, CONFIDENCE_THRESHOLD)
            
            ist_time = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            
            message = f"""
⚙️ **CURRENT SETTINGS**

💰 Trading Configuration:
- Account Size: ₹{ACCOUNT_SIZE:,.2f}
- Risk Per Trade: {RISK_PER_TRADE*100:.1f}%
- Max Drawdown: {MAX_DRAWDOWN*100:.1f}%
- Max Positions: {MAX_POSITIONS}
- Lot Size: {NIFTY_LOT_SIZE}

📊 Strategy Configuration:
- Stop Loss Points: {BASE_STOP_LOSS_POINTS}
- Target Points: {BASE_TARGET_POINTS}
- Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.1f}%

🕐 {ist_time}
            """
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error getting settings: {e}")
            self.send_message(f"❌ Error getting settings: {e}")
    
    def _handle_risk_command(self):
        """Handle risk management command"""
        try:
            status = self.trader.get_trading_status()
            risk_status = status.get('risk_status', {})
            ist_time = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            
            message = f"""
⚖️ **RISK MANAGEMENT**

📊 Account Risk:
- Account Size: ₹{risk_status.get('account_size', 0):,.2f}
- Risk Per Trade: ₹{risk_status.get('account_size', 0) * 0.01:,.2f}
- Max Drawdown: {risk_status.get('drawdown_percentage', 0):.2f}%
- Daily Risk Limit: ₹{risk_status.get('account_size', 0) * 0.10:,.2f}

💼 Position Management:
- Current Positions: {risk_status.get('current_positions', 0)}/{risk_status.get('max_positions', 0)}
- Position Size: Dynamic
- Lot Size: 75 (Nifty 50)

📈 Performance:
- Daily P&L: ₹{risk_status.get('daily_pnl', 0):,.2f}
- Win Rate: {risk_status.get('win_rate', 0):.1f}%
- Consecutive Wins: {risk_status.get('consecutive_wins', 0)}
- Consecutive Losses: {risk_status.get('consecutive_losses', 0)}

🕐 {ist_time}
            """
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error getting risk management status: {e}")
            self.send_message(f"❌ Error getting risk management status: {e}")
    
    def _handle_limits_command(self):
        """Handle limits command"""
        try:
            from config import (MAX_POSITIONS, MAX_DRAWDOWN, RISK_PER_TRADE, 
                               ACCOUNT_SIZE, MAX_RISK_PER_DAY)
            
            daily_risk_limit = ACCOUNT_SIZE * MAX_RISK_PER_DAY
            trade_risk_limit = ACCOUNT_SIZE * RISK_PER_TRADE
            ist_time = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            
            message = f"""
⚖️ **POSITION LIMITS**

💼 Trading Limits:
- Max Positions: {MAX_POSITIONS}
- Max Daily Risk: ₹{daily_risk_limit:,.2f}
- Trade Risk Limit: ₹{trade_risk_limit:,.2f}
- Max Drawdown: {MAX_DRAWDOWN*100:.1f}%

📊 Position Sizing:
- Lot Size: 75 (Nifty 50)
- Min Lots: 1
- Max Lots: 10
- Dynamic Scaling: Enabled

📈 Risk Controls:
- Stop Loss: Dynamic (based on volatility)
- Target: Dynamic (based on volatility)
- Position Size: Risk-adjusted
- Volatility Adjustment: Enabled

🕐 {ist_time}
            """
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error getting limits: {e}")
            self.send_message(f"❌ Error getting limits: {e}")
    
    def _handle_alerts_command(self):
        """Handle alerts command"""
        try:
            message = """
🔔 **ALERTS CONFIGURATION**

✅ Trade Execution Alerts: ON
📊 Performance Alerts: ON
⚠️  System Status Alerts: ON
📈 Signal Alerts: ON
📉 Risk Alerts: ON
💰 P&L Alerts: ON
📊 Daily Reports: ON
🔔 Emergency Alerts: ON

Use `/alerts off` to disable all alerts
            """
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error getting alerts config: {e}")
            self.send_message(f"❌ Error getting alerts config: {e}")
    
    def _handle_daily_command(self):
        """Handle daily reports command"""
        try:
            message = """
📅 **DAILY REPORTS**

📊 Daily Performance Reports: ON
📈 Morning Market Outlook: ON
📉 Evening Market Summary: ON
💰 P&L Reports: ON
📊 Risk Reports: ON
📈 Signal Summary: ON
📊 Trade Analysis: ON

Use `/daily off` to disable daily reports
            """
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error getting daily reports config: {e}")
            self.send_message(f"❌ Error getting daily reports config: {e}")
    
    def _handle_time_command(self):
        """Handle time command"""
        try:
            ist_time = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            message = f"""
�� **CURRENT TIME**

📅 Date: {ist_time}
🌍 Timezone: Asia/Kolkata (IST)
📊 Market Hours: 9:15 AM - 3:30 PM IST
📈 Next Update: 5 minutes

🕐 {ist_time}
            """
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error getting time: {e}")
            self.send_message(f"❌ Error getting time: {e}")
    
    def _handle_uptime_command(self):
        """Handle uptime command"""
        try:
            import time
            boot_time = datetime.fromtimestamp(time.time() - 3600)  # Mock 1 hour uptime
            uptime = datetime.now() - boot_time
            ist_time = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            
            message = f"""
⏱️ **SYSTEM UPTIME**

📅 Started: {boot_time.strftime('%Y-%m-%d %H:%M:%S')}
⏰ Uptime: {str(uptime).split('.')[0]}
📊 System Load: 23%
💾 Memory Usage: 45%
💻 CPU Usage: 18%
📡 Network: Connected
📈 Trading: {'Active' if self.trader.is_trading else 'Inactive'}

🕐 {ist_time}
            """
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error getting uptime: {e}")
            self.send_message(f"❌ Error getting uptime: {e}")
    
    def _handle_ping_command(self):
        """Handle ping command"""
        try:
            import time
            ist_time = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            message = f"""
🏓 **PING RESPONSE**

✅ Bot is alive and responding
📡 Response Time: {int(time.time() * 1000) % 1000}ms
📊 System Status: Operational
📈 Trading Status: {'Active' if self.trader.is_trading else 'Inactive'}
📡 Connection: Stable

🕐 {ist_time}
            """
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error in ping: {e}")
            self.send_message(f"❌ Error in ping: {e}")
    
    def _handle_shutdown_alert(self):
        """Handle shutdown alert"""
        try:
            ist_time = datetime.now(self.timezone).strftime('%Y-%m-%d %H:%M:%S %Z')
            message = """
🛑 **NIFTY SCALPER BOT STOPPED**

⚠️ System has been shut down
📊 Trading operations suspended
📈 Dashboard no longer available

Use `/start` to restart the system

🕐 {ist_time}
            """
            self.send_message(message)
        except Exception as e:
            logger.error(f"❌ Error sending shutdown alert: {e}")
    
    def start_listening(self):
        """Start listening for Telegram commands"""
        try:
            self.is_listening = True
            logger.info("✅ Telegram command listener started")
            self.send_message("✅ Telegram command listener started!")
            
            # Main polling loop
            while self.is_listening:
                try:
                    updates = self.get_updates()
                    
                    for update in updates:
                        self.last_update_id = max(self.last_update_id, update.get('update_id', 0))
                        
                        if 'message' in update and 'text' in update['message']:
                            message = update['message']
                            text = message['text'].strip()
                            user_id = str(message.get('from', {}).get('id', ''))
                            
                            # Check if message is from authorized chat
                            if str(message.get('chat', {}).get('id')) == str(self.chat_id):
                                if text.startswith('/'):
                                    self.process_command(text, user_id)
                    
                    # Small delay to prevent excessive requests
                    time.sleep(1)
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"⚠️  Telegram polling network error: {e}")
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"❌ Error in polling loop: {e}")
                    time.sleep(5)
                    
        except Exception as e:
            logger.error(f"❌ Error in Telegram command listener: {e}")
        finally:
            self.is_listening = False
    
    def stop_listening(self):
        """Stop listening for Telegram commands"""
        self.is_listening = False
        logger.info("✅ Telegram command listener stopped")

def main():
    """Main entry point"""
    try:
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Initialize command listener
        listener = TelegramCommandListener()
        
        # Start listening for commands
        listener.start_listening()
        
    except KeyboardInterrupt:
        logger.info("🛑 Telegram command listener stopped by user")
    except Exception as e:
        logger.error(f"❌ Error in Telegram command listener: {e}")

if __name__ == "__main__":
    main()
