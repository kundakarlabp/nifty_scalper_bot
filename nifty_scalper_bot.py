
# nifty_scalper_bot.py - Production Ready Automatic Trading Bot with IST Timezone
import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from threading import Thread, Event, Lock
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, jsonify, request
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import io
import json
import signal
import sys
from typing import Dict, Optional, Tuple
import traceback
import pytz

# ================================
# Configuration & Initialization
# ================================
app = Flask(__name__)

# Global shutdown event and thread lock
shutdown_event = Event()
trade_lock = Lock()

# Indian Standard Time
IST = pytz.timezone('Asia/Kolkata')

def get_ist_now():
    """Get current time in IST"""
    return datetime.now(IST)

# ================================
# SignalEngine - Enhanced AI & Indicators
# ================================
class SignalEngine:
    def __init__(self):
        self.logger = logging.getLogger("SignalEngine")
        
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators with error handling"""
        try:
            # Simple moving averages
            df['ema_9'] = df['close'].ewm(span=9).mean()
            df['ema_21'] = df['close'].ewm(span=21).mean()
            
            # RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Simple MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macdsignal'] = df['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # VWAP
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            return df
        except Exception as e:
            self.logger.error(f"Indicator computation error: {e}")
            return df

    def generate_signal(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Generate trading signals with enhanced logic"""
        try:
            df = self.compute_indicators(df)
            if df.empty or len(df) < 50:
                return 0.0, 0.0
                
            last = df.iloc[-1]
            prev = df.iloc[-2]
            close = last['close']

            # Trend analysis
            trend_up = last['ema_9'] > last['ema_21']
            trend_strength = abs(last['ema_9'] - last['ema_21']) / last['ema_21']
            
            # Momentum indicators
            rsi_oversold = last['rsi'] < 30
            rsi_overbought = last['rsi'] > 70
            rsi_neutral = 30 <= last['rsi'] <= 70
            
            # MACD signals
            macd_bullish = last['macd'] > last['macdsignal'] and prev['macd'] <= prev['macdsignal']
            macd_bearish = last['macd'] < last['macdsignal'] and prev['macd'] >= prev['macdsignal']
            
            # Volume confirmation
            volume_surge = last['volume'] > last['volume_sma'] * 1.5
            
            # Bollinger Band signals
            bb_squeeze = (last['bb_upper'] - last['bb_lower']) / last['bb_middle'] < 0.1
            near_lower_bb = close <= last['bb_lower'] * 1.02
            near_upper_bb = close >= last['bb_upper'] * 0.98
            
            # Price action
            bullish_candle = last['close'] > last['open']
            bearish_candle = last['close'] < last['open']
            
            # CE (Call) scoring - Bullish signals
            buy_ce_score = 0.0
            if trend_up:
                buy_ce_score += 2.0 * min(trend_strength * 10, 1.0)
            if rsi_oversold or (rsi_neutral and last['rsi'] < 50):
                buy_ce_score += 1.5
            if macd_bullish:
                buy_ce_score += 1.8
            if volume_surge and bullish_candle:
                buy_ce_score += 1.2
            if near_lower_bb:
                buy_ce_score += 1.0
            if close > last['vwap']:
                buy_ce_score += 0.8
            if bb_squeeze:
                buy_ce_score += 0.5

            # PE (Put) scoring - Bearish signals
            buy_pe_score = 0.0
            if not trend_up:
                buy_pe_score += 2.0 * min(trend_strength * 10, 1.0)
            if rsi_overbought or (rsi_neutral and last['rsi'] > 50):
                buy_pe_score += 1.5
            if macd_bearish:
                buy_pe_score += 1.8
            if volume_surge and bearish_candle:
                buy_pe_score += 1.2
            if near_upper_bb:
                buy_pe_score += 1.0
            if close < last['vwap']:
                buy_pe_score += 0.8
            if bb_squeeze:
                buy_pe_score += 0.5

            return buy_ce_score, buy_pe_score
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            return 0.0, 0.0

# ================================
# BotController - Complete Trading Logic
# ================================
class BotController:
    def __init__(self):
        self.config = self._load_config()
        self.kite = None
        self.bot = None
        self.updater = None
        self.engine = SignalEngine()
        self.trade_logs = []
        self.current_trade = None
        self.logger = self.setup_logging()
        self.use_webhook = bool(os.getenv('WEBHOOK_URL'))
        
        # Initialize APIs
        self._initialize_kite()
        self._initialize_telegram()
        
    def _load_config(self) -> Dict:
        """Load and validate configuration"""
        config = {
            'ZERODHA_API_KEY': os.getenv('ZERODHA_API_KEY'),
            'ZERODHA_ACCESS_TOKEN': os.getenv('ZERODHA_ACCESS_TOKEN'),
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
            'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID', '0'),
            'DRY_RUN': os.getenv('DRY_RUN', 'true').lower() == 'true',
            'AUTO_TRADE': os.getenv('AUTO_TRADE', 'false').lower() == 'true',
            'MAX_LOSS_PER_DAY': float(os.getenv('MAX_LOSS_PER_DAY', '5000')),
            'MAX_TRADES_PER_DAY': int(os.getenv('MAX_TRADES_PER_DAY', '10')),
            'TRADE_QUANTITY': int(os.getenv('TRADE_QUANTITY', '75')),
            'SIGNAL_THRESHOLD': float(os.getenv('SIGNAL_THRESHOLD', '3.0'))
        }
        
        # Validate required fields
        required_fields = ['TELEGRAM_BOT_TOKEN']
        for field in required_fields:
            if not config[field]:
                raise ValueError(f"Missing required configuration: {field}")
                
        try:
            config['TELEGRAM_CHAT_ID'] = int(config['TELEGRAM_CHAT_ID'])
        except ValueError:
            config['TELEGRAM_CHAT_ID'] = 0
            
        return config
        
    def _initialize_kite(self):
        """Initialize Kite Connect API"""
        try:
            if self.config['ZERODHA_API_KEY'] and self.config['ZERODHA_ACCESS_TOKEN']:
                from kiteconnect import KiteConnect
                self.kite = KiteConnect(api_key=self.config['ZERODHA_API_KEY'])
                self.kite.set_access_token(self.config['ZERODHA_ACCESS_TOKEN'])
                
                # Test connection
                profile = self.kite.profile()
                self.logger.info(f"Kite connected successfully for user: {profile['user_name']}")
            else:
                self.logger.warning("Kite credentials not provided, running in simulation mode")
        except Exception as e:
            self.logger.error(f"Kite initialization failed: {e}")
            if not self.config['DRY_RUN']:
                self.logger.warning("Kite failed but continuing in DRY_RUN mode")
                
    def _initialize_telegram(self):
        """Initialize Telegram bot"""
        try:
            self.bot = telegram.Bot(token=self.config['TELEGRAM_BOT_TOKEN'])
            self.setup_telegram()
            self.logger.info("Telegram bot initialized successfully")
        except Exception as e:
            self.logger.error(f"Telegram initialization failed: {e}")
            raise
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler('bot.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("BotController")

    def setup_telegram(self):
        """Setup Telegram bot handlers"""
        try:
            if self.use_webhook:
                # Webhook mode - don't start polling
                self.logger.info("Using webhook mode for Telegram")
                # We'll handle updates via Flask route
            else:
                # Polling mode
                self.logger.info("Using polling mode for Telegram")
                self.updater = Updater(token=self.config['TELEGRAM_BOT_TOKEN'], use_context=True)
                dp = self.updater.dispatcher
                
                # Command handlers
                self._add_handlers(dp)
                
                # Error handler
                dp.add_error_handler(self.error_handler)
                
                # Start polling in a separate thread
                self.updater.start_polling()
                self.logger.info("Telegram polling started")
                
        except Exception as e:
            self.logger.error(f"Telegram setup failed: {e}")
            raise

    def _add_handlers(self, dp):
        """Add command handlers to dispatcher"""
        dp.add_handler(CommandHandler('start', self.cmd_start))
        dp.add_handler(CommandHandler('status', self.cmd_status))
        dp.add_handler(CommandHandler('summary', self.cmd_summary))
        dp.add_handler(CommandHandler('export', self.cmd_export))
        dp.add_handler(CommandHandler('trade', self.cmd_trade))
        dp.add_handler(CommandHandler('exit', self.cmd_exit))
        dp.add_handler(CommandHandler('sl', self.cmd_sl))
        dp.add_handler(CommandHandler('tp', self.cmd_tp))
        dp.add_handler(CommandHandler('auto', self.cmd_auto))
        dp.add_handler(CommandHandler('config', self.cmd_config))
        dp.add_handler(CommandHandler('help', self.cmd_help))

    def error_handler(self, update, context):
        """Handle Telegram errors"""
        self.logger.error(f"Telegram error: {context.error}")

    def register_webhook(self):
        """Register webhook for Telegram updates"""
        webhook_url = os.getenv('WEBHOOK_URL')
        if webhook_url and self.bot:
            try:
                # Delete existing webhook first
                self.bot.delete_webhook()
                time.sleep(1)
                
                # Set new webhook
                self.bot.set_webhook(url=f"{webhook_url}/telegram")
                self.logger.info(f"Webhook registered at {webhook_url}/telegram")
            except Exception as e:
                self.logger.error(f"Webhook registration failed: {e}")

    def process_telegram_update(self, update_data):
        """Process Telegram update for webhook mode"""
        try:
            if not self.use_webhook:
                return
                
            update = telegram.Update.de_json(update_data, self.bot)
            
            # Simple command processing for webhook mode
            if update.message and update.message.text:
                text = update.message.text
                chat_id = update.message.chat_id
                
                if text.startswith('/start'):
                    self.cmd_start_simple(chat_id)
                elif text.startswith('/status'):
                    self.cmd_status_simple(chat_id)
                elif text.startswith('/help'):
                    self.cmd_help_simple(chat_id)
                elif text.startswith('/trade'):
                    self.cmd_trade_simple(chat_id)
                elif text.startswith('/exit'):
                    self.cmd_exit_simple(chat_id)
                elif text.startswith('/auto'):
                    self.cmd_auto_simple(chat_id)
                else:
                    self._send_message(chat_id, "Unknown command. Type /help for available commands.")
                    
        except Exception as e:
            self.logger.error(f"Error processing Telegram update: {e}")

    # ================================
    # Simplified Command Handlers for Webhook Mode
    # ================================
    
    def cmd_start_simple(self, chat_id):
        """Simplified start command"""
        ist_time = get_ist_now().strftime("%Y-%m-%d %H:%M:%S IST")
        msg = f"🤖 Nifty Scalper Bot v2.0 is Live!\n\n" \
              f"🕐 Current IST Time: {ist_time}\n\n" \
              f"📊 Features:\n" \
              f"• Automated Nifty options trading\n" \
              f"• Advanced technical analysis\n" \
              f"• Risk management & P&L tracking\n" \
              f"• Real-time notifications\n\n" \
              f"📱 Commands:\n" \
              f"/help - Show all commands\n" \
              f"/status - Current bot status\n" \
              f"/trade - Execute manual trade\n" \
              f"/auto - Toggle auto-trading"
        self._send_message(chat_id, msg)

    def cmd_status_simple(self, chat_id):
        """Simplified status command"""
        try:
            ist_now = get_ist_now()
            today = ist_now.date()
            today_trades = [t for t in self.trade_logs if self._parse_ist_timestamp(t['timestamp']).date() == today]
            today_pnl = sum([t.get('pnl', 0) for t in today_trades])
            
            status_msg = f"🔄 Bot Status:\n\n"
            status_msg += f"🕐 IST Time: {ist_now.strftime('%H:%M:%S')}\n"
            status_msg += f"• Mode: {'🧪 DRY RUN' if self.config['DRY_RUN'] else '💰 LIVE TRADING'}\n"
            status_msg += f"• Auto-trading: {'✅ ON' if self.config['AUTO_TRADE'] else '❌ OFF'}\n"
            status_msg += f"• Market: {'🟢 OPEN' if self.is_market_hours() else '🔴 CLOSED'}\n"
            status_msg += f"• Today's trades: {len(today_trades)}/{self.config['MAX_TRADES_PER_DAY']}\n"
            status_msg += f"• Today's P&L: ₹{today_pnl:.2f}\n"
            
            if self.current_trade:
                trade = self.current_trade
                status_msg += f"\n📊 Active Trade:\n"
                status_msg += f"• Symbol: {trade['symbol']}\n"
                status_msg += f"• Type: {trade['type']}\n"
                status_msg += f"• Entry: ₹{trade['entry']:.2f}\n"
                status_msg += f"• Current P&L: ₹{trade.get('pnl', 0):.2f}"
            else:
                status_msg += f"\n💤 No active trades"
                
            self._send_message(chat_id, status_msg)
        except Exception as e:
            self.logger.error(f"Status command error: {e}")
            self._send_message(chat_id, "❌ Error retrieving status")

    def cmd_help_simple(self, chat_id):
        """Simplified help command"""
        msg = "📱 Available Commands:\n\n" \
              "🔄 Trading:\n" \
              "/trade - Execute manual trade\n" \
              "/exit - Exit current trade\n" \
              "/auto - Toggle auto-trading\n\n" \
              "📊 Monitoring:\n" \
              "/status - Current status\n" \
              "/help - Show this help\n" \
              "/start - Welcome message\n\n" \
              "🕐 Market Hours: 9:15 AM - 3:30 PM IST\n" \
              "📅 Trading Days: Monday to Friday"
        self._send_message(chat_id, msg)

    def cmd_trade_simple(self, chat_id):
        """Simplified trade command"""
        try:
            if self.current_trade:
                self._send_message(chat_id, "⚠️ Trade already active! Use /exit to close current trade first.")
                return

            if not self.is_market_hours():
                self._send_message(chat_id, "🔴 Market is closed. Trading hours: 9:15 AM - 3:30 PM IST")
                return

            if not self._check_daily_limits():
                return

            self._send_message(chat_id, "🔍 Analyzing market conditions...")
            
            df = self._get_market_data()
            if df is None:
                self._send_message(chat_id, "❌ Failed to get market data")
                return
            
            buy_ce_score, buy_pe_score = self.engine.generate_signal(df)
            entry_price = df['close'].iloc[-1]
            
            if buy_ce_score > buy_pe_score and buy_ce_score >= self.config['SIGNAL_THRESHOLD']:
                success = self._execute_trade('CE', entry_price, buy_ce_score)
                if success:
                    self._send_message(chat_id, f"✅ Manual CE trade executed!\n📊 Signal Score: {buy_ce_score:.2f}")
            elif buy_pe_score >= self.config['SIGNAL_THRESHOLD']:
                success = self._execute_trade('PE', entry_price, buy_pe_score)
                if success:
                    self._send_message(chat_id, f"✅ Manual PE trade executed!\n📊 Signal Score: {buy_pe_score:.2f}")
            else:
                self._send_message(chat_id, f"⚠️ No strong signals found.\n📊 CE: {buy_ce_score:.2f}, PE: {buy_pe_score:.2f}")

        except Exception as e:
            self.logger.error(f"Manual trade error: {e}")
            self._send_message(chat_id, "❌ Error executing manual trade")

    def cmd_exit_simple(self, chat_id):
        """Simplified exit command"""
        if not self.current_trade:
            self._send_message(chat_id, "💤 No active trade to exit")
            return
            
        self.exit_trade("Manual exit via command")
        self._send_message(chat_id, "✅ Trade exited manually")

    def cmd_auto_simple(self, chat_id):
        """Simplified auto command"""
        self.config['AUTO_TRADE'] = not self.config['AUTO_TRADE']
        state = "✅ enabled" if self.config['AUTO_TRADE'] else "❌ disabled"
        self._send_message(chat_id, f"🤖 Auto-trading {state}")
        self.logger.info(f"Auto-trading {state}")

    # ================================
    # Original Command Handlers (for polling mode)
    # ================================
    
    def cmd_start(self, update, context):
        """Start command handler"""
        self.cmd_start_simple(update.effective_chat.id)

    def cmd_status(self, update, context):
        """Status command handler"""
        self.cmd_status_simple(update.effective_chat.id)

    def cmd_help(self, update, context):
        """Help command handler"""
        self.cmd_help_simple(update.effective_chat.id)

    def cmd_config(self, update, context):
        """Configuration command handler"""
        try:
            ist_time = get_ist_now().strftime("%Y-%m-%d %H:%M:%S IST")
            config_msg = f"⚙️ Bot Configuration:\n\n"
            config_msg += f"🕐 Current IST Time: {ist_time}\n"
            config_msg += f"• Dry Run: {'✅' if self.config['DRY_RUN'] else '❌'}\n"
            config_msg += f"• Auto Trading: {'✅' if self.config['AUTO_TRADE'] else '❌'}\n"
            config_msg += f"• Max Daily Loss: ₹{self.config['MAX_LOSS_PER_DAY']}\n"
            config_msg += f"• Max Daily Trades: {self.config['MAX_TRADES_PER_DAY']}\n"
            config_msg += f"• Trade Quantity: {self.config['TRADE_QUANTITY']}\n"
            config_msg += f"• Signal Threshold: {self.config['SIGNAL_THRESHOLD']}\n"
            config_msg += f"• Market Hours: 9:15 AM - 3:30 PM IST\n"
            config_msg += f"• Market Status: {'🟢 OPEN' if self.is_market_hours() else '🔴 CLOSED'}"
            
            self._send_message(update.effective_chat.id, config_msg)
        except Exception as e:
            self.logger.error(f"Config command error: {e}")
            self._send_message(update.effective_chat.id, "❌ Error retrieving configuration")

    def cmd_summary(self, update, context):
        """Generate and send P&L summary with chart"""
        try:
            if not self.trade_logs:
                self._send_message(update.effective_chat.id, "📊 No completed trades yet!")
                return

            # Calculate statistics
            total_trades = len(self.trade_logs)
            winning_trades = len([t for t in self.trade_logs if t.get('pnl', 0) > 0])
            losing_trades = total_trades - winning_trades
            total_pnl = sum([t.get('pnl', 0) for t in self.trade_logs])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Send summary text
            summary_text = f"📊 Trading Performance Summary:\n\n"
            summary_text += f"📈 Overall Stats:\n"
            summary_text += f"• Total Trades: {total_trades}\n"
            summary_text += f"• Win Rate: {win_rate:.1f}%\n"
            summary_text += f"• Total P&L: ₹{total_pnl:.2f}\n"
            summary_text += f"• Avg P&L/Trade: ₹{total_pnl/total_trades:.2f}\n\n"
            
            summary_text += f"💰 Win/Loss Analysis:\n"
            summary_text += f"• Winning Trades: {winning_trades}\n"
            summary_text += f"• Losing Trades: {losing_trades}\n"

            self._send_message(update.effective_chat.id, summary_text)

        except Exception as e:
            self.logger.error(f"Summary generation error: {e}")
            self._send_message(update.effective_chat.id, "❌ Error generating summary")

    def cmd_export(self, update, context):
        """Export trade logs as CSV"""
        try:
            if not self.trade_logs:
                self._send_message(update.effective_chat.id, "📊 No trades to export!")
                return

            # Create simple export
            export_text = "📈 Trade History:\n\n"
            for i, trade in enumerate(self.trade_logs[-10:], 1):  # Last 10 trades
                trade_time = self._parse_ist_timestamp(trade['timestamp'])
                export_text += f"{i}. {trade['symbol']} - {trade['type']}\n"
                export_text += f"   Entry: ₹{trade['entry']:.2f}\n"
                export_text += f"   P&L: ₹{trade.get('pnl', 0):.2f}\n"
                export_text += f"   Time: {trade_time.strftime('%d-%m-%Y %H:%M IST')}\n\n"

            self._send_message(update.effective_chat.id, export_text)

        except Exception as e:
            self.logger.error(f"Export error: {e}")
            self._send_message(update.effective_chat.id, "❌ Error exporting trades")

    def cmd_trade(self, update, context):
        """Manual trade execution"""
        self.cmd_trade_simple(update.effective_chat.id)

    def cmd_exit(self, update, context):
        """Exit current trade"""
        self.cmd_exit_simple(update.effective_chat.id)

    def cmd_sl(self, update, context):
        """Set/view stop loss"""
        if not self.current_trade:
            self._send_message(update.effective_chat.id, "💤 No active trade")
            return
            
        try:
            if context.args:
                new_sl = float(context.args[0])
                self.current_trade['sl'] = new_sl
                self._send_message(update.effective_chat.id, f"✅ Stop-loss updated to ₹{new_sl:.2f}")
                self.logger.info(f"Stop-loss updated to {new_sl}")
            else:
                current_sl = self.current_trade.get('sl', 'Not set')
                self._send_message(update.effective_chat.id, f"📊 Current SL: ₹{current_sl}\n💡 Usage: /sl <price>")
        except ValueError:
            self._send_message(update.effective_chat.id, "❌ Invalid price format. Usage: /sl <price>")

    def cmd_tp(self, update, context):
        """Set/view take profit"""
        if not self.current_trade:
            self._send_message(update.effective_chat.id, "💤 No active trade")
            return
            
        try:
            if context.args:
                new_tp = float(context.args[0])
                self.current_trade['tp'] = new_tp
                self._send_message(update.effective_chat.id, f"✅ Take-profit updated to ₹{new_tp:.2f}")
                self.logger.info(f"Take-profit updated to {new_tp}")
            else:
                current_tp = self.current_trade.get('tp', 'Not set')
                self._send_message(update.effective_chat.id, f"📊 Current TP: ₹{current_tp}\n💡 Usage: /tp <price>")
        except ValueError:
            self._send_message(update.effective_chat.id, "❌ Invalid price format. Usage: /tp <price>")

    def cmd_auto(self, update, context):
        """Toggle auto-trading"""
        self.cmd_auto_simple(update.effective_chat.id)

    # ================================
    # Helper Methods
    # ================================
    
    def _send_message(self, chat_id: int, message: str):
        """Send Telegram message with error handling"""
        try:
            self.bot.send_message(chat_id=chat_id, text=message)
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")

    def _parse_ist_timestamp(self, timestamp_str: str):
        """Parse timestamp string and convert to IST"""
        try:
            # Parse the ISO format timestamp
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            # Convert to IST
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=pytz.UTC)
            return dt.astimezone(IST)
        except:
            # Fallback to current IST time
            return get_ist_now()

    def _get_market_data(self) -> Optional[pd.DataFrame]:
        """Get market data with error handling"""
        try:
            if self.kite:
                # Nifty 50 instrument token
                instrument_token = 256265
                ist_now = get_ist_now()
                from_date = ist_now - timedelta(days=5)
                to_date = ist_now
                
                data = self.kite.historical_data(instrument_token, from_date, to_date, '15minute')
                df = pd.DataFrame(data)
                
                if df.empty:
                    self.logger.error("Empty market data received")
                    return None
                    
                return df
            else:
                # Generate dummy data for testing
                ist_now = get_ist_now()
                dates = pd.date_range(start=ist_now - timedelta(days=1), end=ist_now, freq='15min')
                np.random.seed(42)
                base_price = 24000
                prices = base_price + np.cumsum(np.random.randn(len(dates)) * 10)
                
                df = pd.DataFrame({
                    'date': dates,
                    'open': prices,
                    'high': prices + np.random.rand(len(dates)) * 20,
                    'low': prices - np.random.rand(len(dates)) * 20,
                    'close': prices,
                    'volume': np.random.randint(1000, 10000, len(dates))
                })
                return df
                
        except Exception as e:
            self.logger.error(f"Market data error: {e}")
            return None

    def _check_daily_limits(self) -> bool:
        """Check if daily trading limits are exceeded"""
        ist_now = get_ist_now()
        today = ist_now.date()
        today_trades = [t for t in self.trade_logs if self._parse_ist_timestamp(t['timestamp']).date() == today]
        today_pnl = sum([t.get('pnl', 0) for t in today_trades])
        
        # Check trade count limit
        if len(today_trades) >= self.config['MAX_TRADES_PER_DAY']:
            self._send_message(self.config['TELEGRAM_CHAT_ID'], 
                f"🚫 Daily trade limit reached ({self.config['MAX_TRADES_PER_DAY']})")
            return False
            
        # Check loss limit
        if today_pnl <= -self.config['MAX_LOSS_PER_DAY']:
            self._send_message(self.config['TELEGRAM_CHAT_ID'], 
                f"🚫 Daily loss limit reached (₹{self.config['MAX_LOSS_PER_DAY']})")
            return False
            
        return True

    def _execute_trade(self, trade_type: str, entry_price: float, score: float) -> bool:
        """Execute a trade with proper risk management"""
        try:
            with trade_lock:
                # Calculate strike price
                strike = self._get_nearest_strike(entry_price)
                symbol = f"NIFTY{strike}{trade_type}"
                
                # Calculate SL and TP
                if trade_type == 'CE':
                    sl = entry_price * 0.97  # 3% SL
                    tp = entry_price * 1.06  # 6% TP
                else:
                    sl = entry_price * 1.03  # 3% SL
                    tp = entry_price * 0.94  # 6% TP
                    
                # Create trade record with IST timestamp
                ist_now = get_ist_now()
                self.current_trade = {
                    'symbol': symbol,
                    'entry': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'type': trade_type,
                    'score': score,
                    'timestamp': ist_now.isoformat(),
                    'pnl': 0,
                    'highest': entry_price if trade_type == 'CE' else entry_price,
                    'lowest': entry_price if trade_type == 'PE' else entry_price,
                    'quantity': self.config['TRADE_QUANTITY']
                }
                
                # Place actual order if not in dry run
                if not self.config['DRY_RUN'] and self.kite:
                    try:
                        order_id = self.kite.place_order(
                            tradingsymbol=symbol,
                            exchange=self.kite.EXCHANGE_NFO,
                            transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                            quantity=self.config['TRADE_QUANTITY'],
                            order_type=self.kite.ORDER_TYPE_MARKET,
                            product=self.kite.PRODUCT_MIS,
                            variety=self.kite.VARIETY_REGULAR
                        )
                        self.current_trade['order_id'] = order_id
                        self.logger.info(f"Order placed: {order_id}")
                    except Exception as e:
                        self.logger.error(f"Order placement failed: {e}")
                        self.current_trade = None
                        return False
                
                # Send notification
                notification = f"📈 {'🟢 BUY CE' if trade_type == 'CE' else '🔴 BUY PE'}\n\n" \
                              f"🎯 Symbol: {symbol}\n" \
                              f"💰 Entry: ₹{entry_price:.2f}\n" \
                              f"📊 Score: {score:.2f}\n" \
                              f"🎯 Target: ₹{tp:.2f}\n" \
                              f"🛡️ Stop Loss: ₹{sl:.2f}\n" \
                              f"📦 Quantity: {self.config['TRADE_QUANTITY']}\n" \
                              f"🕐 Time: {ist_now.strftime('%H:%M:%S IST')}\n" \
                              f"🧪 Mode: {'DRY RUN' if self.config['DRY_RUN'] else 'LIVE'}"
                
                self._send_message(self.config['TELEGRAM_CHAT_ID'], notification)
                self.logger.info(f"Trade executed: {symbol} at {entry_price}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            return False

    def _get_nearest_strike(self, price: float) -> int:
        """Get nearest option strike price"""
        return int(round(price / 50) * 50)

    def is_market_hours(self) -> bool:
        """Check if market is open (Indian Standard Time)"""
        ist_now = get_ist_now()
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if ist_now.weekday() > 4:  # Saturday or Sunday
            return False
            
        current_time = ist_now.time()
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()
        
        return market_open <= current_time <= market_close

    def exit_trade(self, reason: str = "Manual exit"):
        """Exit current trade with P&L calculation"""
        if not self.current_trade:
            return
            
        try:
            with trade_lock:
                # Calculate final P&L (simplified for dry run)
                if self.config['DRY_RUN'] or not self.kite:
                    # Simulate some P&L for dry run
                    import random
                    pnl_factor = random.uniform(-0.05, 0.08)  # -5% to +8%
                    self.current_trade['pnl'] = self.current_trade['entry'] * self.config['TRADE_QUANTITY'] * pnl_factor
                    self.current_trade['exit_price'] = self.current_trade['entry'] * (1 + pnl_factor)
                else:
                    # Get actual market price
                    symbol = self.current_trade['symbol']
                    try:
                        ticker = self.kite.quote(f"NFO:{symbol}")
                        last_price = ticker[f"NFO:{symbol}"]["last_price"]
                        
                        if self.current_trade['type'] == 'CE':
                            self.current_trade['pnl'] = (last_price - self.current_trade['entry']) * self.config['TRADE_QUANTITY']
                        else:
                            self.current_trade['pnl'] = (self.current_trade['entry'] - last_price) * self.config['TRADE_QUANTITY']
                            
                        self.current_trade['exit_price'] = last_price
                    except Exception as e:
                        self.logger.error(f"Price fetch error during exit: {e}")
                        self.current_trade['pnl'] = 0
                        self.current_trade['exit_price'] = self.current_trade['entry']
                    
                # Complete trade record with IST timestamp
                ist_now = get_ist_now()
                self.current_trade['exit_time'] = ist_now.isoformat()
                self.current_trade['exit_reason'] = reason
                
                # Log trade
                self.trade_logs.append(self.current_trade.copy())
                
                # Send notification
                pnl = self.current_trade['pnl']
                pnl_emoji = "💰" if pnl > 0 else "💸" if pnl < 0 else "➖"
                
                notification = f"✅ Trade Closed\n\n" \
                              f"🎯 Symbol: {self.current_trade['symbol']}\n" \
                              f"📊 Entry: ₹{self.current_trade['entry']:.2f}\n" \
                              f"📊 Exit: ₹{self.current_trade['exit_price']:.2f}\n" \
                              f"{pnl_emoji} P&L: ₹{pnl:.2f}\n" \
                              f"📝 Reason: {reason}\n" \
                              f"⏱️ Duration: {self._get_trade_duration()}\n" \
                              f"🕐 Exit Time: {ist_now.strftime('%H:%M:%S IST')}"
                
                self._send_message(self.config['TELEGRAM_CHAT_ID'], notification)
                self.logger.info(f"Trade closed: {self.current_trade['symbol']}, P&L: {pnl:.2f}")
                
                # Clear current trade
                self.current_trade = None
                
        except Exception as e:
            self.logger.error(f"Exit trade error: {e}")

    def _get_trade_duration(self) -> str:
        """Get formatted trade duration"""
        if not self.current_trade:
            return "N/A"
            
        try:
            entry_time = self._parse_ist_timestamp(self.current_trade['timestamp'])
            duration = get_ist_now() - entry_time
            return str(duration).split('.')[0]  # Remove microseconds
        except:
            return "N/A"

    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down bot...")
        
        # Exit any open trade
        if self.current_trade:
            self.exit_trade("Bot shutdown")
            
        # Stop Telegram updater
        if self.updater:
            self.updater.stop()
            
        # Set shutdown event
        shutdown_event.set()
        
        self.logger.info("Bot shutdown complete")

# ================================
# Trading Functions (moved outside main)
# ================================

def auto_trade_job(controller):
    """Automatically find and execute trades"""
    try:
        if not controller.is_market_hours():
            return
            
        if controller.current_trade:
            controller.logger.debug("Trade already active, skipping auto-trade")
            return
            
        # Check daily limits
        if not controller._check_daily_limits():
            return
            
        # Get market data
        df = controller._get_market_data()
        if df is None:
            return
            
        # Generate signals
        buy_ce_score, buy_pe_score = controller.engine.generate_signal(df)
        entry_price = df['close'].iloc[-1]
        
        # Execute trade if signal is strong enough
        if buy_ce_score >= controller.config['SIGNAL_THRESHOLD'] and buy_ce_score > buy_pe_score:
            controller._execute_trade('CE', entry_price, buy_ce_score)
        elif buy_pe_score >= controller.config['SIGNAL_THRESHOLD']:
            controller._execute_trade('PE', entry_price, buy_pe_score)
            
    except Exception as e:
        controller.logger.error(f"Auto-trade job error: {e}")

def monitor_trades(controller):
    """Continuously monitor open trades for exits"""
    while not shutdown_event.is_set():
        try:
            time.sleep(30)  # Check every 30 seconds
            
            if not controller.current_trade or not controller.is_market_hours():
                continue
                
            # Get current price (simplified for dry run)
            if controller.config['DRY_RUN'] or not controller.kite:
                # Simulate price movement
                import random
                price_change = random.uniform(-0.02, 0.02)  # ±2%
                current_price = controller.current_trade['entry'] * (1 + price_change)
            else:
                # Get actual price
                try:
                    symbol = controller.current_trade['symbol']
                    ticker = controller.kite.quote(f"NFO:{symbol}")
                    current_price = ticker[f"NFO:{symbol}"]["last_price"]
                except Exception as e:
                    controller.logger.error(f"Price fetch error: {e}")
                    continue
            
            # Update P&L and tracking
            with trade_lock:
                trade = controller.current_trade
                if not trade:
                    continue
                    
                if trade['type'] == 'CE':
                    trade['pnl'] = (current_price - trade['entry']) * controller.config['TRADE_QUANTITY']
                    trade['highest'] = max(trade['highest'], current_price)
                else:
                    trade['pnl'] = (trade['entry'] - current_price) * controller.config['TRADE_QUANTITY']
                    trade['lowest'] = min(trade['lowest'], current_price)
                
                # Check exit conditions
                exit_reason = None
                
                # Stop Loss
                if ((trade['type'] == 'CE' and current_price <= trade['sl']) or 
                    (trade['type'] == 'PE' and current_price >= trade['sl'])):
                    exit_reason = "Stop Loss Hit"
                    
                # Take Profit
                elif ((trade['type'] == 'CE' and current_price >= trade['tp']) or 
                      (trade['type'] == 'PE' and current_price <= trade['tp'])):
                    exit_reason = "Take Profit Hit"
                    
                # Trailing Stop Loss (3% from highest/lowest)
                elif (trade['type'] == 'CE' and current_price <= trade['highest'] * 0.97):
                    exit_reason = "Trailing Stop Loss"
                elif (trade['type'] == 'PE' and current_price >= trade['lowest'] * 1.03):
                    exit_reason = "Trailing Stop Loss"
                    
                # Time-based exit (if trade is open for more than 4 hours)
                entry_time = controller._parse_ist_timestamp(trade['timestamp'])
                if get_ist_now() - entry_time > timedelta(hours=4):
                    exit_reason = "Time-based Exit"
                    
                # Exit if condition met
                if exit_reason:
                    controller.exit_trade(exit_reason)
                    
        except Exception as e:
            controller.logger.error(f"Trade monitoring error: {e}")
            time.sleep(60)  # Wait longer on error

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    if 'controller' in globals():
        controller.logger.info(f"Received signal {signum}")
        controller.shutdown()
    sys.exit(0)

# ================================
# Flask Routes
# ================================

@app.route('/')
def home():
    """Health check endpoint"""
    try:
        ist_time = get_ist_now().strftime("%Y-%m-%d %H:%M:%S IST")
        return jsonify({
            "status": "running",
            "version": "2.0",
            "ist_time": ist_time,
            "auto_trading": controller.config['AUTO_TRADE'],
            "market_hours": controller.is_market_hours(),
            "current_trade": bool(controller.current_trade),
            "total_trades": len(controller.trade_logs),
            "dry_run": controller.config['DRY_RUN'],
            "webhook_mode": controller.use_webhook
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/telegram', methods=['POST'])
def telegram_webhook():
    """Handle Telegram webhook updates"""
    try:
        update_data = request.get_json(force=True)
        controller.process_telegram_update(update_data)
        return jsonify({"status": "ok"})
    except Exception as e:
        controller.logger.error(f"Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/trades')
def get_trades():
    """API endpoint to get trade history"""
    try:
        return jsonify({
            "trades": controller.trade_logs,
            "current_trade": controller.current_trade,
            "total_pnl": sum([t.get('pnl', 0) for t in controller.trade_logs]),
            "ist_time": get_ist_now().strftime("%Y-%m-%d %H:%M:%S IST")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================================
# Initialization & Startup
# ================================

if __name__ == '__main__':
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize controller
        controller = BotController()
        
        # Register webhook if needed
        if controller.use_webhook:
            controller.register_webhook()
        
        # Send startup notification
        ist_time = get_ist_now().strftime("%Y-%m-%d %H:%M:%S IST")
        startup_msg = f"🚀 Nifty Scalper Bot v2.0 Started!\n\n" \
                     f"🕐 IST Time: {ist_time}\n\n" \
                     f"⚙️ Configuration:\n" \
                     f"• Mode: {'🧪 DRY RUN' if controller.config['DRY_RUN'] else '💰 LIVE TRADING'}\n" \
                     f"• Auto-trading: {'✅ ON' if controller.config['AUTO_TRADE'] else '❌ OFF'}\n" \
                     f"• Market: {'🟢 OPEN' if controller.is_market_hours() else '🔴 CLOSED'}\n" \
                     f"• Signal Threshold: {controller.config['SIGNAL_THRESHOLD']}\n" \
                     f"• Max Daily Trades: {controller.config['MAX_TRADES_PER_DAY']}\n" \
                     f"• Max Daily Loss: ₹{controller.config['MAX_LOSS_PER_DAY']}\n" \
                     f"• Market Hours: 9:15 AM - 3:30 PM IST\n" \
                     f"• Telegram Mode: {'Webhook' if controller.use_webhook else 'Polling'}"
        
        if controller.config['TELEGRAM_CHAT_ID']:
            controller._send_message(controller.config['TELEGRAM_CHAT_ID'], startup_msg)
        
        # Start background threads
        monitor_thread = Thread(target=monitor_trades, args=(controller,), daemon=True)
        monitor_thread.start()
        
        # Start auto-trading scheduler if enabled
        if controller.config['AUTO_TRADE']:
            def auto_trade_scheduler():
                while not shutdown_event.is_set():
                    try:
                        if controller.is_market_hours() and not controller.current_trade:
                            auto_trade_job(controller)
                        time.sleep(300)  # Check every 5 minutes
                    except Exception as e:
                        controller.logger.error(f"Auto-trade scheduler error: {e}")
                        time.sleep(60)  # Wait 1 minute on error
                        
            auto_thread = Thread(target=auto_trade_scheduler, daemon=True)
            auto_thread.start()
        
        # Start Flask server
        try:
            from waitress import serve
            port = int(os.getenv('PORT', 8000))
            controller.logger.info(f"Starting server on port {port}")
            serve(app, host='0.0.0.0', port=port)
        except ImportError:
            # Fallback to Flask dev server
            port = int(os.getenv('PORT', 8000))
            controller.logger.warning("Waitress not available, using Flask dev server")
            app.run(host='0.0.0.0', port=port, debug=False)
            
    except Exception as e:
        print(f"Startup error: {e}")
        traceback.print_exc()
        sys.exit(1)
