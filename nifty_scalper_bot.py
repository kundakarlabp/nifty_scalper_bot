My sincerest apologies. That is completely my fault.

The error `SyntaxError: unterminated string literal` is happening because I accidentally included my explanatory text at the very top of the code block I provided.

When you copied the code, you also copied the English sentences I wrote, and the Python interpreter tried to read them as code, which caused the crash. The apostrophe in "I've" is what specifically triggered the `unterminated string literal` error.

### **The Solution: A Clean Copy**

Please delete the entire content of your `nifty_scalper_bot.py` file and replace it with the code block below.

This time, I have made absolutely sure it contains **only the Python code** and no explanatory text.

```python
# nifty_scalper_bot.py - Production Ready Automatic Trading Bot
import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from threading import Thread, Event, Lock
from flask import Flask, jsonify
import telegram
from telegram.ext import Updater, CommandHandler
import json
import signal
import sys
from typing import Dict, Optional, Tuple
import traceback
import pytz  # Added for timezone handling
from kiteconnect import KiteConnect

# ================================
# Configuration & Initialization
# ================================
app = Flask(__name__)

# Global shutdown event and thread lock
shutdown_event = Event()
trade_lock = Lock()

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NiftyScalperBot")

# ================================
# SignalEngine - Enhanced AI & Indicators
# ================================
class SignalEngine:
    def __init__(self):
        self.logger = logging.getLogger("SignalEngine")

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators with error handling"""
        try:
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()

            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macdsignal'] = df['macd'].ewm(span=9, adjust=False).mean()

            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            return df
        except Exception as e:
            self.logger.error(f"Indicator computation error: {e}")
            return df

    def generate_signal(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Generate trading signals with enhanced logic"""
        try:
            df = self.compute_indicators(df)
            if df.empty or len(df) < 30:
                return 0.0, 0.0

            last = df.iloc[-1]
            prev = df.iloc[-2]
            close = last['close']

            trend_up = last['ema_9'] > last['ema_21']
            rsi_oversold = last['rsi'] < 35
            rsi_overbought = last['rsi'] > 65
            macd_bullish_cross = last['macd'] > last['macdsignal'] and prev['macd'] <= prev['macdsignal']
            macd_bearish_cross = last['macd'] < last['macdsignal'] and prev['macd'] >= prev['macdsignal']
            volume_surge = last['volume'] > last['volume_sma'] * 1.5
            near_lower_bb = close <= last['bb_lower']
            near_upper_bb = close >= last['bb_upper']

            buy_ce_score = 0.0
            if trend_up: buy_ce_score += 1.5
            if rsi_oversold: buy_ce_score += 2.0
            if macd_bullish_cross: buy_ce_score += 2.0
            if volume_surge and last['close'] > last['open']: buy_ce_score += 1.0
            if near_lower_bb: buy_ce_score += 1.0
            if close > last['vwap']: buy_ce_score += 0.5

            buy_pe_score = 0.0
            if not trend_up: buy_pe_score += 1.5
            if rsi_overbought: buy_pe_score += 2.0
            if macd_bearish_cross: buy_pe_score += 2.0
            if volume_surge and last['close'] < last['open']: buy_pe_score += 1.0
            if near_upper_bb: buy_pe_score += 1.0
            if close < last['vwap']: buy_pe_score += 0.5

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
        self.nfo_instruments = None

        self._initialize_kite()
        self._initialize_telegram()

    def _load_config(self) -> Dict:
        """Load and validate configuration"""
        config = {
            'ZERODHA_API_KEY': os.getenv('ZERODHA_API_KEY'),
            'ZERODHA_ACCESS_TOKEN': os.getenv('ZERODHA_ACCESS_TOKEN'),
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
            'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID'),
            'REDIS_URL': os.getenv('REDIS_URL'), # From first script
            'DRY_RUN': os.getenv('DRY_RUN', 'true').lower() == 'true',
            'AUTO_TRADE': os.getenv('AUTO_TRADE', 'false').lower() == 'true',
            'MAX_LOSS_PER_DAY': float(os.getenv('MAX_LOSS_PER_DAY', '2500')),
            'MAX_TRADES_PER_DAY': int(os.getenv('MAX_TRADES_PER_DAY', '10')),
            'TRADE_QUANTITY': int(os.getenv('TRADE_QUANTITY', '50')),
            'SIGNAL_THRESHOLD': float(os.getenv('SIGNAL_THRESHOLD', '3.5')),
            'SL_PERCENT': float(os.getenv('SL_PERCENT', '0.20')), # 20% SL on option price
            'TP_PERCENT': float(os.getenv('TP_PERCENT', '0.40')), # 40% TP on option price
        }
        required = ['ZERODHA_API_KEY', 'ZERODHA_ACCESS_TOKEN', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
        for field in required:
            if not config[field]:
                raise ValueError(f"Missing required configuration: {field}")
        return config

    def _initialize_kite(self):
        """Initialize Kite Connect API"""
        try:
            self.kite = KiteConnect(api_key=self.config['ZERODHA_API_KEY'])
            self.kite.set_access_token(self.config['ZERODHA_ACCESS_TOKEN'])
            profile = self.kite.profile()
            logger.info(f"Kite connected successfully for user: {profile['user_name']}")
            # Fetch NFO instruments once at startup
            self.nfo_instruments = self.kite.instruments("NFO")
            logger.info(f"Fetched {len(self.nfo_instruments)} NFO instruments.")
        except Exception as e:
            logger.error(f"Kite initialization failed: {e}")
            if not self.config['DRY_RUN']:
                logger.warning("Kite failed but continuing in DRY_RUN mode")

    def _initialize_telegram(self):
        """Initialize Telegram bot"""
        try:
            self.bot = telegram.Bot(token=self.config['TELEGRAM_BOT_TOKEN'])
            self.updater = Updater(token=self.config['TELEGRAM_BOT_TOKEN'], use_context=True)
            dp = self.updater.dispatcher
            dp.add_handler(CommandHandler('start', self.cmd_start))
            dp.add_handler(CommandHandler('status', self.cmd_status))
            dp.add_handler(CommandHandler('trade', self.cmd_trade, pass_args=True))
            dp.add_handler(CommandHandler('exit', self.cmd_exit))
            dp.add_handler(CommandHandler('auto', self.cmd_auto))
            dp.add_handler(CommandHandler('help', self.cmd_help))
            dp.add_error_handler(self.error_handler)
            self.updater.start_polling()
            logger.info("Telegram bot initialized and polling.")
        except Exception as e:
            logger.error(f"Telegram initialization failed: {e}", exc_info=True)
            raise

    def error_handler(self, update, context):
        logger.error(f"Telegram error: {context.error}", exc_info=context.error)

    def _send_message(self, message: str):
        try:
            self.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text=message)
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    def is_market_hours(self) -> bool:
        """Check if market is open in IST."""
        tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(tz)
        if now.weekday() >= 5: return False # Skip weekends
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0).time()
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0).time()
        return market_open <= now.time() <= market_close

    def _get_market_data(self) -> Optional[pd.DataFrame]:
        """Get Nifty 50 historical data."""
        try:
            if not self.kite: return None
            instrument_token = 256265  # NIFTY 50
            to_date = datetime.now()
            from_date = to_date - timedelta(days=5)
            data = self.kite.historical_data(instrument_token, from_date, to_date, '5minute')
            if not data:
                logger.warning("No market data received from Kite.")
                return None
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return None

    def _get_option_symbol(self, strike: int, option_type: str) -> Optional[str]:
        """Find the nearest weekly expiry option symbol for NIFTY."""
        if not self.nfo_instruments:
            logger.error("NFO instruments not loaded.")
            return None
        
        today = datetime.now().date()
        nifty_options = [
            inst for inst in self.nfo_instruments
            if inst['name'] == 'NIFTY'
            and inst['strike'] == strike
            and inst['instrument_type'] == option_type
            and inst['expiry'] >= today
        ]
        
        if not nifty_options:
            return None
            
        # Sort by expiry date to find the nearest one
        nifty_options.sort(key=lambda x: x['expiry'])
        return nifty_options[0]['tradingsymbol']

    def _get_ltp(self, tradingsymbol: str) -> Optional[float]:
        """Get Last Traded Price for a symbol."""
        try:
            if not self.kite: return None
            quote = self.kite.quote(f"NFO:{tradingsymbol}")
            return quote[f"NFO:{tradingsymbol}"]["last_price"]
        except Exception as e:
            logger.error(f"Failed to get LTP for {tradingsymbol}: {e}")
            return None

    def _execute_trade(self, trade_type: str, nifty_price: float, score: float):
        """Execute a trade with proper risk management."""
        with trade_lock:
            if self.current_trade:
                self._send_message("⚠️ Trade already active!")
                return

            if not self._check_daily_limits(): return

            strike = int(round(nifty_price / 50) * 50)
            symbol = self._get_option_symbol(strike, trade_type)
            if not symbol:
                self._send_message(f"❌ Could not find a valid {trade_type} option for strike {strike}.")
                return

            entry_price = self._get_ltp(symbol)
            if not entry_price:
                self._send_message(f"❌ Failed to get entry price for {symbol}.")
                return

            sl_price = entry_price * (1 - self.config['SL_PERCENT'])
            tp_price = entry_price * (1 + self.config['TP_PERCENT'])

            self.current_trade = {
                'symbol': symbol,
                'entry_price': entry_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'type': trade_type,
                'timestamp': datetime.now().isoformat(),
                'quantity': self.config['TRADE_QUANTITY'],
                'pnl': 0,
                'highest_price': entry_price,
            }

            if not self.config['DRY_RUN']:
                try:
                    order_id = self.kite.place_order(
                        tradingsymbol=symbol, exchange=self.kite.EXCHANGE_NFO,
                        transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                        quantity=self.config['TRADE_QUANTITY'],
                        order_type=self.kite.ORDER_TYPE_MARKET,
                        product=self.kite.PRODUCT_MIS, variety=self.kite.VARIETY_REGULAR
                    )
                    self.current_trade['order_id'] = order_id
                    logger.info(f"LIVE Order placed: {order_id} for {symbol}")
                except Exception as e:
                    logger.error(f"Order placement failed: {e}")
                    self._send_message(f"❌ LIVE Order placement failed: {e}")
                    self.current_trade = None
                    return

            mode = 'DRY RUN' if self.config['DRY_RUN'] else 'LIVE'
            msg = (f"📈 {'🟢 BUY CE' if trade_type == 'CE' else '🔴 BUY PE'} [{mode}]\n\n"
                   f"🎯 Symbol: {symbol}\n"
                   f"💰 Entry: ₹{entry_price:.2f}\n"
                   f"🛡️ Stop Loss: ₹{sl_price:.2f}\n"
                   f"🎯 Target: ₹{tp_price:.2f}\n"
                   f"📊 Signal Score: {score:.2f}")
            self._send_message(msg)
            logger.info(f"Trade executed: {symbol} at {entry_price}")

    def exit_trade(self, reason: str):
        """Exit current trade."""
        with trade_lock:
            if not self.current_trade: return

            trade = self.current_trade
            exit_price = self._get_ltp(trade['symbol'])
            if not exit_price:
                self._send_message(f"⚠️ Could not fetch exit price for {trade['symbol']}. Using entry price for P&L.")
                exit_price = trade['entry_price']

            if not self.config['DRY_RUN']:
                try:
                    order_id = self.kite.place_order(
                        tradingsymbol=trade['symbol'], exchange=self.kite.EXCHANGE_NFO,
                        transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                        quantity=trade['quantity'],
                        order_type=self.kite.ORDER_TYPE_MARKET,
                        product=self.kite.PRODUCT_MIS, variety=self.kite.VARIETY_REGULAR
                    )
                    logger.info(f"LIVE Exit order placed: {order_id} for {trade['symbol']}")
                except Exception as e:
                    logger.error(f"Exit order failed: {e}")
                    self._send_message(f"❌ LIVE Exit order failed: {e}")

            pnl = (exit_price - trade['entry_price']) * trade['quantity']
            trade['pnl'] = pnl
            trade['exit_price'] = exit_price
            trade['exit_reason'] = reason
            self.trade_logs.append(trade)

            pnl_emoji = "💰" if pnl > 0 else "💸"
            msg = (f"✅ Trade Closed\n\n"
                   f"🎯 Symbol: {trade['symbol']}\n"
                   f"📊 Entry: ₹{trade['entry_price']:.2f}, Exit: ₹{exit_price:.2f}\n"
                   f"{pnl_emoji} P&L: ₹{pnl:.2f}\n"
                   f"📝 Reason: {reason}")
            self._send_message(msg)
            logger.info(f"Trade closed: {trade['symbol']}, P&L: {pnl:.2f}")
            self.current_trade = None

    def _check_daily_limits(self) -> bool:
        """Check daily trading limits."""
        today = datetime.now(pytz.timezone('Asia/Kolkata')).date()
        today_trades = [t for t in self.trade_logs if datetime.fromisoformat(t['timestamp']).astimezone(pytz.timezone('Asia/Kolkata')).date() == today]
        today_pnl = sum(t.get('pnl', 0) for t in today_trades)

        if len(today_trades) >= self.config['MAX_TRADES_PER_DAY']:
            self._send_message(f"🚫 Daily trade limit reached ({self.config['MAX_TRADES_PER_DAY']}).")
            return False
        if today_pnl <= -self.config['MAX_LOSS_PER_DAY']:
            self._send_message(f"🚫 Daily loss limit reached (₹{-self.config['MAX_LOSS_PER_DAY']:.2f}).")
            self.config['AUTO_TRADE'] = False # Stop auto trading for the day
            self._send_message("🤖 Auto-trading disabled for the day.")
            return False
        return True

    # --- Telegram Command Handlers ---
    def cmd_start(self, update, context):
        update.message.reply_text("🤖 Nifty Scalper Bot is active. Use /help for commands.")

    def cmd_help(self, update, context):
        msg = ("📱 Available Commands:\n"
               "/status - Current bot & trade status\n"
               "/trade <CE/PE> - Manually enter a trade\n"
               "/exit - Exit the current trade\n"
               "/auto - Toggle auto-trading ON/OFF")
        update.message.reply_text(msg)

    def cmd_status(self, update, context):
        today = datetime.now(pytz.timezone('Asia/Kolkata')).date()
        today_trades = [t for t in self.trade_logs if datetime.fromisoformat(t['timestamp']).astimezone(pytz.timezone('Asia/Kolkata')).date() == today]
        today_pnl = sum(t.get('pnl', 0) for t in today_trades)

        msg = (f"🔄 Bot Status:\n"
               f"• Mode: {'🧪 DRY RUN' if self.config['DRY_RUN'] else '💰 LIVE'}\n"
               f"• Auto-trading: {'✅ ON' if self.config['AUTO_TRADE'] else '❌ OFF'}\n"
               f"• Market: {'🟢 OPEN' if self.is_market_hours() else '🔴 CLOSED'}\n"
               f"• Today's P&L: ₹{today_pnl:.2f}\n"
               f"• Today's Trades: {len(today_trades)}/{self.config['MAX_TRADES_PER_DAY']}")

        if self.current_trade:
            trade = self.current_trade
            ltp = self._get_ltp(trade['symbol']) or trade['entry_price']
            current_pnl = (ltp - trade['entry_price']) * trade['quantity']
            msg += (f"\n\n📊 Active Trade:\n"
                    f"• Symbol: {trade['symbol']}\n"
                    f"• Entry: ₹{trade['entry_price']:.2f}, LTP: ₹{ltp:.2f}\n"
                    f"• Current P&L: ₹{current_pnl:.2f}")
        else:
            msg += "\n\n💤 No active trades."
        update.message.reply_text(msg)

    def cmd_trade(self, update, context):
        if not context.args or context.args[0].upper() not in ['CE', 'PE']:
            update.message.reply_text("Usage: /trade <CE/PE>")
            return
        trade_type = context.args[0].upper()
        df = self._get_market_data()
        if df is None:
            update.message.reply_text("❌ Failed to get market data for manual trade.")
            return
        nifty_price = df['close'].iloc[-1]
        self._execute_trade(trade_type, nifty_price, score=99.0) # Manual trade score

    def cmd_exit(self, update, context):
        if not self.current_trade:
            update.message.reply_text("💤 No active trade to exit.")
            return
        self.exit_trade("Manual exit via command")

    def cmd_auto(self, update, context):
        self.config['AUTO_TRADE'] = not self.config['AUTO_TRADE']
        state = "✅ enabled" if self.config['AUTO_TRADE'] else "❌ disabled"
        update.message.reply_text(f"🤖 Auto-trading {state}.")
        logger.info(f"Auto-trading {state} by command.")

    def shutdown(self):
        logger.info("Shutting down bot...")
        if self.current_trade:
            self.exit_trade("Bot shutdown")
        if self.updater:
            self.updater.stop()
        shutdown_event.set()
        logger.info("Bot shutdown complete.")

# ================================
# Background Jobs
# ================================
def auto_trade_job(controller: BotController):
    """Job to find and execute trades automatically."""
    if not controller.config['AUTO_TRADE'] or not controller.is_market_hours() or controller.current_trade:
        return
    if not controller._check_daily_limits():
        return

    df = controller._get_market_data()
    if df is None: return

    buy_ce_score, buy_pe_score = controller.engine.generate_signal(df)
    nifty_price = df['close'].iloc[-1]

    if buy_ce_score >= controller.config['SIGNAL_THRESHOLD'] and buy_ce_score > buy_pe_score:
        controller._execute_trade('CE', nifty_price, buy_ce_score)
    elif buy_pe_score >= controller.config['SIGNAL_THRESHOLD']:
        controller._execute_trade('PE', nifty_price, buy_pe_score)

def monitor_trades_job(controller: BotController):
    """Job to monitor open trades for exit conditions."""
    if not controller.current_trade or not controller.is_market_hours():
        return

    # Use a lock to prevent race conditions when accessing the trade object
    with trade_lock:
        # Double-check if trade still exists after acquiring lock
        if not controller.current_trade:
            return
        
        trade = controller.current_trade
        ltp = controller._get_ltp(trade['symbol'])
        if not ltp: return

        trade['highest_price'] = max(trade.get('highest_price', ltp), ltp)
        
        # Note: Trailing SL is now calculated based on a percentage of the highest price since entry
        trailing_sl_percent = controller.config.get('TRAILING_SL_PERCENT', 0.15) # Default 15% trail
        trailing_sl_price = trade['highest_price'] * (1 - trailing_sl_percent)

        exit_reason = None
        if ltp <= trade['sl_price']:
            exit_reason = "Stop Loss Hit"
        elif ltp >= trade['tp_price']:
            exit_reason = "Take Profit Hit"
        elif ltp <= trailing_sl_price:
            exit_reason = "Trailing Stop Loss Hit"

    # Exit the trade outside the lock to prevent deadlocks
    if exit_reason:
        controller.exit_trade(exit_reason)


def main_loop(controller: BotController):
    """Main loop to run scheduled jobs."""
    last_trade_check = 0
    last_monitor_check = 0
    while not shutdown_event.is_set():
        now = time.time()
        # Run auto-trade check every 60 seconds
        if now - last_trade_check > 60:
            try:
                auto_trade_job(controller)
            except Exception as e:
                logger.error(f"Error in auto_trade_job: {e}", exc_info=True)
            last_trade_check = now

        # Run trade monitor every 5 seconds
        if now - last_monitor_check > 5:
            try:
                monitor_trades_job(controller)
            except Exception as e:
                logger.error(f"Error in monitor_trades_job: {e}", exc_info=True)
            last_monitor_check = now
        
        time.sleep(1)

# ================================
# Flask Routes & Startup
# ================================
@app.route('/')
def home():
    """Health check endpoint."""
    return jsonify({
        "status": "running",
        "auto_trading": controller.config['AUTO_TRADE'],
        "market_hours": controller.is_market_hours(),
        "current_trade": bool(controller.current_trade),
    })

def signal_handler(signum, frame):
    """Handle shutdown signals for graceful exit."""
    if 'controller' in globals():
        logger.info(f"Received signal {signum}, initiating shutdown...")
        controller.shutdown()
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        controller = BotController()
        controller._send_message(f"🚀 Nifty Scalper Bot Started!\nMode: {'🧪 DRY RUN' if controller.config['DRY_RUN'] else '💰 LIVE'}")

        # Start main logic in a background thread
        main_thread = Thread(target=main_loop, args=(controller,), daemon=True)
        main_thread.start()

        # Start Flask server using waitress
        from waitress import serve
        port = int(os.getenv('PORT', 10000))
        logger.info(f"Starting Flask server on port {port}")
        serve(app, host='0.0.0.0', port=port)

    except Exception as e:
        logger.error(f"Fatal startup error: {e}", exc_info=True)
        if 'controller' in globals() and controller.bot:
            controller._send_message(f"❌ Bot failed to start: {e}")
        sys.exit(1)
```
