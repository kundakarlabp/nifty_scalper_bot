# nifty_scalper_bot.py - Fully Automatic Trading Bot
import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from threading import Thread
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, jsonify, request
from rq import Queue
from redis import Redis
from kiteconnect import KiteConnect
import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# ================================
# Configuration & Initialization
# ================================
app = Flask(__name__)
redis_conn = Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'), decode_responses=True)
task_queue = Queue(connection=redis_conn)

# ================================
# SignalEngine - Enhanced AI & Indicators
# ================================
class SignalEngine:
    def __init__(self):
        self.logger = logging.getLogger("SignalEngine")
        
    def compute_indicators(self, df):
        try:
            import pandas_ta as ta
            df['ema_9'] = ta.ema(df['close'], length=9)
            df['ema_21'] = ta.ema(df['close'], length=21)
            df['rsi'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'])
            df['macd'] = macd['MACD_12_26_9']
            df['macdsignal'] = macd['SIGNAL_12_26_9']
            df['adx'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
            df['bb_upper'] = ta.bbands(df['close'], length=20)['BBU_20_2.0']
            df['bb_lower'] = ta.bbands(df['close'], length=20)['BBL_20_2.0']
            df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            return df
        except Exception as e:
            self.logger.error(f"Indicator error: {e}")
            return df

    def generate_signal(self, df):
        df = self.compute_indicators(df)
        if df.empty:
            return 0, 0
            
        last = df.iloc[-1]
        close = last['close']

        # Enhanced scoring system with weights
        buy_ce_score = sum([
            (last['ema_9'] > last['ema_21']) * 1.2,
            (last['rsi'] < 35) * 1.0,
            (last['macd'] > last['macdsignal']) * 0.9,
            (last['adx'] > 25) * 0.7,
            (last['mfi'] < 20) * 0.6,
            (close > last['bb_lower']) * 0.4,
            (close > last['vwap']) * 0.3
        ])

        buy_pe_score = sum([
            (last['ema_9'] < last['ema_21']) * 1.2,
            (last['rsi'] > 65) * 1.0,
            (last['macd'] < last['macdsignal']) * 0.9,
            (last['adx'] > 25) * 0.7,
            (last['mfi'] > 80) * 0.6,
            (close < last['bb_upper']) * 0.4,
            (close < last['vwap']) * 0.3
        ])

        return buy_ce_score, buy_pe_score

# ================================
# BotController - Complete Trading Logic
# ================================
class BotController:
    def __init__(self):
        self.config = {
            'ZERODHA_API_KEY': os.getenv('ZERODHA_API_KEY'),
            'ZERODHA_ACCESS_TOKEN': os.getenv('ZERODHA_ACCESS_TOKEN'),
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
            'TELEGRAM_CHAT_ID': int(os.getenv('TELEGRAM_CHAT_ID')),
            'DRY_RUN': os.getenv('DRY_RUN', 'false').lower() == 'true',
            'AUTO_TRADE': os.getenv('AUTO_TRADE', 'true').lower() == 'true'
        }
        self.kite = KiteConnect(api_key=self.config['ZERODHA_API_KEY'])
        self.kite.set_access_token(self.config['ZERODHA_ACCESS_TOKEN'])
        self.bot = telegram.Bot(token=self.config['TELEGRAM_BOT_TOKEN'])
        self.engine = SignalEngine()
        self.trade_logs = []
        self.current_trade = None
        self.logger = self.setup_logging()
        self.setup_telegram()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler('bot.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("Bot")

    def setup_telegram(self):
        self.updater = Updater(token=self.config['TELEGRAM_BOT_TOKEN'], use_context=True)
        dp = self.updater.dispatcher
        
        # Command handlers
        dp.add_handler(CommandHandler('start', self.cmd_start))
        dp.add_handler(CommandHandler('status', self.cmd_status))
        dp.add_handler(CommandHandler('summary', self.cmd_summary))
        dp.add_handler(CommandHandler('export', self.cmd_export))
        dp.add_handler(CommandHandler('trade', self.cmd_trade))
        dp.add_handler(CommandHandler('exit', self.cmd_exit))
        dp.add_handler(CommandHandler('sl', self.cmd_sl))
        dp.add_handler(CommandHandler('tp', self.cmd_tp))
        dp.add_handler(CommandHandler('auto', self.cmd_auto))
        
        # Start polling
        self.updater.start_polling()
        self.logger.info("Telegram bot started")

    def register_webhook(self):
        """Register webhook for Telegram updates"""
        webhook_url = os.getenv('WEBHOOK_URL')
        if webhook_url:
            self.bot.set_webhook(url=f"{webhook_url}/telegram")
            self.logger.info(f"Webhook registered at {webhook_url}/telegram")

    # Telegram command implementations
    def cmd_start(self, update, context):
        msg = "🤖 Nifty Scalper Bot is Live!\n\n" \
              "Commands:\n" \
              "/trade - Manual trade\n" \
              "/exit - Exit current trade\n" \
              "/sl [price] - Set stop-loss\n" \
              "/tp [price] - Set take-profit\n" \
              "/status - Current status\n" \
              "/summary - P&L chart\n" \
              "/export - Export trades\n" \
              "/auto - Toggle auto-trading"
        context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

    def cmd_status(self, update, context):
        status_msg = f"🔄 Bot Status:\n"
        status_msg += f"• Auto-trading: {'✅ ON' if self.config['AUTO_TRADE'] else '❌ OFF'}\n"
        status_msg += f"• Market hours: {'✅ OPEN' if self.is_market_hours() else '❌ CLOSED'}\n"
        
        if self.current_trade:
            status_msg += f"\n📊 Current Trade:\n"
            status_msg += f"• Symbol: {self.current_trade['symbol']}\n"
            status_msg += f"• Type: {self.current_trade['type']}\n"
            status_msg += f"• Entry: ₹{self.current_trade['entry']:.2f}\n"
            status_msg += f"• SL: ₹{self.current_trade.get('sl', 'N/A')}\n"
            status_msg += f"• TP: ₹{self.current_trade.get('tp', 'N/A')}\n"
            status_msg += f"• P&L: ₹{self.current_trade.get('pnl', 0):.2f}"
        else:
            status_msg += "\n💤 No active trades"
            
        context.bot.send_message(chat_id=update.effective_chat.id, text=status_msg)

    def cmd_auto(self, update, context):
        self.config['AUTO_TRADE'] = not self.config['AUTO_TRADE']
        state = "✅ enabled" if self.config['AUTO_TRADE'] else "❌ disabled"
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Auto-trading {state}"
        )

    # ... (other command methods remain the same as before)

    def is_market_hours(self):
        now = datetime.now().time()
        market_open = datetime.strptime("09:15", "%H:%M").time()
        market_close = datetime.strptime("15:30", "%H:%M").time()
        return market_open <= now <= market_close

    def exit_trade(self, reason="Manual exit"):
        if not self.current_trade:
            return
            
        # Calculate final P&L
        try:
            symbol = self.current_trade['symbol']
            ticker = self.kite.quote(f"NFO:{symbol}")
            last_price = ticker[f"NFO:{symbol}"]["last_price"]
            
            if self.current_trade['type'] == 'CE':
                self.current_trade['pnl'] = (last_price - self.current_trade['entry']) * 75
            else:
                self.current_trade['pnl'] = (self.current_trade['entry'] - last_price) * 75
                
            self.current_trade['exit_price'] = last_price
            self.current_trade['exit_time'] = datetime.now().isoformat()
            self.current_trade['exit_reason'] = reason
        except Exception as e:
            self.logger.error(f"Exit calculation error: {e}")

        # Log and notify
        self.trade_logs.append(self.current_trade.copy())
        self.bot.send_message(
            chat_id=self.config['TELEGRAM_CHAT_ID'],
            text=f"✅ Trade exited ({reason})\n" \
                 f"Symbol: {self.current_trade['symbol']}\n" \
                 f"P&L: ₹{self.current_trade['pnl']:.2f}"
        )
        self.current_trade = None

# ================================
# Trading Functions
# ================================
def auto_trade_job():
    """Automatically find and execute trades"""
    if not controller.is_market_hours():
        return
        
    if controller.current_trade:
        controller.logger.info("Trade already active, skipping auto-trade")
        return
        
    try:
        # Get Nifty 50 data
        df = controller.kite.historical_data(256265, datetime.now() - timedelta(days=3), datetime.now(), '15minute')
        df = pd.DataFrame(df)
        
        # Generate signals
        buy_ce_score, buy_pe_score = controller.engine.generate_signal(df)
        entry = df['close'].iloc[-1]
        
        # Check if we should trade
        if buy_ce_score >= 2.5:  # Higher threshold for better quality
            execute_trade('CE', entry, buy_ce_score)
        elif buy_pe_score >= 2.5:
            execute_trade('PE', entry, buy_pe_score)
            
    except Exception as e:
        controller.logger.error(f"Auto-trade error: {e}")

def execute_trade(trade_type, entry_price, score):
    """Execute a trade with proper risk management"""
    try:
        if trade_type == 'CE':
            symbol = f"NIFTY{get_nearest_strike(entry_price)}CE"
            sl = entry_price * 0.98  # 2% SL
            tp = entry_price * 1.04  # 4% TP
        else:
            symbol = f"NIFTY{get_nearest_strike(entry_price)}PE"
            sl = entry_price * 1.02  # 2% SL
            tp = entry_price * 0.96  # 4% TP
            
        # Create trade record
        controller.current_trade = {
            'symbol': symbol,
            'entry': entry_price,
            'sl': sl,
            'tp': tp,
            'type': trade_type,
            'score': score,
            'timestamp': datetime.now().isoformat(),
            'pnl': 0,
            'highest': entry_price if trade_type == 'CE' else None,
            'lowest': entry_price if trade_type == 'PE' else None
        }
        
        # Place actual order if not in dry run
        if not controller.config['DRY_RUN']:
            try:
                # Get instrument token
                instruments = controller.kite.instruments("NFO")
                instrument = next((i for i in instruments if i['tradingsymbol'] == symbol), None)
                
                if instrument:
                    order_id = controller.kite.place_order(
                        tradingsymbol=symbol,
                        exchange=controller.kite.EXCHANGE_NFO,
                        transaction_type=controller.kite.TRANSACTION_TYPE_BUY,
                        quantity=75,
                        order_type=controller.kite.ORDER_TYPE_MARKET,
                        product=controller.kite.PRODUCT_MIS,
                        variety=controller.kite.VARIETY_REGULAR
                    )
                    controller.current_trade['order_id'] = order_id
            except Exception as e:
                controller.logger.error(f"Order placement error: {e}")
                controller.current_trade = None
                return
        
        # Notify
        controller.bot.send_message(
            chat_id=controller.config['TELEGRAM_CHAT_ID'],
            text=f"📈 {'BUY CE' if trade_type == 'CE' else 'BUY PE'} - {symbol}\n" \
                 f"Entry: ₹{entry_price:.2f}\n" \
                 f"Score: {score:.2f}\n" \
                 f"TP: ₹{tp:.2f}\n" \
                 f"SL: ₹{sl:.2f}"
        )
        
    except Exception as e:
        controller.logger.error(f"Trade execution error: {e}")

def get_nearest_strike(price):
    """Get nearest option strike price"""
    strike = round(price / 50) * 50
    return int(strike)

def monitor_trades():
    """Continuously monitor open trades for exits"""
    while True:
        time.sleep(30)  # Check every 30 seconds
        
        if not controller.current_trade or not controller.is_market_hours():
            continue
            
        try:
            symbol = controller.current_trade['symbol']
            ticker = controller.kite.quote(f"NFO:{symbol}")
            last_price = ticker[f"NFO:{symbol}"]["last_price"]
            
            # Update P&L
            if controller.current_trade['type'] == 'CE':
                controller.current_trade['pnl'] = (last_price - controller.current_trade['entry']) * 75
                controller.current_trade['highest'] = max(controller.current_trade['highest'], last_price)
            else:
                controller.current_trade['pnl'] = (controller.current_trade['entry'] - last_price) * 75
                controller.current_trade['lowest'] = min(controller.current_trade['lowest'], last_price)
            
            # Check exit conditions
            exit_reason = None
            if controller.current_trade['type'] == 'CE':
                if last_price <= controller.current_trade['sl']:
                    exit_reason = "SL hit"
                elif last_price >= controller.current_trade['tp']:
                    exit_reason = "TP hit"
                elif last_price <= controller.current_trade['highest'] * 0.97:  # Trailing SL
                    exit_reason = "Trailing SL"
            else:
                if last_price >= controller.current_trade['sl']:
                    exit_reason = "SL hit"
                elif last_price <= controller.current_trade['tp']:
                    exit_reason = "TP hit"
                elif last_price >= controller.current_trade['lowest'] * 1.03:  # Trailing SL
                    exit_reason = "Trailing SL"
                    
            # Exit if needed
            if exit_reason:
                controller.exit_trade(exit_reason)
                
        except Exception as e:
            controller.logger.error(f"Trade monitoring error: {e}")

# ================================
# Flask Routes
# ================================
@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "auto_trading": controller.config['AUTO_TRADE'],
        "market_hours": controller.is_market_hours(),
        "current_trade": bool(controller.current_trade)
    })

@app.route('/telegram', methods=['POST'])
def telegram_webhook():
    """Handle Telegram webhook updates"""
    update = telegram.Update.de_json(request.get_json(force=True), controller.bot)
    controller.updater.dispatcher.process_update(update)
    return jsonify({"status": "ok"})

# ================================
# Initialization & Startup
# ================================
if __name__ == '__main__':
    # Initialize controller
    controller = BotController()
    controller.register_webhook()
    
    # Start background threads
    Thread(target=monitor_trades, daemon=True).start()
    
    if controller.config['AUTO_TRADE']:
        def auto_trade_scheduler():
            while True:
                if controller.is_market_hours() and not controller.current_trade:
                    task_queue.enqueue(auto_trade_job)
                time.sleep(300)  # Check every 5 minutes
                
        Thread(target=auto_trade_scheduler, daemon=True).start()
    
    # Start Flask server
    from waitress import serve
    port = int(os.getenv('PORT', 8000))
    serve(app, host='0.0.0.0', port=port)
