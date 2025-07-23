# nifty_scalper_bot.py - Final Optimized Version
import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from threading import Thread
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, jsonify
from rq import Queue
from redis import Redis
from kiteconnect import KiteConnect
import telegram
from telegram.ext import Updater, CommandHandler

# ================================
# Flask & Redis Setup
# ================================
app = Flask(__name__)
redis_conn = Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'), decode_responses=True)
task_queue = Queue(connection=redis_conn)

# ================================
# SignalEngine – AI & Indicators
# ================================
class SignalEngine:
    def __init__(self):
        pass

    def compute_indicators(self, df):
        import pandas_ta as ta
        df['ema_9'] = ta.ema(df['Close'], length=9)
        df['ema_21'] = ta.ema(df['Close'], length=21)
        df['rsi'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macdsignal'] = macd['SIGNAL_12_26_9']
        df['adx'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
        df['mfi'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
        df['bb_upper'] = ta.bbands(df['Close'], length=20)['BBU_20_2.0']
        df['bb_lower'] = ta.bbands(df['Close'], length=20)['BBL_20_2.0']
        return df

    def generate_signal(self, df):
        df = self.compute_indicators(df)
        df.dropna(inplace=True)
        last = df.iloc[-1]
        score = sum([
            (last['ema_9'] > last['ema_21']) * 1.0,
            (last['rsi'] < 30) * 0.8,
            (last['macd'] > last['macdsignal']) * 0.7,
            (last['adx'] > 25) * 0.5,
            (last['mfi'] < 20) * 0.5
        ])
        return score

# ================================
# BotController – Trading Logic
# ================================
class BotController:
    def __init__(self):
        self.config = {
            'ZERODHA_API_KEY': os.getenv('ZERODHA_API_KEY'),
            'ZERODHA_ACCESS_TOKEN': os.getenv('ZERODHA_ACCESS_TOKEN'),
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
            'TELEGRAM_CHAT_ID': int(os.getenv('TELEGRAM_CHAT_ID')),
            'DRY_RUN': os.getenv('DRY_RUN', 'true').lower() == 'true'
        }
        self.kite = KiteConnect(api_key=self.config['ZERODHA_API_KEY'])
        self.kite.set_access_token(self.config['ZERODHA_ACCESS_TOKEN'])
        self.bot = telegram.Bot(token=self.config['TELEGRAM_BOT_TOKEN'])
        self.engine = SignalEngine()
        self.trade_logs = []
        self.current_trade = None
        self.logger = self.setup_logging()
        self.updater = Updater(token=self.config['TELEGRAM_BOT_TOKEN'], use_context=True)
        self.register_commands()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
        return logging.getLogger("Bot")

    def register_commands(self):
        dp = self.updater.dispatcher
        dp.add_handler(CommandHandler('start', self.cmd_start))
        dp.add_handler(CommandHandler('summary', self.cmd_summary))
        dp.add_handler(CommandHandler('export', self.cmd_export))
        dp.add_handler(CommandHandler('trade', self.cmd_trade))
        dp.add_handler(CommandHandler('exit', self.cmd_exit))
        dp.add_handler(CommandHandler('sl', self.cmd_sl))
        dp.add_handler(CommandHandler('tp', self.cmd_tp))
        self.updater.start_polling()

    def cmd_start(self, update, context):
        msg = "🤖 Nifty Scalper Bot is Live!\n\n" \
              "Commands:\n" \
              "/trade – Start a trade\n" \
              "/exit – Exit current trade\n" \
              "/sl 190 – Set stop-loss\n" \
              "/tp 210 – Set take-profit\n" \
              "/summary – Show P&L chart\n" \
              "/export – Export trade history"
        context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text=msg)

    def cmd_summary(self, update, context):
        if not self.trade_logs:
            context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text="📊 No trades yet.")
            return
        df = pd.DataFrame(self.trade_logs)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='timestamp', y='pnl')
        plt.title("P&L Over Time")
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = "pnl_plot.png"
        plt.savefig(path)
        plt.close()
        context.bot.send_photo(chat_id=self.config['TELEGRAM_CHAT_ID'], photo=open(path, 'rb'))

    def cmd_export(self, update, context):
        if not self.trade_logs:
            context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text="📥 No trades to export.")
            return
        df = pd.DataFrame(self.trade_logs)
        path = f"trades_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        df.to_excel(path, index=False)
        context.bot.send_document(chat_id=self.config['TELEGRAM_CHAT_ID'], document=open(path, 'rb'))

    def cmd_trade(self, update, context):
        if self.current_trade:
            context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text="⚠️ Already in a trade!")
            return
        task_queue.enqueue(self.auto_trade)
        context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text="🔍 Looking for trade...")

    def cmd_exit(self, update, context):
        if not self.current_trade:
            context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text="❌ No active trade.")
            return
        self.exit_trade()
        context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text="✅ Trade exited manually.")

    def cmd_sl(self, update, context):
        if not self.current_trade:
            context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text="❌ No active trade.")
            return
        try:
            new_sl = float(context.args[0])
            self.current_trade['sl'] = new_sl
            context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text=f"📉 SL updated to {new_sl}")
        except:
            context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text="❌ Usage: /sl 190")

    def cmd_tp(self, update, context):
        if not self.current_trade:
            context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text="❌ No active trade.")
            return
        try:
            new_tp = float(context.args[0])
            self.current_trade['tp'] = new_tp
            context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text=f"🎯 TP updated to {new_tp}")
        except:
            context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text="❌ Usage: /tp 210")

    def auto_trade(self):
        if not self.is_market_hours():
            return
        if self.current_trade:
            return
        try:
            df = self.kite.historical_data(260105, datetime.now() - timedelta(days=5), datetime.now(), '5minute')
            df = pd.DataFrame(df)
            score = self.engine.generate_signal(df)
            entry = df['close'].iloc[-1]
            sl = entry * 0.97
            tp = entry * 1.05

            if score > 0.7:
                self.current_trade = {
                    'symbol': 'NIFTY22500CE',
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'pnl': 0,
                    'highest': entry,
                    'timestamp': datetime.now().isoformat()
                }
                self.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text=f"📈 BUY CE – ₹{entry:.2f}\n🎯 TP: {tp:.2f}\n🛑 SL: {sl:.2f}")
        except Exception as e:
            self.logger.error(f"Auto-trade error: {e}")

    def exit_trade(self):
        if not self.current_trade:
            return
        self.trade_logs.append(self.current_trade.copy())
        self.current_trade = None
        self.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text="✅ Trade exited.")

    def monitor_trade_with_trailing_sl(self):
        while True:
            time.sleep(60)  # Check every 60 seconds
            if not self.current_trade or not self.is_market_hours():
                continue
            try:
                ticker = self.kite.ltp(self.current_trade['symbol'])
                last_price = ticker[self.current_trade['symbol']]['last_price']
                new_high = max(self.current_trade['highest'], last_price)
                trailing_sl = new_high * 0.97
                self.current_trade['highest'] = new_high
                self.current_trade['sl'] = trailing_sl

                if last_price <= trailing_sl:
                    self.exit_trade()
                    self.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text=f"🛑 Trailing SL Hit! Price: {last_price:.2f}")
                else:
                    self.current_trade['pnl'] = (last_price - self.current_trade['entry']) * 75
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")

    def is_market_hours(self):
        now = datetime.now().time()
        return datetime.strptime("09:15", "%H:%M").time() <= now <= datetime.strptime("15:30", "%H:%M").time()

# ================================
# Initialize Everything
# ================================
controller = BotController()
Thread(target=controller.monitor_trade_with_trailing_sl, daemon=True).start()

# ================================
# Auto-Scheduler (Every 1 Minute)
# ================================
def run_scheduler():
    while True:
        if controller.is_market_hours() and controller.current_trade is None:
            task_queue.enqueue(controller.auto_trade)
        time.sleep(60)  # Every 1 minute

Thread(target=run_scheduler, daemon=True).start()

# ================================
# Flask Routes
# ================================
@app.route('/')
def home():
    return "<h1>Nifty Scalper Bot is Live 🎉</h1><p>Use <code>/trade</code> in Telegram to start.</p>"

@app.route('/trade', methods=['POST'])
def trigger_trade():
    if controller.current_trade:
        return jsonify({'status': 'Already in trade'})
    job = task_queue.enqueue(controller.auto_trade)
    return jsonify({'status': 'Trade queued', 'job_id': job.id})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'trades': len(controller.trade_logs),
        'current_trade': controller.current_trade
    })

# ================================
# Start Server
# ================================
if __name__ == '__main__':
    from waitress import serve
    port = int(os.getenv('PORT', 10000))
    Thread(target=lambda: serve(app, host='0.0.0.0', port=port), daemon=True).start()
    controller.updater.idle()
