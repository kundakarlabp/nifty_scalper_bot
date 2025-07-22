# NiftyScalperBot – Full System Rewrite with SignalEngine, CLI, Flask Webhooks, and Redis Task Queues

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from threading import Thread
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, request, jsonify
from rq import Queue
from redis import Redis
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from kiteconnect import KiteConnect
import telegram
from telegram.ext import Updater, CommandHandler

app = Flask(__name__)
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_conn = Redis.from_url(redis_url)
task_queue = Queue(connection=redis_conn)

# ================================
# SignalEngine – all AI logic here
# ================================
class SignalEngine:
    def __init__(self, fii_dii_score=0, option_sentiment_score=0):
        self.fii_dii_score = fii_dii_score
        self.option_sentiment_score = option_sentiment_score
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5)
        self.scaler = StandardScaler()

    def train_from_csv(self, path):
        df = pd.read_csv(path)
        df = self.compute_indicators(df)
        df.dropna(inplace=True)
        df['label'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        X = df[['ema_9', 'ema_21', 'rsi']]
        y = df['label']
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
        self.model.fit(X_train, y_train)

    def compute_indicators(self, df):
        import talib
        df['ema_9'] = talib.EMA(df['Close'], timeperiod=9)
        df['ema_21'] = talib.EMA(df['Close'], timeperiod=21)
        df['rsi'] = talib.RSI(df['Close'], timeperiod=14)
        df['macd'], df['macdsignal'], _ = talib.MACD(df['Close'])
        df['adx'] = talib.ADX(df['High'], df['Low'], df['Close'])
        df['mfi'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'])
        return df

    def generate_signal(self, df):
        df = self.compute_indicators(df)
        last = df.iloc[-1]
        score = sum([
            last['ema_9'] > last['ema_21'],
            last['rsi'] < 30,
            last['macd'] > last['macdsignal'],
            last['adx'] > 25,
            last['mfi'] < 20
        ])
        lstm_score = self.lstm_forecast(df)
        combined_score = score * 0.4 + self.fii_dii_score * 0.2 + self.option_sentiment_score * 0.2 + lstm_score * 0.2
        return combined_score

    def lstm_forecast(self, df):
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)
        series = df['log_ret'].values[-60:]
        X = [series[i:i+10] for i in range(len(series)-10)]
        y = series[10:]
        X = np.array(X).reshape(-1, 10, 1)
        model = Sequential()
        model.add(LSTM(50, input_shape=(10, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, verbose=0)
        return model.predict(X[-1].reshape(1, 10, 1))[0][0]

# ==========================================
# BotController (CLI + Telegram + Webhooks)
# ==========================================
class BotController:
    def __init__(self):
        self.config = self.load_config()
        self.kite = KiteConnect(api_key=self.config['ZERODHA_API_KEY'])
        self.kite.set_access_token(self.config['ZERODHA_ACCESS_TOKEN'])
        self.bot = telegram.Bot(token=self.config['TELEGRAM_BOT_TOKEN'])
        self.engine = SignalEngine()
        self.trade_logs = []
        self.logger = self.setup_logging()
        self.updater = Updater(token=self.config['TELEGRAM_BOT_TOKEN'], use_context=True)
        self.register_commands()

    def load_config(self):
        return {
            'ZERODHA_API_KEY': os.getenv('ZERODHA_API_KEY'),
            'ZERODHA_ACCESS_TOKEN': os.getenv('ZERODHA_ACCESS_TOKEN'),
            'TELEGRAM_BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN'),
            'TELEGRAM_CHAT_ID': os.getenv('TELEGRAM_CHAT_ID')
        }

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s [%(levelname)s] %(message)s',
                            handlers=[logging.StreamHandler(),
                                      logging.FileHandler("scalper.log")])
        return logging.getLogger("Bot")

    def register_commands(self):
        dp = self.updater.dispatcher
        dp.add_handler(CommandHandler('start', self.cmd_start))
        dp.add_handler(CommandHandler('summary', self.cmd_summary))
        dp.add_handler(CommandHandler('export', self.cmd_export))
        self.updater.start_polling()

    def cmd_start(self, update, context):
        context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text="🤖 NiftyScalper Started")

    def cmd_summary(self, update, context):
        self.visualize_dashboard()
        context.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'], text="📊 Trade Summary Sent!")

    def cmd_export(self, update, context):
        df = pd.DataFrame(self.trade_logs)
        path = f"export_trades_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        df.to_excel(path, index=False)
        context.bot.send_document(chat_id=self.config['TELEGRAM_CHAT_ID'], document=open(path, 'rb'))

    def visualize_dashboard(self):
        df = pd.DataFrame(self.trade_logs)
        if df.empty:
            return
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='timestamp', y='pnl', marker='o')
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = f"pnl_plot_{datetime.now().strftime('%H%M')}.png"
        plt.savefig(path)
        self.bot.send_photo(chat_id=self.config['TELEGRAM_CHAT_ID'], photo=open(path, 'rb'))

    def auto_trade(self):
        df = self.kite.historical_data(260105, datetime.now() - timedelta(days=5), datetime.now(), '5minute')
        df = pd.DataFrame(df)
        score = self.engine.generate_signal(df)
        if score > 0.75:
            self.place_order(df['close'].iloc[-1])
            self.bot.send_message(chat_id=self.config['TELEGRAM_CHAT_ID'],
                                  text=f"✅ Auto Trade Triggered | Score: {score:.2f}")

    def place_order(self, entry_price):
        symbol = "NIFTY22500CE"  # placeholder
        pnl = round(np.random.uniform(100, 300), 2)  # Simulated
        self.trade_logs.append({
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': entry_price + pnl / 75,
            'pnl': pnl,
            'timestamp': datetime.now().isoformat()
        })

controller = BotController()

@app.route('/trade', methods=['POST'])
def trigger_trade():
    job = task_queue.enqueue(controller.auto_trade)
    return jsonify({'status': 'Trade queued', 'job_id': job.id})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'trades': len(controller.trade_logs)})

if __name__ == '__main__':
    from waitress import serve
    port = int(os.getenv('PORT', 10000))
    Thread(target=lambda: serve(app, host='0.0.0.0', port=port), daemon=True).start()
    while True:
        controller.auto_trade()
        time.sleep(300)
@app.route('/')
def home():
    return """
    <h1>Nifty Scalper Bot is Live 🎉</h1>
    <p>Available endpoints:</p>
    <ul>
        <li><a href="/status">/status</a></li>
        <li><a href="/trade (use POST)">/trade</a></li>
    </ul>
    <p>Use <code>curl -X POST https://nifty-scalper-bot.onrender.com/trade</code> to trigger a trade.</p>
    """
