# config.py
import os

class Config:
    # Kite Connect API
    KITE_API_KEY = os.getenv('KITE_API_KEY')
    KITE_ACCESS_TOKEN = os.getenv('KITE_ACCESS_TOKEN')

    # Trading symbol
    UNDERLYING_SYMBOL = os.getenv('UNDERLYING_SYMBOL', 'NIFTY50')  # e.g. 'NIFTY50'

    # Signal & risk settings
    SIGNAL_THRESHOLD = float(os.getenv('SIGNAL_THRESHOLD', 7.0))
    RISK_PER_TRADE_PCT = float(os.getenv('RISK_PER_TRADE_PCT', 0.01))  # 1%
    MAX_DAILY_TRADES = int(os.getenv('MAX_DAILY_TRADES', 10))
    MAX_DAILY_LOSS_PCT = float(os.getenv('MAX_DAILY_LOSS_PCT', 0.05))  # 5%
    LOOP_DELAY = int(os.getenv('LOOP_DELAY', 5))  # seconds between loops

    # Position sizing
    DEFAULT_LOT_SIZE = int(os.getenv('DEFAULT_LOT_SIZE', 75))
    MIN_QUANTITY = int(os.getenv('MIN_QUANTITY', 75))
    MAX_QUANTITY = int(os.getenv('MAX_QUANTITY', 750))
    DEFAULT_QUANTITY = DEFAULT_LOT_SIZE

    # Signal timing
    MIN_SIGNAL_INTERVAL = int(os.getenv('MIN_SIGNAL_INTERVAL', 60))  # seconds

    # Circuit breaker
    CIRCUIT_BREAKER_MAX_LOSSES = int(os.getenv('CIRCUIT_BREAKER_MAX_LOSSES', 3))
    CIRCUIT_BREAKER_PAUSE_MINUTES = int(os.getenv('CIRCUIT_BREAKER_PAUSE_MINUTES', 60))

    # Market hours
    MARKET_START_HOUR = 9
    MARKET_START_MINUTE = 15
    MARKET_END_HOUR = 15
    MARKET_END_MINUTE = 30

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')


# monitor.py
#!/usr/bin/env python3
"""
Bot Monitoring Script
Monitors the trading bot and sends alerts if issues are detected
"""

import requests
import time
import logging
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import json
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('monitor.log')]
)
logger = logging.getLogger(__name__)

class BotMonitor:
    """Monitor the trading bot and send alerts"""
    def __init__(self, bot_url: str = "http://localhost:10000"):
        self.bot_url = bot_url
        self.last_status = None
        self.last_alert_time = None

    def check_health(self) -> Dict[str, Any]:
        try:
            r = requests.get(f"{self.bot_url}/health", timeout=10)
            if r.status_code == 200:
                return {'status': 'healthy', 'data': r.json(), 'timestamp': datetime.now()}
            return {'status': 'unhealthy', 'error': f"HTTP {r.status_code}", 'timestamp': datetime.now()}
        except requests.RequestException as e:
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now()}

    def check_trading_status(self) -> Optional[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.bot_url}/status", timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.error(f"Error fetching trading status: {e}")
        return None

    def check_for_issues(self, status: Dict[str, Any]) -> list:
        issues = []
        if status['status'] != 'healthy':
            issues.append(f"Bot unhealthy: {status.get('error')}")
            return issues
        ts = self.check_trading_status()
        if not ts:
            issues.append("Unable to fetch trading status")
            return issues
        if ts.get('circuit_breaker'):
            issues.append("Circuit breaker is active")
        if ts.get('daily_pnl', 0) < -5000:
            issues.append(f"High daily loss: ₹{ts['daily_pnl']:.2f}")
        if not ts.get('auto_trade', True):
            issues.append("Auto-trading disabled")
        return issues

    def send_alert(self, msg: str):
        now = datetime.now()
        if self.last_alert_time and now - self.last_alert_time < timedelta(minutes=30):
            return
        logger.warning(f"ALERT: {msg}")
        self.last_alert_time = now

    def monitor_loop(self, interval: int = 60):
        logger.info(f"Starting monitor every {interval}s")
        while True:
            st = self.check_health()
            issues = self.check_for_issues(st)
            if issues:
                self.send_alert("\n".join(issues))
            else:
                logger.info("Bot healthy")
            time.sleep(interval)

# utils.py
#!/usr/bin/env python3
import logging
import os
import pytz
from datetime import datetime, time, timedelta
from typing import Any, Dict
import pandas as pd
from config import Config

# Logging
def setup_logging():
    lvl = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(level=getattr(logging, lvl), format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)
logger = setup_logging()

# Market utils
def is_market_open() -> bool:
    try:
        tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(tz)
        if now.weekday() >= 5:
            return False
        start = time(Config.MARKET_START_HOUR, Config.MARKET_START_MINUTE)
        end = time(Config.MARKET_END_HOUR, Config.MARKET_END_MINUTE)
        return start <= now.time() <= end
    except:
        return False

def get_market_status() -> str:
    return "🟢 OPEN" if is_market_open() else "🔴 CLOSED"

def time_until_market_open() -> str:
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz)
    if now.weekday() >= 5:
        days = 7 - now.weekday()
        return f"Opens Monday (in {days}d)"
    open_time = now.replace(hour=Config.MARKET_START_HOUR, minute=Config.MARKET_START_MINUTE, second=0)
    if now < open_time:
        delta = open_time - now
        h, r = divmod(delta.seconds, 3600)
        m, _ = divmod(r, 60)
        return f"Opens in {h}h {m}m"
    return "Opens tomorrow at 9:15 AM"

# Safe converters
def safe_float(v: Any, d: float = 0.0) -> float:
    try: return float(v)
    except: return d

def safe_int(v: Any, d: int = 0) -> int:
    try: return int(v)
    except: return d

# Technical indicators (as before, omitted for brevity)
# ...

# telegram_bot.py
#!/usr/bin/env python3
import logging
import asyncio
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from config import Config
from utils import is_market_open, get_market_status, time_until_market_open

logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self, trading_bot=None):
        self.trading_bot = trading_bot
        self.app = None
        self.is_running = False

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        status = get_market_status()
        mode = '✅ ON' if self.trading_bot and self.trading_bot.auto_trade else '❌ OFF'
        msg = f"🚀 Bot Started!\nAuto-trading: {mode}\nMarket: {status}"
        await update.message.reply_text(msg)

    # Additional handlers (/stop, /status, etc.) as refactored above
    # ...

    def setup_handlers(self):
        for cmd, fn in [
            ('start', self.start_command),
            # etc.
        ]:
            self.app.add_handler(CommandHandler(cmd, fn))

    async def start_bot(self):
        if not Config.TELEGRAM_BOT_TOKEN:
            return
        self.app = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()
        self.setup_handlers()
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        self.is_running = True

# kite_client.py
#!/usr/bin/env python3
from kiteconnect import KiteConnect
from config import Config

class KiteClient:
    def __init__(self):
        self.kite = KiteConnect(api_key=Config.KITE_API_KEY)

    def connect(self) -> bool:
        try:
            self.kite.set_access_token(Config.KITE_ACCESS_TOKEN)
            return True
        except:
            return False

    def get_instrument_token(self, symbol: str) -> int:
        instruments = self.kite.instruments('NFO')
        for inst in instruments:
            if inst['tradingsymbol'] == symbol:
                return inst['instrument_token']
        return None

# signal_generator.py
#!/usr/bin/env python3
from typing import Dict, Any

class SignalGenerator:
    def __init__(self):
        pass

    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: implement logic
        return {}

# nifty_scalper_bot.py
#!/usr/bin/env python3
import os
import sys
import logging
import threading
import time
import asyncio
import schedule
from datetime import datetime
import pandas as pd
from flask import Flask, jsonify
from config import Config
from kite_client import KiteClient
from signal_generator import SignalGenerator
from monitor import BotMonitor
from utils import is_market_open, get_market_status, time_until_market_open
from telegram_bot import TelegramBot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NiftyScalperBot:
    def __init__(self):
        self.kite_client = KiteClient()
        self.signal_generator = SignalGenerator()
        self.monitor = BotMonitor()
        self.telegram_bot = TelegramBot(self)
        # ... initialize risk manager, state
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/')
        def health():
            return jsonify(status='running')
        # /status and /trades as before

    def start(self):
        # Connect, start threads, etc.
        pass

    def stop(self):
        pass

    # trading_loop, execute_trade, place_order, close_position, etc.

if __name__ == '__main__':
    bot = NiftyScalperBot()
    bot.start()
    bot.app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
