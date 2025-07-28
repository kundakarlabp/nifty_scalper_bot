import os
import pytz
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Set India timezone (IST - India Standard Time)
IST_TZ = pytz.timezone('Asia/Kolkata')

def get_current_ist_time():
    """Get current time in IST"""
    return datetime.now(IST_TZ)

def format_ist_time(dt=None):
    """Format datetime in IST"""
    if dt is None:
        dt = get_current_ist_time()
    return dt.strftime('%Y-%m-%d %H:%M:%S %Z')

# Zerodha API Configuration
ZERODHA_API_KEY = os.getenv('ZERODHA_API_KEY', '')
ZERODHA_API_SECRET = os.getenv('ZERODHA_API_SECRET', '')
ZERODHA_REQUEST_TOKEN = os.getenv('ZERODHA_REQUEST_TOKEN', '')
ZERODHA_ACCESS_TOKEN = os.getenv('ZERODHA_ACCESS_TOKEN', '')

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Trading Configuration
TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'NIFTY 50')
NIFTY_LOT_SIZE = int(os.getenv('NIFTY_LOT_SIZE', '75'))  # Updated to 75
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '1'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.01'))  # 1% of capital
ACCOUNT_SIZE = float(os.getenv('ACCOUNT_SIZE', '100000'))
MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '0.05'))  # 5%
MAX_RISK_PER_DAY = float(os.getenv('MAX_RISK_PER_DAY', '0.10'))  # 10%

# Dynamic Position Sizing Configuration
MIN_LOTS = int(os.getenv('MIN_LOTS', '1'))
MAX_LOTS = int(os.getenv('MAX_LOTS', '10'))
BASE_STOP_LOSS_POINTS = int(os.getenv('BASE_STOP_LOSS_POINTS', '20'))
BASE_TARGET_POINTS = int(os.getenv('BASE_TARGET_POINTS', '40'))

# Strategy Configuration
STRATEGY_NAME = os.getenv('STRATEGY_NAME', 'dynamic_scalping')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))

# Database Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trading_data.db')

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Timezone Configuration
TIMEZONE = os.getenv('TIMEZONE', 'Asia/Kolkata')
CURRENT_TIMEZONE = pytz.timezone(TIMEZONE)

print("✅ Configuration loaded successfully!")
print(f"🕐 Current IST Time: {format_ist_time()}")
