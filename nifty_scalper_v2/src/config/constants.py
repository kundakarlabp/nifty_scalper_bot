"""All magic numbers and compile-time constants for NiftyScalperBot v2.

Importing this module is zero-cost — it defines only primitive values.
Never put mutable state or imports with side-effects here.
"""

from datetime import time

# ---------------------------------------------------------------------------
# Instrument constants
# ---------------------------------------------------------------------------
LOT_SIZE: int = 50                     # Nifty options lot size (NSE)
TICK_SIZE: float = 0.05                # Minimum price movement
NIFTY_STRIKE_INTERVAL: float = 50.0   # Distance between consecutive strikes

# ---------------------------------------------------------------------------
# Session times (IST — Asia/Kolkata)
# ---------------------------------------------------------------------------
SESSION_START: time = time(9, 20)      # Avoids first-5-min chaos
SESSION_END: time = time(15, 15)       # Avoids last-15-min spread blow-out
PRE_SESSION: time = time(9, 0)         # Instrument warmup start
EOD_CLEANUP: time = time(15, 30)       # Final position force-close
DAILY_RESET: time = time(9, 0)         # Risk counters reset

# ---------------------------------------------------------------------------
# ORB (Opening Range Breakout) timing
# ---------------------------------------------------------------------------
ORB_CLOSE_TIME: time = time(9, 35)     # 15-minute opening range closes at
ORB_VALID_UNTIL: time = time(13, 0)    # ORB signals expire after this

# ---------------------------------------------------------------------------
# OI Max Pain
# ---------------------------------------------------------------------------
OI_MIN_BUILD_TIME: time = time(11, 0)  # OI meaningful after 11:00 IST

# ---------------------------------------------------------------------------
# Tuesday Gamma
# ---------------------------------------------------------------------------
TUESDAY_GAMMA_START: time = time(14, 0)  # Gamma ramp active after 14:00 IST on Tue
TUESDAY_WEEKDAY: int = 1               # Monday=0, Tuesday=1
THURSDAY_EXPIRY_WEEKDAY: int = 3       # Nifty weekly expiry

# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
WARMUP_TICKS: int = 100                # Ticks per symbol before strategies activate
WARMUP_CANDLES: int = 20              # Min completed candles before indicators valid

# ---------------------------------------------------------------------------
# Brokerage (Zerodha flat-fee model)
# ---------------------------------------------------------------------------
BROKERAGE_PER_LOT: float = 20.0        # INR per lot (flat fee per order leg)
STT_SELL_PCT: float = 0.05             # STT on sell side (options) as % of premium
EXCHANGE_TXNS_PCT: float = 0.0005      # NSE transaction charge
GST_PCT: float = 18.0                  # GST on brokerage

# ---------------------------------------------------------------------------
# Risk defaults (also configurable via env)
# ---------------------------------------------------------------------------
DEFAULT_CAPITAL: float = 1_000_000.0
DEFAULT_DAILY_LOSS_PCT: float = 2.0
DEFAULT_MAX_CONCURRENT: int = 3

# Aliases expected by settings.py
CONSECUTIVE_LOSS_COOLDOWN: int = 3
COOLDOWN_MINUTES: int = 30

# ---------------------------------------------------------------------------
# Indicator periods
# ---------------------------------------------------------------------------
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
BB_PERIOD: int = 20
BB_STD: float = 2.0
ADX_PERIOD: int = 14
EMA_FAST: int = 9
EMA_MID: int = 21
EMA_SLOW: int = 50
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
VWAP_RESET: str = "day"                # VWAP resets daily

# ---------------------------------------------------------------------------
# Option selection
# ---------------------------------------------------------------------------
DEFAULT_STRIKES_FROM_ATM: int = 5      # Subscribe ATM ± N strikes
MIN_OPTION_OI: int = 100
MAX_SPREAD_PCT: float = 3.0            # Reject if bid-ask > 3% of mid
MIN_OPTION_PRICE: float = 5.0          # Min premium to trade
MAX_OPTION_PRICE: float = 500.0        # Max premium to trade

# ---------------------------------------------------------------------------
# Black-Scholes constants
# ---------------------------------------------------------------------------
RISK_FREE_RATE: float = 0.065          # 6.5% annualized (approximate RBI repo)
TRADING_DAYS_PER_YEAR: int = 252

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
CANDLE_TIMEFRAMES: list[str] = ["1m", "5m", "15m"]
SIGNAL_DEDUP_WINDOW_SEC: int = 60      # Suppress identical signals within 60s
MAX_INDICATOR_HISTORY: int = 500       # Rolling window size per symbol+timeframe
LOOP_CADENCE_MS: int = 100             # Main event loop interval
REGIME_CACHE_MINUTES: int = 5          # Regime recalculated every 5 min
