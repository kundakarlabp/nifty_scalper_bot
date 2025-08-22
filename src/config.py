# src/config.py
"""
Simple config loader: all settings pulled from .env
No pydantic, no BaseModel, no callbacks.
"""

import os
from types import SimpleNamespace
from dotenv import load_dotenv

# Load .env file
load_dotenv()


def get_str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def get_int(key: str, default: int = 0) -> int:
    try:
        return int(os.getenv(key, default))
    except Exception:
        return default


def get_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.getenv(key, default))
    except Exception:
        return default


def get_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


class AppSettings:
    def __init__(self):
        # --- Modes / Logging ---
        self.enable_live_trading = get_bool("ENABLE_LIVE_TRADING", False)
        self.allow_offhours_testing = get_bool("ALLOW_OFFHOURS_TESTING", False)
        self.log_level = get_str("LOG_LEVEL", "INFO")

        # --- Scheduling / Hours ---
        self.time_filter_start = get_str("TIME_FILTER_START", "09:20")
        self.time_filter_end = get_str("TIME_FILTER_END", "15:20")
        self.lookback_minutes = get_int("DATA__LOOKBACK_MINUTES", 15)

        # --- Instruments ---
        self.instruments = SimpleNamespace(
            spot_symbol=get_str("INSTRUMENTS__SPOT_SYMBOL", "NSE:NIFTY 50"),
            trade_symbol=get_str("INSTRUMENTS__TRADE_SYMBOL", "NIFTY"),
            trade_exchange=get_str("INSTRUMENTS__TRADE_EXCHANGE", "NFO"),
            instrument_token=get_int("INSTRUMENTS__INSTRUMENT_TOKEN", 256265),
            nifty_lot_size=get_int("INSTRUMENTS__NIFTY_LOT_SIZE", 50),
            strike_range=get_int("INSTRUMENTS__STRIKE_RANGE", 3),
            min_lots=get_int("INSTRUMENTS__MIN_LOTS", 1),
            max_lots=get_int("INSTRUMENTS__MAX_LOTS", 15),
        )

        # --- Strategy ---
        self.strategy = SimpleNamespace(
            min_signal_score=get_float("STRATEGY__MIN_SIGNAL_SCORE", 2.0),
            confidence_threshold=get_float("STRATEGY__CONFIDENCE_THRESHOLD", 2.0),
            atr_period=get_int("STRATEGY__ATR_PERIOD", 14),
            atr_sl_multiplier=get_float("STRATEGY__ATR_SL_MULTIPLIER", 1.5),
            atr_tp_multiplier=get_float("STRATEGY__ATR_TP_MULTIPLIER", 3.0),
            sl_confidence_adj=get_float("STRATEGY__SL_CONFIDENCE_ADJ", 0.12),
            tp_confidence_adj=get_float("STRATEGY__TP_CONFIDENCE_ADJ", 0.35),
            min_bars_for_signal=get_int("STRATEGY__MIN_BARS_FOR_SIGNAL", 10),
        )

        # --- Risk ---
        self.risk = SimpleNamespace(
            default_equity=get_float("RISK__DEFAULT_EQUITY", 30000.0),
            risk_per_trade=get_float("RISK__RISK_PER_TRADE", 0.02),
            max_trades_per_day=get_int("RISK__MAX_TRADES_PER_DAY", 20),
            consecutive_loss_limit=get_int("RISK__CONSECUTIVE_LOSS_LIMIT", 3),
            max_daily_drawdown_pct=get_float("RISK__MAX_DAILY_DRAWDOWN_PCT", 0.05),
        )

        # --- Executor ---
        self.executor = SimpleNamespace(
            exchange=get_str("EXECUTOR__EXCHANGE", "NFO"),
            order_product=get_str("EXECUTOR__ORDER_PRODUCT", "NRML"),
            order_variety=get_str("EXECUTOR__ORDER_VARIETY", "regular"),
            entry_order_type=get_str("EXECUTOR__ENTRY_ORDER_TYPE", "LIMIT"),
            tick_size=get_float("EXECUTOR__TICK_SIZE", 0.05),
            exchange_freeze_qty=get_int("EXECUTOR__EXCHANGE_FREEZE_QTY", 1800),
            preferred_exit_mode=get_str("EXECUTOR__PREFERRED_EXIT_MODE", "REGULAR"),
            use_slm_exit=get_bool("EXECUTOR__USE_SLM_EXIT", True),
            partial_tp_enable=get_bool("EXECUTOR__PARTIAL_TP_ENABLE", True),
            tp1_qty_ratio=get_float("EXECUTOR__TP1_QTY_RATIO", 0.5),
            breakeven_ticks=get_int("EXECUTOR__BREAKEVEN_TICKS", 2),
            enable_trailing=get_bool("EXECUTOR__ENABLE_TRAILING", True),
            trailing_atr_multiplier=get_float("EXECUTOR__TRAILING_ATR_MULTIPLIER", 1.5),
            fee_per_lot=get_float("EXECUTOR__FEE_PER_LOT", 20.0),
        )

        # --- API Keys ---
        self.telegram = SimpleNamespace(
            bot_token=get_str("TELEGRAM_BOT_TOKEN", ""),
            chat_id=get_str("TELEGRAM_CHAT_ID", ""),
        )
        self.zerodha = SimpleNamespace(
            api_key=get_str("ZERODHA_API_KEY", ""),
            api_secret=get_str("ZERODHA_API_SECRET", ""),
            access_token=get_str("ZERODHA_ACCESS_TOKEN", ""),
        )

    def summary(self) -> str:
        return (
            f"Trading={'LIVE' if self.enable_live_trading else 'DRY'} | "
            f"Risk={self.risk.risk_per_trade*100:.1f}% per trade | "
            f"Equity=₹{self.risk.default_equity:.0f} | "
            f"Lots={self.instruments.min_lots}-{self.instruments.max_lots}"
        )


settings = AppSettings()