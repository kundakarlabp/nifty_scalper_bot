# src/config.py
from __future__ import annotations

import logging
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


class AppSettings(BaseSettings):
    # --- Modes / Logging ---
    enable_live_trading: bool = Field(default=False, alias="ENABLE_LIVE_TRADING")
    allow_offhours_testing: bool = Field(default=False, alias="ALLOW_OFFHOURS_TESTING")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- Scheduling / Hours ---
    time_filter_start: str = Field(default="09:20", alias="TIME_FILTER_START")
    time_filter_end: str = Field(default="15:20", alias="TIME_FILTER_END")
    data__lookback_minutes: int = Field(default=15, alias="DATA__LOOKBACK_MINUTES")

    # --- Instruments ---
    instruments__spot_symbol: str = Field(default="NSE:NIFTY 50", alias="INSTRUMENTS__SPOT_SYMBOL")
    instruments__trade_symbol: str = Field(default="NIFTY", alias="INSTRUMENTS__TRADE_SYMBOL")
    instruments__trade_exchange: str = Field(default="NFO", alias="INSTRUMENTS__TRADE_EXCHANGE")
    instruments__instrument_token: int = Field(default=256265, alias="INSTRUMENTS__INSTRUMENT_TOKEN")
    instruments__nifty_lot_size: int = Field(default=50, alias="INSTRUMENTS__NIFTY_LOT_SIZE")
    instruments__strike_range: int = Field(default=3, alias="INSTRUMENTS__STRIKE_RANGE")
    instruments__min_lots: int = Field(default=1, alias="INSTRUMENTS__MIN_LOTS")
    instruments__max_lots: int = Field(default=100, alias="INSTRUMENTS__MAX_LOTS")

    # --- Strategy Core ---
    strategy__min_signal_score: int = Field(default=2, alias="STRATEGY__MIN_SIGNAL_SCORE")
    strategy__confidence_threshold: float = Field(default=2.0, alias="STRATEGY__CONFIDENCE_THRESHOLD")
    strategy__atr_period: int = Field(default=14, alias="STRATEGY__ATR_PERIOD")
    strategy__atr_sl_multiplier: float = Field(default=1.5, alias="STRATEGY__ATR_SL_MULTIPLIER")
    strategy__atr_tp_multiplier: float = Field(default=3.0, alias="STRATEGY__ATR_TP_MULTIPLIER")
    strategy__sl_confidence_adj: float = Field(default=0.12, alias="STRATEGY__SL_CONFIDENCE_ADJ")
    strategy__tp_confidence_adj: float = Field(default=0.35, alias="STRATEGY__TP_CONFIDENCE_ADJ")
    strategy__min_bars_for_signal: int = Field(default=10, alias="STRATEGY__MIN_BARS_FOR_SIGNAL")

    # --- Risk / Limits ---
    risk__default_equity: float = Field(default=30000, alias="RISK__DEFAULT_EQUITY")
    risk__risk_per_trade: float = Field(default=0.01, alias="RISK__RISK_PER_TRADE")
    risk__max_trades_per_day: int = Field(default=20, alias="RISK__MAX_TRADES_PER_DAY")
    risk__consecutive_loss_limit: int = Field(default=3, alias="RISK__CONSECUTIVE_LOSS_LIMIT")
    risk__max_daily_drawdown_pct: float = Field(default=0.05, alias="RISK__MAX_DAILY_DRAWDOWN_PCT")

    # --- Execution ---
    executor__exchange: str = Field(default="NFO", alias="EXECUTOR__EXCHANGE")
    executor__order_product: str = Field(default="NRML", alias="EXECUTOR__ORDER_PRODUCT")
    executor__order_variety: str = Field(default="regular", alias="EXECUTOR__ORDER_VARIETY")
    executor__entry_order_type: str = Field(default="LIMIT", alias="EXECUTOR__ENTRY_ORDER_TYPE")
    executor__tick_size: float = Field(default=0.05, alias="EXECUTOR__TICK_SIZE")
    executor__exchange_freeze_qty: int = Field(default=1800, alias="EXECUTOR__EXCHANGE_FREEZE_QTY")
    executor__preferred_exit_mode: str = Field(default="REGULAR", alias="EXECUTOR__PREFERRED_EXIT_MODE")
    executor__use_slm_exit: bool = Field(default=True, alias="EXECUTOR__USE_SLM_EXIT")
    executor__partial_tp_enable: bool = Field(default=True, alias="EXECUTOR__PARTIAL_TP_ENABLE")
    executor__tp1_qty_ratio: float = Field(default=0.5, alias="EXECUTOR__TP1_QTY_RATIO")
    executor__breakeven_ticks: int = Field(default=2, alias="EXECUTOR__BREAKEVEN_TICKS")
    executor__enable_trailing: bool = Field(default=True, alias="EXECUTOR__ENABLE_TRAILING")
    executor__trailing_atr_multiplier: float = Field(default=1.5, alias="EXECUTOR__TRAILING_ATR_MULTIPLIER")
    executor__fee_per_lot: float = Field(default=20.0, alias="EXECUTOR__FEE_PER_LOT")

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings()


settings = get_settings()
logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")