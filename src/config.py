# src/config.py
from __future__ import annotations

import logging
from functools import lru_cache
from pydantic import BaseSettings, Field

log = logging.getLogger(__name__)


# --- Server ---
class ServerSettings(BaseSettings):
    host: str = Field("0.0.0.0", env="SERVER__HOST")
    port: int = Field(8000, env="SERVER__PORT")


# --- Instruments ---
class InstrumentsSettings(BaseSettings):
    spot_symbol: str = Field(..., env="INSTRUMENTS__SPOT_SYMBOL")
    trade_symbol: str = Field(..., env="INSTRUMENTS__TRADE_SYMBOL")
    trade_exchange: str = Field(..., env="INSTRUMENTS__TRADE_EXCHANGE")
    instrument_token: int = Field(..., env="INSTRUMENTS__INSTRUMENT_TOKEN")
    nifty_lot_size: int = Field(50, env="INSTRUMENTS__NIFTY_LOT_SIZE")
    strike_range: int = Field(3, env="INSTRUMENTS__STRIKE_RANGE")
    min_lots: int = Field(1, env="INSTRUMENTS__MIN_LOTS")
    max_lots: int = Field(100, env="INSTRUMENTS__MAX_LOTS")


# --- Strategy ---
class StrategySettings(BaseSettings):
    min_signal_score: int = Field(2, env="STRATEGY__MIN_SIGNAL_SCORE")
    confidence_threshold: float = Field(2.0, env="STRATEGY__CONFIDENCE_THRESHOLD")
    atr_period: int = Field(14, env="STRATEGY__ATR_PERIOD")
    atr_sl_multiplier: float = Field(1.5, env="STRATEGY__ATR_SL_MULTIPLIER")
    atr_tp_multiplier: float = Field(3.0, env="STRATEGY__ATR_TP_MULTIPLIER")
    sl_confidence_adj: float = Field(0.12, env="STRATEGY__SL_CONFIDENCE_ADJ")
    tp_confidence_adj: float = Field(0.35, env="STRATEGY__TP_CONFIDENCE_ADJ")
    min_bars_for_signal: int = Field(10, env="STRATEGY__MIN_BARS_FOR_SIGNAL")


# --- Risk ---
class RiskSettings(BaseSettings):
    default_equity: float = Field(30000, env="RISK__DEFAULT_EQUITY")
    risk_per_trade: float = Field(0.02, env="RISK__RISK_PER_TRADE")
    max_trades_per_day: int = Field(30, env="RISK__MAX_TRADES_PER_DAY")
    consecutive_loss_limit: int = Field(3, env="RISK__CONSECUTIVE_LOSS_LIMIT")
    max_daily_drawdown_pct: float = Field(0.05, env="RISK__MAX_DAILY_DRAWDOWN_PCT")


# --- Execution ---
class ExecutorSettings(BaseSettings):
    exchange: str = Field("NFO", env="EXECUTOR__EXCHANGE")
    order_product: str = Field("NRML", env="EXECUTOR__ORDER_PRODUCT")
    order_variety: str = Field("regular", env="EXECUTOR__ORDER_VARIETY")
    entry_order_type: str = Field("LIMIT", env="EXECUTOR__ENTRY_ORDER_TYPE")
    tick_size: float = Field(0.05, env="EXECUTOR__TICK_SIZE")
    exchange_freeze_qty: int = Field(1800, env="EXECUTOR__EXCHANGE_FREEZE_QTY")
    preferred_exit_mode: str = Field("REGULAR", env="EXECUTOR__PREFERRED_EXIT_MODE")
    use_slm_exit: bool = Field(True, env="EXECUTOR__USE_SLM_EXIT")
    partial_tp_enable: bool = Field(True, env="EXECUTOR__PARTIAL_TP_ENABLE")
    tp1_qty_ratio: float = Field(0.5, env="EXECUTOR__TP1_QTY_RATIO")
    breakeven_ticks: int = Field(2, env="EXECUTOR__BREAKEVEN_TICKS")
    enable_trailing: bool = Field(True, env="EXECUTOR__ENABLE_TRAILING")
    trailing_atr_multiplier: float = Field(1.5, env="EXECUTOR__TRAILING_ATR_MULTIPLIER")
    fee_per_lot: float = Field(20.0, env="EXECUTOR__FEE_PER_LOT")


# --- AppSettings ---
class AppSettings(BaseSettings):
    enable_live_trading: bool = Field(False, env="ENABLE_LIVE_TRADING")
    allow_offhours_testing: bool = Field(False, env="ALLOW_OFFHOURS_TESTING")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    time_filter_start: str = Field("09:20", env="TIME_FILTER_START")
    time_filter_end: str = Field("15:20", env="TIME_FILTER_END")
    data_lookback_minutes: int = Field(15, env="DATA__LOOKBACK_MINUTES")

    instruments: InstrumentsSettings = InstrumentsSettings()
    strategy: StrategySettings = StrategySettings()
    risk: RiskSettings = RiskSettings()
    executor: ExecutorSettings = ExecutorSettings()
    server: ServerSettings = ServerSettings()  # ✅ added back for main.py health server


@lru_cache()
def get_settings() -> AppSettings:
    settings = AppSettings()
    log.info("Settings loaded successfully.")
    return settings


settings = get_settings()