# Path: src/config.py
from __future__ import annotations

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class TradingSettings(BaseSettings):
    # --- CORE ---
    enable_live_trading: bool = Field(default=False, env="ENABLE_LIVE_TRADING")
    allow_offhours_testing: bool = Field(default=False, env="ALLOW_OFFHOURS_TESTING")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    data_lookback_minutes: int = Field(default=30, env="DATA_LOOKBACK_MINUTES")
    time_filter_start: str = Field(default="09:20", env="TIME_FILTER_START")
    time_filter_end: str = Field(default="15:20", env="TIME_FILTER_END")

    # --- BROKER (Zerodha) ---
    zerodha_api_key: str = Field(default="", env="ZERODHA_API_KEY")
    zerodha_api_secret: str = Field(default="", env="ZERODHA_API_SECRET")
    zerodha_access_token: str = Field(default="", env="ZERODHA_ACCESS_TOKEN")

    # --- TELEGRAM ---
    telegram_enabled: bool = Field(default=True, env="TELEGRAM_ENABLED")
    telegram_bot_token: str = Field(default="", env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", env="TELEGRAM_CHAT_ID")

    # --- INSTRUMENTS ---
    instruments_spot_symbol: str = Field(default="NSE:NIFTY 50", env="INSTRUMENTS_SPOT_SYMBOL")
    instruments_trade_symbol: str = Field(default="NIFTY", env="INSTRUMENTS_TRADE_SYMBOL")
    instruments_trade_exchange: str = Field(default="NFO", env="INSTRUMENTS_TRADE_EXCHANGE")
    instruments_instrument_token: int = Field(default=256265, env="INSTRUMENTS_INSTRUMENT_TOKEN")
    instruments_nifty_lot_size: int = Field(default=75, env="INSTRUMENTS_NIFTY_LOT_SIZE")
    instruments_strike_range: int = Field(default=0, env="INSTRUMENTS_STRIKE_RANGE")
    instruments_min_lots: int = Field(default=1, env="INSTRUMENTS_MIN_LOTS")
    instruments_max_lots: int = Field(default=10, env="INSTRUMENTS_MAX_LOTS")

    # --- STRATEGY ---
    strategy_min_signal_score: int = Field(default=3, env="STRATEGY_MIN_SIGNAL_SCORE")
    strategy_confidence_threshold: float = Field(default=55.0, env="STRATEGY_CONFIDENCE_THRESHOLD")
    strategy_min_bars_for_signal: int = Field(default=50, env="STRATEGY_MIN_BARS_FOR_SIGNAL")
    strategy_ema_fast: int = Field(default=9, env="STRATEGY_EMA_FAST")
    strategy_ema_slow: int = Field(default=21, env="STRATEGY_EMA_SLOW")
    strategy_rsi_period: int = Field(default=14, env="STRATEGY_RSI_PERIOD")
    strategy_bb_period: int = Field(default=20, env="STRATEGY_BB_PERIOD")
    strategy_bb_std: float = Field(default=2.0, env="STRATEGY_BB_STD")
    strategy_atr_period: int = Field(default=14, env="STRATEGY_ATR_PERIOD")
    strategy_atr_sl_multiplier: float = Field(default=1.3, env="STRATEGY_ATR_SL_MULTIPLIER")
    strategy_atr_tp_multiplier: float = Field(default=2.2, env="STRATEGY_ATR_TP_MULTIPLIER")

    # --- RISK (live equity controls) ---
    risk_use_live_equity: bool = Field(default=True, env="RISK_USE_LIVE_EQUITY")
    risk_default_equity: float = Field(default=30000.0, env="RISK_DEFAULT_EQUITY")
    risk_min_equity_floor: float = Field(default=25000.0, env="RISK_MIN_EQUITY_FLOOR")
    equity_refresh_seconds: int = Field(default=60, env="EQUITY_REFRESH_SECONDS")
    risk_risk_per_trade: float = Field(default=0.01, env="RISK_RISK_PER_TRADE")
    risk_max_trades_per_day: int = Field(default=12, env="RISK_MAX_TRADES_PER_DAY")
    risk_consecutive_loss_limit: int = Field(default=3, env="RISK_CONSECUTIVE_LOSS_LIMIT")
    risk_max_daily_drawdown_pct: float = Field(default=0.04, env="RISK_MAX_DAILY_DRAWDOWN_PCT")
    risk_max_position_size_pct: float = Field(default=0.10, env="RISK_MAX_POSITION_SIZE_PCT")

    # --- EXECUTION ---
    executor_exchange: str = Field(default="NFO", env="EXECUTOR_EXCHANGE")
    executor_order_product: str = Field(default="NRML", env="EXECUTOR_ORDER_PRODUCT")
    executor_order_variety: str = Field(default="regular", env="EXECUTOR_ORDER_VARIETY")
    executor_entry_order_type: str = Field(default="LIMIT", env="EXECUTOR_ENTRY_ORDER_TYPE")
    executor_tick_size: float = Field(default=0.05, env="EXECUTOR_TICK_SIZE")
    executor_exchange_freeze_qty: int = Field(default=1800, env="EXECUTOR_EXCHANGE_FREEZE_QTY")
    executor_preferred_exit_mode: str = Field(default="REGULAR", env="EXECUTOR_PREFERRED_EXIT_MODE")
    executor_use_slm_exit: bool = Field(default=True, env="EXECUTOR_USE_SLM_EXIT")
    executor_partial_tp_enable: bool = Field(default=True, env="EXECUTOR_PARTIAL_TP_ENABLE")
    executor_tp1_qty_ratio: float = Field(default=0.5, env="EXECUTOR_TP1_QTY_RATIO")
    executor_breakeven_ticks: int = Field(default=2, env="EXECUTOR_BREAKEVEN_TICKS")
    executor_enable_trailing: bool = Field(default=True, env="EXECUTOR_ENABLE_TRAILING")
    executor_trailing_atr_multiplier: float = Field(default=1.4, env="EXECUTOR_TRAILING_ATR_MULTIPLIER")
    executor_fee_per_lot: float = Field(default=20.0, env="EXECUTOR_FEE_PER_LOT")

    # --- MONITORING / SYSTEM ---
    enable_health_server: bool = Field(default=True, env="ENABLE_HEALTH_SERVER")
    health_server_port: int = Field(default=8080, env="HEALTH_SERVER_PORT")
    enable_performance_tracking: bool = Field(default=True, env="ENABLE_PERFORMANCE_TRACKING")
    enable_trade_logging: bool = Field(default=True, env="ENABLE_TRADE_LOGGING")
    max_api_calls_per_second: float = Field(default=8.0, env="MAX_API_CALLS_PER_SECOND")
    websocket_reconnect_attempts: int = Field(default=5, env="WEBSOCKET_RECONNECT_ATTEMPTS")
    order_timeout_seconds: int = Field(default=30, env="ORDER_TIMEOUT_SECONDS")
    position_sync_interval: int = Field(default=60, env="POSITION_SYNC_INTERVAL")

    @validator("risk_risk_per_trade")
    def _v_risk_pct(cls, v: float) -> float:
        if not 0.001 <= v <= 0.10:
            raise ValueError("Risk per trade must be between 0.1% and 10%")
        return v

    @validator("risk_max_daily_drawdown_pct")
    def _v_dd_pct(cls, v: float) -> float:
        if not 0.01 <= v <= 0.20:
            raise ValueError("Max daily drawdown must be between 1% and 20%")
        return v

    @validator("strategy_confidence_threshold")
    def _v_conf(cls, v: float) -> float:
        if not 0 <= v <= 100:
            raise ValueError("Confidence threshold must be 0..100")
        return v

    @validator("instruments_min_lots", "instruments_max_lots")
    def _v_lots(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Lot sizes must be positive")
        return v

    @validator("time_filter_start", "time_filter_end")
    def _v_time(cls, v: str) -> str:
        from datetime import datetime
        datetime.strptime(v, "%H:%M")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class DevelopmentSettings(TradingSettings):
    enable_live_trading: bool = False
    allow_offhours_testing: bool = True
    log_level: str = "DEBUG"
    risk_max_trades_per_day: int = 5


class ProductionSettings(TradingSettings):
    enable_live_trading: bool = True
    allow_offhours_testing: bool = False
    log_level: str = "INFO"


def get_settings() -> TradingSettings:
    env = os.getenv("TRADING_ENV", "development").lower()
    if env == "production":
        return ProductionSettings()
    if env == "development":
        return DevelopmentSettings()
    return TradingSettings()


settings = get_settings()


def validate_critical_settings() -> None:
    errs = []

    if settings.enable_live_trading:
        if not settings.zerodha_api_key:
            errs.append("ZERODHA_API_KEY is required for live trading")
        if not settings.zerodha_api_secret:
            errs.append("ZERODHA_API_SECRET is required for live trading")
        if not settings.zerodha_access_token:
            errs.append("ZERODHA_ACCESS_TOKEN is required for live trading")

    if settings.telegram_enabled:
        if not settings.telegram_bot_token:
            errs.append("TELEGRAM_BOT_TOKEN is required when Telegram is enabled")
        if not settings.telegram_chat_id:
            errs.append("TELEGRAM_CHAT_ID is required when Telegram is enabled")

    if settings.instruments_max_lots < settings.instruments_min_lots:
        errs.append("INSTRUMENTS_MAX_LOTS must be >= INSTRUMENTS_MIN_LOTS")

    if settings.strategy_ema_fast >= settings.strategy_ema_slow:
        errs.append("STRATEGY_EMA_FAST must be < STRATEGY_EMA_SLOW")

    if settings.strategy_atr_tp_multiplier <= settings.strategy_atr_sl_multiplier:
        errs.append("STRATEGY_ATR_TP_MULTIPLIER should be > STRATEGY_ATR_SL_MULTIPLIER")

    if settings.risk_use_live_equity and settings.risk_min_equity_floor <= 0:
        errs.append("RISK_MIN_EQUITY_FLOOR must be > 0 when RISK_USE_LIVE_EQUITY=true")

    if errs:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errs))


validate_critical_settings()
