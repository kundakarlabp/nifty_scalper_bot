# src/config.py
"""
Centralized, validated configuration for the Nifty Scalper Bot using Pydantic.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    PositiveFloat,
    PositiveInt,
    NonNegativeFloat,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExecutorConfig(BaseModel):
    """Configuration for the OrderExecutor."""
    default_product: Literal["MIS", "NRML"] = Field("MIS", alias="DEFAULT_PRODUCT")
    default_order_type: Literal["MARKET", "LIMIT"] = Field("MARKET", alias="DEFAULT_ORDER_TYPE")
    default_validity: Literal["DAY", "IOC"] = Field("DAY", alias="DEFAULT_VALIDITY")
    preferred_exit_mode: Literal["AUTO", "GTT", "REGULAR"] = Field("AUTO", alias="PREFERRED_EXIT_MODE")
    use_slm_exit: bool = Field(True, alias="USE_SLM_EXIT")
    sl_limit_offset_ticks: PositiveInt = Field(2, alias="SL_LIMIT_OFFSET_TICKS")
    trail_cooldown_sec: NonNegativeFloat = Field(12.0, alias="TRAIL_COOLDOWN_SEC")
    partial_tp_enable: bool = Field(True, alias="PARTIAL_TP_ENABLE")
    partial_tp_ratio: float = Field(0.5, gt=0, lt=1, alias="PARTIAL_TP_RATIO")
    partial_tp2_r_mult: PositiveFloat = Field(2.0, alias="PARTIAL_TP2_R_MULT")
    breakeven_after_tp1_enable: bool = Field(True, alias="BREAKEVEN_AFTER_TP1_ENABLE")
    breakeven_offset_ticks: NonNegativeFloat = Field(1.0, alias="BREAKEVEN_OFFSET_TICKS")
    nfo_freeze_qty: PositiveInt = Field(1800, alias="NFO_FREEZE_QTY")
    nifty_lot_size: PositiveInt = Field(50, alias="NIFTY_LOT_SIZE")
    tick_size: PositiveFloat = Field(0.05, alias="TICK_SIZE")


class APIConfig(BaseModel):
    """Credentials for brokers and services."""
    zerodha_api_key: str = Field("DUMMY_KEY", alias="ZERODHA_API_KEY")
    zerodha_api_secret: str = Field("DUMMY_SECRET", alias="ZERODHA_API_SECRET")
    zerodha_access_token: str = Field("DUMMY_TOKEN", alias="ZERODHA_ACCESS_TOKEN")


class TelegramConfig(BaseModel):
    """Configuration for Telegram notifications."""
    bot_token: str = Field("DUMMY_BOT_TOKEN", alias="TELEGRAM_BOT_TOKEN")
    chat_id: str = Field("12345", alias="TELEGRAM_CHAT_ID")


class RiskConfig(BaseModel):
    """Parameters controlling risk management and position sizing."""
    max_daily_drawdown_pct: float = Field(0.03, alias="MAX_DAILY_DRAWDOWN_PCT", gt=0, lt=0.5)
    max_trades_per_day: PositiveInt = Field(15, alias="MAX_TRADES_PER_DAY")
    consecutive_loss_limit: PositiveInt = Field(3, alias="CONSECUTIVE_LOSS_LIMIT")
    risk_per_trade_pct: float = Field(0.01, alias="RISK_PER_TRADE_PCT", gt=0, lt=0.2)
    min_lots: PositiveInt = Field(1, alias="MIN_LOTS")
    max_lots: PositiveInt = Field(10, alias="MAX_LOTS")


class StrategyConfig(BaseModel):
    """Parameters for tuning the trading strategy logic."""
    min_signal_score: float = Field(5.0, alias="MIN_SIGNAL_SCORE")
    confidence_threshold: float = Field(6.0, alias="CONFIDENCE_THRESHOLD", ge=0, le=10)
    time_filter_start: str = Field("09:20", alias="TIME_FILTER_START")
    time_filter_end: str = Field("15:15", alias="TIME_FILTER_END")
    atr_period: PositiveInt = Field(14, alias="ATR_PERIOD")
    atr_sl_multiplier: PositiveFloat = Field(1.5, alias="ATR_SL_MULTIPLIER")
    atr_tp_multiplier: PositiveFloat = Field(3.0, alias="ATR_TP_MULTIPLIER")
    spot_symbol: str = Field("NSE:NIFTY 50", alias="SPOT_SYMBOL")
    strike_selection_range: PositiveInt = Field(3, alias="STRIKE_SELECTION_RANGE")
    min_bars_for_signal: PositiveInt = Field(30, alias="MIN_BARS_FOR_SIGNAL")


class AppSettings(BaseSettings):
    """Main application settings, composing all other configuration models."""

    @staticmethod
    def find_dotenv() -> Path | None:
        """Find the .env file by searching upward from the current directory."""
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            env_file = parent / ".env"
            if env_file.exists():
                return env_file
        return None

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )
    enable_live_trading: bool = Field(False, alias="ENABLE_LIVE_TRADING")
    enable_telegram: bool = Field(True, alias="ENABLE_TELEGRAM")
    allow_offhours_testing: bool = Field(False, alias="ALLOW_OFFHOURS_TESTING")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    api: APIConfig = Field(default_factory=APIConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

try:
    settings = AppSettings()
except Exception as e:
    print(f"FATAL: Could not load application settings. Error: {e}")
    exit(1)

if __name__ == "__main__":
    print("--- Application Settings ---")
    print(settings.model_dump_json(indent=2))