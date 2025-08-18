# src/config.py
"""
Centralized, validated configuration for the Nifty Scalper Bot (Pydantic v2).

- Provides structured settings grouped by concern (API/Telegram/Risk/Strategy/Executor).
- Loads from .env (searched upward from CWD) with well-named UPPERCASE aliases.
- Exposes a modern `settings` object (AppSettings) for new code.
- Exposes a backward-compat `Config` class so legacy imports keep working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    BaseSettings,
    Field,
    PositiveInt,
    PositiveFloat,
    NonNegativeFloat,
    field_validator,
)
from pydantic_settings import SettingsConfigDict


# ============================ Sub-models ============================ #

class ExecutorConfig(BaseModel):
    """
    Order execution & session timing parameters.
    Many of these are thin wrappers around exchange/broker realities.
    """
    trade_symbol: str = Field("NIFTY", alias="TRADE_SYMBOL")
    trade_exchange: str = Field("NFO", alias="TRADE_EXCHANGE")

    # Session / time gates (HH:MM 24h IST)
    market_open: str = Field("09:15", alias="TIME_FILTER_START")
    market_close: str = Field("15:30", alias="TIME_FILTER_END")

    # Data / lookback
    data_lookback_minutes: PositiveInt = Field(60, alias="DATA_LOOKBACK_MINUTES")

    # Breakeven & tick geometry
    breakeven_after_tp1_enable: bool = Field(True, alias="BREAKEVEN_AFTER_TP1_ENABLE")
    breakeven_offset_ticks: NonNegativeFloat = Field(1.0, alias="BREAKEVEN_OFFSET_TICKS")
    tick_size: PositiveFloat = Field(0.05, alias="TICK_SIZE")

    # Exchange specifics
    nfo_freeze_qty: PositiveInt = Field(1800, alias="NFO_FREEZE_QTY")
    # NIFTY lot size changed historically (75 → 50). Keep default 50 to match the refactor branch.
    nifty_lot_size: PositiveInt = Field(50, alias="NIFTY_LOT_SIZE")


class APIConfig(BaseModel):
    """Credentials for brokers and services (env only; never print)."""
    zerodha_api_key: str = Field("DUMMY_KEY", alias="ZERODHA_API_KEY")
    zerodha_api_secret: str = Field("DUMMY_SECRET", alias="ZERODHA_API_SECRET")
    zerodha_access_token: str = Field("DUMMY_TOKEN", alias="ZERODHA_ACCESS_TOKEN")


class TelegramConfig(BaseModel):
    """Configuration for Telegram notifications & control."""
    bot_token: str = Field("DUMMY_BOT_TOKEN", alias="TELEGRAM_BOT_TOKEN")
    chat_id: str = Field("12345", alias="TELEGRAM_CHAT_ID")


class RiskConfig(BaseModel):
    """Risk & position sizing policy."""
    max_daily_drawdown_pct: float = Field(0.03, alias="MAX_DAILY_DRAWDOWN_PCT", gt=0, lt=0.5)
    max_trades_per_day: PositiveInt = Field(15, alias="MAX_TRADES_PER_DAY")
    consecutive_loss_limit: PositiveInt = Field(3, alias="CONSECUTIVE_LOSS_LIMIT")
    risk_per_trade_pct: float = Field(0.01, alias="RISK_PER_TRADE_PCT", gt=0, lt=0.2)
    min_lots: PositiveInt = Field(1, alias="MIN_LOTS")
    max_lots: PositiveInt = Field(10, alias="MAX_LOTS")


class StrategyConfig(BaseModel):
    """
    Core strategy thresholds & knobs.
    Keep semantics identical in LEGACY mode; place add-ons behind toggles elsewhere.
    """
    # Core scoring / thresholds
    min_signal_score: int = Field(1, alias="MIN_SIGNAL_SCORE")
    confidence_threshold: float = Field(4.0, alias="CONFIDENCE_THRESHOLD")

    # ATR / SL / TP model
    atr_period: PositiveInt = Field(14, alias="ATR_PERIOD")
    atr_sl_multiplier: PositiveFloat = Field(1.5, alias="ATR_SL_MULTIPLIER")
    atr_tp_multiplier: PositiveFloat = Field(3.0, alias="ATR_TP_MULTIPLIER")
    sl_confidence_adj: float = Field(0.2, alias="SL_CONFIDENCE_ADJ")
    tp_confidence_adj: float = Field(0.3, alias="TP_CONFIDENCE_ADJ")

    # Instruments / selection
    spot_symbol: str = Field("NSE:NIFTY 50", alias="SPOT_SYMBOL")
    strike_selection_range: PositiveInt = Field(3, alias="STRIKE_SELECTION_RANGE")

    # Warmup / indicators
    min_bars_for_signal: PositiveInt = Field(30, alias="MIN_BARS_FOR_SIGNAL")


# ============================ Root settings ============================ #

class AppSettings(BaseSettings):
    """
    Main application settings, composing all other configuration models.
    Loads from a discovered .env file (search upward from CWD) and environment.
    """

    @staticmethod
    def find_dotenv() -> Path | None:
        """Find a .env file by searching upward from the current directory."""
        cwd = Path.cwd()
        for parent in (cwd, *cwd.parents):
            env_file = parent / ".env"
            if env_file.exists():
                return env_file
        return None

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",  # allow NESTED__FIELD style
        extra="ignore",
    )

    # Top-level toggles & logging
    enable_live_trading: bool = Field(False, alias="ENABLE_LIVE_TRADING")
    enable_telegram: bool = Field(True, alias="ENABLE_TELEGRAM")
    allow_offhours_testing: bool = Field(False, alias="ALLOW_OFFHOURS_TESTING")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Sub-models
    api: APIConfig = Field(default_factory=APIConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = (v or "").upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {sorted(valid)}")
        return upper


# Instantiate settings (raise, don’t exit, on failure)
try:
    settings = AppSettings()
except Exception as e:
    raise RuntimeError(f"Could not load application settings: {e}") from e


# ============================ Legacy shim ============================ #
# Many legacy files import `from src.config import Config` and expect UPPERCASE attributes.
# Keep that contract to avoid widespread edits.

class Config:  # noqa: N801  (legacy name)
    # top-level toggles and logging
    ENABLE_LIVE_TRADING = settings.enable_live_trading
    ENABLE_TELEGRAM = settings.enable_telegram
    ALLOW_OFFHOURS_TESTING = settings.allow_offhours_testing
    LOG_LEVEL = settings.log_level

    # Zerodha API
    ZERODHA_API_KEY = settings.api.zerodha_api_key
    ZERODHA_API_SECRET = settings.api.zerodha_api_secret
    ZERODHA_ACCESS_TOKEN = settings.api.zerodha_access_token

    # Telegram
    TELEGRAM_BOT_TOKEN = settings.telegram.bot_token
    TELEGRAM_CHAT_ID = settings.telegram.chat_id

    # Risk
    MAX_DAILY_DRAWDOWN_PCT = settings.risk.max_daily_drawdown_pct
    MAX_TRADES_PER_DAY = settings.risk.max_trades_per_day
    CONSECUTIVE_LOSS_LIMIT = settings.risk.consecutive_loss_limit
    RISK_PER_TRADE_PCT = settings.risk.risk_per_trade_pct
    MIN_LOTS = settings.risk.min_lots
    MAX_LOTS = settings.risk.max_lots

    # Strategy
    MIN_SIGNAL_SCORE = settings.strategy.min_signal_score
    CONFIDENCE_THRESHOLD = settings.strategy.confidence_threshold
    ATR_PERIOD = settings.strategy.atr_period
    ATR_SL_MULTIPLIER = settings.strategy.atr_sl_multiplier
    ATR_TP_MULTIPLIER = settings.strategy.atr_tp_multiplier
    SL_CONFIDENCE_ADJ = settings.strategy.sl_confidence_adj
    TP_CONFIDENCE_ADJ = settings.strategy.tp_confidence_adj

    # Instruments
    SPOT_SYMBOL = settings.strategy.spot_symbol
    TRADE_SYMBOL = settings.executor.trade_symbol
    TRADE_EXCHANGE = settings.executor.trade_exchange
    NIFTY_LOT_SIZE = settings.executor.nifty_lot_size
    STRIKE_RANGE = settings.strategy.strike_selection_range

    # Time/Data
    WARMUP_BARS = settings.strategy.min_bars_for_signal
    DATA_LOOKBACK_MINUTES = settings.executor.data_lookback_minutes
    TIME_FILTER_START = settings.executor.market_open
    TIME_FILTER_END = settings.executor.market_close


if __name__ == "__main__":
    # Small self-check (prints sanitized summary; no secrets)
    import json

    summary = {
        "enable_live_trading": settings.enable_live_trading,
        "enable_telegram": settings.enable_telegram,
        "log_level": settings.log_level,
        "spot_symbol": settings.strategy.spot_symbol,
        "market_open": settings.executor.market_open,
        "market_close": settings.executor.market_close,
        "nifty_lot_size": settings.executor.nifty_lot_size,
    }
    print("--- Application Settings (sanitized) ---")
    print(json.dumps(summary, indent=2))
