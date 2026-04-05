"""
Pydantic-Settings configuration for NiftyScalperBot v2.

Environment variable prefix convention:
    BROKER__*           BrokerSettings
    TRADING__*          TradingSettings
    RISK__*             RiskSettings
    LIFECYCLE__*        LifecycleSettings
    EXECUTION__*        ExecutionSettings
    STREAMING__*        StreamingSettings
    STRATEGY_<NAME>__*  Per-strategy StrategySettings
    TELEGRAM__*         TelegramSettings
    INFRA__*            InfraSettings

The double-underscore delimiter is set via SettingsConfigDict(env_nested_delimiter="__").
"""

from __future__ import annotations

import os
from datetime import time
from enum import Enum
from functools import lru_cache
from typing import Any, Optional

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import (
    BROKERAGE_PER_LOT,
    CONSECUTIVE_LOSS_COOLDOWN,
    COOLDOWN_MINUTES,
    DEFAULT_DAILY_LOSS_PCT as DAILY_LOSS_PCT,
    LOT_SIZE,
    MAX_OPTION_PRICE,
    MAX_SPREAD_PCT,
    MIN_OPTION_OI,
    MIN_OPTION_PRICE,
    NIFTY_STRIKE_INTERVAL,
    TICK_SIZE,
    WARMUP_CANDLES,
    WARMUP_TICKS,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExecutionMode(str, Enum):
    PAPER = "PAPER"
    LIVE = "LIVE"


class SlippageModel(str, Enum):
    SPREAD_HALF = "SPREAD_HALF"
    FIXED_TICKS = "FIXED_TICKS"
    ZERO = "ZERO"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SLM = "SL-M"


class LogFormat(str, Enum):
    JSON = "json"
    CONSOLE = "console"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class PreferredExpiry(str, Enum):
    WEEKLY = "weekly"
    MONTHLY = "monthly"


# ---------------------------------------------------------------------------
# BrokerSettings
# ---------------------------------------------------------------------------


class BrokerSettings(BaseSettings):
    """Zerodha Kite Connect credentials and connection tuning."""

    model_config = SettingsConfigDict(
        env_prefix="BROKER__",
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
    )

    api_key: SecretStr = Field(default=SecretStr(""), description="Zerodha Kite API key")
    api_secret: SecretStr = Field(default=SecretStr(""), description="Zerodha Kite API secret")
    access_token: SecretStr = Field(
        default=SecretStr(""),
        description="Session access token (refreshed daily)",
    )
    base_url: str = Field(default="https://api.kite.trade", description="Kite Connect REST base URL")
    ws_url: str = Field(default="wss://ws.kite.trade", description="Kite Connect WebSocket URL")
    timeout_seconds: float = Field(default=10.0, ge=1.0, le=120.0)
    connect_retries: int = Field(default=3, ge=0, le=10)
    rate_limit_orders_per_sec: float = Field(default=5.0, gt=0.0)
    rate_limit_rest_per_sec: float = Field(default=10.0, gt=0.0)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BrokerSettings("
            f"api_key=***, "
            f"base_url={self.base_url!r}, "
            f"ws_url={self.ws_url!r}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"connect_retries={self.connect_retries}"
            f")"
        )


# ---------------------------------------------------------------------------
# TradingSettings
# ---------------------------------------------------------------------------


class TradingSettings(BaseSettings):
    """Core trading instrument and lot-size configuration."""

    model_config = SettingsConfigDict(
        env_prefix="TRADING__",
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
    )

    execution_mode: ExecutionMode = Field(default=ExecutionMode.PAPER)
    enable_live: bool = Field(
        default=False,
        description="Hard safety gate — must be True AND execution_mode=LIVE to place real orders",
    )
    instrument: str = Field(default="NIFTY")
    exchange: str = Field(default="NFO")
    lot_size: int = Field(default=LOT_SIZE, gt=0)
    tick_size: float = Field(default=TICK_SIZE, gt=0.0)
    strike_interval: int = Field(default=int(NIFTY_STRIKE_INTERVAL), gt=0)
    max_strikes_from_atm: int = Field(default=5, ge=1, le=20)
    preferred_expiry: PreferredExpiry = Field(default=PreferredExpiry.WEEKLY)

    @model_validator(mode="after")
    def _live_guard(self) -> "TradingSettings":
        if self.execution_mode == ExecutionMode.LIVE and not self.enable_live:
            raise ValueError(
                "TRADING__EXECUTION_MODE=LIVE requires TRADING__ENABLE_LIVE=true. "
                "Set explicitly to acknowledge live trading risks."
            )
        return self


# ---------------------------------------------------------------------------
# RiskSettings
# ---------------------------------------------------------------------------


class RiskSettings(BaseSettings):
    """Position sizing, drawdown limits, and filter thresholds."""

    model_config = SettingsConfigDict(
        env_prefix="RISK__",
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
    )

    capital: float = Field(default=1_000_000.0, gt=0.0, description="Notional capital in INR")
    max_daily_trades: int = Field(default=30, ge=1)
    daily_loss_pct: float = Field(
        default=DAILY_LOSS_PCT,
        gt=0.0,
        le=100.0,
        description="Max intraday loss as % of capital before halt",
    )
    max_order_notional: float = Field(default=200_000.0, gt=0.0)
    max_concurrent_positions: int = Field(default=3, ge=1)
    max_positions_per_symbol: int = Field(default=1, ge=1)
    consecutive_loss_cooldown: int = Field(default=CONSECUTIVE_LOSS_COOLDOWN, ge=1)
    cooldown_minutes: int = Field(default=COOLDOWN_MINUTES, ge=1)

    # Option quality filters
    min_option_oi: int = Field(default=MIN_OPTION_OI, ge=0)
    max_spread_pct: float = Field(default=MAX_SPREAD_PCT, gt=0.0)
    min_option_price: float = Field(default=MIN_OPTION_PRICE, ge=0.0)
    max_option_price: float = Field(default=MAX_OPTION_PRICE, gt=0.0)

    # VIX regime
    vix_high_threshold: float = Field(default=20.0, gt=0.0)
    vix_extreme_threshold: float = Field(default=30.0, gt=0.0)

    @model_validator(mode="after")
    def _vix_thresholds_ordered(self) -> "RiskSettings":
        if self.vix_high_threshold >= self.vix_extreme_threshold:
            raise ValueError(
                "RISK__VIX_HIGH_THRESHOLD must be less than RISK__VIX_EXTREME_THRESHOLD"
            )
        return self

    @property
    def daily_loss_limit(self) -> float:
        """Absolute INR loss limit derived from capital and daily_loss_pct."""
        return self.capital * self.daily_loss_pct / 100.0


# ---------------------------------------------------------------------------
# LifecycleSettings
# ---------------------------------------------------------------------------


class LifecycleSettings(BaseSettings):
    """Trade lifecycle: take-profit ratios, trailing stop, time stop."""

    model_config = SettingsConfigDict(
        env_prefix="LIFECYCLE__",
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
    )

    tp1_r: float = Field(default=1.0, gt=0.0, description="First TP as R-multiple of initial risk")
    tp1_partial: float = Field(
        default=0.6,
        gt=0.0,
        lt=1.0,
        description="Fraction of position to close at TP1",
    )
    tp2_r_trend: float = Field(default=1.8, gt=0.0)
    tp2_r_range: float = Field(default=1.4, gt=0.0)
    trail_atr_mult: float = Field(default=0.8, gt=0.0)
    trail_activation_r: float = Field(default=1.0, gt=0.0)
    time_stop_min: int = Field(
        default=12,
        ge=1,
        description="Force-close trade if open for this many minutes without hitting TP/SL",
    )

    @model_validator(mode="after")
    def _tp2_greater_than_tp1(self) -> "LifecycleSettings":
        if self.tp2_r_trend <= self.tp1_r:
            raise ValueError("LIFECYCLE__TP2_R_TREND must be > LIFECYCLE__TP1_R")
        if self.tp2_r_range <= self.tp1_r:
            raise ValueError("LIFECYCLE__TP2_R_RANGE must be > LIFECYCLE__TP1_R")
        return self


# ---------------------------------------------------------------------------
# ExecutionSettings
# ---------------------------------------------------------------------------


class ExecutionSettings(BaseSettings):
    """Order execution parameters and slippage model."""

    model_config = SettingsConfigDict(
        env_prefix="EXECUTION__",
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
    )

    order_type: OrderType = Field(default=OrderType.MARKET)
    limit_offset_ticks: int = Field(
        default=2,
        ge=0,
        description="Ticks away from LTP when placing LIMIT orders",
    )
    slippage_model: SlippageModel = Field(default=SlippageModel.SPREAD_HALF)
    reconcile_interval_sec: int = Field(default=30, ge=5)
    max_fill_wait_sec: int = Field(default=60, ge=5)


# ---------------------------------------------------------------------------
# StreamingSettings
# ---------------------------------------------------------------------------


class StreamingSettings(BaseSettings):
    """WebSocket / tick streaming configuration."""

    model_config = SettingsConfigDict(
        env_prefix="STREAMING__",
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
    )

    websocket_disabled: bool = Field(default=False)
    polling_interval_ms: int = Field(default=1000, ge=100)
    warmup_ticks: int = Field(default=WARMUP_TICKS, ge=1)
    warmup_candles: int = Field(default=WARMUP_CANDLES, ge=1)
    max_reconnect_attempts: int = Field(default=10, ge=0)
    reconnect_backoff_base: float = Field(default=2.0, ge=1.0)


# ---------------------------------------------------------------------------
# Per-strategy settings
# ---------------------------------------------------------------------------


class StrategySettings(BaseSettings):
    """
    Per-strategy configuration block.
    Instantiated once per strategy with its own env_prefix.
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
    )

    enabled: bool = Field(default=True)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    cooldown_sec: int = Field(default=300, ge=0)
    max_daily_signals: int = Field(default=10, ge=1)


def _make_strategy_settings(prefix: str) -> StrategySettings:
    """
    Build a StrategySettings instance reading env vars prefixed with
    ``STRATEGY_{prefix}__``, e.g. ``STRATEGY_SMC_LIQUIDITY__ENABLED``.
    """

    class _StratCfg(StrategySettings):
        model_config = SettingsConfigDict(
            env_prefix=f"STRATEGY_{prefix}__",
            env_nested_delimiter="__",
            case_sensitive=False,
            frozen=True,
        )

    return _StratCfg()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TelegramSettings
# ---------------------------------------------------------------------------


class TelegramSettings(BaseSettings):
    """Telegram notification settings."""

    model_config = SettingsConfigDict(
        env_prefix="TELEGRAM__",
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
    )

    bot_token: SecretStr = Field(default=SecretStr(""))
    chat_id: str = Field(default="")
    webhook_url: str = Field(default="")
    use_polling: bool = Field(default=True)
    rate_limit_per_min: int = Field(default=20, ge=1)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TelegramSettings("
            f"bot_token=***, "
            f"chat_id={self.chat_id!r}, "
            f"use_polling={self.use_polling}"
            f")"
        )


# ---------------------------------------------------------------------------
# InfraSettings
# ---------------------------------------------------------------------------


class InfraSettings(BaseSettings):
    """Infrastructure: ports, DB path, logging."""

    model_config = SettingsConfigDict(
        env_prefix="INFRA__",
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
    )

    metrics_port: int = Field(default=9090, ge=1024, le=65535)
    api_port: int = Field(default=8000, ge=1024, le=65535)
    db_path: str = Field(default="./data/journal.db")
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format: LogFormat = Field(default=LogFormat.JSON)
    daily_summary_time: str = Field(
        default="15:20",
        description="HH:MM in IST to send daily P&L summary",
    )

    @field_validator("daily_summary_time", mode="before")
    @classmethod
    def _validate_time_str(cls, v: Any) -> str:
        parts = str(v).split(":")
        if len(parts) != 2:
            raise ValueError("INFRA__DAILY_SUMMARY_TIME must be HH:MM")
        h, m = int(parts[0]), int(parts[1])
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError("INFRA__DAILY_SUMMARY_TIME HH:MM out of range")
        return v

    @property
    def daily_summary_time_parsed(self) -> time:
        h, m = self.daily_summary_time.split(":")
        return time(int(h), int(m))


# ---------------------------------------------------------------------------
# Master Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Root settings object.  All sub-groups are nested models; pydantic-settings
    resolves them from environment using the ``__`` delimiter.

    Example .env::

        BROKER__API_KEY=abc123
        TRADING__EXECUTION_MODE=PAPER
        RISK__CAPITAL=500000
        STRATEGY_ORB_PRO__ENABLED=true
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        case_sensitive=False,
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    broker: BrokerSettings = Field(default_factory=BrokerSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    lifecycle: LifecycleSettings = Field(default_factory=LifecycleSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    streaming: StreamingSettings = Field(default_factory=StreamingSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    infra: InfraSettings = Field(default_factory=InfraSettings)

    # Per-strategy blocks
    strategy_smc_liquidity: StrategySettings = Field(
        default_factory=lambda: _make_strategy_settings("SMC_LIQUIDITY")
    )
    strategy_vwap_pro: StrategySettings = Field(
        default_factory=lambda: _make_strategy_settings("VWAP_PRO")
    )
    strategy_rsi_divergence: StrategySettings = Field(
        default_factory=lambda: _make_strategy_settings("RSI_DIVERGENCE")
    )
    strategy_gamma_scalping: StrategySettings = Field(
        default_factory=lambda: _make_strategy_settings("GAMMA_SCALPING")
    )
    strategy_oi_max_pain: StrategySettings = Field(
        default_factory=lambda: _make_strategy_settings("OI_MAX_PAIN")
    )
    strategy_orb_pro: StrategySettings = Field(
        default_factory=lambda: _make_strategy_settings("ORB_PRO")
    )
    strategy_bb_squeeze: StrategySettings = Field(
        default_factory=lambda: _make_strategy_settings("BB_SQUEEZE")
    )
    strategy_cpr_breakout: StrategySettings = Field(
        default_factory=lambda: _make_strategy_settings("CPR_BREAKOUT")
    )
    strategy_order_flow: StrategySettings = Field(
        default_factory=lambda: _make_strategy_settings("ORDER_FLOW")
    )
    strategy_straddle_theta: StrategySettings = Field(
        default_factory=lambda: _make_strategy_settings("STRADDLE_THETA")
    )
    strategy_trend_momentum: StrategySettings = Field(
        default_factory=lambda: _make_strategy_settings("TREND_MOMENTUM")
    )
    strategy_tuesday_gamma: StrategySettings = Field(
        default_factory=lambda: _make_strategy_settings("TUESDAY_GAMMA")
    )

    def get_strategy(self, name: str) -> StrategySettings:
        """
        Look up a strategy settings block by canonical strategy name string
        (e.g. ``"ORB_PRO"``).  Raises ``KeyError`` for unknown names.
        """
        mapping: dict[str, StrategySettings] = {
            "SMC_LIQUIDITY": self.strategy_smc_liquidity,
            "VWAP_PRO": self.strategy_vwap_pro,
            "RSI_DIVERGENCE": self.strategy_rsi_divergence,
            "GAMMA_SCALPING": self.strategy_gamma_scalping,
            "OI_MAX_PAIN": self.strategy_oi_max_pain,
            "ORB_PRO": self.strategy_orb_pro,
            "BB_SQUEEZE": self.strategy_bb_squeeze,
            "CPR_BREAKOUT": self.strategy_cpr_breakout,
            "ORDER_FLOW": self.strategy_order_flow,
            "STRADDLE_THETA": self.strategy_straddle_theta,
            "TREND_MOMENTUM": self.strategy_trend_momentum,
            "TUESDAY_GAMMA": self.strategy_tuesday_gamma,
        }
        try:
            return mapping[name.upper()]
        except KeyError:
            raise KeyError(f"Unknown strategy name: {name!r}. Valid: {list(mapping)}")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Settings("
            f"execution_mode={self.trading.execution_mode.value}, "
            f"instrument={self.trading.instrument}, "
            f"capital={self.risk.capital:,.0f}, "
            f"log_level={self.infra.log_level.value}"
            f")"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance (cached after first call).

    In tests override by calling ``get_settings.cache_clear()`` then
    patching env vars before the next call.
    """
    return Settings()
