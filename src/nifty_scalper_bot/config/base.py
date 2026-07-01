"""Typed configuration models for the Nifty scalper bot."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class BrokerConfig(BaseModel):
    """Broker credentials and endpoint configuration."""

    model_config = {"extra": "forbid"}

    api_key: str = "DUMMY_API_KEY"
    api_secret: str = "DUMMY_API_SECRET"
    access_token: str | None = None
    base_url: HttpUrl = "https://api.kite.trade"
    websocket_url: str = "wss://ws.kite.trade"

    @field_validator("websocket_url")
    @classmethod
    def _validate_websocket_url(cls, value: str) -> str:
        if not value.startswith(("ws://", "wss://")):
            raise ValueError("WebSocket URL must start with ws:// or wss://")
        return value


class RiskConfig(BaseModel):
    """Risk controls enforced before submitting orders."""

    model_config = {"extra": "forbid"}

    max_daily_trades: int = Field(default=20, ge=1)
    max_order_notional: float = Field(default=200_000.0, gt=0)
    allow_short: bool = True
    max_drawdown_pct: float = Field(default=10.0, ge=0.0)
    max_daily_loss_pct: float = Field(default=5.0, ge=0.0)
    max_position_size_pct: float = Field(default=20.0, ge=0.0, le=100.0)
    max_total_exposure_pct: float = Field(default=80.0, ge=0.0, le=100.0)
    max_concurrent_positions: int = Field(default=3, ge=1)
    max_positions_per_symbol: int = Field(default=1, ge=1)
    min_risk_reward_ratio: float = Field(default=2.0, ge=1.0)


class LoggingConfig(BaseModel):
    """Application logging preferences."""

    model_config = {"extra": "forbid"}

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class TelegramConfig(BaseModel):
    """Telegram credentials and webhook configuration for single-user control."""

    model_config = {"extra": "forbid"}

    bot_token: str | None = None
    chat_id: int | None = Field(default=None)
    webhook_url: HttpUrl | None = None
    webhook_path: str = Field(default="/telegram_webhook")
    webhook_secret_token: str | None = None
    webhook_max_failures: int = Field(default=5, ge=1)
    enable_polling_fallback: bool = False
    polling_interval_seconds: float = Field(default=5.0, gt=0.0, le=60.0)
    webhook_listen_host: str = Field(default="0.0.0.0")
    webhook_listen_port: int = Field(default=8000, ge=1, le=65_535)

    @field_validator("webhook_path")
    @classmethod
    def _normalize_webhook_path(cls, value: str) -> str:
        if not value:
            raise ValueError("webhook_path must not be empty")
        if not value.startswith("/"):
            return f"/{value}"
        return value

    @model_validator(mode="after")
    def _require_pair(self) -> "TelegramConfig":
        if bool(self.bot_token) != bool(self.chat_id):
            raise ValueError(
                "Telegram configuration requires both bot_token and chat_id or neither."
            )
        return self


class RateLimitBucketConfig(BaseModel):
    """Leaky bucket configuration for a single rate limited endpoint."""

    model_config = {"extra": "forbid"}

    capacity: int = Field(default=1, ge=1)
    refill_rate_per_sec: float = Field(default=1.0, gt=0)


class RateLimitConfig(BaseModel):
    """Rate limits grouped by broker endpoint type."""

    model_config = {"extra": "forbid"}

    orders: RateLimitBucketConfig = Field(default_factory=RateLimitBucketConfig)
    rest: RateLimitBucketConfig = Field(default_factory=RateLimitBucketConfig)
    hist: RateLimitBucketConfig = Field(default_factory=RateLimitBucketConfig)


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = {"extra": "forbid"}

    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ratelimit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    quote_stale_threshold_ms: int = Field(default=1500, gt=0)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)


__all__ = [
    "AppConfig",
    "BrokerConfig",
    "LoggingConfig",
    "TelegramConfig",
    "RateLimitBucketConfig",
    "RateLimitConfig",
    "RiskConfig",
]
