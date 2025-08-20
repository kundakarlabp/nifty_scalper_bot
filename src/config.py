# src/config.py
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, AliasChoices, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────── leaf configs ─────────────────────────── #

class ServerConfig(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)


class TelegramConfig(BaseModel):
    enabled: bool = Field(default=True)
    # Tests expect chat_id as *string*; controller converts to int safely.
    bot_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("TELEGRAM_BOT_TOKEN", "telegram_bot_token"),
    )
    chat_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("TELEGRAM_CHAT_ID", "telegram_chat_id"),
    )


class ZerodhaConfig(BaseModel):
    api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ZERODHA_API_KEY", "KITE_API_KEY"),
    )
    api_secret: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ZERODHA_API_SECRET", "KITE_API_SECRET"),
    )
    access_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ZERODHA_ACCESS_TOKEN", "KITE_ACCESS_TOKEN", "ACCESS_TOKEN"),
    )


class RiskConfig(BaseSettings):
    """
    Standalone env-driven risk config (tests instantiate this directly).
    Env:
      - MAX_DAILY_DRAWDOWN_PCT  (default 0.03)
      - RISK_PER_TRADE_PCT      (default 0.01)
      - DEFAULT_EQUITY          (default 30000)
    """
    model_config = SettingsConfigDict(env_prefix="", extra="ignore", case_sensitive=False)

    max_daily_drawdown_pct: float = Field(
        default=0.03,
        validation_alias=AliasChoices("MAX_DAILY_DRAWDOWN_PCT", "max_daily_drawdown_pct"),
    )
    risk_per_trade_pct: float = Field(
        default=0.01,
        validation_alias=AliasChoices("RISK_PER_TRADE_PCT", "risk_per_trade_pct"),
    )
    default_equity: float = Field(
        default=30000.0,
        validation_alias=AliasChoices("DEFAULT_EQUITY", "default_equity"),
    )

    @field_validator("max_daily_drawdown_pct")
    @classmethod
    def _chk_drawdown(cls, v: float) -> float:
        v = float(v)
        if not (0.0 < v < 0.5):
            raise ValueError("max_daily_drawdown_pct must be between 0 and 0.5")
        return v

    @field_validator("risk_per_trade_pct")
    @classmethod
    def _chk_risk(cls, v: float) -> float:
        v = float(v)
        if not (0.0 < v <= 0.2):
            raise ValueError("risk_per_trade_pct must be between 0 and 0.2")
        return v


# ─────────────────────────── root settings ─────────────────────────── #

class AppSettings(BaseSettings):
    """
    Centralized, env-first settings. Always import/use the singleton
    `settings` from this module rather than importing sub-config classes.
    """
    model_config = SettingsConfigDict(
        env_prefix="",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # Feature flags / mode
    enable_live_trading: bool = Field(
        default=False,
        validation_alias=AliasChoices("ENABLE_LIVE_TRADING", "LIVE_TRADING", "enable_live_trading"),
    )

    # Optional logging level
    log_level: str = Field(default="INFO")

    # Sub-configs
    server: ServerConfig = Field(default_factory=ServerConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    zerodha: ZerodhaConfig = Field(default_factory=ZerodhaConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)

    # Back-compat flat aliases (some legacy code paths still read these)
    TELEGRAM_BOT_TOKEN: Optional[str] = Field(default=None)
    TELEGRAM_CHAT_ID: Optional[str] = Field(default=None)
    ZERODHA_API_KEY: Optional[str] = Field(default=None)
    ZERODHA_API_SECRET: Optional[str] = Field(default=None)
    ZERODHA_ACCESS_TOKEN: Optional[str] = Field(default=None)

    # Derived/compat
    @property
    def preferred_exit_mode(self) -> str:
        return "AUTO"

    class _ApiShim(BaseModel):
        zerodha_api_key: Optional[str] = None
        zerodha_api_secret: Optional[str] = None
        zerodha_access_token: Optional[str] = None

    @property
    def api(self) -> "AppSettings._ApiShim":
        return AppSettings._ApiShim(
            zerodha_api_key=self.zerodha.api_key or self.ZERODHA_API_KEY,
            zerodha_api_secret=self.zerodha.api_secret or self.ZERODHA_API_SECRET,
            zerodha_access_token=self.zerodha.access_token or self.ZERODHA_ACCESS_TOKEN,
        )


# Singleton settings instance — the *only* thing other modules should import
settings = AppSettings()
