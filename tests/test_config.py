"""Lightweight tests for configuration-like Pydantic models."""

import os
from pydantic import BaseModel, ValidationError, field_validator
from pydantic_settings import BaseSettings
import pytest

# Remove env file effects and unrelated variables
for _k in ("TRADING_ENV", "trading_env"):
    os.environ.pop(_k, None)


class RiskSettings(BaseModel):
    max_daily_drawdown_pct: float = 0.04
    risk_per_trade: float = 0.01

    @field_validator("max_daily_drawdown_pct")
    @classmethod
    def _max_dd(cls, v: float) -> float:
        if not 0.0 <= v < 0.2:
            raise ValueError("max_daily_drawdown_pct must be < 0.2")
        return v


class AppSettings(BaseSettings):
    enable_live_trading: bool = True
    log_level: str = "INFO"
    zerodha_api_key: str = ""
    zerodha_api_secret: str = ""
    zerodha_access_token: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: int = 0
    risk: RiskSettings = RiskSettings()

    class Config:
        env_nested_delimiter = "__"
        extra = "forbid"


def test_load_from_env():
    env = {
        "ZERODHA_API_KEY": "test_key",
        "ZERODHA_API_SECRET": "test_secret",
        "ZERODHA_ACCESS_TOKEN": "test_token",
        "TELEGRAM_BOT_TOKEN": "test_bot_token",
        "TELEGRAM_CHAT_ID": "12345",
        "RISK__MAX_DAILY_DRAWDOWN_PCT": "0.1",
        "RISK__RISK_PER_TRADE": "0.02",
        "ENABLE_LIVE_TRADING": "true",
    }
    with pytest.MonkeyPatch().context() as mp:
        for k, v in env.items():
            mp.setenv(k, v)
        settings = AppSettings(_env_file=None)
    assert settings.zerodha_api_key == "test_key"
    assert settings.telegram_chat_id == 12345
    assert settings.risk.max_daily_drawdown_pct == 0.1
    assert settings.risk.risk_per_trade == 0.02
    assert settings.enable_live_trading is True


def test_validation_error_on_invalid_data():
    with pytest.raises(ValidationError):
        RiskSettings(max_daily_drawdown_pct=2.0)


def test_default_values():
    env = {
        "ZERODHA_API_KEY": "test_key",
        "ZERODHA_API_SECRET": "test_secret",
        "ZERODHA_ACCESS_TOKEN": "test_token",
        "TELEGRAM_BOT_TOKEN": "test_bot_token",
        "TELEGRAM_CHAT_ID": "12345",
    }
    with pytest.MonkeyPatch().context() as mp:
        for k, v in env.items():
            mp.setenv(k, v)
        settings = AppSettings(_env_file=None)
    assert settings.risk.max_daily_drawdown_pct == 0.04
    assert settings.log_level == "INFO"
    assert settings.enable_live_trading is True
