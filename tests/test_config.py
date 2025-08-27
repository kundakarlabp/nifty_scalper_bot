# tests/test_config.py

"""
Tests for the Pydantic configuration system.
"""

import os
from unittest import mock

import pytest
from pydantic import ValidationError

import src.config as config
from src.config import AppSettings, RiskSettings, validate_critical_settings, TelegramSettings, ZerodhaSettings

def test_load_from_env():
    """Tests that settings are correctly loaded from environment variables."""
    test_env = {
        "ZERODHA__API_KEY": "test_key",
        "ZERODHA__API_SECRET": "test_secret",
        "ZERODHA__ACCESS_TOKEN": "test_token",
        "TELEGRAM__BOT_TOKEN": "test_bot_token",
        "TELEGRAM__CHAT_ID": "12345",
        "RISK__MAX_DAILY_DRAWDOWN_PCT": "0.1",
        "RISK__RISK_PER_TRADE": "0.02",
        "ENABLE_LIVE_TRADING": "true",
    }
    with mock.patch.dict(os.environ, test_env, clear=True):
        settings = AppSettings(_env_file=None)
        assert settings.zerodha.api_key == "test_key"
        assert settings.telegram.chat_id == 12345
        assert settings.risk.max_daily_drawdown_pct == 0.1
        assert settings.risk.risk_per_trade == 0.02
        assert settings.enable_live_trading is True

def test_validation_error_on_invalid_data():
    """Tests that Pydantic raises a ValidationError for out-of-bounds data."""
    with pytest.raises(ValidationError):
        # Direct instantiation to trigger validation errors
        RiskSettings(max_daily_drawdown_pct=2.0)

def test_default_values():
    """Tests that default values are used when environment variables are not set."""
    # Ensure these are not in the environment
    test_env = {
        "ZERODHA__API_KEY": "test_key",
        "ZERODHA__API_SECRET": "test_secret",
        "ZERODHA__ACCESS_TOKEN": "test_token",
        "TELEGRAM__BOT_TOKEN": "test_bot_token",
        "TELEGRAM__CHAT_ID": "12345",
    }
    with mock.patch.dict(os.environ, test_env, clear=True):
        settings = AppSettings(_env_file=None)
        assert settings.risk.max_daily_drawdown_pct == 0.04  # Default value
        assert settings.log_level == "INFO"  # Default value
        assert settings.enable_live_trading is False  # Default value
        assert settings.system.log_buffer_capacity == 4000  # Default value


def test_validate_critical_settings_live_requires_zerodha():
    """Live trading should demand Zerodha credentials."""
    live_settings = AppSettings(
        enable_live_trading=True,
        telegram=TelegramSettings(bot_token="x", chat_id=1),
        _env_file=None,
    )
    with mock.patch.object(config, "settings", live_settings):
        with pytest.raises(ValueError):
            validate_critical_settings()


def test_validate_critical_settings_paper_mode_skips_zerodha():
    """Paper mode should not require Zerodha credentials."""
    paper_settings = AppSettings(
        enable_live_trading=False,
        telegram=TelegramSettings(bot_token="x", chat_id=1),
        zerodha=ZerodhaSettings(),
        _env_file=None,
    )
    with mock.patch.object(config, "settings", paper_settings):
        validate_critical_settings()
