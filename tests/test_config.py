"""
Tests for the Pydantic configuration system.
"""

import os
from unittest import mock
import pytest
from pydantic import ValidationError

from src.config import AppSettings, RiskConfig

def test_load_from_env():
    """Tests that settings are correctly loaded from environment variables."""
    test_env = {
        "ZERODHA_API_KEY": "test_key",
        "ZERODHA_API_SECRET": "test_secret",
        "ZERODHA_ACCESS_TOKEN": "test_token",
        "TELEGRAM_BOT_TOKEN": "test_bot_token",
        "TELEGRAM_CHAT_ID": "12345",
        "MAX_DAILY_DRAWDOWN_PCT": "0.1",
        "RISK_PER_TRADE_PCT": "0.02",
        "ENABLE_LIVE_TRADING": "true",
    }
    with mock.patch.dict(os.environ, test_env):
        settings = AppSettings()
        assert settings.api.zerodha_api_key == "test_key"
        assert settings.telegram.chat_id == "12345"
        assert settings.risk.max_daily_drawdown_pct == 0.1
        assert settings.risk.risk_per_trade_pct == 0.02
        assert settings.enable_live_trading is True

def test_validation_error_on_invalid_data():
    """Tests that Pydantic raises a ValidationError for out-of-bounds data."""
    test_env = {
        "MAX_DAILY_DRAWDOWN_PCT": "2.0",  # Invalid (must be < 0.5)
    }
    with mock.patch.dict(os.environ, test_env):
        # The validation happens at the level of the specific model
        with pytest.raises(ValidationError):
            RiskConfig()

def test_default_values():
    """Tests that default values are used when environment variables are not set."""
    # Ensure these are not in the environment
    test_env = {
        "ZERODHA_API_KEY": "test_key",
        "ZERODHA_API_SECRET": "test_secret",
        "ZERODHA_ACCESS_TOKEN": "test_token",
        "TELEGRAM_BOT_TOKEN": "test_bot_token",
        "TELEGRAM_CHAT_ID": "12345",
    }
    with mock.patch.dict(os.environ, test_env, clear=True):
        settings = AppSettings()
        assert settings.risk.max_daily_drawdown_pct == 0.03 # Default value
        assert settings.log_level == "INFO" # Default value
        assert settings.enable_live_trading is False # Default value
