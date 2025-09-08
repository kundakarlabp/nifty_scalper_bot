# tests/test_config.py

"""
Tests for the Pydantic configuration system.
"""

import logging
import os
from unittest import mock

import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import AppSettings, DataSettings, RiskSettings
from src.boot.validate_env import validate_critical_settings

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
    with mock.patch.dict(os.environ, test_env):
        settings = AppSettings()
        assert settings.zerodha.api_key == "test_key"
        assert settings.telegram.chat_id == 12345
        assert settings.risk.max_daily_drawdown_pct == 0.1
        assert settings.risk.risk_per_trade == 0.02
        assert settings.enable_live_trading is True

def test_validation_error_on_invalid_data():
    """Tests that Pydantic raises a ValidationError for out-of-bounds data."""
    with pytest.raises(ValidationError):
        # Direct instantiation to trigger validation errors
        RiskSettings(max_daily_drawdown_pct=2.0)  # Invalid (must be < 0.2)

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
        assert settings.enable_live_trading is True
        assert settings.system.log_buffer_capacity == 4000  # Default value
        assert settings.data.lookback_minutes == 20  # Updated default
        assert settings.data.lookback_padding_bars == 5  # Default padding
        assert settings.strategy.min_bars_for_signal == 20  # Updated default
        assert settings.strategy.rr_threshold == 1.5  # Default risk-reward threshold


def test_telegram_disabled_without_creds():
    """Telegram should auto-disable when credentials are missing."""
    env = {
        "ZERODHA__API_KEY": "k",
        "ZERODHA__API_SECRET": "s",
        "ZERODHA__ACCESS_TOKEN": "t",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = AppSettings(_env_file=None)
    assert settings.telegram.enabled is False


def test_legacy_env_names_supported():
    """Old flat env var names should still be recognized."""
    env = {
        "KITE_API_KEY": "old_key",
        "KITE_API_SECRET": "old_secret",
        "KITE_ACCESS_TOKEN": "old_token",
        "TELEGRAM_BOT_TOKEN": "old_bot",
        "TELEGRAM_CHAT_ID": "12345",
        "ENABLE_LIVE_TRADING": "false",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = AppSettings(_env_file=None)
        assert settings.zerodha.api_key == "old_key"
        assert settings.zerodha.api_secret == "old_secret"
        assert settings.zerodha.access_token == "old_token"
        assert settings.telegram.bot_token == "old_bot"
        assert settings.telegram.chat_id == 12345


def test_warmup_bars_env():
    env = {
        "TELEGRAM__ENABLED": "false",
        "ENABLE_LIVE_TRADING": "false",
        "WARMUP_BARS": "40",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = AppSettings(_env_file=None)
    assert settings.warmup_bars == 40


def test_historical_timeframe_alias():
    env = {
        "TELEGRAM__ENABLED": "false",
        "ENABLE_LIVE_TRADING": "false",
        "HISTORICAL_TIMEFRAME": "5minute",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = AppSettings(_env_file=None)
    assert settings.data.timeframe == "5minute"


def test_enable_trading_alias():
    env = {
        "ENABLE_TRADING": "true",
        "ZERODHA__API_KEY": "k",
        "ZERODHA__API_SECRET": "s",
        "ZERODHA__ACCESS_TOKEN": "t",
        "TELEGRAM__BOT_TOKEN": "b",
        "TELEGRAM__CHAT_ID": "1",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = AppSettings(_env_file=None)
    assert settings.enable_live_trading is True


def test_skip_broker_validation_allows_missing_creds(caplog):
    env = {
        "ENABLE_LIVE_TRADING": "true",
        "SKIP_BROKER_VALIDATION": "true",
        "TELEGRAM__BOT_TOKEN": "b",
        "TELEGRAM__CHAT_ID": "1",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        import importlib
        import src.boot.validate_env as validate_env

        validate_env = importlib.reload(validate_env)
        settings = AppSettings(_env_file=None)
        with mock.patch("src.config.settings", settings), mock.patch(
            "src.boot.validate_env.settings", settings
        ), caplog.at_level(logging.WARNING):
            validate_env.validate_critical_settings()
    import importlib
    import src.boot.validate_env as validate_env
    importlib.reload(validate_env)
    assert any("SKIP_BROKER_VALIDATION" in r.message for r in caplog.records)


def test_zerodha_creds_required_when_live():
    """Zerodha credentials must be present when live trading is enabled."""
    env = {
        "ENABLE_LIVE_TRADING": "true",
        "TELEGRAM__BOT_TOKEN": "bot",
        "TELEGRAM__CHAT_ID": "12345",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = AppSettings(_env_file=None)
    with mock.patch("src.config.settings", settings), mock.patch(
        "src.boot.validate_env.settings", settings
    ):
        with pytest.raises(ValueError) as exc:
            validate_critical_settings()
    msg = str(exc.value)
    assert "ZERODHA__API_KEY" in msg and "KITE_API_KEY" in msg


def test_zerodha_creds_optional_when_paper():
    """Zerodha credentials are not required in paper trading mode."""
    env = {
        "ENABLE_LIVE_TRADING": "false",
        "TELEGRAM__BOT_TOKEN": "bot",
        "TELEGRAM__CHAT_ID": "12345",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = AppSettings(_env_file=None)
    with mock.patch("src.config.settings", settings), mock.patch(
        "src.boot.validate_env.settings", settings
    ):
        # Should not raise
        validate_critical_settings()


def test_negative_chat_id_allowed():
    """Telegram chat IDs can be negative for group chats."""
    env = {
        "TELEGRAM__BOT_TOKEN": "bot",
        "TELEGRAM__CHAT_ID": "-12345",
        "ENABLE_LIVE_TRADING": "false",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = AppSettings(_env_file=None)
    with mock.patch("src.config.settings", settings), mock.patch(
        "src.boot.validate_env.settings", settings
    ):
        # Should not raise for negative IDs
        validate_critical_settings()
        assert settings.telegram.chat_id == -12345


def test_invalid_instrument_token_detected(monkeypatch):
    env = {
        "ENABLE_LIVE_TRADING": "true",
        "ZERODHA__API_KEY": "k",
        "ZERODHA__API_SECRET": "s",
        "ZERODHA__ACCESS_TOKEN": "t",
        "TELEGRAM__BOT_TOKEN": "b",
        "TELEGRAM__CHAT_ID": "123",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = AppSettings(_env_file=None)

    class DummySource:
        def __init__(self, kite):
            pass

        def connect(self) -> None:
            pass

        def fetch_ohlc(self, *_, **__):
            return pd.DataFrame()

    with mock.patch("src.config.settings", settings), mock.patch(
        "src.boot.validate_env.settings", settings
    ), monkeypatch.context() as m:
        m.setattr("src.boot.validate_env.LiveKiteSource", DummySource)
        with pytest.raises(ValueError) as exc:
            validate_critical_settings()
    assert "valid F&O token" in str(exc.value)


def test_decimal_instrument_token_coerced():
    """Decimal strings for instrument tokens are coerced to integers."""
    env = {
        "TELEGRAM__BOT_TOKEN": "bot",
        "TELEGRAM__CHAT_ID": "12345",
        "INSTRUMENTS__INSTRUMENT_TOKEN": "33712.5",
        "ENABLE_LIVE_TRADING": "false",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = AppSettings(_env_file=None)
    assert settings.instruments.instrument_token == 33712


def test_valid_token_with_no_candles_falls_back_to_ltp(monkeypatch):
    """A valid token with empty OHLC data should pass via LTP fallback."""

    env = {
        "ENABLE_LIVE_TRADING": "true",
        "ZERODHA__API_KEY": "k",
        "ZERODHA__API_SECRET": "s",
        "ZERODHA__ACCESS_TOKEN": "t",
        "TELEGRAM__BOT_TOKEN": "b",
        "TELEGRAM__CHAT_ID": "123",
        "INSTRUMENTS__INSTRUMENT_TOKEN": "111",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = AppSettings(_env_file=None)

    class DummySource:
        def __init__(self, kite):
            pass

        def connect(self) -> None:
            pass

        def fetch_ohlc(self, *_, **__):
            return pd.DataFrame()

        def get_last_price(self, token):
            return 1.0

    class DummyKite:
        def __init__(self, api_key):
            pass

        def set_access_token(self, token):
            pass

    with mock.patch("src.config.settings", settings), mock.patch(
        "src.boot.validate_env.settings", settings
    ), monkeypatch.context() as m:
        m.setattr("src.boot.validate_env.LiveKiteSource", DummySource)
        m.setattr("src.boot.validate_env.KiteConnect", DummyKite)
        # Should not raise
        validate_critical_settings()


def test_lookback_less_than_min_bars():
    env = {
        "TELEGRAM__BOT_TOKEN": "bot",
        "TELEGRAM__CHAT_ID": "12345",
        "DATA__LOOKBACK_MINUTES": "20",
        "STRATEGY__MIN_BARS_FOR_SIGNAL": "50",
        "ENABLE_LIVE_TRADING": "false",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = AppSettings(_env_file=None)
    with mock.patch("src.config.settings", settings), mock.patch(
        "src.boot.validate_env.settings", settings
    ):
        with pytest.raises(ValueError) as exc:
            validate_critical_settings()
    assert "LOOKBACK_MINUTES" in str(exc.value)


def test_instrument_token_validation_skips_on_network_error(monkeypatch):
    env = {
        "ENABLE_LIVE_TRADING": "true",
        "ZERODHA__API_KEY": "k",
        "ZERODHA__API_SECRET": "s",
        "ZERODHA__ACCESS_TOKEN": "t",
        "TELEGRAM__BOT_TOKEN": "b",
        "TELEGRAM__CHAT_ID": "123",
        "INSTRUMENTS__INSTRUMENT_TOKEN": "111",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        settings = AppSettings(_env_file=None)

    class DummySource:
        def __init__(self, kite):
            pass

        def connect(self) -> None:
            pass

        def fetch_ohlc(self, *_, **__):
            raise RuntimeError("network down")

    class DummyKite:
        def __init__(self, api_key):
            pass

        def set_access_token(self, token):
            pass

    with mock.patch("src.config.settings", settings), mock.patch(
        "src.boot.validate_env.settings", settings
    ), monkeypatch.context() as m:
        m.setattr("src.boot.validate_env.LiveKiteSource", DummySource)
        m.setattr("src.boot.validate_env.KiteConnect", DummyKite)
        # Should not raise even though fetch_ohlc errors
        validate_critical_settings()


def test_invalid_timeframe_defaults_to_minute(caplog):
    """Unsupported timeframe values should fall back to 'minute' with a warning."""
    with caplog.at_level(logging.WARNING, logger="config"):
        ds = DataSettings(timeframe="hourly")
    assert ds.timeframe == "minute"
    assert any("Unsupported timeframe" in r.message for r in caplog.records)
