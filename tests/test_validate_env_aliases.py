from __future__ import annotations

import importlib
import os
from unittest import mock

from src.config import AppSettings


def _reload_validate_env():
    import src.boot.validate_env as validate_env

    return importlib.reload(validate_env)


def test_runtime_env_accepts_legacy_zerodha_env_names(monkeypatch):
    env = {
        "ZERODHA_API_KEY": "key",
        "ZERODHA_API_SECRET": "secret",
        "ZERODHA_ACCESS_TOKEN": "token",
        "ENABLE_LIVE_TRADING": "true",
        "TELEGRAM__BOT_TOKEN": "bot",
        "TELEGRAM__CHAT_ID": "123",
        "INSTRUMENTS_CSV": "data/instruments_sample.csv",
    }
    with mock.patch.dict(os.environ, env, clear=True):
        validate_env = _reload_validate_env()

        class DummyKite:
            def __init__(self, api_key):
                pass

            def set_access_token(self, token):
                pass

        monkeypatch.setattr(validate_env, "KiteConnect", DummyKite)
        validate_env.validate_runtime_env(AppSettings(_env_file=None))


def test_runtime_env_accepts_nested_instruments_csv(tmp_path):
    csv_path = tmp_path / "instruments.csv"
    csv_path.write_text("token\n", encoding="utf-8")
    env = {
        "TELEGRAM__ENABLED": "false",
        "ENABLE_LIVE_TRADING": "false",
        "INSTRUMENTS__CSV": str(csv_path),
    }
    with mock.patch.dict(os.environ, env, clear=True):
        validate_env = _reload_validate_env()
        settings = AppSettings(_env_file=None)
        assert settings.INSTRUMENTS_CSV == str(csv_path)
        validate_env.validate_runtime_env(settings)
