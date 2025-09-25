"""Tests covering Telegram settings parsing."""

from __future__ import annotations

import os

import pytest

from src.config import TelegramSettings


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = [
        "TELEGRAM__ENABLED",
        "TELEGRAM_ENABLED",
        "TELEGRAM__BOT_TOKEN",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM__CHAT_ID",
        "TELEGRAM_CHAT_ID",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_telegram_disabled_when_blank_env(caplog: pytest.LogCaptureFixture) -> None:
    caplog.clear()
    settings = TelegramSettings.from_env()
    assert settings.enabled is False
    assert settings.chat_id == 0

    os.environ["TELEGRAM__ENABLED"] = "true"
    settings = TelegramSettings.from_env()
    assert settings.enabled is False
    assert "Telegram credentials missing" in caplog.text


def test_telegram_enabled_with_credentials() -> None:
    os.environ["TELEGRAM__ENABLED"] = "true"
    os.environ["TELEGRAM__BOT_TOKEN"] = "token"
    os.environ["TELEGRAM__CHAT_ID"] = "42"

    settings = TelegramSettings.from_env()
    assert settings.enabled is True
    assert settings.chat_id == 42
    assert settings.bot_token == "token"
