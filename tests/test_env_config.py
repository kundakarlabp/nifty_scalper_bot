"""Tests for environment-driven configuration parsing."""

from __future__ import annotations

import pytest
from pydantic import HttpUrl

from nifty_scalper_bot.config import env


def test_env_uses_http_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROKER_API_KEY", "key")
    monkeypatch.setenv("BROKER_API_SECRET", "secret")
    monkeypatch.setenv("BROKER_BASE_URL", "https://api.kite.trade")

    config = env.load_from_env()
    broker = config.broker

    assert isinstance(broker.base_url, HttpUrl)
    assert str(broker.base_url) == "https://api.kite.trade"

    broker_cfg = env.build_broker_config()
    assert isinstance(broker_cfg.base_url, HttpUrl)
    assert str(broker_cfg.base_url) == "https://api.kite.trade"
