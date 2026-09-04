from __future__ import annotations

import os

import pytest

from nifty_scalper_bot.config.env_utils import (
    LIVE_DAILY_LOSS_PCT,
    LIVE_PER_TRADE_RISK_PCT,
    normalise_live_env_defaults,
)
from nifty_scalper_bot.config.settings import _build_risk_settings


@pytest.fixture(autouse=True)
def _not_production_migration(monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_PLATFORM", raising=False)
    monkeypatch.delenv("PRODUCTION_LIVE_DEFAULT_INITIALIZED", raising=False)
    monkeypatch.delenv("PRODUCTION_DEFAULT_LIVE", raising=False)


def test_live_mode_restores_canonical_conservative_risk_envelope(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("RISK__PER_TRADE_RISK_PCT", "7.0")
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "7.0")
    monkeypatch.setenv("RISK_DAILY_LOSS_PCT", "7.0")
    monkeypatch.setenv("RISK_DAILY_PNL_CAP_PCT", "7.0")
    monkeypatch.setenv("RISK_MAX_DAILY_LOSS_PCT", "7.0")
    monkeypatch.setenv("DAILY_PNL_CAP_PCT", "7.0")

    normalise_live_env_defaults()

    assert LIVE_PER_TRADE_RISK_PCT == "0.75"
    assert LIVE_DAILY_LOSS_PCT == "2.0"
    assert os.environ["RISK__PER_TRADE_RISK_PCT"] == "0.75"
    assert os.environ["RISK_PER_TRADE_PCT"] == "0.75"
    assert os.environ["RISK_DAILY_LOSS_PCT"] == "2.0"
    assert os.environ["RISK_DAILY_PNL_CAP_PCT"] == "2.0"
    assert os.environ["RISK_MAX_DAILY_LOSS_PCT"] == "2.0"
    assert os.environ["DAILY_PNL_CAP_PCT"] == "2.0"

    settings = _build_risk_settings()
    assert settings.per_trade_risk_pct == pytest.approx(0.75)
    assert settings.daily_loss_pct == pytest.approx(2.0)
    assert settings.daily_pnl_cap_pct == pytest.approx(2.0)


def test_non_live_mode_does_not_relax_explicit_risk_limits(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    monkeypatch.setenv("EXECUTION_MODE", "PAPER")
    monkeypatch.setenv("RISK__PER_TRADE_RISK_PCT", "1.0")
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "1.0")
    monkeypatch.setenv("RISK_DAILY_LOSS_PCT", "2.0")
    monkeypatch.setenv("RISK_DAILY_PNL_CAP_PCT", "2.0")

    normalise_live_env_defaults()

    assert os.environ["RISK__PER_TRADE_RISK_PCT"] == "1.0"
    assert os.environ["RISK_PER_TRADE_PCT"] == "1.0"
    assert os.environ["RISK_DAILY_LOSS_PCT"] == "2.0"
    assert os.environ["RISK_DAILY_PNL_CAP_PCT"] == "2.0"
