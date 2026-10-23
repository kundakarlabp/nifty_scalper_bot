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


@pytest.mark.parametrize("previous_limit", ["2.0", "7.0"])
def test_live_mode_restores_canonical_five_percent_risk_envelope(
    monkeypatch, previous_limit
) -> None:
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("RISK__PER_TRADE_RISK_PCT", previous_limit)
    monkeypatch.setenv("RISK_PER_TRADE_PCT", previous_limit)
    monkeypatch.setenv("RISK_DAILY_LOSS_PCT", previous_limit)
    monkeypatch.setenv("RISK_DAILY_PNL_CAP_PCT", previous_limit)
    monkeypatch.setenv("RISK_MAX_DAILY_LOSS_PCT", previous_limit)
    monkeypatch.setenv("DAILY_PNL_CAP_PCT", previous_limit)

    normalise_live_env_defaults()

    assert LIVE_PER_TRADE_RISK_PCT == "5.0"
    assert LIVE_DAILY_LOSS_PCT == "5.0"
    assert os.environ["RISK__PER_TRADE_RISK_PCT"] == "5.0"
    assert os.environ["RISK_PER_TRADE_PCT"] == "5.0"
    assert os.environ["RISK_DAILY_LOSS_PCT"] == "5.0"
    assert os.environ["RISK_DAILY_PNL_CAP_PCT"] == "5.0"
    assert os.environ["RISK_MAX_DAILY_LOSS_PCT"] == "5.0"
    assert os.environ["DAILY_PNL_CAP_PCT"] == "5.0"

    settings = _build_risk_settings()
    assert settings.per_trade_risk_pct == pytest.approx(5.0)
    assert settings.daily_loss_pct == pytest.approx(5.0)
    assert settings.daily_pnl_cap_pct == pytest.approx(5.0)


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


def _track_admission_test_environment(monkeypatch) -> None:
    """Restore env mutated directly by normalisation after each new test."""
    for key in (
        "SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED",
        "VWAP_TREND_RETEST_EXTENSION_ENABLED",
        "ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED",
        "ENABLE_LIVE_TRADING",
        "ORDERS__ENABLE_LIVE",
        "PAPER__ENABLED",
        "PAPER_MODE",
        "SHADOW_MODE",
        "RISK__PER_TRADE_RISK_PCT",
        "RISK_PER_TRADE_PCT",
        "RISK_DAILY_LOSS_PCT",
        "RISK_DAILY_PNL_CAP_PCT",
        "RISK_MAX_DAILY_LOSS_PCT",
        "DAILY_PNL_CAP_PCT",
    ):
        monkeypatch.setenv(key, os.getenv(key, ""))


def test_lightsail_live_smc_confirmation_default_and_explicit_rollback(
    monkeypatch, caplog
) -> None:
    """Only production LIVE inherits admission; explicit false survives reinit."""
    _track_admission_test_environment(monkeypatch)
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("PRODUCTION_LIVE_DEFAULT_INITIALIZED", "true")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.delenv("SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED", raising=False)

    normalise_live_env_defaults()
    assert os.environ["SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED"] == "true"
    assert "SMC_SUSTAINED_FLOW_ADMISSION_CONFIG enabled=True" in caplog.text

    monkeypatch.setenv("SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED", "false")
    normalise_live_env_defaults()
    assert os.environ["SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED"] == "false"
    assert "SMC_SUSTAINED_FLOW_ADMISSION_CONFIG enabled=False" in caplog.text


def test_nonproduction_and_nonlive_do_not_enable_smc_live_admission(
    monkeypatch,
) -> None:
    _track_admission_test_environment(monkeypatch)
    monkeypatch.delenv("SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED", raising=False)
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "railway")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    normalise_live_env_defaults()
    assert os.getenv("SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED") is None

    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("PRODUCTION_LIVE_DEFAULT_INITIALIZED", "true")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("EXECUTION_MODE", "PAPER")
    normalise_live_env_defaults()
    assert os.getenv("SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED") is None


def test_lightsail_live_vwap_and_ofi_default_on_with_independent_rollback(
    monkeypatch, caplog
) -> None:
    """The two existing bounded paths activate only in actual Lightsail LIVE."""
    _track_admission_test_environment(monkeypatch)
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("PRODUCTION_LIVE_DEFAULT_INITIALIZED", "true")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.delenv("VWAP_TREND_RETEST_EXTENSION_ENABLED", raising=False)
    monkeypatch.delenv("ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED", raising=False)

    normalise_live_env_defaults()
    assert os.environ["VWAP_TREND_RETEST_EXTENSION_ENABLED"] == "true"
    assert os.environ["ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED"] == "true"
    assert "LIVE_ENTRY_QUALITY_CONFIG vwap_retest=True adaptive_ofi=True" in caplog.text

    monkeypatch.setenv("VWAP_TREND_RETEST_EXTENSION_ENABLED", "false")
    normalise_live_env_defaults()
    assert os.environ["VWAP_TREND_RETEST_EXTENSION_ENABLED"] == "false"
    assert os.environ["ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED"] == "true"
    assert "LIVE_ENTRY_QUALITY_CONFIG vwap_retest=False adaptive_ofi=True" in caplog.text

    monkeypatch.setenv("ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED", "false")
    normalise_live_env_defaults()
    assert os.environ["VWAP_TREND_RETEST_EXTENSION_ENABLED"] == "false"
    assert os.environ["ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED"] == "false"
    assert "LIVE_ENTRY_QUALITY_CONFIG vwap_retest=False adaptive_ofi=False" in caplog.text


def test_lightsail_paper_and_nonproduction_live_do_not_implicitly_activate_both(
    monkeypatch,
) -> None:
    _track_admission_test_environment(monkeypatch)
    monkeypatch.delenv("VWAP_TREND_RETEST_EXTENSION_ENABLED", raising=False)
    monkeypatch.delenv("ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED", raising=False)
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "railway")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    normalise_live_env_defaults()
    assert os.getenv("VWAP_TREND_RETEST_EXTENSION_ENABLED") is None
    assert os.getenv("ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED") is None

    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("PRODUCTION_LIVE_DEFAULT_INITIALIZED", "true")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("EXECUTION_MODE", "PAPER")
    normalise_live_env_defaults()
    assert os.getenv("VWAP_TREND_RETEST_EXTENSION_ENABLED") is None
    assert os.getenv("ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED") is None
