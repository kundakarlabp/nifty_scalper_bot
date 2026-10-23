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


@pytest.mark.parametrize(
    ("previous_limit", "expected"), [("2.0", "2.0"), ("7.0", "5.0")]
)
def test_live_mode_caps_risk_without_overwriting_stricter_settings(
    monkeypatch, previous_limit, expected
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
    assert os.environ["RISK__PER_TRADE_RISK_PCT"] == expected
    assert os.environ["RISK_PER_TRADE_PCT"] == expected
    assert os.environ["RISK_DAILY_LOSS_PCT"] == expected
    assert os.environ["RISK_DAILY_PNL_CAP_PCT"] == expected
    assert os.environ["RISK_MAX_DAILY_LOSS_PCT"] == expected
    assert os.environ["DAILY_PNL_CAP_PCT"] == expected

    settings = _build_risk_settings()
    assert settings.per_trade_risk_pct == pytest.approx(float(expected))
    assert settings.daily_loss_pct == pytest.approx(float(expected))
    assert settings.daily_pnl_cap_pct == pytest.approx(float(expected))


def test_live_conflicting_aliases_choose_strictest_limit(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("RISK__PER_TRADE_RISK_PCT", "4.0")
    monkeypatch.setenv("RISK_PER_TRADE_PCT", "0.75")
    monkeypatch.setenv("RISK_DAILY_LOSS_PCT", "4.5")
    monkeypatch.setenv("RISK_DAILY_PNL_CAP_PCT", "1.5")
    monkeypatch.setenv("RISK_MAX_DAILY_LOSS_PCT", "2.5")
    monkeypatch.setenv("DAILY_PNL_CAP_PCT", "6.0")

    normalise_live_env_defaults()

    assert os.environ["RISK__PER_TRADE_RISK_PCT"] == "0.75"
    assert os.environ["RISK_PER_TRADE_PCT"] == "0.75"
    assert os.environ["RISK_DAILY_LOSS_PCT"] == "1.5"
    assert os.environ["RISK_DAILY_PNL_CAP_PCT"] == "1.5"
    assert os.environ["RISK_MAX_DAILY_LOSS_PCT"] == "1.5"
    assert os.environ["DAILY_PNL_CAP_PCT"] == "1.5"
    assert _build_risk_settings().per_trade_risk_pct == pytest.approx(0.75)


@pytest.mark.parametrize("invalid", ["garbage", "-1", "0", "nan", "inf"])
def test_invalid_live_risk_limit_fails_closed_before_authorization(
    monkeypatch, invalid
) -> None:
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("RISK__PER_TRADE_RISK_PCT", invalid)

    with pytest.raises(ValueError, match="RISK__PER_TRADE_RISK_PCT"):
        normalise_live_env_defaults()
    assert os.getenv("ENABLE_LIVE_TRADING") != "true" or (
        os.getenv("ORDERS__ENABLE_LIVE") != "true"
    )




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
    """Restore values normalized directly into os.environ after tests."""
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


_ENTRY_FLAGS = (
    "SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED",
    "VWAP_TREND_RETEST_EXTENSION_ENABLED",
    "ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED",
)


@pytest.mark.parametrize("platform", ["aws_lightsail", "railway", ""])
def test_live_entry_defaults_use_execution_mode_not_host(
    monkeypatch, caplog, platform
) -> None:
    _track_admission_test_environment(monkeypatch)
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", platform)
    monkeypatch.setenv("PRODUCTION_LIVE_DEFAULT_INITIALIZED", "true")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    for key in _ENTRY_FLAGS:
        monkeypatch.delenv(key, raising=False)

    normalise_live_env_defaults()
    assert all(os.environ[key] == "true" for key in _ENTRY_FLAGS)
    assert "SMC_SUSTAINED_FLOW_ADMISSION_CONFIG enabled=True" in caplog.text
    assert "LIVE_ENTRY_QUALITY_CONFIG vwap_retest=True adaptive_ofi=True" in caplog.text


@pytest.mark.parametrize("disabled_flag", _ENTRY_FLAGS)
def test_live_entry_flags_retain_independent_false_override(
    monkeypatch, disabled_flag
) -> None:
    _track_admission_test_environment(monkeypatch)
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "railway")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    for key in _ENTRY_FLAGS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(disabled_flag, "false")

    normalise_live_env_defaults()
    for key in _ENTRY_FLAGS:
        assert os.environ[key] == ("false" if key == disabled_flag else "true")
    normalise_live_env_defaults()
    assert os.environ[disabled_flag] == "false"


@pytest.mark.parametrize("platform", ["aws_lightsail", "railway", ""])
@pytest.mark.parametrize("mode", ["PAPER", "SHADOW"])
def test_nonlive_entry_features_remain_unset(monkeypatch, platform, mode) -> None:
    _track_admission_test_environment(monkeypatch)
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", platform)
    monkeypatch.setenv("PRODUCTION_LIVE_DEFAULT_INITIALIZED", "true")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("EXECUTION_MODE", mode)
    for key in _ENTRY_FLAGS:
        monkeypatch.delenv(key, raising=False)

    normalise_live_env_defaults()
    assert all(os.getenv(key) is None for key in _ENTRY_FLAGS)


@pytest.mark.parametrize(
    ("mode", "enable_live"),
    [("LIVE", "false"), ("PAPER", "true")],
)
def test_inconsistent_mode_flags_do_not_admit_live_features(
    monkeypatch, mode, enable_live
) -> None:
    _track_admission_test_environment(monkeypatch)
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "railway")
    monkeypatch.setenv("EXECUTION_MODE", mode)
    monkeypatch.setenv("ENABLE_LIVE", enable_live)
    for key in _ENTRY_FLAGS:
        monkeypatch.delenv(key, raising=False)

    normalise_live_env_defaults()
    assert all(os.getenv(key) is None for key in _ENTRY_FLAGS)
