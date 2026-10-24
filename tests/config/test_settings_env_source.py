"""Regression tests for deterministic, fail-closed environment-file precedence."""

from __future__ import annotations

import os

import pytest

from nifty_scalper_bot.config.settings import _ensure_env_loaded_before_settings


@pytest.fixture(autouse=True)
def isolate_mode_environment(monkeypatch):
    """Restore keys modified in-place by the normalizer and dotenv."""
    keys = (
        "ENABLE_LIVE",
        "EXECUTION_MODE",
        "ENABLE_LIVE_TRADING",
        "ORDERS__ENABLE_LIVE",
        "PAPER_MODE",
        "PAPER__ENABLED",
        "SHADOW_MODE",
        "RISK__PER_TRADE_RISK_PCT",
        "RISK_PER_TRADE_PCT",
        "RISK_DAILY_LOSS_PCT",
        "RISK_DAILY_PNL_CAP_PCT",
        "RISK_MAX_DAILY_LOSS_PCT",
        "DAILY_PNL_CAP_PCT",
        "SMC_SUSTAINED_FLOW_CONFIRMATION_ENABLED",
        "VWAP_TREND_RETEST_EXTENSION_ENABLED",
        "ORDERFLOW_OFI_ADAPTIVE_HOLDOFF_ENABLED",
    )
    for key in keys:
        monkeypatch.setenv(key, os.getenv(key, ""))
        monkeypatch.delenv(key)


def test_explicit_protected_env_is_only_file_loaded(monkeypatch, tmp_path) -> None:
    protected = tmp_path / "secrets.env"
    protected.write_text("ENABLE_LIVE=false\nEXECUTION_MODE=SHADOW\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "ENABLE_LIVE=true\nEXECUTION_MODE=LIVE\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("BOT_ENV_FILE", str(protected))

    _ensure_env_loaded_before_settings()

    assert os.environ["ENABLE_LIVE"] == "false"
    assert os.environ["EXECUTION_MODE"] == "SHADOW"
    assert os.environ["ORDERS__ENABLE_LIVE"] == "false"


def test_missing_explicit_env_does_not_fall_back_to_repo_dotenv(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / ".env").write_text(
        "ENABLE_LIVE=true\nEXECUTION_MODE=LIVE\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("BOT_ENV_FILE", str(tmp_path / "missing.env"))

    _ensure_env_loaded_before_settings()

    assert os.environ["ENABLE_LIVE"] == "false"
    assert os.environ["ORDERS__ENABLE_LIVE"] == "false"
    assert os.environ["EXECUTION_MODE"] != "LIVE"


def test_production_without_file_path_does_not_discover_dotenv(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / ".env").write_text(
        "ENABLE_LIVE=true\nEXECUTION_MODE=LIVE\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.delenv("BOT_ENV_FILE", raising=False)

    _ensure_env_loaded_before_settings()

    assert os.environ["ENABLE_LIVE"] == "false"
    assert os.environ["ORDERS__ENABLE_LIVE"] == "false"


def test_explicit_systemd_live_settings_have_priority_over_dotenv(
    monkeypatch, tmp_path
) -> None:
    protected = tmp_path / "niftybot.env"
    protected.write_text("ENABLE_LIVE=false\nEXECUTION_MODE=SHADOW\n", encoding="utf-8")
    monkeypatch.setenv("BOT_ENV_FILE", str(protected))
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")

    _ensure_env_loaded_before_settings()

    assert os.environ["ENABLE_LIVE"] == "true"
    assert os.environ["EXECUTION_MODE"] == "LIVE"
    assert os.environ["ORDERS__ENABLE_LIVE"] == "true"


def test_development_retains_local_dotenv_discovery(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text(
        "ENABLE_LIVE=false\nEXECUTION_MODE=PAPER\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BOT_ENV_FILE", raising=False)
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "development")

    _ensure_env_loaded_before_settings()

    assert os.environ["ENABLE_LIVE"] == "false"
    assert os.environ["EXECUTION_MODE"] == "PAPER"


def test_effective_profile_reports_only_source_and_normalized_mode(
    monkeypatch, tmp_path, caplog
) -> None:
    protected = tmp_path / "operator-settings.env"
    protected.write_text(
        "ENABLE_LIVE=false\nEXECUTION_MODE=SHADOW\n", encoding="utf-8"
    )
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("BOT_ENV_FILE", str(protected))

    _ensure_env_loaded_before_settings()

    assert "CONFIG_EFFECTIVE_PROFILE source=operator_env_file" in caplog.text
    assert "mode=SHADOW live_orders=false" in caplog.text
    assert str(protected) not in caplog.text


def test_missing_production_source_reports_fail_closed_profile(
    monkeypatch, tmp_path, caplog
) -> None:
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("BOT_ENV_FILE", str(tmp_path / "absent.env"))
    (tmp_path / ".env").write_text(
        "ENABLE_LIVE=true\nEXECUTION_MODE=LIVE\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    _ensure_env_loaded_before_settings()

    assert "CONFIG_SOURCE_MISSING_FAIL_CLOSED" in caplog.text
    assert "CONFIG_EFFECTIVE_PROFILE source=operator_env_missing" in caplog.text
    assert "mode=SHADOW live_orders=false" in caplog.text
    assert str(tmp_path) not in caplog.text
