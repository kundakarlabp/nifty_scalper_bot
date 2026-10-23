from __future__ import annotations

import os

from nifty_scalper_bot.config.env_utils import normalise_live_env_defaults


DERIVED_KEYS = (
    'ENABLE_LIVE',
    'ENABLE_LIVE_TRADING',
    'EXECUTION_MODE',
    'ORDERS__ENABLE_LIVE',
    'PAPER__ENABLED',
    'PAPER_MODE',
    'SHADOW_MODE',
)


def _clear_derived(monkeypatch) -> None:
    for key in DERIVED_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_minimal_railway_live_env_derives_flags(monkeypatch) -> None:
    _clear_derived(monkeypatch)
    monkeypatch.setenv('ENABLE_LIVE', 'true')
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')

    normalise_live_env_defaults()

    assert os.getenv('ENABLE_LIVE_TRADING') == 'true'
    assert os.getenv('ORDERS__ENABLE_LIVE') == 'true'
    assert os.getenv('PAPER__ENABLED') == 'false'
    assert os.getenv('PAPER_MODE') == 'false'
    assert os.getenv('SHADOW_MODE') == 'false'


def test_execution_mode_live_alone_must_not_enable_real_trading(monkeypatch) -> None:
    _clear_derived(monkeypatch)
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")

    normalise_live_env_defaults()

    assert os.getenv("ENABLE_LIVE") == "false"
    assert os.getenv("ENABLE_LIVE_TRADING") == "false"
    assert os.getenv("ORDERS__ENABLE_LIVE") == "false"
    assert os.getenv("EXECUTION_MODE") == "SHADOW"
    assert os.getenv("PAPER_MODE") == "true"


def test_enable_live_without_matching_mode_is_inert(monkeypatch) -> None:
    _clear_derived(monkeypatch)
    monkeypatch.setenv("ENABLE_LIVE", "true")

    normalise_live_env_defaults()

    assert os.getenv("ENABLE_LIVE") == "false"
    assert os.getenv("EXECUTION_MODE") == "SHADOW"
    assert os.getenv("ENABLE_LIVE_TRADING") == "false"



def test_paper_defaults_when_no_live_requested(monkeypatch) -> None:
    _clear_derived(monkeypatch)

    normalise_live_env_defaults()

    assert os.getenv('ENABLE_LIVE') == 'false'
    assert os.getenv('ENABLE_LIVE_TRADING') == 'false'
    assert os.getenv('EXECUTION_MODE') == 'PAPER'
    assert os.getenv('ORDERS__ENABLE_LIVE') == 'false'
    assert os.getenv('PAPER__ENABLED') == 'true'
    assert os.getenv('PAPER_MODE') == 'true'
    assert os.getenv('SHADOW_MODE') == 'true'


def test_explicit_railway_override_preserved(monkeypatch) -> None:
    _clear_derived(monkeypatch)
    monkeypatch.setenv('ENABLE_LIVE', 'true')
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('PAPER_MODE', 'true')

    normalise_live_env_defaults()

    assert os.getenv('PAPER_MODE') == 'false'


def test_first_lightsail_boot_never_promotes_legacy_shadow_to_live(
    monkeypatch, tmp_path
) -> None:
    _clear_derived(monkeypatch)
    monkeypatch.delenv("PRODUCTION_DEFAULT_LIVE", raising=False)
    monkeypatch.delenv("PRODUCTION_LIVE_DEFAULT_INITIALIZED", raising=False)
    monkeypatch.setenv("DEPLOYMENT_PLATFORM", "aws_lightsail")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    env_file = tmp_path / "niftybot.env"
    original = "ENABLE_LIVE=false\\nEXECUTION_MODE=SHADOW\\nKITE_API_SECRET=keep-me\\n"
    env_file.write_text(original, encoding="utf-8")
    monkeypatch.setenv("BOT_ENV_FILE", str(env_file))

    normalise_live_env_defaults()

    assert os.getenv("ENABLE_LIVE") == "false"
    assert os.getenv("EXECUTION_MODE") == "SHADOW"
    assert os.getenv("ENABLE_LIVE_TRADING") == "false"
    assert os.getenv("ORDERS__ENABLE_LIVE") == "false"
    assert os.getenv("PRODUCTION_LIVE_DEFAULT_INITIALIZED") is None
    assert env_file.read_text(encoding="utf-8") == original



def test_lightsail_explicit_default_live_opt_out_preserves_shadow(monkeypatch) -> None:
    _clear_derived(monkeypatch)
    monkeypatch.setenv('DEPLOYMENT_PLATFORM', 'aws_lightsail')
    monkeypatch.setenv('PRODUCTION_DEFAULT_LIVE', 'false')
    monkeypatch.delenv('PRODUCTION_LIVE_DEFAULT_INITIALIZED', raising=False)
    monkeypatch.setenv('ENABLE_LIVE', 'false')
    monkeypatch.setenv('EXECUTION_MODE', 'SHADOW')

    normalise_live_env_defaults()

    assert os.getenv('ENABLE_LIVE') == 'false'
    assert os.getenv('EXECUTION_MODE') == 'SHADOW'


def test_lightsail_initialized_shadow_choice_remains_authoritative(monkeypatch) -> None:
    _clear_derived(monkeypatch)
    monkeypatch.setenv('DEPLOYMENT_PLATFORM', 'aws_lightsail')
    monkeypatch.setenv('PRODUCTION_LIVE_DEFAULT_INITIALIZED', 'true')
    monkeypatch.setenv('ENABLE_LIVE', 'false')
    monkeypatch.setenv('ENABLE_LIVE_TRADING', 'true')
    monkeypatch.setenv('EXECUTION_MODE', 'SHADOW')
    monkeypatch.setenv('ORDERS__ENABLE_LIVE', 'true')
    monkeypatch.setenv('PAPER__ENABLED', 'false')
    monkeypatch.setenv('PAPER_MODE', 'false')
    monkeypatch.setenv('SHADOW_MODE', 'false')

    normalise_live_env_defaults()

    assert os.getenv('ENABLE_LIVE') == 'false'
    assert os.getenv('ENABLE_LIVE_TRADING') == 'false'
    assert os.getenv('EXECUTION_MODE') == 'SHADOW'
    assert os.getenv('ORDERS__ENABLE_LIVE') == 'false'
    assert os.getenv('PAPER__ENABLED') == 'true'
    assert os.getenv('PAPER_MODE') == 'true'
    assert os.getenv('SHADOW_MODE') == 'true'
