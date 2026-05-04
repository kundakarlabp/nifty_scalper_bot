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


def test_execution_mode_live_alone_derives_enable_live(monkeypatch) -> None:
    _clear_derived(monkeypatch)
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')

    normalise_live_env_defaults()

    assert os.getenv('ENABLE_LIVE') == 'true'
    assert os.getenv('ENABLE_LIVE_TRADING') == 'true'
    assert os.getenv('ORDERS__ENABLE_LIVE') == 'true'
    assert os.getenv('PAPER__ENABLED') == 'false'
    assert os.getenv('PAPER_MODE') == 'false'
    assert os.getenv('SHADOW_MODE') == 'false'


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

    assert os.getenv('PAPER_MODE') == 'true'
