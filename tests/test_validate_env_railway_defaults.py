"""Tests for Railway-hosted environment defaults in validate_env."""

from __future__ import annotations

import importlib


def _reload_validate_env():
    import src.boot.validate_env as validate_env

    return importlib.reload(validate_env)


def test_railway_defaults(monkeypatch):
    """Flags default to true on Railway unless overridden."""

    monkeypatch.delenv("RAILWAY_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAILWAY_STATIC_URL", raising=False)
    monkeypatch.delenv("DATA__WARMUP_DISABLE", raising=False)
    ve = _reload_validate_env()
    assert ve.IS_HOSTED_RAILWAY is False
    assert ve.data_warmup_disable() is False

    monkeypatch.setenv("RAILWAY_PROJECT_ID", "123")
    ve = _reload_validate_env()
    assert ve.IS_HOSTED_RAILWAY is True
    assert ve.data_warmup_disable() is True

    monkeypatch.setenv("DATA__WARMUP_DISABLE", "false")
    ve = _reload_validate_env()
    assert ve.data_warmup_disable() is False

    monkeypatch.delenv("RAILWAY_PROJECT_ID", raising=False)
    monkeypatch.delenv("DATA__WARMUP_DISABLE", raising=False)
    _reload_validate_env()

