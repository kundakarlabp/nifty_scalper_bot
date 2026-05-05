"""Tests for runner env flag parsing."""

from __future__ import annotations

from nifty_scalper_bot.strategies.runner import _env_flag


def test_env_flag_returns_default_when_unset(monkeypatch) -> None:
    """Validate default path. Args: monkeypatch. Returns: none. Raises: none."""
    monkeypatch.delenv("RUNNER_ALLOW_DYNAMIC_SYMBOL_ADD", raising=False)

    assert _env_flag("RUNNER_ALLOW_DYNAMIC_SYMBOL_ADD", True) is True


def test_env_flag_false_when_disabled(monkeypatch) -> None:
    """Validate false values. Args: monkeypatch. Returns: none. Raises: none."""
    monkeypatch.setenv("RUNNER_ALLOW_DYNAMIC_SYMBOL_ADD", "false")

    assert _env_flag("RUNNER_ALLOW_DYNAMIC_SYMBOL_ADD", True) is False


def test_env_flag_true_when_enabled(monkeypatch) -> None:
    """Validate true values. Args: monkeypatch. Returns: none. Raises: none."""
    monkeypatch.setenv("RUNNER_ALLOW_DYNAMIC_SYMBOL_ADD", "true")

    assert _env_flag("RUNNER_ALLOW_DYNAMIC_SYMBOL_ADD", False) is True


def test_env_flag_invalid_value_falls_back_to_default(monkeypatch) -> None:
    """Validate invalid values fallback. Args: monkeypatch. Returns: none. Raises: none."""
    monkeypatch.setenv("RUNNER_ALLOW_DYNAMIC_SYMBOL_ADD", "not-a-bool")

    assert _env_flag("RUNNER_ALLOW_DYNAMIC_SYMBOL_ADD", True) is True
    assert _env_flag("RUNNER_ALLOW_DYNAMIC_SYMBOL_ADD", False) is False
