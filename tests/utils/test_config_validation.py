"""Unit tests for configuration validation helpers."""

import pytest

from nifty_scalper_bot.utils.config_validation import validate_execution_config


@pytest.fixture(name="required_vars")
def fixture_required_vars() -> tuple[str, ...]:
    """Return required environment variable names.

    Args:
        None.

    Returns:
        tuple[str, ...]: Tuple of required variable names.

    Raises:
        None.
    """

    return (
        "EXECUTION_MODE",
        "LIFECYCLE_TP1_R",
        "LIFECYCLE_TP2_R_TREND",
    )


def _clear_required(
    monkeypatch: pytest.MonkeyPatch,
    variables: tuple[str, ...],
) -> None:
    """Unset required environment variables for validation tests.

    Args:
        monkeypatch: pytest monkeypatch fixture for environment mutation.
        variables: Collection of environment variable names to clear.

    Returns:
        None.

    Raises:
        None.
    """

    for var in variables:
        monkeypatch.delenv(var, raising=False)


def test_validate_execution_config_missing(
    monkeypatch: pytest.MonkeyPatch,
    required_vars: tuple[str, ...],
) -> None:
    """Report missing errors when required variables are absent.

    Args:
        monkeypatch: pytest monkeypatch fixture for environment mutation.
        required_vars: Required environment variable names.

    Returns:
        None.

    Raises:
        None.
    """

    _clear_required(monkeypatch, required_vars)
    errors = validate_execution_config()
    for var in required_vars:
        assert f"Missing env var: {var}" in errors


def test_validate_execution_config_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    required_vars: tuple[str, ...],
) -> None:
    """Flag invalid numeric values and execution modes.

    Args:
        monkeypatch: pytest monkeypatch fixture for environment mutation.
        required_vars: Required environment variable names.

    Returns:
        None.

    Raises:
        None.
    """

    _clear_required(monkeypatch, required_vars)
    monkeypatch.setenv("EXECUTION_MODE", "invalid")
    monkeypatch.setenv("LIFECYCLE_TP1_R", "not-a-number")
    monkeypatch.setenv("LIFECYCLE_TP2_R_TREND", "-5")
    errors = validate_execution_config()
    assert "LIFECYCLE_TP1_R must be a number" in errors
    assert "LIFECYCLE_TP2_R_TREND must be positive" in errors
    assert "Invalid EXECUTION_MODE: INVALID" in errors


def test_validate_execution_config_success(
    monkeypatch: pytest.MonkeyPatch,
    required_vars: tuple[str, ...],
) -> None:
    """Return an empty list when configuration is valid.

    Args:
        monkeypatch: pytest monkeypatch fixture for environment mutation.
        required_vars: Required environment variable names.

    Returns:
        None.

    Raises:
        None.
    """

    _clear_required(monkeypatch, required_vars)
    monkeypatch.setenv("EXECUTION_MODE", "shadow")
    monkeypatch.setenv("LIFECYCLE_TP1_R", "1.5")
    monkeypatch.setenv("LIFECYCLE_TP2_R_TREND", "2.0")
    errors = validate_execution_config()
    assert errors == []
