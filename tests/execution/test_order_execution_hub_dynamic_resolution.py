"""Tests for dynamic ATR and regime resolution in OrderExecutionHub."""

from __future__ import annotations

from unittest.mock import Mock

from nifty_scalper_bot.execution.order_execution_hub import OrderExecutionHub


def _build_hub(**overrides: object) -> OrderExecutionHub:
    """Construct a hub instance for tests.

    Args:
        **overrides: Optional dependency overrides for the hub.

    Returns:
        OrderExecutionHub: Configured hub with mocked dependencies.

    Raises:
        None.
    """

    state_tracker = Mock()
    state_tracker.get_position_state.return_value = None
    preflight_validator = Mock()
    lifecycle_manager = Mock()
    order_queue = Mock()
    execution_router = Mock()
    post_fill_monitor = Mock()
    data_hub = overrides.get("data_hub")
    regime_manager = overrides.get("regime_manager")
    risk_manager = overrides.get("risk_manager")
    hub = OrderExecutionHub(
        state_tracker=state_tracker,
        preflight_validator=preflight_validator,
        lifecycle_manager=lifecycle_manager,
        order_queue=order_queue,
        execution_router=execution_router,
        post_fill_monitor=post_fill_monitor,
        data_hub=data_hub,
        regime_manager=regime_manager,
        risk_manager=risk_manager,
    )
    hub._state_tracker = state_tracker
    hub._data_hub = data_hub
    hub._regime_manager = regime_manager
    hub._risk_manager = risk_manager
    return hub


def test_get_atr_from_metadata() -> None:
    """ATR should be resolved from metadata when provided.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the resolved ATR mismatches expectations.
    """

    hub = _build_hub()
    metadata = {"atr": 45.5}
    atr = hub._get_atr_for_symbol("NIFTY25OCT25000CE", metadata)
    assert atr == 45.5


def test_get_atr_from_datahub() -> None:
    """ATR should fallback to DataHub indicator when metadata missing.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If ATR fallback logic fails.
    """

    data_hub = Mock()
    data_hub.get_indicator = Mock(return_value=52.3)
    hub = _build_hub(data_hub=data_hub)
    atr = hub._get_atr_for_symbol("NIFTY25OCT25000CE")
    assert atr == 52.3
    data_hub.get_indicator.assert_called_once_with("NIFTY25OCT25000CE", "atr")


def test_get_atr_default_for_unknown_symbol() -> None:
    """Unknown symbols should use conservative default ATR.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the default ATR is unexpected.
    """

    hub = _build_hub()
    atr = hub._get_atr_for_symbol("XYZOPT202501CE")
    assert atr == 10.0


def test_get_regime_from_manager() -> None:
    """Regime should be sourced from the regime manager when available.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If the regime lookup does not match expectations.
    """

    regime_manager = Mock()
    regime_manager.get_current_regime = Mock(return_value="TREND")
    hub = _build_hub(regime_manager=regime_manager)
    regime = hub._get_current_regime("NIFTY25OCT25000CE")
    assert regime == "TREND"
    regime_manager.get_current_regime.assert_called_once()


def test_circuit_breaker_pauses_incremented() -> None:
    """Circuit breaker halts should increment pause counter.

    Args:
        None.

    Returns:
        None.

    Raises:
        AssertionError: If pause counters are not incremented.
    """

    risk_manager = Mock()
    risk_manager.is_circuit_breaker_tripped = Mock(return_value=(True, "max_loss"))
    hub = _build_hub(risk_manager=risk_manager)
    halted = hub._should_halt_processing()
    assert halted is True
    assert hub._stats["circuit_breaker_pauses"] == 1
