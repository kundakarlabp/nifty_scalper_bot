"""Integration test covering the order execution hub pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nifty_scalper_bot.execution.execution_router import ExecutionResult
from nifty_scalper_bot.execution.order_execution_hub import OrderExecutionHub
from nifty_scalper_bot.execution.order_queue import OrderRequest
from nifty_scalper_bot.execution.preflight_validator import ValidationResult


@pytest.fixture
def mock_components():
    """Return an execution hub instance with mocked dependencies."""

    state_tracker = Mock()
    state_tracker.get_open_positions.return_value = []
    state_tracker.get_position_state.return_value = None
    state_tracker.add_order = Mock()
    state_tracker.update_position = Mock()

    regime_manager = Mock()
    regime_manager.get_current_regime.return_value = "TREND"
    regime_manager.get_regime_confidence.return_value = 0.85

    preflight_validator = Mock()
    preflight_validator.validate.return_value = ValidationResult(
        allowed=True,
        reasons=[],
        blocking_level="NONE",
    )
    preflight_validator._regime_manager = regime_manager

    lifecycle_manager = Mock()
    lifecycle_manager.start = AsyncMock()
    lifecycle_manager.shutdown = AsyncMock()
    lifecycle_manager.on_fill = Mock()

    order_queue = Mock()
    order_queue.get_queue_snapshot.return_value = []
    order_queue.submit_order_request = Mock()
    order_queue.get_next_request = Mock(return_value=None)

    execution_router = Mock()
    execution_router.get_stats.return_value = {}
    fill_result = ExecutionResult(
        status="FILLED",
        order_id="OID123",
        fill_quantity=75,
        fill_price=25000.0,
        rejection_reason=None,
    )
    execution_router.execute = Mock(return_value=fill_result)

    post_fill_monitor = Mock()
    post_fill_monitor.start = AsyncMock()
    post_fill_monitor.stop = AsyncMock()
    post_fill_monitor.get_stats.return_value = {}

    datahub = Mock()
    datahub.get_indicator = Mock(return_value=12.5)

    hub = OrderExecutionHub(
        state_tracker=state_tracker,
        preflight_validator=preflight_validator,
        lifecycle_manager=lifecycle_manager,
        order_queue=order_queue,
        execution_router=execution_router,
        post_fill_monitor=post_fill_monitor,
        data_hub=datahub,
    )
    return SimpleNamespace(
        hub=hub,
        state_tracker=state_tracker,
        lifecycle_manager=lifecycle_manager,
        execution_router=execution_router,
        order_queue=order_queue,
        preflight_validator=preflight_validator,
        fill_result=fill_result,
        datahub=datahub,
        regime_manager=regime_manager,
    )


@pytest.mark.asyncio
async def test_entry_to_tp2_full_cycle(mock_components):
    """Ensure entry execution triggers lifecycle hooks with ATR and regime."""

    hub = mock_components.hub
    lifecycle_manager = mock_components.lifecycle_manager
    execution_router = mock_components.execution_router
    order_queue = mock_components.order_queue

    request = OrderRequest(
        symbol="NIFTY25OCT25000CE",
        side="BUY",
        quantity=75,
        intent="ENTRY",
        source="test",
    )

    order_queue.get_next_request.side_effect = [request, None]
    submission_id = hub.submit_order_request(request)
    assert submission_id.startswith("req_")

    validation = hub._run_preflight(request)
    assert validation is not None

    await hub._dispatch_request(request)

    execution_router.execute.assert_called_once_with(request)
    lifecycle_manager.on_fill.assert_called_once()
    kwargs = lifecycle_manager.on_fill.call_args.kwargs
    assert kwargs["symbol"] == "NIFTY25OCT25000CE"
    assert kwargs["atr"] == pytest.approx(12.5)
    assert kwargs["regime"] == "TREND"
    assert kwargs["quantity"] == 75
    assert kwargs["entry_price"] == pytest.approx(25000.0)
