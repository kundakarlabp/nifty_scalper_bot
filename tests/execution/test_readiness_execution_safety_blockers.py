from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core.app import _reconcile_state
from nifty_scalper_bot.execution.readiness import normalize_readiness_blockers


def test_position_reconciliation_failure_blocks_live_arming_until_successful_recompute():
    failed = normalize_readiness_blockers(
        ["position_reconciliation_failed"],
        market_state="open",
        live_mode=True,
        evaluation_ready=True,
        execution_ready=True,
    )
    assert failed.live_orders_armed is False
    assert failed.primary_blocker == "position_reconciliation_failed"

    cleared = normalize_readiness_blockers(
        [],
        market_state="open",
        live_mode=True,
        evaluation_ready=True,
        execution_ready=True,
    )
    assert cleared.live_orders_armed is True


def test_unresolved_exit_position_blocks_live_arming():
    decision = normalize_readiness_blockers(
        ["unresolved_exit_position"],
        market_state="open",
        live_mode=True,
        evaluation_ready=True,
        execution_ready=True,
    )
    assert decision.live_orders_armed is False
    assert decision.primary_blocker == "unresolved_exit_position"


class _FlatBroker:
    def get_positions(self):
        return []


class _PositionManager:
    def __init__(self):
        self.synced = None

    def synchronize_with_broker(self, positions):
        self.synced = list(positions)

    def get_open_positions(self):
        return []


class _OrderManager:
    def __init__(self):
        self._bracket_manager = None
        self.status_reports = 0

    def reconcile_open_orders_with_broker(self):
        return None

    def _log_status_report(self):
        self.status_reports += 1


@pytest.mark.asyncio
async def test_successful_flat_reconcile_clears_previous_unprotected_blocker():
    ctx = SimpleNamespace(
        broker_client=SimpleNamespace(client=_FlatBroker()),
        order_manager=_OrderManager(),
        position_manager=_PositionManager(),
        data_hub=None,
        unprotected_broker_position=True,
        position_reconciliation_failed=True,
        live_orders_armed=False,
    )

    await _reconcile_state(ctx)

    assert ctx.position_reconciliation_failed is False
    assert ctx.unprotected_broker_position is False
    assert ctx.position_manager.synced == []

    decision = normalize_readiness_blockers(
        [],
        market_state="open",
        live_mode=True,
        evaluation_ready=True,
        execution_ready=True,
    )
    assert decision.live_orders_armed is True
