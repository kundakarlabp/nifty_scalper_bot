import asyncio
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app
from nifty_scalper_bot.data.data_hub import SubscriptionState
from nifty_scalper_bot.execution.readiness import ReadinessDecision
from nifty_scalper_bot.utils.errors import BrokerReconciliationError


def test_attach_canonical_entry_guard_blocks_missing_readiness():
    order_manager = SimpleNamespace(guard=None)
    order_manager.set_entry_execution_guard = lambda guard: setattr(order_manager, "guard", guard)
    ctx = SimpleNamespace(
        order_manager=order_manager,
        broker_auth_invalid=False,
        broker_session_invalid=False,
        broker_balance_valid=True,
        position_reconciliation_failed=False,
        position_reconciliation_completed=True,
        unresolved_reconciliation_symbols=set(),
        unprotected_broker_positions=set(),
        readiness_decision=None,
    )

    app.attach_canonical_entry_execution_guard(ctx)

    assert order_manager.guard is not None
    assert order_manager.guard() == (False, "readiness_snapshot_missing")


@pytest.mark.parametrize("payload", [{}, None, 1, {"status": "error"}, {"error_type": "x"}])
def test_normalize_broker_positions_payload_rejects_invalid(payload):
    with pytest.raises(BrokerReconciliationError):
        app._normalize_broker_positions_payload(payload)


def test_normalize_broker_positions_payload_accepts_verified_empty():
    assert app._normalize_broker_positions_payload({"net": [], "day": []}) == []


@pytest.mark.asyncio
async def test_auth_failure_callback_style_recompute_increments_generation(monkeypatch):
    calls = []
    ctx = SimpleNamespace(
        active_trading_universe={},
        active_contract_basket={},
        market_data_manager=None,
        data_hub=None,
        settings=SimpleNamespace(execution_mode="LIVE"),
        strategy_runner=SimpleNamespace(
            get_status=lambda: {"running": False},
            set_runtime_readiness=lambda **kwargs: calls.append(kwargs),
        ),
        order_manager=object(),
        broker_client=SimpleNamespace(is_connected=lambda: True),
        readiness_generation=0,
        broker_auth_invalid=True,
        broker_session_invalid=False,
        broker_balance_valid=False,
        position_reconciliation_completed=False,
        position_reconciliation_failed=False,
        unprotected_broker_positions=set(),
        selected_ce=None,
        selected_pe=None,
    )
    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(app, "_ensure_context_history_hydrated", _noop)
    monkeypatch.setattr(app, "_ensure_selected_options_hydrated", _noop)

    await app._recompute_and_push_runtime_readiness(ctx, reason="broker_auth_invalid")

    assert ctx.readiness_generation == 1
    assert ctx.readiness_decision.primary_blocker == "broker_auth_invalid"
    assert ctx.live_orders_armed is False


def test_subscription_record_live_state_required():
    class Hub:
        def __init__(self, state):
            self.state = state
        def get_subscription_snapshot(self, symbol):
            return SimpleNamespace(state=self.state)

    assert Hub(SubscriptionState.LIVE).get_subscription_snapshot("x").state is SubscriptionState.LIVE
