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

class _BrokerWithPositions:
    def __init__(self, positions):
        self._positions = positions

    def get_positions(self):
        return list(self._positions)


class _BrokerSyncedPositionManager(_PositionManager):
    def __init__(self):
        super().__init__()
        self._open = []

    def synchronize_with_broker(self, positions):
        self.synced = list(positions)
        self._open = [
            SimpleNamespace(
                symbol=p["symbol"],
                quantity=abs(int(p["quantity"])),
                side="LONG" if int(p["quantity"]) > 0 else "SHORT",
                average_price=p.get("average_price", 100.0),
                last_price=p.get("last_price", 100.0),
            )
            for p in self.synced
            if int(p.get("quantity", 0)) != 0
        ]

    def get_open_positions(self):
        return list(self._open)

    def broker_exposure_state(self, _symbol):
        from nifty_scalper_bot.execution.position_snapshot import BrokerExposureState

        return BrokerExposureState.NONZERO


class _BracketManagerFake:
    def __init__(self):
        self._managed = set()
        self._brackets = {}
        self._symbol_map = {}

    def is_symbol_managed(self, symbol):
        return symbol in self._managed

    def get_bracket(self, bracket_id):
        return self._brackets.get(bracket_id)

    def manage(self, symbol, bracket_id="guard-1"):
        self._managed.add(symbol)
        self._symbol_map.setdefault(symbol, set()).add(bracket_id)
        self._brackets[bracket_id] = SimpleNamespace(
            symbol=symbol,
            status="ACTIVE",
            quantity=65,
            stop_loss=90.0,
        )
        return bracket_id


class _GuardOrderManager(_OrderManager):
    def __init__(self, bm, result=None, raises=False):
        super().__init__()
        self._bracket_manager = bm
        self.result = result
        self.raises = raises
        self.calls = []

    def guard_orphan_position(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise RuntimeError("guard failed")
        if self.result == "manage":
            return self._bracket_manager.manage(kwargs["symbol"])
        return self.result


@pytest.mark.asyncio
async def test_reconcile_retains_unprotected_blocker_when_guard_returns_none(caplog):
    symbol = "NFO:NIFTY2660923100CE"
    bm = _BracketManagerFake()
    ctx = SimpleNamespace(
        broker_client=SimpleNamespace(client=_BrokerWithPositions([{"symbol": symbol, "quantity": 65, "average_price": 100.0, "last_price": 100.0, "product": "MIS"}])),
        order_manager=_GuardOrderManager(bm, result=None),
        position_manager=_BrokerSyncedPositionManager(),
        data_hub=None,
        unprotected_broker_position=False,
        unprotected_broker_positions=set(),
        position_reconciliation_failed=False,
        live_orders_armed=False,
    )

    await _reconcile_state(ctx, source="test")

    assert ctx.position_reconciliation_failed is False
    assert ctx.unprotected_broker_position is True
    assert symbol in ctx.unprotected_broker_positions
    assert "POSITION_ORPHAN_GUARD_FAILED" in caplog.text
    assert "POSITION_ADOPTED_TO_BRACKET" not in caplog.text


@pytest.mark.asyncio
async def test_reconcile_clears_only_verified_orphan_protection(caplog):
    symbol = "NFO:NIFTY2660923100CE"
    bm = _BracketManagerFake()
    ctx = SimpleNamespace(
        broker_client=SimpleNamespace(client=_BrokerWithPositions([{"symbol": symbol, "quantity": 65, "average_price": 100.0, "last_price": 100.0, "product": "MIS"}])),
        order_manager=_GuardOrderManager(bm, result="manage"),
        position_manager=_BrokerSyncedPositionManager(),
        data_hub=None,
        unprotected_broker_position=False,
        unprotected_broker_positions=set(),
        position_reconciliation_failed=False,
        live_orders_armed=False,
    )

    await _reconcile_state(ctx, source="test")

    assert ctx.unprotected_broker_position is False
    assert ctx.unprotected_broker_positions == set()
    assert "POSITION_ADOPTED_TO_BRACKET" in caplog.text
    assert ctx.position_reconciliation_last_run["source"] == "test"
    assert ctx.position_reconciliation_last_run["reconcile_run_id"]
