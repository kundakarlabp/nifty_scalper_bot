from nifty_scalper_bot.execution.bracket_manager import ExitExecutionResult
import time
from types import SimpleNamespace
from nifty_scalper_bot.execution.bracket_manager import BracketManager, BracketState


def test_exit_execution_result_instantiation_safe() -> None:
    result = ExitExecutionResult(submitted=False, confirmed=False, order_id=None, filled_qty=0, reason='test')
    assert result.reason == 'test'
    assert not hasattr(result, 'stop_loss_price')
    assert not hasattr(result, 'target_price')


def _mk_bm_with_broker(broker):
    om = SimpleNamespace(_broker=broker)
    bm = BracketManager(om)
    bm.shutdown()
    bm._watchdog_thread.join(timeout=1.0)
    bm._pending_entry_reconcile_after_sec = 0.0
    return bm


def _add_pending(bm, bracket):
    bm._brackets[bracket.entry_order_id] = bracket
    bm._symbol_map[bracket.symbol] = [bracket.entry_order_id]


def _pending(order_id, *, age=10):
    return BracketState(
        order_id,
        "NFO:NIFTYCE",
        "BUY",
        1,
        100.0,
        95.0,
        110.0,
        created_at=time.time() - age,
        active=False,
        entry_confirmed=False,
    )


def test_pending_bracket_reconciles_filled_order_from_get_order_status():
    class _Broker:
        def get_order_status(self, oid):
            return {"status": "COMPLETE", "average_price": 101.5}
    bm = _mk_bm_with_broker(_Broker())
    bracket = BracketState("E1", "NFO:NIFTYCE", "BUY", 1, 100.0, 95.0, 110.0, created_at=time.time() - 10, active=False, entry_confirmed=False)
    bm._brackets["E1"] = bracket
    bm._reconcile_pending_entry(bracket)
    assert bracket.entry_confirmed is True
    assert bracket.active is True


def test_pending_bracket_reconciles_filled_order_from_get_orders():
    class _Broker:
        def get_order_status(self, oid):
            return {}
        def get_orders(self):
            return [{"order_id": "E2", "status": "COMPLETE", "avg_price": 102.0}]
    bm = _mk_bm_with_broker(_Broker())
    bracket = BracketState("E2", "NFO:NIFTYCE", "BUY", 1, 100.0, 95.0, 110.0, created_at=time.time() - 10, active=False, entry_confirmed=False)
    bm._brackets["E2"] = bracket
    bm._reconcile_pending_entry(bracket)
    assert bracket.entry_confirmed is True
    assert bracket.active is True


def test_pending_bracket_not_reconciled_before_threshold():
    class _Broker:
        def get_order_status(self, oid):
            return {"status": "COMPLETE", "average_price": 101.0}
    bm = _mk_bm_with_broker(_Broker())
    bm._pending_entry_reconcile_after_sec = 30.0
    bracket = BracketState("E3", "NFO:NIFTYCE", "BUY", 1, 100.0, 95.0, 110.0, created_at=time.time(), active=False, entry_confirmed=False)
    bm._brackets["E3"] = bracket
    bm._reconcile_pending_entry(bracket)
    assert bracket.entry_confirmed is False


def test_terminal_unfilled_pending_entry_closes_only_when_broker_flat():
    class _Broker:
        def get_order_status(self, oid):
            return {"order_id": oid, "status": "REJECTED"}

        def get_positions(self):
            return []

    bm = _mk_bm_with_broker(_Broker())
    bracket = _pending("E4")
    _add_pending(bm, bracket)

    bm._reconcile_pending_entry(bracket)

    assert bm.get_bracket("E4") is None
    assert not bm.is_symbol_managed("NFO:NIFTYCE")


def test_terminal_pending_entry_is_preserved_when_position_state_unknown():
    class _Broker:
        def get_order_status(self, oid):
            return {"order_id": oid, "status": "CANCELLED"}

        def get_positions(self):
            raise RuntimeError("broker unavailable")

    bm = _mk_bm_with_broker(_Broker())
    bracket = _pending("E5")
    _add_pending(bm, bracket)

    bm._reconcile_pending_entry(bracket)

    assert bm.get_bracket("E5") is bracket


def test_live_or_exposed_pending_entry_is_preserved():
    class _Broker:
        status = "OPEN"
        positions = []

        def get_order_status(self, oid):
            return {"order_id": oid, "status": self.status}

        def get_positions(self):
            return list(self.positions)

    broker = _Broker()
    bm = _mk_bm_with_broker(broker)
    bracket = _pending("E7", age=300)
    _add_pending(bm, bracket)

    bm._reconcile_pending_entry(bracket)
    assert bm.get_bracket("E7") is bracket

    broker.status = "REJECTED"
    broker.positions = [{"symbol": "NFO:NIFTYCE", "quantity": 1}]
    bm._reconcile_pending_entry(bracket)
    assert bm.get_bracket("E7") is bracket


def test_old_absent_entry_closes_after_authoritative_order_and_position_checks():
    class _Broker:
        def get_orders(self):
            return []

        def get_positions(self):
            return []

    bm = _mk_bm_with_broker(_Broker())
    bm._pending_entry_stale_after_sec = 120.0
    bracket = _pending("E6", age=121)
    _add_pending(bm, bracket)

    bm._reconcile_pending_entry(bracket)

    assert bm.get_bracket("E6") is None


def test_recent_absent_entry_is_preserved_during_fill_grace():
    class _Broker:
        def get_orders(self):
            return []

        def get_positions(self):
            return []

    bm = _mk_bm_with_broker(_Broker())
    bracket = _pending("E8", age=30)
    _add_pending(bm, bracket)

    bm._reconcile_pending_entry(bracket)

    assert bm.get_bracket("E8") is bracket
