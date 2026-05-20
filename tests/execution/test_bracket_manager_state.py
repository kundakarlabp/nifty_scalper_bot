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
    bm._pending_entry_reconcile_after_sec = 0.0
    return bm


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
