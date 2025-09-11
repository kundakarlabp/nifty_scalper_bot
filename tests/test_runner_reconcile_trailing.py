from types import SimpleNamespace

from src.execution.order_executor import OrderExecutor
from src.state.store import StateStore
from src.strategies.registry import ActiveComponents
from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - capture only
        pass

    def send_eod_summary(self) -> None:  # pragma: no cover - capture only
        pass


class Kite:
    def orders(self):
        return [{"tag": "RID1", "status": "OPEN", "order_id": "OID"}]

    def positions(self):
        return {"day": [{"tradingsymbol": "TEST", "quantity": 1, "last_price": 103.0}]}

def test_reconcile_restores_trailing(monkeypatch, tmp_path):
    store = StateStore(str(tmp_path / "state.json"))
    payload = {
        "symbol": "TEST",
        "action": "BUY",
        "quantity": 1,
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "trailing_enabled": True,
        "trailing_atr_mult": 1.0,
        "client_oid": "RID1",
    }
    store.record_order("RID1", payload)
    store.record_position("TEST", {"last_price": 103.0})

    kite = Kite()
    ex = OrderExecutor(kite=kite, state_store=store)

    def fake_init(settings, **kwargs):
        strat = SimpleNamespace()
        data = SimpleNamespace(connect=lambda: None)
        return ActiveComponents(strategy=strat, data_provider=data, order_connector=ex, names={})

    monkeypatch.setattr("src.strategies.runner.init_default_registries", fake_init)
    monkeypatch.setattr("src.strategies.runner.StateStore", lambda path: store)

    telegram = DummyTelegram()
    _ = StrategyRunner(telegram_controller=telegram)
    rec = ex.get_active_orders()[0]
    assert rec.sl_price and rec.sl_price > 95.0
