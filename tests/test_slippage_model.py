from src.execution.order_executor import OrderExecutor


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - stub
        pass


def test_entry_price_uses_half_spread() -> None:
    ex = OrderExecutor(kite=None, telegram_controller=DummyTelegram())
    rid = ex.place_order(
        {
            "action": "BUY",
            "quantity": 150,
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 105.0,
            "symbol": "TEST",
            "bid": 100.0,
            "ask": 100.2,
            "depth": 1000,
        }
    )
    assert rid is not None
    rec = ex._active[rid]
    assert rec.entry_price == 100.2
