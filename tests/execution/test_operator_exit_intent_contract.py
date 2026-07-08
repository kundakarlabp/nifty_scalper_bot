from types import SimpleNamespace

from nifty_scalper_bot.execution.operator_control_patch import _place_flatten_order, emergency_stop


class FakeRuntimeOrderManager:
    def __init__(self):
        self.orders = []
        self.cancelled = []
        self._broker = SimpleNamespace(
            get_positions=lambda: [
                {"tradingsymbol": "NIFTY24JUL24000CE", "quantity": 75}
            ],
            get_orders=lambda: [],
        )

    def place_order(self, **kwargs):
        self.orders.append(dict(kwargs))
        return f"OID{len(self.orders)}"


def test_flatten_order_is_explicit_reduce_intent():
    manager = FakeRuntimeOrderManager()

    order_id = _place_flatten_order(manager, "NFO:NIFTY24JUL24000CE", 75)

    assert order_id == "OID1"
    assert manager.orders == [
        {
            "symbol": "NFO:NIFTY24JUL24000CE",
            "side": "SELL",
            "quantity": 75,
            "order_type": "MARKET",
            "tag": "EXIT_FLATTEN_TELEGRAM",
            "check_risk": False,
            "product": "MIS",
            "intent": "REDUCE",
            "strategy_name": "operator_flatten",
        }
    ]


def test_emergency_stop_pauses_and_flattens_open_exposure():
    manager = FakeRuntimeOrderManager()

    result = emergency_stop(manager, reason="test_emergency")

    assert result["kill_switch"] is True
    assert result["flatten"]["submitted"] == [
        {
            "symbol": "NFO:NIFTY24JUL24000CE",
            "qty": 75,
            "side": "SELL",
            "order_id": "OID1",
        }
    ]
    assert manager.orders[0]["intent"] == "REDUCE"
    assert manager.orders[0]["check_risk"] is False
