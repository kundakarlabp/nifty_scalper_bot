"""Regression coverage for broker stop-order restoration from saved orders."""

import json
import logging
from threading import RLock

import pytest

from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType


def _restore(monkeypatch, tmp_path, raw_type):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "orders.json").write_text(
        json.dumps(
            {
                "stop-1": {
                    "order_id": "stop-1",
                    "symbol": "NFO:NIFTY2671423950CE",
                    "side": "SELL",
                    "quantity": 65,
                    "price": 95.0,
                    "order_type": raw_type,
                    "status": "SUBMITTED",
                    "client_order_id": "stop:entry-1",
                    "tag": "protective-stop",
                    "intent": "EXIT",
                }
            }
        ),
        encoding="utf-8",
    )
    manager = object.__new__(OrderManager)
    manager._lock = RLock()
    manager._orders = {}
    manager._client_order_index = {}
    manager._logger = logging.getLogger("test_broker_stop_order_restore")
    manager._load_orders()
    return manager


@pytest.mark.parametrize(
    ("raw_type", "expected"),
    [
        ("SL", OrderType.STOP_LOSS),
        ("SL-M", OrderType.STOP_LOSS_MARKET),
        ("STOP_LOSS", OrderType.STOP_LOSS),
        ("STOP_LOSS_MARKET", OrderType.STOP_LOSS_MARKET),
        ("LIMIT", OrderType.LIMIT),
    ],
)
def test_saved_order_restores_type_and_client_identity(
    monkeypatch, tmp_path, raw_type, expected
):
    manager = _restore(monkeypatch, tmp_path, raw_type)
    restored = manager._orders["stop-1"]
    assert restored.order_type == expected
    assert restored.status.name == "SUBMITTED"
    assert restored.client_order_id == "stop:entry-1"
    assert manager._client_order_index["stop:entry-1"] == "stop-1"
    assert restored.tag == "protective-stop"
    assert restored.intent == "EXIT"


@pytest.mark.parametrize("raw_type", ["UNKNOWN_BROKER_ORDER", None])
def test_unknown_or_missing_stop_type_never_becomes_limit_or_market(
    monkeypatch, tmp_path, raw_type
):
    manager = _restore(monkeypatch, tmp_path, raw_type)
    assert manager._orders == {}
    assert manager._client_order_index == {}
