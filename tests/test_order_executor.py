from __future__ import annotations

import pytest

from nifty_scalper_bot.config.base import RiskConfig
from nifty_scalper_bot.execution.order_executor import OrderExecutor
from nifty_scalper_bot.utils.errors import OrderPlacementError


class DummyMDM:
    def get_last_quote(self, symbol: str) -> dict[str, float]:
        return {"bid": 100.0, "ask": 101.0, "ts_ns": 1}


class SuccessfulBroker:
    def place_order(self, payload: dict[str, object]) -> dict[str, object]:
        return {"order_id": "TEST123", "payload": payload}


def test_order_executor_rejects_large_notional() -> None:
    risk = RiskConfig(max_daily_trades=5, max_order_notional=100.0, allow_short=True)
    executor = OrderExecutor(SuccessfulBroker(), risk, DummyMDM())
    with pytest.raises(OrderPlacementError):
        executor.place_market_order(symbol="NIFTY", side="BUY", qty=1, price=150.0)

from nifty_scalper_bot.execution.order_executor import ExecutionError


class KwargsBroker:
    def __init__(self): self.kwargs=None
    def place_order(self, **kwargs):
        self.kwargs = kwargs
        return {"order_id":"K1"}


class LegacyBroker:
    def place_order(self, payload):
        return {"order_id":"L1", "payload":payload}


def test_order_executor_kwargs_contract() -> None:
    risk = RiskConfig(max_daily_trades=5, max_order_notional=1_000_000.0, allow_short=True)
    broker = KwargsBroker()
    executor = OrderExecutor(broker, risk, DummyMDM())
    oid = executor.place_market_order(symbol="NIFTY", side="BUY", qty=1, price=100.0)
    assert oid == 'K1'
    assert broker.kwargs['quantity'] == 1
    assert broker.kwargs['order_type'] == 'LIMIT'


def test_order_executor_legacy_payload_contract() -> None:
    risk = RiskConfig(max_daily_trades=5, max_order_notional=1_000_000.0, allow_short=True)
    executor = OrderExecutor(LegacyBroker(), risk, DummyMDM())
    assert executor.place_market_order(symbol="NIFTY", side="BUY", qty=1, price=100.0) == 'L1'


def test_order_executor_invalid_response_raises() -> None:
    class BadBroker:
        def place_order(self, **kwargs): return 'x'
    risk = RiskConfig(max_daily_trades=5, max_order_notional=1_000_000.0, allow_short=True)
    executor = OrderExecutor(BadBroker(), risk, DummyMDM())
    with pytest.raises(ExecutionError):
        executor.place_market_order(symbol="NIFTY", side="BUY", qty=1, price=100.0)


def test_order_executor_kwargs_payload_uses_strict_keys() -> None:
    risk = RiskConfig(max_daily_trades=5, max_order_notional=1_000_000.0, allow_short=True)
    broker = KwargsBroker()
    executor = OrderExecutor(broker, risk, DummyMDM())
    executor.place_market_order(symbol="NIFTY", side="BUY", qty=1, price=100.0)
    assert "qty" not in broker.kwargs
    assert "type" not in broker.kwargs
    assert broker.kwargs["quantity"] == 1
    assert broker.kwargs["order_type"] == "LIMIT"


def test_order_executor_quote_context_handles_malformed_quote() -> None:
    class BadMDM:
        def get_last_quote(self, symbol: str): return {"bid": "bad", "ask": None, "ts_ns": "bad"}
    risk = RiskConfig(max_daily_trades=5, max_order_notional=1_000_000.0, allow_short=True)
    executor = OrderExecutor(KwargsBroker(), risk, BadMDM())
    assert executor.place_market_order(symbol="NIFTY", side="BUY", qty=1, price=100.0) == "K1"


def test_order_executor_find_open_order_scans_get_orders() -> None:
    class B:
        def get_orders(self):
            return [{"client_order_id": "CID1", "status": "OPEN", "order_id": "OID1"}]
    ex = OrderExecutor(B(), RiskConfig(max_daily_trades=5, max_order_notional=1_000_000.0, allow_short=True), DummyMDM())
    assert ex._find_open_order("CID1")["order_id"] == "OID1"


def test_order_executor_find_open_order_ignores_terminal_statuses() -> None:
    class B:
        def get_orders(self):
            return [
                {"client_order_id": "CID1", "status": "COMPLETE", "order_id": "OID1"},
                {"client_order_id": "CID1", "status": "REJECTED", "order_id": "OID2"},
            ]
    ex = OrderExecutor(B(), RiskConfig(max_daily_trades=5, max_order_notional=1_000_000.0, allow_short=True), DummyMDM())
    assert ex._find_open_order("CID1") is None


def test_order_executor_find_open_order_matches_tag_suffix() -> None:
    class B:
        def get_orders(self):
            return [{"tag": "BOT_ABCD1234", "status": "OPEN", "order_id": "OID3"}]
    ex = OrderExecutor(B(), RiskConfig(max_daily_trades=5, max_order_notional=1_000_000.0, allow_short=True), DummyMDM())
    assert ex._find_open_order("bot_abcd1234")["order_id"] == "OID3"
