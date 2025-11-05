from pathlib import Path
import tempfile

import pytest

import nifty_scalper_bot.execution.order_manager as order_manager_module
from nifty_scalper_bot.execution.order_manager import (
    AtomicLeg,
    OrderManager,
    OrderType,
)
from nifty_scalper_bot.utils.errors import BrokerError, OrderPlacementError


class _Limiter:
    def acquire(self, *_args, **_kwargs) -> None:
        return None

    def snapshot(self) -> dict[str, object]:
        return {}


class _Positions:
    def __init__(self) -> None:
        self.updates: list[tuple[str, str]] = []

    def add_pending_order(
        self,
        *,
        order_id: str,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        order_type: str,
    ) -> None:
        self.updates.append((order_id, "PENDING"))

    def update_order_status(
        self, order_id: str, status: str, _price: float | None = None
    ) -> None:
        self.updates.append((order_id, status))

    def get_position(self, _symbol: str) -> None:
        return None

    def get_all_positions(self) -> list[object]:
        return []

    def get_realized_pnl(self) -> float:
        return 0.0

    def get_total_exposure(self) -> float:
        return 0.0


class _RiskManagerStub:
    def __init__(self, allowed: bool, reasons: tuple[str, ...]):
        self._allowed = allowed
        self._reasons = reasons

    def risk_gate_should_trade(self) -> tuple[bool, tuple[str, ...]]:
        return self._allowed, self._reasons


class _StubMarketDataManager:
    def __init__(
        self,
        quote: dict[str, float | None],
        age_ms: float,
        cached_mid: tuple[float | None, int | None] = (None, None),
    ) -> None:
        self._quote = quote
        self._age_ms = age_ms
        self._cached_mid = cached_mid

    def get_quote(self, symbol: str) -> dict[str, float | None]:
        return dict(self._quote)

    def quote_age_ms(self, symbol: str) -> float:
        return float(self._age_ms)

    def cached_mid(self, symbol: str) -> tuple[float | None, int | None]:
        return self._cached_mid


class _Broker:
    def __init__(self) -> None:
        self.orders: dict[str, dict[str, object]] = {}
        self.by_client: dict[str, dict[str, object]] = {}
        self.cancelled: list[str] = []
        self.reconcile_calls = 0
        self._counter = 0

    def place_order(self, payload: dict[str, object]) -> dict[str, object]:
        self._counter += 1
        order_id = f"OID-{self._counter}"
        record = {
            "order_id": order_id,
            "status": "complete",
            "client_order_id": payload.get("client_order_id"),
        }
        client_id = payload.get("client_order_id")
        if isinstance(client_id, str):
            self.by_client[client_id] = record
        self.orders[order_id] = record
        return record

    def cancel_order(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
        return True

    def get_order_status(self, order_id: str) -> dict[str, object]:
        return self.orders.get(order_id, {"order_id": order_id, "status": "open"})

    def get_order_by_client_order_id(
        self, client_order_id: str
    ) -> dict[str, object] | None:
        return self.by_client.get(client_order_id)

    def get_open_orders(self) -> list[dict[str, object]]:
        self.reconcile_calls += 1
        return list(self.orders.values())


class _RejectingBroker(_Broker):
    def place_order(self, payload: dict[str, object]) -> dict[str, object]:
        record = super().place_order(payload)
        if self._counter == 2:
            record["status"] = "rejected"
        return record


class _FlakyBroker(_Broker):
    def __init__(self) -> None:
        super().__init__()
        self._failed = False

    def place_order(self, payload: dict[str, object]) -> dict[str, object]:
        client_id = payload.get("client_order_id")
        if not self._failed and isinstance(client_id, str):
            self._failed = True
            self._counter += 1
            record = {
                "order_id": f"OID-{self._counter}",
                "status": "open",
                "client_order_id": client_id,
            }
            self.by_client[client_id] = record
            self.orders[record["order_id"]] = record
            raise BrokerError("timeout")
        return super().place_order(payload)


def _make_order_manager(broker: _Broker) -> OrderManager:
    positions = _Positions()
    limiter = _Limiter()
    history_path = Path(tempfile.mkdtemp(prefix="order-manager-history-")) / (
        "history.json"
    )
    manager = OrderManager(broker, positions, limiter, history_path=history_path)
    manager.set_risk_manager(_RiskManagerStub(True, ()))

    class _Resolver:
        def lot_size_for_symbol(self, symbol: str) -> int:
            return 75

    manager.set_instrument_resolver(_Resolver())
    return manager


def test_atomic_entry_success() -> None:
    broker = _Broker()
    om = _make_order_manager(broker)
    legs = [
        AtomicLeg(
            symbol="NIFTY25APR25000CE",
            side="BUY",
            quantity=75,
            order_type=OrderType.MARKET,
        ),
        AtomicLeg(
            symbol="NIFTY25APR25000PE",
            side="SELL",
            quantity=75,
            order_type=OrderType.MARKET,
        ),
    ]

    order_ids = om.place_atomic_entry(legs)

    assert len(order_ids) == 2
    assert order_ids[0] in broker.orders
    assert order_ids[1] in broker.orders


def test_atomic_entry_leg_failure_triggers_cancel() -> None:
    broker = _RejectingBroker()
    om = _make_order_manager(broker)
    legs = [
        AtomicLeg(symbol="NIFTY25APR25000CE", side="BUY", quantity=75),
        AtomicLeg(symbol="NIFTY25APR25000PE", side="SELL", quantity=75),
    ]

    with pytest.raises(OrderPlacementError):
        om.place_atomic_entry(legs)

    assert "OID-1" in broker.cancelled
    assert broker.reconcile_calls >= 1


def test_atomic_entry_retry_uses_client_order_lookup() -> None:
    broker = _FlakyBroker()
    om = _make_order_manager(broker)
    legs = [
        AtomicLeg(symbol="NIFTY25APR25000CE", side="BUY", quantity=75),
        AtomicLeg(symbol="NIFTY25APR25000PE", side="SELL", quantity=75),
    ]

    order_ids = om.place_atomic_entry(legs)

    assert len(order_ids) == 2
    # First leg placement should succeed despite the initial BrokerError.
    assert broker._failed is True
    # Only one logical order should be recorded for the first leg.
    assert broker.orders["OID-1"]["status"] == "open"


def test_reference_price_logging_includes_source_and_age(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mdm = _StubMarketDataManager({"bid": 100.0, "ask": 101.0}, 150.0)
    with caplog.at_level("INFO"):
        price, meta = order_manager_module.resolve_reference_price(
            "NIFTY",
            require_depth=True,
            max_age_ms=2_000,
            allow_ltp_fallback=True,
            allow_market_protect=False,
            protect_slippage_bps=12,
            mdm=mdm,
        )
    assert price == pytest.approx(100.5)
    assert isinstance(meta, order_manager_module.RefPriceMeta)
    assert meta.source == "mid"
    records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("reference_price")
    ]
    assert records
    latest = records[-1]
    assert "source=mid" in latest.getMessage()
    assert getattr(latest, "source", None) == "mid"
    assert getattr(latest, "age_ms", None) == pytest.approx(150.0)


def test_atomic_entry_blocked_by_risk_state() -> None:
    broker = _Broker()
    om = _make_order_manager(broker)
    om.set_risk_manager(_RiskManagerStub(False, ("STALE",)))
    legs = [
        AtomicLeg(symbol="NIFTY25APR25000CE", side="BUY", quantity=75),
        AtomicLeg(symbol="NIFTY25APR25000PE", side="SELL", quantity=75),
    ]

    order_ids = om.place_atomic_entry(legs)

    assert order_ids == []
    assert broker.orders == {}
