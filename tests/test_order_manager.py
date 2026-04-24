from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

import nifty_scalper_bot.config.settings as app_settings
from nifty_scalper_bot.core.trading_switch import trading_switch
from nifty_scalper_bot.execution.margin_engine import MarginDecision
from nifty_scalper_bot.execution.order_manager import (
    OrderManager,
    OrderStatus,
    OrderType,
)
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.utils.circuit_breaker import CircuitBreaker
from nifty_scalper_bot.utils.errors import (
    BrokerError,
    OrderPlacementError,
    RateLimitError,
)
from nifty_scalper_bot.utils.rate_limiter import RateLimiter


class FakeBrokerClient:
    """Test double simulating broker lifecycle responses."""

    def __init__(self) -> None:
        self._counter = 0
        self.orders: dict[str, dict[str, Any]] = {}

    def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._counter += 1
        order_id = f"ORD-{self._counter}"
        order = {
            "order_id": order_id,
            "status": "open",
            "filled_quantity": 0,
            "average_price": payload.get("price"),
            "message": "",
            "payload": payload,
            "quantity": payload.get("quantity", 0),
        }
        self.orders[order_id] = order
        return {"order_id": order_id, "status": "open"}

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        return self.orders.get(order_id, {})

    def cancel_order(self, order_id: str, variety: str = "regular") -> dict[str, Any]:
        order = self.orders.get(order_id)
        if order is None:
            return {}
        order["status"] = "cancelled"
        return {"order_id": order_id, "status": "cancelled"}

    def modify_order(
        self,
        order_id: str,
        quantity: int | None = None,
        price: float | None = None,
        variety: str = "regular",
    ) -> dict[str, Any]:
        order = self.orders.get(order_id)
        if order is None:
            return {}
        if quantity is not None:
            order["quantity"] = quantity
            if isinstance(order.get("payload"), dict):
                order["payload"]["quantity"] = quantity
        if price is not None:
            order["average_price"] = price
        return {"order_id": order_id}

    def fill(self, order_id: str, quantity: int, price: float) -> None:
        order = self.orders[order_id]
        order["status"] = "complete"
        order["filled_quantity"] = quantity
        order["average_price"] = price


class RejectingBroker:
    def __init__(self, message: str) -> None:
        self.attempts = 0
        self.message = message

    def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.attempts += 1
        return {
            "order_id": f"RJ-{self.attempts}",
            "status": "rejected",
            "message": self.message,
        }


class FailingBroker:
    def __init__(self) -> None:
        self.calls = 0

    def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        raise BrokerError("temporary failure")


@pytest.fixture()
def rate_limiter() -> RateLimiter:
    limiter = RateLimiter()
    limiter.configure_bucket("orders", capacity=5, refill_rate_per_sec=5)
    return limiter


@pytest.fixture()
def position_manager(tmp_path: Path) -> PositionManager:
    return PositionManager(state_file=str(tmp_path / "positions.json"))


@pytest.fixture()
def fake_broker() -> FakeBrokerClient:
    return FakeBrokerClient()


@pytest.fixture()
def order_manager(
    tmp_path: Path,
    position_manager: PositionManager,
    rate_limiter: RateLimiter,
    fake_broker: FakeBrokerClient,
) -> OrderManager:
    manager = OrderManager(
        fake_broker,
        position_manager,
        rate_limiter,
        history_path=tmp_path / "history.json",
    )
    manager.POLL_INTERVAL_SEC = 0.05

    class _Resolver:
        def lot_size_for_symbol(self, symbol: str) -> int:
            return 75

    manager.set_instrument_resolver(_Resolver())
    return manager


def test_place_order_tracks_fill(
    order_manager: OrderManager,
    position_manager: PositionManager,
    fake_broker: FakeBrokerClient,
) -> None:
    order_id = order_manager.place_order(
        symbol="NSE:INFY",
        side="BUY",
        quantity=10,
        order_type=OrderType.LIMIT,
        price=1500.0,
    )
    assert order_manager.get_order_status(order_id) == OrderStatus.SUBMITTED

    fake_broker.fill(order_id, quantity=10, price=1501.5)
    assert order_manager.wait_for_fill(order_id, timeout_sec=2)
    assert order_manager.get_order_status(order_id) == OrderStatus.FILLED
    position = position_manager.get_position("NSE:INFY")
    assert position is not None
    assert position.quantity == 10
    assert order_manager.get_fill_price(order_id) == pytest.approx(1501.5)


def test_place_order_emits_submitted_not_filled_on_ack(
    order_manager: OrderManager,
    fake_broker: FakeBrokerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    original_log_event = order_manager._log_trade_event

    def _capture(event_type: str, **kwargs: Any) -> None:
        events.append(event_type)
        original_log_event(event_type, **kwargs)

    monkeypatch.setattr(order_manager, "_log_trade_event", _capture)
    monkeypatch.setattr(order_manager, "_confirm_fill_fast", lambda *_args, **_kwargs: False)

    order_id = order_manager.place_order(
        symbol="NSE:SBIN",
        side="BUY",
        quantity=5,
        order_type=OrderType.LIMIT,
        price=600.0,
        stop_loss=585.0,
        take_profit=630.0,
    )

    assert order_id in fake_broker.orders
    assert "ORDER_SUBMIT_ATTEMPT" in events
    assert "ORDER_SUBMITTED" in events
    assert "ORDER_FILL_CONFIRMED" not in events


def test_place_bracket_order_creates_children(
    order_manager: OrderManager,
    fake_broker: FakeBrokerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_fill(order_manager, fake_broker, 5, 600.0, monkeypatch)
    order_id = order_manager.place_order(
        symbol="NSE:SBIN",
        side="BUY",
        quantity=5,
        order_type=OrderType.LIMIT,
        price=600.0,
        stop_loss=585.0,
        take_profit=630.0,
    )
    history = order_manager.get_order_history()
    entry = next(order for order in history if order.order_id == order_id)
    assert len(entry.child_order_ids) == 2
    for child_id in entry.child_order_ids:
        child = next(order for order in history if order.order_id == child_id)
        assert child.parent_order_id == order_id
        assert child.side == "SELL"


def test_place_order_truncates_quantity_to_lot(
    order_manager: OrderManager, fake_broker: FakeBrokerClient
) -> None:
    order_id = order_manager.place_order(
        symbol="NFO:NIFTY24MAY24000CE",
        side="BUY",
        quantity=160,
        order_type=OrderType.MARKET,
    )
    assert order_id
    payload = fake_broker.orders[order_id]["payload"]
    assert payload["quantity"] <= 160
    assert payload["quantity"] % 75 == 0


def test_resolve_lot_size_supports_get_lot_size_interface(
    order_manager: OrderManager,
) -> None:
    class _GetLotResolver:
        def get_lot_size(self, symbol: str) -> int:
            assert symbol in {"NFO:NIFTY26APR23800PE", "NIFTY26APR23800PE"}
            return 65

    order_manager.set_instrument_resolver(_GetLotResolver())
    assert order_manager.resolve_lot_size("NFO:NIFTY26APR23800PE") == 65


def test_resolve_lot_size_supports_bare_symbol_normalization(
    order_manager: OrderManager,
) -> None:
    class _LotResolver:
        def lot_size_for_symbol(self, symbol: str) -> int | None:
            return 65 if symbol == "NIFTY26APR23800PE" else None

    order_manager.set_instrument_resolver(_LotResolver())
    assert order_manager.resolve_lot_size("NFO:NIFTY26APR23800PE") == 65


def test_resolve_lot_size_supports_lot_size_for_symbol_interface(
    order_manager: OrderManager,
) -> None:
    class _LotResolver:
        def lot_size_for_symbol(self, symbol: str) -> int | None:
            if symbol in {"NFO:NIFTY26APR23800PE", "NIFTY26APR23800PE"}:
                return 65
            return None

    order_manager.set_instrument_resolver(_LotResolver())
    assert order_manager.resolve_lot_size("NFO:NIFTY26APR23800PE") == 65
    assert order_manager.resolve_lot_size("NIFTY26APR23800PE") == 65


def test_place_order_rounding_to_zero_skips(
    order_manager: OrderManager, fake_broker: FakeBrokerClient
) -> None:
    result = order_manager.place_order(
        symbol="NFO:NIFTY24MAY24000CE",
        side="BUY",
        quantity=30,
        order_type=OrderType.MARKET,
    )
    assert result == ""
    assert order_manager.consume_skip_reason() == "lot_rounding_zero"
    assert fake_broker.orders == {}


def test_on_order_update_adopts_filled_manual_order(
    order_manager: OrderManager,
) -> None:
    class _Notifier:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def send_alert(self, message: str) -> None:
            self.messages.append(message)

    class _BracketManager:
        def __init__(self) -> None:
            self.register_calls: list[dict[str, Any]] = []
            self.confirm_calls: list[tuple[str, float]] = []
            self.notifier: Any | None = None

        def set_notifier(self, notifier: Any | None) -> None:
            self.notifier = notifier

        def get_bracket(self, _order_id: str) -> None:
            return None

        def register_virtual_bracket(
            self,
            *,
            order_id: str,
            symbol: str,
            side: str,
            qty: int,
            price: float,
            sl: float,
            tp: float,
            tag: str | None = None,
        ) -> None:
            self.register_calls.append(
                {
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'qty': qty,
                    'price': price,
                    'sl': sl,
                    'tp': tp,
                    'tag': tag,
                }
            )

        def confirm_entry_fill(self, order_id: str, fill_price: float) -> None:
            self.confirm_calls.append((order_id, fill_price))

    notifier = _Notifier()
    bracket_manager = _BracketManager()
    order_manager.set_notifier(notifier)
    order_manager.set_bracket_manager(bracket_manager)

    order_manager.on_order_update(
        {
            'order_id': 'MAN-1',
            'status': 'COMPLETE',
            'filled_quantity': 75,
            'average_price': 100.0,
            'transaction_type': 'BUY',
            'tradingsymbol': 'NFO:NIFTYTEST',
            'price': 100.0,
            'quantity': 75,
        }
    )

    assert bracket_manager.register_calls
    assert bracket_manager.confirm_calls == [('MAN-1', 100.0)]
    register_call = bracket_manager.register_calls[0]
    assert register_call['sl'] == 90.0
    assert register_call['tp'] == 120.0
    assert any('ORDER_ADOPTED' in message for message in notifier.messages)


def test_sell_without_position_triggers_reduce_only(
    order_manager: OrderManager, fake_broker: FakeBrokerClient
) -> None:
    result = order_manager.place_order(
        symbol="NFO:NIFTY24MAY24000CE",
        side="SELL",
        quantity=75,
        order_type=OrderType.MARKET,
    )
    assert result == ""
    assert order_manager.consume_skip_reason() == "reduce_only_violation"
    assert fake_broker.orders == {}


def test_sell_allowed_when_naked_shorts_enabled(
    order_manager: OrderManager,
    fake_broker: FakeBrokerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_settings, "ORDER_ALLOW_NAKED_SHORTS", True)
    order_id = order_manager.place_order(
        symbol="NFO:NIFTY24MAY24000CE",
        side="SELL",
        quantity=75,
        order_type=OrderType.MARKET,
    )
    assert order_id
    assert order_manager.consume_skip_reason() is None


def _patch_entry_fill(
    order_manager: OrderManager,
    fake_broker: FakeBrokerClient,
    quantity: int,
    price: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch ``wait_for_fill`` to immediately mark the entry as filled."""

    def _fill(self: OrderManager, order_id: str, timeout_sec: float = 30.0) -> bool:
        fake_broker.fill(order_id, quantity=quantity, price=price)
        self._update_from_response(
            self._orders[order_id],
            {
                "status": "complete",
                "filled_quantity": quantity,
                "average_price": price,
            },
        )
        return True

    monkeypatch.setattr(
        order_manager, "wait_for_fill", types.MethodType(_fill, order_manager)
    )


def test_execute_bracket_trade_partial_profit(
    order_manager: OrderManager,
    fake_broker: FakeBrokerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_fill(order_manager, fake_broker, 4, 600.0, monkeypatch)
    entry_id, stop_id, tp1_id = order_manager.execute_bracket_trade(
        symbol="NSE:SBIN",
        side="BUY",
        quantity=4,
        entry_price=600.0,
        stop_loss=590.0,
        take_profit=610.0,
        partial_profit_fraction=0.5,
        second_target_price=620.0,
    )
    assert entry_id
    assert stop_id
    assert tp1_id

    state = order_manager._brackets[entry_id]
    assert state.tp_secondary_id is None
    assert state.tp_secondary_qty == 2

    fake_broker.fill(tp1_id, quantity=2, price=610.0)
    order_manager._update_from_response(
        order_manager._orders[tp1_id],
        {"status": "complete", "filled_quantity": 2, "average_price": 610.0},
    )

    state_after = order_manager._brackets[entry_id]
    assert state_after.tp_secondary_id is not None
    second_id = state_after.tp_secondary_id
    assert fake_broker.orders[second_id]["payload"]["quantity"] == 2
    assert order_manager._orders[state_after.stop_order_id].quantity == 2

    fake_broker.fill(second_id, quantity=2, price=620.0)
    order_manager._update_from_response(
        order_manager._orders[second_id],
        {"status": "complete", "filled_quantity": 2, "average_price": 620.0},
    )

    assert entry_id not in order_manager._brackets
    stop_snapshot = fake_broker.orders[stop_id]
    assert stop_snapshot["status"] == "cancelled"


def test_bracket_stop_fill_cancels_targets(
    order_manager: OrderManager,
    fake_broker: FakeBrokerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_fill(order_manager, fake_broker, 3, 500.0, monkeypatch)
    entry_id, stop_id, tp_id = order_manager.execute_bracket_trade(
        symbol="NSE:ITC",
        side="BUY",
        quantity=3,
        entry_price=500.0,
        stop_loss=490.0,
        take_profit=510.0,
    )

    assert entry_id
    assert stop_id
    assert tp_id

    fake_broker.fill(stop_id, quantity=3, price=490.0)
    order_manager._update_from_response(
        order_manager._orders[stop_id],
        {"status": "complete", "filled_quantity": 3, "average_price": 490.0},
    )

    assert entry_id not in order_manager._brackets
    assert fake_broker.orders[tp_id]["status"] == "cancelled"


def test_bracket_partial_tp_resizes_stop(
    order_manager: OrderManager,
    fake_broker: FakeBrokerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_fill(order_manager, fake_broker, 6, 200.0, monkeypatch)
    entry_id, stop_id, tp_id = order_manager.execute_bracket_trade(
        symbol="NSE:HDFCBANK",
        side="BUY",
        quantity=6,
        entry_price=200.0,
        stop_loss=190.0,
        take_profit=205.0,
    )

    assert entry_id
    assert stop_id
    assert tp_id

    order_manager._update_from_response(
        order_manager._orders[tp_id],
        {
            "status": "partial",
            "filled_quantity": 2,
            "average_price": 205.0,
        },
    )

    stop_details = order_manager._orders[stop_id]
    assert stop_details.quantity == 4


def test_execute_bracket_trade_uses_margin_inputs(
    order_manager: OrderManager,
    fake_broker: FakeBrokerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float | None] = {}

    def fake_pretrade(
        self: OrderManager,
        *,
        symbol: str,
        side: str,
        quantity: int,
        product: str | None,
        price: float | None = None,
        stop_loss: float | None = None,
        atr: float | None = None,
    ) -> tuple[MarginDecision, dict[str, object]]:
        captured["price"] = price
        captured["stop_loss"] = stop_loss
        decision = MarginDecision(
            ok=True,
            reason="",
            order_type="NRML",
            quantity=quantity,
            est_required=0.0,
            available=1_000_000.0,
        )
        return decision, {}

    monkeypatch.setattr(
        order_manager,
        "_pre_trade_decision",
        types.MethodType(fake_pretrade, order_manager),
    )
    _patch_entry_fill(order_manager, fake_broker, 2, 150.0, monkeypatch)

    order_manager.execute_bracket_trade(
        symbol="NSE:AXISBANK",
        side="BUY",
        quantity=2,
        entry_price=150.0,
        stop_loss=140.0,
        take_profit=160.0,
    )

    assert captured["price"] == pytest.approx(150.0)
    assert captured["stop_loss"] == pytest.approx(140.0)


def test_execute_bracket_trade_blocks_on_margin(
    order_manager: OrderManager,
    fake_broker: FakeBrokerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_margin(
        self: OrderManager,
        symbol: str,
        side: str,
        quantity: int,
        product: str | None,
        *,
        dry_run: bool = False,
        price: float | None = None,
        stop_loss: float | None = None,
    ) -> tuple[bool, str, dict[str, object]]:
        return False, "insufficient_margin", {"needed": 1000.0, "available": 100.0}

    monkeypatch.setattr(
        order_manager,
        "_precheck_margin",
        types.MethodType(fake_margin, order_manager),
    )

    entry_id, stop_id, tp_id = order_manager.execute_bracket_trade(
        symbol="NSE:RELIANCE",
        side="BUY",
        quantity=75,
        entry_price=2500.0,
        stop_loss=2450.0,
        take_profit=2550.0,
    )

    assert (entry_id, stop_id, tp_id) == ("", "", "")
    assert fake_broker.orders == {}


def test_non_retryable_rejection_does_not_retry(
    rate_limiter: RateLimiter,
    position_manager: PositionManager,
    tmp_path: Path,
) -> None:
    broker = RejectingBroker("Insufficient funds available")
    manager = OrderManager(
        broker,
        position_manager,
        rate_limiter,
        history_path=tmp_path / "history.json",
    )
    with pytest.raises(OrderPlacementError) as exc_info:
        manager.place_order(
            symbol="NSE:TCS", side="BUY", quantity=1, order_type=OrderType.MARKET
        )
    assert "MARGIN" in str(exc_info.value).upper()
    assert broker.attempts == 1


def test_precheck_margin_blocks_when_available_low(
    order_manager: OrderManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    order_manager._margin_factor = 1.0
    order_manager._margin_buffer = 1.0

    def fake_pretrade(
        self: OrderManager,
        *,
        symbol: str,
        side: str,
        quantity: int,
        product: str | None,
        price: float | None = None,
        stop_loss: float | None = None,
        atr: float | None = None,
    ) -> tuple[MarginDecision, dict[str, object]]:
        decision = MarginDecision(
            ok=False,
            reason="MARGIN insufficient",
            order_type="NRML",
            quantity=0,
            est_required=100.0,
            available=50.0,
        )
        return decision, {"available_source": "stub"}

    monkeypatch.setattr(
        order_manager,
        "_pre_trade_decision",
        types.MethodType(fake_pretrade, order_manager),
    )

    ok, reason, meta = order_manager._precheck_margin(
        "NSE:NIFTY",
        "BUY",
        1,
        None,
        dry_run=True,
    )

    assert not ok
    assert reason.startswith("MARGIN")
    assert pytest.approx(meta.get("needed"), rel=1e-3) == 100.0
    assert pytest.approx(meta.get("available"), rel=1e-3) == 50.0
    assert meta.get("available_source") == "stub"


def test_order_blocked_when_guard_disallows(order_manager: OrderManager) -> None:
    order_manager.set_session_guard_getter(
        lambda: {
            "session_valid": False,
            "rate_limits_ok": True,
            "risk_green": True,
            "market_open": False,
            "override_out_of_hours": False,
            "broker_session_valid": True,
            "reasons": ["Outside trading window"],
        }
    )
    order_manager.set_trade_mode_getters(
        enable_live=lambda: True,
        shadow_mode=lambda: False,
    )
    with pytest.raises(OrderPlacementError):
        order_manager.place_order(
            symbol="NSE:ITC", side="BUY", quantity=1, order_type=OrderType.MARKET
        )


def test_order_blocked_when_trading_paused(order_manager: OrderManager) -> None:
    switch = trading_switch()
    switch.pause()
    with pytest.raises(OrderPlacementError):
        order_manager.place_order(
            symbol="NSE:SBIN",
            side="BUY",
            quantity=1,
        )
    switch.resume()


def test_soft_risk_reason_does_not_raise(order_manager: OrderManager, caplog) -> None:
    class _SoftRisk:
        def __init__(self) -> None:
            self.calls = 0

        def risk_gate_should_trade(self) -> tuple[bool, tuple[str, ...]]:
            self.calls += 1
            return False, ("STALE",)

        def cooldown_remaining(self) -> float:
            return 0.0

    risk = _SoftRisk()
    order_manager.set_risk_manager(risk)

    with caplog.at_level("INFO"):
        allowed = order_manager._ensure_trading_allowed(
            symbol="NIFTY", side="BUY", quantity=1
        )

    assert risk.calls == 1
    assert allowed is False
    assert any("order_blocked_soft" in record.message for record in caplog.records)


def test_soft_risk_block_reports_freshness(order_manager: OrderManager, caplog) -> None:
    class _SoftRisk:
        def risk_gate_should_trade(self) -> tuple[bool, tuple[str, ...]]:
            return False, ("STALE",)

        def cooldown_remaining(self) -> float:
            return 12.34

    class _Hub:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def is_fresh(self, symbol: str) -> tuple[bool, dict[str, float | str | None]]:
            self.calls.append(symbol)
            return (
                False,
                {
                    "effective_ms": 1234.0,
                    "mono_age_ms": 1200.0,
                    "server_age_ms": 1300.0,
                    "threshold_ms": 2000.0,
                    "reason": "stale",
                },
            )

    hub = _Hub()
    risk = _SoftRisk()
    order_manager.attach_data_hub(hub)
    order_manager.set_risk_manager(risk)

    with caplog.at_level("INFO"):
        allowed = order_manager._ensure_trading_allowed(
            symbol="NIFTY", side="BUY", quantity=1
        )

    assert allowed is False
    assert hub.calls == ["NIFTY"]
    record = next(
        entry for entry in caplog.records if "order_blocked_soft" in entry.message
    )
    assert record.tick_fresh is False
    assert record.tick_reason == "stale"
    assert record.tick_threshold_ms == 2000.0
    assert record.tick_age_ms == 1234.0
    assert record.tick_mono_age_ms == 1200.0
    assert record.tick_server_age_ms == 1300.0


def test_circuit_breaker_blocks_after_repeated_failures(
    rate_limiter: RateLimiter,
    position_manager: PositionManager,
    tmp_path: Path,
) -> None:
    broker = FailingBroker()
    manager = OrderManager(
        broker,
        position_manager,
        rate_limiter,
        history_path=tmp_path / "history.json",
    )
    manager._broker_circuit = CircuitBreaker(
        fail_threshold=1,
        window_s=10.0,
        cooldown_s=5.0,
        clock=lambda: 0.0,
    )
    with pytest.raises(RateLimitError):
        manager.place_order(
            symbol="NSE:HDFCBANK",
            side="BUY",
            quantity=1,
            order_type=OrderType.MARKET,
        )
    assert broker.calls == 1
    assert manager._broker_circuit.is_open


def test_order_blocked_when_shadow_mode(order_manager: OrderManager) -> None:
    order_manager.set_session_guard_getter(
        lambda: {
            "session_valid": True,
            "rate_limits_ok": True,
            "risk_green": True,
            "market_open": True,
            "override_out_of_hours": False,
            "broker_session_valid": True,
            "reasons": [],
        }
    )
    order_manager.set_trade_mode_getters(
        enable_live=lambda: True,
        shadow_mode=lambda: True,
    )
    with pytest.raises(OrderPlacementError):
        order_manager.place_order(
            symbol="NSE:HDFCBANK", side="BUY", quantity=1, order_type=OrderType.MARKET
        )


def test_exit_allowed_when_guard_blocks(
    order_manager: OrderManager,
    position_manager: PositionManager,
    fake_broker: FakeBrokerClient,
) -> None:
    position_manager.open_position(
        symbol="NSE:INFY",
        side="LONG",
        quantity=10,
        entry_price=1500.0,
    )
    order_manager.set_session_guard_getter(
        lambda: {
            "session_valid": False,
            "rate_limits_ok": False,
            "risk_green": False,
            "market_open": False,
            "override_out_of_hours": False,
            "broker_session_valid": False,
            "reasons": ["BREAKER"],
        }
    )
    order_manager.set_trade_mode_getters(
        enable_live=lambda: False,
        shadow_mode=lambda: True,
    )

    order_id = order_manager.place_order(
        symbol="NSE:INFY",
        side="SELL",
        quantity=10,
        order_type=OrderType.MARKET,
    )

    assert order_id in fake_broker.orders


def test_force_exit_infers_side_from_net_quantity(
    order_manager: OrderManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NetOnlyPosition:
        def __init__(self, net_quantity: int) -> None:
            self.net_quantity = net_quantity
            self.quantity = 0

    monkeypatch.setattr(
        order_manager._positions,  # type: ignore[attr-defined]
        "get_position",
        lambda symbol: NetOnlyPosition(-15) if symbol == "NSE:SBIN" else None,
    )

    assert order_manager._is_force_exit(symbol="NSE:SBIN", side="BUY", quantity=10)
    assert not order_manager._is_force_exit(symbol="NSE:SBIN", side="SELL", quantity=10)
    assert not order_manager._is_force_exit(symbol="NSE:SBIN", side="BUY", quantity=20)


def test_order_payload_omits_tradingsymbol_for_plain_symbol(
    order_manager: OrderManager,
    fake_broker: FakeBrokerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSTRUMENTS__TRADE_EXCHANGE", "NFO")

    order_id = order_manager.place_order(
        symbol="NIFTY",
        side="BUY",
        quantity=1,
        order_type=OrderType.MARKET,
        price=None,
    )

    payload = fake_broker.orders[order_id]["payload"]
    assert payload.get("exchange") == "NFO"
    assert "tradingsymbol" not in payload


def test_order_payload_includes_tradingsymbol_for_explicit_symbol(
    order_manager: OrderManager,
    fake_broker: FakeBrokerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSTRUMENTS__TRADE_EXCHANGE", "NFO")

    order_id = order_manager.place_order(
        symbol="NFO:NIFTY25OCT19500CE",
        side="BUY",
        quantity=75,
        order_type=OrderType.MARKET,
        price=None,
    )

    payload = fake_broker.orders[order_id]["payload"]
    assert payload.get("tradingsymbol") == "NIFTY25OCT19500CE"


def test_order_payload_includes_tradingsymbol_for_derivative_suffix(
    order_manager: OrderManager,
    fake_broker: FakeBrokerClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSTRUMENTS__TRADE_EXCHANGE", "NFO")

    order_id = order_manager.place_order(
        symbol="NIFTY25OCT19500CE",
        side="BUY",
        quantity=75,
        order_type=OrderType.MARKET,
        price=None,
    )

    payload = fake_broker.orders[order_id]["payload"]
    assert payload.get("tradingsymbol") == "NIFTY25OCT19500CE"
