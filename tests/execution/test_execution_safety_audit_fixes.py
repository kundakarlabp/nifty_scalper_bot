from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution import bracket_core
from nifty_scalper_bot.execution.bracket_core import BracketManager, BracketState
from nifty_scalper_bot.execution.execution_policy import ExecutionPolicy
from nifty_scalper_bot.execution.margin_engine import MarginInputs
from nifty_scalper_bot.execution.position_manager import Order, Position, PositionManager
from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.utils.errors import BrokerError
from nifty_scalper_bot.utils.errors import OrderPlacementError


SYMBOL = "NFO:NIFTY2662324050PE"


def _position() -> Position:
    return Position(
        symbol=SYMBOL,
        side="LONG",
        quantity=65,
        entry_price=100.0,
        entry_time=datetime.now(timezone.utc),
        current_price=101.0,
    )


def _position_manager(tmp_path) -> PositionManager:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager._schedule_retry_after_failure = lambda *_args, **_kwargs: None
    manager._positions[SYMBOL] = _position()
    return manager


def test_none_broker_snapshot_fails_closed_and_preserves_position(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    manager.set_broker_client(SimpleNamespace(get_positions=lambda: None))

    assert manager.reconcile_now() is False
    assert manager.get_position(SYMBOL) is not None


def test_malformed_broker_snapshot_fails_closed_and_preserves_position(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    manager.set_broker_client(
        SimpleNamespace(
            get_positions=lambda: [
                {
                    "symbol": SYMBOL,
                    "product": "MIS",
                    "quantity": "not-a-number",
                    "average_price": 100.0,
                    "last_price": 101.0,
                }
            ]
        )
    )

    assert manager.reconcile_now() is False
    assert manager.get_position(SYMBOL) is not None


def test_explicit_empty_snapshot_is_authoritative_flat(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    flattened: list[list[str]] = []
    manager.set_on_symbols_flat(lambda symbols: flattened.append(list(symbols)))
    manager.set_broker_client(SimpleNamespace(get_positions=lambda: []))

    assert manager.reconcile_now() is True
    assert manager.get_position(SYMBOL) is None
    assert flattened == [[SYMBOL]]


def test_broker_realised_field_updates_daily_realised_without_using_total_pnl(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    manager.synchronize_with_broker(
        [
            {
                "symbol": SYMBOL,
                "product": "MIS",
                "quantity": 0,
                "realised": -125.5,
                "pnl": 9999.0,
                "m2m": 9999.0,
            }
        ]
    )
    assert manager.get_realized_pnl() == pytest.approx(-125.5)


def test_update_from_order_uses_fill_price_and_existing_fill_lifecycle(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    order = Order(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        order_type="MARKET",
        quantity=65,
        price=100.0,
        status="FILLED",
        filled_quantity=65,
        fill_price=101.0,
    )
    manager.update_from_order(order)
    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.entry_price == pytest.approx(101.0)
    assert position.quantity == 65


def _stop(manager: BracketManager) -> None:
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)


def test_direct_long_bracket_registration_normalizes_and_triggers_sl(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="LONG",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert bracket.side == "BUY"
    action = manager._evaluate_exit_fast(bracket, 89.0)
    assert action is not None
    assert action["type"] == "SL"


def test_bracket_state_is_written_and_restored_with_ledger_recovery_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    bracket._ledger_pending_exit_order_id = "exit-1"
    bracket._ledger_pending_exit_quantity = 65
    bracket._ledger_pending_exit_price = 89.5
    manager.save_state()

    restored = BracketManager(order_manager=SimpleNamespace())
    _stop(restored)
    restored.load_state()
    restored_bracket = restored.get_bracket("entry-1")
    assert restored_bracket is not None
    assert restored_bracket._ledger_pending_exit_order_id == "exit-1"
    assert restored_bracket._ledger_pending_exit_quantity == 65
    assert restored_bracket._ledger_pending_exit_price == pytest.approx(89.5)


def test_confirmed_fill_remains_active_when_snapshot_persistence_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
    )
    monkeypatch.setattr(manager, "save_state", lambda: (_ for _ in ()).throw(OSError("disk")))
    manager.confirm_entry_fill("entry-1", 101.0)
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert bracket.active is True
    assert bracket.entry_confirmed is True


def test_metrics_failure_does_not_undo_registered_protection(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)

    class _Counter:
        def inc(self) -> None:
            raise RuntimeError("metrics down")

    monkeypatch.setattr(bracket_core, "METRICS_AVAILABLE", True)
    monkeypatch.setattr(bracket_core, "METRICS", SimpleNamespace(brackets_created=_Counter()))
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        activate_immediately=True,
    )
    assert manager.get_bracket("entry-1") is not None


def test_margin_inputs_are_immutable() -> None:
    inputs = MarginInputs(
        symbol=SYMBOL,
        side="BUY",
        price=100.0,
        stop_loss=90.0,
        atr=5.0,
        requested_qty=65,
        product="MIS",
        lot_size=65,
        balance=100000.0,
        per_trade_risk_pct=1.0,
        per_trade_cap_pct=10.0,
        margin_factor=1.0,
        margin_buffer=0.95,
        contract_multiplier=1.0,
        ist_now=datetime.now(timezone.utc),
        min_lots_per_trade=1,
        max_lots_per_trade=2,
        atr_multiple=1.5,
    )
    with pytest.raises(FrozenInstanceError):
        inputs.balance = 1.0


class _Hub:
    def __init__(self, quote):
        self.quote = quote

    def get_quote(self, symbol: str, allow_pull: bool = True):
        return dict(self.quote)


def test_zero_spread_limit_is_strict_and_none_explicitly_disables_guard() -> None:
    quote = {"best_bid": 100.0, "best_ask": 101.0}
    with pytest.raises(OrderPlacementError):
        ExecutionPolicy(_Hub(quote), max_spread_pct=0.0).build_plan("NSE:NIFTY", "BUY")
    plan = ExecutionPolicy(_Hub(quote), max_spread_pct=None).build_plan("NSE:NIFTY", "BUY")
    assert plan.spread_pct > 0


def test_invalid_later_row_does_not_partially_mutate_existing_position(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    original = manager.get_position(SYMBOL)
    assert original is not None
    original_price = original.current_price
    with pytest.raises(ValueError):
        manager.synchronize_with_broker(
            [
                {
                    "symbol": SYMBOL,
                    "product": "MIS",
                    "quantity": 65,
                    "average_price": 100.0,
                    "last_price": 150.0,
                },
                {
                    "symbol": "NFO:NIFTY2662324000CE",
                    "product": "MIS",
                    "quantity": "invalid",
                    "average_price": 100.0,
                    "last_price": 101.0,
                },
            ]
        )
    preserved = manager.get_position(SYMBOL)
    assert preserved is not None
    assert preserved.current_price == pytest.approx(original_price)


def _zerodha_positions_client(response):
    client = object.__new__(ZerodhaKiteClient)
    client._GENERAL_BUCKET = "general"
    client._positions_cache = None
    client._log_time_fn = lambda: 0.0
    client._acquire_bucket = lambda *_args, **_kwargs: None
    client._make_request = lambda *_args, **_kwargs: response
    client._ensure_json = lambda payload: payload
    client._build_retry_handlers = lambda **_kwargs: (lambda *_args, **_kw: False, None)
    client._execute_with_retry = lambda **kwargs: kwargs["operation"]()
    client._load_rest_cache = lambda *_args, **_kwargs: None
    return client


def test_zerodha_missing_net_snapshot_raises_instead_of_returning_flat() -> None:
    client = _zerodha_positions_client({"status": "success", "data": {}})
    with pytest.raises(BrokerError):
        client.get_positions()


def test_zerodha_authoritative_empty_net_does_not_fall_back_to_day_rows() -> None:
    client = _zerodha_positions_client(
        {
            "status": "success",
            "data": {
                "net": [],
                "day": [{"symbol": SYMBOL, "quantity": 65}],
            },
        }
    )
    assert client.get_positions() == []


def test_flat_verification_rejects_missing_and_malformed_snapshots() -> None:
    for response in (None, [None], [{"symbol": SYMBOL}], [{"quantity": 0}]):
        broker = SimpleNamespace(get_positions=lambda response=response: response)
        manager = BracketManager(order_manager=SimpleNamespace(_broker=broker))
        _stop(manager)
        assert manager._verify_position_closed(SYMBOL) is False


def test_flat_verification_accepts_only_valid_explicit_flat_snapshot() -> None:
    broker = SimpleNamespace(
        get_positions=lambda: [
            {"symbol": SYMBOL, "quantity": 0},
            {"symbol": "NFO:NIFTY2662324000CE", "quantity": 65},
        ]
    )
    manager = BracketManager(order_manager=SimpleNamespace(_broker=broker))
    _stop(manager)
    assert manager._verify_position_closed(SYMBOL) is True

    broker.get_positions = lambda: [{"symbol": SYMBOL, "quantity": 65}]
    assert manager._verify_position_closed(SYMBOL) is False


def test_duplicate_managed_position_rows_reject_snapshot_atomically(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    original = manager.get_position(SYMBOL)
    assert original is not None
    original_price = original.current_price
    duplicate = {
        "symbol": SYMBOL,
        "product": "MIS",
        "quantity": 65,
        "average_price": 100.0,
        "last_price": 150.0,
    }
    with pytest.raises(ValueError, match="duplicate broker position"):
        manager.synchronize_with_broker([duplicate, dict(duplicate)])
    preserved = manager.get_position(SYMBOL)
    assert preserved is not None
    assert preserved.current_price == pytest.approx(original_price)
