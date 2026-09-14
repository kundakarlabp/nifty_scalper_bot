from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import nifty_scalper_bot.execution  # noqa: F401 - install remaining compatibility overlays
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.execution.position_reconciliation_identity import (
    _prepare_broker_positions,
)

SYMBOL = "NFO:NIFTY24JAN100CE"


def test_native_position_manager_owns_reconciliation_identity_contract() -> None:
    manager_source = Path(
        "src/nifty_scalper_bot/execution/position_manager.py"
    ).read_text(encoding="utf-8")
    overlay_source = Path(
        "src/nifty_scalper_bot/execution/position_identity_extension.py"
    ).read_text(encoding="utf-8")

    assert "def _reconcile_positions_from_broker" in manager_source
    assert "self._single_reconcile_lock" in manager_source
    assert "_prepare_broker_positions(self, broker_positions)" in manager_source
    assert "cls.reconcile_now = reconcile_now" not in overlay_source
    assert "cls.synchronize_with_broker = synchronize_with_broker" not in overlay_source


def test_native_sync_preserves_bot_owned_fill_basis_and_lifecycle(tmp_path) -> None:
    manager = PositionManager(str(tmp_path / "positions.json"))
    manager.open_position(
        SYMBOL,
        "LONG",
        65,
        100.0,
        stop_loss=90.0,
        take_profit=120.0,
        order_id="ENTRY-1",
    )

    manager.synchronize_with_broker(
        [
            {
                "tradingsymbol": "NIFTY24JAN100CE",
                "quantity": 65,
                "average_price": 117.5,
                "last_price": 118.0,
                "product": "MIS",
            }
        ]
    )

    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.entry_price == 100.0
    assert position.current_price == 118.0
    assert position.order_id == "ENTRY-1"
    assert position.stop_loss == 90.0
    assert position.take_profit == 120.0


def test_native_single_flight_coalescing_contract_survives_outer_order_ledger(
    tmp_path,
) -> None:
    manager = PositionManager(str(tmp_path / "positions.json"))
    lock = manager._single_reconcile_lock
    lock.acquire()
    try:
        assert manager.reconcile_now() is False
    finally:
        lock.release()

    assert manager._single_reconcile_coalesced == 1


def _local_position(*, side: str, quantity: int, entry_price: float = 100.0):
    return SimpleNamespace(
        symbol=SYMBOL,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        order_id="ENTRY-1",
    )


def _prepare_missing_average(*, side: str, local_qty: int, broker_qty: int):
    manager = SimpleNamespace(
        _positions={SYMBOL: _local_position(side=side, quantity=local_qty)}
    )
    return _prepare_broker_positions(
        manager,
        [
            {
                "tradingsymbol": "NIFTY24JAN100CE",
                "quantity": broker_qty,
                "average_price": 0,
                "last_price": 110.0,
                "product": "MIS",
            }
        ],
    )


def test_missing_average_same_side_reduction_reuses_owned_basis() -> None:
    prepared, unresolved = _prepare_missing_average(
        side="LONG",
        local_qty=65,
        broker_qty=30,
    )

    assert unresolved == set()
    assert prepared[0]["average_price"] == 100.0


def test_missing_average_same_side_scale_up_is_unresolved() -> None:
    prepared, unresolved = _prepare_missing_average(
        side="LONG",
        local_qty=65,
        broker_qty=130,
    )

    assert unresolved == {SYMBOL}
    assert prepared[0]["average_price"] == 0


def test_missing_average_side_reversal_is_unresolved() -> None:
    prepared, unresolved = _prepare_missing_average(
        side="LONG",
        local_qty=65,
        broker_qty=-65,
    )

    assert unresolved == {SYMBOL}
    assert prepared[0]["average_price"] == 0


def test_missing_average_short_reduction_reuses_owned_basis() -> None:
    prepared, unresolved = _prepare_missing_average(
        side="SHORT",
        local_qty=65,
        broker_qty=-30,
    )

    assert unresolved == set()
    assert prepared[0]["average_price"] == 100.0


def test_missing_average_short_scale_up_is_unresolved() -> None:
    prepared, unresolved = _prepare_missing_average(
        side="SHORT",
        local_qty=65,
        broker_qty=-130,
    )

    assert unresolved == {SYMBOL}
    assert prepared[0]["average_price"] == 0


def test_missing_average_short_to_long_reversal_is_unresolved() -> None:
    prepared, unresolved = _prepare_missing_average(
        side="SHORT",
        local_qty=65,
        broker_qty=65,
    )

    assert unresolved == {SYMBOL}
    assert prepared[0]["average_price"] == 0
