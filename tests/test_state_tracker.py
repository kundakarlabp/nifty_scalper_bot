from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Generator

import pytest

from nifty_scalper_bot.execution.state_tracker import StateTracker, StateTrackerSettings

FIXED_TIME = 1_700_000_000.0


@pytest.fixture(autouse=True)
def _freeze_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: FIXED_TIME)


@pytest.fixture()
def temp_db(tmp_path: Path) -> str:
    db_path = tmp_path / "state.db"
    return str(db_path)


@pytest.fixture()
def state_tracker(temp_db: str) -> Generator[StateTracker, None, None]:
    tracker = StateTracker(StateTrackerSettings(db_path=temp_db))
    try:
        yield tracker
    finally:
        tracker.close()


def test_position_state_crud(state_tracker: StateTracker) -> None:
    state_tracker.update_position(
        "NIFTY",
        {
            "quantity": 5,
            "entry_price": 201.5,
            "entry_time": FIXED_TIME,
            "unrealized_pnl": 10.0,
            "stop_loss": 190.0,
            "tp1": 210.0,
            "tp2": 220.0,
            "trail_active": True,
            "lifecycle_stage": "ACTIVE",
        },
    )
    snapshot = state_tracker.get_position_state("NIFTY")
    assert snapshot is not None
    assert snapshot["quantity"] == 5
    state_tracker.update_position("NIFTY", {"quantity": 2, "unrealized_pnl": 5.0})
    snapshot = state_tracker.get_position_state("NIFTY")
    assert snapshot is not None
    assert snapshot["quantity"] == 2
    state_tracker.update_position("NIFTY", {"delete": True})
    assert state_tracker.get_position_state("NIFTY") is None


def test_order_insertion_and_retrieval(state_tracker: StateTracker) -> None:
    order_id = state_tracker.add_order(
        {
            "order_id": "order-1",
            "symbol": "NIFTY",
            "side": "BUY",
            "quantity": 10,
            "status": "OPEN",
            "intent": "ENTRY",
        }
    )
    assert order_id == "order-1"
    state_tracker.add_order(
        {
            "order_id": "order-2",
            "symbol": "NIFTY",
            "side": "BUY",
            "quantity": 10,
            "status": "FILLED",
            "intent": "ENTRY",
        }
    )
    open_orders = state_tracker.get_orders_by_status("OPEN")
    assert len(open_orders) == 1
    assert open_orders[0]["order_id"] == "order-1"


def test_lifecycle_event_logging(state_tracker: StateTracker) -> None:
    state_tracker.record_lifecycle_event(
        "NIFTY",
        "TP1_TAKEN",
        {"pnl": 1000.0},
        timestamp=FIXED_TIME,
    )
    events = state_tracker.list_lifecycle_events("NIFTY")
    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "TP1_TAKEN"
    assert event["details_json"]["pnl"] == 1000.0


def test_reconciliation_broker_drift(state_tracker: StateTracker) -> None:
    state_tracker.update_position(
        "NIFTY",
        {
            "quantity": 5,
            "entry_price": 210.0,
            "entry_time": FIXED_TIME,
            "unrealized_pnl": 50.0,
            "stop_loss": 180.0,
            "tp1": 220.0,
            "tp2": 230.0,
            "trail_active": True,
            "lifecycle_stage": "ACTIVE",
        },
    )
    state_tracker.reconcile_with_broker(
        [
            {
                "symbol": "NIFTY",
                "quantity": 0,
            },
            {
                "symbol": "BANKNIFTY",
                "quantity": 3,
                "entry_price": 400.0,
                "entry_time": FIXED_TIME,
                "unrealized_pnl": 5.0,
                "stop_loss": 350.0,
                "tp1": 420.0,
                "tp2": 450.0,
                "trail_active": False,
                "lifecycle_stage": "NEW",
            },
        ]
    )
    assert state_tracker.get_position_state("NIFTY") is None
    bank = state_tracker.get_position_state("BANKNIFTY")
    assert bank is not None
    assert bank["quantity"] == 3


def test_thread_safety(state_tracker: StateTracker) -> None:
    barrier = threading.Barrier(5)
    errors: list[Exception] = []

    def _worker(symbol: str, qty: int) -> None:
        try:
            barrier.wait()
            state_tracker.update_position(symbol, {"quantity": qty})
        except Exception as exc:  # pragma: no cover - diagnostic
            errors.append(exc)

    threads = [
        threading.Thread(target=_worker, args=("NIFTY", idx)) for idx in range(5)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    snapshot = state_tracker.get_position_state("NIFTY")
    assert snapshot is not None
    assert snapshot["quantity"] in {0, 1, 2, 3, 4}
