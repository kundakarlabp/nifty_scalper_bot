from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from threading import RLock
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.order_manager import (
    OrderManager,
    OrderStatus,
    OrderType,
)
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.utils.rate_limiter import RateLimiter


SYMBOL = "NFO:NIFTY2671423950CE"


class _Broker:
    is_simulated_adapter = True

    def __init__(self) -> None:
        self.calls = 0
        self.payloads: list[dict] = []

    def place_order(self, **kwargs):
        self.calls += 1
        self.payloads.append(dict(kwargs))
        return {"order_id": "EXIT-1", "status": "SUBMITTED"}


def _manager(monkeypatch, tmp_path) -> tuple[OrderManager, _Broker]:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = _Broker()
    manager = OrderManager(
        broker_client=broker,
        position_manager=PositionManager(str(tmp_path / "positions.json")),
        rate_limiter=RateLimiter(),
    )
    manager._positions.open_position(
        SYMBOL, "LONG", 65, 100.0, order_id="entry-1"
    )
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)
    manager._kill_switch_engaged_at = datetime.now(timezone.utc)
    manager._kill_switch_allow_auto_reset = False
    manager._kill_switch_reason = "test_entry_kill_switch"
    manager._consecutive_failures = 3
    return manager, broker


@pytest.mark.parametrize("intent", ["EXIT", "REDUCE"])
def test_explicit_exit_intent_bypasses_entry_kill_switch_without_tag(
    monkeypatch, tmp_path, intent
) -> None:
    """Reducing orders must not depend on tag text to bypass entry-only gates."""
    manager, broker = _manager(monkeypatch, tmp_path)

    order_id = manager.place_order(
        symbol=SYMBOL,
        side="SELL",
        quantity=65,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent=intent,
        tag=None,
    )

    assert order_id == "EXIT-1"
    assert broker.calls == 1


def test_protective_exit_does_not_create_entry_trade_identity(
    monkeypatch, tmp_path
) -> None:
    """A protective exit belongs to the existing trade, not a new manual trade."""
    manager, broker = _manager(monkeypatch, tmp_path)
    events: list[dict] = []
    manager._trade_journal = SimpleNamespace(
        log_event=lambda payload: events.append(dict(payload))
    )

    order_id = manager.place_order(
        symbol=SYMBOL,
        side="SELL",
        quantity=65,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent="EXIT",
        bracket_id="entry-1",
        linked_entry_order_id="entry-1",
        trade_lifecycle_id="entry-1",
        tag="EXIT_HARD_SL",
    )

    assert order_id == "EXIT-1"
    assert broker.calls == 1
    assert manager._orders["EXIT-1"].signal_id is None
    assert not any(
        event["event_type"] in {"ORDER_SUBMIT_ATTEMPT", "ORDER_SUBMITTED"}
        for event in events
    )


def test_reversal_remains_subject_to_entry_kill_switch(monkeypatch, tmp_path) -> None:
    """REVERSAL creates new exposure and must not inherit protective-exit bypass."""
    manager, broker = _manager(monkeypatch, tmp_path)

    order_id = manager.place_order(
        symbol=SYMBOL,
        side="BUY",
        quantity=65,
        order_type=OrderType.LIMIT,
        price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        check_risk=False,
        intent="REVERSAL",
        tag=None,
    )

    assert order_id is None
    assert broker.calls == 0
    assert manager._last_order_decision["block_reason"] == "kill_switch_active"


@pytest.mark.parametrize(
    "intent,order_type,linked_entry_id,should_cancel",
    [
        ("ENTRY", OrderType.LIMIT, None, True),
        ("SCALE_IN", OrderType.LIMIT, None, True),
        ("REVERSAL", OrderType.LIMIT, None, True),
        ("EXIT", OrderType.STOP_LOSS, "entry-1", False),
        ("REDUCE", OrderType.MARKET, "entry-1", False),
        ("UNKNOWN", OrderType.LIMIT, None, False),
        ("ENTRY", OrderType.STOP_LOSS, None, False),
        ("ENTRY", "SL", None, False),
        ("ENTRY", OrderType.LIMIT, "entry-1", False),
    ],
)
def test_zombie_cleanup_never_cancels_protective_orders(
    intent, order_type, linked_entry_id, should_cancel
) -> None:
    """An aged exchange stop cannot be treated as a stale entry order."""
    cancelled: list[str] = []
    order = SimpleNamespace(
        intent=intent,
        order_type=order_type,
        linked_entry_order_id=linked_entry_id,
        status=OrderStatus.PENDING,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=60),
    )
    manager = SimpleNamespace(
        _lock=RLock(),
        _orders={"aged": order},
        _logger=logging.getLogger(__name__),
        cancel_order=cancelled.append,
    )

    OrderManager._check_zombie_orders(manager)

    assert cancelled == (["aged"] if should_cancel else [])


def test_zombie_cleanup_retains_recent_entry_order() -> None:
    cancelled: list[str] = []
    order = SimpleNamespace(
        intent="ENTRY",
        order_type=OrderType.LIMIT,
        linked_entry_order_id=None,
        status=OrderStatus.PENDING,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=5),
    )
    manager = SimpleNamespace(
        _lock=RLock(),
        _orders={"recent": order},
        _logger=logging.getLogger(__name__),
        cancel_order=cancelled.append,
    )

    OrderManager._check_zombie_orders(manager)

    assert cancelled == []
