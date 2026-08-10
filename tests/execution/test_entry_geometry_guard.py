from threading import RLock
from types import SimpleNamespace

from nifty_scalper_bot.execution import order_entry_guard_patch as guard_patch
from nifty_scalper_bot.execution.order_entry_guard_patch import _entry_geometry_block_reason
from nifty_scalper_bot.utils.symbols import normalize_symbol


def test_entry_geometry_blocks_low_reward_to_risk():
    reason = _entry_geometry_block_reason(
        SimpleNamespace(),
        symbol="NFO:NIFTY24JUL24000CE",
        side="BUY",
        price=223.90,
        stop_loss=214.61,
        take_profit=227.76,
        intent="ENTRY",
    )

    assert reason is not None
    assert reason["block_reason"] == "entry_rr_below_floor"
    assert reason["rr"] < reason["rr_floor"]


def test_entry_geometry_allows_good_reward_to_risk():
    reason = _entry_geometry_block_reason(
        SimpleNamespace(),
        symbol="NFO:NIFTY24JUL24000CE",
        side="BUY",
        price=140.55,
        stop_loss=135.50,
        take_profit=151.00,
        intent="ENTRY",
    )

    assert reason is None


def test_entry_geometry_does_not_block_protective_exit():
    reason = _entry_geometry_block_reason(
        SimpleNamespace(),
        symbol="NFO:NIFTY24JUL24000CE",
        side="SELL",
        price=135.50,
        stop_loss=None,
        take_profit=None,
        intent="EXIT",
    )

    assert reason is None


def test_explicit_prebroker_rejection_releases_entry_reservation(monkeypatch):
    symbol = "NFO:NIFTY2681124600CE"
    normalized = normalize_symbol(symbol)
    manager = SimpleNamespace(
        is_live_mode=lambda: True,
        _lock=RLock(),
        _entries_in_flight={normalized: 1.0},
        _last_order_decision={"allowed": False, "broker_attempted": False},
    )
    monkeypatch.setattr(
        guard_patch,
        "_ORIGINAL_PLACE_ORDER",
        lambda _self, *args, **kwargs: None,
    )

    result = guard_patch._patched_place_order(
        manager,
        symbol=symbol,
        side="BUY",
        quantity=65,
        intent="ENTRY",
    )

    assert result is None
    assert normalized not in manager._entries_in_flight


def test_broker_attempted_rejection_keeps_entry_reservation(monkeypatch):
    symbol = "NFO:NIFTY2681124600CE"
    normalized = normalize_symbol(symbol)
    manager = SimpleNamespace(
        is_live_mode=lambda: True,
        _lock=RLock(),
        _entries_in_flight={normalized: 1.0},
        _last_order_decision={"allowed": False, "broker_attempted": True},
    )
    monkeypatch.setattr(
        guard_patch,
        "_ORIGINAL_PLACE_ORDER",
        lambda _self, *args, **kwargs: None,
    )

    guard_patch._patched_place_order(
        manager,
        symbol=symbol,
        side="BUY",
        quantity=65,
        intent="ENTRY",
    )

    assert normalized in manager._entries_in_flight
