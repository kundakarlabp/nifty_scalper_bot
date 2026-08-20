from threading import RLock
from types import SimpleNamespace

from nifty_scalper_bot.execution.entry_geometry import (
    entry_geometry_block_reason,
    entry_identity_block_reason,
    release_prebroker_entry_reservation,
)
from nifty_scalper_bot.execution.runtime_order_manager import RuntimeOrderManager
from nifty_scalper_bot.utils.symbols import normalize_symbol


def test_entry_geometry_blocks_low_reward_to_risk():
    reason = entry_geometry_block_reason(
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
    reason = entry_geometry_block_reason(
        symbol="NFO:NIFTY24JUL24000CE",
        side="BUY",
        price=140.55,
        stop_loss=135.50,
        take_profit=151.00,
        intent="ENTRY",
    )

    assert reason is None


def test_entry_geometry_uses_canonical_net_rr_threshold(monkeypatch):
    monkeypatch.setenv("ENTRY_MIN_RR", "3.0")
    monkeypatch.setenv("MIN_BRACKET_RR", "3.0")
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "2.1")

    reason = entry_geometry_block_reason(
        symbol="NFO:NIFTY24JUL24000CE",
        side="BUY",
        price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        intent="ENTRY",
    )

    assert reason is not None
    assert reason["block_reason"] == "entry_rr_below_floor"
    assert reason["rr_floor"] == 2.1


def test_runtime_order_manager_is_not_entry_geometry_monkeypatched():
    assert not getattr(RuntimeOrderManager, "_order_entry_geometry_patch", False)


def test_entry_geometry_does_not_block_protective_exit():
    reason = entry_geometry_block_reason(
        symbol="NFO:NIFTY24JUL24000CE",
        side="SELL",
        price=135.50,
        stop_loss=None,
        take_profit=None,
        intent="EXIT",
    )

    assert reason is None


def test_explicit_sell_entry_without_stop_is_blocked():
    reason = entry_geometry_block_reason(
        symbol="NFO:NIFTY24JUL24000CE",
        side="SELL",
        price=140.0,
        stop_loss=None,
        take_profit=125.0,
        intent="ENTRY",
    )

    assert reason is not None
    assert reason["block_reason"] == "entry_stop_loss_required"


def test_explicit_entry_with_exit_like_tag_is_blocked():
    reason = entry_identity_block_reason(
        intent="ENTRY",
        tag="strategy_exit_probe",
        symbol="NFO:NIFTY24JUL24000CE",
    )

    assert reason is not None
    assert reason["block_reason"] == "entry_exit_tag_conflict"


def test_exit_intent_is_not_blocked_by_entry_identity_guard():
    assert (
        entry_identity_block_reason(
            intent="EXIT",
            tag="strategy_exit_probe",
            symbol="NFO:NIFTY24JUL24000CE",
        )
        is None
    )


def test_explicit_prebroker_rejection_releases_entry_reservation():
    symbol = "NFO:NIFTY2681124600CE"
    normalized = normalize_symbol(symbol)
    manager = SimpleNamespace(
        is_live_mode=lambda: True,
        _lock=RLock(),
        _entries_in_flight={normalized: 1.0},
        _last_order_decision={"allowed": False, "broker_attempted": False},
    )
    released = release_prebroker_entry_reservation(
        manager,
        {"symbol": symbol, "intent": "ENTRY"},
    )

    assert released is True
    assert normalized not in manager._entries_in_flight


def test_broker_attempted_rejection_keeps_entry_reservation():
    symbol = "NFO:NIFTY2681124600CE"
    normalized = normalize_symbol(symbol)
    manager = SimpleNamespace(
        is_live_mode=lambda: True,
        _lock=RLock(),
        _entries_in_flight={normalized: 1.0},
        _last_order_decision={"allowed": False, "broker_attempted": True},
    )
    released = release_prebroker_entry_reservation(
        manager,
        {"symbol": symbol, "intent": "ENTRY"},
    )

    assert released is False
    assert normalized in manager._entries_in_flight
