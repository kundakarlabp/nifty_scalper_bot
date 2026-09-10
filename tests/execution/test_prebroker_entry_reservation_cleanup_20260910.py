from __future__ import annotations

from threading import RLock
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution import order_manager as order_manager_module


SYMBOL = "NFO:NIFTY2691523400PE"


class _Logger:
    def info(self, *args, **kwargs):
        return None


def _manager(*, broker_attempted: bool, allowed: bool = False):
    return SimpleNamespace(
        _entries_in_flight={SYMBOL: 1.0},
        _entry_inflight_owners={SYMBOL: "live-regression"},
        _last_order_decision={
            "allowed": allowed,
            "broker_attempted": broker_attempted,
            "block_reason": "risk_manager_blocked",
        },
        _lock=RLock(),
        _logger=_Logger(),
    )


def test_live_risk_rejection_releases_entry_reservation_immediately(monkeypatch) -> None:
    """10-Sep live regression: a local net-RR reject must not wedge next entry."""
    manager = _manager(broker_attempted=False)

    monkeypatch.setattr(
        order_manager_module,
        "_original_runtime_place_order",
        lambda self, *args, **kwargs: None,
    )

    result = order_manager_module._place_order_with_prebroker_reservation_cleanup(
        manager,
        symbol=SYMBOL,
        side="BUY",
        intent="ENTRY",
        check_risk=True,
    )

    assert result is None
    assert SYMBOL not in manager._entries_in_flight
    assert SYMBOL not in manager._entry_inflight_owners


def test_broker_attempted_entry_reservation_is_not_released(monkeypatch) -> None:
    """Once broker submission may have occurred, local cleanup must fail closed."""
    manager = _manager(broker_attempted=True)

    monkeypatch.setattr(
        order_manager_module,
        "_original_runtime_place_order",
        lambda self, *args, **kwargs: None,
    )

    order_manager_module._place_order_with_prebroker_reservation_cleanup(
        manager,
        symbol=SYMBOL,
        side="BUY",
        intent="ENTRY",
        check_risk=True,
    )

    assert SYMBOL in manager._entries_in_flight
    assert SYMBOL in manager._entry_inflight_owners


def test_allowed_path_does_not_release_reservation(monkeypatch) -> None:
    """Cleanup is scoped only to explicit local rejections."""
    manager = _manager(broker_attempted=False, allowed=True)
    sentinel = object()

    monkeypatch.setattr(
        order_manager_module,
        "_original_runtime_place_order",
        lambda self, *args, **kwargs: sentinel,
    )

    result = order_manager_module._place_order_with_prebroker_reservation_cleanup(
        manager,
        symbol=SYMBOL,
        side="BUY",
        intent="ENTRY",
        check_risk=True,
    )

    assert result is sentinel
    assert SYMBOL in manager._entries_in_flight
    assert SYMBOL in manager._entry_inflight_owners


def test_exception_path_preserves_reservation_fail_closed(monkeypatch) -> None:
    """Unexpected local exceptions must never clear an in-flight entry guard."""
    manager = _manager(broker_attempted=False)

    def _raise(self, *args, **kwargs):
        raise RuntimeError("synthetic local failure")

    monkeypatch.setattr(order_manager_module, "_original_runtime_place_order", _raise)

    with pytest.raises(RuntimeError, match="synthetic local failure"):
        order_manager_module._place_order_with_prebroker_reservation_cleanup(
            manager,
            symbol=SYMBOL,
            side="BUY",
            intent="ENTRY",
            check_risk=True,
        )

    assert SYMBOL in manager._entries_in_flight
    assert SYMBOL in manager._entry_inflight_owners


@pytest.mark.parametrize("owner", ["first-entry", "retry-of-same-setup"])
def test_rejected_contender_preserves_existing_entry_reservation(monkeypatch, owner):
    """A pre-broker gate rejection belongs to the contender, not its predecessor."""
    manager = _manager(broker_attempted=False)
    manager._entry_inflight_owners[SYMBOL] = owner
    manager._last_order_decision = {}

    def reject_contender(self, **kwargs):
        self._last_order_decision = {
            "allowed": False,
            "broker_attempted": False,
            "block_reason": f"single_position_gate:entry_in_flight:{SYMBOL}",
        }
        return None

    monkeypatch.setattr(
        order_manager_module, "_original_runtime_place_order", reject_contender
    )
    for _ in range(2):
        result = order_manager_module._place_order_with_prebroker_reservation_cleanup(
            manager,
            symbol=SYMBOL,
            side="BUY",
            intent="ENTRY",
            signal_id="retry-of-same-setup",
        )
        assert result is None
        assert manager._entries_in_flight == {SYMBOL: 1.0}
        assert manager._entry_inflight_owners == {SYMBOL: owner}
