from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from nifty_scalper_bot.execution.bracket_manager import BracketManager

SYMBOL = "NFO:NIFTY2682725000CE"


@pytest.fixture(autouse=True)
def isolated_bracket_store(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRACKET_STALE_REFRESH_MIN_INTERVAL_SEC", "1.0")


class _MarketDataManager:
    def __init__(
        self,
        *,
        age_seconds: float | None,
        stale_after_seconds: float = 2.0,
    ) -> None:
        self.age_seconds = age_seconds
        self.stale_after_seconds = stale_after_seconds
        self.refresh_calls: list[tuple[str, str]] = []

    def time_since_last_tick(self, _symbol: str) -> float | None:
        return self.age_seconds

    def _ltp_stale_threshold_for_symbol(self, _symbol: str) -> float:
        return self.stale_after_seconds

    def request_fallback_refresh(self, symbol: str, *, reason: str) -> bool:
        self.refresh_calls.append((symbol, reason))
        return True


class _DataHub:
    def __init__(self, mdm: _MarketDataManager) -> None:
        self._mdm = mdm


class _OrderManager:
    def __init__(self) -> None:
        self._broker = SimpleNamespace(
            get_positions=lambda: [{"symbol": SYMBOL, "quantity": 65}]
        )
        self.place_calls: list[dict[str, Any]] = []

    def place_order(self, **kwargs: Any) -> str:
        self.place_calls.append(dict(kwargs))
        return "unexpected-exit"


def _active_manager(mdm: _MarketDataManager) -> tuple[BracketManager, _OrderManager]:
    order_manager = _OrderManager()
    manager = BracketManager(
        order_manager=order_manager,
        market_data=_DataHub(mdm),
    )
    manager.register_virtual_bracket(
        order_id="entry-stale-ltp",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        activate_immediately=True,
    )
    return manager, order_manager


def _wait_until(predicate, timeout: float = 1.25) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def test_active_bracket_stale_ltp_requests_canonical_fallback_refresh() -> None:
    mdm = _MarketDataManager(age_seconds=10.0, stale_after_seconds=2.0)
    manager, order_manager = _active_manager(mdm)
    try:
        assert _wait_until(lambda: bool(mdm.refresh_calls))
        assert mdm.refresh_calls[0] == (SYMBOL, "bracket_ltp_stale")
        assert order_manager.place_calls == []
    finally:
        manager.shutdown()
        manager._watchdog_thread.join(timeout=1.0)


def test_active_bracket_fresh_ltp_does_not_request_fallback_refresh() -> None:
    mdm = _MarketDataManager(age_seconds=0.25, stale_after_seconds=2.0)
    manager, order_manager = _active_manager(mdm)
    try:
        time.sleep(0.40)
        assert mdm.refresh_calls == []
        assert order_manager.place_calls == []
    finally:
        manager.shutdown()
        manager._watchdog_thread.join(timeout=1.0)
