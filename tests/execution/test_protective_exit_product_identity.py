from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from nifty_scalper_bot.execution import live_safety_identity  # noqa: F401
from nifty_scalper_bot.execution.bracket_core import BracketManager


SYMBOL = "NFO:NIFTY2690124150PE"


class _Positions:
    def add_pending_order(self, *args: Any, **kwargs: Any) -> None:
        return None

    def bind_pending_order_id(self, *args: Any, **kwargs: Any) -> None:
        return None

    def remove_pending_order(self, *args: Any, **kwargs: Any) -> None:
        return None


class _OrderManager:
    def __init__(self) -> None:
        self._positions = _Positions()
        self._broker = SimpleNamespace(get_positions=lambda: [{"symbol": SYMBOL, "quantity": 65, "product": "NRML"}])
        self._last_order_decision: dict[str, Any] = {}
        self.place_calls: list[dict[str, Any]] = []

    def place_order(self, **kwargs: Any) -> str:
        self.place_calls.append(dict(kwargs))
        return f"exit-{len(self.place_calls)}"

    def cancel_order(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


def _manager(tmp_path, monkeypatch) -> tuple[BracketManager, _OrderManager]:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    order_manager = _OrderManager()
    manager = BracketManager(order_manager=order_manager)
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=65.85,
        sl=61.15,
        tp=75.65,
        product="NRML",
        activate_immediately=True,
    )
    manager._price_exit_order = lambda **_kwargs: (
        "MARKET",
        None,
        {
            "mode": "market",
            "bid": 67.65,
            "ask": 67.75,
            "ltp": 67.70,
            "fallback": False,
            "quote_missing": False,
        },
    )
    return manager, order_manager


def test_protective_exit_preserves_entry_product(tmp_path, monkeypatch) -> None:
    manager, order_manager = _manager(tmp_path, monkeypatch)

    result = manager.submit_exit_order(
        symbol=SYMBOL,
        qty=65,
        reason="HARD_SL_BREACH",
        bracket_id="entry-1",
        preferred_order_type="MARKET",
        correlation_tag="exit-product-test",
    )

    assert result.accepted is True
    assert order_manager.place_calls[-1]["intent"] == "EXIT"
    assert order_manager.place_calls[-1]["product"] == "NRML"


def test_escalation_market_exit_preserves_entry_product(tmp_path, monkeypatch) -> None:
    manager, order_manager = _manager(tmp_path, monkeypatch)
    manager._exit_force_market_on_escalation = True
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    bracket.exit_pending = True
    bracket.remaining_quantity = 65
    bracket.last_exit_error = "fatal_order_error"

    with manager._lock:
        manager._escalate_exit_locked(bracket, "fatal_or_retry_exhausted")

    assert order_manager.place_calls
    assert order_manager.place_calls[-1]["intent"] == "EXIT"
    assert order_manager.place_calls[-1]["product"] == "NRML"
