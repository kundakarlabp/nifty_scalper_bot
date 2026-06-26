from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from nifty_scalper_bot.execution.ledger_bracket_manager import LedgerBracketManager
from nifty_scalper_bot.execution.quote_readiness import evaluate_execution_quote
from nifty_scalper_bot.execution.runtime_bracket_manager import RuntimeBracketManager


SYMBOL = "NFO:NIFTY26JUN24050CE"


def test_execution_quote_accepts_fresh_full_depth_without_top_level_bid_ask() -> None:
    verdict = evaluate_execution_quote(
        SYMBOL,
        {
            "last_price": 101.0,
            "tick_age_ms": 150.0,
            "timestamp_ms": 1_782_443_000_000,
            "depth": {
                "buy": [{"price": 100.9, "quantity": 75}],
                "sell": [{"price": 101.1, "quantity": 50}],
            },
        },
        live_mode=True,
        max_tick_age_ms=2_000.0,
        max_spread_pct=1.0,
        require_depth=True,
    )

    assert verdict.allowed is True
    assert verdict.reason == "ready"
    assert verdict.bid == 100.9
    assert verdict.ask == 101.1
    assert verdict.depth_available is True
    assert verdict.tradable_quote is True


def test_execution_quote_keeps_one_sided_depth_fail_closed() -> None:
    verdict = evaluate_execution_quote(
        SYMBOL,
        {
            "last_price": 101.0,
            "tick_age_ms": 150.0,
            "depth": {"buy": [{"price": 100.9, "quantity": 75}], "sell": []},
        },
        live_mode=True,
        max_tick_age_ms=2_000.0,
        max_spread_pct=1.0,
        require_depth=True,
    )

    assert verdict.allowed is False
    assert verdict.reason == "bid_ask_missing"
    assert verdict.tradable_quote is False


class _ReleaseStore:
    def __init__(self) -> None:
        self.cleared: list[str] = []

    def clear(self, bracket_id: str) -> None:
        self.cleared.append(str(bracket_id))


@dataclass
class _Broker:
    positions: Any
    raises: bool = False

    def get_positions(self) -> Any:
        if self.raises:
            raise RuntimeError("broker unavailable")
        return self.positions


@dataclass
class _OrderManager:
    _broker: _Broker


def _orphan_manager(*, payload: dict[str, Any], broker: _Broker) -> RuntimeBracketManager:
    manager = object.__new__(RuntimeBracketManager)
    manager._ledger_blocked = {
        "old-bracket": {"reason": "final_exit_accounting_failed", "payload": payload}
    }
    manager._release_store = _ReleaseStore()
    manager.order_manager = _OrderManager(broker)
    manager._find_bracket_by_id = lambda _bracket_id: None  # type: ignore[method-assign]
    manager._retry_ledger_block = lambda _bracket: False  # type: ignore[method-assign]
    return manager


def test_runtime_block_persists_restart_reconciliation_identity(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _capture(
        _self: Any,
        _bracket: Any,
        *,
        reason: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        captured["reason"] = reason
        captured["payload"] = dict(payload or {})

    monkeypatch.setattr(LedgerBracketManager, "_block_ledger_release", _capture)
    manager = object.__new__(RuntimeBracketManager)
    bracket = SimpleNamespace(
        symbol=SYMBOL,
        bracket_id="entry-1",
        exit_state="CLOSED",
        remaining_quantity=0,
    )

    manager._block_ledger_release(
        bracket,
        reason="final_exit_accounting_failed",
        payload={"order_id": "exit-1"},
    )

    assert captured["payload"] == {
        "order_id": "exit-1",
        "symbol": SYMBOL,
        "bracket_id": "entry-1",
        "exit_state": "CLOSED",
        "remaining_quantity": 0,
    }


def test_symbol_orphan_clears_only_after_explicit_broker_flat_proof() -> None:
    manager = _orphan_manager(payload={"symbol": SYMBOL}, broker=_Broker([]))

    manager._retry_blocked_releases()

    assert manager._ledger_blocked == {}
    assert manager._release_store.cleared == ["old-bracket"]


def test_symbol_orphan_remains_blocked_when_position_is_open() -> None:
    manager = _orphan_manager(
        payload={"symbol": SYMBOL},
        broker=_Broker([{"symbol": SYMBOL, "quantity": 50}]),
    )

    manager._retry_blocked_releases()

    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []


def test_symbol_orphan_remains_blocked_when_broker_truth_is_unknown() -> None:
    manager = _orphan_manager(
        payload={"symbol": SYMBOL}, broker=_Broker([], raises=True)
    )

    manager._retry_blocked_releases()

    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []


def test_legacy_orphan_without_symbol_clears_when_whole_account_is_flat() -> None:
    manager = _orphan_manager(payload={}, broker=_Broker({"net": [], "day": []}))

    manager._retry_blocked_releases()

    assert manager._ledger_blocked == {}
    assert manager._release_store.cleared == ["old-bracket"]


def test_legacy_orphan_without_symbol_retains_block_for_open_position() -> None:
    manager = _orphan_manager(
        payload={},
        broker=_Broker({"net": [{"symbol": SYMBOL, "quantity": 50}]}),
    )

    manager._retry_blocked_releases()

    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []
