from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from nifty_scalper_bot.execution.fill_ledger import BracketFillLedgerStore, FillLeg
from nifty_scalper_bot.execution.ledger_bracket_manager import LedgerBracketManager
from nifty_scalper_bot.execution.quote_readiness import evaluate_execution_quote
from nifty_scalper_bot.execution.runtime_bracket_manager import RuntimeBracketManager


SYMBOL = "NFO:NIFTY26JUN24050CE"


def test_execution_quote_accepts_fresh_full_depth_without_top_level_bid_ask(
    monkeypatch,
) -> None:
    import nifty_scalper_bot.execution.readiness as readiness

    monkeypatch.setattr(readiness.time, "time", lambda: 1_782_443_000.15)
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
    orders: Any = None
    orderbook_raises: bool = False
    strict_orderbook_reads: int = 0

    def get_positions(self) -> Any:
        if self.raises:
            raise RuntimeError("broker unavailable")
        return self.positions

    def get_orders(self, *, allow_cache_fallback: bool = True) -> Any:
        if not allow_cache_fallback:
            self.strict_orderbook_reads += 1
        if self.orderbook_raises:
            raise RuntimeError("orderbook unavailable")
        return [] if self.orders is None else self.orders


@dataclass
class _OrderManager:
    _broker: _Broker


def _orphan_manager(*, payload: dict[str, Any], broker: _Broker) -> RuntimeBracketManager:
    manager = object.__new__(RuntimeBracketManager)
    manager._ledger_blocked = {
        "old-bracket": {"reason": "final_exit_accounting_failed", "payload": payload}
    }
    manager._release_store = _ReleaseStore()
    manager._fill_ledger = SimpleNamespace(
        realized_pnl=lambda _bracket_id: SimpleNamespace(complete=True)
    )
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
        product="MIS",
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
        "product": "MIS",
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


def test_product_qualified_orphan_releases_without_waiting_for_other_product() -> None:
    manager = _orphan_manager(
        payload={"symbol": SYMBOL, "product": "MIS"},
        broker=_Broker(
            [
                {"symbol": SYMBOL, "product": "MIS", "quantity": 0},
                {"symbol": SYMBOL, "product": "NRML", "quantity": 65},
            ]
        ),
    )

    manager._retry_blocked_releases()

    assert manager._ledger_blocked == {}
    assert manager._release_store.cleared == ["old-bracket"]


def test_product_qualified_orphan_keeps_block_when_its_position_is_open() -> None:
    manager = _orphan_manager(
        payload={"symbol": SYMBOL, "product": "MIS"},
        broker=_Broker(
            [
                {"symbol": SYMBOL, "product": "MIS", "quantity": 65},
                {"symbol": SYMBOL, "product": "NRML", "quantity": 0},
            ]
        ),
    )

    manager._retry_blocked_releases()

    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []


def test_product_qualified_orphan_rejects_unlabeled_broker_position() -> None:
    manager = _orphan_manager(
        payload={"symbol": SYMBOL, "product": "MIS"},
        broker=_Broker([{"symbol": SYMBOL, "quantity": 0}]),
    )

    manager._retry_blocked_releases()

    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []


def test_legacy_orphan_does_not_net_across_products() -> None:
    manager = _orphan_manager(
        payload={"symbol": SYMBOL},
        broker=_Broker(
            [
                {"symbol": SYMBOL, "product": "MIS", "quantity": 0},
                {"symbol": SYMBOL, "product": "NRML", "quantity": 65},
            ]
        ),
    )

    manager._retry_blocked_releases()

    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []


def test_orphan_with_invalid_persisted_product_remains_blocked() -> None:
    manager = _orphan_manager(
        payload={"symbol": SYMBOL, "product": "INVALID"},
        broker=_Broker([]),
    )

    manager._retry_blocked_releases()

    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []


def test_flat_orphan_remains_blocked_without_durable_fills(tmp_path) -> None:
    manager = _orphan_manager(payload={"symbol": SYMBOL}, broker=_Broker([]))
    manager._fill_ledger = BracketFillLedgerStore(tmp_path / "orphan-ledger.db")

    manager._retry_blocked_releases()

    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []


def test_orphan_requires_complete_entry_and_exit_fill_ledger(tmp_path) -> None:
    manager = _orphan_manager(
        payload={"symbol": SYMBOL, "product": "MIS"},
        broker=_Broker([]),
    )
    ledger = BracketFillLedgerStore(tmp_path / "orphan-ledger.db")
    manager._fill_ledger = ledger
    ledger.record_fill(
        FillLeg(
            fill_id="ENTRY:entry-order",
            bracket_id="old-bracket",
            order_id="entry-order",
            kind="ENTRY",
            side="BUY",
            quantity=65,
            price=100.0,
        )
    )
    manager._retry_blocked_releases()
    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []

    ledger.record_fill(
        FillLeg(
            fill_id="EXIT:exit-order",
            bracket_id="old-bracket",
            order_id="exit-order",
            kind="EXIT",
            side="SELL",
            quantity=65,
            price=99.0,
        )
    )
    manager._retry_blocked_releases()

    assert manager._ledger_blocked == {}
    assert manager._release_store.cleared == ["old-bracket"]


def test_orphan_fails_closed_when_ledger_unavailable() -> None:
    manager = _orphan_manager(payload={"symbol": SYMBOL}, broker=_Broker([]))
    manager._fill_ledger = None

    manager._retry_blocked_releases()

    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []


def test_orphan_keeps_entry_block_when_durable_clear_fails() -> None:
    manager = _orphan_manager(payload={"symbol": SYMBOL}, broker=_Broker([]))

    def reject_clear(_bracket_id: str) -> None:
        raise OSError("database not writable")

    manager._release_store.clear = reject_clear
    manager._retry_blocked_releases()

    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []


def test_flat_orphan_does_not_release_with_resting_exchange_stop() -> None:
    broker = _Broker(
        [],
        orders=[
            {
                "order_id": "STOP-1",
                "exchange": "NFO",
                "tradingsymbol": SYMBOL.split(":", 1)[1],
                "product": "MIS",
                "order_type": "SL",
                "transaction_type": "SELL",
                "status": "TRIGGER PENDING",
            }
        ],
    )
    manager = _orphan_manager(
        payload={"symbol": SYMBOL, "product": "MIS"}, broker=broker
    )
    manager._is_live_execution = lambda: True  # type: ignore[method-assign]

    manager._retry_blocked_releases()

    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []
    assert broker.strict_orderbook_reads == 1


def test_flat_orphan_releases_after_broker_stop_cancelled() -> None:
    broker = _Broker(
        [],
        orders=[
            {
                "order_id": "STOP-1",
                "tradingsymbol": SYMBOL.split(":", 1)[1],
                "product": "MIS",
                "status": "CANCELLED",
            }
        ],
    )
    manager = _orphan_manager(
        payload={"symbol": SYMBOL, "product": "MIS"}, broker=broker
    )
    manager._is_live_execution = lambda: True  # type: ignore[method-assign]

    manager._retry_blocked_releases()

    assert manager._ledger_blocked == {}
    assert manager._release_store.cleared == ["old-bracket"]
    assert broker.strict_orderbook_reads == 1


def test_flat_orphan_does_not_block_for_unrelated_product_stop() -> None:
    manager = _orphan_manager(
        payload={"symbol": SYMBOL, "product": "MIS"},
        broker=_Broker(
            [],
            orders=[
                {
                    "tradingsymbol": SYMBOL.split(":", 1)[1],
                    "product": "NRML",
                    "status": "OPEN",
                }
            ],
        ),
    )

    manager._retry_blocked_releases()

    assert manager._ledger_blocked == {}
    assert manager._release_store.cleared == ["old-bracket"]


def test_flat_orphan_blocks_for_unknown_order_product_or_identity() -> None:
    manager = _orphan_manager(
        payload={"symbol": SYMBOL, "product": "MIS"},
        broker=_Broker([], orders=[{"status": "OPEN"}]),
    )

    manager._retry_blocked_releases()

    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []


def test_flat_orphan_fails_closed_when_fresh_orderbook_unavailable() -> None:
    broker = _Broker([], orderbook_raises=True)
    manager = _orphan_manager(
        payload={"symbol": SYMBOL, "product": "MIS"}, broker=broker
    )
    manager._is_live_execution = lambda: True  # type: ignore[method-assign]

    manager._retry_blocked_releases()

    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []
    assert broker.strict_orderbook_reads == 1


def test_flat_orphan_rejects_malformed_orderbook_response() -> None:
    manager = _orphan_manager(
        payload={"symbol": SYMBOL}, broker=_Broker([], orders=[None])
    )

    manager._retry_blocked_releases()

    assert "old-bracket" in manager._ledger_blocked
    assert manager._release_store.cleared == []
