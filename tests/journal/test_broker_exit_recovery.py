"""Regression proof for broker-backed local trade-journal recovery."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.execution.fill_ledger import (
    BracketFillLedgerStore,
    FillLeg,
)
from nifty_scalper_bot.journal.broker_exit_recovery import (
    HistoricalExitRecoveryError,
    recover_broker_proven_exit,
)
from nifty_scalper_bot.journal.trade_journal import TradeJournal

_SYMBOL = "NFO:NIFTY26N0323050PE"
_ENTRY = "2116084443126362112"
_EXIT = "2116084452060229632"
_TRADE = "TRD_7b90049669d40650"
_IST = ZoneInfo("Asia/Kolkata")
_OPEN = datetime(2026, 10, 30, 13, 57, 3, tzinfo=_IST).timestamp()
_CLOSE = datetime(2026, 10, 30, 13, 59, 3, tzinfo=_IST).timestamp()
_NOW = datetime(2026, 10, 30, 18, 10, 0, tzinfo=_IST).timestamp()


def _broker_orders() -> list[dict]:
    return [
        {
            "order_id": _ENTRY,
            "exchange": "NFO",
            "tradingsymbol": _SYMBOL.removeprefix("NFO:"),
            "product": "MIS",
            "transaction_type": "BUY",
            "order_type": "LIMIT",
            "status": "COMPLETE",
            "quantity": 65,
            "filled_quantity": 65,
            "average_price": 148.80,
            "exchange_update_timestamp": "2026-10-30 13:57:03",
        },
        {
            "order_id": _EXIT,
            "exchange": "NFO",
            "tradingsymbol": _SYMBOL.removeprefix("NFO:"),
            "product": "MIS",
            "transaction_type": "SELL",
            "order_type": "SL",
            "status": "COMPLETE",
            "quantity": 65,
            "filled_quantity": 65,
            "average_price": 150.60,
            "exchange_update_timestamp": "2026-10-30 13:59:03",
        },
    ]


@pytest.fixture()
def stores(tmp_path: Path) -> tuple[Path, Path]:
    journal_path = tmp_path / "trades.db"
    fill_path = tmp_path / "bot_state.db"
    journal = TradeJournal(str(journal_path))
    connection = journal._ensure_connection(None)
    events = [
        journal._normalize_event(
            {
                "event_type": typ,
                "timestamp": _OPEN - 0.5 if typ == "BRACKET_ARMED" else _OPEN,
                "symbol": _SYMBOL,
                "side": "BUY",
                "qty": 65,
                "price": 148.8,
                "order_id": _ENTRY,
                "meta": {
                    "trade_id": _TRADE,
                    "signal_id": _TRADE.removeprefix("TRD_"),
                    "strategy": "SMC",
                    "trace_id": "trace-evidence",
                    "stop_price": 141.65,
                    "target_price": 163.10,
                },
            }
        )
        for typ in ("ORDER_FILL_CONFIRMED", "BRACKET_ARMED")
    ]
    journal._flush_batch(events, connection)
    connection.close()
    ledger = BracketFillLedgerStore(fill_path)
    ledger.record_fill(
        FillLeg(
            fill_id=f"ENTRY:{_ENTRY}",
            bracket_id=_ENTRY,
            order_id=_ENTRY,
            kind="ENTRY",
            side="BUY",
            quantity=65,
            price=148.8,
            reason="ENTRY_CONFIRMED",
            metadata={"symbol": _SYMBOL},
        )
    )
    return journal_path, fill_path


def _recover(stores, *, orders=None, positions=None, commit=False, now=_NOW):
    return recover_broker_proven_exit(
        journal_db=stores[0],
        fill_db=stores[1],
        orders=_broker_orders() if orders is None else orders,
        positions=[] if positions is None else positions,
        today="2026-10-30",
        commit=commit,
        now=now,
    )


def _read(journal_path: Path):
    with sqlite3.connect(journal_path) as connection:
        row = connection.execute(
            """SELECT state, exit_order_id, exit_price, gross_pnl,
                      estimated_costs, net_pnl, ledger_complete, closed_at,
                      updated_at FROM trade_ledger WHERE trade_id = ?""",
            (_TRADE,),
        ).fetchone()
        event_count = connection.execute(
            """SELECT COUNT(*) FROM trade_events
               WHERE trade_id = ?
                 AND event_name IN
                 ('exit.filled', 'trade.closed',
                  'historical.exit.recovery.verified')""",
            (_TRADE,),
        ).fetchone()[0]
    return row, event_count


def test_dry_run_then_exact_recovery_and_idempotent_replay(stores):
    before, events_before = _read(stores[0])
    assert before[0] == "OPEN"
    assert events_before == 0

    preview = _recover(stores)
    assert len(preview) == 1
    assert preview[0]["status"] == "eligible"
    assert preview[0]["gross_pnl"] == 117.0
    assert _read(stores[0]) == (before, 0)
    assert len(BracketFillLedgerStore(stores[1]).load_fills(_ENTRY)) == 1

    recovered = _recover(stores, commit=True)
    assert len(recovered) == 1
    assert recovered[0]["status"] == "recovered"
    result, events = _read(stores[0])
    assert result[0] == "CLOSED"
    assert result[1] == _EXIT
    assert result[2] == pytest.approx(150.60)
    assert result[3] == pytest.approx(117.0)
    assert result[4] > 0
    assert result[5] is None  # Never mislabel model costs as final broker net.
    assert result[6] == 1
    assert result[7] == pytest.approx(_CLOSE)
    assert result[8] == pytest.approx(_NOW)
    assert events == 3

    assert _recover(stores, commit=True) == []
    assert _read(stores) == (result, 3)
    fill = BracketFillLedgerStore(stores[1]).realized_pnl(_ENTRY)
    assert fill.complete is True
    assert fill.exit_quantity == 65
    assert fill.gross_pnl == 117.0


@pytest.mark.parametrize("change", ["partial_exit", "manual_exit", "ambiguous"])
def test_unsupported_orderbook_does_not_write(stores, change):
    orders = _broker_orders()
    if change == "partial_exit":
        orders[1]["filled_quantity"] = 25
    elif change == "manual_exit":
        orders[1]["order_type"] = "MARKET"
    else:
        extra = dict(orders[1])
        extra["order_id"] = "other-sell"
        extra["average_price"] = 165.0
        orders.append(extra)
    assert _recover(stores, orders=orders, commit=True) == []
    assert _read(stores)[0][0] == "OPEN"
    assert len(BracketFillLedgerStore(stores[1]).load_fills(_ENTRY)) == 1


def test_nonflat_broker_position_never_reconciles(stores):
    pos = [{
        "exchange": "NFO",
        "tradingsymbol": _SYMBOL.removeprefix("NFO:"),
        "product": "MIS",
        "quantity": 65,
    }]
    assert _recover(stores, positions=pos, commit=True) == []
    assert _read(stores)[1] == 0


def test_missing_durable_entry_fill_blocks_recovery(stores):
    with sqlite3.connect(stores[1]) as connection:
        connection.execute("DELETE FROM bracket_fill_legs")
    assert _recover(stores, commit=True) == []
    assert _read(stores)[0][0] == "OPEN"


def test_other_day_and_unverified_timestamp_fail_closed(stores):
    with pytest.raises(HistoricalExitRecoveryError, match="only_current_session"):
        _recover(stores, commit=True, now=_NOW + 86400)
    orders = _broker_orders()
    del orders[1]["exchange_update_timestamp"]
    with pytest.raises(HistoricalExitRecoveryError, match="timestamp_missing"):
        _recover(stores, orders=orders, commit=True)
    assert _read(stores)[0][0] == "OPEN"
