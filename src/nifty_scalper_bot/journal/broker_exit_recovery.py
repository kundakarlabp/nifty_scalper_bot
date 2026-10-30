"""Broker-evidence-only recovery of a missing local protective-stop exit.

This operation never submits, modifies, or cancels broker orders. It may restore
a *historical* journal and fill leg only after a fresh day orderbook, position
snapshot, and the existing entry fill all agree. It is not an entry path.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from nifty_scalper_bot.execution.fill_ledger import (
    BracketFillLedgerStore,
    FillLeg,
)
from nifty_scalper_bot.journal.trade_ledger import materialize_trade_events
from nifty_scalper_bot.risk.cost_model import estimate_round_trip_cost

_IST = ZoneInfo("Asia/Kolkata")
_TERMINAL = {"COMPLETE", "FILLED"}


class HistoricalExitRecoveryError(ValueError):
    """Broker proof is absent, ambiguous, or inconsistent with durable fills."""


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise HistoricalExitRecoveryError("invalid_broker_economics") from exc
    if not math.isfinite(result):
        raise HistoricalExitRecoveryError("nonfinite_broker_economics")
    return result


def _fill_timestamp(value: Any) -> float:
    """Require the broker order's actual terminal update, never submit time."""
    if not isinstance(value, str) or not value.strip():
        raise HistoricalExitRecoveryError("broker_terminal_timestamp_missing")
    try:
        moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise HistoricalExitRecoveryError("broker_terminal_timestamp_invalid") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_IST)
    return moment.timestamp()


def _symbol(order: Mapping[str, Any]) -> str:
    exchange = str(order.get("exchange") or "").strip().upper()
    name = str(order.get("tradingsymbol") or "").strip().upper()
    return f"{exchange}:{name}"


def _order_quantity(order: Mapping[str, Any]) -> int:
    requested = _number(order.get("quantity"))
    filled = _number(order.get("filled_quantity"))
    if requested <= 0 or not requested.is_integer() or filled != requested:
        raise HistoricalExitRecoveryError("broker_order_not_fully_filled")
    return int(requested)


def _safe_day(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=_IST).date().isoformat()


def recover_broker_proven_exit(
    *,
    journal_db: str | Path,
    fill_db: str | Path,
    orders: Sequence[Mapping[str, Any]],
    positions: Sequence[Mapping[str, Any]],
    today: str,
    commit: bool = False,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Recover only unique same-session BUY + filled exchange-SL SELL pairs.

    Caller must fetch both broker snapshots without cached fallback. Default is
    read-only. Durable writes require `commit=True`; never infer a missing
    order, an exit price, a quantity, or settled contract-note charges.
    """
    if not isinstance(orders, (list, tuple)) or not isinstance(
        positions, (list, tuple)
    ):
        raise HistoricalExitRecoveryError("fresh_broker_snapshots_required")
    try:
        datetime.strptime(today, "%Y-%m-%d")
    except ValueError as exc:
        raise HistoricalExitRecoveryError("invalid_session_date") from exc
    timestamp = time.time() if now is None else _number(now)
    if _safe_day(timestamp) != today:
        raise HistoricalExitRecoveryError("only_current_session_recovery_allowed")
    path = Path(journal_db)
    if not path.is_file():
        return []
    order_rows = [dict(row) for row in orders if isinstance(row, Mapping)]
    if len(order_rows) != len(orders):
        raise HistoricalExitRecoveryError("malformed_broker_orders")
    if len({str(row.get("order_id")) for row in order_rows}) != len(order_rows):
        raise HistoricalExitRecoveryError("duplicate_broker_order_identity")
    position_rows = [dict(row) for row in positions if isinstance(row, Mapping)]
    if len(position_rows) != len(positions):
        raise HistoricalExitRecoveryError("malformed_broker_positions")

    store = BracketFillLedgerStore(fill_db)
    results: list[dict[str, Any]] = []
    with sqlite3.connect(path, timeout=10.0) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT trade_id, signal_id, trace_id, strategy, symbol, side,
                      entry_order_id, entry_price, entry_filled_at, quantity,
                      build_sha
               FROM trade_ledger
               WHERE state = 'OPEN' AND entry_filled_at IS NOT NULL
               ORDER BY entry_filled_at"""
        ).fetchall()
        for row in rows:
            trade_id = str(row["trade_id"])
            entry_id = str(row["entry_order_id"] or "")
            symbol = str(row["symbol"] or "").upper()
            quantity = int(row["quantity"] or 0)
            entry_price = _number(row["entry_price"])
            opened_at = _number(row["entry_filled_at"])
            if (
                not entry_id
                or str(row["side"]).upper() != "BUY"
                or quantity <= 0
                or entry_price <= 0
                or _safe_day(opened_at) != today
            ):
                continue

            matching = [
                o for o in order_rows
                if _symbol(o) == symbol and
                str(o.get("product") or "").upper() == "MIS" and
                _number(o.get("filled_quantity") or 0) > 0
            ]
            if len(matching) != 2:
                continue  # Ambiguous historical ownership: never guess.
            entries = [o for o in matching if str(o.get("order_id")) == entry_id]
            exits = [o for o in matching if str(o.get("order_id")) != entry_id]
            if len(entries) != 1 or len(exits) != 1:
                continue
            entry_order, exit_order = entries[0], exits[0]
            exit_id = str(exit_order["order_id"])
            if (
                str(entry_order.get("transaction_type")).upper() != "BUY"
                or str(exit_order.get("transaction_type")).upper() != "SELL"
                or str(exit_order.get("order_type")).upper() not in {"SL", "SL-M"}
                or str(entry_order.get("status")).upper() not in _TERMINAL
                or str(exit_order.get("status")).upper() not in _TERMINAL
            ):
                continue
            if (
                _order_quantity(entry_order) != quantity
                or _order_quantity(exit_order) != quantity
                or abs(_number(entry_order.get("average_price")) - entry_price) > 0.01
            ):
                continue
            exit_price = _number(exit_order.get("average_price"))
            if exit_price <= 0:
                continue
            exited_at = _fill_timestamp(
                exit_order.get("exchange_update_timestamp")
            )
            if (
                exited_at < opened_at
                or exited_at > timestamp
                or _safe_day(exited_at) != today
            ):
                continue

            if any(
                _symbol(pos) == symbol and
                str(pos.get("product") or "").upper() == "MIS" and
                _number(pos.get("quantity") or 0) != 0
                for pos in position_rows
            ):
                continue
            entry_leg = store.get_fill(f"ENTRY:{entry_id}")
            if (
                entry_leg is None
                or entry_leg.bracket_id != entry_id
                or entry_leg.order_id != entry_id
                or entry_leg.kind != "ENTRY"
                or entry_leg.side != "BUY"
                or entry_leg.quantity != quantity
                or abs(entry_leg.price - entry_price) > 0.01
                or any(leg.kind == "EXIT" for leg in store.load_fills(entry_id))
            ):
                continue
            gross = round((exit_price - entry_price) * quantity, 2)
            costs = asdict(
                estimate_round_trip_cost(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                )
            )
            finding = {
                "trade_id": trade_id,
                "entry_order_id": entry_id,
                "exit_order_id": exit_id,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross,
                "estimated_costs": round(costs["total"], 2),
                "net_pnl_verified": False,
                "exit_filled_at": exited_at,
                "status": "eligible",
            }
            if not commit:
                results.append(finding)
                continue

            # The exit leg is independently idempotent. On interruption after
            # this write, a retry finds it; the journal transaction still must
            # be completed rather than claiming an already-closed trade.
            exit_leg = FillLeg(
                fill_id=f"EXIT:{exit_id}",
                bracket_id=entry_id,
                order_id=exit_id,
                kind="EXIT",
                side="SELL",
                quantity=quantity,
                price=exit_price,
                target="FINAL",
                reason="exchange_protective_stop_fill",
                recorded_at=exited_at,
                metadata={"symbol": symbol, "source": "broker_orderbook_recovery"},
            )
            store.record_fill(exit_leg)
            if not store.realized_pnl(entry_id).complete:
                raise HistoricalExitRecoveryError("fill_ledger_not_converged")

            metadata = {
                "trade_id": trade_id,
                "signal_id": row["signal_id"],
                "trace_id": row["trace_id"],
                "strategy": row["strategy"],
                "entry_order_id": entry_id,
                "exit_order_id": exit_id,
                "reason": "exchange_protective_stop_fill",
                "fill_price": exit_price,
                "broker_confirmed_fill": True,
                "recovered_historical": True,
            }
            outcome = {
                "trade_id": trade_id,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross,
                "net_pnl": None,
                "estimated_costs": costs,
                "cost_source": "estimated_model_unverified",
                "ledger_complete": True,
                "exit_reason": "exchange_protective_stop_fill",
                "close_source": "broker_orderbook_history_recovery",
                "charges_verified": False,
            }

            def event(kind: str, occurred_at: float, meta: dict[str, Any]):
                return {
                    "event_type": kind,
                    "event_name": {
                        "EXIT_FILLED": "exit.filled",
                        "BRACKET_CLOSED": "trade.closed",
                        "HISTORICAL_EXIT_RECOVERY_VERIFIED":
                            "historical.exit.recovery.verified",
                    }[kind],
                    "timestamp": occurred_at,
                    "symbol": symbol,
                    "side": "SELL",
                    "qty": quantity,
                    "price": exit_price,
                    "order_id": exit_id,
                    "trade_id": trade_id,
                    "signal_id": row["signal_id"],
                    "trace_id": row["trace_id"],
                    "strategy": row["strategy"],
                    "reason_code": "exchange_protective_stop_fill",
                    "build_sha": row["build_sha"],
                    "meta": meta,
                }

            events = [
                event("EXIT_FILLED", exited_at, dict(metadata)),
                event(
                    "BRACKET_CLOSED",
                    exited_at,
                    {**metadata, "completed_trade": outcome},
                ),
                event(
                    "HISTORICAL_EXIT_RECOVERY_VERIFIED",
                    timestamp,
                    {**metadata, "verified_at": timestamp},
                ),
            ]
            connection.execute("BEGIN IMMEDIATE")
            try:
                latest = connection.execute(
                    "SELECT state FROM trade_ledger WHERE trade_id = ?",
                    (trade_id,),
                ).fetchone()
                has_exit = connection.execute(
                    """SELECT 1 FROM trade_events
                       WHERE trade_id = ? AND event_name IN
                       ('exit.filled', 'trade.closed') LIMIT 1""",
                    (trade_id,),
                ).fetchone()
                if latest is None or latest["state"] != "OPEN" or has_exit:
                    connection.execute("ROLLBACK")
                    finding["status"] = "already_reconciled"
                    results.append(finding)
                    continue
                for item in events:
                    connection.execute(
                        """INSERT INTO trade_events (
                           timestamp, event_type, event_name, symbol, side,
                           qty, price, order_id, trade_id, signal_id, trace_id,
                           strategy, reason_code, build_sha, meta_json, event_json
                           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            item["timestamp"], item["event_type"],
                            item["event_name"], item["symbol"], item["side"],
                            item["qty"], item["price"], item["order_id"],
                            item["trade_id"], item["signal_id"], item["trace_id"],
                            item["strategy"], item["reason_code"],
                            item["build_sha"], json.dumps(item["meta"]),
                            json.dumps(item),
                        ),
                    )
                materialize_trade_events(connection, events)
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
            finding["status"] = "recovered"
            results.append(finding)
    return results
