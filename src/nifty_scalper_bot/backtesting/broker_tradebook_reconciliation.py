"""Read-only ledger-to-broker tradebook reconciliation.

Accept an explicitly exported Kite tradebook CSV. This is not a live broker
connection, account-position check, or final contract-note charge authority.
Only exact entry/exit order-ID matches are eligible for fill verification.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REQUIRED = {
    "trade_id": ("trade_id", "broker_trade_id"),
    "order_id": ("order_id",),
    "symbol": ("tradingsymbol", "symbol"),
    "side": ("transaction_type", "side"),
    "quantity": ("quantity",),
    "price": ("average_price", "price"),
}


def _valid_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_broker_tradebook_csv(path: Path) -> list[dict[str, Any]]:
    """Parse only known tradebook columns; reject incomplete or duplicate rows."""
    with path.expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        columns = {
            key: next((option for option in aliases if option in headers), None)
            for key, aliases in _REQUIRED.items()
        }
        missing = [key for key, column in columns.items() if column is None]
        if missing:
            raise ValueError(
                "broker_tradebook_missing_columns:" + ",".join(sorted(missing))
            )
        resolved: list[dict[str, Any]] = []
        seen_trade_ids: set[str] = set()
        for number, raw in enumerate(reader, start=2):
            row = {
                key: str(raw.get(column) or "").strip()
                for key, column in columns.items()
                if column is not None
            }
            quantity_value = _valid_number(row["quantity"])
            price_value = _valid_number(row["price"])
            if (
                any(not row[key] for key in ("trade_id", "order_id", "symbol", "side"))
                or row["side"].upper() not in {"BUY", "SELL"}
                or quantity_value is None
                or quantity_value <= 0
                or not quantity_value.is_integer()
                or price_value is None
                or price_value <= 0
            ):
                raise ValueError(f"broker_tradebook_invalid_row:{number}")
            if row["trade_id"] in seen_trade_ids:
                raise ValueError(f"broker_tradebook_duplicate_trade_id:{number}")
            seen_trade_ids.add(row["trade_id"])
            resolved.append(
                {
                    "trade_id": row["trade_id"],
                    "order_id": row["order_id"],
                    "symbol": row["symbol"].upper(),
                    "side": row["side"].upper(),
                    "quantity": int(quantity_value),
                    "price": price_value,
                }
            )
    if not resolved:
        raise ValueError("broker_tradebook_empty")
    return resolved


def _normalized_symbol(value: Any) -> str:
    return str(value or "").strip().upper().rsplit(":", 1)[-1]


def _order_fill_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_symbol: str,
    expected_side: str,
    expected_quantity: int,
    expected_price: float,
    price_tolerance: float,
) -> str:
    if not rows:
        return "missing_order_fills"
    if any(
        _normalized_symbol(row["symbol"]) != expected_symbol
        or row["side"] != expected_side
        for row in rows
    ):
        return "symbol_or_side_mismatch"
    quantity = sum(int(row["quantity"]) for row in rows)
    if quantity < expected_quantity:
        return "partial_order_fills"
    if quantity > expected_quantity:
        return "quantity_mismatch"
    vwap = sum(float(row["price"]) * int(row["quantity"]) for row in rows) / quantity
    if abs(vwap - expected_price) > price_tolerance + 1e-9:
        return "fill_price_mismatch"
    return "matched"


def summarize_broker_tradebook_reconciliation(
    ledger_rows: Sequence[Mapping[str, Any]],
    broker_fills: Sequence[Mapping[str, Any]],
    *,
    price_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Verify exact order-fill evidence, never infer charges or live positions."""
    if not math.isfinite(price_tolerance) or price_tolerance < 0:
        raise ValueError("invalid_price_tolerance")
    by_order: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    broker_trade_ids: set[str] = set()
    for fill in broker_fills:
        trade_id = str(fill.get("trade_id") or "").strip()
        order_id = str(fill.get("order_id") or "").strip()
        if not trade_id or not order_id or trade_id in broker_trade_ids:
            raise ValueError("broker_tradebook_duplicate_or_missing_fill_identity")
        broker_trade_ids.add(trade_id)
        by_order[order_id].append(fill)

    results: list[dict[str, str]] = []
    status_counts: dict[str, int] = defaultdict(int)
    referenced_order_ids: set[str] = set()
    complete_rows = 0
    for ledger in ledger_rows:
        if str(ledger.get("state") or "").upper() != "CLOSED" or (
            ledger.get("ledger_complete") not in (1, True)
        ):
            continue
        complete_rows += 1
        trade_id = str(ledger.get("trade_id") or "")
        entry_order_id = str(ledger.get("entry_order_id") or "").strip()
        exit_order_id = str(ledger.get("exit_order_id") or "").strip()
        quantity_value = _valid_number(ledger.get("quantity"))
        entry_price = _valid_number(ledger.get("entry_price"))
        exit_price = _valid_number(ledger.get("exit_price"))
        symbol = _normalized_symbol(ledger.get("symbol"))
        side = str(ledger.get("side") or "").strip().upper()
        if (
            not trade_id
            or not entry_order_id
            or not exit_order_id
            or entry_order_id == exit_order_id
            or not symbol
            or side != "BUY"
            or quantity_value is None
            or quantity_value <= 0
            or not quantity_value.is_integer()
            or entry_price is None
            or exit_price is None
        ):
            status = "insufficient_ledger_identity"
        else:
            referenced_order_ids.update((entry_order_id, exit_order_id))
            entry_status = _order_fill_evidence(
                by_order.get(entry_order_id, ()),
                expected_symbol=symbol,
                expected_side="BUY",
                expected_quantity=int(quantity_value),
                expected_price=entry_price,
                price_tolerance=price_tolerance,
            )
            exit_status = _order_fill_evidence(
                by_order.get(exit_order_id, ()),
                expected_symbol=symbol,
                expected_side="SELL",
                expected_quantity=int(quantity_value),
                expected_price=exit_price,
                price_tolerance=price_tolerance,
            )
            status = (
                "fills_matched"
                if entry_status == exit_status == "matched"
                else f"entry:{entry_status};exit:{exit_status}"
            )
        status_counts[status] += 1
        results.append({"trade_id": trade_id, "fill_verification": status})

    matched = status_counts.get("fills_matched", 0)
    return {
        "source": "explicit_user_supplied_broker_tradebook_csv",
        "broker_tradebook_fill_rows": len(broker_fills),
        "completed_ledger_rows": complete_rows,
        "fills_verified_trade_count": matched,
        "unverified_completed_trade_count": complete_rows - matched,
        "statuses": dict(sorted(status_counts.items())),
        "trades": results,
        "unreferenced_broker_order_count": len(
            set(by_order).difference(referenced_order_ids)
        ),
        "actual_contract_note_charges_verified": False,
        "final_net_pnl_broker_reconciled": False,
        "live_positions_inferred": False,
        "limitations": (
            "Fill-only verification; missing or split exit orders cannot be "
            "assumed flat. Final net P&L requires separate settled contract "
            "notes and charge attribution."
        ),
    }
