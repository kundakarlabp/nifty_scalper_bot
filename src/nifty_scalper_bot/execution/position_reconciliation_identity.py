"""Pure broker-position identity helpers for PositionManager reconciliation.

This module contains no runtime patch installation.  PositionManager owns the
broker-position reconciliation lifecycle; compatibility overlays may reuse these
helpers while the remaining ordered patch stack is migrated incrementally.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Mapping

from nifty_scalper_bot.execution.position_snapshot import (
    PositionSnapshotError,
    decode_position_snapshot,
)
from nifty_scalper_bot.utils.symbols import normalize_symbol

_SYMBOL_FIELDS = ("symbol", "tradingsymbol", "trading_symbol")
_AVG_PRICE_FIELDS = ("average_price", "avg_price", "buy_price", "price")
_QTY_FIELDS = ("quantity", "net_qty", "net_quantity", "netQuantity", "net")
_LOCAL_LIFECYCLE_FIELDS = (
    "entry_time",
    "order_id",
    "stop_loss",
    "take_profit",
    "trailing_stop_distance",
    "state",
)


def _canonical_key(symbol: object) -> str:
    return normalize_symbol(str(symbol or ""))


def _canonicalize_payload_symbol(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    cloned = dict(payload)
    for key in _SYMBOL_FIELDS:
        value = cloned.get(key)
        if isinstance(value, str) and value.strip():
            cloned[key] = _canonical_key(value)
    return cloned


def _canonicalize_broker_positions(broker_positions: Any) -> Any:
    if broker_positions is None:
        return None
    if isinstance(broker_positions, dict):
        # Preserve canonical complete broker snapshot mappings such as Zerodha
        # {"net": [...], "day": [...]} instead of wrapping them as one row.
        if "net" in broker_positions or "positions" in broker_positions:
            cloned = dict(broker_positions)
            for key in ("net", "positions"):
                value = cloned.get(key)
                if isinstance(value, list):
                    cloned[key] = [_canonicalize_payload_symbol(row) for row in value]
            return cloned
        return [_canonicalize_payload_symbol(broker_positions)]
    try:
        return [_canonicalize_payload_symbol(position) for position in broker_positions]
    except TypeError:
        return broker_positions


def _positive_float(payload: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        with suppress(Exception):
            value = float(payload.get(key) or 0.0)
            if value > 0.0:
                return value
    return 0.0


def _net_quantity(payload: Mapping[str, Any]) -> int:
    for key in _QTY_FIELDS:
        if key not in payload:
            continue
        with suppress(Exception):
            return int(float(payload.get(key) or 0))
    return 0


def _prepared_row_symbol(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    return _canonical_key(row.get("symbol") or row.get("tradingsymbol"))


def _prepare_broker_positions(
    manager: Any, broker_positions: Any
) -> tuple[Any, set[str]]:
    canonicalized = _canonicalize_broker_positions(broker_positions)
    if isinstance(canonicalized, dict):
        try:
            snapshot = decode_position_snapshot(canonicalized)
        except PositionSnapshotError:
            return canonicalized, set()
        canonicalized = snapshot.raw_rows()
    if not isinstance(canonicalized, list):
        return canonicalized, set()
    positions = getattr(manager, "_positions", {})
    unresolved: set[str] = set()
    prepared: list[Any] = []
    for row in canonicalized:
        if not isinstance(row, dict):
            prepared.append(row)
            continue
        cloned = dict(row)
        symbol = _canonical_key(cloned.get("tradingsymbol") or cloned.get("symbol"))
        if symbol:
            cloned["tradingsymbol"] = symbol
            cloned["symbol"] = symbol
        net_qty = _net_quantity(cloned)
        avg_price = _positive_float(cloned, _AVG_PRICE_FIELDS)
        existing = positions.get(symbol) if isinstance(positions, dict) else None
        existing_entry = (
            float(getattr(existing, "entry_price", 0.0) or 0.0) if existing else 0.0
        )
        existing_qty = int(getattr(existing, "quantity", 0) or 0) if existing else 0
        existing_side = (
            str(getattr(existing, "side", "") or "").strip().upper() if existing else ""
        )
        same_side = bool(
            (net_qty > 0 and existing_side == "LONG")
            or (net_qty < 0 and existing_side == "SHORT")
        )
        owned_same_exposure = bool(
            existing
            and str(getattr(existing, "order_id", "") or "").strip()
            and abs(net_qty) == existing_qty
            and same_side
        )
        safe_local_basis_reuse = bool(
            existing
            and existing_entry > 0.0
            and existing_qty > 0
            and same_side
            and abs(net_qty) <= existing_qty
        )
        if net_qty != 0 and owned_same_exposure and existing_entry > 0.0:
            # Zerodha's day-position average can span earlier closed trades in the
            # same contract. Once this exact exposure is locally owned, the
            # broker-confirmed order fill is the authoritative lifecycle basis.
            cloned["average_price"] = existing_entry
        elif net_qty != 0 and avg_price <= 0.0:
            # A local basis remains valid for an unchanged/reduced same-side
            # exposure. A scale-up or side reversal introduces broker exposure
            # whose acquisition basis is unknown, so fail closed instead of
            # inheriting stale cost basis from the earlier lifecycle.
            if safe_local_basis_reuse:
                cloned["average_price"] = existing_entry
            else:
                unresolved.add(symbol)
        prepared.append(cloned)
    return prepared, unresolved


def _cost_basis_quarantine_exposure(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build the canonical unmanaged exposure for one unresolved broker row."""

    out = dict(row)
    symbol = _canonical_key(out.get("symbol") or out.get("tradingsymbol"))
    signed_qty = _net_quantity(out)
    out.update(
        {
            "symbol": symbol,
            "tradingsymbol": symbol,
            "quantity": abs(signed_qty),
            "signed_quantity": signed_qty,
            "side": "LONG" if signed_qty > 0 else "SHORT" if signed_qty < 0 else "FLAT",
            "status": "BROKER_POSITION_QUARANTINED",
            "reason": "cost_basis_unresolved",
            "cost_basis_unresolved": True,
            "managed_position": False,
            "entry_accounting_allowed": False,
            "realized_pnl_accounting_allowed": False,
            "requires_history_recovery": True,
        }
    )
    return out


def _merge_cost_basis_quarantine(
    existing: Mapping[str, Mapping[str, Any]] | None,
    prepared: Any,
    unresolved: set[str],
) -> dict[str, dict[str, Any]]:
    """Refresh only cost-basis rows while preserving stronger quarantine sources."""

    fresh: dict[str, dict[str, Any]] = {}
    if isinstance(prepared, list) and unresolved:
        for row in prepared:
            if not isinstance(row, Mapping):
                continue
            symbol = _prepared_row_symbol(row)
            if symbol in unresolved:
                fresh[symbol] = _cost_basis_quarantine_exposure(row)

    preserved: dict[str, dict[str, Any]] = {}
    if isinstance(existing, Mapping):
        for raw_symbol, exposure in existing.items():
            if not isinstance(exposure, Mapping):
                continue
            if str(exposure.get("reason") or "") == "cost_basis_unresolved":
                continue
            symbol = _canonical_key(
                exposure.get("symbol") or exposure.get("tradingsymbol") or raw_symbol
            )
            if symbol:
                preserved[symbol] = dict(exposure)

    # Stronger non-cost-basis ownership wins a same-symbol collision.
    return {**fresh, **preserved}


def _snapshot_owned_position_lifecycle(manager: Any) -> dict[str, dict[str, Any]]:
    """Capture local-only lifecycle identity for positions already owned by the bot."""

    positions = getattr(manager, "_positions", None)
    if not isinstance(positions, dict):
        return {}
    snapshot: dict[str, dict[str, Any]] = {}
    for raw_key, position in list(positions.items()):
        order_id = str(getattr(position, "order_id", "") or "").strip()
        if not order_id:
            continue
        symbol = _canonical_key(getattr(position, "symbol", raw_key))
        if not symbol:
            continue
        values = {
            field: getattr(position, field, None) for field in _LOCAL_LIFECYCLE_FIELDS
        }
        values["side"] = str(getattr(position, "side", "") or "").strip().upper()
        snapshot[symbol] = values
    return snapshot


def _restore_owned_position_lifecycle(
    manager: Any,
    snapshot: dict[str, dict[str, Any]],
) -> int:
    """Restore local lifecycle fields only when broker truth still shows the same side."""

    if not snapshot:
        return 0
    positions = getattr(manager, "_positions", None)
    if not isinstance(positions, dict):
        return 0
    restored = 0
    for symbol, values in snapshot.items():
        position = positions.get(symbol)
        if position is None:
            continue
        saved_side = str(values.get("side") or "").strip().upper()
        current_side = str(getattr(position, "side", "") or "").strip().upper()
        if saved_side and current_side and saved_side != current_side:
            continue
        try:
            if int(getattr(position, "quantity", 0) or 0) == 0:
                continue
        except (TypeError, ValueError):
            continue
        for field in _LOCAL_LIFECYCLE_FIELDS:
            with suppress(Exception):
                setattr(position, field, values.get(field))
        restored += 1
    return restored


def _canonicalize_position_store(manager: Any) -> None:
    positions = getattr(manager, "_positions", None)
    if not isinstance(positions, dict):
        return
    canonical: dict[str, Any] = {}
    for raw_key, position in list(positions.items()):
        key = _canonical_key(getattr(position, "symbol", raw_key))
        if not key:
            key = str(raw_key).strip().upper()
        with suppress(Exception):
            position.symbol = key
        existing = canonical.get(key)
        if existing is None:
            canonical[key] = position
            continue
        with suppress(Exception):
            if abs(int(getattr(position, "quantity", 0) or 0)) > abs(
                int(getattr(existing, "quantity", 0) or 0)
            ):
                canonical[key] = position
    positions.clear()
    positions.update(canonical)


__all__ = [
    "_canonical_key",
    "_canonicalize_payload_symbol",
    "_canonicalize_broker_positions",
    "_prepared_row_symbol",
    "_prepare_broker_positions",
    "_cost_basis_quarantine_exposure",
    "_merge_cost_basis_quarantine",
    "_snapshot_owned_position_lifecycle",
    "_restore_owned_position_lifecycle",
    "_canonicalize_position_store",
]
