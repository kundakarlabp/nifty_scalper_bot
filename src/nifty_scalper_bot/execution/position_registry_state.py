"""Canonical persistence owner for broker-order and quarantine registries.

PositionManager remains the lifecycle/accounting implementation.  This module is
installed explicitly after the legacy position/risk wrappers and before broker-order
classification overlays.  It owns the durable broker-order ledger and
quarantine registry as part of the same atomic positions.json snapshot, avoiding
the prior read/merge/second-write persistence race.
"""

from __future__ import annotations

import copy
import json
from contextlib import suppress
from typing import Any, Mapping

from nifty_scalper_bot.execution import position_manager as _position_manager
from nifty_scalper_bot.execution.position_reconciliation_identity import (
    _build_cost_basis_exposures,
    _merge_cost_basis_exposures,
    _prepare_broker_positions,
)
from nifty_scalper_bot.execution.position_risk_state_patch import (
    _restore_risk_state,
    _risk_state_snapshot,
)
from nifty_scalper_bot.utils.symbols import normalize_symbol

_PATCH_APPLIED = False
_ORIGINALS: dict[str, Any] = {}
_ACTIVE_LEDGER_CLASSIFICATIONS = {
    "active_external_order",
    "broker_position_quarantined",
    "broker_state_unverified",
}


def _canonical(symbol: object) -> str:
    value = str(symbol or "").strip()
    return normalize_symbol(value) or value.upper()


def _read_state(self: Any) -> dict[str, Any]:
    path = getattr(self, "_state_path", None)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _hydrate_registry_state(self: Any, payload: Mapping[str, Any]) -> None:
    ledger_raw = payload.get("broker_order_ledger", {})
    ledger = (
        {
            str(order_id): dict(row)
            for order_id, row in ledger_raw.items()
            if isinstance(row, Mapping)
        }
        if isinstance(ledger_raw, Mapping)
        else {}
    )

    exposures_raw = payload.get("quarantined_broker_exposures", {})
    exposures: dict[str, dict[str, Any]] = {}
    if isinstance(exposures_raw, Mapping):
        for raw_key, row in exposures_raw.items():
            if not isinstance(row, Mapping):
                continue
            key = _canonical(raw_key) or _canonical(
                row.get("symbol") or row.get("tradingsymbol")
            )
            if key:
                exposures[key] = dict(row)

    persisted_unresolved_raw = payload.get("cost_basis_unresolved_symbols", [])
    persisted_unresolved = {
        _canonical(symbol)
        for symbol in persisted_unresolved_raw
        if _canonical(symbol)
    } if isinstance(persisted_unresolved_raw, (list, tuple, set)) else set()
    exposure_unresolved = {
        symbol
        for symbol, row in exposures.items()
        if str(row.get("reason") or "") == "cost_basis_unresolved"
    }

    with self._lock:
        self._broker_order_ledger = ledger
        self._quarantined_broker_exposures = exposures
        self._cost_basis_unresolved_symbols = (
            persisted_unresolved | exposure_unresolved
        )


def _registry_snapshot_locked(self: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = {
        str(order_id): dict(row)
        for order_id, row in self._broker_order_ledger.items()
        if isinstance(row, Mapping)
    }
    exposures = {
        str(symbol): dict(row)
        for symbol, row in self._quarantined_broker_exposures.items()
        if isinstance(row, Mapping)
    }
    return ledger, exposures


def _ledger_blocks_symbol(self: Any, symbol: str) -> bool:
    wanted = _canonical(symbol)
    with self._lock:
        rows = list(self._broker_order_ledger.values())
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("classification") or "") not in _ACTIVE_LEDGER_CLASSIFICATIONS:
            continue
        row_symbol = _canonical(row.get("symbol") or row.get("tradingsymbol"))
        if row_symbol == wanted:
            return True
    return False


def apply_patches() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    cls = getattr(_position_manager, "PositionManager", None)
    if cls is None or getattr(cls, "_canonical_registry_state_owner", False):
        _PATCH_APPLIED = True
        return

    for name in (
        "__init__",
        "save_state",
        "load_state",
        "synchronize_with_broker",
        "current_entry_protection_blocker",
    ):
        if hasattr(cls, name):
            _ORIGINALS[f"PositionManager.{name}"] = getattr(cls, name)

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        # Native __init__ invokes self.load_state(); initialise the registries
        # first so the patched load path can hydrate them safely.
        self._broker_order_ledger = {}
        self._quarantined_broker_exposures = {}
        self._registry_state_write_generation = 0
        _ORIGINALS["PositionManager.__init__"](self, *args, **kwargs)

    def save_state(self: Any) -> None:
        """Persist one coherent native + broker-registry state snapshot."""

        with self._lock:
            ledger, exposures = _registry_snapshot_locked(self)
            state = {
                "positions": [
                    position.to_dict() for position in self._positions.values()
                ],
                "orders": [order.to_dict() for order in self._orders.values()],
                "terminal_orders": {
                    order_id: metadata.to_dict()
                    for order_id, metadata in self._terminal_orders.items()
                },
                "unresolved_terminal_orders": {
                    order_id: metadata.to_dict()
                    for order_id, metadata in self._unresolved_terminal_orders.items()
                },
                "exit_lifecycles": {
                    order_id: lifecycle.to_dict()
                    for order_id, lifecycle in self._exit_lifecycles.items()
                },
                "broker_order_ledger": ledger,
                "quarantined_broker_exposures": exposures,
                "cost_basis_unresolved_symbols": sorted(
                    set(self._cost_basis_unresolved_symbols)
                ),
                "_risk_runtime": _risk_state_snapshot(self),
                "daily_realized_pnl": self._daily_realized_pnl,
                "local_realized_pnl": self._local_realized_pnl,
                "broker_realized_pnl": self._broker_realized_pnl,
                "local_provisional_realized_pnl": self._local_provisional_realized_pnl,
                "authoritative_realized_pnl": self._authoritative_realized_pnl,
                "pnl_authority": self._pnl_authority,
                "pnl_reconciliation_status": self._pnl_reconciliation_status,
                "pnl_snapshot_at": (
                    self._pnl_snapshot_at.isoformat() if self._pnl_snapshot_at else None
                ),
                "session_opening_realized_baseline": (
                    self._session_opening_realized_baseline
                ),
                "pnl_trading_date": self._pnl_trading_date,
                "pnl_account_fingerprint": self._pnl_account_fingerprint,
                "pnl_product_scope": self._pnl_product_scope,
                "baseline_established_at": (
                    self._baseline_established_at.isoformat()
                    if self._baseline_established_at
                    else None
                ),
                "baseline_source": self._baseline_source,
                "require_pnl_baseline_for_entries": (
                    self._require_pnl_baseline_for_entries
                ),
                "active_contracts": [
                    contract.to_dict() for contract in self._active_contracts.values()
                ],
            }
            reconciled_snapshot = copy.deepcopy(self._positions)

        try:
            _position_manager._atomic_write_json(self._state_path, state)
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failed to save position state: %s", exc)
            return

        self._persist_positions_snapshot()
        with self._lock:
            self._last_reconciled_state = reconciled_snapshot
            self._registry_state_write_generation = int(
                getattr(self, "_registry_state_write_generation", 0)
            ) + 1
        self._maybe_flush_persistent_state()

    def load_state(self: Any) -> None:
        # Preserve the reviewed native recovery semantics for positions/orders/P&L,
        # then hydrate the extra registries from the same canonical state file.
        _ORIGINALS["PositionManager.load_state"](self)
        payload = _read_state(self)
        _hydrate_registry_state(self, payload)
        _restore_risk_state(self, payload.get("_risk_runtime"))

    def synchronize_with_broker(self: Any, broker_positions: Any) -> Any:
        prepared, unresolved = _prepare_broker_positions(self, broker_positions)
        fresh_cost_basis = _build_cost_basis_exposures(prepared, set(unresolved))
        with self._lock:
            previous_unresolved = set(self._cost_basis_unresolved_symbols)
            merged = _merge_cost_basis_exposures(
                self._quarantined_broker_exposures,
                fresh_cost_basis,
            )
            quarantine_changed = merged != self._quarantined_broker_exposures
            unresolved_changed = previous_unresolved != set(unresolved)
            self._quarantined_broker_exposures = merged
            before_write_generation = int(
                getattr(self, "_registry_state_write_generation", 0)
            )

        result = _ORIGINALS["PositionManager.synchronize_with_broker"](
            self, broker_positions
        )

        if quarantine_changed or unresolved_changed:
            with self._lock:
                write_generation = int(
                    getattr(self, "_registry_state_write_generation", 0)
                )
            if write_generation == before_write_generation:
                self.save_state()
        return result

    def current_entry_protection_blocker(
        self: Any, symbol: str | None = None
    ) -> str | None:
        wanted = _canonical(symbol) if symbol else None
        with self._lock:
            exposures = dict(self._quarantined_broker_exposures)
            unresolved = set(self._cost_basis_unresolved_symbols)

        if wanted is not None:
            exposure = exposures.get(wanted)
            if exposure is not None:
                if str(exposure.get("reason") or "") == "broker_state_unverified":
                    return "broker_state_unverified"
                return "broker_exposure_quarantined"
        elif exposures:
            if any(
                str(row.get("reason") or "") == "broker_state_unverified"
                for row in exposures.values()
                if isinstance(row, Mapping)
            ):
                return "broker_state_unverified"
            return "broker_exposure_quarantined"

        if unresolved and (wanted is None or wanted in unresolved):
            return "cost_basis_unresolved"

        original = _ORIGINALS.get("PositionManager.current_entry_protection_blocker")
        if callable(original):
            return original(self, wanted)
        return None

    def get_quarantined_broker_exposures(
        self: Any, symbol: str | None = None
    ) -> dict[str, dict[str, Any]] | list[dict[str, Any]]:
        wanted = _canonical(symbol) if symbol else None
        with self._lock:
            exposures = {
                key: dict(value)
                for key, value in self._quarantined_broker_exposures.items()
                if isinstance(value, Mapping)
            }
        if wanted is None:
            return exposures
        exposure = exposures.get(wanted)
        return [dict(exposure)] if exposure is not None else []

    def clear_quarantined_broker_exposure(self: Any, symbol: str) -> bool:
        wanted = _canonical(symbol)
        if not wanted or _ledger_blocks_symbol(self, wanted):
            return False
        with self._lock:
            removed = self._quarantined_broker_exposures.pop(wanted, None)
            if removed is None:
                return False
            if str(removed.get("reason") or "") == "cost_basis_unresolved":
                self._cost_basis_unresolved_symbols.discard(wanted)
        self.save_state()
        return True

    cls.__init__ = __init__
    cls.save_state = save_state
    cls.load_state = load_state
    cls.synchronize_with_broker = synchronize_with_broker
    if "PositionManager.current_entry_protection_blocker" in _ORIGINALS:
        cls.current_entry_protection_blocker = current_entry_protection_blocker
    cls.get_quarantined_broker_exposures = get_quarantined_broker_exposures
    cls.clear_quarantined_broker_exposure = clear_quarantined_broker_exposure
    cls._canonical_registry_state_owner = True
    cls._canonical_registry_state_owner_name = __name__
    _PATCH_APPLIED = True


__all__ = ["apply_patches"]
