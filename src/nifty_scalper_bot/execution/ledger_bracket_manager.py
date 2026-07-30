"""Canonical bracket lifecycle with durable confirmed-fill accounting.

Protective execution never depends on the ledger being healthy.  A persistence
failure therefore cannot stop SL/TP handling, but it does block the runner from
re-arming until fill history and broker-flat state are reconciled.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from nifty_scalper_bot.execution import bracket_manager as _legacy
from nifty_scalper_bot.execution.canonical_bracket_manager import (
    CanonicalBracketManager,
)
from nifty_scalper_bot.execution.fill_ledger import (
    BracketFillLedgerStore,
    FillLedgerError,
    FillLeg,
    FillValidationError,
)
from nifty_scalper_bot.risk.cost_model import estimate_round_trip_cost


class _LedgerReleaseStore:
    """Persist entry-freeze markers independently of in-memory bracket state."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS bracket_ledger_blocks (
                    bracket_id TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def block(self, bracket_id: str, reason: str, payload: Mapping[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO bracket_ledger_blocks (
                    bracket_id, reason, payload_json, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(bracket_id) DO UPDATE SET
                    reason = excluded.reason,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    str(bracket_id),
                    str(reason),
                    json.dumps(dict(payload), sort_keys=True, default=str),
                    time.time(),
                ),
            )

    def clear(self, bracket_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM bracket_ledger_blocks WHERE bracket_id = ?",
                (str(bracket_id),),
            )

    def load(self) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT bracket_id, reason, payload_json FROM bracket_ledger_blocks"
            ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for bracket_id, reason, payload_json in rows:
            try:
                payload = json.loads(str(payload_json or "{}"))
            except json.JSONDecodeError:
                payload = {}
            result[str(bracket_id)] = {
                "reason": str(reason),
                "payload": payload if isinstance(payload, Mapping) else {},
            }
        return result


class LedgerBracketManager(CanonicalBracketManager):
    """Runtime bracket authority with durable fills and release gating."""

    _LEDGER_ERROR_PREFIX = "fill_ledger_degraded"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        data_dir = Path(os.getenv("DATA_DIR", "data"))
        configured = os.getenv("BRACKET_FILL_LEDGER_PATH")
        self._ledger_path = Path(configured) if configured else data_dir / "bot_state.db"
        self._fill_ledger: BracketFillLedgerStore | None = None
        self._release_store: _LedgerReleaseStore | None = None
        self._ledger_blocked: dict[str, dict[str, Any]] = {}
        try:
            self._fill_ledger = BracketFillLedgerStore(self._ledger_path)
            self._release_store = _LedgerReleaseStore(self._ledger_path)
            self._ledger_blocked = self._release_store.load()
        except Exception as exc:  # noqa: BLE001 - protection must still initialize
            _legacy.LOGGER.critical(
                "FILL_LEDGER_INIT_FAILED path=%s error=%s",
                self._ledger_path,
                exc,
                extra={
                    "event": "FILL_LEDGER_INIT_FAILED",
                    "path": str(self._ledger_path),
                    "error_type": type(exc).__name__,
                },
            )
        super().__init__(*args, **kwargs)

    @staticmethod
    def _entry_fill_id(order_id: str) -> str:
        return f"ENTRY:{order_id}"

    @staticmethod
    def _exit_fill_id(order_id: str) -> str:
        return f"EXIT:{order_id}"

    def _find_bracket_by_id(self, bracket_id: str) -> Any | None:
        with self._lock:
            for bracket in self._brackets.values():
                if str(bracket.bracket_id) == str(bracket_id):
                    return bracket
        return None

    def _block_ledger_release(
        self,
        bracket: Any,
        *,
        reason: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        details = {"reason": str(reason), "payload": dict(payload or {})}
        self._ledger_blocked[str(bracket.bracket_id)] = details
        with self._lock:
            bracket.last_exit_error = f"{self._LEDGER_ERROR_PREFIX}:{reason}"
            bracket.updated_at = time.time()
        if self._release_store is not None:
            with suppress(Exception):
                self._release_store.block(
                    str(bracket.bracket_id),
                    str(reason),
                    dict(payload or {}),
                )
        _legacy.LOGGER.critical(
            "FILL_LEDGER_RELEASE_BLOCKED bracket_id=%s symbol=%s reason=%s payload=%s",
            bracket.bracket_id,
            bracket.symbol,
            reason,
            dict(payload or {}),
            extra={
                "event": "FILL_LEDGER_RELEASE_BLOCKED",
                "bracket_id": bracket.bracket_id,
                "symbol": bracket.symbol,
                "reason": reason,
            },
        )
        with suppress(Exception):
            self._notify_event(
                "FILL_LEDGER_DEGRADED",
                {
                    "symbol": bracket.symbol,
                    "bracket_id": bracket.bracket_id,
                    "reason": reason,
                    "message": "Broker protection remains active; new entries are blocked until fill accounting is reconciled.",
                },
            )

    def _clear_ledger_release(self, bracket: Any) -> None:
        self._ledger_blocked.pop(str(bracket.bracket_id), None)
        if self._release_store is not None:
            with suppress(Exception):
                self._release_store.clear(str(bracket.bracket_id))
        with self._lock:
            if str(bracket.last_exit_error or "").startswith(self._LEDGER_ERROR_PREFIX):
                bracket.last_exit_error = None

    def _record_fill(self, leg: FillLeg) -> bool:
        if self._fill_ledger is None:
            raise FillLedgerError("fill ledger is unavailable")
        inserted = self._fill_ledger.record_fill(leg)
        _legacy.LOGGER.info(
            "FILL_LEDGER_RECORDED bracket_id=%s fill_id=%s kind=%s side=%s qty=%s price=%.2f inserted=%s",
            leg.bracket_id,
            leg.fill_id,
            leg.kind,
            leg.side,
            leg.quantity,
            leg.price,
            inserted,
            extra={
                "event": "FILL_LEDGER_RECORDED",
                "bracket_id": leg.bracket_id,
                "fill_id": leg.fill_id,
                "kind": leg.kind,
                "qty": leg.quantity,
                "price": leg.price,
                "inserted": inserted,
            },
        )
        return inserted

    def _record_entry_fill(self, bracket: Any, fill_price: float) -> None:
        self._record_fill(
            FillLeg(
                fill_id=self._entry_fill_id(str(bracket.entry_order_id)),
                bracket_id=str(bracket.bracket_id),
                order_id=str(bracket.entry_order_id),
                kind="ENTRY",
                side="BUY" if bracket.side == "BUY" else "SELL",
                quantity=int(bracket.quantity),
                price=float(fill_price),
                reason="ENTRY_CONFIRMED",
                metadata={"symbol": bracket.symbol},
            )
        )

    def _record_exit_fill(
        self,
        bracket: Any,
        *,
        order_id: str,
        quantity: int,
        price: float,
        target: str | None,
        reason: str | None,
    ) -> None:
        self._record_fill(
            FillLeg(
                fill_id=self._exit_fill_id(str(order_id)),
                bracket_id=str(bracket.bracket_id),
                order_id=str(order_id),
                kind="EXIT",
                side="SELL" if bracket.side == "BUY" else "BUY",
                quantity=int(quantity),
                price=float(price),
                target=target,
                reason=reason,
                metadata={"symbol": bracket.symbol},
            )
        )

    def confirm_entry_fill(self, order_id: str, fill_price: float) -> None:
        bracket = self.get_bracket(order_id)
        ledger_ok = True
        if bracket is not None:
            entry_intent = str(
                getattr(bracket, "entry_order_intent", "ENTRY") or "ENTRY"
            ).upper()
            if entry_intent not in {"ENTRY", "SCALE_IN", "REVERSAL"}:
                super().confirm_entry_fill(order_id, fill_price)
                return
            try:
                self._record_entry_fill(bracket, float(fill_price))
            except Exception as exc:  # noqa: BLE001
                ledger_ok = False
                setattr(bracket, "_ledger_pending_entry_price", float(fill_price))
                self._block_ledger_release(
                    bracket,
                    reason="entry_fill_persist_failed",
                    payload={"order_id": order_id, "error": str(exc)},
                )
        super().confirm_entry_fill(order_id, fill_price)
        bracket = self.get_bracket(order_id)
        if bracket is not None and ledger_ok:
            self._clear_ledger_release(bracket)

    def _resume_after_partial_target(
        self,
        bracket: Any,
        *,
        target: Any,
        residual_quantity: int,
        order_id: str,
        status_payload: Mapping[str, Any],
        requested_by: str,
    ) -> bool:
        previous_remaining = int(bracket.remaining_quantity or 0)
        filled_quantity = previous_remaining - int(residual_quantity)
        fill_price = self._extract_status_price(status_payload)
        ledger_ok = True
        try:
            if filled_quantity <= 0 or fill_price is None:
                raise FillValidationError("partial exit fill quantity/price unavailable")
            self._record_exit_fill(
                bracket,
                order_id=order_id,
                quantity=filled_quantity,
                price=fill_price,
                target=str(target.name),
                reason=str(bracket.exit_reason or requested_by),
            )
        except Exception as exc:  # noqa: BLE001
            ledger_ok = False
            setattr(bracket, "_ledger_pending_exit_order_id", str(order_id))
            setattr(bracket, "_ledger_pending_exit_quantity", max(filled_quantity, 0))
            setattr(bracket, "_ledger_pending_exit_price", fill_price)
            setattr(bracket, "_ledger_pending_exit_target", str(target.name))
            self._block_ledger_release(
                bracket,
                reason="partial_exit_persist_failed",
                payload={"order_id": order_id, "error": str(exc)},
            )

        resumed = super()._resume_after_partial_target(
            bracket,
            target=target,
            residual_quantity=residual_quantity,
            order_id=order_id,
            status_payload=status_payload,
            requested_by=requested_by,
        )
        if resumed and ledger_ok and self._fill_ledger is not None:
            pnl = self._fill_ledger.realized_pnl(str(bracket.bracket_id))
            self._clear_ledger_release(bracket)
            with suppress(Exception):
                self._notify_event(
                    "PARTIAL_EXIT_ACCOUNTED",
                    {
                        "symbol": bracket.symbol,
                        "target": str(target.name),
                        "remaining_qty": residual_quantity,
                        "realized_gross_pnl": pnl.gross_pnl,
                        "realized_net_pnl": pnl.net_pnl,
                    },
                )
        return resumed

    def _resolved_exit_price(self, bracket: Any, exit_price: float | None) -> float | None:
        if exit_price is not None and float(exit_price) > 0:
            return float(exit_price)
        order_id = bracket.exit_order_id or bracket.pending_exit_order_id
        if not order_id:
            return None
        with suppress(Exception):
            return self._extract_status_price(
                self._get_broker_order_status(str(order_id))
            )
        return None

    def _completed_trade_outcome(
        self,
        bracket: Any,
        *,
        ledger_pnl: Any | None,
        gross_pnl: float | None,
        exit_price: float | None,
        ledger_complete: bool,
    ) -> dict[str, Any]:
        """Build the strategy-facing outcome from confirmed bracket economics."""
        entry_price = (
            ledger_pnl.entry_vwap
            if ledger_pnl is not None
            else getattr(bracket, "entry_fill_price", None)
            or getattr(bracket, "entry_price", None)
        )
        resolved_exit = (
            ledger_pnl.exit_vwap
            if ledger_pnl is not None and ledger_pnl.exit_vwap is not None
            else exit_price
        )
        quantity = int(
            getattr(ledger_pnl, "entry_quantity", 0)
            or getattr(bracket, "quantity", 0)
            or 0
        )
        executed_orders = 2
        if self._fill_ledger is not None:
            with suppress(Exception):
                executed_orders = max(
                    2,
                    len(self._fill_ledger.load_fills(str(bracket.bracket_id))),
                )
        costs = None
        net_pnl = gross_pnl
        if (
            gross_pnl is not None
            and entry_price is not None
            and resolved_exit is not None
            and quantity > 0
        ):
            costs = estimate_round_trip_cost(
                entry_price=float(entry_price),
                exit_price=float(resolved_exit),
                quantity=quantity,
                executed_orders=executed_orders,
            )
            net_pnl = round(float(gross_pnl) - costs.total, 2)

        side = str(getattr(bracket, "side", "BUY") or "BUY").upper()
        high = float(
            getattr(bracket, "highest_ltp", entry_price) or entry_price or 0.0
        )
        low = float(
            getattr(bracket, "lowest_ltp", entry_price) or entry_price or 0.0
        )
        entry = float(entry_price or 0.0)
        if side == "SELL":
            mfe_points = max(0.0, entry - low)
            mae_points = max(0.0, high - entry)
        else:
            mfe_points = max(0.0, high - entry)
            mae_points = max(0.0, entry - low)
        closed_at = float(getattr(bracket, "closed_at", None) or time.time())
        opened_at = float(
            getattr(bracket, "entry_fill_ts", None)
            or getattr(bracket, "created_at", closed_at)
            or closed_at
        )
        provenance = dict(getattr(bracket, "trade_provenance", {}) or {})
        return {
            **provenance,
            "bracket_id": str(bracket.bracket_id),
            "symbol": str(bracket.symbol),
            "side": side,
            "quantity": quantity,
            "entry_price": float(entry_price) if entry_price is not None else None,
            "exit_price": (
                float(resolved_exit) if resolved_exit is not None else None
            ),
            "gross_pnl": gross_pnl,
            "estimated_costs": asdict(costs) if costs is not None else None,
            "net_pnl": net_pnl,
            "mfe_points": round(mfe_points, 4),
            "mae_points": round(mae_points, 4),
            "mfe_pnl": round(mfe_points * quantity, 2),
            "mae_pnl": round(mae_points * quantity, 2),
            "holding_seconds": round(max(0.0, closed_at - opened_at), 3),
            "exit_reason": str(
                getattr(bracket, "exit_reason", None) or bracket.close_source or ""
            ),
            "close_source": str(bracket.close_source or ""),
            "ledger_complete": bool(ledger_complete),
        }

    def _close_bracket(
        self,
        bracket: Any,
        *,
        close_source: str,
        exit_price: float | None = None,
    ) -> None:
        # Idempotency + single-flight for FINAL accounting.
        #
        # A second closure of an already-accounted bracket re-entered the
        # accounting block with remaining_quantity already 0, raised
        # "final exit identity, quantity or fill price unavailable", and
        # latched ledger_blocked=True with
        # fill_ledger_degraded:final_exit_accounting_failed. That latch does
        # not self-heal: _retry_ledger_block() sees a pending exit marker with
        # quantity 0 and returns before reaching its closed-and-complete
        # recovery, so entries stayed blocked with
        # ENTRY_BLOCKED_NATIVE_GATE reason=unresolved_exit_position even after
        # position reconciliation succeeded.
        #
        # Both orderings must be covered: a sequential re-close, and two
        # callers (watchdog / tick processing / reconciliation) beginning
        # final accounting concurrently.
        with self._lock:
            already_accounted = bool(
                getattr(bracket, "exit_executed", False)
                and int(getattr(bracket, "remaining_quantity", 0) or 0) <= 0
            )
            if already_accounted or getattr(bracket, "_final_close_in_progress", False):
                _legacy.LOGGER.info(
                    "BRACKET_CLOSE_ALREADY_ACCOUNTED bracket_id=%s close_source=%s",
                    getattr(bracket, "bracket_id", None),
                    close_source,
                    extra={
                        "event": "BRACKET_CLOSE_ALREADY_ACCOUNTED",
                        "bracket_id": str(getattr(bracket, "bracket_id", "") or ""),
                        "symbol": getattr(bracket, "symbol", None),
                        "close_source": close_source,
                        "concurrent": not already_accounted,
                    },
                )
                return
            setattr(bracket, "_final_close_in_progress", True)

        order_id = str(bracket.exit_order_id or bracket.pending_exit_order_id or "")
        closing_quantity = int(bracket.remaining_quantity or 0)
        resolved_price = self._resolved_exit_price(bracket, exit_price)
        ledger_complete = False
        ledger_pnl = None
        try:
            if not order_id or closing_quantity <= 0 or resolved_price is None:
                raise FillValidationError("final exit identity, quantity or fill price unavailable")
            self._record_exit_fill(
                bracket,
                order_id=order_id,
                quantity=closing_quantity,
                price=resolved_price,
                target="FINAL",
                reason=str(bracket.exit_reason or close_source),
            )
            if self._fill_ledger is None:
                raise FillLedgerError("fill ledger unavailable after final fill")
            ledger_pnl = self._fill_ledger.realized_pnl(str(bracket.bracket_id))
            if not ledger_pnl.complete:
                raise FillValidationError(
                    f"ledger incomplete entry={ledger_pnl.entry_quantity} exit={ledger_pnl.exit_quantity}"
                )
            ledger_complete = True
        except Exception as exc:  # noqa: BLE001
            setattr(bracket, "_ledger_pending_exit_order_id", order_id or None)
            setattr(bracket, "_ledger_pending_exit_quantity", closing_quantity)
            setattr(bracket, "_ledger_pending_exit_price", resolved_price)
            setattr(bracket, "_ledger_pending_exit_target", "FINAL")
            self._block_ledger_release(
                bracket,
                reason="final_exit_accounting_failed",
                payload={"order_id": order_id, "error": str(exc)},
            )

        with self._lock:
            setattr(bracket, "_final_close_in_progress", False)
            bracket.remaining_quantity = 0
            bracket.exit_executed = True
            bracket.exit_pending = False
            bracket.exit_in_progress = False
            bracket.active = False
            bracket.position_flat_confirmed = True
            bracket.exit_state = _legacy.BracketExitLifecycle.CLOSED.value
            bracket.entry_status = "CLOSED"
            bracket.pending_exit_order_id = None
            positions = getattr(self.order_manager, "_positions", None)
            position_zero = True
            unresolved_exit = False
            if positions is not None:
                getter = getattr(positions, "get_position", None)
                if callable(getter):
                    with suppress(Exception):
                        position_zero = getter(bracket.symbol) is None
                checker = getattr(positions, "is_exit_converging", None)
                if callable(checker):
                    with suppress(Exception):
                        unresolved_exit = bool(checker(bracket.symbol))
            if position_zero and not unresolved_exit:
                bracket.exit_submission_inflight = False
                bracket.exit_intent = None
                bracket.expected_exit_side = None
                bracket.expected_exit_qty = 0
                bracket.exit_correlation_id = None
            bracket.close_source = close_source
            bracket.closed_at = time.time()
            if resolved_price is not None:
                bracket.exit_price = resolved_price
            bracket.updated_at = bracket.closed_at
            self._exit_cooldowns.pop(bracket.entry_order_id, None)
            setattr(
                bracket,
                "ledger_realized_pnl",
                ledger_pnl.to_dict() if ledger_pnl is not None else None,
            )

        snapshot_persisted = True
        try:
            self.save_state()
        except Exception as exc:  # noqa: BLE001
            snapshot_persisted = False
            self._block_ledger_release(
                bracket,
                reason="closed_snapshot_persist_failed",
                payload={"error": str(exc)},
            )

        gross_pnl = ledger_pnl.gross_pnl if ledger_pnl is not None else None
        net_pnl = ledger_pnl.net_pnl if ledger_pnl is not None else None
        if gross_pnl is None and resolved_price is not None:
            entry_price = getattr(bracket, "entry_fill_price", None) or getattr(bracket, "entry_price", None)
            if entry_price is not None:
                try:
                    side_mult = -1.0 if str(getattr(bracket, "side", "BUY")).upper() == "SELL" else 1.0
                    gross_pnl = round((float(resolved_price) - float(entry_price)) * int(bracket.quantity or 0) * side_mult, 2)
                except (TypeError, ValueError):
                    gross_pnl = None
        outcome = self._completed_trade_outcome(
            bracket,
            ledger_pnl=ledger_pnl,
            gross_pnl=gross_pnl,
            exit_price=resolved_price,
            ledger_complete=ledger_complete,
        )
        setattr(bracket, "_completed_trade_outcome", outcome)
        net_pnl = outcome["net_pnl"]
        _legacy.LOGGER.info(
            "BRACKET_CLOSED bracket_id=%s symbol=%s close_source=%s side=%s qty=%s entry=%s exit=%s pnl=%s net_pnl=%s ledger_complete=%s",
            bracket.bracket_id,
            bracket.symbol,
            close_source,
            bracket.side,
            int(bracket.quantity or 0),
            ledger_pnl.entry_vwap if ledger_pnl is not None else bracket.entry_price,
            ledger_pnl.exit_vwap if ledger_pnl is not None else resolved_price,
            gross_pnl,
            net_pnl,
            ledger_complete,
        )
        self._log_bracket_event(
            "BRACKET_CLOSED",
            bracket,
            meta={
                "close_source": close_source,
                "exit_order_id": order_id,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "ledger_complete": ledger_complete,
                "completed_trade": outcome,
            },
        )
        self._notify_open_position_priority("close", bracket.symbol)
        with suppress(Exception):
            self._notify_event(
                "BRACKET_CLOSED",
                {
                    "symbol": bracket.symbol,
                    "quantity": int(bracket.quantity or 0),
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "ledger_complete": ledger_complete,
                    "close_source": close_source,
                    "completed_trade": outcome,
                },
            )

        if ledger_complete and snapshot_persisted:
            self._clear_ledger_release(bracket)
            hook = self._on_exit_complete_hook
            if hook is not None and not getattr(bracket, "_ledger_release_hook_fired", False):
                try:
                    hook(bracket.symbol)
                    setattr(bracket, "_ledger_release_hook_fired", True)
                except Exception:
                    _legacy.LOGGER.exception(
                        "BRACKET_EXIT_COMPLETE_HOOK_FAILED symbol=%s", bracket.symbol
                    )

    def _retry_ledger_block(self, bracket: Any) -> bool:
        try:
            replayed = False
            pending_entry = getattr(bracket, "_ledger_pending_entry_price", None)
            if pending_entry is not None:
                self._record_entry_fill(bracket, float(pending_entry))
                delattr(bracket, "_ledger_pending_entry_price")
                replayed = True

            pending_order = getattr(bracket, "_ledger_pending_exit_order_id", None)
            if pending_order:
                quantity = int(getattr(bracket, "_ledger_pending_exit_quantity", 0) or 0)
                price = getattr(bracket, "_ledger_pending_exit_price", None)
                target = getattr(bracket, "_ledger_pending_exit_target", None)
                if quantity <= 0 or price is None:
                    return False
                self._record_exit_fill(
                    bracket,
                    order_id=str(pending_order),
                    quantity=quantity,
                    price=float(price),
                    target=str(target or "RECOVERED"),
                    reason="LEDGER_RECONCILIATION",
                )
                for name in (
                    "_ledger_pending_exit_order_id",
                    "_ledger_pending_exit_quantity",
                    "_ledger_pending_exit_price",
                    "_ledger_pending_exit_target",
                ):
                    with suppress(AttributeError):
                        delattr(bracket, name)
                replayed = True

            if (
                bracket.exit_state != _legacy.BracketExitLifecycle.CLOSED.value
                and not replayed
            ):
                # Durable block on a still-open bracket with nothing to replay
                # (markers lost, e.g. the state snapshot itself failed to persist
                # before a restart). Keep the entry freeze latched; closure
                # reconciliation clears it once flat + ledger-complete.
                return False

            if bracket.exit_state == _legacy.BracketExitLifecycle.CLOSED.value:
                if not self._safe_position_flat(bracket.symbol):
                    return False
                if self._fill_ledger is None:
                    return False
                pnl = self._fill_ledger.realized_pnl(str(bracket.bracket_id))
                if not pnl.complete:
                    return False
                setattr(bracket, "ledger_realized_pnl", pnl.to_dict())
                self.save_state()
                hook = self._on_exit_complete_hook
                if hook is not None and not getattr(
                    bracket, "_ledger_release_hook_fired", False
                ):
                    setattr(
                        bracket,
                        "_completed_trade_outcome",
                        self._completed_trade_outcome(
                            bracket,
                            ledger_pnl=pnl,
                            gross_pnl=pnl.gross_pnl,
                            exit_price=pnl.exit_vwap,
                            ledger_complete=True,
                        ),
                    )
                    hook(bracket.symbol)
                    setattr(bracket, "_ledger_release_hook_fired", True)
            self._clear_ledger_release(bracket)
            return True
        except Exception as exc:  # noqa: BLE001
            _legacy.LOGGER.error(
                "FILL_LEDGER_RECONCILE_FAILED bracket_id=%s error=%s",
                bracket.bracket_id,
                exc,
            )
            return False

    def _retry_blocked_releases(self) -> None:
        for bracket_id in list(self._ledger_blocked):
            bracket = self._find_bracket_by_id(bracket_id)
            if bracket is not None:
                self._retry_ledger_block(bracket)

    def has_unresolved_exit(self) -> bool:
        self._retry_blocked_releases()
        return bool(self._ledger_blocked) or super().has_unresolved_exit()

    def get_first_unresolved_exit_bracket_id(self) -> str | None:
        self._retry_blocked_releases()
        if self._ledger_blocked:
            return next(iter(self._ledger_blocked))
        return super().get_first_unresolved_exit_bracket_id()


__all__ = ["LedgerBracketManager"]
