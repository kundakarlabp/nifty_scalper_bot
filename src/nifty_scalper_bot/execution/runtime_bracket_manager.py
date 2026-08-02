"""Runtime bracket authority with strict LIVE and compatible non-live closure.

LIVE execution uses the durable ledger release gate. PAPER/SHADOW/SIMULATION
retain legacy close and hook behaviour so tests, dry runs and dashboards do not
lose functionality when broker fill identity is intentionally absent.
"""

from __future__ import annotations

import math
import os
import time
from contextlib import suppress
from typing import Any, Mapping

from nifty_scalper_bot.execution import bracket_manager as _legacy
from nifty_scalper_bot.execution.ledger_bracket_manager import LedgerBracketManager


class RuntimeBracketManager(LedgerBracketManager):
    """Final runtime export for the staged lifecycle implementation."""

    def _minimum_locked_profit_r(self) -> float:
        """Return the cached post-activation net-profit floor in initial-risk units."""
        cached = getattr(self, "_trail_min_locked_profit_r", None)
        if cached is not None:
            return float(cached)
        try:
            value = float(os.getenv("TRAIL_MIN_LOCKED_PROFIT_R", "0.10"))
        except (TypeError, ValueError):
            value = 0.10
        value = value if math.isfinite(value) else 0.10
        self._trail_min_locked_profit_r = max(0.0, value)
        return self._trail_min_locked_profit_r

    def _calculate_tiered_trailing_sl(
        self,
        bracket: Any,
        ltp: float,
        profit_pct: float,
        high_water: float,
        atr: float,
    ) -> float | None:
        """Delegate to the canonical ladder, then enforce one net-profit invariant.

        The inherited core ladder is the only tier implementation. This override
        corrects its Tier-4 dimensional comparison by supplying the canonical
        R-metric only for that legacy percentage check, then floors the single
        returned candidate at round-trip costs plus ``TRAIL_MIN_LOCKED_PROFIT_R``.
        """
        entry = float(getattr(bracket, "entry_price", 0.0) or 0.0)
        current_sl = float(getattr(bracket, "sl_trigger_price", 0.0) or 0.0)
        initial_sl = float(
            getattr(bracket, "initial_sl_trigger_price", 0.0) or current_sl
        )
        initial_risk = abs(entry - initial_sl)
        side = str(getattr(bracket, "side", "") or "").upper()
        profit_points = (ltp - entry) if side == "BUY" else (entry - ltp)
        tier_metric = (profit_points / initial_risk) if initial_risk > 0 else None

        corrected_profit_pct = float(profit_pct)
        if tier_metric is not None and tier_metric >= float(self._trail_tier4_r):
            # bracket_core's final legacy guard compares profit_pct with an R
            # threshold. Supplying the already-computed R value fixes only that
            # dimensional mismatch; all tier calculations still live in super().
            corrected_profit_pct = max(corrected_profit_pct, tier_metric)

        candidate = super()._calculate_tiered_trailing_sl(
            bracket=bracket,
            ltp=ltp,
            profit_pct=corrected_profit_pct,
            high_water=high_water,
            atr=atr,
        )
        if candidate is None or initial_risk <= 0 or tier_metric is None:
            return candidate

        activation_r = float(self._trail_activation_r(bracket))
        if tier_metric < activation_r:
            return candidate

        cost_points = max(0.0, float(self._breakeven_cost_per_unit(bracket)))
        # The positive-profit floor starts only after the canonical 0.75R
        # progress threshold. Strategies that deliberately arm cost-adjusted
        # breakeven earlier retain that established behaviour without installing
        # a noise-sensitive positive-profit stop before meaningful progress.
        locked_r = self._minimum_locked_profit_r() if tier_metric >= 0.75 else 0.0
        locked_points = cost_points + (initial_risk * locked_r)
        room = max(cost_points, 0.05)

        if side == "BUY":
            floor = entry + locked_points
            proposed = max(float(candidate), floor)
            if proposed >= (float(ltp) - room):
                return None
        else:
            floor = entry - locked_points
            proposed = min(float(candidate), floor)
            if proposed <= (float(ltp) + room):
                return None

        if proposed != float(candidate):
            _legacy.LOGGER.info(
                "TRAIL_MIN_PROFIT_FLOOR_APPLIED symbol=%s side=%s tier_metric_r=%.3f locked_r=%.3f cost_points=%.3f candidate_sl=%.2f floored_sl=%.2f ltp=%.2f",
                getattr(bracket, "symbol", ""),
                side,
                tier_metric,
                locked_r,
                cost_points,
                float(candidate),
                proposed,
                float(ltp),
                extra={
                    "event": "TRAIL_MIN_PROFIT_FLOOR_APPLIED",
                    "symbol": getattr(bracket, "symbol", ""),
                    "side": side,
                    "initial_risk_points": initial_risk,
                    "tier_metric_r": tier_metric,
                    "minimum_locked_profit_r": locked_r,
                    "minimum_locked_profit_points": initial_risk * locked_r,
                    "estimated_cost_points": cost_points,
                    "candidate_sl": float(candidate),
                    "applied_sl": proposed,
                    "ltp": float(ltp),
                },
            )
        return proposed

    def _block_ledger_release(
        self,
        bracket: Any,
        *,
        reason: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist enough identity to reconcile a ledger block after restart."""
        release_payload = dict(payload or {})
        release_payload.setdefault("symbol", str(bracket.symbol))
        release_payload.setdefault("bracket_id", str(bracket.bracket_id))
        release_payload.setdefault("exit_state", str(bracket.exit_state or ""))
        release_payload.setdefault(
            "remaining_quantity", int(bracket.remaining_quantity or 0)
        )
        super()._block_ledger_release(
            bracket,
            reason=reason,
            payload=release_payload,
        )

    def _broker_all_positions_flat(self) -> bool | None:
        """Return explicit all-account flatness, or ``None`` when unknowable."""
        try:
            return self._authoritative_position_snapshot().all_flat
        except Exception as exc:  # noqa: BLE001
            _legacy.LOGGER.error(
                "FILL_LEDGER_ORPHAN_POSITION_CHECK_FAILED error=%s",
                exc,
                extra={
                    "event": "FILL_LEDGER_ORPHAN_POSITION_CHECK_FAILED",
                    "error_type": type(exc).__name__,
                },
            )
            return None

    def _retry_orphan_ledger_block(
        self,
        bracket_id: str,
        details: Mapping[str, Any] | None,
    ) -> bool:
        """Clear a restart orphan only after authoritative broker-flat proof."""
        payload = details.get("payload", {}) if isinstance(details, Mapping) else {}
        payload = payload if isinstance(payload, Mapping) else {}
        symbol = str(payload.get("symbol") or "").strip()

        flat: bool | None
        if symbol:
            try:
                quantity = self._broker_position_quantity(symbol)
            except Exception as exc:  # noqa: BLE001 - unknown remains blocked
                _legacy.LOGGER.error(
                    "FILL_LEDGER_ORPHAN_POSITION_CHECK_FAILED bracket_id=%s symbol=%s error=%s",
                    bracket_id,
                    symbol,
                    exc,
                    extra={
                        "event": "FILL_LEDGER_ORPHAN_POSITION_CHECK_FAILED",
                        "bracket_id": str(bracket_id),
                        "symbol": symbol,
                        "error_type": type(exc).__name__,
                    },
                )
                flat = None
            else:
                flat = None if quantity is None else quantity == 0
        else:
            # Legacy rows predate persisted symbol metadata. They may be released
            # only when the broker authoritatively reports the entire account flat.
            flat = self._broker_all_positions_flat()

        if flat is not True:
            _legacy.LOGGER.warning(
                "FILL_LEDGER_ORPHAN_BLOCK_RETAINED bracket_id=%s symbol=%s broker_flat=%s",
                bracket_id,
                symbol or "unknown",
                flat,
                extra={
                    "event": "FILL_LEDGER_ORPHAN_BLOCK_RETAINED",
                    "bracket_id": str(bracket_id),
                    "symbol": symbol or None,
                    "broker_flat": flat,
                },
            )
            return False

        self._ledger_blocked.pop(str(bracket_id), None)
        if self._release_store is not None:
            with suppress(Exception):
                self._release_store.clear(str(bracket_id))
        _legacy.LOGGER.warning(
            "FILL_LEDGER_ORPHAN_BLOCK_CLEARED bracket_id=%s symbol=%s",
            bracket_id,
            symbol or "account_flat",
            extra={
                "event": "FILL_LEDGER_ORPHAN_BLOCK_CLEARED",
                "bracket_id": str(bracket_id),
                "symbol": symbol or None,
            },
        )
        return True

    def _retry_blocked_releases(self) -> None:
        """Retry live brackets and reconcile restart-orphaned ledger rows."""
        for bracket_id in list(self._ledger_blocked):
            bracket = self._find_bracket_by_id(bracket_id)
            if bracket is not None:
                self._retry_ledger_block(bracket)
                continue
            self._retry_orphan_ledger_block(
                str(bracket_id), self._ledger_blocked.get(str(bracket_id))
            )

    def _strict_ledger_release_required(self) -> bool:
        checker = getattr(self.order_manager, "is_live_mode", None)
        if callable(checker):
            with suppress(Exception):
                return bool(checker())
        mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").upper()
        live_flag = str(
            os.getenv("ENABLE_LIVE")
            or os.getenv("ENABLE_LIVE_TRADING")
            or "false"
        ).lower() in {"1", "true", "yes", "on"}
        return mode == "LIVE" and live_flag

    def confirm_partial_entry_fill(
        self,
        order_id: str,
        filled_quantity: int,
        fill_price: float,
    ) -> bool:
        """Arm protection for the final broker-confirmed partial entry quantity."""

        bracket = self.get_bracket(order_id)
        if bracket is None:
            _legacy.LOGGER.critical(
                "PARTIAL_ENTRY_BRACKET_MISSING order_id=%s qty=%s price=%s",
                order_id,
                filled_quantity,
                fill_price,
                extra={
                    "event": "PARTIAL_ENTRY_BRACKET_MISSING",
                    "order_id": order_id,
                    "filled_quantity": filled_quantity,
                },
            )
            return False
        try:
            quantity = int(filled_quantity)
            price = float(fill_price)
        except (TypeError, ValueError):
            return False
        if quantity <= 0 or price <= 0:
            return False

        with self._lock:
            planned_quantity = int(bracket.quantity or 0)
            if planned_quantity > 0:
                quantity = min(quantity, planned_quantity)
            bracket.quantity = quantity
            bracket.remaining_quantity = quantity
            allocation_left = quantity
            retained_targets = []
            for target in bracket.tp_levels:
                allocated = min(int(target.quantity or 0), allocation_left)
                if allocated <= 0:
                    continue
                target.quantity = allocated
                retained_targets.append(target)
                allocation_left -= allocated
            bracket.tp_levels = retained_targets
            bracket.updated_at = time.time()

        self.confirm_entry_fill(order_id, price)
        with suppress(Exception):
            self.save_state()
        _legacy.LOGGER.critical(
            "PARTIAL_ENTRY_PROTECTED order_id=%s symbol=%s planned_qty=%s filled_qty=%s fill_price=%.2f",
            order_id,
            bracket.symbol,
            planned_quantity,
            quantity,
            price,
            extra={
                "event": "PARTIAL_ENTRY_PROTECTED",
                "order_id": order_id,
                "symbol": bracket.symbol,
                "planned_quantity": planned_quantity,
                "filled_quantity": quantity,
                "fill_price": price,
            },
        )
        with suppress(Exception):
            self._notify_event(
                "PARTIAL_ENTRY_PROTECTED",
                {
                    "symbol": bracket.symbol,
                    "planned_quantity": planned_quantity,
                    "filled_quantity": quantity,
                    "fill_price": round(price, 2),
                    "sl": round(float(bracket.sl_trigger_price), 2),
                    "tp": round(float(bracket.tp_trigger_price), 2),
                },
            )
        return True

    def _close_bracket(
        self,
        bracket: Any,
        *,
        close_source: str,
        exit_price: float | None = None,
    ) -> None:
        if self._strict_ledger_release_required():
            super()._close_bracket(
                bracket,
                close_source=close_source,
                exit_price=exit_price,
            )
            return

        order_id = str(bracket.exit_order_id or bracket.pending_exit_order_id or "")
        closing_quantity = int(bracket.remaining_quantity or 0)
        resolved_price = self._resolved_exit_price(bracket, exit_price)
        ledger_pnl = None
        if order_id and closing_quantity > 0 and resolved_price is not None:
            with suppress(Exception):
                self._record_exit_fill(
                    bracket,
                    order_id=order_id,
                    quantity=closing_quantity,
                    price=resolved_price,
                    target="FINAL",
                    reason=str(bracket.exit_reason or close_source),
                )
                if self._fill_ledger is not None:
                    candidate = self._fill_ledger.realized_pnl(str(bracket.bracket_id))
                    if candidate.complete:
                        ledger_pnl = candidate

        with self._lock:
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

        entry_px = (
            ledger_pnl.entry_vwap
            if ledger_pnl is not None
            else (
                bracket.entry_fill_price
                if bracket.entry_fill_price is not None
                else bracket.entry_price
            )
        )
        final_exit_px = (
            ledger_pnl.exit_vwap if ledger_pnl is not None else bracket.exit_price
        )
        filled_qty = int(bracket.quantity or 0)
        realized_pnl = ledger_pnl.gross_pnl if ledger_pnl is not None else None
        if realized_pnl is None:
            try:
                if entry_px is not None and final_exit_px is not None and filled_qty > 0:
                    if bracket.side == "BUY":
                        realized_pnl = round(
                            (float(final_exit_px) - float(entry_px)) * filled_qty,
                            2,
                        )
                    else:
                        realized_pnl = round(
                            (float(entry_px) - float(final_exit_px)) * filled_qty,
                            2,
                        )
            except Exception:  # noqa: BLE001
                realized_pnl = None

        try:
            self.save_state()
        except Exception as exc:  # noqa: BLE001
            _legacy.LOGGER.error(
                "BRACKET_CLOSE_PERSIST_FAILED bracket_id=%s error=%s",
                bracket.bracket_id,
                exc,
            )

        outcome = self._completed_trade_outcome(
            bracket,
            ledger_pnl=ledger_pnl,
            gross_pnl=realized_pnl,
            exit_price=final_exit_px,
            ledger_complete=bool(ledger_pnl and ledger_pnl.complete),
        )
        setattr(bracket, "_completed_trade_outcome", outcome)
        _legacy.LOGGER.info(
            "BRACKET_CLOSED bracket_id=%s symbol=%s close_source=%s side=%s qty=%s entry=%s exit=%s pnl=%s",
            bracket.bracket_id,
            bracket.symbol,
            close_source,
            bracket.side,
            filled_qty,
            entry_px,
            final_exit_px,
            realized_pnl,
        )
        self._log_bracket_event(
            "BRACKET_CLOSED",
            bracket,
            meta={
                "close_source": close_source,
                "exit_order_id": order_id,
                "side": bracket.side,
                "qty": filled_qty,
                "entry": entry_px,
                "exit": final_exit_px,
                "pnl": realized_pnl,
                "net_pnl": outcome["net_pnl"],
                "ledger_complete": bool(ledger_pnl and ledger_pnl.complete),
                "completed_trade": outcome,
            },
        )
        self._notify_open_position_priority("close", bracket.symbol)
        with suppress(Exception):
            self._notify_event(
                "BRACKET_CLOSED",
                {
                    "symbol": bracket.symbol,
                    "quantity": filled_qty,
                    "gross_pnl": realized_pnl,
                    "net_pnl": outcome["net_pnl"],
                    "ledger_complete": bool(ledger_pnl and ledger_pnl.complete),
                    "close_source": close_source,
                    "completed_trade": outcome,
                },
            )
        self._clear_ledger_release(bracket)
        hook = self._on_exit_complete_hook
        if hook is not None:
            try:
                hook(bracket.symbol)
            except Exception:
                _legacy.LOGGER.exception(
                    "BRACKET_EXIT_COMPLETE_HOOK_FAILED symbol=%s", bracket.symbol
                )


__all__ = ["RuntimeBracketManager"]
