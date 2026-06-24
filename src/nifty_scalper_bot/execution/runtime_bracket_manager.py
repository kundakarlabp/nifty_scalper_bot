"""Runtime bracket authority with strict LIVE and compatible non-live closure.

LIVE execution uses the durable ledger release gate.  PAPER/SHADOW/SIMULATION
retain legacy close and hook behaviour so tests, dry runs and dashboards do not
lose functionality when broker fill identity is intentionally absent.
"""

from __future__ import annotations

from contextlib import suppress
import os
import time
from typing import Any

from nifty_scalper_bot.execution import bracket_manager as _legacy
from nifty_scalper_bot.execution.ledger_bracket_manager import LedgerBracketManager


class RuntimeBracketManager(LedgerBracketManager):
    """Final runtime export for the staged lifecycle implementation."""

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
                "ledger_complete": bool(ledger_pnl and ledger_pnl.complete),
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
                    "ledger_complete": bool(ledger_pnl and ledger_pnl.complete),
                    "close_source": close_source,
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
