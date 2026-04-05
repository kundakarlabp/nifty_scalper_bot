"""Position lifecycle: SL/TP/trailing stop evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .bracket import BracketManager, ExitSignal

if TYPE_CHECKING:
    from ..indicators.engine import IndicatorSnapshot
    from ..config.settings import LifecycleSettings


class LifecycleManager:
    """Evaluates all open brackets every 100ms loop iteration."""

    def __init__(self, bracket_manager: BracketManager, settings: "LifecycleSettings") -> None:
        self._brackets = bracket_manager
        self._cfg = settings
        self._position_open_times: dict[str, datetime] = {}

    def register_position(self, position_id: str) -> None:
        self._position_open_times[position_id] = datetime.now(timezone.utc)

    def evaluate_all(
        self, snapshots: dict[str, "IndicatorSnapshot"]
    ) -> list[ExitSignal]:
        """Return list of exit signals for positions that hit SL/TP/trail/time."""
        exits = []
        now = datetime.now(timezone.utc)

        for bracket in self._brackets.get_all():
            sym = bracket.symbol
            snap = snapshots.get(sym)
            if snap is None:
                continue

            current = snap.ltp
            atr = snap.atr_14 or 0.0
            bracket.last_evaluated_at = now

            # Time stop
            opened_at = self._position_open_times.get(bracket.position_id)
            if opened_at:
                elapsed_min = (now - opened_at).total_seconds() / 60.0
                if elapsed_min >= self._cfg.time_stop_min:
                    exits.append(ExitSignal(
                        position_id=bracket.position_id,
                        bracket_id=bracket.bracket_id,
                        symbol=sym,
                        quantity=bracket.remaining_quantity,
                        reason="TIME",
                    ))
                    continue

            # SL check (long only — current < sl triggers exit)
            if current <= bracket.sl_price:
                exits.append(ExitSignal(
                    position_id=bracket.position_id,
                    bracket_id=bracket.bracket_id,
                    symbol=sym,
                    quantity=bracket.remaining_quantity,
                    reason="SL",
                ))
                continue

            # Trailing stop check
            if bracket.trail_active and bracket.trail_price is not None:
                if current <= bracket.trail_price:
                    exits.append(ExitSignal(
                        position_id=bracket.position_id,
                        bracket_id=bracket.bracket_id,
                        symbol=sym,
                        quantity=bracket.remaining_quantity,
                        reason="TRAIL",
                    ))
                    continue
                # Update trail
                if bracket.peak_price is None or current > bracket.peak_price:
                    bracket.peak_price = current
                if bracket.peak_price and atr > 0:
                    new_trail = bracket.peak_price - (atr * self._cfg.trail_atr_mult)
                    new_trail = max(new_trail, bracket.entry_price)  # never below breakeven
                    bracket.trail_price = new_trail

            # TP1 (partial exit)
            if not bracket.tp1_hit and current >= bracket.tp1_price:
                partial_qty = max(1, int(bracket.remaining_quantity * self._cfg.tp1_partial))
                exits.append(ExitSignal(
                    position_id=bracket.position_id,
                    bracket_id=bracket.bracket_id,
                    symbol=sym,
                    quantity=partial_qty,
                    reason="TP1",
                ))
                bracket.tp1_hit = True
                bracket.remaining_quantity -= partial_qty
                bracket.trail_active = True
                bracket.peak_price = current
                if atr > 0:
                    bracket.trail_price = current - (atr * self._cfg.trail_atr_mult)
                continue

            # TP2 (full remaining exit)
            if bracket.tp1_hit and current >= bracket.tp2_price:
                exits.append(ExitSignal(
                    position_id=bracket.position_id,
                    bracket_id=bracket.bracket_id,
                    symbol=sym,
                    quantity=bracket.remaining_quantity,
                    reason="TP2",
                ))

        return exits

    def remove_position(self, position_id: str) -> None:
        self._position_open_times.pop(position_id, None)
