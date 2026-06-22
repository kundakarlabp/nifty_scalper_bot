"""Safety-hardened adaptive trailing controller.

This module preserves the public controller contract while correcting live-money
edge cases in activation, stale ATR degradation, tick rounding, and stop ratchets.
"""

from __future__ import annotations

import math
import time
from typing import Any

from nifty_scalper_bot.execution import adaptive_trailing as _legacy


_LegacyAdaptiveTrailingController = _legacy.AdaptiveTrailingController


class HardenedAdaptiveTrailingController(_LegacyAdaptiveTrailingController):
    """Adaptive trailing controller with deterministic protective invariants."""

    _TICK_SIZE = 0.05

    def on_tick(self, tick: dict | None = None) -> None:
        """Process one tick without ever weakening an established stop."""
        if self._halted:
            if time.monotonic() < self._halt_until:
                return
            self._halted = False
            self.failed_modifications = 0
            self._logger.info(
                "Trailing stop backoff expired for %s — resuming",
                self.symbol,
            )

        ltp: Any = None
        if tick:
            ltp = tick.get("ltp") or tick.get("last_price")
        if not ltp:
            ltp = self._get_ltp(self.symbol)
        try:
            ltp = float(ltp)
        except (TypeError, ValueError):
            return
        if not math.isfinite(ltp) or ltp <= 0 or self.entry_price <= 0:
            return

        if self.side == "LONG":
            self.highest_price = max(float(self.highest_price), ltp)
        else:
            self.lowest_price = min(float(self.lowest_price), ltp)

        atr_snapshot = self._atr.get_atr(self.symbol, fallback=self.spec.trail_by)
        atr_fresh = bool(
            atr_snapshot is not None
            and getattr(atr_snapshot, "is_fresh", lambda **_: True)(max_age_sec=60.0)
        )
        if atr_fresh:
            try:
                atr_value = float(atr_snapshot.value)
            except (TypeError, ValueError, AttributeError):
                atr_value = 0.0
        else:
            atr_value = 0.0

        if not math.isfinite(atr_value) or atr_value <= 0:
            # Continue protecting the position when ATR is temporarily stale.
            configured = max(float(self.spec.trail_by or 0.0), self._TICK_SIZE)
            trail_distance = min(configured, self.entry_price * 0.10)
            trail_distance = max(trail_distance, self.entry_price * 0.005)
            self._last_atr_value = trail_distance
            self._logger.warning(
                "TRAILING_ATR_DEGRADED symbol=%s fallback_distance=%.2f",
                self.symbol,
                trail_distance,
                extra={
                    "event": "TRAILING_ATR_DEGRADED",
                    "symbol": self.symbol,
                    "fallback_distance": trail_distance,
                },
            )
        else:
            self._last_atr_value = atr_value
            trail_distance = self._calculate_trail_distance(atr_snapshot, ltp)

        configured_activation = max(float(self.spec.activation or 0.0), 0.0)
        volatility_activation = (
            (self._last_atr_value * 0.2) / self.entry_price
        ) * 100.0
        activation_pct = max(configured_activation, volatility_activation, 0.10)
        profit_pct = self._calculate_profit_pct(ltp)
        if not self.trailing_active:
            if profit_pct < activation_pct:
                return
            self.trailing_active = True
            self._logger.info(
                "TRAILING_STOP_ACTIVATED symbol=%s profit_pct=%.3f threshold_pct=%.3f",
                self.symbol,
                profit_pct,
                activation_pct,
                extra={
                    "event": "TRAILING_STOP_ACTIVATED",
                    "symbol": self.symbol,
                    "profit_pct": profit_pct,
                    "activation_pct": activation_pct,
                },
            )

        new_sl = self._calculate_new_sl(ltp, trail_distance)
        if not math.isfinite(new_sl) or new_sl <= 0:
            return

        # A protective stop must remain on the non-triggered side of the current
        # market price. The bracket manager independently enforces the same rule.
        if self.side == "LONG":
            new_sl = min(new_sl, ltp - self._TICK_SIZE)
        else:
            new_sl = max(new_sl, ltp + self._TICK_SIZE)
        if new_sl <= 0:
            return

        try:
            if not self._should_update_sl(new_sl):
                return
            success = self._execute_sl_update(new_sl)
            if success:
                self.failed_modifications = 0
                return
            self.failed_modifications += 1
            if self.failed_modifications >= 5:
                self._emergency_halt("5 consecutive SL modification failures")
        except Exception as exc:  # noqa: BLE001 - trailing must not stop hard SL checks
            self._logger.error(
                "TRAILING_TICK_FAILED symbol=%s error=%s",
                self.symbol,
                exc,
                extra={
                    "event": "TRAILING_TICK_FAILED",
                    "symbol": self.symbol,
                    "error_type": type(exc).__name__,
                },
            )

    def _calculate_trail_distance(self, atr: Any, ltp: float) -> float:
        distance = float(super()._calculate_trail_distance(atr, ltp))
        if not math.isfinite(distance) or distance <= 0:
            return max(self.entry_price * 0.005, self._TICK_SIZE)
        # Prevent a malformed ATR snapshot from effectively disabling the stop.
        return min(distance, max(ltp * 0.20, self._TICK_SIZE))

    def _execute_sl_update(self, new_sl: float) -> bool:
        """Commit a broker-compatible 0.05-tick ratchet update."""
        old_sl = float(self.current_sl)
        rounded = round(round(float(new_sl) / self._TICK_SIZE) * self._TICK_SIZE, 2)
        if self.side == "LONG" and rounded <= old_sl:
            return False
        if self.side == "SHORT" and rounded >= old_sl:
            return False
        try:
            result = self._modify(self.sl_order_id, rounded)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "TRAILING_SL_MODIFY_FAILED symbol=%s error=%s",
                self.symbol,
                exc,
                extra={
                    "event": "TRAILING_SL_MODIFY_FAILED",
                    "symbol": self.symbol,
                    "error_type": type(exc).__name__,
                },
            )
            return False
        if not result:
            return False

        self.current_sl = rounded
        self.last_update_time = time.time()
        self.update_count += 1
        if hasattr(self._journal, "set"):
            self._journal.set(
                self.sl_order_id,
                {
                    "current_sl": rounded,
                    "last_update": self.last_update_time,
                    "update_count": self.update_count,
                },
            )
        self._logger.info(
            "TRAILING_SL_UPDATED symbol=%s old_sl=%.2f new_sl=%.2f",
            self.symbol,
            old_sl,
            rounded,
            extra={
                "event": "TRAILING_SL_UPDATED",
                "symbol": self.symbol,
                "old_sl": old_sl,
                "new_sl": rounded,
            },
        )
        return True


__all__ = ["HardenedAdaptiveTrailingController"]
