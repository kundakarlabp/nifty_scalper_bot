"""
World-class ATR-based trailing stop controller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from nifty_scalper_bot.indicators.atr_provider import SafeATRProvider
from nifty_scalper_bot.utils.logging import get_logger

# --- DATA STRUCTURES ---


@dataclass
class TrailingSpec:
    """Configuration for trailing stop behavior."""

    trail_by: float  # Initial/Fallback trailing distance
    step: float  # Minimum price move to update SL
    activation: float  # Profit % before trailing activates


# --- CONTROLLER ---


class AdaptiveTrailingController:
    """
    World-class ATR-based trailing stop with:
    - Volatility regime detection
    - Data staleness protection
    - Graceful degradation
    """

    def __init__(
        self,
        symbol: str,
        side: Literal["LONG", "SHORT"],
        entry: float,
        sl_order_id: str,
        variety: str,
        spec: TrailingSpec,
        get_ltp: Callable[[str], float | None],
        modify_order: Callable,
        atr_provider: SafeATRProvider,
        journal: Any,  # Accepts AtomicKV or MockJournal
        atr_multiplier: float = 2.0,
    ):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry
        self.current_sl = entry  # Will be updated
        self.sl_order_id = sl_order_id
        self.variety = variety

        self.spec = spec
        self._get_ltp = get_ltp
        self._modify = modify_order
        self._atr = atr_provider
        self._journal = journal
        self._logger = get_logger(__name__)
        self.atr_multiplier = atr_multiplier

        # State tracking
        self.trailing_active = False
        self.highest_price = entry if side == "LONG" else entry
        self.lowest_price = entry if side == "SHORT" else entry
        self.last_update_time = time.time()
        self.update_count = 0
        self.failed_modifications = 0
        self._last_atr_value = 0.0

        # Recovery backoff — instead of permanent halt, use exponential backoff.
        # After each failure group we wait longer, but always eventually retry.
        self._halted = False
        self._halt_reason: str | None = None
        self._halt_until: float = 0.0        # monotonic; 0 = not halted
        self._halt_backoff_sec: float = 5.0  # doubles on each halt, max 300 s

    def on_tick(self, tick: dict | None = None) -> None:
        """Process price tick and update trailing stop if needed"""

        # 1. SAFETY: Check if halted (timed backoff, NOT permanent)
        if self._halted:
            if time.monotonic() < self._halt_until:
                return  # Still in backoff window
            # Backoff window expired — reset and retry
            self._halted = False
            self.failed_modifications = 0
            self._logger.info(
                "Trailing stop backoff expired for %s — resuming",
                self.symbol,
            )

        # 2. VALIDATE LTP
        # We accept tick data or fetch it
        ltp = None
        if tick:
            ltp = tick.get("ltp") or tick.get("last_price")

        if not ltp:
            ltp = self._get_ltp(self.symbol)

        if ltp is None or float(ltp) <= 0:
            return  # Silent return to avoid log spam on missing ticks

        ltp = float(ltp)

        # Update High/Low Watermarks
        if self.side == "LONG":
            if ltp > self.highest_price:
                self.highest_price = ltp
        else:
            if ltp < self.lowest_price:
                self.lowest_price = ltp

        # 3. FETCH ATR WITH VALIDATION
        atr_snapshot = self._atr.get_atr(self.symbol, fallback=self.spec.trail_by)
        if atr_snapshot is None:
            trail_distance = self.entry_price * 0.005
            self._last_atr_value = float(trail_distance)
        else:
            if not atr_snapshot.is_fresh(max_age_sec=60.0):
                return
            self._last_atr_value = float(atr_snapshot.value)
            trail_distance = self._calculate_trail_distance(atr_snapshot, ltp)

        # 4. CHECK ACTIVATION
        profit_pct = self._calculate_profit_pct(ltp)
        activation_pct = max(
            ((self._last_atr_value * 0.2) / self.entry_price) * 100,
            0.5,
        )
        if not self.trailing_active:
            if profit_pct >= activation_pct:
                self.trailing_active = True
                self._logger.info(
                    "🚀 Trailing stop ACTIVATED for %s at %.2f%% profit",
                    self.symbol,
                    profit_pct,
                    extra={"event": "trailing_activated", "profit_pct": profit_pct},
                )
            else:
                return  # Not profitable enough yet

        # 5. UPDATE STOP LOSS IF NEEDED
        new_sl = self._calculate_new_sl(ltp, trail_distance)

        try:
            if self._should_update_sl(new_sl):
                success = self._execute_sl_update(new_sl)

                if not success:
                    self.failed_modifications += 1
                    if self.failed_modifications >= 5:
                        self._emergency_halt("5 consecutive SL modification failures")
                else:
                    self.failed_modifications = 0
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in on_tick: %s", exc)

    def _calculate_trail_distance(self, atr: Any, ltp: float) -> float:
        """Calculate dynamic trail distance based on ATR and regime."""
        base_atr = atr.value

        # Detect volatility regime
        atr_pct_of_price = (base_atr / ltp) * 100

        multiplier = self.atr_multiplier
        if atr_pct_of_price < 1.0:
            multiplier = self.atr_multiplier * 0.75  # Tighten
        elif atr_pct_of_price > 3.0:
            multiplier = self.atr_multiplier * 1.5  # Widen

        return base_atr * multiplier

    def _calculate_new_sl(self, ltp: float, trail_distance: float) -> float:
        """Calculate new stop loss based on current price and trail distance"""
        if self.side == "LONG":
            # For LONG, SL is Price - Distance.
            # We peg to Highest Price to prevent SL moving down when price drops.
            anchor = self.highest_price
            return anchor - trail_distance
        else:  # SHORT
            anchor = self.lowest_price
            return anchor + trail_distance

    def _should_update_sl(self, new_sl: float) -> bool:
        """Check if SL should be updated (Ratchet Logic)."""
        if self.side == "LONG":
            # Only move UP
            if new_sl <= self.current_sl:
                return False
            improvement = new_sl - self.current_sl
        else:  # SHORT
            # Only move DOWN
            if new_sl >= self.current_sl:
                return False
            improvement = self.current_sl - new_sl

        # Check minimum step
        min_step = max(self.spec.step, self._last_atr_value * 0.1)
        if improvement < min_step:
            return False

        return True

    def _execute_sl_update(self, new_sl: float) -> bool:
        """Execute stop loss modification."""
        old_sl = self.current_sl

        try:
            new_sl_rounded = round(new_sl, 1)

            # Attempt modification via callback
            result = self._modify(self.sl_order_id, new_sl_rounded)

            if result:
                self.current_sl = new_sl_rounded
                self.last_update_time = time.time()
                self.update_count += 1

                # Try to log to journal if available
                if hasattr(self._journal, "set"):
                    self._journal.set(
                        self.sl_order_id,
                        {
                            "current_sl": new_sl_rounded,
                            "last_update": self.last_update_time,
                            "update_count": self.update_count,
                        },
                    )

                self._logger.info(
                    f"✅ Trailing SL updated: {old_sl:.2f} → {new_sl_rounded:.2f}",
                    extra={
                        "event": "trailing_sl_updated",
                        "symbol": self.symbol,
                        "new_sl": new_sl_rounded,
                    },
                )
                return True
            else:
                return False

        except Exception as exc:
            self._logger.error(f"SL mod error: {exc}")
            return False

    def _emergency_halt(self, reason: str) -> None:
        """Temporarily pause trailing updates with exponential backoff.
        
        Instead of a permanent halt (which leaves positions unprotected for the
        entire session after any transient broker API error), we back off for an
        increasing interval and then automatically retry.
        """
        if self._halted:
            return
        self._halted = True
        self._halt_reason = reason
        self._halt_until = time.monotonic() + self._halt_backoff_sec
        # Double backoff for next event, capped at 5 minutes
        self._halt_backoff_sec = min(self._halt_backoff_sec * 2, 300.0)
        self._logger.warning(
            "⏸️ Trailing backoff for %s (%.0fs): %s",
            self.symbol,
            self._halt_backoff_sec / 2,  # log the actual wait, not next
            reason,
        )

    def _calculate_profit_pct(self, ltp: float) -> float:
        """Calculate current profit percentage"""
        if self.side == "LONG":
            return ((ltp - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - ltp) / self.entry_price) * 100
