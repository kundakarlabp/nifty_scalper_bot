"""Trailing stop controller built on top of broker order modifications."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Optional

from nifty_scalper_bot.storage.journal import AtomicKV
from nifty_scalper_bot.utils.errors import BrokerError, OrderPlacementError
from nifty_scalper_bot.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class TrailingSpec:
    """Configuration describing how the trailing stop should behave."""

    trail_by: float = 10.0
    step: float = 2.0
    min_gap: float = 1.0
    debounce_ms: int = 800


class TrailingStopController:
    """Manage trailing stop adjustments for a single position."""

    def __init__(
        self,
        *,
        symbol: str,
        side: str,
        entry: float,
        sl_order_id: str,
        variety: str,
        spec: TrailingSpec,
        get_ltp: Callable[[str], Optional[float]],
        modify_order: Callable[[str, str, Optional[int], Optional[float]], dict],
        journal: AtomicKV | None = None,
    ) -> None:
        self.symbol = symbol
        self.side = side.upper()
        self.entry = float(entry)
        self.order_id = sl_order_id
        self.variety = variety
        self.spec = spec
        self._get_ltp = get_ltp
        self._modify_order = modify_order
        self._journal = journal
        self._last_trigger: Optional[float] = None
        self._last_move_monotonic = 0.0

        if journal is not None:
            stored = journal.get(self.order_id)
            if isinstance(stored, (int, float)):
                self._last_trigger = float(stored)

    def on_tick(self, tick: dict[str, float] | None = None) -> None:
        """Process the latest price and adjust the stop when required."""

        ltp = self._extract_ltp(tick)
        if ltp is None or ltp <= 0:
            return

        desired = self._compute_trigger(float(ltp))
        if desired is None:
            return

        if not self._should_move(desired):
            return

        try:
            self._modify_order(self.variety, self.order_id, None, desired)
        except (OrderPlacementError, BrokerError) as exc:
            log.warning(
                "Trailing SL modify failed",
                extra={"symbol": self.symbol, "error": str(exc)},
            )
            return

        self._record_trigger(desired)
        log.info(
            "Trailing SL updated",
            extra={
                "symbol": self.symbol,
                "trigger": round(desired, 2),
                "ltp": round(ltp, 2),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    def _extract_ltp(self, tick: dict[str, float] | None) -> float | None:
        if tick is not None:
            value = tick.get("ltp")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass
        price = self._get_ltp(self.symbol)
        if price is None:
            return None
        try:
            return float(price)
        except (TypeError, ValueError):
            return None

    def _compute_trigger(self, ltp: float) -> float | None:
        trail_by = float(self.spec.trail_by)
        min_gap = float(self.spec.min_gap)
        if trail_by <= 0 or min_gap < 0:
            return None
        if self.side == "LONG":
            base = max(ltp - trail_by, self.entry - trail_by)
            if self._last_trigger is not None:
                base = max(base, self._last_trigger)
            trigger = min(base, ltp - min_gap)
        else:
            base = min(ltp + trail_by, self.entry + trail_by)
            if self._last_trigger is not None:
                base = min(base, self._last_trigger)
            trigger = max(base, ltp + min_gap)
        return float(trigger)

    def _should_move(self, new_value: float) -> bool:
        now = time.monotonic()
        if self._last_move_monotonic and (now - self._last_move_monotonic) < (
            self.spec.debounce_ms / 1000.0
        ):
            return False
        last = self._last_trigger
        if last is None:
            return True
        if abs(new_value - last) < float(self.spec.step):
            return False
        if self.side == "LONG" and new_value < last:
            return False
        if self.side == "SHORT" and new_value > last:
            return False
        return True

    def _record_trigger(self, value: float) -> None:
        self._last_trigger = float(value)
        self._last_move_monotonic = time.monotonic()
        if self._journal is not None:
            self._journal.set(self.order_id, self._last_trigger)


__all__ = ["TrailingSpec", "TrailingStopController"]
