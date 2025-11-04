"""Deterministic entry price model used before submitting orders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["BUY", "SELL"]


@dataclass(slots=True)
class EntryPriceModel:
    """Compute tick-aligned entry prices from the live quote."""

    tick: float = 0.05
    pct_of_spread: float = 0.75
    min_ticks_from_best: int = 0
    clamp_spread: float | None = None

    def compute(self, *, side: Side, bid: float, ask: float) -> float:
        """Return the executable price based on the latest book state.

        Args:
            side: Trading side (``"BUY"`` or ``"SELL"``).
            bid: Current best bid price.
            ask: Current best ask price.

        Returns:
            Tick-aligned price positioned inside the spread.

        Raises:
            ValueError: If the quote is invalid (``ask <= bid``).
        """

        normalized_side = side.upper()
        if ask <= bid:
            raise ValueError("Invalid book (ask <= bid)")
        spread = ask - bid
        if self.clamp_spread is not None:
            spread = min(spread, self.clamp_spread)
        if normalized_side == "BUY":
            raw = bid + self.pct_of_spread * spread
        else:
            raw = ask - self.pct_of_spread * spread
        nudge = self.min_ticks_from_best * self.tick
        if normalized_side == "BUY":
            raw = max(raw, min(ask, bid + nudge))
        else:
            raw = min(raw, max(bid, ask - nudge))
        return round(round(raw / self.tick) * self.tick + 1e-12, 2)


__all__ = ["EntryPriceModel", "Side"]
