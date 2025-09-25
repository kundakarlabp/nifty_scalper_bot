"""Risk management helpers tailored for the simplified scalper flow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskManager:
    """Compute protective stop levels for option entries."""

    max_percent: float = 0.05
    atr_multiplier: float = 1.5

    def calculate_stop_loss(
        self, entry_price: float, atr: float, *, side: str = "LONG"
    ) -> float:
        """Return an ATR-based stop-loss for the given ``entry_price``.

        Parameters
        ----------
        entry_price:
            Executed option premium.
        atr:
            Current Average True Range of the option or underlying.
        side:
            Position direction. ``"LONG"`` caps downside while ``"SHORT"`` caps
            upside.  The default mirrors the long-straddle use case.
        """

        if entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if atr < 0:
            raise ValueError("atr must be non-negative")

        buffer = self.atr_multiplier * atr
        direction = side.upper()
        if direction == "SHORT":
            raw = entry_price + buffer
            cap = entry_price * (1.0 + self.max_percent)
            return min(raw, cap)

        # Default: long position.  Clamp so that the stop never trails more than
        # ``max_percent`` below the entry.
        raw = entry_price - buffer
        floor = entry_price * (1.0 - self.max_percent)
        return max(raw, floor)


__all__ = ["RiskManager"]

