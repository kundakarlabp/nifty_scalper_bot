"""Execution policy helpers for NSE index options."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time
import uuid

from nifty_scalper_bot.utils.errors import OrderPlacementError


@dataclass(slots=True)
class OptionsExecutionPolicy:
    """Apply tick, lot and latency constraints before firing orders.

    Example:
        >>> policy = OptionsExecutionPolicy()
        >>> policy.round_to_tick(123.72)
        123.7
    """

    tick_size: float = 0.05
    max_spread: float = 3.0
    max_tick_age_ms: int = 400
    max_order_value: float = 2_000_000
    lot_size_lookup: Callable[[str], int | None] | None = field(
        default=None, repr=False
    )
    time_ns: Callable[[], int] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Nothing to initialise yet; attributes are configured via setters.
        return

    def set_lot_size_lookup(self, lookup: Callable[[str], int | None] | None) -> None:
        """Provide a callable resolving lot sizes for option symbols."""

        self.lot_size_lookup = lookup

    def set_time_provider(self, provider: Callable[[], int] | None) -> None:
        """Override the nanosecond clock source used for staleness checks."""

        self.time_ns = provider

    def _resolve_lot_size(self, symbol: str) -> int:
        lookup = self.lot_size_lookup
        if not callable(lookup):
            raise OrderPlacementError("Lot size resolver unavailable")
        try:
            resolved = lookup(symbol)
        except Exception as exc:  # noqa: BLE001 - surface resolver failure
            raise OrderPlacementError("Failed to resolve lot size") from exc
        if resolved is None:
            raise OrderPlacementError("Lot size resolver returned no value")
        try:
            lot_size = int(resolved)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise OrderPlacementError(
                "Lot size resolver returned invalid value"
            ) from exc
        if lot_size <= 0:
            raise OrderPlacementError("Lot size must be positive")
        return lot_size

    def round_to_tick(self, price: float) -> float:
        """Round an order price to the nearest allowed tick.

        Args:
            price: Raw price from the strategy.

        Returns:
            Price aligned to the configured tick size.
        """

        rounded = round(price / self.tick_size) * self.tick_size
        return round(rounded + 1e-12, 2)

    def validate_qty(self, symbol: str, qty: int) -> None:
        """Ensure quantity respects the configured lot size.

        Args:
            symbol: Trading symbol (e.g. ``"NIFTY"``).
            qty: Requested quantity.

        Raises:
            OrderPlacementError: If the quantity violates policy rules.
        """

        if qty <= 0:
            raise OrderPlacementError("Quantity must be positive")
        normalized = (symbol or "").strip().upper()
        lot_size = self._resolve_lot_size(normalized or symbol)
        if qty % lot_size != 0:
            raise OrderPlacementError(f"Quantity must be a multiple of {lot_size}")

    def validate_notional(self, price: float, qty: int) -> None:
        """Verify the order value stays under the configured ceiling.

        Args:
            price: Price after tick rounding.
            qty: Quantity to be traded.

        Raises:
            OrderPlacementError: If the notional exceeds limits.
        """

        notional = price * qty
        if notional > self.max_order_value:
            raise OrderPlacementError("Order value exceeds execution policy limit")

    def price_guard(self, bid: float, ask: float, last_ts_ns: int) -> None:
        """Validate spread and staleness bounds before submitting.

        Args:
            bid: Current best bid price.
            ask: Current best ask price.
            last_ts_ns: Timestamp of the quote in nanoseconds.

        Raises:
            OrderPlacementError: If the quote is stale or too wide.
        """

        if ask <= bid:
            raise OrderPlacementError("Invalid market: ask <= bid")
        spread = ask - bid
        if spread > self.max_spread:
            raise OrderPlacementError("Market spread too wide")
        time_provider = self.time_ns
        if callable(time_provider):
            now_ns = int(time_provider())
        else:
            now_ns = time.time_ns()
        age_ms = (now_ns - last_ts_ns) / 1_000_000
        if age_ms > self.max_tick_age_ms:
            raise OrderPlacementError("Quote is stale")

    def client_order_id(
        self,
        symbol: str,
        side: str,
        qty: int,
        *,
        nonce: str | None = None,
    ) -> str:
        """Return a deterministic client order identifier.

        Args:
            symbol: Trading symbol.
            side: Side of the order (``"BUY"`` or ``"SELL"``).
            qty: Quantity for the order.

        Returns:
            Deterministic UUID string scoped to the order payload.
        """

        payload_nonce = nonce if nonce is not None else uuid.uuid4().hex
        payload = f"{symbol.upper()}|{side.upper()}|{qty}|{payload_nonce}"
        identifier = uuid.uuid5(uuid.NAMESPACE_URL, payload)
        return str(identifier)


__all__ = ["OptionsExecutionPolicy"]
