"""Margin planning helpers for order execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, time
from typing import Any, Literal, Mapping

import nifty_scalper_bot.config.settings as app_settings
from nifty_scalper_bot.utils.logging import get_logger

OrderSide = Literal["BUY", "SELL"]


@dataclass(frozen=True, slots=True)
class MarginInputs:
    """Immutable inputs required for pre-trade margin assessment."""

    symbol: str
    side: OrderSide
    price: float
    stop_loss: float | None
    atr: float | None
    requested_qty: int
    product: str | None
    lot_size: int
    balance: float
    per_trade_risk_pct: float
    per_trade_cap_pct: float
    margin_factor: float
    margin_buffer: float
    contract_multiplier: float
    ist_now: datetime
    min_lots_per_trade: int
    max_lots_per_trade: int
    atr_multiple: float


@dataclass(slots=True)
class MarginDecision:
    """Result of a margin evaluation before placing an order."""

    ok: bool
    reason: str | None
    order_type: str
    quantity: int
    est_required: float
    available: float
    sizing: "SizingResult | None" = None


@dataclass(slots=True)
class SizingResult:
    """Detailed sizing context for logging and skip decisions."""

    qty: int
    reason: str | None = None
    needed: float | None = None
    available: float | None = None


class _BrokerMarginUnavailable(RuntimeError):
    """Concrete broker margin operation exists but did not return usable data."""


MIS_CUTOFF = time(15, 25)


class MarginEngine:
    """Evaluate margin availability and sizing before broker submission."""

    def __init__(
        self,
        *,
        broker: Any,
        data_hub: Any,
        lot_size_resolver: Any,
        clock: Any,
    ) -> None:
        self._broker = broker
        self._lots = lot_size_resolver
        self._clock = clock
        self._logger = get_logger(__name__)
        self._data_hub: Any | None = None
        self.set_data_hub(data_hub)

    def set_data_hub(self, hub: Any | None) -> None:
        self._data_hub = hub
        self._logger.info(
            "Condition met: margin_engine_data_hub_attached",
            extra={
                "event": "margin_engine_data_hub_attached",
                "attached": hub is not None,
            },
        )

    def plan(self, inputs: MarginInputs) -> MarginDecision:
        """Return the largest safe whole-lot quantity not exceeding the request."""
        fallback_balance = max(0.0, float(inputs.balance))
        requested_qty = int(inputs.requested_qty)
        lot_size = max(1, int(inputs.lot_size))
        order_type = self._resolve_order_type(inputs.ist_now, inputs.product)

        if requested_qty <= 0 or requested_qty % lot_size:
            return MarginDecision(
                ok=False,
                reason=(
                    "invalid_requested_quantity"
                    if requested_qty <= 0
                    else "invalid_lot_quantity"
                ),
                order_type=order_type,
                quantity=0,
                est_required=0.0,
                available=fallback_balance,
            )

        session_reason = self._session_reason(inputs.ist_now, order_type)
        if session_reason:
            return MarginDecision(
                ok=False,
                reason=session_reason,
                order_type=order_type,
                quantity=0,
                est_required=0.0,
                available=fallback_balance,
            )

        available = self._resolve_available_margin(fallback_balance)
        effective_inputs = replace(
            inputs,
            balance=available if available > 0 else fallback_balance,
        )
        max_units = self._max_qty_from_risk(effective_inputs)
        max_lot_qty = lot_size * max(int(inputs.max_lots_per_trade), 0)
        if max_lot_qty > 0:
            max_units = min(max_units, max_lot_qty)

        qty = self._snap_lot(min(requested_qty, max_units), lot_size)
        min_qty = lot_size * max(1, int(inputs.min_lots_per_trade))
        if qty < min_qty:
            sizing = SizingResult(
                qty=0,
                reason="insufficient_risk_capacity",
                available=available,
            )
            return MarginDecision(
                ok=False,
                reason="MARGIN no_qty_after_risk",
                order_type=order_type,
                quantity=0,
                est_required=0.0,
                available=available,
                sizing=sizing,
            )

        buffer = inputs.margin_buffer if inputs.margin_buffer > 0 else 1.0
        try:
            needed = self._estimate_required(
                symbol=inputs.symbol,
                side=inputs.side,
                quantity=qty,
                order_type=order_type,
                inputs=inputs,
            )
            while qty > min_qty and available * buffer < needed:
                qty -= lot_size
                needed = self._estimate_required(
                    symbol=inputs.symbol,
                    side=inputs.side,
                    quantity=qty,
                    order_type=order_type,
                    inputs=inputs,
                )
        except _BrokerMarginUnavailable as exc:
            self._logger.error(
                "margin_plan_required_unavailable",
                extra={"event": "margin_plan_required_unavailable", "error": str(exc)},
            )
            return MarginDecision(
                ok=False,
                reason="broker_margin_unavailable",
                order_type=order_type,
                quantity=0,
                est_required=0.0,
                available=available,
                sizing=SizingResult(
                    qty=0,
                    reason="broker_margin_unavailable",
                    available=available,
                ),
            )

        sizing = SizingResult(qty=qty, needed=needed, available=available)
        if needed <= 0 or available * buffer < needed:
            reason = "margin_no_qty" if needed <= 0 else f"MARGIN needed={needed:.2f}"
            sizing.reason = "margin_no_qty" if needed <= 0 else "insufficient_margin"
            return MarginDecision(
                ok=False,
                reason=reason,
                order_type=order_type,
                quantity=0 if needed <= 0 else qty,
                est_required=max(needed, 0.0),
                available=available,
                sizing=sizing,
            )

        return MarginDecision(
            ok=True,
            reason=None,
            order_type=order_type,
            quantity=qty,
            est_required=needed,
            available=available,
            sizing=sizing,
        )

    def _resolve_available_margin(self, fallback: float, *, force: bool = False) -> float:
        """Prefer a positive DataHub balance, otherwise use the supplied balance."""
        hub = self._data_hub
        if hub is not None and hasattr(hub, "get_available_balance"):
            try:
                raw = hub.get_available_balance(force=force)
                available = float(raw) if raw is not None else 0.0
                if available > 0:
                    return available
            except Exception:  # noqa: BLE001
                pass
            try:
                extracted = self._extract_balance_from_snapshot(
                    hub.get_account_snapshot(force=force)
                )
                if extracted is not None:
                    return extracted
            except Exception:  # noqa: BLE001
                pass
        return max(float(fallback), 0.0)

    def _extract_balance_from_snapshot(self, snapshot: object) -> float | None:
        if not isinstance(snapshot, Mapping):
            return None
        paths = (
            ("equity", "available", "live_balance"),
            ("equity", "available", "available_cash"),
            ("equity", "available", "cash"),
            ("available", "live_balance"),
            ("available", "available_cash"),
            ("available", "cash"),
            ("available", "opening_balance"),
            ("net", "available"),
            ("available_cash",),
            ("available",),
            ("balance",),
        )
        for path in paths:
            value: object = snapshot
            for key in path:
                if not isinstance(value, Mapping) or key not in value:
                    break
                value = value[key]
            else:
                try:
                    parsed = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    return parsed
        return None

    def _resolve_order_type(self, now_ist: datetime, preferred: str | None) -> str:
        del now_ist
        product = (preferred or "NRML").upper()
        return product if product in {"MIS", "NRML"} else "NRML"

    def _max_qty_from_risk(self, inputs: MarginInputs) -> int:
        """Size in complete lots and convert to broker units exactly once."""
        balance = max(float(inputs.balance), 0.0)
        price = max(float(inputs.price), 0.0)
        lot_size = max(1, int(inputs.lot_size))

        risk_per_unit = 0.0
        if inputs.stop_loss and inputs.stop_loss > 0 and price > 0:
            risk_per_unit = abs(price - float(inputs.stop_loss))
        elif inputs.atr and inputs.atr > 0 and inputs.atr_multiple > 0:
            risk_per_unit = float(inputs.atr) * float(inputs.atr_multiple)
        if risk_per_unit <= 0 and price > 0:
            risk_per_unit = price * (
                max(app_settings.RISK_FALLBACK_PRICE_MOVE_PCT, 0.0) / 100.0
            )
        if risk_per_unit <= 0 or price <= 0 or balance <= 0:
            return 0

        one_lot_cost = price * lot_size
        lots_by_risk = int(
            (balance * max(float(inputs.per_trade_risk_pct), 0.0) / 100.0)
            // (risk_per_unit * lot_size)
        )
        lots_by_cap = int(
            (balance * max(float(inputs.per_trade_cap_pct), 0.0) / 100.0)
            // one_lot_cost
        )

        # Do not let a percentage cap alone suppress one indivisible lot when
        # stop-risk and actual cash both permit that requested lot.
        if (
            inputs.requested_qty >= lot_size
            and lots_by_risk >= 1
            and balance >= one_lot_cost
        ):
            lots_by_cap = max(lots_by_cap, 1)

        max_lots = min(lots_by_risk, lots_by_cap)
        configured_max = max(int(inputs.max_lots_per_trade), 0)
        if configured_max:
            max_lots = min(max_lots, configured_max)
        return max(max_lots, 0) * lot_size

    def _snap_lot(self, quantity: int, lot_size: int) -> int:
        lot = max(1, int(lot_size))
        return (max(0, int(quantity)) // lot) * lot

    def _estimate_required(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: str,
        inputs: MarginInputs,
    ) -> float:
        """Use concrete broker margin data when implemented; otherwise premium cost."""
        client = getattr(self._broker, "_client", None) or self._broker
        fetcher = getattr(client, "get_required_margin", None)
        if callable(fetcher):
            try:
                response = fetcher(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    product=order_type,
                )
                if not isinstance(response, Mapping):
                    raise _BrokerMarginUnavailable("invalid_broker_margin_payload")
                required = float(response.get("required", 0.0))
                if required <= 0:
                    raise _BrokerMarginUnavailable("invalid_broker_margin_value")
                return required
            except _BrokerMarginUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001
                raise _BrokerMarginUnavailable(type(exc).__name__) from exc

        return (
            max(float(inputs.price), 0.0)
            * max(int(quantity), 0)
            * max(float(inputs.margin_factor), 1.0)
        )

    def _session_reason(self, now_ist: datetime, order_type: str) -> str | None:
        if order_type == "MIS" and now_ist.timetz() >= MIS_CUTOFF:
            return "MIS_WINDOW_CLOSED"
        return None


__all__ = [
    "MarginEngine",
    "MarginDecision",
    "MarginInputs",
    "MIS_CUTOFF",
    "SizingResult",
]
