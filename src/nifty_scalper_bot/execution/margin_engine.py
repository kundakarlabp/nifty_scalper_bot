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


MIS_CUTOFF = time(15, 25)


class MarginEngine:
    """Evaluate margin availability and sizing prior to broker submission."""

    def __init__(
        self,
        *,
        broker: Any,
        data_hub: Any,
        lot_size_resolver: Any,
        clock: Any,
    ) -> None:
        """Create a margin engine with the dependencies required for planning.

        Args:
            broker: Broker client used for required margin lookups.
            data_hub: Shared data hub instance for market data access.
            lot_size_resolver: Resolver able to provide lot-size context.
            clock: Callable returning monotonic timestamps (unused hook).
        """

        self._broker = broker
        self._lots = lot_size_resolver
        self._clock = clock
        self._logger = get_logger(__name__)
        self._data_hub: Any | None = None
        self.set_data_hub(data_hub)

    def set_data_hub(self, hub: Any | None) -> None:
        """Attach the shared Data Hub for balance hydration.

        Args:
            hub: Data hub instance exposing ``get_available_balance``.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered MarginEngine.set_data_hub",
            extra={"event": "margin_engine_set_data_hub_start"},
        )
        try:
            self._data_hub = hub
            self._logger.info(
                "Condition met: margin_engine_data_hub_attached",
                extra={
                    "event": "margin_engine_data_hub_attached",
                    "attached": hub is not None,
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in MarginEngine.set_data_hub: %s",
                exc,
                extra={"event": "margin_engine_set_data_hub_failed"},
                exc_info=exc,
            )
            raise

    def plan(self, inputs: MarginInputs) -> MarginDecision:
        """Compute whether an order should proceed given risk and margin.

        Args:
            inputs: Structured information describing the intended trade.

        Returns:
            MarginDecision describing feasibility, size, and order type.

        Raises:
            None.
        """

        self._logger.debug(
            "margin_plan_start",
            extra={
                "event": "margin_plan_start",
                "symbol": inputs.symbol,
                "side": inputs.side,
                "requested_qty": inputs.requested_qty,
                "product": inputs.product,
            },
        )
        fallback_balance = max(0.0, float(inputs.balance))
        available_margin = self._resolve_available_margin(fallback_balance)
        effective_balance = (
            available_margin if available_margin > 0 else fallback_balance
        )
        effective_inputs = replace(inputs, balance=effective_balance)
        order_type = self._resolve_order_type(
            effective_inputs.ist_now, effective_inputs.product
        )
        max_units = self._max_qty_from_risk(effective_inputs)
        raw_lot = max(1, int(inputs.lot_size))
        min_lot_qty = raw_lot * max(1, int(inputs.min_lots_per_trade))
        max_lot_qty = raw_lot * max(int(inputs.max_lots_per_trade), 0)
        if 0 < max_lot_qty < min_lot_qty:
            min_lot_qty = max_lot_qty
        if max_lot_qty > 0:
            max_units = min(max_units, max_lot_qty)
        qty = max(0, min(inputs.requested_qty, max_units))
        qty = self._snap_lot(qty, inputs.lot_size)
        if max_lot_qty > 0:
            qty = min(qty, max_lot_qty)
        if qty <= 0:
            self._logger.warning(
                "Risk/margin block: order sizing reduced to zero",
                extra={
                    "event": "margin_sizing_zero_qty",
                    "symbol": inputs.symbol,
                    "side": inputs.side,
                    "requested_qty": inputs.requested_qty,
                    "max_units": max_units,
                    "available_margin": round(float(available_margin), 2),
                    "effective_balance": round(float(effective_balance), 2),
                    "per_trade_risk_pct": inputs.per_trade_risk_pct,
                    "per_trade_cap_pct": inputs.per_trade_cap_pct,
                    "max_lots_per_trade": inputs.max_lots_per_trade,
                },
            )
        sizing = SizingResult(qty=qty)
        if qty <= 0:
            min_lots = max(1, int(app_settings.MIN_LOTS_PER_TRADE))
            min_lots = max(min_lots, int(inputs.min_lots_per_trade))
            lot_units = max(1, int(inputs.lot_size))
            min_qty = min_lots * lot_units
            if max_units < min_qty:
                sizing = SizingResult(
                    qty=0,
                    reason="insufficient_risk_capacity",
                    needed=None,
                    available=available_margin,
                )
                self._logger.debug(
                    "margin_plan_skip insufficient_risk_capacity",
                    extra={
                        "event": "margin_plan_skip",
                        "reason": "insufficient_risk_capacity",
                        "symbol": inputs.symbol,
                        "side": inputs.side,
                        "max_units": max_units,
                        "min_qty": min_qty,
                    },
                )
                return MarginDecision(
                    ok=False,
                    reason="MARGIN no_qty_after_risk",
                    order_type=order_type,
                    quantity=0,
                    est_required=0.0,
                    available=available_margin,
                    sizing=sizing,
                )
            needed_min_qty = self._estimate_required(
                symbol=inputs.symbol,
                side=inputs.side,
                quantity=min_qty,
                order_type=order_type,
                inputs=inputs,
            )
            sizing = SizingResult(
                qty=min_qty,
                reason="clamped_min_lot",
                needed=needed_min_qty,
                available=available_margin,
            )
            buffer = (
                inputs.margin_buffer
                if inputs.margin_buffer and inputs.margin_buffer > 0
                else 1.0
            )
            if available_margin * buffer >= needed_min_qty > 0:
                qty = min_qty
                self._logger.info(
                    "margin_plan_clamp_min_lot",
                    extra={
                        "event": "margin_plan_clamp_min_lot",
                        "symbol": inputs.symbol,
                        "side": inputs.side,
                        "quantity": min_qty,
                        "needed": needed_min_qty,
                        "available": available_margin,
                    },
                )
            else:
                self._logger.debug(
                    "margin_plan_skip margin_no_qty",
                    extra={
                        "event": "margin_plan_skip",
                        "reason": "margin_no_qty",
                        "symbol": inputs.symbol,
                        "side": inputs.side,
                        "quantity": min_qty,
                        "needed": needed_min_qty,
                        "available": available_margin,
                    },
                )
                return MarginDecision(
                    ok=False,
                    reason="margin_no_qty",
                    order_type=order_type,
                    quantity=0,
                    est_required=needed_min_qty,
                    available=available_margin,
                    sizing=SizingResult(
                        qty=0,
                        reason="margin_no_qty",
                        needed=needed_min_qty,
                        available=available_margin,
                    ),
                )

        needed = self._estimate_required(
            symbol=inputs.symbol,
            side=inputs.side,
            quantity=qty,
            order_type=order_type,
            inputs=inputs,
        )
        sizing.needed = needed
        sizing.available = available_margin
        if inputs.product and inputs.product.upper() == "MIS" and order_type != "MIS":
            self._logger.info(
                "margin_plan_block_session",
                extra={
                    "event": "margin_plan_block",
                    "reason": "MIS_WINDOW_CLOSED",
                    "symbol": inputs.symbol,
                    "side": inputs.side,
                    "quantity": qty,
                    "needed": needed,
                    "available": available_margin,
                },
            )
            return MarginDecision(
                ok=False,
                reason="MIS_WINDOW_CLOSED",
                order_type=order_type,
                quantity=qty,
                est_required=needed,
                available=available_margin,
                sizing=sizing,
            )
        reason = self._session_reason(inputs.ist_now, order_type)
        if reason:
            self._logger.info(
                "margin_plan_block_session",
                extra={
                    "event": "margin_plan_block",
                    "reason": reason,
                    "symbol": inputs.symbol,
                    "side": inputs.side,
                    "quantity": qty,
                    "needed": needed,
                    "available": available_margin,
                },
            )
            return MarginDecision(
                ok=False,
                reason=reason,
                order_type=order_type,
                quantity=qty,
                est_required=needed,
                available=available_margin,
                sizing=sizing,
            )

        if available_margin * inputs.margin_buffer < needed:
            reason_text = f"MARGIN needed={needed:.2f}"
            self._logger.info(
                "margin_plan_block_balance",
                extra={
                    "event": "margin_plan_block",
                    "reason": reason_text,
                    "symbol": inputs.symbol,
                    "side": inputs.side,
                    "quantity": qty,
                    "needed": needed,
                    "available": available_margin,
                    "buffer": inputs.margin_buffer,
                },
            )
            return MarginDecision(
                ok=False,
                reason=reason_text,
                order_type=order_type,
                quantity=qty,
                est_required=needed,
                available=available_margin,
                sizing=SizingResult(
                    qty=qty,
                    reason="insufficient_margin",
                    needed=needed,
                    available=available_margin,
                ),
            )

        decision = MarginDecision(
            ok=True,
            reason=None,
            order_type=order_type,
            quantity=qty,
            est_required=needed,
            available=available_margin,
            sizing=sizing,
        )
        self._logger.debug(
            "margin_plan_ok",
            extra={
                "event": "margin_plan_ok",
                "symbol": inputs.symbol,
                "side": inputs.side,
                "quantity": qty,
                "needed": needed,
                "available": available_margin,
                "order_type": order_type,
            },
        )
        return decision

    def _resolve_available_margin(
        self, fallback: float, *, force: bool = False
    ) -> float:
        """Resolve available margin using the data hub when attached.

        Args:
            fallback: Local fallback balance supplied by the caller.
            force: Force data hub refresh when ``True``.

        Returns:
            float: Non-negative available margin for sizing decisions.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered MarginEngine._resolve_available_margin",
            extra={
                "event": "margin_resolve_available_start",
                "force": force,
                "fallback": round(max(fallback, 0.0), 2),
            },
        )
        available: float | None = None
        hub = getattr(self, "_data_hub", None)
        if hub is None or not hasattr(hub, "get_available_balance"):
            self._logger.debug(
                "Condition met: margin_resolve_available_no_hub",
                extra={"event": "margin_resolve_available_no_hub"},
            )
        else:
            raw_available: float | None = None
            try:
                raw_available = hub.get_available_balance(force=force)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in MarginEngine._resolve_available_margin: %s",
                    exc,
                    extra={"event": "margin_resolve_available_error"},
                    exc_info=exc,
                )
            if raw_available is not None:
                try:
                    available = float(raw_available)
                except (TypeError, ValueError) as exc:  # noqa: BLE001
                    self._logger.error(
                        "Failure in MarginEngine._resolve_available_margin: %s",
                        exc,
                        extra={
                            "event": "margin_resolve_available_invalid",
                            "raw_value": raw_available,
                        },
                        exc_info=exc,
                    )
                    available = None
                else:
                    self._logger.info(
                        "Condition met: margin_resolve_available_broker_payload",
                        extra={
                            "event": "margin_resolve_available_broker_payload",
                            "balance": round(available, 2),
                        },
                    )
            if available is None:
                try:
                    snapshot = hub.get_account_snapshot(force=force)
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Failure in MarginEngine._resolve_available_margin: %s",
                        exc,
                        extra={"event": "margin_resolve_available_snapshot_error"},
                        exc_info=exc,
                    )
                else:
                    extracted = self._extract_balance_from_snapshot(snapshot)
                    if extracted is not None:
                        available = extracted
                        self._logger.info(
                            "Condition met: margin_resolve_available_snapshot_payload",
                            extra={
                                "event": "margin_resolve_available_snapshot_payload",
                                "balance": round(extracted, 2),
                            },
                        )

        if available is not None and available > 0:
            resolved = float(available)
            self._logger.info(
                "Condition met: margin_resolve_available_data_hub",
                extra={
                    "event": "margin_resolve_available_data_hub",
                    "balance": round(resolved, 2),
                },
            )
            return resolved

        if available is not None:
            self._logger.warning(
                "margin_resolve_available_non_positive",
                extra={
                    "event": "margin_resolve_available_non_positive",
                    "balance": round(float(available), 2),
                },
            )

        if fallback > 0:
            self._logger.info(
                "Condition met: margin_resolve_available_fallback",
                extra={
                    "event": "margin_resolve_available_fallback",
                    "balance": round(float(fallback), 2),
                },
            )
            return float(fallback)

        self._logger.warning(
            "margin_resolve_available_unavailable",
            extra={"event": "margin_resolve_available_unavailable"},
        )
        return 0.0

    def _extract_balance_from_snapshot(self, snapshot: object) -> float | None:
        """Extract a usable balance value from a margin snapshot payload.

        Args:
            snapshot: Raw snapshot payload returned by the data hub.

        Returns:
            float | None: Positive balance when discovered, otherwise ``None``.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered MarginEngine._extract_balance_from_snapshot",
            extra={"event": "margin_resolve_extract_start"},
        )
        try:
            if not isinstance(snapshot, Mapping):
                return None
            candidates: tuple[tuple[str, ...], ...] = (
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
            for path in candidates:
                cursor: object = snapshot
                for key in path:
                    if isinstance(cursor, Mapping) and key in cursor:
                        cursor = cursor[key]
                    else:
                        break
                else:
                    try:
                        value = float(cursor)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        return value
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in MarginEngine._extract_balance_from_snapshot: %s",
                exc,
                extra={"event": "margin_resolve_extract_error"},
                exc_info=exc,
            )
        return None

    def _resolve_order_type(self, now_ist: datetime, preferred: str | None) -> str:
        """Determine the effective order product based on time and preference.

        Args:
            now_ist: Current timestamp in IST.
            preferred: Requested product, typically 'MIS' or 'NRML'.

        Returns:
            Canonical product string acceptable to the broker.

        Raises:
            None.
        """

        product = (preferred or "").upper()
        if product == "MIS" and now_ist.timetz() >= MIS_CUTOFF:
            self._logger.info(
                "margin_plan_downgrade_mis",
                extra={"event": "margin_plan_downgrade", "target": "NRML"},
            )
            return "NRML"
        if product in {"MIS", "NRML"}:
            return product
        return "NRML"

    def _max_qty_from_risk(self, inputs: MarginInputs) -> int:
        """Derive the maximum quantity allowed by risk and capital caps.

        Args:
            inputs: Structured trade description including risk context.

        Returns:
            Maximum whole-unit quantity before lot snapping.

        Raises:
            None.
        """

        balance = max(0.0, float(inputs.balance))
        price = max(inputs.price, 0.0)
        cap_pct = max(inputs.per_trade_cap_pct, inputs.per_trade_risk_pct)
        cap_from_pct = balance * (cap_pct / 100.0)
        risk_pct = max(inputs.per_trade_risk_pct, 0.0) / 100.0

        risk_per_unit = 0.0
        if inputs.stop_loss and inputs.stop_loss > 0 and price > 0:
            risk_per_unit = abs(price - inputs.stop_loss)
        elif inputs.atr and inputs.atr > 0 and inputs.atr_multiple > 0:
            risk_per_unit = inputs.atr * inputs.atr_multiple
        if risk_per_unit <= 0 and price > 0:
            fallback_move_pct = (
                max(app_settings.RISK_FALLBACK_PRICE_MOVE_PCT, 0.0) / 100.0
            )
            risk_per_unit = price * fallback_move_pct

        lot_size = max(1, int(inputs.lot_size))
        # Per-lot economics derived from lot_size ALONE. contract_multiplier is
        # deliberately not applied here: production sets it to the lot size, so
        # using both would count contract size twice. Contract size is counted
        # exactly once, via lot_size, and converted to broker units once at the
        # end of this method.
        one_lot_risk = risk_per_unit * lot_size
        if one_lot_risk <= 0:
            return 0
        lots_by_risk = int((balance * risk_pct) // one_lot_risk)
        one_lot_cost = price * lot_size
        lots_by_cap = int(cap_from_pct // one_lot_cost) if one_lot_cost > 0 else 0
        lots_by_margin = 0
        margin_estimator = getattr(self._broker, "estimate_margin", None)
        if callable(margin_estimator) and price > 0:
            try:
                # QUANTITY CONTRACT: `estimate_margin(symbol, quantity, price)`
                # is an OPTIONAL broker capability -- no adapter in this repo
                # implements it today (it is probed via getattr), so its
                # quantity convention is not established by any concrete
                # implementation. We therefore ask for the margin of ONE WHOLE
                # LOT directly by passing `lot_size` as the quantity, rather
                # than estimating one unit and scaling by lot_size. If the
                # broker interprets quantity as broker units this is exactly
                # one lot; the result is used as-is and never multiplied by
                # lot_size again, so no convention mismatch can inflate it.
                margin_for_one_lot = float(
                    margin_estimator(inputs.symbol, lot_size, price)
                )
                if margin_for_one_lot > 0:
                    lots_by_margin = int(balance // margin_for_one_lot)
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "margin_plan_estimate_margin_error",
                    extra={"event": "margin_plan_estimate_margin_error", "error": str(exc)},
                )
        max_lots = max(
            0, min(lots_by_risk, lots_by_cap, lots_by_margin or lots_by_cap)
        )
        configured_max_lots = max(int(inputs.max_lots_per_trade), 0)
        if configured_max_lots > 0:
            max_lots = min(max_lots, configured_max_lots)
        # Single conversion point: complete lots -> broker units.
        max_units = max_lots * lot_size
        self._logger.debug(
            "margin_plan_qty",
            extra={
                "event": "margin_plan_qty",
                "symbol": inputs.symbol,
                "lots_by_risk": lots_by_risk,
                "lots_by_cap": lots_by_cap,
                "lots_by_margin": lots_by_margin,
                "max_lots": max_lots,
                "lot_size": lot_size,
                "max_units": max_units,
            },
        )
        return max_units

    def _snap_lot(self, quantity: int, lot_size: int) -> int:
        """Snap a quantity down to the nearest valid lot multiple.

        Args:
            quantity: Raw unit quantity derived from sizing.
            lot_size: Instrument lot size (minimum tradable chunk).

        Returns:
            Quantity respecting lot-size constraints.

        Raises:
            None.
        """

        lot = max(1, int(lot_size))
        snapped = (max(0, int(quantity)) // lot) * lot
        self._logger.debug(
            "margin_plan_snap",
            extra={
                "event": "margin_plan_snap",
                "quantity": quantity,
                "lot": lot,
                "snapped": snapped,
            },
        )
        return snapped

    def _estimate_required(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: str,
        inputs: MarginInputs,
    ) -> float:
        """Estimate margin required using broker API or fall back model.

        Args:
            symbol: Tradable instrument identifier.
            side: Trade direction.
            quantity: Planned lot-adjusted quantity.
            order_type: Selected product for the order.
            inputs: Complete margin context for fallback calculations.

        Returns:
            Estimated required capital for broker submission.

        Raises:
            None.
        """

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
                if isinstance(response, Mapping):
                    required_raw = response.get("required")
                    required = float(required_raw) if required_raw is not None else 0.0
                    if required > 0:
                        return required
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "margin_plan_required_fetch_failed",
                    extra={"event": "margin_plan_required_fallback", "error": str(exc)},
                )
        premium = max(inputs.price, 0.0)
        # `quantity` is already broker units (lots x lot_size), and
        # `contract_multiplier` IS the lot size, so applying it here would
        # square the contract size (e.g. 100 x 65 x 65 = 422500 instead of
        # 6500). Contract size is counted exactly once, inside `quantity`.
        estimate = (
            premium * max(quantity, 0) * max(inputs.margin_factor, 1.0)
        )
        return estimate

    def _session_reason(self, now_ist: datetime, order_type: str) -> str | None:
        """Resolve session-derived block reasons for the chosen order type.

        Args:
            now_ist: Current IST timestamp.
            order_type: Product selected for the order.

        Returns:
            Canonical soft-block reason if session disallows the order, else None.

        Raises:
            None.
        """

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
