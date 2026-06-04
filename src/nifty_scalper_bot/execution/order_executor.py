"""Order execution with risk checks.

NOT THE LIVE ORDER PATH. The single production order path is
StrategyRunner -> OrderManager (wrapped by SafeOrderManager) -> broker,
wired in core/app.py. OrderManager enforces idempotency, the trading-window
gate, the circuit breaker, risk gating, and bracket/OCO coordination.

OrderExecutor is a lighter, self-contained executor retained for unit tests
and simulation only. It deliberately does NOT replicate OrderManager's full
gate stack, so routing live orders through it would bypass those protections.
To prevent that, it refuses to submit to a live broker unless the caller sets
allow_live=True explicitly (used only by tests/sims). This keeps exactly one
live order path in production.
"""

from __future__ import annotations

import os
from itertools import count
import math
import time
from typing import Any, Dict, Iterable, Optional, Sequence, cast

from nifty_scalper_bot.config.base import RiskConfig
from nifty_scalper_bot.execution.entry_price import EntryPriceModel, Side
from nifty_scalper_bot.execution.options_policy import OptionsExecutionPolicy
from nifty_scalper_bot.utils.errors import BrokerError, OrderPlacementError
from nifty_scalper_bot.utils.logging import get_logger


class ExecutionError(OrderPlacementError):
    """Raised when broker execution fails hard."""


def _safe_float(value: object, default: float) -> float:
    if value in (None, ""):
        return float(default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number


def _safe_int(value: object, default: int) -> int:
    if value in (None, ""):
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)

class OrderExecutor:
    """Execute orders while enforcing risk limits."""

    def __init__(
        self,
        broker_client: Any,
        risk_config: RiskConfig,
        mdm: Any,
        *,
        allow_live: bool = False,
    ) -> None:
        self._broker = broker_client
        self._risk = risk_config
        self._mdm = mdm
        # Guard: OrderExecutor is not the live path (see module docstring).
        # It only submits against a live broker when explicitly permitted
        # (tests/sims). In production EXECUTION_MODE=LIVE without allow_live,
        # this raises so the sole live path stays OrderManager.
        self._allow_live = bool(allow_live)
        self._logger = get_logger(__name__)
        self._daily_trade_count = 0
        self._period_start = time.time()
        self._policy = OptionsExecutionPolicy()
        self._entry_model = EntryPriceModel(tick=self._policy.tick_size)
        self._nonce_sequence = count()
        self._nonce_cache: dict[tuple[str, str, int], str] = {}
        self._configure_policy()
        self._open_orders: dict[str, Dict[str, object]] = {}

    def _configure_policy(self) -> None:
        lookup: Any = None
        time_provider: Any = None
        if self._mdm is not None:
            time_provider = getattr(self._mdm, "now_ns", None)
            lookup = getattr(self._mdm, "lot_size_for_symbol", None)
            if not callable(lookup):
                resolver = getattr(self._mdm, "resolver", None) or getattr(
                    self._mdm, "_resolver", None
                )
                lookup = getattr(resolver, "lot_size_for_symbol", None)
        if callable(time_provider):
            self._policy.set_time_provider(time_provider)
        if callable(lookup):
            self._policy.set_lot_size_lookup(lookup)
        self._entry_model.tick = self._policy.tick_size

    def _reset_if_needed(self) -> None:
        if time.time() - self._period_start >= 24 * 60 * 60:
            self._daily_trade_count = 0
            self._period_start = time.time()

    def _check_risk(self, symbol: str, side: str, qty: int, price: float) -> None:
        self._reset_if_needed()
        if self._daily_trade_count >= self._risk.max_daily_trades:
            raise OrderPlacementError("Daily trade limit exceeded")
        notional = qty * price
        if notional > self._risk.max_order_notional:
            raise OrderPlacementError("Order notional exceeds limit")
        if side.upper() == "SELL" and not self._risk.allow_short:
            raise OrderPlacementError("Short selling is disabled")

    def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        *,
        client_order_id: str | None = None,
    ) -> str:
        self._policy.validate_qty(symbol, qty)
        bid, ask, ts_ns = self._quote_context(symbol, price)
        self._policy.price_guard(bid, ask, ts_ns)
        entry_price = price
        side_token = side.upper()
        try:
            entry_price = self._entry_model.compute(
                side=cast(Side, side_token), bid=bid, ask=ask
            )
        except ValueError:
            entry_price = price
        rounded_price = self._policy.round_to_tick(entry_price)
        self._policy.validate_notional(rounded_price, qty)
        self._check_risk(symbol, side, qty, rounded_price)
        key = (symbol.upper(), side.upper(), qty)
        if client_order_id is None:
            nonce = self._nonce_cache.get(key)
            if nonce is None:
                nonce = f"executor:{next(self._nonce_sequence)}"
                self._nonce_cache[key] = nonce
            client_order_id = self._policy.client_order_id(
                symbol, side, qty, nonce=nonce
            )
        else:
            self._nonce_cache.pop(key, None)
        broker_payload: Dict[str, object] = {
            "symbol": symbol,
            "side": side.upper(),
            "quantity": qty,
            "order_type": "LIMIT",
            "price": rounded_price,
            "client_order_id": client_order_id,
        }
        legacy_payload: Dict[str, object] = {
            **broker_payload,
            "qty": qty,
            "type": "LIMIT",
        }
        response: Dict[str, object] | None = None
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = self._submit_broker_order(
                    broker_payload, legacy_payload=legacy_payload
                )
                break
            except BrokerError as exc:
                last_error = exc
                self._logger.warning(
                    '{"event":"ORDER_RETRY","symbol":"%s","attempt":%s}',
                    symbol,
                    attempt,
                )
                response = self._find_open_order(client_order_id)
                if response is not None:
                    break
                if attempt >= 3:
                    raise ExecutionError(
                        "Order placement failed after retries"
                    ) from exc
                time.sleep(0.1 * attempt)
        if response is None:
            raise ExecutionError("Order placement failed") from last_error
        if not isinstance(response, dict):
            raise ExecutionError("Invalid broker response payload")
        if "order_id" not in response:
            existing = self._find_open_order(client_order_id)
            if existing and "order_id" in existing:
                response = existing
            else:
                raise ExecutionError("Broker response missing order_id")
        self._daily_trade_count += 1
        order_id = str(response["order_id"])
        if not order_id:
            raise ExecutionError("Order placement failed")
        self._logger.info(
            '{"event":"ORDER_PLACED","symbol":"%s","order_id":"%s"}', symbol, order_id
        )
        self._open_orders[client_order_id] = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
        }
        self._nonce_cache.pop(key, None)
        return order_id

    def _submit_broker_order(
        self, payload: dict[str, object], *, legacy_payload: dict[str, object] | None = None
    ) -> dict[str, object]:
        # Enforce single live order path: OrderExecutor must not submit live
        # orders in production (that path is OrderManager). Allowed only when
        # explicitly permitted (tests/sims) or outside LIVE mode.
        if not self._allow_live and str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper() == "LIVE":
            raise ExecutionError(
                "OrderExecutor is not the live order path; use OrderManager "
                "(set allow_live=True only in tests/sims)"
            )
        try:
            response = self._broker.place_order(**payload)
        except TypeError as exc:
            if legacy_payload is None:
                raise
            try:
                response = self._broker.place_order(legacy_payload)
            except TypeError:
                raise exc
        if not isinstance(response, dict):
            raise ExecutionError("Invalid broker response payload")
        return response

    def _quote_context(
        self, symbol: str, fallback_price: float
    ) -> tuple[float, float, int]:
        bid = fallback_price
        ask = fallback_price
        time_provider = getattr(self._mdm, "now_ns", None)
        if callable(time_provider):
            ts_ns = int(time_provider())
        else:
            ts_ns = time.time_ns()
        if self._mdm is None:
            return bid, ask, ts_ns
        quote_getter = getattr(self._mdm, "get_last_quote", None)
        if not callable(quote_getter):
            return bid, ask, ts_ns
        quote = quote_getter(symbol)
        if not isinstance(quote, dict):
            return bid, ask, ts_ns
        bid = _safe_float(
            quote.get("bid")
            or quote.get("best_bid")
            or quote.get("best_bid_price")
            or quote.get("buy_price"),
            bid,
        )
        ask = _safe_float(
            quote.get("ask")
            or quote.get("best_ask")
            or quote.get("best_ask_price")
            or quote.get("sell_price"),
            ask,
        )
        ts_val = quote.get("ts_ns") or quote.get("timestamp_ns")
        ts_ns = _safe_int(ts_val, ts_ns)
        return bid, ask, ts_ns



    def is_execution_ready(self, symbol: str | None = None) -> tuple[bool, str]:
        """Validate broker, margin, and instrument readiness. Args: symbol. Returns: (ready, reason). Raises: none."""

        try:
            if self._broker is None:
                return (False, 'broker_unavailable')
            health_fn = getattr(self._broker, 'is_connected', None)
            if callable(health_fn) and not bool(health_fn()):
                return (False, 'broker_disconnected')
            margin_fn = getattr(self._broker, 'available_margin', None)
            if callable(margin_fn):
                margin = float(margin_fn())
                if margin <= 0:
                    return (False, 'insufficient_margin')
            if symbol:
                resolver = getattr(self._mdm, 'resolver', None) if self._mdm is not None else None
                if resolver is not None:
                    lookup = getattr(resolver, 'lookup', None)
                    if callable(lookup) and lookup(symbol) is None:
                        return (False, 'invalid_instrument')
            return (True, 'ready')
        except Exception as e:
            self._logger.exception('Failure in is_execution_ready: %s', e)
            return (False, 'execution_readiness_error')

    def reconcile_open_orders(self) -> dict[str, Dict[str, object]]:
        """Refresh cached open-order metadata using client order identifiers.

        Returns:
            Dictionary keyed by client order identifier containing the latest
            broker snapshots for open orders.
        """

        reconciled: dict[str, Dict[str, object]] = {}
        for client_order_id in list(self._open_orders.keys()):
            snapshot = self._find_open_order(client_order_id)
            if snapshot:
                reconciled[client_order_id] = snapshot
                self._open_orders[client_order_id] = snapshot
            else:
                self._open_orders.pop(client_order_id, None)
        return reconciled

    def place_multi_leg_market_order(
        self,
        legs: Sequence[dict[str, object]],
        *,
        partial_tolerance: float = 1.0,
    ) -> list[str]:
        """Submit multiple legs atomically, cancelling on failure or partial fills.

        Args:
            legs: Sequence of order payload dictionaries describing each leg.
            partial_tolerance: Maximum acceptable shortfall ratio before the
                entire strategy is cancelled.

        Returns:
            Ordered list of broker order identifiers, one per leg.

        Raises:
            OrderPlacementError: If any leg fails policy checks, broker
            submission, or breaches the fill tolerance.
        """

        if not legs:
            raise OrderPlacementError("At least one leg required")
        executed: list[tuple[str, str, int, str]] = []
        order_ids: list[str] = []
        try:
            for index, leg in enumerate(legs):
                symbol = str(leg["symbol"])
                side = str(leg["side"])
                qty_raw = leg.get("qty")
                if isinstance(qty_raw, int):
                    qty = qty_raw
                elif isinstance(qty_raw, str):
                    qty = int(qty_raw)
                else:
                    raise OrderPlacementError("Leg quantity must be numeric")
                price_raw = leg.get("price", 0.0)
                if isinstance(price_raw, (int, float)):
                    price = float(price_raw)
                elif isinstance(price_raw, str):
                    price = float(price_raw)
                else:
                    raise OrderPlacementError("Leg price must be numeric")
                client_order_id_obj = leg.get("client_order_id")
                if client_order_id_obj is None:
                    client_order_id = None
                else:
                    client_order_id = str(client_order_id_obj)
                order_id = self.place_market_order(
                    symbol,
                    side,
                    qty,
                    price,
                    client_order_id=client_order_id,
                )
                order_ids.append(order_id)
                executed.append((order_id, symbol, qty, side))
                if partial_tolerance >= 1.0:
                    continue
                if not self._is_fill_within_tolerance(order_id, qty, partial_tolerance):
                    raise OrderPlacementError(f"Leg {index} fill below tolerance")
        except OrderPlacementError:
            self._cancel_orders(order_id for order_id, *_ in executed)
            self.reconcile_open_orders()
            raise
        return order_ids

    def _is_fill_within_tolerance(
        self, order_id: str, qty: int, partial_tolerance: float
    ) -> bool:
        if partial_tolerance <= 0.0:
            return self._filled_ratio(order_id, qty) >= 1.0
        ratio = self._filled_ratio(order_id, qty)
        return ratio >= 1.0 - partial_tolerance

    def _filled_ratio(self, order_id: str, qty: int) -> float:
        if qty <= 0:
            return 1.0
        getter = getattr(self._broker, "get_order_status", None)
        if not callable(getter):
            return 1.0
        status = getter(order_id)
        if not isinstance(status, dict):
            return 1.0
        filled = float(status.get("filled_qty", qty))
        return max(0.0, min(1.0, filled / qty))

    def _cancel_orders(self, order_ids: Iterable[str]) -> None:
        cancel = getattr(self._broker, "cancel_order", None)
        if not callable(cancel):
            return
        for order_id in order_ids:
            if not order_id:
                continue
            try:
                cancel(order_id)
            except BrokerError:
                self._logger.warning("Failed to cancel order %s", order_id)

    def _find_open_order(self, client_order_id: str) -> Optional[Dict[str, object]]:
        wanted = str(client_order_id or "").strip()
        if not wanted:
            return None
        for attr in ("get_order_by_client_order_id", "find_order_by_client_order_id"):
            finder = getattr(self._broker, attr, None)
            if callable(finder):
                result = finder(wanted)
                if isinstance(result, dict):
                    return result
        get_orders = getattr(self._broker, "get_orders", None)
        if not callable(get_orders):
            get_orders = getattr(self._broker, "orders", None)
        if not callable(get_orders):
            return None
        try:
            orders = get_orders() or []
        except Exception:
            return None
        terminal = {"CANCELLED", "CANCELED", "REJECTED", "COMPLETE", "COMPLETED"}
        wanted_upper = wanted.upper()
        wanted_suffix = wanted_upper[-8:]
        for order in orders:
            if not isinstance(order, dict):
                continue
            status = str(order.get("status") or "").strip().upper()
            if status in terminal:
                continue
            values = [
                order.get("client_order_id"),
                order.get("tag"),
                order.get("order_id"),
                order.get("guid"),
                order.get("exchange_order_id"),
                order.get("parent_order_id"),
            ]
            candidates = {str(v or "").strip().upper() for v in values if v}
            if wanted_upper in candidates:
                return dict(order)
            tag_value = str(order.get("tag") or "").strip().upper()
            if tag_value and wanted_suffix and wanted_suffix in tag_value:
                return dict(order)
        return None


__all__ = ["OrderExecutor"]
