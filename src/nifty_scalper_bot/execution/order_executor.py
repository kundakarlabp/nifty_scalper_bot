"""Order execution with risk checks."""

from __future__ import annotations

from itertools import count
import time
from typing import Any, Dict, Iterable, Optional, Sequence, cast

from nifty_scalper_bot.config.base import RiskConfig
from nifty_scalper_bot.execution.entry_price import EntryPriceModel, Side
from nifty_scalper_bot.execution.options_policy import OptionsExecutionPolicy
from nifty_scalper_bot.utils.errors import BrokerError, OrderPlacementError
from nifty_scalper_bot.utils.logging import get_logger


class OrderExecutor:
    """Execute orders while enforcing risk limits."""

    def __init__(
        self,
        broker_client: Any,
        risk_config: RiskConfig,
        mdm: Any,
    ) -> None:
        self._broker = broker_client
        self._risk = risk_config
        self._mdm = mdm
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
        payload: Dict[str, object] = {
            "symbol": symbol,
            "side": side.upper(),
            "qty": qty,
            "type": "MARKET",
            "price": rounded_price,
            "client_order_id": client_order_id,
        }
        try:
            response = self._broker.place_order(payload)
        except BrokerError as exc:
            response = self._find_open_order(client_order_id)
            if response is None:
                raise OrderPlacementError("Broker rejected order") from exc
        if "order_id" not in response:
            existing = self._find_open_order(client_order_id)
            if existing and "order_id" in existing:
                response = existing
            else:
                raise OrderPlacementError("Broker response missing order_id")
        self._daily_trade_count += 1
        order_id = str(response["order_id"])
        self._logger.info("Placed order %s for %s", order_id, symbol)
        self._open_orders[client_order_id] = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
        }
        self._nonce_cache.pop(key, None)
        return order_id

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
        bid = float(quote.get("bid", bid))
        ask = float(quote.get("ask", ask))
        ts_val = quote.get("ts_ns")
        if ts_val is not None:
            ts_ns = int(ts_val)
        return bid, ask, ts_ns

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
        for attr in ("get_order_by_client_order_id", "find_order_by_client_order_id"):
            finder = getattr(self._broker, attr, None)
            if callable(finder):
                result = finder(client_order_id)
                if result:
                    return result
        return None


__all__ = ["OrderExecutor"]
