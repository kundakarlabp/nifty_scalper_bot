"""Safety wrapper for :class:`~nifty_scalper_bot.execution.order_manager.OrderManager`.

This module provides guardrails around the underlying order manager.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Literal, Mapping

from nifty_scalper_bot.config.settings import OrderSettings
from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager
from nifty_scalper_bot.execution.order_manager import (
    ExitIntent,
    OrderManager,
    OrderType,
)
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.infra.structured_logger import emit_diag
from nifty_scalper_bot.utils.errors import OrderPlacementError
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.metrics import Counter, Gauge
from nifty_scalper_bot.utils.reasons import canonical

OrderSide = Literal["BUY", "SELL"]


@dataclass(slots=True)
class SafeOrderManager:
    """Enforce rate limits, sane defaults and metrics around the base order manager."""

    order_manager: OrderManager
    settings: OrderSettings
    on_order_rejected: Callable[[str, str], None] | None = None
    post_order_hook: Callable[[str, OrderSide, int, float | None], None] | None = None
    regime_manager: MarketRegimeManager | None = None
    _logger: Any = field(init=False, repr=False)
    _lock: threading.RLock = field(init=False, repr=False)
    _per_second: Deque[float] = field(init=False, repr=False)
    _per_minute: Deque[float] = field(init=False, repr=False)
    _per_day: Deque[float] = field(init=False, repr=False)
    _throttled: int = field(init=False, repr=False, default=0)
    _rejections: int = field(init=False, repr=False, default=0)
    _m_throttled: Any = field(init=False, repr=False)
    _m_rejections: Any = field(init=False, repr=False)
    _m_active: Any = field(init=False, repr=False)
    _last_order_at: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        self._logger = get_logger(__name__)
        self._lock = threading.RLock()
        self._per_second: Deque[float] = deque()
        self._per_minute: Deque[float] = deque()
        self._per_day: Deque[float] = deque()
        self._throttled = 0
        self._rejections = 0
        self._m_throttled = Counter(
            "orders_throttled_total",
            "Orders blocked by SafeOrderManager throttling",
        )
        self._m_rejections = Counter(
            "orders_rejected_total",
            "Orders rejected downstream of SafeOrderManager",
        )
        self._m_active = Gauge(
            "orders_live_enabled",
            "1 when live routing is allowed, 0 otherwise",
        )
        self._update_live_gauge()
        self._last_order_at = 0.0

    # ------------------------------------------------------------------
    def _propagate_skip_reason(self, reason: str) -> None:
        """Forward recognised skip reasons to the wrapped order manager.

        Args:
            reason: Candidate reason emitted by downstream layers.

        Returns:
            None.

        Raises:
            None.
        """

        token = canonical(reason)
        if not token or token == "OK":
            return
        setter = getattr(self.order_manager, "set_last_skip_reason", None)
        if not callable(setter):
            return
        try:
            setter(token)
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "propagate_skip_reason_failed",
                extra={
                    "event": "propagate_skip_reason_failed",
                    "reason": token,
                    "error": str(exc),
                },
            )

    # ------------------------------------------------------------------
    def _regime_gate_decision(
        self,
        *,
        symbol: str,
        side: OrderSide,
        context: Mapping[str, object] | None = None,
    ) -> tuple[bool, tuple[str, ...]]:
        """Evaluate regime gating and return allowance with reasons.

        Args:
            symbol: Trading symbol being evaluated.
            side: Order side associated with the request.
            context: Additional metadata for diagnostics.

        Returns:
            tuple[bool, tuple[str, ...]]: Decision flag and associated reasons.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered SafeOrderManager._regime_gate_decision",
            extra={
                "event": "safe_order_regime_gate_decision",
                "symbol": symbol,
                "side": side,
            },
        )
        manager = self.regime_manager
        if manager is None:
            return True, tuple()
        payload: dict[str, object] = {
            "component": "safe_order_manager",
            "symbol": symbol,
            "side": side,
        }
        if context is not None:
            for key, value in context.items():
                if isinstance(key, str):
                    payload[key] = value
        try:
            allowed = manager.can_trade(context=payload)
            reasons = manager.get_filter_reasons()
            return allowed, reasons
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _regime_gate_decision: %s",
                exc,
                extra={"event": "safe_order_regime_gate_error", "symbol": symbol},
                exc_info=exc,
            )
            return True, tuple()

    # ------------------------------------------------------------------
    def _update_live_gauge(self) -> None:
        try:
            self._m_active.set(1 if self.settings.enable_live else 0)
        except Exception:  # pragma: no cover - optional metrics
            pass

    def _resolve_reduce_only_side(self, intent: ExitIntent) -> OrderSide:
        """Infer reduce-only exit side for *intent* quantity semantics.

        Args:
            intent: Exit intent describing the quantity sign semantics.

        Returns:
            OrderSide: ``"BUY"`` when closing shorts else ``"SELL"``.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered SafeOrderManager._resolve_reduce_only_side",
            extra={
                "event": "safe_reduce_only_side_infer",
                "symbol": intent.symbol,
            },
        )
        try:
            qty = int(intent.qty)
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in _resolve_reduce_only_side: %s",
                exc,
                extra={"event": "safe_reduce_only_side_error", "symbol": intent.symbol},
                exc_info=exc,
            )
            return "SELL"
        side: OrderSide = "BUY" if qty < 0 else "SELL"
        self._logger.info(
            "Condition met: safe_reduce_only_side_resolved",
            extra={
                "event": "safe_reduce_only_side_resolved",
                "symbol": intent.symbol,
                "qty": qty,
                "side": side,
            },
        )
        return side

    def set_live_enabled(self, enabled: bool) -> None:
        """Update live trading toggle and metric."""

        with self._lock:
            self.settings.enable_live = bool(enabled)
            self._update_live_gauge()

    def throttled_count(self) -> int:
        return self._throttled

    def rejection_count(self) -> int:
        return self._rejections

    def consume_skip_reason(self) -> str | None:
        """Return the latest skip reason from the wrapped order manager.

        Args:
            None.

        Returns:
            str | None: Most recent skip reason or ``None`` when unavailable.

        Raises:
            None.
        """

        self._logger.debug("Entered consume_skip_reason")
        try:
            handler = getattr(self.order_manager, "consume_skip_reason", None)
            if handler is None:
                self._logger.info(
                    "Condition met: no skip reason handler available",
                    extra={"event": "consume_skip_reason_missing"},
                )
                return None
            reason = handler()
            self._logger.info(
                "Condition met: retrieved skip reason",
                extra={"event": "consume_skip_reason", "reason": reason},
            )
            return reason
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in consume_skip_reason: %s",
                exc,
                extra={"event": "consume_skip_reason_error"},
            )
            return None

    # ------------------------------------------------------------------
    def place_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        product: str | None = None,
        tag: str | None = None,
    ) -> str:
        """Submit an order with throttling, retries and sane defaults."""

        self._logger.debug(
            "Entered SafeOrderManager.place_order",
            extra={
                "event": "safe_order_place_enter",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "order_type": order_type.name if order_type else None,
                "price": price,
            },
        )
        allowed, reasons = self._regime_gate_decision(
            symbol=symbol,
            side=side,
            context={"quantity": quantity, "order_type": order_type.name},
        )
        if not allowed:
            reason_text = ", ".join(reasons) if reasons else "regime_gate_block"
            primary_reason = reasons[0] if reasons else "regime_gate"
            self._logger.warning(
                "Condition met: safe_order_regime_block",
                extra={
                    "event": "safe_order_regime_block",
                    "symbol": symbol,
                    "side": side,
                    "reasons": list(reasons),
                },
            )
            self._propagate_skip_reason(primary_reason)
            self._record_rejection(symbol, reason_text)
            raise OrderPlacementError(f"Regime gate blocked order: {reason_text}")
        try:
            METRICS.increment_retry_event(
                label="order.place", stage="enter", outcome="start"
            )
        except Exception:  # pragma: no cover - optional metrics
            self._logger.debug("order_place_enter_metric_failed", exc_info=True)

        now = time.time()
        if not self.settings.enable_live:
            self._logger.info(
                "Condition met: order_skip_live_disabled",
                extra={
                    "event": "order_skip_live_disabled",
                    "symbol": symbol,
                    "side": side,
                },
            )
            self._throttled += 1
            try:
                self._m_throttled.inc()
            except Exception:  # pragma: no cover - optional metrics
                pass
            self._record_rejection(symbol, "live trading disabled")
            try:
                METRICS.increment_retry_event(
                    label="order.place", stage="skip", outcome="live_disabled"
                )
            except Exception:  # pragma: no cover - optional metrics
                self._logger.debug("order_skip_metric_failed", exc_info=True)
            raise OrderPlacementError("Live trading disabled")

        spacing = max(self.settings.min_spacing_seconds, 0.0)

        with self._lock:
            if spacing > 0:
                wait = spacing - (time.time() - self._last_order_at)
                if wait > 0:
                    time.sleep(wait)
                self._last_order_at = time.time()
            now = time.time()
            if not self._check_throttle(now):
                self._throttled += 1
                try:
                    self._m_throttled.inc()
                except Exception:  # pragma: no cover - optional metrics
                    pass
                self._logger.info(
                    "Condition met: order_skip_throttle",
                    extra={
                        "event": "order_skip_throttle",
                        "symbol": symbol,
                        "side": side,
                    },
                )
                self._record_rejection(symbol, "order rate limited")
                try:
                    METRICS.increment_retry_event(
                        label="order.place", stage="skip", outcome="throttle"
                    )
                except Exception:  # pragma: no cover - optional metrics
                    self._logger.debug("order_throttle_metric_failed", exc_info=True)
                raise OrderPlacementError("Order rate limited")

        chosen_order_type = order_type
        if chosen_order_type is None:
            try:
                chosen_order_type = OrderType[self.settings.default_order_type.upper()]
            except KeyError:
                self._logger.debug(
                    "Unknown default order type %s; using LIMIT",
                    self.settings.default_order_type,
                )
                chosen_order_type = OrderType.LIMIT
        adj_price = self._apply_price_offset(side, price)

        try:
            chosen_order_type, adj_price = self._apply_queue_policy(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=chosen_order_type,
                price=adj_price,
            )
        except OrderPlacementError as exc:
            reason = canonical(str(exc) or "queue_policy")
            self._propagate_skip_reason(reason)
            self._record_rejection(symbol, reason)
            try:
                METRICS.increment_retry_event(
                    label="order.place", stage="skip", outcome=reason
                )
            except Exception:  # pragma: no cover - optional metrics
                self._logger.debug("order_queue_metric_failed", exc_info=True)
            raise

        attempt = 0
        last_error: Exception | None = None
        max_attempts = max(1, self.settings.retry_attempts)
        while attempt < max_attempts:
            attempt += 1
            self._logger.debug(
                "order_attempt_start",
                extra={
                    "event": "order_attempt_start",
                    "symbol": symbol,
                    "side": side,
                    "attempt": attempt,
                    "order_type": chosen_order_type.name,
                    "price": adj_price,
                },
            )
            try:
                METRICS.increment_retry_event(
                    label="order.place", stage="attempt", outcome="start"
                )
            except Exception:  # pragma: no cover - optional metrics
                self._logger.debug("order_attempt_metric_failed", exc_info=True)
            try:
                order_id = self.order_manager.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=chosen_order_type,
                    price=adj_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    product=product or self.settings.default_product,
                    tag=tag,
                )
                if not order_id:
                    self._logger.info(
                        "Condition met: order_ack_empty",
                        extra={
                            "event": "order_ack_empty",
                            "symbol": symbol,
                            "side": side,
                            "attempt": attempt,
                        },
                    )
                    return ""
                if chosen_order_type == OrderType.LIMIT and adj_price is not None:
                    self._chase_fill(order_id, side, float(adj_price))
                if self.post_order_hook is not None:
                    try:
                        self.post_order_hook(symbol, side, quantity, adj_price)
                    except Exception:  # pragma: no cover - defensive
                        self._logger.debug("post_order_hook failed", exc_info=True)
                self._logger.info(
                    "Condition met: order_submit_success",
                    extra={
                        "event": "order_submit_success",
                        "symbol": symbol,
                        "side": side,
                        "order_id": order_id,
                        "attempt": attempt,
                    },
                )
                try:
                    METRICS.increment_retry_event(
                        label="order.place", stage="attempt", outcome="success"
                    )
                except Exception:  # pragma: no cover - optional metrics
                    self._logger.debug("order_success_metric_failed", exc_info=True)
                return order_id
            except OrderPlacementError as exc:
                last_error = exc
                raw_reason = canonical(str(exc) or "order_error")
                self._propagate_skip_reason(raw_reason)
                if attempt >= max_attempts:
                    self._logger.info(
                        "Condition met: order_submit_failure",
                        extra={
                            "event": "order_submit_failure",
                            "symbol": symbol,
                            "side": side,
                            "attempt": attempt,
                            "reason": raw_reason,
                        },
                    )
                    self._record_rejection(symbol, raw_reason)
                    try:
                        METRICS.increment_error(error_type="order_placement_failure")
                    except Exception:  # pragma: no cover - optional metrics
                        self._logger.debug(
                            "Failed to increment error metric", exc_info=True
                        )
                    try:
                        METRICS.increment_retry_event(
                            label="order.place", stage="attempt", outcome="failure"
                        )
                    except Exception:  # pragma: no cover - optional metrics
                        self._logger.debug("order_failure_metric_failed", exc_info=True)
                    raise
                try:
                    METRICS.increment_retry_event(
                        label="order.place", stage="attempt", outcome="retry"
                    )
                except Exception:  # pragma: no cover - optional metrics
                    self._logger.debug("order_retry_metric_failed", exc_info=True)
                time.sleep(self.settings.retry_delay_seconds)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                reason_text = canonical(str(exc) or "order_exception")
                self._record_rejection(symbol, reason_text)
                try:
                    METRICS.increment_error(error_type="order_unexpected_error")
                except Exception:  # pragma: no cover - optional metrics
                    self._logger.debug(
                        "Failed to increment error metric", exc_info=True
                    )
                self._logger.error(
                    "Failure in order submission: %s",
                    exc,
                    extra={
                        "event": "order_submit_exception",
                        "symbol": symbol,
                        "side": side,
                        "attempt": attempt,
                    },
                )
                if attempt >= max_attempts:
                    try:
                        METRICS.increment_retry_event(
                            label="order.place", stage="attempt", outcome="failure"
                        )
                    except Exception:  # pragma: no cover - optional metrics
                        self._logger.debug(
                            "order_exception_metric_failed", exc_info=True
                        )
                    raise OrderPlacementError(str(exc)) from exc
                try:
                    METRICS.increment_retry_event(
                        label="order.place", stage="attempt", outcome="retry"
                    )
                except Exception:  # pragma: no cover - optional metrics
                    self._logger.debug(
                        "order_exception_retry_metric_failed", exc_info=True
                    )
                time.sleep(self.settings.retry_delay_seconds)

        if last_error is not None:
            reason_text = canonical(str(last_error) or "order_failure")
            self._propagate_skip_reason(reason_text)
            try:
                METRICS.increment_error(error_type="order_last_error")
            except Exception:  # pragma: no cover - optional metrics
                self._logger.debug("Failed to increment error metric", exc_info=True)
            raise OrderPlacementError(str(last_error)) from last_error

        raise OrderPlacementError("Order placement failed without error")

    def place_reduce_only_exit(self, intent: ExitIntent) -> str | None:
        """Submit a reduce-only exit through the wrapped order manager.

        Args:
            intent: Exit intent containing instrument details and size.

        Returns:
            str | None: Broker order identifier when routed else ``None``.

        Raises:
            OrderPlacementError: If reduce-only exits are not supported or fail.
        """

        self._logger.debug(
            "Entered place_reduce_only_exit",
            extra={
                "event": "safe_reduce_only_exit_enter",
                "symbol": intent.symbol,
                "qty": intent.qty,
            },
        )
        side = self._resolve_reduce_only_side(intent)
        allowed, reasons = self._regime_gate_decision(
            symbol=intent.symbol,
            side=side,
            context={
                "intent_qty": intent.qty,
                "action": "reduce_only",
                "intent_side": side,
            },
        )
        if not allowed:
            reason_text = ", ".join(reasons) if reasons else "regime_gate_block"
            primary_reason = reasons[0] if reasons else "regime_gate"
            self._logger.warning(
                "Condition met: safe_reduce_only_regime_block",
                extra={
                    "event": "safe_reduce_only_regime_block",
                    "symbol": intent.symbol,
                    "reasons": list(reasons),
                },
            )
            self._propagate_skip_reason(primary_reason)
            self._record_rejection(intent.symbol, reason_text)
            raise OrderPlacementError(
                f"Regime gate blocked reduce-only exit: {reason_text}"
            )
        handler = getattr(self.order_manager, "place_reduce_only_exit", None)
        if handler is None:
            self._logger.error(
                "Failure in place_reduce_only_exit: handler_missing",
                extra={
                    "event": "safe_reduce_only_exit_missing",
                    "symbol": intent.symbol,
                },
            )
            raise OrderPlacementError("Reduce-only exits unsupported")
        try:
            order_id = handler(intent)
        except OrderPlacementError as exc:
            self._logger.error(
                "Failure in place_reduce_only_exit: %s",
                exc,
                extra={
                    "event": "safe_reduce_only_exit_error",
                    "symbol": intent.symbol,
                },
            )
            raise
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in place_reduce_only_exit: %s",
                exc,
                extra={
                    "event": "safe_reduce_only_exit_error",
                    "symbol": intent.symbol,
                },
            )
            raise OrderPlacementError("Reduce-only exit failed") from exc

        if order_id:
            self._logger.info(
                "Condition met: reduce-only exit placed",
                extra={
                    "event": "safe_reduce_only_exit_success",
                    "symbol": intent.symbol,
                    "qty": intent.qty,
                    "order_id": order_id,
                },
            )
        else:
            self._logger.info(
                "Condition met: reduce-only exit skipped",
                extra={
                    "event": "safe_reduce_only_exit_skipped",
                    "symbol": intent.symbol,
                    "qty": intent.qty,
                },
            )
        return order_id

    # ------------------------------------------------------------------
    def _chase_fill(self, order_id: str, side: OrderSide, price: float) -> None:
        attempts = max(int(self.settings.replace_attempts), 0)
        if attempts <= 0 or price <= 0:
            return
        timeout = max(float(self.settings.replace_timeout_seconds), 0.0)
        step_pct = max(float(self.settings.replace_price_step_pct), 0.0) / 100.0
        backoff = max(float(self.settings.replace_backoff_seconds), 0.0)
        current_price = float(price)
        wait_for_fill = getattr(self.order_manager, "wait_for_fill", None)
        modify_order = getattr(self.order_manager, "modify_order", None)
        if not callable(wait_for_fill) or not callable(modify_order):
            return
        for attempt in range(1, attempts + 1):
            if timeout > 0 and wait_for_fill(order_id, timeout_sec=timeout):
                return
            if step_pct <= 0 or current_price <= 0:
                return
            delta = current_price * step_pct
            if side == "BUY":
                current_price += delta
            else:
                current_price = max(current_price - delta, 0.0)
            try:
                modified = modify_order(order_id, new_price=current_price)
            except NotImplementedError:
                return
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "order_replace_failed",
                    extra={
                        "event": "order_replace_failed",
                        "order_id": order_id,
                        "err": str(exc),
                    },
                )
                return
            if not modified:
                continue
            self._logger.info(
                "order_replaced",
                extra={
                    "event": "order_replace",
                    "order_id": order_id,
                    "attempt": attempt,
                    "new_price": current_price,
                },
            )
            if timeout > 0:
                timeout *= 1.5
            if backoff > 0:
                time.sleep(backoff)
                backoff *= 1.5

    # ------------------------------------------------------------------
    def _apply_price_offset(self, side: OrderSide, price: float | None) -> float | None:
        if price is None:
            return None
        offset_pct = max(self.settings.limit_offset_pct, 0.0) / 100.0
        if offset_pct == 0:
            return price
        delta = price * offset_pct
        if side == "BUY":
            return max(price - delta, 0.0)
        return price + delta

    def _apply_queue_policy(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType,
        price: float | None,
    ) -> tuple[OrderType, float | None]:
        """Apply queue aware adjustments returning updated order settings.

        Args:
            symbol: Instrument identifier under evaluation.
            side: Order side requested by the caller.
            quantity: Absolute quantity requested.
            order_type: Current order type requested by the caller.
            price: Price after limit offset adjustments.

        Returns:
            Tuple containing the resolved order type and price.

        Raises:
            OrderPlacementError: When the policy requests waiting out the queue.
        """

        self._logger.debug(
            "Entered _apply_queue_policy",
            extra={
                "event": "queue_policy_enter",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            },
        )
        settings = self.settings
        if not settings.enable_queue_policy:
            return order_type, price
        manager = self.order_manager
        if manager is None:
            return order_type, price

        best_bid: float | None = None
        best_ask: float | None = None
        depth_snapshot: Mapping[str, object] | None = None
        try:
            depth_fn = getattr(manager, "get_best_bid_ask_depth", None)
            if callable(depth_fn):
                depth_raw = depth_fn(symbol)  # type: ignore[call-arg]
                if isinstance(depth_raw, Mapping):
                    depth_snapshot = depth_raw
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _apply_queue_policy depth lookup: %s",
                exc,
                extra={
                    "event": "queue_policy_depth_error",
                    "symbol": symbol,
                },
            )
        if depth_snapshot is not None:
            bid_val = depth_snapshot.get("bid")
            ask_val = depth_snapshot.get("ask")
            if isinstance(bid_val, (int, float)):
                best_bid = float(bid_val)
            if isinstance(ask_val, (int, float)):
                best_ask = float(ask_val)

        queue_price = price
        if queue_price is None:
            if side == "BUY":
                queue_price = best_ask
            else:
                queue_price = best_bid

        queue_position = 0
        try:
            calc_fn = getattr(manager, "calculate_queue_position", None)
            if callable(calc_fn) and queue_price is not None and float(queue_price) > 0:
                queue_position = int(
                    calc_fn(symbol, side, float(queue_price))  # type: ignore[misc]
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _apply_queue_policy queue calc: %s",
                exc,
                extra={
                    "event": "queue_policy_calc_error",
                    "symbol": symbol,
                },
            )

        wait_threshold = max(int(settings.queue_wait_threshold), 0)
        cross_threshold = max(int(settings.queue_cross_threshold), 0)
        spread_bps: float | None = None
        if best_bid and best_ask and best_bid > 0 and best_ask > 0:
            spread = float(best_ask) - float(best_bid)
            mid = (float(best_ask) + float(best_bid)) / 2.0
            if mid > 0:
                spread_bps = (spread / mid) * 10_000.0

        if cross_threshold > 0 and queue_position >= cross_threshold:
            if spread_bps is not None and spread_bps <= float(
                settings.queue_cross_spread_bps
            ):
                self._logger.info(
                    "Condition met: queue_policy_cross",
                    extra={
                        "event": "queue_policy_cross",
                        "symbol": symbol,
                        "side": side,
                        "queue_position": queue_position,
                        "spread_bps": spread_bps,
                        "use_market": settings.queue_cross_use_market,
                    },
                )
                if settings.queue_cross_use_market:
                    return OrderType.MARKET, None
                cross_price = best_ask if side == "BUY" else best_bid
                if cross_price is not None:
                    return OrderType.LIMIT, float(cross_price)
            else:
                self._logger.info(
                    "Condition met: queue_policy_cross_block",
                    extra={
                        "event": "queue_policy_cross_block",
                        "symbol": symbol,
                        "side": side,
                        "queue_position": queue_position,
                        "spread_bps": spread_bps,
                        "threshold_bps": settings.queue_cross_spread_bps,
                    },
                )

        if wait_threshold > 0 and queue_position >= wait_threshold:
            self._logger.info(
                "Condition met: queue_policy_wait",
                extra={
                    "event": "queue_policy_wait",
                    "symbol": symbol,
                    "side": side,
                    "queue_position": queue_position,
                    "wait_threshold": wait_threshold,
                },
            )
            raise OrderPlacementError("queue_wait")

        return order_type, price

    def _check_throttle(self, now: float) -> bool:
        self._prune_window(self._per_second, now, 1.0)
        self._prune_window(self._per_minute, now, 60.0)
        self._prune_window(self._per_day, now, 86400.0)
        if len(self._per_second) >= self.settings.max_per_second:
            return False
        if len(self._per_minute) >= self.settings.max_per_minute:
            return False
        if len(self._per_day) >= self.settings.max_per_day:
            return False
        self._per_second.append(now)
        self._per_minute.append(now)
        self._per_day.append(now)
        return True

    @staticmethod
    def _prune_window(window: Deque[float], now: float, interval: float) -> None:
        cutoff = now - interval
        while window and window[0] < cutoff:
            window.popleft()

    def _record_rejection(self, symbol: str, reason: str) -> None:
        token = canonical(reason)
        reason_token = token if token != "OK" else "unknown"
        emit_diag(
            self._logger,
            "safe_order_rejection",
            reason=reason_token,
            severity="warning",
            alert=True,
            symbol=symbol,
        )
        self._rejections += 1
        try:
            self._m_rejections.inc()
        except Exception:  # pragma: no cover - optional metrics
            pass
        try:
            METRICS.record_order_rejection(reason=reason_token)
        except Exception:  # pragma: no cover - optional metrics
            self._logger.debug("record_order_rejection metric failed", exc_info=True)
        if self.on_order_rejected is not None:
            try:
                self.on_order_rejected(symbol, reason_token)
            except Exception:  # pragma: no cover - defensive
                self._logger.debug("on_order_rejected hook failed", exc_info=True)


__all__ = ["SafeOrderManager"]
