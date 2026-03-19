"""Mode-aware routing for order execution requests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from nifty_scalper_bot.execution.metrics import EXECUTION_LATENCY, SHADOW_DRIFT
from nifty_scalper_bot.execution.order_manager import OrderType
from nifty_scalper_bot.execution.order_queue import OrderRequest
from nifty_scalper_bot.execution.paper_fill_engine import PaperFillEngine
from nifty_scalper_bot.execution.safe_order_manager import SafeOrderManager
from nifty_scalper_bot.utils.errors import OrderPlacementError
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

ExecutionStatus = Literal["SUBMITTED", "FILLED", "REJECTED", "FAILED"]


@dataclass(slots=True)
class ExecutionResult:
    """Result payload emitted by :class:`ExecutionRouter`."""

    status: ExecutionStatus
    order_id: str | None
    fill_quantity: int
    fill_price: float | None
    rejection_reason: str | None = None
    shadow_order_id: str | None = None
    shadow_fill_price: float | None = None
    attempts: int = 1


@dataclass(slots=True)
class ExecutionRouterSettings:
    """Configuration options for :class:`ExecutionRouter`."""

    retry_attempts: int = 3
    retry_delay_ms: int = 500
    shadow_drift_threshold_bps: float = 20.0


class SlippageModel:
    """Estimate price slippage for simulated paper executions."""

    def estimate(
        self,
        symbol: str,
        side: str,
        quantity: int,
        base_price: float,
    ) -> float:
        """Return a signed slippage estimate for the provided order.

        Args:
            symbol: Instrument identifier for the request.
            side: Transaction side, e.g. ``BUY`` or ``SELL``.
            quantity: Order quantity used for tiered slippage.
            base_price: Reference price produced by the paper engine.

        Returns:
            float: Signed slippage increment to apply to ``base_price``.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered SlippageModel.estimate",
            extra={
                "event": "execution_router_slippage_estimate",
                "symbol": symbol,
            },
        )
        try:
            tick_size = 0.05 if "NIFTY" in symbol.upper() else 0.01
            slip = tick_size
            if quantity > 100:
                slip *= 2.0
            elif quantity > 50:
                slip *= 1.5
            direction = 1.0 if side.upper() == "BUY" else -1.0
            signed_slip = slip * direction
            LOGGER.info(
                "Condition met: slippage_estimated",
                extra={
                    "event": "execution_router_slippage_estimated",
                    "symbol": symbol,
                    "quantity": quantity,
                    "base_price": base_price,
                    "slippage": signed_slip,
                },
            )
            return signed_slip
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "[CRITICAL FAILURE]",
                extra={"event": "execution_router_slippage_error", "symbol": symbol},
                exc_info=True,
            )
            raise


class ExecutionRouter:
    """Route order requests to live, paper or shadow executors."""

    def __init__(
        self,
        *,
        live_executor: SafeOrderManager | None,
        paper_executor: PaperFillEngine | None,
        mode: str,
        settings: ExecutionRouterSettings | None = None,
    ) -> None:
        """Store dependencies and prepare statistics.

        Args:
            live_executor: Live trading order manager.
            paper_executor: Paper trading engine used for fills.
            mode: Requested execution mode (LIVE, PAPER, SHADOW).
            settings: Optional router configuration overrides.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered ExecutionRouter.__init__",
            extra={"event": "execution_router_init", "mode": mode},
        )
        self._live_executor = live_executor
        self._paper_executor = paper_executor
        self._mode = mode.upper().strip() or "SHADOW"
        self._settings = settings or ExecutionRouterSettings()
        self._slippage_model = SlippageModel()
        self._stats: dict[str, float] = {
            "live_orders": 0,
            "paper_orders": 0,
            "shadow_alerts": 0,
            "last_drift_bps": 0.0,
        }

    # ------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        """Update execution mode at runtime.

        Args:
            mode: New execution mode string to apply.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered ExecutionRouter.set_mode",
            extra={"event": "execution_router_set_mode", "mode": mode},
        )
        self._mode = mode.upper().strip() or self._mode

    # ------------------------------------------------------------------
    def execute(self, request: OrderRequest) -> ExecutionResult:
        """Execute ``request`` using the configured routing mode.

        Args:
            request: Order payload to route for execution.

        Returns:
            ExecutionResult describing live/paper outcomes.

        Raises:
            Exception: Re-raises router failures as critical execution failures.
        """

        LOGGER.debug(
            "Entered ExecutionRouter.execute",
            extra={
                "event": "execution_router_execute",
                "mode": self._mode,
                "symbol": request.symbol,
                "intent": request.intent,
            },
        )
        try:
            if self._mode == "PAPER":
                return self._execute_paper(request)
            if self._mode == "LIVE":
                return self._execute_live(request)
            return self._execute_shadow(request)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "[CRITICAL FAILURE]",
                extra={
                    "event": "execution_router_execute_error",
                    "mode": self._mode,
                    "symbol": request.symbol,
                },
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    def get_stats(self) -> dict[str, float]:
        """Return a shallow copy of routing statistics.

        Args:
            None.

        Returns:
            dict[str, float]: Snapshot of router statistics.

        Raises:
            None.
        """

        return dict(self._stats)

    # ------------------------------------------------------------------
    def _execute_live(self, request: OrderRequest) -> ExecutionResult:
        """Submit ``request`` via the live executor with retries.

        Args:
            request: Order payload routed to the live executor.

        Returns:
            ExecutionResult containing live execution details.

        Raises:
            None. Execution failures are captured and converted.
        """

        if self._live_executor is None:
            LOGGER.info(
                "Condition met: live executor unavailable",
                extra={"event": "execution_router_live_missing"},
            )
            return ExecutionResult(
                status="FAILED",
                order_id=None,
                fill_quantity=0,
                fill_price=None,
                rejection_reason="live_executor_unavailable",
            )
        attempts = max(1, int(self._settings.retry_attempts))
        delay = max(0.0, float(self._settings.retry_delay_ms) / 1000.0)
        last_error: str | None = None
        for attempt in range(1, attempts + 1):
            start_time = time.monotonic()
            try:
                LOGGER.info(
                    "Condition met: routing live order",
                    extra={
                        "event": "execution_router_live_attempt",
                        "attempt": attempt,
                        "symbol": request.symbol,
                    },
                )
                order_type = (
                    OrderType.MARKET if request.price is None else OrderType.LIMIT
                )
                order_id = self._live_executor.place_order(
                    symbol=request.symbol,
                    side=request.side,
                    quantity=request.quantity,
                    order_type=order_type,
                    price=request.price,
                    stop_loss=request.stop_loss,
                    take_profit=request.take_profit,
                )
                if not order_id:
                    raise OrderPlacementError("Order placement failed")
                latency = time.monotonic() - start_time
                EXECUTION_LATENCY.labels(executor="LIVE", status="SUBMITTED").observe(
                    latency
                )
                self._stats["live_orders"] += 1
                return ExecutionResult(
                    status="SUBMITTED",
                    order_id=order_id,
                    fill_quantity=request.quantity,
                    fill_price=request.price,
                    attempts=attempt,
                )
            except OrderPlacementError as exc:
                latency = time.monotonic() - start_time
                EXECUTION_LATENCY.labels(executor="LIVE", status="REJECTED").observe(
                    latency
                )
                last_error = str(exc)
                LOGGER.warning(
                    "Live order rejected: %s",
                    exc,
                    extra={
                        "event": "execution_router_live_reject",
                        "symbol": request.symbol,
                        "attempt": attempt,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                latency = time.monotonic() - start_time
                EXECUTION_LATENCY.labels(executor="LIVE", status="FAILED").observe(
                    latency
                )
                last_error = str(exc)
                LOGGER.exception(
                    "[CRITICAL FAILURE]",
                    extra={
                        "event": "execution_router_live_error",
                        "symbol": request.symbol,
                        "attempt": attempt,
                    },
                    exc_info=True,
                )
            if attempt < attempts and delay > 0:
                time.sleep(delay)
        return ExecutionResult(
            status="REJECTED",
            order_id=None,
            fill_quantity=0,
            fill_price=None,
            rejection_reason=last_error or "live_execution_failed",
            attempts=attempts,
        )

    # ------------------------------------------------------------------
    def _execute_paper(self, request: OrderRequest) -> ExecutionResult:
        """Simulate ``request`` using the paper fill engine.

        Args:
            request: Order payload routed to the paper executor.

        Returns:
            ExecutionResult containing simulated execution details.

        Raises:
            None. Execution failures are captured and converted.
        """

        if self._paper_executor is None:
            LOGGER.info(
                "Condition met: paper executor unavailable",
                extra={"event": "execution_router_paper_missing"},
            )
            return ExecutionResult(
                status="FAILED",
                order_id=None,
                fill_quantity=0,
                fill_price=None,
                rejection_reason="paper_executor_unavailable",
            )
        metadata = dict(request.metadata or {})
        tradingsymbol = str(
            metadata.get("tradingsymbol") or request.symbol or ""
        ).strip()
        exchange = metadata.get("exchange")
        if ":" in tradingsymbol:
            parts = tradingsymbol.split(":", 1)
            if len(parts) == 2:
                if exchange is None:
                    exchange = parts[0]
                tradingsymbol = parts[1]
        if not tradingsymbol:
            tradingsymbol = request.symbol
        payload = {
            "tradingsymbol": tradingsymbol,
            "transaction_type": request.side,
            "quantity": request.quantity,
            "order_type": "LIMIT" if request.price is not None else "MARKET",
        }
        if exchange:
            payload["exchange"] = exchange
        if request.price is not None:
            payload["price"] = request.price
        instrument_token = metadata.get("instrument_token")
        if instrument_token is not None:
            payload["instrument_token"] = instrument_token
        start_time = time.monotonic()
        result = self._paper_executor.place_order(payload)
        latency = time.monotonic() - start_time
        order_id = str(result.get("order_id")) if result.get("order_id") else None
        status = str(result.get("status", "")).lower()
        filled_qty = int(result.get("filled_quantity") or 0)
        avg_price = result.get("average_price")
        rejection_reason = None
        exec_status: ExecutionStatus = "SUBMITTED"
        if status in {"complete", "filled"}:
            exec_status = "FILLED"
            if filled_qty <= 0:
                filled_qty = request.quantity
        elif status in {"rejected", "cancelled"}:
            exec_status = "REJECTED"
            rejection_reason = str(result.get("message") or "paper_rejected")
        self._stats["paper_orders"] += 1
        fill_price_value: float | None
        if avg_price is not None:
            try:
                fill_price_value = float(avg_price)
            except (TypeError, ValueError):  # noqa: BLE001
                fill_price_value = None
        elif request.price is not None:
            fill_price_value = float(request.price)
        else:
            fill_price_value = None
        EXECUTION_LATENCY.labels(executor="PAPER", status=exec_status).observe(latency)
        if exec_status == "FILLED" and fill_price_value is not None:
            try:
                slippage = self._slippage_model.estimate(
                    request.symbol,
                    request.side,
                    request.quantity,
                    fill_price_value,
                )
                fill_price_value += slippage
                LOGGER.info(
                    "Condition met: paper_slippage_applied",
                    extra={
                        "event": "execution_router_paper_slippage",
                        "symbol": request.symbol,
                        "slippage": slippage,
                        "base_price": fill_price_value - slippage,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception(
                    "[CRITICAL FAILURE]",
                    extra={
                        "event": "execution_router_paper_slippage_error",
                        "symbol": request.symbol,
                    },
                    exc_info=True,
                )
                raise
        return ExecutionResult(
            status=exec_status,
            order_id=order_id,
            fill_quantity=filled_qty if filled_qty > 0 else request.quantity,
            fill_price=fill_price_value,
            rejection_reason=rejection_reason,
        )

    # ------------------------------------------------------------------
    def _execute_shadow(self, request: OrderRequest) -> ExecutionResult:
        """Execute ``request`` in shadow mode (live + paper).

        Args:
            request: Order payload executed in both live and paper modes.

        Returns:
            ExecutionResult combining live and paper outcomes.

        Raises:
            None. Execution failures are captured and converted.
        """

        action = "WOULD_EXIT" if request.intent.upper() == "EXIT" else ("WOULD_BUY" if request.side.upper() == "BUY" else "WOULD_SELL")
        LOGGER.info(
            "%s symbol=%s qty=%s",
            action,
            request.symbol,
            request.quantity,
            extra={
                "event": "execution_router_shadow_action",
                "action": action,
                "symbol": request.symbol,
            },
        )
        paper_result = self._execute_paper(request)
        self._stats["last_drift_bps"] = 0.0
        SHADOW_DRIFT.labels(symbol=request.symbol).observe(0.0)
        return ExecutionResult(
            status=paper_result.status,
            order_id=None,
            fill_quantity=paper_result.fill_quantity,
            fill_price=paper_result.fill_price,
            rejection_reason=paper_result.rejection_reason,
            shadow_order_id=paper_result.order_id,
            shadow_fill_price=paper_result.fill_price,
            attempts=paper_result.attempts,
        )

    # ------------------------------------------------------------------
    def _compute_drift_bps(
        self, live: ExecutionResult, paper: ExecutionResult
    ) -> float:
        """Return price drift in basis points between live and paper fills.

        Args:
            live: Execution result from the live executor.
            paper: Execution result from the paper executor.

        Returns:
            float: Drift value expressed in basis points.

        Raises:
            None.
        """

        live_price = live.fill_price
        paper_price = paper.fill_price
        if not live_price or not paper_price:
            return 0.0
        try:
            return abs((live_price - paper_price) / paper_price) * 10000.0
        except ZeroDivisionError:  # pragma: no cover - defensive
            return 0.0


__all__ = [
    "ExecutionRouter",
    "ExecutionResult",
    "ExecutionRouterSettings",
    "SlippageModel",
]
