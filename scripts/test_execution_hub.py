"""Smoke test for the OrderExecutionHub integration."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Iterable

from nifty_scalper_bot.execution.execution_router import ExecutionResult
from nifty_scalper_bot.execution.lifecycle_manager import LifecycleManager
from nifty_scalper_bot.execution.order_execution_hub import OrderExecutionHub
from nifty_scalper_bot.execution.order_queue import OrderQueue, OrderRequest
from nifty_scalper_bot.execution.post_fill_monitor import PostFillMonitor
from nifty_scalper_bot.execution.preflight_validator import (
    PreFlightValidator,
    PreFlightValidatorSettings,
    ValidationResult,
)
from nifty_scalper_bot.execution.state_tracker import StateTracker, StateTrackerSettings
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class SmokePositionManager:
    """In-memory position manager used by the smoke test."""

    positions: list[dict[str, Any]] = field(default_factory=list)

    def get_all_positions(self) -> list[dict[str, Any]]:
        """Return stored positions for compatibility.

        Args:
            None.

        Returns:
            list[dict[str, Any]]: Snapshot of tracked positions.

        Raises:
            None.
        """

        return list(self.positions)


class SmokeRiskManager:
    """Stub risk manager satisfying :class:`PreFlightValidator` hooks."""

    def __init__(self) -> None:
        """Initialise default metrics for the smoke test.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self.current_balance = 100_000.0
        self._daily_realized = 0.0
        self._losses = 0
        self.settings = type("Settings", (), {"max_concurrent_positions": 5})()
        self.position_manager = SmokePositionManager()

    def snapshot(self) -> Any:
        """Return a minimal snapshot object.

        Args:
            None.

        Returns:
            Any: Object exposing attributes used by the validator.

        Raises:
            None.
        """

        return type(
            "Snapshot",
            (),
            {
                "daily_realized": self._daily_realized,
                "daily_loss_limit": 25_000.0,
                "losses_in_row": self._losses,
            },
        )()

    def get_margin_available(self) -> float:
        """Return available margin for smoke validation.

        Args:
            None.

        Returns:
            float: Margin headroom.

        Raises:
            None.
        """

        return 50_000.0

    def get_consecutive_losses(self) -> int:
        """Return number of consecutive losses (zero for smoke test).

        Args:
            None.

        Returns:
            int: Loss streak count.

        Raises:
            None.
        """

        return self._losses

    def get_daily_pnl(self) -> float:
        """Return accumulated daily PnL used for drawdown checks.

        Args:
            None.

        Returns:
            float: Realised PnL value.

        Raises:
            None.
        """

        return self._daily_realized


class SmokeRegimeManager:
    """Stub regime detector emitting a calm regime."""

    def get_current_regime(self) -> str:
        """Return a deterministic regime label.

        Args:
            None.

        Returns:
            str: Regime label.

        Raises:
            None.
        """

        return "CALM"

    def get_regime_confidence(self) -> float:
        """Return regime confidence score.

        Args:
            None.

        Returns:
            float: Confidence value.

        Raises:
            None.
        """

        return 0.5


class SmokeSessionGuard:
    """Stub session guard allowing trading by default."""

    def cantradenow(self) -> tuple[bool, str]:
        """Return session allowance tuple.

        Args:
            None.

        Returns:
            tuple[bool, str]: Trading allowed flag and diagnostic string.

        Raises:
            None.
        """

        return True, ""


class SmokeDataHub:
    """Minimal data hub providing quotes, candles, and IV."""

    def __init__(self) -> None:
        """Initialise quote caches and handler registry.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self._handlers: Dict[str, list[Callable[[dict[str, Any]], None]]] = {}
        self._price = 100.0
        self._timestamp = datetime.now(timezone.utc)

    def getlatesttick(
        self, symbol: str
    ) -> dict[str, Any]:  # noqa: D401 - compatibility
        return {
            "symbol": symbol,
            "timestamp": self._timestamp.timestamp(),
            "best_bid": self._price - 0.25,
            "best_ask": self._price + 0.25,
            "last_price": self._price,
        }

    def subscribe_ticks(
        self, symbol: str, handler: Callable[[dict[str, Any]], None]
    ) -> None:
        """Register ``handler`` for tick updates.

        Args:
            symbol: Instrument identifier.
            handler: Callable invoked for every tick.

        Returns:
            None.

        Raises:
            None.
        """

        self._handlers.setdefault(symbol, []).append(handler)

    def unsubscribe_ticks(
        self, symbol: str, handler: Callable[[dict[str, Any]], None]
    ) -> None:
        """Remove ``handler`` subscription when present.

        Args:
            symbol: Instrument identifier.
            handler: Callable previously registered.

        Returns:
            None.

        Raises:
            None.
        """

        handlers = self._handlers.get(symbol, [])
        if handler in handlers:
            handlers.remove(handler)
        if not handlers:
            self._handlers.pop(symbol, None)

    def emit_tick(self, symbol: str, price: float) -> None:
        """Emit synthetic ticks to lifecycle subscribers.

        Args:
            symbol: Instrument identifier.
            price: Last traded price to deliver.

        Returns:
            None.

        Raises:
            None.
        """

        self._price = price
        self._timestamp = datetime.now(timezone.utc)
        payload = {
            "symbol": symbol,
            "last_price": price,
            "best_bid": price - 0.25,
            "best_ask": price + 0.25,
        }
        for handler in list(self._handlers.get(symbol, [])):
            handler(payload)

    def get_recent_candles(
        self, symbol: str, limit: int = 15
    ) -> Iterable[dict[str, float]]:  # noqa: D401
        del symbol, limit
        base = self._price
        candles: list[dict[str, float]] = []
        for idx in range(16):
            offset = idx * 0.1
            candles.append({"high": base + offset, "low": base - offset, "close": base})
        return candles

    def get_iv(self, symbol: str) -> float:  # noqa: D401 - compatibility
        del symbol
        return 0.3


class InstrumentedPreFlightValidator(PreFlightValidator):
    """Preflight validator that records validation invocations."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialise validator and invocation counter.

        Args:
            **kwargs: Keyword arguments forwarded to ``PreFlightValidator``.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__(**kwargs)
        self.call_count = 0

    def validate(
        self, symbol: str, *, context: dict[str, Any] | None = None
    ) -> ValidationResult:  # type: ignore[override]
        """Run validation and increment the invocation counter.

        Args:
            symbol: Instrument identifier under evaluation.
            context: Optional supplemental context payload.

        Returns:
            ValidationResult: Outcome from the parent validator.

        Raises:
            None.
        """

        self.call_count += 1
        return super().validate(symbol, context=context)


@dataclass(slots=True)
class StaticExecutionRouter:
    """Deterministic execution router used for smoke testing."""

    def __init__(self) -> None:
        """Initialise execution statistics.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        self.calls: list[OrderRequest] = []

    def execute(self, request: OrderRequest) -> ExecutionResult:
        """Return an immediate fill for ``request``.

        Args:
            request: Order payload to execute.

        Returns:
            ExecutionResult: Synthetic fill result.

        Raises:
            None.
        """

        self.calls.append(request)
        fill_price = request.price or 101.0
        return ExecutionResult(
            status="FILLED",
            order_id=f"SIM-{len(self.calls)}",
            fill_quantity=request.quantity,
            fill_price=fill_price,
        )

    def get_stats(self) -> dict[str, float]:
        """Return routing statistics.

        Args:
            None.

        Returns:
            dict[str, float]: Count of executed orders.

        Raises:
            None.
        """

        return {"live_orders": float(len(self.calls))}


class SmokeBrokerClient:
    """Stub broker client returning empty positions."""

    def get_positions(self) -> list[dict[str, Any]]:
        """Return empty broker positions.

        Args:
            None.

        Returns:
            list[dict[str, Any]]: Empty snapshot.

        Raises:
            None.
        """

        return []


async def run_smoke(dry_run: bool) -> int:
    """Execute the smoke test sequence.

    Args:
        dry_run: Whether to suppress non-zero exit codes on failure.

    Returns:
        int: Exit code (``0`` on success).

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered run_smoke",
        extra={"event": "execution_hub_smoke_start", "dry_run": dry_run},
    )
    try:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "execution_state.db"
            tracker = StateTracker(StateTrackerSettings(db_path=str(db_path)))
            queue = OrderQueue()
            data_hub = SmokeDataHub()
            lifecycle = LifecycleManager(
                data_hub=data_hub,
                order_queue=queue,
                state_tracker=tracker,
                clock=lambda: datetime.now(timezone.utc),
            )
            risk = SmokeRiskManager()
            regime = SmokeRegimeManager()
            session = SmokeSessionGuard()
            validator = InstrumentedPreFlightValidator(
                risk_manager=risk,
                regime_manager=regime,
                datahub=data_hub,
                session_guard=session,
                settings=PreFlightValidatorSettings(),
            )
            router = StaticExecutionRouter()
            broker = SmokeBrokerClient()
            monitor = PostFillMonitor(
                broker_client=broker,
                state_tracker=tracker,
                interval_sec=30,
                alert_on_mismatch=True,
            )
            hub = OrderExecutionHub(
                state_tracker=tracker,
                preflight_validator=validator,
                lifecycle_manager=lifecycle,
                order_queue=queue,
                execution_router=router,
                post_fill_monitor=monitor,
                data_hub=data_hub,
            )
            await hub.start()
            symbol = "NIFTY25OCT25000CE"
            request = OrderRequest(
                symbol=symbol,
                side="BUY",
                quantity=25,
                intent="ENTRY",
                source="smoke_test",
                metadata={"atr": 5.0, "regime": "TREND"},
            )
            request.price = 100.5
            hub.submit_order_request(request)
            # Wait for the entry order to process.
            for _ in range(40):
                await asyncio.sleep(0.05)
                orders = tracker.get_orders_by_status("filled")
                if orders:
                    break
            orders = tracker.get_orders_by_status("filled")
            if not orders:
                raise RuntimeError("Entry order was not recorded")
            # Emit ticks to trigger lifecycle transitions.
            data_hub.emit_tick(symbol, 108.0)
            await asyncio.sleep(0.1)
            data_hub.emit_tick(symbol, 115.0)
            await asyncio.sleep(0.2)
            # Allow queued lifecycle exits to process.
            for _ in range(40):
                await asyncio.sleep(0.05)
                if not tracker.get_open_positions():
                    break
            await monitor.reconcile()
            stats = hub.get_stats()
            summary = {
                "orders_recorded": len(orders),
                "validator_calls": validator.call_count,
                "router_calls": len(router.calls),
                "positions_open": tracker.get_open_positions(),
                "hub_stats": stats,
                "reconciliation": monitor.get_stats(),
            }
            await hub.shutdown()
            tracker.close()
            LOGGER.info(
                "execution_hub_smoke_summary",
                extra={"event": "execution_hub_smoke_summary", **summary},
            )
            success = (
                summary["orders_recorded"] >= 1
                and summary["validator_calls"] >= 1
                and summary["router_calls"] >= 1
                and not summary["positions_open"]
            )
            if success:
                print("✅ All checks passed")
                return 0
            print("❌ Smoke test incomplete")
            return 0 if dry_run else 1
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in run_smoke: %s",
            exc,
            extra={"event": "execution_hub_smoke_error"},
            exc_info=exc,
        )
        print(f"❌ Smoke test failed: {exc}")
        return 0 if dry_run else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Raw argument list.

    Returns:
        argparse.Namespace: Parsed arguments.

    Raises:
        None.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Return success even if checks fail (useful for local experimentation)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI execution.

    Args:
        argv: Optional list of command line arguments.

    Returns:
        int: Process exit code.

    Raises:
        None.
    """

    LOGGER.debug("Entered main", extra={"event": "execution_hub_smoke_main"})
    args = parse_args(argv or sys.argv[1:])
    return asyncio.run(run_smoke(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
