"""FastAPI health and metrics endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

try:  # pragma: no cover - optional dependency
    from prometheus_client import multiprocess  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional dependency
    multiprocess = None  # type: ignore[assignment]

from nifty_scalper_bot.infra.metrics import METRICS, MetricsCollector
from nifty_scalper_bot.risk import RiskManager
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


if TYPE_CHECKING:
    from nifty_scalper_bot.execution.safe_order_manager import SafeOrderManager


def _prepare_registry(
    registry: CollectorRegistry | None,
) -> CollectorRegistry | None:
    """Return a CollectorRegistry configured for this process.

    Args:
        registry: Optional registry provided by the caller.

    Returns:
        CollectorRegistry | None: Prepared registry when creation succeeds.

    Raises:
        None.
    """

    if registry is not None:
        return registry
    try:
        prepared = CollectorRegistry()
        if multiprocess is not None and os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            multiprocess.MultiProcessCollector(prepared)  # type: ignore[attr-defined]
        return prepared
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure while preparing Prometheus registry: %s",
            exc,
            extra={"event": "prometheus_registry_error"},
            exc_info=exc,
        )
        return None


@dataclass(slots=True)
class HealthState:
    streamer: Any
    order_manager: SafeOrderManager
    risk_manager: RiskManager
    live_enabled: Callable[[], bool]
    registry: CollectorRegistry | None = None
    session_guard: object | None = None
    stream_supervisor: object | None = None
    metrics_collector: MetricsCollector | None = None
    websocket_enabled: bool = True
    selfchecker: object | None = None


def create_health_app(state: HealthState) -> FastAPI:
    registry = _prepare_registry(state.registry)
    if registry is not None:
        state.registry = registry
    app = FastAPI()

    @app.get("/health", response_class=JSONResponse)
    async def health() -> dict[str, object]:
        collector = state.metrics_collector or METRICS
        streamer_obj = state.streamer
        supervisor = getattr(state, "stream_supervisor", None)
        streaming_snapshot: dict[str, object] | None = None
        ws_enabled = bool(getattr(state, "websocket_enabled", True))
        if supervisor is not None:
            try:
                streaming_snapshot = supervisor.snapshot()  # type: ignore[assignment]
            except Exception as exc:  # pragma: no cover - defensive snapshot
                interval = float(
                    getattr(getattr(supervisor, "streamer", None), "_interval_s", 0.0)
                    or 0.0
                )
                batch = int(
                    getattr(getattr(supervisor, "streamer", None), "_batch_size", 0)
                    or 0
                )
                streaming_snapshot = {
                    "running": False,
                    "status": f"error: {exc}",
                    "tokens": 0,
                    "interval_s": interval,
                    "batch_size": batch,
                }
        connected = True
        backlog = 0
        if streamer_obj is not None:
            is_connected_fn = getattr(streamer_obj, "is_connected", None)
            if callable(is_connected_fn):
                try:
                    connected = bool(is_connected_fn())
                except Exception:  # pragma: no cover - defensive
                    connected = False
            tracked_tokens_fn = getattr(streamer_obj, "tracked_tokens", None)
            backlog_fn = getattr(streamer_obj, "backlog_size", None)
            if callable(backlog_fn):
                try:
                    backlog = int(backlog_fn())
                except Exception:  # pragma: no cover - defensive
                    backlog = 0
            elif callable(tracked_tokens_fn):
                try:
                    backlog = len(tracked_tokens_fn())
                except Exception:  # pragma: no cover - defensive
                    backlog = 0

        status_label = "ok"
        if ws_enabled and not connected:
            status_label = "degraded"
        if streaming_snapshot is not None and not bool(
            streaming_snapshot.get("running", False)
        ):
            status_label = "degraded"

        response: dict[str, object] = {
            "status": status_label,
            "websocket": {
                "enabled": ws_enabled,
                "connected": connected,
                "backlog": backlog,
                "mode": "websocket" if ws_enabled else "polling",
            },
            "orders": {
                "throttled": state.order_manager.throttled_count(),
                "rejections": state.order_manager.rejection_count(),
            },
            "risk": {
                "cooldown_seconds": state.risk_manager.cooldown_remaining(),
            },
            "live_enabled": state.live_enabled(),
        }

        if streaming_snapshot is not None:
            response["streaming"] = streaming_snapshot

        guard = getattr(state, "session_guard", None)
        if guard is not None:
            status_dict: dict[str, object] | None = None
            label: str | None = None
            try:
                last_status = getattr(guard, "last_status", None)
                current = last_status() if callable(last_status) else None
                if current is None:
                    evaluator = getattr(guard, "evaluate", None)
                    if callable(evaluator):
                        current = evaluator()
                if current is not None:
                    as_dict = current.as_dict() if hasattr(current, "as_dict") else None
                    if isinstance(as_dict, dict):
                        status_dict = as_dict
                        label = "ok" if as_dict.get("session_valid") else "blocked"
            except Exception:  # pragma: no cover - defensive
                label = "error"

            if label is not None:
                response["guard"] = label
            if status_dict is not None:
                response["guard_details"] = status_dict
                reasons = status_dict.get("reasons")
                if isinstance(reasons, list):
                    response["reasons"] = list(reasons)

        checker = getattr(state, "selfchecker", None)
        last_self_run = getattr(checker, "last_run", None) if checker else None
        if isinstance(last_self_run, datetime):
            last_self_test = last_self_run.isoformat()
        elif last_self_run is None:
            last_self_test = "never"
        else:
            last_self_test = str(last_self_run)
        last_trip_time = getattr(state.risk_manager, "_last_trip_time", None)
        if isinstance(last_trip_time, datetime):
            last_trip = last_trip_time.isoformat()
        elif last_trip_time is None:
            last_trip = "never"
        else:
            last_trip = str(last_trip_time)
        response["runtime_checks"] = {
            "last_self_test": last_self_test,
            "last_breaker_trip": last_trip,
        }

        if collector is not None:
            try:
                response["metrics"] = collector.metric_summary()
                response["alerts"] = collector.snapshot_alerts()
            except Exception as exc:  # pragma: no cover - defensive
                response["metrics_error"] = str(exc)

        return response

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> PlainTextResponse:
        active_registry = state.registry or registry
        payload = (
            generate_latest(active_registry)
            if active_registry is not None
            else generate_latest()
        )
        return PlainTextResponse(
            payload.decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


__all__ = ["HealthState", "create_health_app"]
