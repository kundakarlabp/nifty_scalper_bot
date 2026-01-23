"""Core orchestration for the Nifty scalper trading bot.

Polling mode is more reliable for Railway/Heroku/Cloud deploys; WebSocket/
webhook should be used only on static public IP/server with trusted domain and
TLS certificate.
"""

# ruff: noqa: I001

from __future__ import annotations

import threading
import asyncio  # Required for startup reconciliation and background tasks
from contextlib import suppress
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, time, timedelta, timezone
from importlib import import_module
import inspect
import os
from pathlib import Path
from nifty_scalper_bot.data.robust_provider import RobustDataProvider, CircuitBreakerConfig
from nifty_scalper_bot.data.instruments import ensure_sqlite, load_rows_for_resolver
from nifty_scalper_bot.infra.watchdog import start_watchdog
from collections import OrderedDict
import random
import pytz
import sqlite3
import time as time_module
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Iterable,
    Literal,
    Mapping,
    TypedDict,
    TypeVar,
    cast,
)
import logging

LOGGER = logging.getLogger("nifty_scalper_bot.core.app")

from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

from nifty_scalper_bot.config.base import AppConfig
from nifty_scalper_bot.config.settings import Settings, get_settings
from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager
from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.core.message_bus import (
    MessageBus, 
    Message, 
    MessageType
)
from nifty_scalper_bot.core.unified_manager import UnifiedManager
from nifty_scalper_bot.data import (
    InstrumentResolver,
    InstrumentUniverseStatus,
    ensure_sqlite,
    load_rows_for_resolver,
    refresh_from_csv,
)
from nifty_scalper_bot.data.assess_data import assess_datahub_fresh
from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.data.market_regime import MarketRegimeDetector
from nifty_scalper_bot.data.persistent_state import PersistentStateManager
from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.data.rest.zerodha_ws_adapter import build_kite_ticker
from nifty_scalper_bot.data.websocket.manager import WebSocketManager
from nifty_scalper_bot.execution.bracket_manager import (
    BracketManager,
    SupportsCancelOrder,
)
from nifty_scalper_bot.execution.execution_router import (
    ExecutionRouter,
    ExecutionRouterSettings,
)
from nifty_scalper_bot.execution.lifecycle_manager import LifecycleManager
from nifty_scalper_bot.execution.order_execution_hub import OrderExecutionHub
from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType
from nifty_scalper_bot.execution.order_queue import OrderQueue
from nifty_scalper_bot.execution.paper_fill_engine import PaperFillEngine
from nifty_scalper_bot.execution.position_manager import ActiveContract, PositionManager
from nifty_scalper_bot.execution.post_fill_monitor import PostFillMonitor
from nifty_scalper_bot.execution.preflight_validator import PreFlightValidator
from nifty_scalper_bot.execution.safe_order_manager import SafeOrderManager
from nifty_scalper_bot.execution.state_tracker import StateTracker
from nifty_scalper_bot.infra.cron_refresh import schedule_instrument_refresh
from nifty_scalper_bot.infra.health import HealthState, create_health_app
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.infra.scheduled_tasks import start_background_tasks
from nifty_scalper_bot.infra.structured_logger import (
    emit_diag,
    setup_structured_logging,
)
from nifty_scalper_bot.notifications.telegram_commands import (
    Services as TelegramCommandServices,
    register_telegram_commands,
)
from nifty_scalper_bot.notifications.telegram_enhanced import TelegramEnhancedNotifier
from nifty_scalper_bot.notifications.telegram_webhook_enhanced import (
    TelegramWebhookController,
    register_webhook,
)
from nifty_scalper_bot.options.strike_selector import StrikeSelector
from nifty_scalper_bot.risk import RiskManager, RiskSnapshot, RiskState
from nifty_scalper_bot.risk.session_gate import build_session_guard
from nifty_scalper_bot.server import selftest_router
from nifty_scalper_bot.shadow.shadow_paper import ShadowPaperTrader
from nifty_scalper_bot.storage import HubStore
from nifty_scalper_bot.strategies.elite_strategies.builder import (
    build_elite_strategies,
    get_strategy_tags as elite_strategy_tags,
)
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.orchestrator import StrategyOrchestrator
from nifty_scalper_bot.strategies.runner import StrategyRunner, StrategyRunnerConfig
from nifty_scalper_bot.streaming import (
    PollingStreamer,
    ResilientStreamer,
    StreamSupervisor,
)
from nifty_scalper_bot.utils.config_validation import validate_execution_config
from nifty_scalper_bot.utils.env import (
    coalesce_bool,
    coalesce_float,
    coalesce_int,
    coalesce_str,
    get_bool,
    get_csv,
    get_str,
    normalize_path,
)
from nifty_scalper_bot.utils.errors import ConfigurationError
from nifty_scalper_bot.utils.logging import get_logger, setup_logging
from nifty_scalper_bot.utils.metrics import ensure_multiproc_dir
from nifty_scalper_bot.utils.rate_limiter import RateLimiter
from nifty_scalper_bot.utils.reasons import SOFT, canonical

from nifty_scalper_bot.execution.order_processor import OrderProcessor

if TYPE_CHECKING:
    from nifty_scalper_bot.notifications.telegram_controller import TelegramBot
    from nifty_scalper_bot.notifications.telegram_webhook_enhanced import (
        TelegramEnhancedNotifier,
    )
    from telegram.ext import Application

LOGGER = logging.getLogger("nifty_scalper_bot.core.app")

_ComponentT = TypeVar("_ComponentT")


def _require_component(component: _ComponentT | None, name: str) -> _ComponentT:
    """Return *component* when present, otherwise raise ``RuntimeError``.

    Args:
        component: Optional component instance to validate.
        name: Human-readable component name for diagnostics.

    Returns:
        _ComponentT: The validated component instance.

    Raises:
        RuntimeError: If ``component`` is ``None``.
    """

    LOGGER.debug(
        "Entered _require_component",
        extra={"event": "require_component", "component": name},
    )
    try:
        if component is None:
            raise RuntimeError(f"{name} is not configured")
        return component
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _require_component: %s",
            exc,
            extra={"event": "require_component_failure", "component": name},
            exc_info=exc,
        )
        raise


@dataclass(slots=True)
class InstrumentWarmResult:
    """Snapshot capturing the resolver warm-up outcome."""

    status: InstrumentUniverseStatus
    tokens: int
    options: int
    source: str
    csv_path: str | None
    db_path: str | None
    connection: sqlite3.Connection | None
    last_refresh: datetime | None


def _try_warm_instruments(
    settings: Settings,
    resolver: InstrumentResolver,
    logger: Any,
) -> InstrumentWarmResult:
    """Warm resolver from CSV when available without blocking startup.

    Args:
        settings: Runtime settings carrying instrument configuration.
        resolver: Resolver that should be populated with instrument data.
        logger: Structured logger instance for diagnostics.

    Returns:
        InstrumentWarmResult: Outcome describing the warm-up state.

    Raises:
        None.
    """

    logger.debug(
        "Entered _try_warm_instruments",
        extra={"event": "instrument_warm_enter"},
    )
    status = InstrumentUniverseStatus()
    instrument_tokens = 0
    instrument_options = 0
    instrument_source = "broker"
    last_refresh_ts: datetime | None = None
    csv_hint: str | None = None
    db_hint: str | None = None
    conn: sqlite3.Connection | None = None
    cache_settings = getattr(settings, "instruments", None)
    try:
        if cache_settings is None:
            logger.info(
                "Instrument settings not provided; warming resolver via broker.",
                extra={"event": "instrument_warm_broker_only"},
            )
            resolver.warm()
            instrument_tokens = len(getattr(resolver, "_symbol_by_token", {}))
            instrument_options = instrument_tokens
        else:
            csv_path = getattr(cache_settings, "csv_path", None)
            db_path = getattr(cache_settings, "db_path", None)
            if csv_path:
                csv_hint = str(csv_path)
            if db_path:
                db_hint = str(db_path)
            if db_path:
                conn = ensure_sqlite(str(db_path))
            if csv_path is None:
                logger.info(
                    "Instrument CSV path not configured; skipping CSV warm-up.",
                    extra={"event": "instrument_warm_csv_missing"},
                )
            elif not Path(csv_path).exists():
                logger.warning(
                    "Instrument CSV not found at %s; skipping warm-up.",
                    csv_path,
                    extra={
                        "event": "instrument_warm_csv_not_found",
                        "csv_path": str(csv_path),
                    },
                )
            elif conn is None:
                logger.warning(
                    "Instrument database path not configured; skipping cache refresh.",
                    extra={"event": "instrument_warm_db_missing"},
                )
            else:
                summary = refresh_from_csv(conn, str(csv_path))
                instrument_options = int(summary.get("stored") or instrument_options)
                instrument_source = "csv"
                last_refresh_ts = datetime.now(timezone.utc)
            if conn is not None:
                rows = load_rows_for_resolver(conn)
                if rows:
                    resolver.warm_from_broker_dump(rows)
                    instrument_tokens = len(rows)
                    if instrument_options <= 0:
                        instrument_options = len(rows)
                    if instrument_source != "csv":
                        instrument_source = "sqlite"
                else:
                    logger.info(
                        "Condition met: instrument_cache_empty; "
                        "falling back to broker warm-up.",
                        extra={"event": "instrument_warm_empty_cache"},
                    )
                    resolver.warm()
                    instrument_tokens = len(getattr(resolver, "_symbol_by_token", {}))
                    instrument_options = instrument_tokens
                    instrument_source = "broker"
            else:
                resolver.warm()
                instrument_tokens = len(getattr(resolver, "_symbol_by_token", {}))
                instrument_options = instrument_tokens
                instrument_source = "broker"
            if conn is not None:
                try:
                    count_row = conn.execute(
                        "SELECT COUNT(1) FROM instruments"
                    ).fetchone()
                    if count_row and count_row[0]:
                        instrument_tokens = instrument_tokens or int(count_row[0])
                        instrument_options = instrument_options or int(count_row[0])
                    ts_row = conn.execute(
                        "SELECT MAX(updated_at) FROM instruments"
                    ).fetchone()
                    if ts_row and ts_row[0]:
                        last_refresh_ts = datetime.fromisoformat(str(ts_row[0]))
                except Exception as meta_exc:  # noqa: BLE001
                    logger.error(
                        "Failure in _try_warm_instruments metadata query: %s",
                        meta_exc,
                        extra={"event": "instrument_warm_metadata_error"},
                        exc_info=meta_exc,
                    )
        if instrument_tokens <= 0:
            instrument_tokens = len(getattr(resolver, "_symbol_by_token", {}))
        if instrument_options <= 0:
            instrument_options = instrument_tokens
        status.record_refresh(
            tokens=instrument_tokens,
            options=instrument_options,
            source=instrument_source,
            path=csv_hint or db_hint,
            timestamp=last_refresh_ts,
        )
        logger.info(
            "Instrument warm-up complete: tokens=%s options=%s source=%s path=%s",
            status.tokens,
            status.options,
            status.last_source,
            csv_hint or db_hint or "n/a",
            extra={
                "event": "instrument_warm_complete",
                "tokens": status.tokens,
                "options": status.options,
                "source": status.last_source,
                "csv_path": csv_hint,
                "db_path": db_hint,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Instrument warm-up failed: %s",
            exc,
            extra={"event": "instrument_warm_failed"},
            exc_info=exc,
        )
        try:
            resolver.warm()
            fallback_tokens = len(getattr(resolver, "_symbol_by_token", {}))
        except Exception as fallback_exc:
            logger.warning(
                "Fallback resolver warm failed: %s",
                fallback_exc,
                extra={"event": "instrument_fallback_warm_failed"}
            )
            fallback_tokens = 0

        status.record_refresh(
            tokens=fallback_tokens,
            options=fallback_tokens,
            source="broker",
            path=csv_hint or db_hint,
        )
        instrument_tokens = status.tokens
        instrument_options = status.options
        instrument_source = status.last_source
    return InstrumentWarmResult(
        status=status,
        tokens=instrument_tokens,
        options=instrument_options,
        source=instrument_source,
        csv_path=csv_hint,
        db_path=db_hint,
        connection=conn,
        last_refresh=status.last_refresh,
    )


_HTTP_APP: FastAPI | None = None
_HTTP_NOTIFIER: TelegramEnhancedNotifier | None = None
_HTTP_CONTROLLER: TelegramWebhookController | None = None
_LATEST_CTX: "BotContext | None" = None


class _LifecycleTrackerAdapter:
    """Adapter exposing tracker hooks required by the lifecycle manager."""

    def __init__(self, tracker: StateTracker) -> None:
        """Store state tracker reference for delegation.

        Args:
            tracker: Concrete state tracker implementation.

        Returns:
            None.

        Raises:
            None.
        """

        self._tracker = tracker

    def record_lifecycle_event(
        self, symbol: str, event_type: str, payload: Mapping[str, Any] | None = None
    ) -> None:
        """Forward lifecycle events to the underlying tracker.

        Args:
            symbol: Trading symbol associated with the event.
            event_type: Lifecycle event type identifier.
            payload: Optional metadata describing the event.

        Returns:
            None.

        Raises:
            None. Errors are logged to preserve observability.
        """

        try:
            details = dict(payload or {})
            self._tracker.record_lifecycle_event(symbol, event_type, details)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleTrackerAdapter.record_lifecycle_event: %s",
                exc,
                extra={"event": "lifecycle_adapter_record_error", "symbol": symbol},
                exc_info=exc,
            )

    def get_open_positions(self) -> Iterable[Mapping[str, Any]]:
        """Return open positions from the underlying tracker.

        Args:
            None.

        Returns:
            Iterable[Mapping[str, Any]]: Snapshot of open positions.

        Raises:
            None. Errors are logged and an empty list returned.
        """

        try:
            return list(self._tracker.get_open_positions())
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleTrackerAdapter.get_open_positions: %s",
                exc,
                extra={"event": "lifecycle_adapter_positions_error"},
                exc_info=exc,
            )
            return []


try:  # pragma: no cover - optional dependency guard
    import prometheus_client  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - defensive fallback
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    generate_latest = None  # type: ignore[assignment]
    multiprocess = None  # type: ignore[assignment]
    CollectorRegistryRef: type[Any] | None = None
else:
    CONTENT_TYPE_LATEST = getattr(
        prometheus_client,
        "CONTENT_TYPE_LATEST",
        "text/plain; version=0.0.4; charset=utf-8",
    )
    generate_latest = getattr(prometheus_client, "generate_latest", None)
    CollectorRegistryRef = cast(
        type[Any] | None, getattr(prometheus_client, "CollectorRegistry", None)
    )
    try:  # pragma: no cover - optional dependency guard
        multiprocess = import_module("prometheus_client.multiprocess")
    except Exception:  # pragma: no cover - defensive fallback
        multiprocess = None  # type: ignore[assignment]


def _render_prometheus_metrics() -> tuple[str, str]:
    """Render Prometheus exposition payload and media type.

    Args:
        None.

    Returns:
        tuple[str, str]: Tuple containing payload text and media type string.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered _render_prometheus_metrics",
        extra={"event": "render_prometheus_metrics_enter"},
    )
    if generate_latest is None:
        LOGGER.error(
            "Failure in _render_prometheus_metrics: prometheus_client missing",
            extra={"event": "render_prometheus_metrics_missing_client"},
        )
        return "# prometheus_client_unavailable\n", "text/plain; charset=utf-8"
    try:
        registry_obj: Any | None = None
        if CollectorRegistryRef is not None:
            registry_obj = CollectorRegistryRef()
            if multiprocess is not None and os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
                try:
                    multiprocess.MultiProcessCollector(registry_obj)  # type: ignore[attr-defined]
                except Exception as exc:  # noqa: BLE001 - defensive multiprocess init
                    LOGGER.error(
                        "Failure in multiprocess collector setup: %s",
                        exc,
                        extra={"event": "render_prometheus_metrics_multiprocess_error"},
                        exc_info=exc,
                    )
                    registry_obj = None
        payload_bytes = (
            generate_latest(registry_obj)
            if registry_obj is not None
            else generate_latest()
        )
        media_type = CONTENT_TYPE_LATEST or "text/plain; version=0.0.4"
        return payload_bytes.decode("utf-8"), media_type
    except Exception as exc:  # noqa: BLE001 - defensive fallback
        LOGGER.error(
            "Failure in _render_prometheus_metrics: %s",
            exc,
            extra={"event": "render_prometheus_metrics_error"},
            exc_info=exc,
        )
        return "# prometheus_metrics_error\n", "text/plain; charset=utf-8"


def _normalize_broker_positions(snapshot: Any) -> list[Mapping[str, object]]:
    """Normalize broker position snapshot into mappings.

    Args:
        snapshot: Raw payload returned from the broker `get_positions` call.

    Returns:
        A list containing mapping entries for each broker position.

    Raises:
        ValueError: If the payload cannot be interpreted as position mappings.
    """

    LOGGER.debug("Entered _normalize_broker_positions")
    try:
        if snapshot is None:
            LOGGER.info(
                "Condition met: broker_position_snapshot_empty",
                extra={"event": "broker_position_snapshot_empty"},
            )
            return []
        if isinstance(snapshot, Mapping):
            LOGGER.info(
                "Condition met: broker_position_single_entry",
                extra={"event": "broker_position_single_entry"},
            )
            return [cast(Mapping[str, object], snapshot)]
        if isinstance(snapshot, Iterable) and not isinstance(snapshot, (str, bytes)):
            normalized: list[Mapping[str, object]] = []
            for item in snapshot:
                if isinstance(item, Mapping):
                    normalized.append(cast(Mapping[str, object], item))
            LOGGER.info(
                "Condition met: broker_position_snapshot_normalized",
                extra={
                    "event": "broker_position_snapshot_normalized",
                    "entries": len(normalized),
                },
            )
            return normalized
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _normalize_broker_positions: %s",
            exc,
            exc_info=exc,
            extra={"event": "broker_position_snapshot_normalize_failed"},
        )
        raise
    raise ValueError("Unsupported broker position payload shape")


def _fetch_positions_with_retry(
    broker_client: Any,
    *,
    max_attempts: int,
    backoff_min: float,
    backoff_max: float,
    backoff_multiplier: float,
    jitter_fraction: float,
    total_timeout_sec: float = 60.0,
) -> list[Mapping[str, object]]:
    LOGGER.debug("Entered _fetch_positions_with_retry")
    if broker_client is None:
        LOGGER.error(
            "Failure in _fetch_positions_with_retry: broker client missing",
            extra={"event": "broker_position_sync_missing_client"},
        )
        raise ValueError("broker_client is required for position sync")

    get_positions = getattr(broker_client, "get_positions", None)
    if not callable(get_positions):
        LOGGER.error(
            "Failure in _fetch_positions_with_retry: get_positions unavailable",
            extra={"event": "broker_position_sync_missing_getter"},
        )
        raise ValueError("broker_client.get_positions is unavailable")

    # [FIX] Variables defined at correct indentation level
    attempt = 0
    delay = max(backoff_min, 0.0)
    last_error: Exception | None = None
    start_time = time_module.monotonic()
    
    while attempt < max_attempts and (time_module.monotonic() - start_time) < total_timeout_sec:
        attempt += 1
        try:
            snapshot = get_positions()
            positions = _normalize_broker_positions(snapshot)
            LOGGER.info(
                "Condition met: broker_position_sync_success",
                extra={
                    "event": "broker_position_sync_success",
                    "attempt": attempt,
                    "positions": len(positions),
                },
            )
            return positions
        except Exception as exc:
            last_error = exc
            LOGGER.error(
                "Failure in _fetch_positions_with_retry: %s",
                exc,
                exc_info=exc,
                extra={
                    "event": "broker_position_sync_attempt_failed",
                    "attempt": attempt,
                },
            )
            
            if (time_module.monotonic() - start_time) + delay > total_timeout_sec:
                LOGGER.error("Broker position sync timed out")
                break

            sleep_window = min(backoff_max, max(delay, backoff_min))
            jitter_amplitude = max(0.0, jitter_fraction) * sleep_window
            if jitter_amplitude > 0.0:
                sleep_window += random.uniform(-jitter_amplitude, jitter_amplitude)
                sleep_window = max(backoff_min, sleep_window)
            
            LOGGER.info(
                "Condition met: broker_position_sync_retry_scheduled",
                extra={
                    "event": "broker_position_sync_retry_scheduled",
                    "attempt": attempt + 1,
                    "delay_sec": round(sleep_window, 3),
                },
            )
            time_module.sleep(sleep_window)
            delay = min(backoff_max, max(backoff_min, delay * backoff_multiplier))
            continue

    if last_error is not None:
        raise last_error
    return []


def get_latest_bot_context() -> "BotContext | None":
    """Return the most recently initialized bot context, if any."""

    return _LATEST_CTX


class PollTick(TypedDict, total=False):
    """Shape of normalized poll ticks consumed by the app."""

    instrument_token: int
    token: int | float | str
    source: str
    symbol: str
    ltp: float
    last_price: float
    close: float


@dataclass(slots=True)
class TradingSessionStatus:
    """Compact view of trading-session readiness."""

    session_valid: bool
    rate_limits_ok: bool
    market_open: bool
    risk_green: bool
    reasons: list[str]
    timestamp: datetime
    override_out_of_hours: bool = False
    fail_reason: str | None = None

    def all_clear(self) -> bool:
        return (
            self.session_valid
            and self.rate_limits_ok
            and self.risk_green
            and (self.market_open or self.override_out_of_hours)
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "market_open": self.market_open,
            "override_out_of_hours": self.override_out_of_hours,
            "session_valid": self.session_valid,
            "rate_limits_ok": self.rate_limits_ok,
            "risk_green": self.risk_green,
            "broker_session_valid": self.session_valid,
            "reasons": list(self.reasons),
            "timestamp": self.timestamp.isoformat(),
        }
        if self.fail_reason:
            payload["session_fail_reason"] = self.fail_reason
        return payload


class TradingSessionGuard:
    """Evaluate whether live trading can be safely enabled."""

    def __init__(
        self,
        *,
        rate_limiter: RateLimiter,
        risk_manager: RiskManager,
        market_open: time = time(9, 15),
        market_close: time = time(15, 30),
        session_max_age_hours: float = 22.0,
        timezone_name: str = "Asia/Kolkata",
        allow_out_of_hours: bool = True,
    ) -> None:
        self._rate_limiter = rate_limiter
        self._risk_manager = risk_manager
        self._market_open = market_open
        self._market_close = market_close
        self._session_max_age = timedelta(hours=session_max_age_hours)
        self._tz = ZoneInfo(timezone_name)
        self._session_validated_at: datetime | None = None
        self._last_status: TradingSessionStatus | None = None
        self._allow_out_of_hours = bool(allow_out_of_hours)

    def mark_session_valid(self) -> None:
        self._session_validated_at = datetime.now(timezone.utc)

    def reset_session_validation(self) -> None:
        """Invalidate previously marked broker sessions."""

        self._session_validated_at = None
        self._last_status = None

    def evaluate(self) -> TradingSessionStatus:
        now = datetime.now(timezone.utc)
        base_guard = build_session_guard(
            now=now,
            override=self._allow_out_of_hours,
            market_open=self._market_open,
            market_close=self._market_close,
        )
        raw_reasons = base_guard.get("reasons", [])
        reasons: list[str] = []
        if isinstance(raw_reasons, Iterable):
            reasons = [str(reason) for reason in raw_reasons]

        broker_session_valid = False
        if self._session_validated_at is None:
            reasons.append("Broker session not validated")
        else:
            broker_session_valid = (
                now - self._session_validated_at < self._session_max_age
            )
            
            # [FIX] Auto-refresh stale session
            if not broker_session_valid:
                LOGGER.warning(f"⚠️ Broker session stale (Age: {now - self._session_validated_at}). Attempting auto-refresh...")
                try:
                    # Attempt to fetch profile to validate connectivity
                    ctx = get_latest_bot_context()
                    if ctx and ctx.broker_client:
                        # Use internal broker reference if wrapped
                        client = getattr(ctx.broker_client, "client", ctx.broker_client)
                        if hasattr(client, "get_profile"):
                            client.get_profile() # Will raise if failed
                            self.mark_session_valid()
                            broker_session_valid = True
                            LOGGER.info("✅ Session auto-refreshed successfully.")
                except Exception as e:
                    LOGGER.error(f"❌ Session auto-refresh failed: {e}")
                    reasons.append("Broker session stale")

        budgets_ok = True
        snapshot = self._rate_limiter.snapshot()
        for name, bucket in snapshot.items():
            tokens = float(bucket.get("tokens", 0.0))
            capacity = max(float(bucket.get("capacity", 1.0)), 1.0)
            if tokens <= 0.1:
                budgets_ok = False
                reasons.append(f"Rate limit depleted: {name}")
                break
            if tokens < max(1.0, 0.1 * capacity):
                budgets_ok = False
                reasons.append(f"Rate limit low: {name}")
                break

        market_ok = bool(base_guard.get("market_open", False))
        override_active = bool(base_guard.get("override_out_of_hours", False))

        risk_ok = True
        risk_snapshot: RiskSnapshot | None = None
        risk_fail_reason: str | None = None

        
        if os.getenv("SKIP_APP_RISK", "").lower() == "true":
            risk_ok = True
            LOGGER.warning("⚠️ APP-LEVEL RISK CHECK BYPASSED via SKIP_APP_RISK")
        else:
            try:
                risk_ok = self._risk_manager.is_green()
                if not risk_ok:
                    snap = self._risk_manager.snapshot()
                    if snap:
                        LOGGER.warning(
                            f"⛔ RISK BLOCK: Breaker={snap.breaker_tripped} | "
                            f"Loss={snap.day_loss:.2f}/{snap.max_day_loss:.2f} | "
                            f"Cooldown={snap.cooldown_remaining:.1f}s"
                        )
            except Exception:
                risk_ok = False
                reasons.append("Risk manager unavailable")
                
        if not risk_ok and "Risk manager unavailable" not in reasons:
            try:
                risk_snapshot = self._risk_manager.snapshot()
            except Exception:  # pragma: no cover - defensive
                risk_snapshot = None
            if risk_snapshot is not None:
                if risk_snapshot.breaker_tripped:
                    risk_fail_reason = "BREAKER"
                elif risk_snapshot.cooldown_remaining > 0:
                    risk_fail_reason = "COOLDOWN"
                else:
                    source = (
                        risk_snapshot.last_rejection
                        or risk_snapshot.breaker_reason
                        or ""
                    )
                    risk_fail_reason = canonical(str(source))
                    if risk_fail_reason == "OK":
                        risk_fail_reason = "RISK_CHECK_FAILED"
            if risk_fail_reason is None:
                risk_fail_reason = "RISK_CHECK_FAILED"
            reasons.append(risk_fail_reason)

        status = TradingSessionStatus(
            session_valid=broker_session_valid,
            rate_limits_ok=budgets_ok,
            market_open=market_ok,
            risk_green=risk_ok,
            reasons=reasons,
            timestamp=now.astimezone(self._tz),
            override_out_of_hours=override_active,
            fail_reason=risk_fail_reason,
        )
        self._last_status = status
        return status

    def snapshot(self) -> dict[str, Any]:
        """Evaluate and return the guard payload as a dictionary."""

        status = self.evaluate()
        payload = status.as_dict()
        payload["rate_limits_ok"] = status.rate_limits_ok
        payload["risk_green"] = status.risk_green
        payload["broker_session_valid"] = status.session_valid
        session_ok = status.all_clear()
        fail_reason = "ok"
        if not session_ok:
            try:
                snapshot = self._risk_manager.snapshot()
            except Exception:  # pragma: no cover - defensive
                snapshot = None
            if snapshot is not None and snapshot.breaker_tripped:
                fail_reason = "BREAKER"
            elif snapshot is not None and snapshot.cooldown_remaining > 0:
                fail_reason = "COOLDOWN"
            elif status.fail_reason:
                fail_reason = status.fail_reason
            else:
                fail_reason = canonical(",".join(status.reasons))
                if fail_reason == "OK":
                    fail_reason = "unknown"
        payload["session_fail_reason"] = fail_reason
        return payload

    def _is_market_open(self, now_utc: datetime) -> bool:
        guard = build_session_guard(
            now=now_utc,
            override=self._allow_out_of_hours,
            market_open=self._market_open,
            market_close=self._market_close,
        )
        return bool(guard.get("market_open", False))

    def allow_live(self) -> tuple[bool, TradingSessionStatus]:
        status = self.evaluate()
        return status.all_clear(), status

    def last_status(self) -> TradingSessionStatus | None:
        return self._last_status

    def allow_out_of_hours(self) -> bool:
        return self._allow_out_of_hours

    def set_allow_out_of_hours(self, allow: bool) -> None:
        """Enable or disable trading outside configured hours."""

        self._allow_out_of_hours = bool(allow)

    def set_trading_window(self, start_hhmm: str, end_hhmm: str) -> None:
        """Update the trading window from HH:MM formatted strings."""

        self._market_open = self._parse_hhmm(start_hhmm, self._market_open)
        self._market_close = self._parse_hhmm(end_hhmm, self._market_close)

    def get_trading_window(self) -> tuple[time, time]:
        """Return the current trading window as ``(open, close)`` times."""

        return self._market_open, self._market_close

    @staticmethod
    def _parse_hhmm(value: str, fallback: time) -> time:
        try:
            clean = (value or "").strip()
            if not clean:
                return fallback
            
            if ":" not in clean:
                raise ValueError(f"Invalid time format (missing colon): {clean}")
            
            hour_str, minute_str = clean.split(":", 1)
            hour = int(hour_str)
            minute = int(minute_str)
            
            if not (0 <= hour <= 23):
                raise ValueError(f"Hour must be 0-23, got {hour}")
            if not (0 <= minute <= 59):
                raise ValueError(f"Minute must be 0-59, got {minute}")
            
            return time(hour, minute)
        except ValueError as exc:
            LOGGER.error(
                f"Invalid time format '{value}': {exc}",
                extra={"event": "parse_hhmm_invalid", "value": value}
            )
            raise ConfigurationError(f"Invalid time format '{value}': {exc}")
        except Exception as exc:
            LOGGER.warning(
                f"Unexpected error parsing time '{value}': {exc}",
                extra={"event": "parse_hhmm_error", "value": value}
            )
            return fallback

def _resolve_session_reason(
    status: TradingSessionStatus, snapshot: RiskSnapshot | None
) -> tuple[str, bool]:
    """Return canonical session reason and soft-override eligibility."""

    base_reason = canonical(status.fail_reason or ",".join(status.reasons))
    if snapshot is None:
        reason = (
            base_reason if base_reason != "OK" else canonical(",".join(status.reasons))
        )
        return reason, False

    if snapshot.breaker_tripped:
        return "BREAKER", False

    if snapshot.cooldown_remaining > 0:
        reason = "COOLDOWN"
    else:
        source = status.fail_reason or snapshot.last_rejection or ""
        reason = canonical(source)
        if reason == "OK" and snapshot.last_rejection:
            reason = canonical(snapshot.last_rejection)
        if reason == "OK" and status.reasons:
            reason = canonical(",".join(status.reasons))
    if reason == "OK":
        reason = base_reason

    soft_override = (
        status.session_valid
        and status.rate_limits_ok
        and status.market_open
        and reason in SOFT
    )
    return reason if reason else "OK", soft_override

def get_http_app() -> FastAPI:
    """Return the FastAPI application exposing inbound Telegram webhook."""
    global _HTTP_APP, _HTTP_NOTIFIER, _HTTP_CONTROLLER
    
    # Thread-safe singleton pattern
    if _HTTP_APP is not None:
        return _HTTP_APP
        
    settings = get_settings()

    telemetry_logger = get_logger("telegram.bootstrap")

    raw_webhook_env = os.getenv("TELEGRAM__WEBHOOK_ENABLED")
    if raw_webhook_env is None:
        raw_webhook_env = os.getenv("TELEGRAM_WEBHOOK_ENABLED")
    webhook_env_requested = str(raw_webhook_env or "false").strip().lower() == "true"

    if not webhook_env_requested and settings.notifications.webhook_enabled:
        settings.notifications.webhook_enabled = False

    app = FastAPI()
    app.state.ctx_getter = get_latest_bot_context
    _HTTP_APP = app

    notifier = TelegramEnhancedNotifier.from_settings(settings.notifications)
    _HTTP_NOTIFIER = notifier

    controller: TelegramWebhookController | None = None
    if settings.notifications.enabled:
        if notifier is None:
            telemetry_logger.warning(
                "telegram_controller_skipped",
                extra={"event": "controller_skipped", "reason": "no_notifier"},
            )
        else:
            controller = TelegramWebhookController(
                bot=notifier.bot,
                settings=settings.notifications,
            )
            app.include_router(controller.router)
            _HTTP_CONTROLLER = controller
    else:
        telemetry_logger.info(
            "telegram_disabled",
            extra={"event": "telegram_disabled", "reason": "notifications_disabled"},
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics() -> PlainTextResponse:
        """Serve Prometheus metrics payload for observability scrapes.

        Args:
            None.

        Returns:
            PlainTextResponse: Response containing Prometheus metrics payload.

        Raises:
            None.
        """

        telemetry_logger.debug(
            "Entered prometheus_metrics",
            extra={"event": "http_metrics_enter"},
        )
        try:
            payload, media_type = _render_prometheus_metrics()
        except Exception as exc:  # noqa: BLE001 - defensive
            telemetry_logger.error(
                "Failure in prometheus_metrics: %s",
                exc,
                extra={"event": "http_metrics_render_error"},
                exc_info=exc,
            )
            payload = "# prometheus_metrics_render_error\n"
            media_type = "text/plain; charset=utf-8"
        return PlainTextResponse(payload, media_type=media_type)

    @app.get("/health", response_class=JSONResponse)
    async def http_health() -> JSONResponse:
        ctx = get_latest_bot_context()
        
        # ✅ FIX 1: Handle Startup Gracefully (Return 200, not 503)
        # This prevents Railway from killing the bot while it initializes.
        # ✅ FIX: Return 200 during startup
        if not ctx:
            return JSONResponse(
                status_code=200, 
                content={
                    "status": "starting", 
                    "ready": False, 
                    "reason": "Context initializing...",
                },
            )

        # ✅ FIX 2: Comprehensive Component Checks
        checks = {
            "broker": ctx.broker_client is not None and ctx.broker_client.is_connected(),
            "position_manager": ctx.position_manager is not None,
            "risk_manager": ctx.risk_manager is not None,
            "data_hub": ctx.data_hub is not None,
        }
        
        # Optional: Check Risk Breaker if available
        if ctx.risk_manager:
            try:
                # Assuming snapshot() or similar property exists
                checks["risk_breaker_ok"] = not getattr(ctx.risk_manager, "breaker_tripped", False)
            except Exception:
                checks["risk_breaker_ok"] = False

        # ✅ FIX 3: Determine Status
        all_healthy = all(checks.values())
        status = "healthy" if all_healthy else "degraded"
        
        # We return 200 even if degraded so we can see the status JSON.
        # Only return 503 if something catastrophic (like broker down) happens in Production mode.
        # For now, 200 is safer for stability.
        return JSONResponse(
            status_code=200,
            content={
                "status": status,
                "ready": all_healthy,
                "checks": checks,
                "uptime_seconds": int(time_module.monotonic() - _START_TIME),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    @app.on_event("startup")
    async def _startup_webhook() -> None:
        telegram_logger = get_logger("telegram")
        notif_settings = settings.notifications
        if (
            controller
            and notif_settings.webhook_enabled
            and notif_settings.public_base_url
        ):
            registered = await register_webhook(
                controller.bot,
                notif_settings.public_base_url,
                logger=telegram_logger,
            )
            if registered:
                telegram_logger.info(
                    "telegram_webhook_ready",
                    extra={"event": "webhook_ready", "webhook_ready": True},
                )
            else:
                if notif_settings.allow_poll_fallback:
                    await controller.activate_polling_fallback("webhook setup failed")
                else:
                    telegram_logger.warning(
                        "telegram_webhook_registration_failed",
                        extra={
                            "event": "webhook_failed",
                            "public_url": notif_settings.public_base_url,
                            "fallback_enabled": False,
                        },
                    )
        else:
            if controller and notif_settings.allow_poll_fallback:
                await controller.activate_polling_fallback("webhook url missing")
                telegram_logger.info(
                    "telegram_polling_started_no_webhook",
                    extra={"event": "polling_started", "reason": "no_public_url"},
                )
            else:
                disabled_via_env = (
                    not webhook_env_requested or not notif_settings.webhook_enabled
                )
                webhook_configured = bool(
                    (notif_settings.public_base_url or "").strip()
                )
                extra_payload = {
                    "event": "webhook_not_configured",
                    "public_url_set": webhook_configured,
                    "controller_ready": bool(controller),
                    "enabled": notif_settings.enabled,
                    "fallback_enabled": notif_settings.allow_poll_fallback,
                    "disabled_via_env": disabled_via_env,
                }

                if not notif_settings.webhook_enabled:
                    telegram_logger.info(
                        "Telegram running in polling mode (webhook disabled)",
                        extra=extra_payload,
                    )
                elif not webhook_configured:
                    if webhook_env_requested:
                        telegram_logger.warning(
                            "telegram_webhook_not_configured",
                            extra=extra_payload,
                        )
                    else:
                        telegram_logger.debug(
                            "telegram_webhook_not_configured",
                            extra=extra_payload,
                        )
                else:
                    telegram_logger.debug(
                        "telegram_webhook_configuration_state",
                        extra=extra_payload,
                    )

    # ----------------------------------------------------------------
    # ✅ FIX: Explicit Telegram Startup for Python-Telegram-Bot v20+
    # ----------------------------------------------------------------
    @app.on_event("startup")
    async def _start_telegram_bot_service() -> None:
        """
        Start Telegram Bot and send a rich startup notification.
        """
        # No import needed here; get_latest_bot_context is defined globally in app.py
        
        try:
            ctx = get_latest_bot_context()
            if ctx and ctx.telegram_bot:
                LOGGER.info("🚀 Triggering TelegramBot explicit startup...")
                
                # 1. Start the Bot (Connects to API)
                await ctx.telegram_bot.start()
                
                # 2. Wait for connection stability (Critical for preventing send errors)
                await asyncio.sleep(2.0)
                
                # 3. Determine Bot State details for the message
                mode_icon = "🔴" if ctx.settings.enable_live else "🟡"
                mode_text = "LIVE TRADING" if ctx.settings.enable_live else "PAPER / SHADOW"
                
                # 4. Construct Rich HTML Message
                startup_msg = (
                    f"<b>{mode_icon} Nifty Scalper Bot Online</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🕒 <b>Time:</b> <code>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>\n"
                    f"⚙️ <b>Mode:</b> <code>{mode_text}</code>\n"
                    f"📡 <b>System:</b> <code>Initialized & Monitoring</code>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"<i>Waiting for market data...</i>"
                )
                
                # 5. Send Message using the internal application
                if ctx.telegram_bot._app:
                    await ctx.telegram_bot._app.bot.send_message(
                        chat_id=ctx.settings.notifications.chat_id,
                        text=startup_msg,
                        parse_mode="HTML"
                    )
                    LOGGER.info("✅ Startup message sent to Telegram.")

        except Exception as exc:
            # We log this but don't crash the app if Telegram fails
            LOGGER.error(f"❌ Failed to start Telegram/Send Message: {exc}", exc_info=True)
            
            
    @app.on_event("shutdown")
    async def _stop_telegram_bot_service() -> None:
        """Ensure clean shutdown of Telegram Bot."""
        try:
            ctx = get_latest_bot_context()
            if ctx and ctx.telegram_bot:
                LOGGER.info("🛑 Shutting down Telegram Bot...")
                await ctx.telegram_bot.stop()
        except Exception:
            pass
    # ----------------------------------------------------------------

    app.include_router(selftest_router)
    # Ensure the instrument resolver is warmed from any broker dump / csv
    # when the FastAPI app starts. This guarantees resolver lookups (symbols→tokens)
    # work before handlers or external probes rely on the resolver.
    @app.on_event("startup")
    async def _warm_instrument_resolver_on_startup() -> None:
        """
        Force fresh instrument sync on startup efficiently.
        Uses smart caching to avoid Rate Limits.
        """
        try:
            ctx = get_latest_bot_context()
            if ctx is None:
                LOGGER.debug("Resolver warm skipped: no bot context")
                return

            # [FIX] Initialize variable safely
            resolver = getattr(ctx, "instrument_resolver", None) or getattr(ctx, "resolver", None)
            
            if resolver is None:
                LOGGER.debug("Resolver warm skipped: no resolver found")
                return

            LOGGER.info("Verifying Instrument Cache...")
            # [FIX] Run warm in a thread to prevent blocking the event loop
            await asyncio.to_thread(resolver.warm)

            tokens = len(getattr(resolver, "_symbol_by_token", {}))
            LOGGER.info(
                f"✅ Instrument Resolver Ready. Active Tokens: {tokens}", 
                extra={"event": "instrument_warm_complete", "tokens": tokens}
            )

        except Exception as exc:
            LOGGER.error(f"Instrument warm-up failed: {exc}", exc_info=True)


def get_telegram_notifier() -> TelegramEnhancedNotifier | None:
    """Return the notifier created for webhook delivery, if any."""

    if _HTTP_NOTIFIER is None and _HTTP_APP is None:
        get_http_app()
    return _HTTP_NOTIFIER


@dataclass(slots=True)
class BotContext:
    """Container for all bot components."""

    settings: Settings
    config: AppConfig
    rate_limiter: RateLimiter
    broker_client: ZerodhaKiteClient
    websocket_client: Any | None
    websocket_manager: WebSocketManager | None
    streamer: Any
    stream_supervisor: StreamSupervisor | None
    message_bus: MessageBus
    order_processor: OrderProcessor | None = None
    data_hub: DataHub | None = None
    market_data_manager: MarketDataManager | None = None
    market_regime: MarketRegimeDetector | None = None
    market_regime_manager: MarketRegimeManager | None = None
    indicator_engine: IndicatorEngine | None = None
    position_manager: PositionManager | None = None
    risk_manager: RiskManager | None = None
    persistent_state: PersistentStateManager | None = None
    order_manager: OrderManager | None = None
    bracket_manager: Any | None = None
    paper_engine: PaperFillEngine | None = None
    safe_order_manager: SafeOrderManager | None = None
    order_queue: OrderQueue | None = None
    state_tracker: StateTracker | None = None
    preflight_validator: PreFlightValidator | None = None
    lifecycle_manager: LifecycleManager | None = None
    execution_router: ExecutionRouter | None = None
    post_fill_monitor: PostFillMonitor | None = None
    order_execution_hub: OrderExecutionHub | None = None
    strategy_manager: StrategyManager | None = None
    strategy_runner: StrategyRunner | None = None
    unified_manager: UnifiedManager | None = None
    instrument_resolver: InstrumentResolver | None = None
    instrument_db: sqlite3.Connection | None = None
    instrument_universe: InstrumentUniverseStatus | None = None
    instrument_refresh_task: asyncio.Task[Any] | None = None
    websocket_enabled: bool = True
    shadow_mode_enabled: bool = False
    shadow_trader: ShadowPaperTrader | None = None
    out_of_hours_override: bool = False
    telegram_bot: "TelegramBot | None" = None
    telegram_application: "Application | None" = None
    telegram_notifier: TelegramEnhancedNotifier | None = None
    health_app: FastAPI | None = None
    session_guard: TradingSessionGuard | None = None
    selfchecker: "RuntimeSelfChecker | None" = None
    underlying_spot_prices: OrderedDict[str, float] = field(
        default_factory=lambda: OrderedDict()
    )
    
    def update_spot_price(self, underlying: str, price: float, max_size: int = 100) -> None:
        """Update spot price with LRU eviction."""
        self.underlying_spot_prices[underlying] = price
        # Evict oldest entry if exceeds limit
        while len(self.underlying_spot_prices) > max_size:
            self.underlying_spot_prices.popitem(last=False)

class PersistentHeartbeatFlusher:
    """Flush :class:`PersistentStateManager` data on heartbeat cadence.

    Args:
        manager: Persistent state manager to flush.
        interval_sec: Minimum seconds between consecutive flushes.

    Returns:
        None.

    Raises:
        ValueError: If ``interval_sec`` is non-positive.
    """

    def __init__(
        self, manager: PersistentStateManager, interval_sec: float = 5.0
    ) -> None:
        self._logger = LOGGER
        self._manager = manager
        self._interval = max(float(interval_sec), 0.0)
        if self._interval <= 0.0:
            raise ValueError("interval_sec must be positive")
        self._last_flush = 0.0

    def handle_heartbeat(self, timestamp: float | None) -> None:
        """Flush persistent state when the heartbeat advances enough.

        Args:
            timestamp: Monotonic timestamp captured at the heartbeat.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PersistentHeartbeatFlusher.handle_heartbeat",
            extra={"event": "persistent_heartbeat_handle"},
        )
        now = float(timestamp) if timestamp is not None else time_module.monotonic()
        if self._last_flush > 0.0 and (now - self._last_flush) < self._interval:
            return
        flush_started = time_module.monotonic()
        try:
            self._manager.flush()
        except Exception as exc:  # noqa: BLE001
            flush_latency = time_module.monotonic() - flush_started
            try:
                METRICS.record_heartbeat_flush(
                    success=False,
                    latency_seconds=flush_latency,
                    now=time_module.time(),
                )
            except Exception as metrics_exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in PersistentHeartbeatFlusher.flush metrics: %s",
                    metrics_exc,
                    extra={"event": "persistent_heartbeat_flush_metric_error"},
                    exc_info=metrics_exc,
                )
            emit_diag(
                self._logger,
                "persistent_heartbeat_flush_failure",
                reason="flush_error",
                severity="critical",
                alert=True,
                latency_seconds=flush_latency,
            )
            self._logger.error(
                "Failure in PersistentHeartbeatFlusher.handle_heartbeat: %s",
                exc,
                exc_info=exc,
            )
            return
        flush_latency = time_module.monotonic() - flush_started
        self._last_flush = now
        try:
            METRICS.record_heartbeat_flush(
                success=True,
                latency_seconds=flush_latency,
                now=time_module.time(),
            )
        except Exception as metrics_exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PersistentHeartbeatFlusher.flush metrics: %s",
                metrics_exc,
                extra={"event": "persistent_heartbeat_flush_metric_error"},
                exc_info=metrics_exc,
            )
        emit_diag(
            self._logger,
            "persistent_heartbeat_flush",
            reason="ok",
            severity="info",
            interval_sec=self._interval,
            timestamp=now,
            latency_seconds=flush_latency,
        )


class RuntimeSelfChecker:
    """Run runtime self-tests to detect silent subsystem failures."""

    def __init__(self, context: BotContext, interval_seconds: float = 300.0) -> None:
        """Initialize the runtime self-check helper.

        Args:
            context: Live bot context exposing subsystem references.
            interval_seconds: Desired cadence for periodic checks in seconds.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger = LOGGER
        self._context = context
        self.interval_seconds = max(float(interval_seconds), 60.0)
        self.last_run: datetime | None = None
        self.last_results: dict[str, dict[str, object]] = {}

    def run_full_check(self) -> dict[str, dict[str, object]]:
        """Execute all configured runtime self-checks.

        Args:
            None.

        Returns:
            Mapping of check names to result dictionaries.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered RuntimeSelfChecker.run_full_check",
            extra={"event": "runtime_self_check_enter"},
        )
        results: dict[str, dict[str, object]] = {}
        for name, check in self._collect_checks().items():
            try:
                ok, detail, meta = check()
            except Exception as exc:  # noqa: BLE001 - defensive surface
                self._logger.error(
                    "Failure in RuntimeSelfChecker.run_full_check: %s",
                    exc,
                    extra={"event": "runtime_self_check_error", "check": name},
                    exc_info=exc,
                )
                ok = False
                detail = f"exception:{exc}"[:256]
                meta = {}
            meta_payload = dict(meta or {})
            meta_payload.setdefault("check", name)
            results[name] = {"ok": bool(ok), "detail": detail, "meta": meta_payload}
            if not ok:
                self._logger.error(
                    "Runtime self-test detected failure",
                    extra={
                        "event": "runtime_self_check_failed",
                        "check": name,
                        "detail": detail,
                        "meta": meta_payload,
                    },
                )
        self.last_run = datetime.now(timezone.utc)
        self.last_results = results
        return results

    def _collect_checks(
        self,
    ) -> dict[str, Callable[[], tuple[bool, str, dict[str, object]]]]:
        """Return mapping of check names to callable probes.

        Args:
            None.

        Returns:
            Dictionary mapping check identifiers to callables.

        Raises:
            None.
        """

        return {
            "data_freshness": self._check_data_freshness,
            "streamer": self._check_streamer,
            "risk_breaker": self._check_risk_breaker,
            "session_guard": self._check_session_guard,
        }

    def _check_data_freshness(self) -> tuple[bool, str, dict[str, object]]:
        """Verify recent quote availability for active trading symbols."""
        hub = self._context.data_hub
        if hub is None:
            return True, "no_data_hub", {}

        # [FIX] Use actually tracked symbols from DataHub to prevent false negatives
        # Accessing protected member _quotes is necessary here for introspection
        symbols = list(getattr(hub, "_quotes", {}).keys())

        # During startup / hydration, do NOT block trading if no quotes yet
        if not symbols:
            return True, "no_symbols_yet", {}

        # Pick the most liquid / reliable symbol (Spot index preferred)
        symbol = None
        for s in symbols:
            if "NIFTY" in s and "NSE" in s:
                symbol = s
                break
        
        # Fallback to any available symbol if index not found
        if symbol is None:
            symbol = symbols[0]

        interval = getattr(self._context.streamer, "_interval_s", 0.7) or 0.7
        # Adaptive threshold: 2.5x poll interval, clamped 2s-5s
        adaptive_ms = max(2000, min(5000, int(float(interval) * 1000.0 * 2.5)))

        ok, detail, meta = assess_datahub_fresh(
            hub,
            symbol,
            freshness_ms=adaptive_ms
        )

        payload = cast(dict[str, object], dict(meta or {}))
        payload["symbol_checked"] = symbol
        payload["adaptive_ms"] = adaptive_ms

        return ok, detail, payload

    def _check_streamer(self) -> tuple[bool, str, dict[str, object]]:
        """Assess market data streamer connectivity and backlog state.

        Args:
            None.

        Returns:
            Tuple containing success flag, detail token, and metadata dictionary.

        Raises:
            None.
        """

        streamer = self._context.streamer
        if streamer is None:
            return True, "no_streamer", {}
        connected = True
        backlog = 0
        detail = "ok"
        is_connected_fn = getattr(streamer, "is_connected", None)
        if callable(is_connected_fn):
            with suppress(Exception):
                connected = bool(is_connected_fn())
        backlog_fn = getattr(streamer, "backlog_size", None)
        if callable(backlog_fn):
            with suppress(Exception):
                backlog = int(backlog_fn())
        detail = "disconnected" if not connected else detail
        if backlog > 1000:
            detail = "backlog_high"
            connected = False
        payload = cast(
            dict[str, object],
            {"connected": connected, "backlog": backlog},
        )
        return connected, detail, payload

    def _check_risk_breaker(self) -> tuple[bool, str, dict[str, object]]:
        """Confirm the risk circuit breaker is not engaged.

        Args:
            None.

        Returns:
            Tuple with boolean status, detail string, and metadata.

        Raises:
            None.
        """

        risk_manager = self._context.risk_manager
        if risk_manager is None:
            return False, "risk_manager_missing", {}
        try:
            tripped, reason = risk_manager.is_circuit_breaker_tripped()
        except Exception as exc:  # noqa: BLE001 - defensive
            return False, f"exception:{exc}"[:256], {}
        return (
            (not tripped),
            (reason or "ok") if tripped else "ok",
            {
                "breaker_tripped": tripped,
                "reason": reason or "",
            },
        )

    def _check_session_guard(self) -> tuple[bool, str, dict[str, object]]:
        """Validate trading session guard status remains healthy.

        Args:
            None.

        Returns:
            Tuple indicating success, detail, and metadata dictionary.

        Raises:
            None.
        """

        guard = self._context.session_guard
        if guard is None:
            return True, "no_guard", {}
        try:
            status = guard.evaluate()
        except Exception as exc:  # noqa: BLE001 - guard errors surface
            return False, f"exception:{exc}"[:256], {}
        if status is None:
            return False, "no_status", {}
        as_dict = status.as_dict() if hasattr(status, "as_dict") else {}
        session_valid = bool(getattr(status, "session_valid", False))
        return session_valid, "ok" if session_valid else "blocked", dict(as_dict)

    def _resolve_symbol(self) -> str:
        """Return primary symbol used for runtime data checks."""
        symbols = getattr(self._context.config, "symbols", None)
        if isinstance(symbols, Iterable) and not isinstance(symbols, (str, bytes)):
            for candidate in symbols:
                candidate_str = str(candidate).strip()
                if candidate_str:
                    return candidate_str
        if isinstance(symbols, (str, bytes)) and symbols:
            return str(symbols)
        
        # ✅ FIX: Return the standard Zerodha format
        return "256265"


def _configure_rate_limiter(cfg: Any) -> RateLimiter:
    """Configure the rate limiter from nested configuration."""

    limiter = RateLimiter()
    limiter.configure_bucket(
        "orders",
        capacity=cfg.orders.capacity,
        refill_rate_per_sec=cfg.orders.refill_rate_per_sec,
    )
    limiter.configure_bucket(
        "rest",
        capacity=cfg.rest.capacity,
        refill_rate_per_sec=cfg.rest.refill_rate_per_sec,
    )
    limiter.configure_bucket(
        "hist",
        capacity=cfg.hist.capacity,
        refill_rate_per_sec=cfg.hist.refill_rate_per_sec,
    )
    return limiter

def get_nifty_expiry() -> str:
    """Return the current month's Nifty expiry code (e.g., 25NOV)."""
    from datetime import datetime
    now = datetime.now()
    # Get 2-digit year and upper-case short month (e.g., 25NOV)
    return now.strftime("%y%b").upper()

def get_nifty_atm_strike(nifty_spot):
    """Round to nearest 50 or 100, as in your option chain tokens."""
    return round(nifty_spot / 50) * 50

def _find_existing_nifty_option_symbol(expiry: str, strike: int, opt_type: str = "CE") -> str | None:
    """
    Return a best-effort matching tradingsymbol (without exchange prefix) present
    in the warmed instrument resolver. Checks both NFO-prefixed and unprefixed keys.
    """
    # try to locate a resolver instance on globals or import fallback
    possible_names = ("instrument_resolver", "resolver", "InstrumentResolverInstance", "instrumentResolver")
    resolver = None
    for n in possible_names:
        resolver = globals().get(n)
        if resolver:
            break
    if resolver is None:
        try:
            from nifty_scalper_bot.data import instruments as _instr_mod  # type: ignore
            resolver = getattr(_instr_mod, "instrument_resolver", None) or getattr(_instr_mod, "resolver", None)
        except Exception:
            resolver = None

    # nothing to validate against
    if resolver is None:
        return None

    want_exp = (expiry or "").upper()
    want_str = str(int(strike))
    want_ot = (opt_type or "CE").upper()

    # Candidate formats to try (without exchange and with possible variations):
    candidates_to_try = [
        f"NIFTY{want_exp}{want_str}{want_ot}",
        f"NIFTY{want_exp}{want_str:0>2}{want_ot}",
        f"NIFTY{want_exp}{want_str}{want_ot}".upper(),
    ]

    # Helper to check resolver mapping safely
    def resolver_lookup(key: str):
        try:
            # Many InstrumentResolver implementations expose lookup(symbol) or dict-like maps
            lookup_fn = getattr(resolver, "lookup", None)
            if callable(lookup_fn):
                return lookup_fn(key)
            # some use dict-like accessors with exchange prefix "NFO:"
            for attr in ("_by_symbol", "symbols", "symbol_map", "_symbol_map", "_symbol_by_token"):
                m = getattr(resolver, attr, None)
                if isinstance(m, dict) and key in m:
                    return m[key]
            # also try simple get()
            get_fn = getattr(resolver, "get", None)
            if callable(get_fn):
                return get_fn(key)
        except Exception:
            return None
        return None

    for cand in candidates_to_try:
        # try both unprefixed and exchange-prefixed forms
        for key in (cand, f"NFO:{cand}"):
            meta = resolver_lookup(key)
            if meta:
                # return the canonical trading symbol without exchange prefix
                ts = meta.get("tradingsymbol") or meta.get("symbol") or cand
                return ts
    # fallback: scan resolver keys for approximate match (safely)
    try:
        keys = None
        for attr in ("_by_symbol", "symbols", "keys"):
            m = getattr(resolver, attr, None)
            if isinstance(m, dict):
                keys = m.keys()
                break
        if keys:
            for s in keys:
                s_up = s.upper()
                if not s_up.startswith("NIFTY"):
                    continue
                if not s_up.endswith(want_ot):
                    continue
                if want_exp in s_up and want_str in s_up:
                    return s_up if not s_up.startswith("NFO:") else s_up.split(":", 1)[-1]
    except Exception:
        pass

    return None


def _get_symbols(
    config: AppConfig,
    resolver: InstrumentResolver | None = None,
    broker: Any | None = None
) -> list[str]:
    """
    Return validated option symbols for trading.
    
    PRODUCTION FIX v2.0:
    - Fixed 'contracts' undefined bug
    - Guaranteed option symbol generation
    - Robust price fetching with multiple fallbacks
    """
    import datetime
    import calendar
    from datetime import timedelta

    LOGGER.info("=" * 60)
    LOGGER.info("🔍 _get_symbols() STARTING - Symbol Resolution")

    # 1. Check for explicit configuration (highest priority)
    symbols = getattr(config, "symbols", None)
    if symbols:
        if isinstance(symbols, Iterable) and not isinstance(symbols, (str, bytes)):
            result = [str(s).strip() for s in symbols if str(s).strip()]
            LOGGER.info(f"✅ Using configured SYMBOLS env: {result}")
            LOGGER.info("=" * 60)
            return result
        result = [str(symbols)]
        LOGGER.info(f"✅ Using configured SYMBOLS env: {result}")
        LOGGER.info("=" * 60)
        return result

    final_symbols: list[str] = []
    atm_price: int | None = None
    ltp: float = 0.0

    # 2. Fetch Live Spot Price
    if broker:
        try:
            token_candidates = [256265]
            str_candidates = ["NSE:NIFTY 50", "NIFTY 50", "NIFTY 50 INDEX"]

            inner = getattr(broker, "client", getattr(broker, "_broker", broker))

            def parse_price(data: Any) -> float:
                if not data:
                    return 0.0
                if isinstance(data, (int, float)):
                    return float(data)
                if isinstance(data, dict):
                    for key in ("last_price", "ltp", "close"):
                        val = data.get(key)
                        if val:
                            try:
                                return float(val)
                            except (ValueError, TypeError):
                                continue
                return 0.0

            # Strategy A: get_ltp_bulk
            if ltp == 0 and hasattr(inner, "get_ltp_bulk"):
                try:
                    response = inner.get_ltp_bulk(token_candidates)
                    if response:
                        for t in token_candidates:
                            val = response.get(t) or response.get(str(t))
                            price = parse_price(val)
                            if price > 0:
                                ltp = price
                                LOGGER.info(f"✅ NIFTY price via get_ltp_bulk: {ltp}")
                                break
                except Exception as e:
                    LOGGER.debug(f"get_ltp_bulk failed: {e}")

            # Strategy B: get_ltp
            if ltp == 0 and hasattr(inner, "get_ltp"):
                for s in str_candidates:
                    try:
                        p = inner.get_ltp(s)
                        price = parse_price(p)
                        if price > 0:
                            ltp = price
                            LOGGER.info(f"✅ NIFTY price via get_ltp: {ltp}")
                            break
                    except Exception:
                        pass

            # Strategy C: Standard Kite .ltp()
            if ltp == 0 and hasattr(inner, "ltp"):
                try:
                    q = inner.ltp(str_candidates)
                    for k in str_candidates:
                        if k in q:
                            price = parse_price(q[k])
                            if price > 0:
                                ltp = price
                                LOGGER.info(f"✅ NIFTY price via .ltp(): {ltp}")
                                break
                except Exception as e:
                    LOGGER.debug(f".ltp() failed: {e}")

            # Strategy D: quote()
            if ltp == 0 and hasattr(inner, "quote"):
                try:
                    q = inner.quote(str_candidates)
                    for k in str_candidates:
                        if k in q:
                            price = parse_price(q[k])
                            if price > 0:
                                ltp = price
                                LOGGER.info(f"✅ NIFTY price via .quote(): {ltp}")
                                break
                except Exception as e:
                    LOGGER.debug(f".quote() failed: {e}")

            if ltp > 0:
                atm_price = round(ltp / 50) * 50
                LOGGER.info(f"✅ Live NIFTY Spot: {ltp} -> ATM Strike: {atm_price}")
                global _LATEST_CTX
                if _LATEST_CTX:
                    _LATEST_CTX.update_spot_price("NIFTY", ltp)
            else:
                LOGGER.warning("⚠️ Could not fetch live NIFTY price from broker")

        except Exception as exc:
            LOGGER.error(f"Error fetching live price: {exc}", exc_info=True)

    # 3. Fallback ATM price (Update based on current NIFTY ~25200)
    if atm_price is None or atm_price < 15000:
        fallback_base = 25200
        LOGGER.warning(f"⚠️ Using fallback ATM: {fallback_base}")
        atm_price = fallback_base

    # 4. Generate strike range
    strike_step = 50
    strikes_to_fetch = [
        atm_price - strike_step,
        atm_price,
        atm_price + strike_step,
    ]
    LOGGER.info(f"📊 Strikes to fetch: {strikes_to_fetch}")

    # 5. Try Smart Resolution (FIXED: contracts initialization)
    try:
        from nifty_scalper_bot.utils.smart_symbol import get_next_valid_symbols

        contract_map = {}
        contracts: list = []  # FIX: Initialize contracts list

        if resolver:
            if hasattr(resolver, "option_contracts"):
                try:
                    contracts = resolver.option_contracts("NIFTY") or []
                    LOGGER.info(f"📦 Got {len(contracts)} contracts from option_contracts()")
                except Exception as e:
                    LOGGER.debug(f"option_contracts() failed: {e}")
                    contracts = []

            if not contracts and hasattr(resolver, "_option_contracts"):
                try:
                    raw = getattr(resolver, "_option_contracts", {})
                    for c_list in raw.values():
                        if isinstance(c_list, list):
                            contracts.extend(c_list)
                    LOGGER.info(f"📦 Got {len(contracts)} contracts from _option_contracts")
                except Exception as e:
                    LOGGER.debug(f"_option_contracts access failed: {e}")

            for c in contracts:
                t = c.get("instrument_token")
                if t:
                    try:
                        contract_map[int(t)] = c
                    except (ValueError, TypeError):
                        pass

        if contract_map and get_next_valid_symbols:
            LOGGER.info(f"🔍 Trying smart resolution with {len(contract_map)} contracts")
            results = get_next_valid_symbols(
                strikes_to_fetch,
                opt_types=("CE", "PE"),
                instrument_map=contract_map
            )
            for inst in results:
                ts = inst.get("tradingsymbol") or inst.get("symbol")
                if ts:
                    prefix = "NFO:" if not ts.startswith("NFO:") else ""
                    final_symbols.append(f"{prefix}{ts}")

            if final_symbols:
                LOGGER.info(f"✅ Smart resolution found {len(final_symbols)} symbols: {final_symbols}")

    except Exception as exc:
        LOGGER.warning(f"Smart resolution skipped: {exc}")

    # 6. GUARANTEED Fallback - Always generate options if none found
    if not final_symbols:
        LOGGER.warning("⚠️ Smart resolution failed. Using GUARANTEED fallback.")

        now = datetime.datetime.now()
        today = now.date()

        # Find next Tuesday (weekly expiry)
        target_weekday = 1  # Tuesday
        days_ahead = target_weekday - today.weekday()
        if days_ahead < 0:
            days_ahead += 7
        if days_ahead == 0 and now.hour >= 16:
            days_ahead += 7

        next_expiry = today + timedelta(days=days_ahead)

        # Check if monthly expiry (last Tuesday of month)
        last_day_of_month = calendar.monthrange(next_expiry.year, next_expiry.month)[1]
        potential_monthly = datetime.date(next_expiry.year, next_expiry.month, last_day_of_month)
        while potential_monthly.weekday() != target_weekday:
            potential_monthly -= timedelta(days=1)

        is_monthly = (next_expiry == potential_monthly)

        # Format expiry code (Zerodha convention)
        if is_monthly:
            date_code = next_expiry.strftime("%y%b").upper()
        else:
            y = next_expiry.strftime("%y")
            d = next_expiry.strftime("%d")
            m_map = {10: "O", 11: "N", 12: "D"}
            m = m_map.get(next_expiry.month, str(next_expiry.month))
            date_code = f"{y}{m}{d}"

        LOGGER.info(f"📅 Expiry: {next_expiry} | Type: {'Monthly' if is_monthly else 'Weekly'} | Code: {date_code}")

        for strike in strikes_to_fetch:
            for kind in ("CE", "PE"):
                sym = f"NFO:NIFTY{date_code}{strike}{kind}"
                final_symbols.append(sym)

        LOGGER.info(f"✅ Generated fallback symbols: {final_symbols}")

    LOGGER.info(f"🎯 FINAL SYMBOLS TO TRADE: {final_symbols}")
    LOGGER.info("=" * 60)
    return final_symbols

def _get_strategy_config(config: AppConfig) -> StrategyRunnerConfig:
    """Build strategy runner configuration with environment overrides.
    
    Environment Variables:
        MIN_INDICATOR_BARS: Number of bars required before signal generation (default: 10)
        SIGNAL_COOLDOWN_SECONDS: Cooldown between signals (default: 3.0)
        TRADE_COOLDOWN_SECONDS: Cooldown between trades (default: 10.0)
    """
    cfg = getattr(config, "strategy_config", None)
    if isinstance(cfg, StrategyRunnerConfig):
        return cfg
    
    # Allow environment override for warmup bars (CRITICAL for faster startup)
    warmup_bars_default = 10  # Changed from 50 to 10 for faster signal generation
    warmup_bars = int(os.getenv("MIN_INDICATOR_BARS", str(warmup_bars_default)))
    
    signal_cooldown = float(os.getenv("SIGNAL_COOLDOWN_SECONDS", "3.0"))
    trade_cooldown = float(os.getenv("TRADE_COOLDOWN_SECONDS", "10.0"))
    
    LOGGER.info(
        f"Strategy config: warmup_bars={warmup_bars}, signal_cooldown={signal_cooldown}s, "
        f"trade_cooldown={trade_cooldown}s"
    )
    
    return StrategyRunnerConfig(
        signal_cooldown_seconds=float(
            getattr(cfg, "signal_cooldown_seconds", signal_cooldown) or signal_cooldown
        ),
        trade_cooldown_seconds=float(
            getattr(cfg, "trade_cooldown_seconds", trade_cooldown) or trade_cooldown
        ),
        min_indicator_bars=int(
            getattr(cfg, "min_indicator_bars", warmup_bars) or warmup_bars
        ),
        max_trade_history=int(getattr(cfg, "max_trade_history", 100) or 100),
    )


def _bind_ws_mdm(ctx: BotContext) -> None:
    """Wire WebSocket connectivity events into the market data manager."""
    ws = getattr(ctx, "websocket_manager", None)
    mdm = getattr(ctx, "market_data_manager", None)
    if ws is None or mdm is None:
        return
    
    def _on_connect() -> None:
        try:
            mdm.set_ws_connected(True)
            mdm.bump_heartbeat()
        except Exception as exc:
            LOGGER.warning(
                f"Failed to set WS connected state: {exc}",
                extra={"event": "ws_mdm_connect_failed"}
            )
    
    def _on_disconnect() -> None:
        try:
            mdm.set_ws_connected(False)
        except Exception as exc:
            LOGGER.warning(
                f"Failed to set WS disconnected state: {exc}",
                extra={"event": "ws_mdm_disconnect_failed"}
            )
    
    try:
        ws.set_callbacks(on_connect=_on_connect, on_disconnect=_on_disconnect)
    except Exception as exc:
        LOGGER.error(
            f"Failed to bind WS callbacks: {exc}",
            extra={"event": "ws_mdm_bind_failed"},
            exc_info=True
        )



async def reconcile_positions_on_startup(
    broker_client: Any,
    position_manager: Any,
    order_manager: Any,
    logger: Any,
) -> None:
    """Reconcile local positions with broker state on startup.

    Args:
        broker_client: Broker API client used to fetch live positions.
        position_manager: Local position manager maintaining state.
        order_manager: Order manager reference for diagnostic context.
        logger: Structured logger used for observability.

    Returns:
        None.

    Raises:
        None. Exceptions are logged and re-raised for upstream handling.
    """

    logger.debug(
        "Entered reconcile_positions_on_startup",
        extra={
            "event": "reconcile.start.enter",
            "order_manager": getattr(
                order_manager, "__class__", type(order_manager)
            ).__name__,
        },
    )
    logger.info(
        "Starting position reconciliation",
        extra={"event": "reconcile.start"},
    )

    try:
        broker_snapshot: list[Mapping[str, Any]] = []
        raw_positions = broker_client.get_positions()
        for entry in raw_positions:
            if isinstance(entry, Mapping):
                broker_snapshot.append(entry)

        local_positions = position_manager.get_all_positions()

        def _normalize_symbol(payload: Mapping[str, Any]) -> str:
            raw_symbol = (
                payload.get("tradingsymbol")
                or payload.get("symbol")
                or payload.get("instrument")
                or ""
            )
            symbol = str(raw_symbol).strip().upper()
            if ":" in symbol:
                symbol = symbol.split(":", maxsplit=1)[-1].strip().upper()
            return symbol

        def _extract_quantity(payload: Mapping[str, Any]) -> int:
            """Return the integer quantity from a broker payload.

            Args:
                payload: Raw broker position payload.

            Returns:
                Normalised signed quantity as an integer.

            Raises:
                None.
            """

            quantity_candidate = (
                payload.get("net_qty")
                or payload.get("net_quantity")
                or payload.get("netQuantity")
                or payload.get("quantity")
                or payload.get("net")
            )
            if quantity_candidate is None:
                return 0
            try:
                numeric_quantity = float(quantity_candidate)
            except (TypeError, ValueError):
                return 0
            return int(numeric_quantity)

        def _extract_average_price(payload: Mapping[str, Any]) -> float:
            """Return the average price from a broker payload.

            Args:
                payload: Raw broker position payload.

            Returns:
                Average price when available, otherwise ``0.0``.

            Raises:
                None.
            """

            price_candidate = (
                payload.get("average_price")
                or payload.get("avg_price")
                or payload.get("price")
                or payload.get("buy_price")
                or payload.get("sell_price")
            )
            if price_candidate is None:
                return 0.0
            try:
                return float(price_candidate)
            except (TypeError, ValueError):
                return 0.0

        broker_symbols: dict[str, dict[str, Any]] = {}
        for payload in broker_snapshot:
            symbol = _normalize_symbol(payload)
            if not symbol:
                continue
            quantity = _extract_quantity(payload)
            if quantity == 0:
                continue
            broker_symbols[symbol] = {
                "quantity": quantity,
                "average_price": _extract_average_price(payload),
                "raw": payload,
            }

        local_symbols = {pos.symbol: pos for pos in local_positions}
        orphaned = set(broker_symbols) - set(local_symbols)

        if orphaned:
            logger.warning(
                "Found orphaned positions in broker",
                extra={
                    "event": "reconcile.orphaned",
                    "symbols": sorted(orphaned),
                    "count": len(orphaned),
                },
            )
            for symbol in sorted(orphaned):
                broker_position = broker_symbols[symbol]
                logger.info(
                    "Imported orphaned position",
                    extra={
                        "event": "reconcile.import",
                        "symbol": symbol,
                        "quantity": broker_position["quantity"],
                    },
                )

        mismatch_symbols: list[str] = []
        for symbol, local_position in local_symbols.items():
            broker_pos: dict[str, Any] | None = broker_symbols.get(symbol)
            if broker_pos is None:
                continue
            broker_qty_raw = int(broker_pos.get("quantity", 0))
            broker_qty = abs(broker_qty_raw)
            broker_side = "LONG" if broker_qty_raw > 0 else "SHORT"
            local_qty = int(getattr(local_position, "quantity", 0))
            local_side = str(getattr(local_position, "side", "LONG")).upper()
            if broker_qty != local_qty or broker_side != local_side:
                mismatch_symbols.append(symbol)
                logger.warning(
                    "Position quantity mismatch",
                    extra={
                        "event": "reconcile.mismatch",
                        "symbol": symbol,
                        "broker_qty": broker_qty,
                        "broker_side": broker_side,
                        "local_qty": local_qty,
                        "local_side": local_side,
                    },
                )

        if broker_snapshot:
            position_manager.synchronize_with_broker(broker_snapshot)

        logger.info(
            "Reconciliation complete",
            extra={
                "event": "reconcile.complete",
                "orphaned_count": len(orphaned),
                "mismatch_count": len(mismatch_symbols),
            },
        )

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Position reconciliation failed",
            extra={"event": "reconcile.failed", "error": str(exc)},
            exc_info=exc,
        )
        raise


def parse_nifty_option_symbol(symbol: str) -> dict | None:
    """
    Parse NIFTY option symbol to extract strike, expiry, and option type.
    """
    import re
    from datetime import datetime, timedelta, timezone # Ensure timezone is imported
    import calendar
    
    symbol = symbol.replace("NFO:", "").strip()
    
    # Monthly/Far Weekly Pattern: NIFTY25NOV25950CE
    monthly_match = re.match(r'NIFTY(\d{2})([A-Z]{3})(\d+)(CE|PE)', symbol)
    if monthly_match:
        year, month_str, strike, opt_type = monthly_match.groups()
        month_names = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
        month = month_names.get(month_str)
        if month:
            year_full = 2000 + int(year)
            
            # Find last Thursday of the month (Standard Monthly Expiry Logic)
            last_day = calendar.monthrange(year_full, month)[1]
            expiry = datetime(year_full, month, last_day)
            while expiry.weekday() != 3: # Thursday is 3
                expiry = expiry - timedelta(days=1)
            
            # Use total_seconds for float days_to_expiry
            days_to_expiry = (expiry - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds() / 86400.0
            
            return {
                "strike": int(strike),
                "expiry": expiry,
                "days_to_expiry": max(days_to_expiry, 0.001),
                "option_type": opt_type,
                "symbol_type": "Monthly"
            }
    
    # Simple pattern match for weekly/other (can be expanded)
    weekly_match = re.match(r'NIFTY(\d{2})([A-Z])(\d{2})(\d+)(CE|PE)', symbol)
    if weekly_match:
        # Placeholder logic, needs proper date mapping for weeks
        return {"strike": int(weekly_match.groups()[3]), "expiry": datetime.now(), "days_to_expiry": 3.0, "option_type": weekly_match.groups()[4], "symbol_type": "Weekly (Approx)"}

    return None

def calculate_greeks_simple(
    spot: float,
    strike: float,
    days_to_expiry: float,
    option_type: str,
    volatility: float = 0.20, # 20% IV assumption
) -> dict:
    """
    Simple Greeks approximation (using Black-Scholes principles).
    """
    import math
    
    if days_to_expiry <= 0.0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "days_to_expiry": 0.0, "moneyness": spot/strike}
    
    t = days_to_expiry / 365.25 # Time in years
    moneyness = spot / strike
    
    # Simple delta approximation (ATMs are 0.5/-0.5)
    delta = 0.5
    if option_type.upper() in ["CE", "CALL"]:
        delta = 0.5 + min(0.49, max(0, moneyness - 1.0)) 
    else:  # PUT
        delta = -0.5 - min(0.49, max(0, 1.0 - moneyness)) 
    
    # Gamma (peaks at ATM)
    gamma = 0.01 / (abs(moneyness - 1) + 0.01) * math.sqrt(1 / max(t, 0.01))
    gamma = min(gamma, 0.05)
    
    # Theta (time decay)
    theta_base = spot * volatility / (2 * math.sqrt(max(t, 0.01))) / 365.25
    theta = -1.0 * theta_base
    
    # Vega (volatility sensitivity)
    vega = spot * math.sqrt(max(t, 0.01)) * 0.01
    
    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 2),
        "vega": round(vega, 2),
        "days_to_expiry": round(days_to_expiry, 1),
        "moneyness": round(moneyness, 3),
    }

def _setup_telegram(ctx: BotContext) -> None:
    """Wire the Telegram controller with full access to bot components."""
    settings = ctx.settings.notifications
    
    # 1. Extract Credentials
    bot_token = settings.token
    # Handle Set[int] -> Single ID conversion safely
    chat_id = next(iter(settings.whitelist_chat_ids)) if settings.whitelist_chat_ids else None

    # 2. Validation with Explicit Logging
    if not bot_token or not chat_id:
        LOGGER.warning(
            f"⚠️ Telegram DISABLED: Missing Credentials. Token={'OK' if bot_token else 'MISSING'}, ChatID={'OK' if chat_id else 'MISSING'}"
        )
        return

    try:
        from nifty_scalper_bot.notifications.telegram_controller import (
            TelegramBot,
            TelegramDeps,
        )

        # 3. Initialize with extracted values
        deps = TelegramDeps(
            token=str(bot_token),  # Ensure string format
            chat_id=str(chat_id),  # Ensure string format
            app_version="1.0.0",
            risk_manager=ctx.risk_manager,
            order_manager=ctx.order_manager,
            position_manager=ctx.position_manager,
            strategy_runner=ctx.strategy_runner,
            market_data_manager=ctx.market_data_manager,
            unified_manager=ctx.unified_manager,
            stream_supervisor=getattr(ctx, "stream_supervisor", None),
            data_hub=ctx.data_hub,
            instrument_resolver=getattr(ctx, "instrument_resolver", None),
            enable_polling_fallback=True,
        )

        ctx.telegram_bot = TelegramBot(deps)
        LOGGER.info("✅ Telegram Controller wired successfully.")

    except Exception as e:
        LOGGER.error(f"❌ Telegram setup failed: {e}", exc_info=True)
        
def initialize_components(settings: Settings | None = None) -> BotContext:
    """Initialize all components in correct order."""
    
    import threading

    ensure_multiproc_dir(clear_stale=True)
    settings = settings or get_settings()
    config = settings.app
    raw_ws_disabled = os.getenv("WEBSOCKET__DISABLED")
    if raw_ws_disabled is None:
        raw_ws_disabled = os.getenv("WEBSOCKET_DISABLED")
    websocket_disabled_env = str(raw_ws_disabled or "false").strip().lower() == "true"

    raw_webhook_env = os.getenv("TELEGRAM__WEBHOOK_ENABLED")
    if raw_webhook_env is None:
        raw_webhook_env = os.getenv("TELEGRAM_WEBHOOK_ENABLED")
    telegram_webhook_env_enabled = (
        str(raw_webhook_env or "false").strip().lower() == "true"
    )

    notif_settings = settings.notifications
    if not telegram_webhook_env_enabled and notif_settings.webhook_enabled:
        notif_settings.webhook_enabled = False
    ws_host = urlsplit(str(config.broker.websocket_url)).hostname or ""

    rate_limiter = _configure_rate_limiter(config.ratelimit)
    message_bus = MessageBus()

    from nifty_scalper_bot.data.robust_provider import (
    RobustDataProvider,
    CircuitBreakerConfig
    )

    broker_client = ZerodhaKiteClient(
        api_key=config.broker.api_key,
        api_secret=config.broker.api_secret,
        access_token=config.broker.access_token,
    )

    robust_provider = RobustDataProvider(
        broker_client=broker_client,
        circuit_config=CircuitBreakerConfig(
            failure_threshold=5,
            timeout_seconds=60.0
        ),
        notifier=lambda event, data: asyncio.create_task(
            notifier.send_event(event, data) if notifier else asyncio.sleep(0)
        )
    )
    
    
    
    broker_client.preload_instruments()

    instrument_resolver = InstrumentResolver(broker_client)
    cache_settings = getattr(settings, "instruments", None)
    warm_result = _try_warm_instruments(settings, instrument_resolver, LOGGER)
    instrument_state = warm_result.status
    instrument_conn = warm_result.connection
    instrument_tokens = instrument_state.tokens or warm_result.tokens
    instrument_options = instrument_state.options or warm_result.options
    instrument_source = instrument_state.last_source or warm_result.source
    csv_hint = warm_result.csv_path or warm_result.db_path

    METRICS.record_resolver_tokens(
        source=instrument_source,
        count=instrument_tokens,
    )
    LOGGER.info(
        "resolver_warm_loaded tokens=%s options=%s source=%s",
        instrument_tokens,
        instrument_options,
        instrument_source,
        extra={
            "event": "resolver_warm_loaded",
            "tokens": instrument_tokens,
            "options": instrument_options,
            "source": instrument_source,
            "csv_path": csv_hint,
        },
    )
    ensure_fn = getattr(instrument_resolver, "ensure_core_index_tokens", None)
    if callable(ensure_fn):
        ensure_fn()

    margin_segment_env = os.getenv("BROKER_MARGIN_SEGMENT", "equity") or "equity"
    margin_segment = margin_segment_env.strip().lower()
    if margin_segment not in {"equity", "commodity"}:
        margin_segment = "equity"
    try:
        margin_summary = broker_client.get_margin_summary(segment=margin_segment)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "broker_margin_summary_failed",
            extra={
                "event": "broker_margin_summary_failed",
                "segment": margin_segment,
                "error": str(exc),
            },
            exc_info=exc,
        )
    else:
        LOGGER.info(
            "broker_margin_summary_loaded",
            extra={
                "event": "broker_margin_summary_loaded",
                "segment": margin_segment,
                "available": margin_summary.get("available"),
                "used": margin_summary.get("used"),
                "net": margin_summary.get("net"),
            },
        )

    websocket_enabled = bool(getattr(settings, "websocket_enabled", True))
    if websocket_disabled_env:
        websocket_enabled = False
    stream_mode_raw = coalesce_str("STREAM__MODE", "STREAMING_MODE", default="poll")
    streaming_mode = stream_mode_raw.strip().lower() or "poll"
    poll_enabled = coalesce_bool("POLLING__ENABLED", default=True)
    poll_interval_sec = coalesce_float("POLLING__INTERVAL_SEC", default=0.0)
    if poll_interval_sec > 10.0:
        poll_interval_sec = poll_interval_sec / 1000.0
    poll_interval_ms_fallback = coalesce_int(
        "POLL_INTERVAL_MS",
        "MICRO_QUOTE_POLL_MS",
        default=700,
    )
    if poll_interval_sec <= 0.0:
        poll_interval_sec = float(poll_interval_ms_fallback) / 1000.0
    poll_interval_sec = max(0.2, float(poll_interval_sec))
    poll_interval_ms = int(poll_interval_sec * 1000.0)
    poll_batch_size = max(
        1,
        coalesce_int("POLLING__BATCH_SIZE", "POLL_BATCH_SIZE", default=50),
    )
    poll_require_depth = coalesce_bool(
        "POLLING__REQUIRE_DEPTH",
        "POLL_REQUIRE_DEPTH",
        "EXECUTOR__REQUIRE_DEPTH",
        default=False,
    )
    poll_warn_rate_limit = coalesce_bool(
        "POLLING__WARN_ON_RATE_LIMIT",
        "POLL_WARN_RATE_LIMIT",
        default=True,
    )
    # Normalize symbols (quotes tolerated) and default fallback
    raw_syms = get_csv("POLLING__SYMBOLS")
    if raw_syms:
        poll_symbols = [s.strip().upper() for s in raw_syms if s.strip()]
    else:
        poll_symbols = ["NSE:NIFTY 50", "256265"]

    attach_resolver = getattr(broker_client, "attach_resolver", None)
    if callable(attach_resolver):
        with suppress(Exception):
            attach_resolver(instrument_resolver)

    websocket_client: Any | None = None
    websocket_manager: WebSocketManager | None = None
    streamer: Any
    stream_supervisor: StreamSupervisor | None = None
    data_hub: DataHub | None = None
    hub_store: HubStore | None = None
    try:
        hub_store = HubStore()
    except Exception:  # pragma: no cover - defensive fallback
        LOGGER.exception("hub_store_init_failed")
        hub_store = None

    def _resolve_ws_token() -> str:
        return ""

    def _refresh_ws_session() -> None:  # pragma: no cover - polling default
        return None

    def _ws_token_issued_at() -> float | None:
        return None

    use_polling = (not websocket_enabled) or streaming_mode in {"polling", "poll", ""}
    # [FIX] Container for direct wiring
    strategy_runner_ref: dict[str, Any] = {}
    
    # [FIX 1/2] Container to hold strategy runner reference for direct tick injection
    strategy_runner_ref: dict[str, Any] = {} 

    if not poll_enabled and not websocket_enabled:
        raise ConfigurationError(
            "Polling disabled while websocket transport is disabled; "
            "no streamer available"
        )
    if not poll_enabled and use_polling:
        use_polling = False  # explicit disable beats default
    market_data_mode = "polling" if use_polling else "websocket"
    LOGGER.info(
        "Market data streamer starting in %s mode",
        market_data_mode,
        extra={
            "event": "market_data_mode",
            "mode": market_data_mode,
            "websocket_disabled_env": websocket_disabled_env,
            "streaming_mode": streaming_mode,
        },
    )
    if use_polling:
        market_data_manager = MarketDataManager(
            broker_client,
            None,
            resolver=instrument_resolver,
        )
        
        # [FIX] Start Health Monitor (Watchdog) for Polling Mode
        # This prevents "Zombie Mode" by killing the process if data stops for 3 mins
        def _monitor_data_health():
            import logging, time, os, threading
            logger = logging.getLogger("nifty_scalper_bot.watchdog")
            logger.info("✅ Data Health Monitor Started (Polling Mode)")
            
            while True:
                time.sleep(60) # Check every minute
                
                # Check if data is flowing
                last_tick = getattr(market_data_manager, "last_tick_time", 0)
                
                # If we have received data before (>0) but it's now stale (>180s)
                if last_tick > 0 and (time.time() - last_tick > 180): 
                    logger.critical(f"🚨 FATAL: No data for {int(time.time() - last_tick)}s. Zombie Mode detected. Exiting.")
                    os._exit(1) # Kill process -> Railway auto-restarts -> Connection restored
                    
        health_thread = threading.Thread(target=_monitor_data_health, daemon=True)
        health_thread.start()
        try:
            start_watchdog(market_data_manager)
            LOGGER.info("✅ Data Health Monitor (Watchdog) started")
        except Exception as exc:
            LOGGER.warning(f"Failed to start watchdog: {exc}")
            
        data_hub = DataHub(
            market_data_manager,
            instrument_resolver,
            options_only=True,
            store=hub_store,
            message_bus=message_bus,
        )
        # Explicitly mark WS disconnected in polling mode so health reflects polling
        market_data_manager.set_ws_connected(False)

    # [FIX] Fan-out: every polled tick updates MDM and the supervisor heartbeat
    def _on_poll_tick(tick: dict[str, Any]) -> None:
        """
        Handle incoming poll tick with Robust Validation & Recovery.
        """
        if not tick or not isinstance(tick, dict):
            return

        # 1. Normalize Tick
        t = dict(tick) if isinstance(tick, dict) else {"raw": tick}
        t.setdefault("source", "polling")

        # 2. Token Normalization & Validation
        if "instrument_token" not in t and "token" in t:
            raw_token = t["token"]
            try:
                t["instrument_token"] = int(float(raw_token))
            except (ValueError, TypeError):
                # LOGGER.warning("Invalid token: %s", raw_token)
                pass

        # 3. Symbol Mapping (If token exists but symbol missing)
        # 3. Symbol Mapping (If token exists but symbol missing)
        token_value = t.get("instrument_token")
        if token_value and not t.get("symbol"):
            try:
                mapped = None
                token_int = int(token_value) if token_value else None
                
                # Try instrument_resolver FIRST (most reliable)
                if not mapped and instrument_resolver:
                    try:
                        # Method 1: format_token_as_symbol
                        if hasattr(instrument_resolver, 'format_token_as_symbol'):
                            mapped = instrument_resolver.format_token_as_symbol(token_int)
                        
                        # Method 2: get_instrument_by_token
                        if not mapped and hasattr(instrument_resolver, 'get_instrument_by_token'):
                            inst = instrument_resolver.get_instrument_by_token(token_int)
                            if inst:
                                exchange = getattr(inst, 'exchange', 'NFO')
                                sym = getattr(inst, 'tradingsymbol', None) or getattr(inst, 'symbol', None)
                                if sym:
                                    mapped = f"{exchange}:{sym}"
                        
                        # Method 3: lookup
                        if not mapped and hasattr(instrument_resolver, 'lookup'):
                            info = instrument_resolver.lookup(token_int)
                            if info:
                                exchange = info.get('exchange', 'NFO')
                                sym = info.get('tradingsymbol') or info.get('symbol')
                                if sym:
                                    mapped = f"{exchange}:{sym}"
                    except Exception as e:
                        LOGGER.debug(f"Resolver lookup failed for {token_int}: {e}")
                
                # Fallback to market_data_manager
                if not mapped and market_data_manager:
                    mapped = market_data_manager._symbol_by_token.get(token_int)
                
                if mapped:
                    t["symbol"] = mapped
                    LOGGER.debug(f"✅ Symbol resolved: {token_int} -> {mapped}")
                else:
                    # Log unmapped token (throttled)
                    log_throttled(
                        LOGGER,
                        f"unmapped_token_{token_value}",
                        f"⚠️ UNMAPPED TOKEN: {token_value} - all resolution methods failed",
                        interval_sec=60.0
                    )
            except Exception as e:
                LOGGER.debug(f"Symbol mapping error for token {token_value}: {e}")
                
                if not mapped and instrument_resolver:
                    inst = instrument_resolver.get_instrument_by_token(int(token_value))
                    if inst:
                        exchange = getattr(inst, 'exchange', 'NFO')
                        tradingsymbol = getattr(inst, 'tradingsymbol', None)
                        if tradingsymbol:
                            mapped = f"{exchange}:{tradingsymbol}"
                
                if mapped:
                    t["symbol"] = mapped
                    # ✅ DIAGNOSTIC: Log successful mapping
                    log_throttled(
                        LOGGER,
                        f"symbol_mapped_{token_value}",
                        f"✅ Mapped token {token_value} -> {mapped}",
                        interval_sec=120.0
                    )
                else:
                    # ✅ DIAGNOSTIC: Log failed mapping
                    log_throttled(
                        LOGGER,
                        f"symbol_unmapped_{token_value}",
                        f"⚠️ UNMAPPED TOKEN: {token_value}",
                        interval_sec=60.0
                    )
            except Exception as e:
                LOGGER.debug(f"Symbol mapping error: {e}")

        # ✅ DIAGNOSTIC: Log tick reception at INFO level
        sym = t.get("symbol")
        ltp = t.get("ltp") or t.get("last_price")
        if sym:
            log_throttled(
                LOGGER,
                f"tick_received_{sym}",
                f"📡 TICK: {sym} | LTP: {ltp}",
                interval_sec=30.0
            )

        # 4. LTP Normalization
        if "ltp" not in t:
            t["ltp"] = t.get("last_price") or t.get("close")

        # 5. Inject Timestamp
        if "timestamp" not in t:
            t["timestamp"] = datetime.now(timezone.utc).timestamp()

        # 6. CRITICAL: Direct Strategy Feed (Bypass DataHub if needed)
        # 6. CRITICAL: Direct Strategy Feed
        sym = t.get("symbol")
        ltp = t.get("ltp") or t.get("last_price")
        
        # Diagnostic log (throttled every 30s per symbol)
        if sym and ltp:
            log_throttled(
                LOGGER,
                f"poll_tick_{sym}",
                f"📡 POLL TICK: {sym} | LTP: {ltp}",
                interval_sec=30.0
            )
        
        if strategy_runner_ref and sym:
            runner = strategy_runner_ref.get("instance")
            if runner:
                try:
                    # Prefer _on_tick_safe if available
                    if hasattr(runner, "_on_tick_safe") and callable(runner._on_tick_safe):
                        runner._on_tick_safe(t)
                    elif hasattr(runner, "_on_tick") and callable(runner._on_tick):
                        runner._on_tick(sym, t)
                    
                    # Diagnostic log (throttled every 60s per symbol)
                    log_throttled(
                        LOGGER,
                        f"tick_to_strategy_{sym}",
                        f"✅ TICK -> StrategyRunner: {sym}",
                        interval_sec=60.0
                    )
                except Exception as e:
                    LOGGER.error(f"❌ Strategy handler error for {sym}: {e}", exc_info=True)
        elif not sym:
            log_throttled(
                LOGGER,
                f"no_symbol_skip_{token_value}",
                f"⚠️ SKIPPING tick (no symbol): token={token_value}",
                interval_sec=60.0
            )
                    except Exception:
                        pass # Don't let strategy errors kill the poller

        # 7. DataHub Ingestion (Async)
        if data_hub is not None:
            try:
                # Fire and forget ingest
                loop = asyncio.get_running_loop()
                loop.create_task(data_hub.ingest_tick(t))
            except (RuntimeError, Exception):
                pass 
                
        # ✅ 8. CRITICAL: Feed BracketManager (Virtual Execution)
        _bm_ref = getattr(ctx.order_manager, "_bracket_manager", None) if ctx.order_manager else None
        
        if _bm_ref is not None:
            _sym = t.get("symbol")
            _ltp = t.get("ltp") or t.get("last_price")
            
            if _sym and _ltp:
                try:
                    # Fire the "Sniper" Logic
                    # This now triggers the AdaptiveTrailingController internally
                    _bm_ref.on_tick(str(_sym), float(_ltp))
                except Exception:
                    pass
                    
        # 9. Market Data Manager Processing (With Recovery)
        try:
            # HEARTBEAT FIX: Update timestamp so watchdog is happy
            if hasattr(market_data_manager, "last_tick_time"):
                market_data_manager.last_tick_time = time_module.time()

            market_data_manager._handle_tick(t)
            
            # CIRCUIT BREAKER RESET
            if hasattr(streamer, '_consecutive_errors'):
                streamer._consecutive_errors = 0

        except Exception as exc:
            # Track errors for circuit breaker
            if not hasattr(streamer, '_consecutive_errors'):
                streamer._consecutive_errors = 0
            streamer._consecutive_errors += 1
            
            # If > 10 failures, log error
            if streamer._consecutive_errors > 10:
                 LOGGER.error(f"MDM Tick Failed (x{streamer._consecutive_errors}): {exc}")
        
        finally:
            if stream_supervisor is not None:
                stream_supervisor.on_tick(t)

    # ------------------------------------------------------------------
    # Streamer Selection Logic (Polling vs WebSocket)
    # ------------------------------------------------------------------
    use_websockets = os.getenv("USE_WEBSOCKETS", "false").lower() == "true"
    if use_websockets:
        LOGGER.info("Initializing WebSocket Streamer...")
        
        def _sanitize_ws_token(value: str | None) -> str:
            token = (value or "").strip()
            if ":" in token:
                token = token.split(":", 1)[-1].strip()
            return token

        initial_token = _sanitize_ws_token(cast(str | None, config.broker.access_token))
        _ws_token_state: dict[str, str] = {"token": initial_token}
        _ws_token_timestamp: dict[str, float] = {
            "ts": time_module.time() if initial_token else 0.0,
        }

        def _resolve_ws_token() -> str:  # type: ignore[redefined-outer-name]
            candidates = [
                os.getenv("ZERODHA_ACCESS_TOKEN"),
                cast(str | None, getattr(config.broker, "access_token", None)),
                _ws_token_state.get("token", ""),
            ]
            for candidate in candidates:
                sanitized = _sanitize_ws_token(candidate)
                if sanitized:
                    previous = _ws_token_state.get("token")
                    _ws_token_state["token"] = sanitized
                    if (
                        sanitized != previous
                        or float(_ws_token_timestamp.get("ts", 0.0)) <= 0.0
                    ):
                        _ws_token_timestamp["ts"] = time_module.time()
                    return sanitized
            return _ws_token_state.get("token", "")

        def _refresh_ws_session() -> None:  # type: ignore[redefined-outer-name]
            _resolve_ws_token()

        def _build_ws():
            token = _resolve_ws_token()
            return build_kite_ticker(
                config.broker.api_key,
                token,
                session_refresher=_refresh_ws_session,
            )

        websocket_client = _build_ws()
        websocket_manager = WebSocketManager(
            websocket_client,
            on_tick=lambda tick: None,
            on_error=lambda err: LOGGER.error("WebSocket manager error: %s", err),
            backoff_min_sec=1.0,
            backoff_max_sec=30.0,
        )
        websocket_manager.set_client_factory(_build_ws)

        # Initialize WebSocket Streamer
        from nifty_scalper_bot.streaming.websocket_streamer import WebSocketStreamer
        streamer = WebSocketStreamer(
            api_key=config.broker.api_key,
            access_token_provider=_resolve_ws_token,
            on_tick=lambda tick: None, # Handled via managers below
            subscriptions=set(),
            reconnect_interval=5.0,
        )
        
        # Wire up WebSocket handlers
        market_data_manager = MarketDataManager(
            broker_client,
            websocket_manager,
            settings=settings,
            resolver=instrument_resolver,
        )
        data_hub = DataHub(
            market_data_manager,
            instrument_resolver,
            options_only=True,
            store=hub_store,
            message_bus=message_bus,
        )
        
        # Register Callbacks
        websocket_manager.on_tick = market_data_manager._handle_tick
        streamer.register_handler(market_data_manager._handle_tick) # Redundant but safe
        streamer.register_handler(lambda tick: asyncio.create_task(data_hub.ingest_tick(tick)))

    else:
        LOGGER.info("Initializing Polling Streamer...")
        
        # ✅ FIX: Initialize Managers for Polling Mode FIRST
        # Note: WebSocketManager is None in polling mode
        market_data_manager = MarketDataManager(
            broker_client,
            None, 
            settings=settings,
            resolver=instrument_resolver,
        )
        
        # ✅ FIX: Create DataHub BEFORE PollingStreamer so it's not None
        data_hub = DataHub(
            market_data_manager,
            instrument_resolver,
            options_only=True,
            store=hub_store,
            message_bus=message_bus,
        )
        LOGGER.info(f"✅ DataHub created: {data_hub is not None}")
        
        # ✅ FIX: NOW Initialize Polling Streamer with valid data_hub
        streamer = PollingStreamer(
            broker_client=broker_client,
            on_tick=_on_poll_tick,
            instrument_resolver=instrument_resolver,
            data_hub=data_hub,  # data_hub is now valid!
            poll_interval_ms=int(poll_interval_sec * 1000),
            batch_size=poll_batch_size,
            require_depth=poll_require_depth,
            warn_on_rate_limit=poll_warn_rate_limit,
        )
        LOGGER.info(f"✅ PollingStreamer initialized with DataHub wired")

    # ------------------------------------------------------------------
    # Common Supervisor & Wiring Logic
    # ------------------------------------------------------------------
    if data_hub is None:
        raise ConfigurationError("Data hub initialisation failed")

    LOGGER.info("DataHub initialized. Snapshot deferred to startup sequence.")

    # Initialize Supervisor
    stream_supervisor = StreamSupervisor(
        streamer=streamer,
        resolver=instrument_resolver,
        default_symbols=list(poll_symbols or ["NSE:NIFTY 50"]),
        autostart=True,
        monitor_interval_s=300.0,
    )
    stream_supervisor.bootstrap()
    stream_supervisor.ensure_started()

    # Initialize Indicators & Regime
    indicator_engine = IndicatorEngine()
    for env_key, env_default in (
        ("REGIME_MIN_CONFIDENCE", "0.40"),
        ("REGIME_STALE_AFTER_SEC", "300"),
        ("REGIME_BLOCK_EVENT", "0.80"),
        ("REGIME_BLOCK_VOLATILE", "0.95"),
        ("REGIME_FAIL_CLOSED", "0"),
        ("STRATEGY_ENFORCE_BLOCKLIST", "0"),
    ):
        os.environ.setdefault(env_key, env_default)

    regime_symbol = "NIFTY"
    if poll_symbols:
        regime_symbol = poll_symbols[0]

    market_regime_detector = MarketRegimeDetector()
    market_regime_manager = MarketRegimeManager(
        market_regime_detector,
        datahub=data_hub,
        indicators=indicator_engine,
        regime_settings={
            "symbol": regime_symbol,
            "update_interval_sec": 60,
            "atr_trend_threshold": 1.5,
            "vol_threshold": 25.0,
            "history_length": 1440,
        },
    )

    # Wire Regime Detector to Stream
    try:
        if hasattr(data_hub, "subscribe_ticks") and hasattr(market_regime_detector, "ingest_tick"):
            callback = cast(
                Callable[[dict[str, Any]], None],
                market_regime_detector.ingest_tick,
            )
            data_hub.subscribe_ticks(regime_symbol, callback)
            LOGGER.info(f"Regime detector subscribed via DataHub to {regime_symbol}")
            
        elif hasattr(streamer, "register_handler") and hasattr(market_regime_detector, "ingest_tick"):
            streamer.register_handler(market_regime_detector.ingest_tick)
            LOGGER.info(f"Regime detector subscribed via Streamer to {regime_symbol}")
            
    except Exception as exc:
        LOGGER.error(
            "Regime detector tick subscription failed",
            extra={"event": "regime_subscription_failed", "symbol": regime_symbol},
            exc_info=exc,
        )
    data_dir = os.getenv("DATA_DIR", "data")
    persistent_state = PersistentStateManager(base_path=Path(data_dir))

    heartbeat_interval = max(
        1.0,
        coalesce_float(
            "PERSISTENCE__HEARTBEAT_FLUSH_SEC",
            "PERSISTENCE_HEARTBEAT_FLUSH_SEC",
            default=5.0,
        ),
    )
    heartbeat_flusher = PersistentHeartbeatFlusher(
        persistent_state, interval_sec=heartbeat_interval
    )
    market_data_manager.register_heartbeat_callback(heartbeat_flusher.handle_heartbeat)

    position_state_path = Path(data_dir) / "positions.json"
    position_manager = PositionManager(state_file=str(position_state_path))
    position_manager.attach_persistent_state(persistent_state)
    position_manager.restore_positions(persistent_state.load_positions())
    data_hub.replace_positions(
        {
            "symbol": pos.symbol,
            "quantity": pos.quantity if pos.side == "LONG" else -pos.quantity,
            "average_price": pos.entry_price,
        }
        for pos in position_manager.get_open_positions()
    )

    broker_sync_attempts = max(
        1,
        coalesce_int(
            "BROKER_SYNC_MAX_ATTEMPTS",
            "BROKER_SYNC_RETRY_ATTEMPTS",
            default=5,
        ),
    )
    broker_sync_backoff_min = max(
        0.25,
        coalesce_float(
            "BROKER_SYNC_BACKOFF_MIN_SEC",
            default=1.0,
        ),
    )
    broker_sync_backoff_max = max(
        broker_sync_backoff_min,
        coalesce_float(
            "BROKER_SYNC_BACKOFF_MAX_SEC",
            default=15.0,
        ),
    )
    broker_sync_backoff_multiplier = max(
        1.0,
        coalesce_float(
            "BROKER_SYNC_BACKOFF_MULT",
            default=2.0,
        ),
    )
    broker_sync_jitter = max(
        0.0,
        min(
            0.5,
            coalesce_float(
                "BROKER_SYNC_BACKOFF_JITTER",
                default=0.2,
            ),
        ),
    )

    try:
        # [FIX] Indented correctly inside try
        broker_positions = _fetch_positions_with_retry(
            broker_client,
            max_attempts=broker_sync_attempts,
            backoff_min=broker_sync_backoff_min,
            backoff_max=broker_sync_backoff_max,
            backoff_multiplier=broker_sync_backoff_multiplier,
            jitter_fraction=broker_sync_jitter,
            total_timeout_sec=60.0,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "broker_position_sync_failed",
            extra={"event": "broker_position_sync_failed", "error": str(exc)},
        )
        broker_positions = []
    if broker_positions:
        position_manager.synchronize_with_broker(broker_positions)

    initial_balance = float(
        getattr(config, "initial_balance", 1_000_000.0) or 1_000_000.0
    )

    # 1. Initialize Risk Manager
    risk_manager = RiskManager(
        settings=settings.risk,
        position_manager=position_manager,
        account_balance=initial_balance,
    )

    # 2. Attach Broker (Safe Try/Except)
    try:
        risk_manager.set_broker_client(broker_client)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("risk_manager_attach_broker_failed: %s", exc)

    # 3. Attach Market Data (Safe Try/Except)
    try:
        risk_manager.set_market_data_manager(market_data_manager)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("risk_manager_attach_mdm_failed: %s", exc)

    # 4. Attach Data Hub (Safe Try/Except)
    try:
        risk_manager.attach_data_hub(data_hub)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("risk_manager_attach_data_hub_failed: %s", exc)

    # 5. [FIX] Wire Lot Size Provider (Unconditional)
    # We define this logic regardless of resolver state to ensure sizing always works.
    def _lot_size_lookup(symbol: str) -> int:
        try:
            # A. Try metadata from resolver if available
            if instrument_resolver:
                meta = instrument_resolver.lookup(symbol)
                if meta and "lot_size" in meta:
                    return int(meta["lot_size"])
            
            # B. Fallback Defaults
            sym_upper = symbol.upper()
            if "NIFTY" in sym_upper:
                return 75  # Force 75 for NIFTY
            if "BANKNIFTY" in sym_upper:
                return 15
            return 1
        except Exception:
            # C. Safety Net
            if "NIFTY" in symbol.upper():
                return 75
            return 1
    
    # Always attach the provider
    risk_manager.set_lot_size_provider(_lot_size_lookup)
    LOGGER.info("✅ Wired Lot Size Provider to Risk Manager (NIFTY=75)")

    risk_state: RiskState | None = None
    try:
        spread_mult = max(
            coalesce_float(
                "RISK_STATE_SPREAD_MULT",
                "RISK_STATE_SPREAD_WIDEN_MULT",
                default=3.0,
            ),
            1.0,
        )
        dd_limit = coalesce_float(
            "RISK_STATE_MAX_INTRADAY_DD",
            "RISK_STATE_MAX_DRAWDOWN",
            default=0.0,
        )
        if abs(dd_limit) <= 0.0:
            pct_cap = float(config.risk.max_drawdown_pct or 0.0) / 100.0
            dd_limit = -abs(initial_balance * pct_cap) if pct_cap > 0 else 0.0
        else:
            dd_limit = -abs(dd_limit)
        loss_cap = max(
            coalesce_int(
                "RISK_STATE_MAX_CONSECUTIVE_LOSSES",
                default=settings.risk.max_consecutive_losses,
            ),
            1,
        )
        risk_state = RiskState(
            quote_stale_ms_threshold=int(config.quote_stale_threshold_ms),
            spread_widen_mult=float(spread_mult),
            max_intraday_dd=float(dd_limit),
            max_consecutive_losses=int(loss_cap),
        )
        risk_manager.bind_risk_state(risk_state)
    except Exception:  # pragma: no cover - defensive wiring
        LOGGER.exception("risk_state_init_failed")
        risk_state = None

    paper_engine = PaperFillEngine(data_hub, instrument_resolver)
    live_toggle_env = coalesce_bool(
        "ENABLE_LIVE_TRADING",
        "ENABLE_LIVE",
        default=settings.enable_live,
    )
    live_possible = bool(live_toggle_env and settings.orders.enable_live)
    paper_toggle_env = coalesce_bool("PAPER__ENABLED", default=not live_possible)
    shadow_mode_env = get_bool("SHADOW_MODE", default=not live_possible)
    paper_initial = bool((not live_possible) or paper_toggle_env or shadow_mode_env)
    broker_backend = robust_provider if not paper_initial else paper_engine

    # 9. Initialize Execution
    # [FIX] We inject indicator_engine here to enable Volatility-Adaptive Trailing
    order_manager = OrderManager(
        broker_client=broker_client,
        position_manager=position_manager,
        rate_limiter=rate_limiter,
        instrument_resolver=instrument_resolver,
        indicator_engine=indicator_engine,  # <--- THIS IS THE CRITICAL ADDITION
    )
    order_manager.set_market_data_manager(market_data_manager)
    order_manager.attach_data_hub(data_hub)
    order_manager.set_instrument_resolver(instrument_resolver)
    order_manager.set_risk_manager(risk_manager)
    order_manager.attach_persistent_state(persistent_state)

    bracket_manager: BracketManager | None = None
    # ----------------------------------------------------------------
    # 1. Initialize BracketManager (Clean)
    # ----------------------------------------------------------------
    if settings.execution.enable_bracket_manager:
        try:
            LOGGER.debug("Initializing BracketManager...")
            
            bracket_manager = BracketManager(
                order_manager=order_manager,
                indicator_engine=indicator_engine,
                market_data=market_data_manager
            )
            
            # Configure
            bracket_manager._auto_reduce_sl = settings.execution.bracket_auto_reduce_sl
            bracket_manager._stale_cleanup_age = getattr(settings.execution, "bracket_stale_cleanup_seconds", 86400)
            
            # Attach to OrderManager
            order_manager.set_bracket_manager(bracket_manager=bracket_manager)
            LOGGER.info("✅ BracketManager initialized and attached.")

        except Exception as exc:
            LOGGER.error(f"Failed to initialize BracketManager: {exc}")
  

    
    

    if risk_state is not None:

        def _sync_risk_state_pnl() -> None:
            """Synchronize risk state and Prometheus PnL metrics.

            Args:
                None.

            Returns:
                None.

            Raises:
                None.
            """

            try:
                realized = float(position_manager.get_realized_pnl())
            except Exception:  # pragma: no cover - defensive
                realized = 0.0
            try:
                unrealized = float(position_manager.get_unrealized_pnl())
            except Exception:  # pragma: no cover - defensive
                unrealized = 0.0
            try:
                METRICS.set_pnl_breakdown(
                    book="primary", realized=realized, unrealized=unrealized
                )
            except Exception:  # pragma: no cover - optional metrics
                LOGGER.debug("Unable to sync pnl breakdown", exc_info=True)
            risk_state.on_trade_update(realized_pnl=realized, unrealized_pnl=unrealized)

        def _first_float(payload: Mapping[str, Any], *keys: str) -> float | None:
            for key in keys:
                value = payload.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return None

        def _risk_state_tick_listener(tick: Mapping[str, Any]) -> None:
            bid = _first_float(
                tick,
                "best_bid",
                "best_bid_price",
                "bid",
                "buy_price",
            )
            ask = _first_float(
                tick,
                "best_ask",
                "best_ask_price",
                "ask",
                "sell_price",
            )
            if bid is None or ask is None:
                return
            ts_ns = time_module.time_ns()
            risk_state.on_tick(bid, ask, ts_ns=ts_ns)
            _sync_risk_state_pnl()

        def _risk_state_order_listener(_order: Mapping[str, Any]) -> None:
            _sync_risk_state_pnl()

        _sync_risk_state_pnl()
        risk_symbol = coalesce_str(
            "RISK_STATE_SYMBOL",
            "RISK_STATE__SYMBOL",
           default="NSE:NIFTY 50",
        )
        attach = getattr(risk_state, "attach_data_hub", None)
        if callable(attach):
            try:
                attach(data_hub, symbol=risk_symbol)
            except Exception:  # pragma: no cover - defensive
                LOGGER.debug("risk_state_attach_data_hub_failed", exc_info=True)
        if hasattr(data_hub, "subscribe_ticks") and risk_symbol:
            data_hub.subscribe_ticks(risk_symbol, _risk_state_tick_listener)
        if hasattr(data_hub, "subscribe_orders"):
            data_hub.subscribe_orders(_risk_state_order_listener)

    safe_order_manager = SafeOrderManager(
        order_manager=order_manager,
        settings=settings.orders,
        regime_manager=market_regime_manager,
    )

    session_guard = TradingSessionGuard(
        rate_limiter=rate_limiter,
        risk_manager=risk_manager,
        allow_out_of_hours=coalesce_bool("SESSION_ALLOW_OUT_OF_HOURS", default=True),
    )
    session_allow_override = coalesce_bool(
        "SESSION_ALLOW_OUT_OF_HOURS",
        "SESSION__ALLOW_OUT_OF_HOURS",
        "ALLOW_OFFHOURS_TESTING",
        default=settings.session_allow_out_of_hours,
    )
    session_guard.set_allow_out_of_hours(session_allow_override)
    settings.session_allow_out_of_hours = session_allow_override
    current_open, current_close = session_guard.get_trading_window()
    default_open = f"{current_open.hour:02d}:{current_open.minute:02d}"
    default_close = f"{current_close.hour:02d}:{current_close.minute:02d}"
    # ---- Fix/normalize PNL path if provided ----
    pnl_path_raw = get_str("PNL_PERSIST_PATH")
    pnl_path = normalize_path(pnl_path_raw)
    if pnl_path:
        os.environ["PNL_PERSIST_PATH"] = pnl_path

    trading_window_start = (
        get_str("DATA__TIME_FILTER_START", default_open) or default_open
    )
    trading_window_end = (
        get_str("DATA__TIME_FILTER_END", default_close) or default_close
    )
    session_guard.set_trading_window(trading_window_start, trading_window_end)
    
    # ----------------------------------------------------------------------
    # 6. Build Elite Strategies (Corrected)
    # ----------------------------------------------------------------------
    elite_strategies: list[Any] = []
    try:
        # ✅ FIX: Pass 'indicator_engine' to the builder
        elite_strategies = build_elite_strategies(
            settings.elite,  # Verify if this is settings.elite or settings.strategies in your config
            indicator_engine
        )

        # Inject DataHub into strategies that need it (e.g. OrderFlow, Gamma)
        if elite_strategies and data_hub:
            for strategy in elite_strategies:
                if hasattr(strategy, "set_data_hub"):
                    try:
                        strategy.set_data_hub(data_hub)
                    except Exception as exc:
                        LOGGER.warning(
                            "Failed to inject DataHub into %s: %s", 
                            getattr(strategy, "name", "unknown"), 
                            exc
                        )

    except Exception as exc:  # noqa: BLE001
        # ✅ FIX: Proper indentation for the except block
        LOGGER.error(
            "Failure in build_elite_strategies: %s",
            exc,
            exc_info=exc,
            extra={"event": "elite_build_error"},
        )
    else:
        # ✅ FIX: Logic to run only if NO exception occurred
        if elite_strategies:
            LOGGER.info(
                "Condition met: elite_strategies_loaded",
                extra={
                    "event": "elite_strategies_loaded",
                    "count": len(elite_strategies),
                },
            )
        else:
            LOGGER.warning(
                "No elite strategies enabled; trading will be disabled",
                extra={"event": "elite_strategies_missing"},
                    )

        # Ensure DataHub is injected into strategies that need complex metrics (IV/Greeks)
        if elite_strategies and data_hub:
            for strategy in elite_strategies:
                if hasattr(strategy, "set_data_hub"):
                    try:
                        strategy.set_data_hub(data_hub)
                        LOGGER.debug(f"Injected DataHub into {strategy.name}")
                    except Exception as exc:
                        LOGGER.warning(f"Failed to inject DataHub into {strategy.name}: {exc}")
        
          
    strategy_instances: list[Any] = list(elite_strategies)
    # Ensure DataHub is injected into all strategies that need enriched data (IV/Greeks).
    if strategy_instances and data_hub:
        for strategy in strategy_instances:
            # Check if the strategy has the required setter method (set_data_hub)
            if hasattr(strategy, "set_data_hub"):
                try:
                    strategy.set_data_hub(data_hub)
                    LOGGER.debug(f"Injected DataHub into {strategy.name}")
                except Exception as exc:
                    LOGGER.warning(
                        f"Failed to inject DataHub into {strategy.name}: {exc}"
                    )    
    orchestrator = StrategyOrchestrator(
        risk_manager=risk_manager,
        order_manager=safe_order_manager,
        data_hub=data_hub,
        futures_symbol="NIFTY",
    )
    regime_bias_map: dict[str, dict[str, float]] = {}
    if elite_strategies:
        elite_fraction = settings.elite.position_size_pct / 100.0
        if elite_fraction <= 0:
            LOGGER.warning(
                "Elite position size pct not positive; defaulting to 1% of capital.",
                extra={"event": "elite_fraction_default"},
            )
            elite_fraction = 0.01
        tag_lookup = elite_strategy_tags(settings.elite)
        # Optimization B: Dynamic Sizing based on Regime
        # Trend = Aggressive (100% size), Chop = Conservative (50% size)
        bias_candidates: dict[str, dict[str, float]] = {
            "trend": {},
            "chop": {},
            "volcrush": {},
            "event": {}, 
        }
        for strategy in elite_strategies:
            tags = tag_lookup.get(strategy.name, ("elite",))
            try:
                orchestrator.register_strategy(
                    strategy.name,
                    capital_fraction=elite_fraction,
                    correlation_tags=tags,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Failure registering elite strategy %s: %s",
                    strategy.name,
                    exc,
                    exc_info=exc,
                    extra={"event": "elite_register_error", "strategy": strategy.name},
                )
                continue
            tag_set = {tag.lower() for tag in tags}
            if {"momentum", "opening", "orderflow"} & tag_set:
                bias_candidates["trend"][strategy.name] = 1.15
            if {"mean_reversion", "liquidity"} & tag_set:
                bias_candidates["chop"][strategy.name] = 1.2
            if {"volatility", "income"} & tag_set:
                bias_candidates["volcrush"][strategy.name] = 1.15
            if {"structure", "orderflow", "liquidity"} & tag_set:
                bias_candidates["event"][strategy.name] = 1.1
        regime_bias_map = {
            regime: mapping for regime, mapping in bias_candidates.items() if mapping
        }
    else:
        tag_lookup = {}

    def _regime_signal_snapshot() -> dict[str, object] | None:
        """Return lightweight market regime snapshot for scoring.

        Args:
            None.

        Returns:
            dict[str, object] | None: Normalised regime snapshot when
            available.

        Raises:
            None.
        """

        snapshot = market_regime_manager.get_latest_snapshot()
        if snapshot is None:
            return None
        return {
            "regime": snapshot.regime,
            "confidence": snapshot.confidence,
            "updated_at": snapshot.updated_at,
        }

    strategy_manager = StrategyManager(
        strategies=strategy_instances,
        indicator_engine=indicator_engine,
        position_manager=position_manager,
        data_hub=data_hub,
        orchestrator=orchestrator,
        futures_symbol="NIFTY",
        score_weights=None,
        regime_signal_getter=_regime_signal_snapshot,
        regime_bias_map=regime_bias_map,
        market_regime_manager=market_regime_manager,
    )

    unified_manager: UnifiedManager | None = None
    try:
        LOGGER.debug(
            "Entered initialize_components unified manager wiring",
            extra={"event": "init.unified_manager.enter"},
        )
        unified_manager = UnifiedManager(
            broker=broker_client,
            mdm=market_data_manager,
            ws=websocket_manager,
            risk=risk_manager,
            streamer=streamer,
            data_hub=data_hub,
            orders=safe_order_manager or order_manager,
            strategies=strategy_manager,
            logger=LOGGER.getChild("init.unified_manager"),
        )
        with suppress(Exception):
            market_data_manager.set_unified_manager(unified_manager)
        with suppress(Exception):
            risk_manager.set_unified_manager(unified_manager)
        LOGGER.info(
            "Condition met: unified_manager_wired",
            extra={
                "event": "init.unified_manager.ready",
                "has_safe_order_manager": bool(safe_order_manager),
                "has_strategy_manager": bool(strategy_manager),
            },
        )
    except Exception as exc:  # noqa: BLE001 - defensive wiring
        LOGGER.error(
            "Failure in initialize_components unified manager wiring: %s",
            exc,
            extra={"event": "init.unified_manager.error"},
            exc_info=exc,
        )
        unified_manager = None

    strike_selector: StrikeSelector | None = None
    if data_hub is not None:
        strike_selector = StrikeSelector(
            data_hub=data_hub,
            selector_settings=settings.selector,
            liquidity_settings=settings.liquidity,
        )

    order_queue = OrderQueue()

    state_tracker = StateTracker()
    lifecycle_tracker_adapter = _LifecycleTrackerAdapter(state_tracker)

    preflight_validator = PreFlightValidator(
        risk_manager=risk_manager,
        regime_manager=market_regime_manager,
        datahub=data_hub,
        session_guard=session_guard,
    )
    lifecycle_manager = LifecycleManager(
        data_hub=data_hub,
        order_queue=order_queue,
        state_tracker=lifecycle_tracker_adapter,
    )
    execution_mode_env = (
        coalesce_str("EXECUTION_MODE", default="SHADOW") or "SHADOW"
    ).upper()
    router_settings = ExecutionRouterSettings(
        retry_attempts=int(coalesce_int("EXECUTION_RETRY_ATTEMPTS", default=3)),
        retry_delay_ms=int(coalesce_int("EXECUTION_RETRY_DELAY_MS", default=500)),
        shadow_drift_threshold_bps=float(
            coalesce_float("SHADOW_DRIFT_THRESHOLD_BPS", default=20.0)
        ),
    )
    execution_router = ExecutionRouter(
        live_executor=safe_order_manager,
        paper_executor=paper_engine,
        mode=execution_mode_env,
        settings=router_settings,
    )
    reconciliation_interval = coalesce_int("RECONCILIATION_INTERVAL_SEC", default=30)
    reconciliation_alert = coalesce_bool(
        "RECONCILIATION_ALERT_ON_MISMATCH", default=True
    )
    post_fill_monitor = PostFillMonitor(
        broker_client=robust_provider,
        state_tracker=state_tracker,
        interval_sec=int(reconciliation_interval),
        alert_on_mismatch=bool(reconciliation_alert),
    )
    order_execution_hub = OrderExecutionHub(
        state_tracker=state_tracker,
        preflight_validator=preflight_validator,
        lifecycle_manager=lifecycle_manager,
        order_queue=order_queue,
        execution_router=execution_router,
        post_fill_monitor=post_fill_monitor,
        data_hub=data_hub,
        regime_manager=market_regime_manager,
        risk_manager=risk_manager,
    )
    order_processor = OrderProcessor(
        message_bus=message_bus,
        safe_order_manager=safe_order_manager,
        risk_manager=risk_manager,
        position_manager=position_manager,
        data_hub=data_hub,
    )

    strategy_runner = StrategyRunner(
        market_data_manager=market_data_manager,
        indicator_engine=indicator_engine,
        strategy_manager=strategy_manager,
        order_manager=order_manager,
        risk_manager=risk_manager,
        position_manager=position_manager,
        config=_get_strategy_config(config),
        data_hub=data_hub,
        strike_selector=strike_selector,
        message_bus=message_bus,
        bracket_manager=bracket_manager,
    )
    strategy_runner.attach_persistent_state(persistent_state)
    strategy_runner.restore_trades(persistent_state.load_trades())
    # [FIX 2/2] Populate the reference for Polling Mode to enable the hot-wire
    strategy_runner_ref["instance"] = strategy_runner
    LOGGER.info("✅ Polling Streamer -> StrategyRunner direct wiring established.")

    settings.enable_live = bool(live_toggle_env)
    mandatory_paper = not live_possible
    paper_state: dict[str, bool] = {"enabled": bool(paper_initial or mandatory_paper)}
    ctx_ref: dict[str, BotContext | None] = {"ctx": None}

    def _apply_paper_mode(enabled: bool) -> bool:
        desired = bool(enabled)
        next_state = bool(mandatory_paper or desired)
        paper_state["enabled"] = next_state
        backend = paper_engine if next_state else broker_client
        order_manager.set_broker_client(backend)
        orders_enabled_now = next_state or live_possible
        safe_order_manager.set_live_enabled(orders_enabled_now)
        risk_manager.force_shadow(next_state)
        ctx_obj = ctx_ref.get("ctx")
        if ctx_obj is not None:
            ctx_obj.shadow_mode_enabled = next_state
        target_mode = "PAPER" if next_state else execution_mode_env
        execution_router.set_mode(target_mode)
        return paper_state["enabled"]

    def _paper_mode_enabled() -> bool:
        return paper_state["enabled"]

    _apply_paper_mode(paper_state["enabled"])
    shadow_enabled = paper_state["enabled"]

    notifier = TelegramEnhancedNotifier.from_settings(settings.notifications)
    order_manager.set_notifier(notifier)
    telegram_logger = get_logger("telegram")
    telegram_mode = (
        "webhook"
        if (
            settings.notifications.enabled
            and settings.notifications.webhook_enabled
            and settings.notifications.public_base_url
        )
        else "polling"
    )
    telegram_logger.info(
        "Telegram controller starting in %s mode",
        telegram_mode,
        extra={
            "event": "telegram_mode",
            "mode": telegram_mode,
            "webhook_env_enabled": telegram_webhook_env_enabled,
            "notifications_enabled": settings.notifications.enabled,
        },
    )
    # Reconcile positions on startup: schedule if loop running, otherwise run synchronously.
    async def _reconcile_with_timeout():
        """Wrapper to add timeout protection"""
        try:
            await asyncio.wait_for(
                reconcile_positions_on_startup(
                    broker_client=broker_client,
                    position_manager=position_manager,
                    order_manager=order_manager,
                    logger=LOGGER,
                ),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            LOGGER.error(
                "Position reconciliation timed out after 30s",
                extra={"event": "reconcile_timeout"}
            )
        except Exception as exc:
            LOGGER.error(
                f"Position reconciliation failed: {exc}",
                extra={"event": "reconcile_failed"},
                exc_info=True
            )
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_reconcile_with_timeout())  # ✅ USE WRAPPED VERSION

            LOGGER.info(
                "Scheduled reconcile_positions_on_startup as background task",
                extra={"event": "reconcile_positions_scheduled"},
            )
        else:
            # No running loop: run the coroutine to completion (blocking).
            asyncio.run(
                reconcile_positions_on_startup(
                    broker_client=broker_client,
                    position_manager=position_manager,
                    order_manager=order_manager,
                    logger=LOGGER,
                )    
            )
            LOGGER.info("Completed reconcile_positions_on_startup (blocking run)", extra={"event": "reconcile_positions_completed"})
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Startup reconciliation failed - will retry in background",
            extra={
                "event": "startup.reconcile.error",
                "error": str(exc),
                "severity": "warning",
            },
            exc_info=False,
        )
        # If you want to attempt a background retry, schedule a noop wrapper that will call reconcile later.
        with suppress(Exception):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(reconcile_positions_on_startup(
                        broker_client=broker_client,
                        position_manager=position_manager,
                        order_manager=order_manager,
                        logger=LOGGER,
                    ))
            except Exception:
                pass

    background_tasks: list[asyncio.Task[Any]] = []
    try:
        background_tasks = start_background_tasks(order_manager, LOGGER)
        LOGGER.info(
            "Background tasks started",
            extra={
                "event": "background_tasks.started",
                "count": len(background_tasks),
            },
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failed to start background tasks",
            extra={
                "event": "background_tasks.failed",
                "error": str(exc),
            },
            exc_info=exc,
        )

    refresh_task = schedule_instrument_refresh(
        settings,
        instrument_resolver,
        state=instrument_state,
    )
    if refresh_task is not None:
        background_tasks.append(refresh_task)
        # keep a handle for graceful shutdown
        ctx_refresh_task: asyncio.Task[Any] | None = refresh_task
    else:
        ctx_refresh_task = None

        async def cleanup_stale_brackets_task() -> None:
            """Periodically remove stale bracket state entries.

            Args:
                None.

            Returns:
                None.

            Raises:
                None.
            """

            LOGGER.debug(
                "Entered cleanup_stale_brackets_task",
                extra={"event": "bracket.cleanup.task.enter"},
            )
            while True:
                try:
                    await asyncio.sleep(3600)
                    removed = bracket_manager.cleanup_stale_brackets(
                        max_age_seconds=settings.execution.bracket_stale_cleanup_seconds,
                    )
                    if removed > 0:
                        LOGGER.info(
                            "Cleaned up stale brackets",
                            extra={
                                "event": "bracket.cleanup.completed",
                                "count": removed,
                                "max_age": (
                                    settings.execution.bracket_stale_cleanup_seconds
                                ),
                            },
                        )
                except (
                    asyncio.CancelledError
                ):  # pragma: no cover - cooperative cancellation
                    LOGGER.info(
                        "Bracket cleanup task cancelled",
                        extra={"event": "bracket.cleanup.task.cancelled"},
                    )
                    raise
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error(
                        "Bracket cleanup task error: %s",
                        exc,
                        extra={"event": "bracket.cleanup.task.error"},
                        exc_info=exc,
                    )

        background_tasks.append(asyncio.create_task(cleanup_stale_brackets_task()))

    def _build_health_snapshot() -> dict[str, object]:
        """Return an aggregate health snapshot for out-of-band alerts.

        Args:
            None.

        Returns:
            Dictionary containing guard, risk, order, and market data signals.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered _build_health_snapshot",
            extra={"event": "health_snapshot_build_enter"},
        )
        snapshot: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "shadow" if paper_state["enabled"] else "live",
        }
        try:
            guard_status = session_guard.evaluate()
        except Exception as exc:  # noqa: BLE001 - defensive
            LOGGER.error(
                "Failure in _build_health_snapshot guard: %s",
                exc,
                extra={"event": "health_snapshot_guard_error"},
            )
            snapshot["session"] = {"error": str(exc)}
        else:
            snapshot["session"] = guard_status.as_dict()

        mdm_payload: dict[str, object] = {}
        if market_data_manager is None:
            mdm_payload = {"error": "unavailable"}
        else:
            try:
                status = market_data_manager.mdm_status()
                mdm_payload = {
                    "ws_connected": bool(status.get("ws_connected")),
                    "fallback_enabled": bool(status.get("fallback_enabled")),
                    "heartbeat_age": status.get("heartbeat_age"),
                    "last_tick_source": status.get("last_tick_source", {}),
                    "last_tick_age": status.get("last_tick_age", {}),
                }
            except Exception as exc:  # noqa: BLE001 - defensive
                LOGGER.error(
                    "Failure in _build_health_snapshot mdm: %s",
                    exc,
                    extra={"event": "health_snapshot_mdm_error"},
                )
                mdm_payload = {"error": str(exc)}
        snapshot["market_data"] = mdm_payload

        try:
            risk_snapshot = risk_manager.snapshot()
            snapshot["risk"] = {
                "breaker_tripped": risk_snapshot.breaker_tripped,
                "cooldown_remaining": risk_snapshot.cooldown_remaining,
                "losses_in_row": risk_snapshot.losses_in_row,
                "last_rejection": risk_snapshot.last_rejection,
                "timestamp": risk_snapshot.timestamp.isoformat(),
            }
        except Exception as exc:  # noqa: BLE001 - defensive
            LOGGER.error(
                "Failure in _build_health_snapshot risk: %s",
                exc,
                extra={"event": "health_snapshot_risk_error"},
            )
            snapshot["risk"] = {"error": str(exc)}

        try:
            open_positions = list(position_manager.get_open_positions())
            snapshot["positions"] = {"open": len(open_positions)}
        except Exception as exc:  # noqa: BLE001 - defensive
            LOGGER.error(
                "Failure in _build_health_snapshot positions: %s",
                exc,
                extra={"event": "health_snapshot_position_error"},
            )
            snapshot["positions"] = {"error": str(exc)}

        try:
            recent = order_manager.recent_orders(limit=5)
            snapshot["orders"] = {"recent": len(recent)}
        except Exception as exc:  # noqa: BLE001 - defensive
            LOGGER.error(
                "Failure in _build_health_snapshot orders: %s",
                exc,
                extra={"event": "health_snapshot_orders_error"},
            )
            snapshot["orders"] = {"error": str(exc)}

        return snapshot
    def _notify(event: str, payload: Mapping[str, object] | None = None) -> None:
        if notifier is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - initialization phase
            LOGGER.debug(
                "No running loop to dispatch telegram notification",
                extra={"event": event},
            )
            return
        loop.create_task(notifier.send_event(event, payload))
    def _emit_health_snapshot(trigger: str, detail: str | None = None) -> None:
        """Dispatch a health snapshot notification for high-impact events.

        Args:
            trigger: Identifier describing the initiating condition.
            detail: Optional human-readable detail string.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.info(
            "Condition met: emitting health snapshot",
            extra={
                "event": "health_snapshot_emit",
                "trigger": trigger,
                "detail": detail,
            },
        )
        payload: dict[str, object] = {
            "trigger": trigger,
            "detail": detail,
            "snapshot": _build_health_snapshot(),
        }
        _notify("HEALTH_SNAPSHOT", payload)

    
    shadow_trader: ShadowPaperTrader | None = None
    if settings.shadow.drift_threshold_pct > 0:

        def _disable_live(_reason: str) -> None:
            safe_order_manager.set_live_enabled(False)

        shadow_trader = ShadowPaperTrader(
            settings=settings.shadow,
            live_equity_fn=position_manager.get_net_pnl,
            notifier=_notify if notifier is not None else None,
            disable_live_callback=_disable_live,
        )
        shadow_trader.attach_persistent_state(persistent_state)
        safe_order_manager.post_order_hook = (
            lambda symbol, side, qty, price: shadow_trader.record_order(
                symbol, side, qty, price or 0.0
            )
        )

    health_state = HealthState(
        streamer=streamer,
        order_manager=safe_order_manager,
        risk_manager=risk_manager,
        live_enabled=lambda: safe_order_manager.settings.enable_live,
        session_guard=session_guard,
        stream_supervisor=stream_supervisor,
        websocket_enabled=websocket_enabled,
    )
    health_app = create_health_app(health_state)

    ctx = BotContext(
        settings=settings,
        config=config,
        rate_limiter=rate_limiter,
        broker_client=robust_provider,
        message_bus=message_bus,
        websocket_client=websocket_client,
        websocket_manager=websocket_manager,
        streamer=streamer,
        stream_supervisor=stream_supervisor,
        data_hub=data_hub,
        market_data_manager=market_data_manager,
        market_regime=market_regime_detector,
        market_regime_manager=market_regime_manager,
        indicator_engine=indicator_engine,
        position_manager=position_manager,
        risk_manager=risk_manager,
        persistent_state=persistent_state,
        order_manager=order_manager,
        bracket_manager=bracket_manager,
        paper_engine=paper_engine,
        safe_order_manager=safe_order_manager,
        order_queue=order_queue,
        state_tracker=state_tracker,
        preflight_validator=preflight_validator,
        lifecycle_manager=lifecycle_manager,
        execution_router=execution_router,
        post_fill_monitor=post_fill_monitor,
        order_execution_hub=order_execution_hub,
        strategy_manager=strategy_manager,
        strategy_runner=strategy_runner,
        unified_manager=unified_manager,
        order_processor=order_processor,
        instrument_resolver=instrument_resolver,
        instrument_db=instrument_conn,
        instrument_universe=instrument_state,
        instrument_refresh_task=ctx_refresh_task,
        websocket_enabled=websocket_enabled,
        shadow_mode_enabled=shadow_enabled,
        shadow_trader=shadow_trader,
        out_of_hours_override=False,
        telegram_bot=None,
        telegram_application=None,
        telegram_notifier=notifier,
        health_app=health_app,
        session_guard=session_guard,
    )

    resolver_candidate = ctx.instrument_resolver
    if resolver_candidate is None and ctx.broker_client is not None:
        try:
            resolver_candidate = InstrumentResolver(ctx.broker_client)
            with suppress(Exception):
                resolver_candidate.warm()
            ctx.instrument_resolver = resolver_candidate
        except Exception as _resolver_exc:  # noqa: BLE001 - defensive wiring
            LOGGER.error(
                "Resolver init failed: %s",
                _resolver_exc,
                extra={"event": "resolver_init_failed"},
                exc_info=_resolver_exc,
            )
            ctx.instrument_resolver = None
            resolver_candidate = None
    if resolver_candidate is not None:
        if ctx.market_data_manager is not None:
            try:
                setattr(ctx.market_data_manager, "_resolver", resolver_candidate)
            except Exception as exc:  # noqa: BLE001 - defensive wiring
                LOGGER.error(
                    "Resolver attach to MDM failed: %s",
                    exc,
                    extra={"event": "resolver_attach_mdm_failed"},
                    exc_info=exc,
                )
        if unified_manager is not None:
            try:
                setattr(unified_manager, "resolver", resolver_candidate)
            except Exception as exc:  # noqa: BLE001 - defensive wiring
                LOGGER.error(
                    "Resolver attach to UM failed: %s",
                    exc,
                    extra={"event": "resolver_attach_um_failed"},
                    exc_info=exc,
                )

    ctx_ref["ctx"] = ctx
    runtime_selfchecker = RuntimeSelfChecker(ctx)
    ctx.selfchecker = runtime_selfchecker
    try:
        health_state.selfchecker = runtime_selfchecker
    except Exception:  # pragma: no cover - defensive assignment
        LOGGER.debug("health_state lacks selfchecker attribute", exc_info=True)
    ctx.shadow_mode_enabled = paper_state["enabled"]

    global _LATEST_CTX
    _LATEST_CTX = ctx
    # Initialize Telegram Bot with all components wired
    try:
        _setup_telegram(ctx)
        # ✅ FIX: Only log success if the bot object was actually created
        if ctx.telegram_bot:
            LOGGER.info("✅ Telegram Bot initialized", extra={"event": "telegram_ready"})
        else:
            # This explains why you might see "Initialized" but get no messages
            LOGGER.warning("⚠️ Telegram Bot NOT initialized (Check Token/Chat ID)")
    except Exception as telegram_exc:
        LOGGER.error(f"❌ Telegram initialization failed: {telegram_exc}", exc_info=True)

    if _HTTP_APP is not None:
        try:
            _HTTP_APP.state.bot_context = ctx
        except AttributeError:  # pragma: no cover - FastAPI state guard
            pass

    order_manager.set_session_guard_getter(session_guard.snapshot)
    order_manager.set_trade_mode_getters(
        enable_live=lambda: bool(
            settings.enable_live and safe_order_manager.settings.enable_live
        ),
        shadow_mode=lambda: bool(ctx.shadow_mode_enabled),
    )

    if websocket_manager is not None:
        _bind_ws_mdm(ctx)

    def _risk_snapshot_to_dict(snapshot: RiskSnapshot) -> dict[str, object]:
        return {
            "daily_realized": snapshot.daily_realized,
            "daily_loss_limit": snapshot.daily_loss_limit,
            "day_loss": snapshot.day_loss,
            "max_day_loss": snapshot.max_day_loss,
            "losses_in_row": snapshot.losses_in_row,
            "cooldown_remaining": snapshot.cooldown_remaining,
            "breaker_tripped": snapshot.breaker_tripped,
            "breaker_reason": snapshot.breaker_reason,
            "shadow_forced": snapshot.shadow_forced,
            "per_trade_risk_pct": snapshot.per_trade_risk_pct,
            "last_rejection": snapshot.last_rejection,
            "timestamp": snapshot.timestamp.isoformat(),
        }

    def _flatten_positions(reason: str) -> list[str]:
        flattened: list[str] = []
        for position in position_manager.get_open_positions():
            qty = getattr(position, "quantity", 0)
            if qty <= 0:
                continue
            exit_side: Literal["BUY", "SELL"] = (
                "SELL" if position.side == "LONG" else "BUY"
            )
            price = market_data_manager.get_latest_price(position.symbol)
            if price is None:
                price = getattr(position, "current_price", None) or getattr(
                    position, "entry_price", None
                )
            try:
                order_manager.place_order(
                    symbol=position.symbol,
                    side=exit_side,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                    price=None,
                )
                flattened.append(position.symbol)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "breaker_flatten_failed",
                    extra={
                        "event": "breaker_flatten_failed",
                        "symbol": position.symbol,
                        "err": str(exc),
                    },
                )
        return flattened

    def _handle_risk_breaker(reason: str, snapshot: RiskSnapshot) -> None:
        ctx.shadow_mode_enabled = True
        safe_order_manager.set_live_enabled(False)
        risk_manager.force_shadow(True)
        cancelled: list[str] = []
        try:
            cancelled = order_manager.cancel_pending_orders()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "breaker_cancel_failed",
                extra={"event": "breaker_cancel_failed", "err": str(exc)},
            )
        flattened = _flatten_positions(reason)
        LOGGER.error(
            "risk_breaker_trip",
            extra={
                "event": "risk_breaker_trip",
                "reason": reason,
                "cancelled_orders": cancelled,
                "flattened": flattened,
            },
        )
        session_guard = ctx.session_guard
        if session_guard is not None:
            session_guard.evaluate()
        else:
            LOGGER.debug(
                "Session guard unavailable during breaker handling",
                extra={"event": "risk_breaker_no_session_guard"},
            )
        _notify(
            "RISK_BREAKER",
            {
                "reason": reason,
                "cancelled_orders": cancelled,
                "flattened": flattened,
                "snapshot": _risk_snapshot_to_dict(snapshot),
            },
        )
        _emit_health_snapshot("session_breaker", reason)

    risk_manager.alert_callback = _handle_risk_breaker

    async def _breaker_alert_sender(reason: str) -> None:
        """Dispatch Telegram alert when the risk breaker trips.

        Args:
            reason: Human-readable reason describing the breaker trigger.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered breaker alert sender",
            extra={"event": "risk_breaker_alert_sender_enter", "reason": reason},
        )
        notifier_ref = ctx.telegram_notifier
        if notifier_ref is None:
            LOGGER.info(
                "Breaker alert skipped: notifier unavailable",
                extra={
                    "event": "risk_breaker_alert_sender_missing",
                    "reason": reason,
                },
            )
            return
        payload: dict[str, object] = {
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            snapshot = _require_component(ctx.risk_manager, "risk_manager").snapshot()
            payload["snapshot"] = _risk_snapshot_to_dict(snapshot)
        except Exception as exc:  # noqa: BLE001 - defensive snapshot capture
            LOGGER.error(
                "Failure in breaker alert snapshot capture: %s",
                exc,
                extra={"event": "risk_breaker_alert_snapshot_error"},
                exc_info=exc,
            )
        try:
            await notifier_ref.send_event("RISK_BREAKER_TRIPPED", payload)
        except Exception as exc:  # noqa: BLE001 - defensive notifier surface
            LOGGER.error(
                "Failure in breaker alert send: %s",
                exc,
                extra={
                    "event": "risk_breaker_alert_send_error",
                    "reason": reason,
                },
                exc_info=exc,
            )
        else:
            LOGGER.info(
                "Condition met: breaker alert dispatched",
                extra={
                    "event": "risk_breaker_alert_sent",
                    "reason": reason,
                },
            )

    _require_component(ctx.risk_manager, "risk_manager").breaker_alert_sender = (
        _breaker_alert_sender
    )

    def _handle_order_rejection(symbol: str, reason: str) -> None:
        """Process order rejections and emit health snapshots when needed.

        Args:
            symbol: Instrument identifier associated with the rejection.
            reason: Textual reason describing the rejection.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered _handle_order_rejection",
            extra={
                "event": "order_rejection_handle",
                "symbol": symbol,
                "reason": reason,
            },
        )
        try:
            risk_manager.record_rejection(reason)
        except Exception as exc:  # noqa: BLE001 - defensive
            LOGGER.error(
                "Failure in _handle_order_rejection record: %s",
                exc,
                extra={"event": "order_rejection_record_error"},
            )
        lower_reason = (reason or "").lower()
        if "storm" in lower_reason:
            _emit_health_snapshot("skip_storm", lower_reason)
        if "brownout" in lower_reason:
            _emit_health_snapshot("api_brownout", lower_reason)

    safe_order_manager.on_order_rejected = _handle_order_rejection

    def set_shadow(on: bool) -> bool:
        desired_shadow = bool(on)
        if desired_shadow:
            ctx.out_of_hours_override = False
            _apply_paper_mode(True)
            LOGGER.info(
                "Shadow mode enabled", extra={"event": "shadow_mode", "enabled": True}
            )
            return True

        if not settings.enable_live:
            LOGGER.warning(
                "Live trading toggle rejected; ENABLE_LIVE is false",
                extra={"event": "live_toggle_blocked"},
            )
            ctx.out_of_hours_override = False
            _apply_paper_mode(True)
            return False

        if not live_possible:
            LOGGER.warning(
                "Live trading unavailable; broker backend disabled",
                extra={"event": "live_toggle_blocked"},
            )
            ctx.out_of_hours_override = False
            _apply_paper_mode(True)
            return False

        allowed, status = session_guard.allow_live()
        override_allowed = False
        session_reason = "OK"
        snapshot: RiskSnapshot | None = None
        if (
            not allowed
            and status.override_out_of_hours
            and status.session_valid
            and status.rate_limits_ok
            and status.risk_green
            and not status.market_open
        ):
            override_allowed = True
            LOGGER.warning(
                "Trading session guard override active outside market hours",
                extra={"event": "session_override", **status.as_dict()},
            )

        soft_override = False
        if not allowed and not override_allowed:
            try:
                snapshot = risk_manager.snapshot()
            except Exception:  # pragma: no cover - defensive
                snapshot = None
            session_reason, soft_override = _resolve_session_reason(status, snapshot)
            if soft_override:
                LOGGER.warning(
                    "Trading session guard soft override engaged",
                    extra={
                        "event": "session_soft_override",
                        **status.as_dict(),
                        "risk_reason": session_reason,
                    },
                )
            if not soft_override:
                ctx.out_of_hours_override = False
                _apply_paper_mode(True)
                LOGGER.warning(
                    "Trading session guard denied live toggle",
                    extra={
                        "event": "session_guard_blocked",
                        **status.as_dict(),
                        "risk_reason": session_reason,
                    },
                )
                block_payload = {**status.as_dict(), "session_reason": session_reason}
                _notify("LIVE_TOGGLE_BLOCKED", block_payload)
                return False
        elif not allowed and override_allowed:
            try:
                snapshot = risk_manager.snapshot()
            except Exception:  # pragma: no cover - defensive
                snapshot = None
            session_reason, _ = _resolve_session_reason(status, snapshot)
        else:
            session_reason = "OK"

        ctx.out_of_hours_override = override_allowed
        override_kind: str | None = None
        if override_allowed:
            override_kind = "out_of_hours"
        elif soft_override:
            override_kind = "soft"
        risk_manager.reset_on_start(override_kind is not None)
        payload = {**status.as_dict(), "session_reason": session_reason}
        if override_kind is not None:
            payload["session_override"] = override_kind
        if override_kind == "out_of_hours":
            payload = {
                **payload,
                "override_used": True,
                "override_kind": "out_of_hours",
            }
            LOGGER.info(
                "Live trading enabled via out-of-hours override",
                extra={
                    "event": "shadow_mode",
                    "enabled": False,
                    "override": True,
                    "override_kind": "out_of_hours",
                    "session_override": "out_of_hours",
                },
            )
        elif override_kind == "soft":
            payload = {
                **payload,
                "override_used": True,
                "override_kind": "soft",
            }
            LOGGER.info(
                "Live trading enabled via soft risk override",
                extra={
                    "event": "shadow_mode",
                    "enabled": False,
                    "override": "soft",
                    "override_kind": "soft",
                    "session_override": "soft",
                },
            )
        else:
            LOGGER.info(
                "Live trading enabled", extra={"event": "shadow_mode", "enabled": False}
            )
        _notify("LIVE_MODE_ENABLED", payload)
        _apply_paper_mode(False)
        return True

    def get_shadow() -> bool:
        return bool(ctx.shadow_mode_enabled)

    if shadow_trader is not None:

        def _force_shadow(reason: str) -> None:
            """Disable live routing and enable paper mode on drift breaches.

            Args:
                reason: Reason text supplied by the drift monitor.

            Returns:
                None.

            Raises:
                None.
            """

            LOGGER.info(
                "Condition met: shadow_forced_by_trader",
                extra={"event": "shadow_forced_by_trader", "reason": reason},
            )
            safe_order_manager.set_live_enabled(False)
            set_shadow(True)

        shadow_trader.disable_live_callback = _force_shadow

    telegram_cfg = getattr(config, "telegram", None)
    ctx.telegram_bot = None
    # Ensure controller exists for telegram setup
    controller = _HTTP_CONTROLLER
    if controller is None and settings.notifications.enabled:
        # Create a minimal controller if the HTTP app wasn't started
        try:
            from nifty_scalper_bot.notifications.telegram_controller import TelegramWebhookController
            notifier = ctx.telegram_notifier
            if notifier and hasattr(notifier, 'bot'):
                controller = TelegramWebhookController(
                    bot=notifier.bot,
                    settings=settings.notifications,
                )
                LOGGER.info("✅ Created fallback TelegramWebhookController")
        except Exception as e:
            LOGGER.warning(f"Could not create fallback controller: {e}")
    telegram_bot_instance: TelegramBot | None = None
    telegram_chat_id: int | None = None

    if not settings.notifications.legacy_console_enabled:
        LOGGER.info(
            "legacy_telegram_console_disabled",
            extra={"event": "legacy_console_disabled"},
        )

    try:
        if (
            telegram_cfg
            and getattr(telegram_cfg, "bot_token", None)
            and getattr(telegram_cfg, "chat_id", None) is not None
        ):
            from nifty_scalper_bot.notifications.telegram_controller import (
                TelegramBot,
                TelegramDeps,
            )

            telegram_plain_flag = get_bool("TELEGRAM__PLAIN_TEXT", True)
            polling_enabled_flag = coalesce_bool(
                "TELEGRAM__POLLING_ENABLED",
                default=bool(telegram_cfg.enable_polling_fallback),
            )

            paper_mode_getters: dict[str, Callable[[], bool]] = {
                "orders": _paper_mode_enabled,
            }
            paper_mode_setters: dict[str, Callable[[bool], bool]] = {
                "orders": set_shadow,
            }

            runner_ref = ctx.strategy_runner
            if runner_ref is not None:

                def _strategy_paper_getter() -> bool:
                    """Return whether strategy runner is in paper-only mode."""

                    runner = _require_component(runner_ref, "strategy_runner")
                    with suppress(Exception):
                        status = runner.get_status()
                        if isinstance(status, Mapping):
                            if "trading_paused" in status:
                                return bool(status["trading_paused"])
                    paused_flag = getattr(runner, "paused", None)
                    if paused_flag is not None:
                        return bool(paused_flag)
                    return bool(getattr(runner, "_trading_paused", False))

                def _strategy_paper_setter(enabled: bool) -> bool:
                    """Toggle paper mode on the strategy runner."""

                    runner = _require_component(runner_ref, "strategy_runner")
                    try:
                        if enabled:
                            runner.pause_trading()
                        else:
                            runner.resume_trading()
                    except Exception as exc:  # noqa: BLE001 - defensive
                        LOGGER.warning(
                            "telegram_strategy_paper_toggle_failed",
                            extra={
                                "event": "paper_toggle_failed",
                                "section": "strategy",
                                "err": str(exc),
                            },
                        )
                        return False
                    return True

                paper_mode_getters["strategy"] = _strategy_paper_getter
                paper_mode_setters["strategy"] = _strategy_paper_setter

            supervisor_ref = ctx.stream_supervisor
            if supervisor_ref is not None:

                def _stream_paper_getter(
                    supervisor: StreamSupervisor | None = supervisor_ref,
                ) -> bool:
                    if supervisor is None:
                        return True
                    with suppress(Exception):
                        return not bool(supervisor.is_running())
                    return True

                def _stream_paper_setter(
                    enabled: bool,
                    supervisor: StreamSupervisor | None = supervisor_ref,
                ) -> bool:
                    if supervisor is None:
                        return False
                    try:
                        if enabled:
                            supervisor.stop()
                            return True
                        return bool(supervisor.start())
                    except Exception as exc:  # noqa: BLE001 - defensive
                        LOGGER.warning(
                            "telegram_stream_paper_toggle_failed",
                            extra={
                                "event": "paper_toggle_failed",
                                "section": "stream",
                                "err": str(exc),
                            },
                        )
                        return False

                paper_mode_getters["stream"] = _stream_paper_getter
                paper_mode_setters["stream"] = _stream_paper_setter

            deps = TelegramDeps(
                token=str(telegram_cfg.bot_token),
                chat_id=int(telegram_cfg.chat_id),
                app_version=str(getattr(config, "version", "dev")),
                webhook_url=(
                    str(telegram_cfg.webhook_url)
                    if getattr(telegram_cfg, "webhook_url", None)
                    else None
                ),
                webhook_path=str(telegram_cfg.webhook_path),
                webhook_secret_token=telegram_cfg.webhook_secret_token,
                webhook_max_failures=int(telegram_cfg.webhook_max_failures),
                enable_polling_fallback=polling_enabled_flag,
                polling_interval_seconds=float(telegram_cfg.polling_interval_seconds),
                webhook_listen_host=str(telegram_cfg.webhook_listen_host),
                webhook_listen_port=int(telegram_cfg.webhook_listen_port),
                broker_client=ctx.broker_client,
                websocket_manager=ctx.websocket_manager,
                streamer=ctx.streamer,
                stream_supervisor=ctx.stream_supervisor,
                websocket_enabled=websocket_enabled,
                market_data_manager=ctx.market_data_manager,
                market_regime=ctx.market_regime,
                regime_manager=ctx.market_regime_manager,
                strategy_manager=ctx.strategy_manager,
                strategy_runner=ctx.strategy_runner,
                position_manager=ctx.position_manager,
                order_manager=ctx.order_manager,
                safe_order_manager=ctx.safe_order_manager,
                risk_manager=ctx.risk_manager,
                instrument_resolver=ctx.instrument_resolver,
                resolver=ctx.instrument_resolver,
                instrument_universe=ctx.instrument_universe,
                instrument_db_path=(
                    str(cache_settings.db_path) if cache_settings is not None else None
                ),
                instrument_csv_path=(
                    str(cache_settings.csv_path)
                    if (
                        cache_settings is not None
                        and cache_settings.csv_path is not None
                    )
                    else None
                ),
                metrics=None,
                session_guard=ctx.session_guard,
                rate_limiter=ctx.rate_limiter,
                get_ws_token=_resolve_ws_token,
                get_ws_token_issued_at=_ws_token_issued_at,
                ws_host=ws_host,
                set_shadow_mode=set_shadow,
                get_shadow_mode=get_shadow,
                paper_mode_getters=paper_mode_getters or None,
                paper_mode_setters=paper_mode_setters or None,
                data_hub=ctx.data_hub,
                unified_manager=unified_manager,
                reload_hook=None,
                telegram_plain=telegram_plain_flag,
                selfchecker=ctx.selfchecker,
            )
            telegram_bot_instance = TelegramBot(deps)
            telegram_chat_id = int(telegram_cfg.chat_id)
            LOGGER.info("Telegram configured for chat_id=%s", telegram_chat_id)

            if controller is not None and settings.notifications.enabled:
                try:
                    application = telegram_bot_instance.build_application(
                        bot=controller.bot
                    )
                except Exception as exc:  # noqa: BLE001 - defensive wiring
                    LOGGER.exception(
                        "telegram_application_build_failed",
                        extra={
                            "event": "telegram_application_build_failed",
                            "err": str(exc),
                        },
                    )
                else:
                    ctx.telegram_application = application
                    controller.attach_application(application)
                    LOGGER.info(
                        "telegram_application_attached",
                        extra={"event": "telegram_application_attached"},
                    )
                    version_info = {
                        "build": str(getattr(config, "version", "unknown")),
                        "sha": str(getattr(settings, "git_sha", "unknown")),
                    }
                    services_bundle = TelegramCommandServices(
                        order_manager=ctx.order_manager,
                        risk_manager=ctx.risk_manager,
                        market_data=ctx.market_data_manager,
                        strategy_runner=ctx.strategy_runner,
                        config=config,
                        broker=ctx.broker_client,
                        journal=None,
                        metrics=None,
                        market_regime=ctx.market_regime,
                        order_execution_hub=ctx.order_execution_hub,
                        order_queue=ctx.order_queue,
                        state_tracker=ctx.state_tracker,
                        preflight_validator=ctx.preflight_validator,
                        version_info=version_info,
                        allowed_chat_id=telegram_chat_id,
                    )
                    try:
                        register_telegram_commands(
                            telegram_bot_instance, application, services_bundle
                        )
                    except Exception as exc:  # noqa: BLE001 - defensive wiring
                        LOGGER.warning(
                            "telegram_command_registration_failed",
                            extra={
                                "event": "telegram_command_registration_failed",
                                "err": str(exc),
                            },
                        )
                    hook = getattr(
                        telegram_bot_instance, "after_application_built", None
                    )
                    if callable(hook):
                        result = hook()
                        if inspect.isawaitable(result):
                            awaitable = cast(Coroutine[Any, Any, object], result)
                            try:
                                loop = asyncio.get_running_loop()
                            except RuntimeError:
                                asyncio.run(awaitable)
                            else:
                                loop.create_task(awaitable)
            elif settings.notifications.enabled and controller is None:
                LOGGER.warning(
                    "telegram_application_controller_missing",
                    extra={"event": "telegram_application_controller_missing"},
                )
        else:
            LOGGER.info("Telegram disabled (no token/chat_id provided).")
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Telegram console disabled: %s", exc)

    if (
        settings.notifications.legacy_console_enabled
        and telegram_bot_instance is not None
        and telegram_chat_id is not None
    ):
        ctx.telegram_bot = telegram_bot_instance
        LOGGER.info("Telegram enabled for chat_id=%s", telegram_chat_id)

    return ctx


def _validate_config(config: AppConfig) -> None:
    if not config.broker.api_key or not config.broker.api_secret:
        raise ValueError("Broker credentials are required")
    if not config.broker.access_token:
        raise ValueError("Broker access token is required")
    if config.ratelimit.orders.capacity <= 0:
        raise ValueError("Order rate limit capacity must be positive")
    LOGGER.debug("Configuration validated successfully")
def force_enable_trading_override() -> str:
    """
    Emergency override to force enable trading by resetting all guards.
    Usage: Call from Telegram or REPL.
    """
    ctx = get_latest_bot_context()
    if not ctx:
        return "❌ No Bot Context found."

    logs = []
    
    # 1. Force Session Valid
    if ctx.session_guard:
        ctx.session_guard.mark_session_valid()
        ctx.session_guard.set_allow_out_of_hours(True)
        ctx.out_of_hours_override = True
        logs.append("✅ Session Guard Force-Validated (Out-of-hours allowed)")

    # 2. Reset Risk Breaker
    if ctx.risk_manager:
        ctx.risk_manager.reset_on_start(override=True)
        # Manually clear flags if needed
        if hasattr(ctx.risk_manager, "_breaker_tripped"):
            ctx.risk_manager._breaker_tripped = False
        logs.append("✅ Risk Manager Reset")

    # 3. Enable Live Orders
    if ctx.safe_order_manager:
        ctx.safe_order_manager.set_live_enabled(True)
        ctx.shadow_mode_enabled = False
        logs.append("✅ Live Trading Enabled (Shadow Mode OFF)")

    LOGGER.critical(f"🚨 MANUAL OVERRIDE ACTIVATED: {', '.join(logs)}")
    return "\n".join(logs)

async def startup_sequence(ctx: BotContext) -> None:
    """Execute startup sequence with Smart Hydration and Option-Only Trading."""

    LOGGER.info("Starting Nifty Scalper Bot...")

    # =========================================================
    # Create Data Directory
    # =========================================================
    import os
    try:
        os.makedirs("data", exist_ok=True)
        LOGGER.info("✅ Verified/Created 'data/' directory.")
    except Exception as e:
        LOGGER.critical(f"❌ Failed to create data directory: {e}")

    _validate_config(ctx.config)
    broker_ready = True
    guard = ctx.session_guard

    # ---------------------------------------------------------
    # Telegram notifier helper
    # ---------------------------------------------------------
    async def _notify(event: str, payload: Mapping[str, object] | None = None) -> None:
        notifier = ctx.telegram_notifier
        if notifier is None:
            return
        try:
            await notifier.send_event(event, payload)
        except Exception:
            LOGGER.debug("Startup notifier failed", exc_info=True)

    # ---------------------------------------------------------
    # 1. Broker validation
    # ---------------------------------------------------------
    try:
        broker_proxy = getattr(
            ctx.broker_client,
            "_broker",
            getattr(ctx.broker_client, "broker", ctx.broker_client),
        )
        get_profile_fn = getattr(broker_proxy, "get_profile", None)
        if callable(get_profile_fn):
            profile = await asyncio.to_thread(get_profile_fn)
            LOGGER.info(f"Connected to broker: {profile.get('user_name') or 'User'}")
            if guard:
                guard.mark_session_valid()
    except Exception as e:
        LOGGER.error(f"Broker connection failed: {e}")
        broker_ready = False

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 2. Load instruments AND sync to resolver
    # ---------------------------------------------------------
    if broker_ready:
        try:
            inner = getattr(
                ctx.broker_client,
                "broker",
                getattr(ctx.broker_client, "_broker", ctx.broker_client),
            )
            LOGGER.info("📦 Loading NSE instruments...")
            await asyncio.to_thread(inner.load_instruments, "NSE")
            LOGGER.info("✅ NSE instruments loaded")
            
            LOGGER.info("📦 Loading NFO instruments...")
            await asyncio.to_thread(inner.load_instruments, "NFO")
            LOGGER.info("✅ NFO instruments loaded")
            
            # ✅ CRITICAL FIX: Sync loaded instruments to InstrumentResolver
            # The resolver was warmed BEFORE instruments were loaded, so we must update it now
            if ctx.instrument_resolver:
                LOGGER.info("🔄 Syncing NFO instruments to InstrumentResolver...")
                synced_count = 0
                
                # Get instruments from broker cache
                broker_ref = getattr(ctx.broker_client, "_broker", ctx.broker_client)
                nfo_cache = getattr(broker_ref, "_instrument_cache", {}).get("NFO", {})
                
                if not nfo_cache:
                    # Fallback: try list_instruments
                    list_fn = getattr(broker_ref, "list_instruments", None)
                    if callable(list_fn):
                        all_instruments = list_fn()
                        nfo_cache = {
                            row.get("tradingsymbol", ""): row 
                            for row in all_instruments 
                            if row.get("exchange") == "NFO"
                        }
                
                total_items = len(nfo_cache)
                LOGGER.info(f"📊 NFO cache has {total_items} items to sync")
                
                for idx, (key, row) in enumerate(nfo_cache.items(), 1):
                    try:
                        token = row.get("instrument_token")
                        ts = row.get("tradingsymbol") or row.get("symbol")
                        exchange = row.get("exchange", "NFO")
                        
                        if token and ts:
                            # Add to resolver's internal caches
                            # Note: upsert now logs at DEBUG level (not INFO)
                            ctx.instrument_resolver.upsert(ts, int(token), exchange=exchange)
                            synced_count += 1
                        
                        # Progress logging every 1000 instruments to show activity
                        if idx % 1000 == 0:
                            LOGGER.info(f"📥 NFO sync progress: {idx}/{total_items}")
                            
                    except Exception as e:
                        # Log failures at debug level to avoid spam
                        LOGGER.debug(f"Skip NFO instrument {key}: {e}")
                
                LOGGER.info(f"✅ Synced {synced_count}/{total_items} NFO instruments to resolver")
                
                # Verify resolution now works
                test_sym = "NFO:NIFTY26JAN25200CE"
                test_tok = ctx.instrument_resolver.resolve(test_sym)
                if test_tok:
                    LOGGER.info(f"✅ NFO resolution test PASSED: {test_sym} -> {test_tok}")
                else:
                    LOGGER.error(f"🔴 NFO resolution test FAILED: {test_sym} -> None")
                    LOGGER.error("🔴 Trying alternative symbol formats...")
                    
                    # Try without NFO prefix
                    for alt_sym in ["NIFTY26JAN25200CE", "NIFTY2612725200CE"]:
                        alt_tok = ctx.instrument_resolver.resolve(alt_sym)
                        if alt_tok:
                            LOGGER.info(f"✅ Found with alternate format: {alt_sym} -> {alt_tok}")
                            break
                    
        except Exception as e:
            LOGGER.error(f"Instrument load failed: {e}", exc_info=True)
            
    # ---------------------------------------------------------
    # 3. Symbol resolution + HYDRATION (FIXED)
    # ---------------------------------------------------------
    if broker_ready:
        try:
            targets = _get_symbols(
                ctx.config,
                ctx.instrument_resolver,
                ctx.broker_client,
            )

            # ---------- Futures rollover logic (unchanged) ----------
            from datetime import datetime, timedelta
            import calendar

            now = datetime.now()
            year, month = now.year, now.month
            last_day = calendar.monthrange(year, month)[1]
            expiry_date = datetime(year, month, last_day)

            while expiry_date.weekday() != 1:  # Tuesday
                expiry_date -= timedelta(days=1)

            target_date = expiry_date + timedelta(days=7) if now.date() > expiry_date.date() else now
            y_str = target_date.strftime("%y")
            m_str = target_date.strftime("%b").upper()
            future_symbol = f"NFO:NIFTY{y_str}{m_str}FUT"

            if ctx.instrument_resolver:
                fut_token = ctx.instrument_resolver.resolve(future_symbol)
                if fut_token:
                    LOGGER.info(f"✅ Resolved Futures (Data Only): {future_symbol} -> {fut_token}")
                    targets.append(future_symbol)
                    if ctx.strategy_manager and hasattr(ctx.strategy_manager, "orchestrator"):
                        ctx.strategy_manager.orchestrator.futures_symbol = future_symbol
                else:
                    LOGGER.warning(f"⚠️ Could not resolve Futures: {future_symbol}")

            targets.append("NSE:NIFTY 50")
            targets = list(dict.fromkeys(targets))

            LOGGER.info(f"⏳ Hydrating {len(targets)} symbols: {targets}")

            # ---------- HISTORICAL HYDRATION (RATE-LIMIT SAFE FIX) ----------
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=5)
            from_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            to_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

            engine = ctx.market_regime_manager.indicators
            HYDRATION_DELAY_SEC = 1.3  # Zerodha-safe pacing

            for idx, sym in enumerate(targets, start=1):
                try:
                    LOGGER.info(f"📥 Hydration {idx}/{len(targets)}: {sym}")

                    records = await asyncio.to_thread(
                        ctx.broker_client.get_ohlc,
                        sym,
                        "minute",
                        from_str,
                        to_str,
                    )

                    if records:
                        count = 0
                        # 1. Get Reference to Runner
                        runner = ctx.strategy_runner

                        for c in records:
                            # 2. Robust Parsing
                            if isinstance(c, dict): 
                                ts = c.get("date")
                                o, h, l, c_p, v = c.get("open"), c.get("high"), c.get("low"), c.get("close"), c.get("volume", 0)
                            elif isinstance(c, list): 
                                ts, o, h, l, c_p, v = c[0], c[1], c[2], c[3], c[4], c[5]
                            else: 
                                continue
                            
                            # 3. Timestamp Normalization
                            if isinstance(ts, str):
                                try: ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                                except: pass

                            # 4. Feed The Runner (The Missing Link)
                            if runner and hasattr(runner, "ingest_historical_bar"):
                                bar_data = {
                                    "symbol": sym,
                                    "open": float(o),
                                    "high": float(h),
                                    "low": float(l),
                                    "close": float(c_p),
                                    "volume": float(v),
                                    "timestamp": ts
                                }
                                runner.ingest_historical_bar(bar_data)
                            
                            count += 1

                        LOGGER.info(f"✅ Hydrated {sym}: {count} bars")
                       
                    await asyncio.sleep(HYDRATION_DELAY_SEC)

                except Exception as e:
                    if "rate limit" in str(e).lower():
                        LOGGER.warning(f"⚠️ Rate limit hit for {sym}, skipping hydration")
                    else:
                        LOGGER.warning(f"❌ Failed to hydrate {sym}: {e}")

                    await asyncio.sleep(HYDRATION_DELAY_SEC)

            # =========================================================
            # ✅ WORLD CLASS FIX: FINALIZE RUNNER STATE
            # =========================================================
            # This commits the hydration so Runner doesn't trigger fallback backfill.
            runner = ctx.strategy_runner
            if runner:
                # 1. Collect symbols we attempted to hydrate
                hydrated_symbols = list(targets)
                
                # 2. Commit the state
                if hasattr(runner, "mark_ready"):
                    runner.mark_ready(hydrated_symbols)
            # =========================================================

            await ctx.market_regime_manager.refresh_from_indicators()
            # -------------------------------------------------
            # MARK INDICATORS AS WARM / READY
            # -------------------------------------------------
            ctx.market_regime_manager.indicators_ready = True
            LOGGER.info("🧠 Indicators fully hydrated and READY at startup")


            # ---------- Tracking / execution wiring (UNCHANGED) ----------
            mdm = ctx.market_data_manager
            streamer = ctx.streamer
            tokens_to_poll = []

            LOGGER.info(f"🔧 Processing {len(targets)} symbols for wiring...")
            resolved_count = 0
            unresolved_symbols = []

            for sym in targets:
                if mdm:
                    mdm.ensure_tracking(sym)

                tok = None
                if ctx.instrument_resolver:
                    tok = ctx.instrument_resolver.resolve(sym)
                    if tok:
                        tokens_to_poll.append(tok)
                        resolved_count += 1
                        LOGGER.info(f"✅ Resolved: {sym} -> token {tok}")
                    else:
                        unresolved_symbols.append(sym)
                        LOGGER.warning(f"⚠️ UNRESOLVED (no token): {sym}")

                ctx.strategy_runner.add_symbol(sym)

            LOGGER.info(f"📊 Resolution summary: {resolved_count}/{len(targets)} resolved")
            if unresolved_symbols:
                LOGGER.error(f"🔴 UNRESOLVED SYMBOLS (will NOT be polled): {unresolved_symbols}")
            LOGGER.info(f"✅ tokens_to_poll has {len(tokens_to_poll)} tokens")

            if streamer and hasattr(streamer, "subscribe") and tokens_to_poll:
                streamer.subscribe(tokens_to_poll)
                LOGGER.info(f"✅ Wired {len(tokens_to_poll)} tokens to PollingStreamer")

            # ✅ FIX: Disable MDM polling when PollingStreamer is active
            # MDM polling is redundant - PollingStreamer already feeds DataHub
            if mdm and not ctx.streamer:
                # Only start MDM polling if there's no streamer
                asyncio.create_task(asyncio.to_thread(mdm._rest_poll_loop))
            else:
                LOGGER.info("📡 MDM polling disabled (PollingStreamer is active)")
        except Exception as e:
            LOGGER.error("Hydration/Tracking failed", exc_info=True)

    # ---------------------------------------------------------
    # 4. Start subsystems (UNCHANGED)
    # ---------------------------------------------------------
    if broker_ready:
        try:
            if ctx.order_manager:
                ctx.order_manager.start_monitoring()
            if ctx.strategy_runner:
                ctx.strategy_runner.start()
            if ctx.stream_supervisor:
                ctx.stream_supervisor.start()
            elif hasattr(ctx.streamer, "start"):
                res = ctx.streamer.start()
                if inspect.isawaitable(res):
                    await res

            if ctx.telegram_bot:
                LOGGER.info("🚀 Starting Telegram Bot (Polling Mode)...")
                await ctx.telegram_bot.start()

            LOGGER.info("✅ All subsystems started.")
        except Exception as e:
            LOGGER.critical(f"Subsystem start failed: {e}")

    # ---------------------------------------------------------
    # 5. Kill switch + reconciliation (UNCHANGED)
    # ---------------------------------------------------------
    if broker_ready:
        try:
            orders = await asyncio.to_thread(ctx.broker_client.get_orders)
            for o in orders:
                if o.get("status") == "OPEN":
                    await asyncio.to_thread(
                        ctx.broker_client.cancel_order, o.get("order_id")
                    )
            LOGGER.info("✅ Zombie orders cleared.")

            async def _sync_loop():
                while True:
                    try:
                        await _reconcile_state(ctx)
                    except Exception:
                        pass
                    await asyncio.sleep(15)

            asyncio.create_task(_sync_loop())

        except Exception as e:
            LOGGER.error(f"Post-start tasks failed: {e}")

    # ---------------------------------------------------------
    # 6. Greeks monitoring (UNCHANGED, SAFE)
    # ---------------------------------------------------------
    if ctx.strategy_runner and hasattr(ctx.strategy_runner, "calculate_portfolio_greeks"):
        import threading, time as time_module

        runner = ctx.strategy_runner

        def _log_greeks_periodically():
            time_module.sleep(60)
            stop_event = getattr(ctx, "shutdown_event", None)

            while True:
                if stop_event and stop_event.is_set():
                    break
                if not threading.main_thread().is_alive():
                    break
                try:
                    greeks = runner.calculate_portfolio_greeks()
                    if abs(greeks.get("net_delta", 0)) > 1.0 or greeks.get("net_theta", 0) < -1.0:
                        LOGGER.info(
                            f"📊 Greeks: Delta={greeks['net_delta']:.1f} "
                            f"Theta={greeks['net_theta']:.1f}/day",
                            extra={"event": "greeks_monitor"},
                        )
                    for _ in range(300):
                        if stop_event and stop_event.is_set():
                            break
                        time_module.sleep(1)
                except Exception as exc:
                    LOGGER.debug(f"Greeks monitor error: {exc}")
                    time_module.sleep(60)

        threading.Thread(target=_log_greeks_periodically, daemon=True).start()
        LOGGER.info("✅ Portfolio Greeks monitoring enabled")

    await _notify("BOT_STARTED", {"mode": "LIVE" if not ctx.shadow_mode_enabled else "SHADOW"})

    # ----------------------------------------------------------------
    # ✅ FIX: Enable Persistence & Background Services
    # ----------------------------------------------------------------
    LOGGER.info("⚙️ Finalizing Startup: Restoring State & Services...")

    # 1. Restore Brackets from Disk (Critical for Restarts)
    if ctx.bracket_manager:
        try:
            # Run in thread to avoid blocking the event loop during file I/O
            await asyncio.to_thread(ctx.bracket_manager.load_state)
            stats = ctx.bracket_manager.get_stats()
            LOGGER.info(f"♻️ Restored virtual brackets: {stats}")
        except Exception as e:
            LOGGER.error(f"Failed to restore brackets: {e}")

    # 2. Start the ATR Feed Task (Fire and Forget)
    loop = asyncio.get_running_loop()
    loop.create_task(_run_atr_feed_task(ctx))

    # 3. Reconcile Open Orders (Sync with Broker)
    if ctx.order_manager:
        LOGGER.info("🔍 Running Startup Reconciliation...")
        try:
            await asyncio.to_thread(ctx.order_manager.reconcile_open_orders)
        except Exception as e:
            LOGGER.error(f"Reconciliation failed: {e}")

    # ----------------------------------------------------------------
    # ✅ FIX: Wire DataHub -> Bracket Manager (Corrected Attribute Name)
    # ----------------------------------------------------------------
    # CHANGE: ctx.market_data -> ctx.market_data_manager
    if ctx.bracket_manager and ctx.market_data_manager: 
        try:
            # 1. Define the tick handler using the fully loaded 'ctx'
            def _feed_ticks_to_bracket_safe(sym, tick):
                # Ensure we have LTP
                ltp = tick.get("ltp")
                if ltp and ctx.bracket_manager:
                    ctx.bracket_manager.on_tick(sym, ltp)

            # 2. Subscribe to the DataHub
            # CHANGE: ctx.market_data -> ctx.market_data_manager
            if hasattr(ctx.market_data_manager, "data_hub") and ctx.market_data_manager.data_hub:
                ctx.market_data_manager.data_hub.subscribe("bracket_feed", _feed_ticks_to_bracket_safe)
                LOGGER.info("✅ Wired DataHub ticks to BracketManager")
                
        except Exception as e:
            LOGGER.error(f"Failed to wire bracket ticks: {e}")
            
    LOGGER.info("✅ Startup sequence fully complete.")


async def shutdown_sequence(ctx: BotContext, *, reason: str = "shutdown") -> None:
    """Execute graceful shutdown."""

    LOGGER.info("Shutting down bot...")
    hub = getattr(ctx, "order_execution_hub", None)
    bus = getattr(ctx, "message_bus", None)
    proc = getattr(ctx, "order_processor", None)
    if proc is not None:
        with suppress(Exception):
            await proc.stop()
    if bus is not None:
        with suppress(Exception):
            await bus.stop()

    if hub is not None:
        with suppress(Exception):
            await hub.shutdown()

    refresh_task = getattr(ctx, "instrument_refresh_task", None)
    if refresh_task is not None:
        with suppress(Exception):
            refresh_task.cancel()
        ctx.instrument_refresh_task = None

    strategy_runner = _require_component(ctx.strategy_runner, "strategy_runner")
    order_manager_component = _require_component(ctx.order_manager, "order_manager")
    market_data_manager_component = _require_component(
        ctx.market_data_manager,
        "market_data_manager",
    )
    position_manager_component = _require_component(
        ctx.position_manager,
        "position_manager",
    )
    persistent_state_component = _require_component(
        ctx.persistent_state,
        "persistent_state",
    )

    hub = getattr(ctx, "order_execution_hub", None)
    if hub is not None:
        with suppress(Exception):
            await hub.shutdown()

    with suppress(Exception):
        strategy_runner.pause_trading()

    if getattr(ctx.config, "close_positions_on_shutdown", False):
        with suppress(Exception):
            _close_all_positions(ctx, reason=reason)

    pending = []
    with suppress(Exception):
        pending = list(position_manager_component.get_pending_orders())
    for order in pending:
        with suppress(Exception):
            order_manager_component.cancel_order(order.order_id)

    with suppress(Exception):
        strategy_runner.stop()
    with suppress(Exception):
        order_manager_component.stop_monitoring()
    with suppress(Exception):
        market_data_manager_component.stop()
    with suppress(Exception):
        supervisor = getattr(ctx, "stream_supervisor", None)
        if supervisor is not None:
            supervisor.stop()
        else:
            stop_callable = getattr(ctx.streamer, "stop", None)
            if callable(stop_callable):
                result = stop_callable()
                if inspect.isawaitable(result):
                    await result
    with suppress(Exception):
        position_manager_component.save_state()
    with suppress(Exception):
        persistent_state_component.flush()
    with suppress(Exception):
        persistent_state_component.close()
    instrument_db = getattr(ctx, "instrument_db", None)
    if instrument_db is not None:
        with suppress(Exception):
            instrument_db.close()
        ctx.instrument_db = None
    tracker = getattr(ctx, "state_tracker", None)
    if tracker is not None:
        with suppress(Exception):
            tracker.close()

    LOGGER.info("Bot shutdown complete")

async def _reconcile_state(ctx: BotContext) -> None:
    """
    Syncs local state with Broker (Orders & Positions).
    Features: Non-Blocking Execution, Position Sync, and Auto-Guarding of Orphans.
    """
    # 1. SYNC ORDERS (Non-Blocking Thread)
    if ctx.order_manager:
        try:
            await asyncio.to_thread(ctx.order_manager.reconcile_open_orders_with_broker)
        except Exception as exc:
            LOGGER.debug(f"Order Reconcile Warning: {exc}")

    # 2. SYNC POSITIONS & AUTO-GUARD ORPHANS
    if ctx.position_manager:
        try:
            # A. Fetch Broker Positions (REQUIRED STEP)
            raw_data = await ctx.broker_client.get_positions()
            broker_positions = []
            
            # Robust Parsing of Broker Data
            if isinstance(raw_data, list):
                broker_positions = [p for p in raw_data if isinstance(p, Mapping)]
            elif isinstance(raw_data, Mapping):
                # Handle 'net' or 'day' keys common in broker APIs
                src = raw_data.get("net", raw_data)
                if isinstance(src, list):
                    broker_positions = [p for p in src if isinstance(p, Mapping)]
                else:
                    broker_positions = [src]

            # B. Sync Broker Positions (Pass data to Manager)
            # We run this in a thread to keep the bot responsive
            await asyncio.to_thread(ctx.position_manager.synchronize_with_broker, broker_positions)

            # C. Auto-Guard Orphans (CRITICAL SAFETY LOGIC)
            if ctx.order_manager and ctx.order_manager._bracket_manager:
                om = ctx.order_manager
                bm = ctx.order_manager._bracket_manager
                
                from nifty_scalper_bot.data.data_hub import DataHub

                # Iterate through the FRESHLY synced open positions
                for pos in ctx.position_manager.get_open_positions():
                    if pos.quantity == 0: 
                        continue

                    # 1. Normalize Symbol
                    raw_symbol = pos.symbol
                    norm_symbol = DataHub.normalize(raw_symbol) or raw_symbol
                    
                    # 2. Check if this symbol is actively managed
                    is_managed = bm.is_symbol_managed(norm_symbol)
                    
                    # 3. If NOT managed, it is an Orphan -> Guard it!
                    if not is_managed:
                        LOGGER.warning(
                            f"⚠️ ORPHAN DETECTED: {norm_symbol} (Qty: {pos.quantity}). Auto-Guarding...",
                            extra={"event": "orphan_detected", "symbol": norm_symbol}
                        )
                        
                        avg_price = float(
                            getattr(pos, "average_price", 0.0) or 
                            getattr(pos, "buy_price", 0.0) or 
                            getattr(pos, "last_price", 0.0) or 
                            0.0
                        )

                        # Call the Master Guard Method
                        om.guard_orphan_position(
                            symbol=norm_symbol,
                            quantity=pos.quantity,
                            average_price=avg_price
                        )

            # =================================================================
            # ✅ D. CLEANUP GHOST BRACKETS (Safety Cleanup)
            # =================================================================
            # If a Bracket exists but the Position is gone (Manual Exit), kill the Bracket.
            # This prevents the bot from opening unwanted positions if price hits old levels.
            if bm and ctx.position_manager:
                # 1. Get symbols that currently have active brackets
                # Accessing protected member safely for reconciliation
                if hasattr(bm, "_symbol_map"):
                    managed_symbols = set(bm._symbol_map.keys())
                    
                    # 2. Get symbols that actually have open positions (Real Broker State)
                    real_positions = {
                        p.symbol for p in ctx.position_manager.get_open_positions() 
                        if p.quantity != 0
                    }
                    
                    # 3. Identify Ghosts (Managed but no Position)
                    ghosts = managed_symbols - real_positions
                    
                    for ghost_sym in ghosts:
                        # Double check if it actually has active brackets inside
                        if bm.is_symbol_managed(ghost_sym):
                            LOGGER.warning(
                                f"👻 GHOST BRACKET DETECTED: {ghost_sym} has protection but no Open Position. "
                                "Performing Safety Cleanup..."
                            )
                            # Force kill the bracket so it doesn't misfire and open a new trade
                            bm.manual_override_close(ghost_sym, reason="State Reconciliation (Ghost)")

        except Exception as exc:
            LOGGER.error(f"Position Sync/Adoption Failed: {exc}", exc_info=True)

    # 3. SITUATION REPORT
    if ctx.order_manager:
        ctx.order_manager._log_status_report()

def _close_all_positions(ctx: BotContext, *, reason: str) -> None:
    position_manager = _require_component(ctx.position_manager, "position_manager")
    market_data_manager = _require_component(
        ctx.market_data_manager,
        "market_data_manager",
    )
    for position in position_manager.get_all_positions():
        LOGGER.info("Closing position for %s", position.symbol)
        tick = market_data_manager.get_latest_tick(position.symbol)
        exit_price = position.current_price
        if tick is not None:
            maybe_price = tick.get("ltp") or tick.get("price")
            if isinstance(maybe_price, (int, float)) and maybe_price > 0:
                exit_price = float(maybe_price)
        position_manager.close_position(position.symbol, exit_price, reason)


def _alert_overnight_exposure(ctx: BotContext) -> None:
    """Emit an alert when open positions persist beyond the session.

    Args:
        ctx: Active bot context containing managers and configuration.

    Returns:
        None.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered overnight exposure check",
        extra={"event": "overnight_exposure_check"},
    )
    position_manager = _require_component(ctx.position_manager, "position_manager")
    runner = _require_component(ctx.strategy_runner, "strategy_runner")
    try:
        positions = position_manager.get_all_positions()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _alert_overnight_exposure positions",
            extra={"error": str(exc)},
            exc_info=True
        )
        return
    if not positions:
        setattr(runner, "_overnight_alerted", False)
        return
    session_close = getattr(ctx.settings.regime, "session_close", None)
    if session_close is None:
        setattr(runner, "_overnight_alerted", False)
        return
    session_guard = ctx.session_guard
    tz = getattr(session_guard, "_tz", ZoneInfo("Asia/Kolkata"))
    if not isinstance(tz, ZoneInfo):
        tz = ZoneInfo("Asia/Kolkata")
    now_local = datetime.now(tz)
    close_today = datetime.combine(now_local.date(), session_close, tzinfo=tz)
    overnight_symbols: set[str] = set()
    exposure_details: list[dict[str, object]] = []
    for position in positions:
        symbol = getattr(position, "symbol", "UNKNOWN") or "UNKNOWN"
        entry_time = getattr(position, "entry_time", None)
        entry_local: datetime | None = None
        if isinstance(entry_time, datetime):
            coerced = entry_time
            if coerced.tzinfo is None:
                coerced = coerced.replace(tzinfo=timezone.utc)
            entry_local = coerced.astimezone(tz)
        is_overnight = False
        if entry_local is not None:
            if entry_local.date() < now_local.date():
                is_overnight = True
            elif now_local > close_today:
                is_overnight = True
        elif now_local > close_today:
            is_overnight = True
        if not is_overnight:
            continue
        overnight_symbols.add(symbol)
        if entry_local is not None:
            age_minutes = max((now_local - entry_local).total_seconds() / 60.0, 0.0)
            exposure_details.append(
                {
                    "symbol": symbol,
                    "entry_time": entry_local.isoformat(),
                    "age_minutes": round(age_minutes, 2),
                }
            )
        else:
            exposure_details.append(
                {
                    "symbol": symbol,
                    "entry_time": None,
                    "age_minutes": None,
                }
            )
    if not overnight_symbols:
        setattr(runner, "_overnight_alerted", False)
        return
    if getattr(runner, "_overnight_alerted", False):
        return
    setattr(runner, "_overnight_alerted", True)
    LOGGER.error(
        "Condition met: overnight_exposure_detected",
        extra={
            "event": "overnight_exposure_detected",
            "symbols": sorted(overnight_symbols),
            "details": exposure_details,
        },
    )


def _health_check(ctx: BotContext) -> None:
    strategy_runner = _require_component(ctx.strategy_runner, "strategy_runner")
    status: Mapping[str, Any] = strategy_runner.get_status()
    if not bool(status.get("running")):
        LOGGER.warning("Strategy runner is not active")


def _must_ok(condition: bool, message: str) -> None:
    """Raise :class:`ConfigurationError` when *condition* is falsy."""

    if not condition:
        raise ConfigurationError(message)


class NiftyScalperApp:
    """High level orchestrator exposing lifecycle hooks for the trading stack."""

    def __init__(
        self, config: AppConfig | None = None, settings: Settings | None = None
    ) -> None:
        base_settings = settings or get_settings()
        if config is not None:
            base_settings = replace(base_settings, app=config)
        self._settings = base_settings
        self._config = base_settings.app
        setup_logging(self._config.logging.level)
        setup_structured_logging(self._config.logging.level)
        validation_errors = validate_execution_config()
        if validation_errors:
            joined = "; ".join(validation_errors)
            LOGGER.error(
                "Failure in NiftyScalperApp.__init__: config validation failed",
                extra={
                    "event": "config_validation_failure",
                    "errors": validation_errors,
                },
            )
            raise ConfigurationError(f"Execution configuration invalid: {joined}")
        self._ctx = initialize_components(self._settings)
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._health_task: asyncio.Task[None] | None = None
        self._self_test_task: asyncio.Task[None] | None = None
        self._telegram_task: asyncio.Task[None] | None = None
        self._telegram_application_started = False
        self._self_test_interval = 300.0

    @property
    def config(self) -> AppConfig:
        """Return the loaded configuration."""

        return self._config

    @property
    def settings(self) -> Settings:
        """Return runtime settings including live trading toggles."""

        return self._settings

    @property
    def health_app(self) -> FastAPI:
        """Expose FastAPI app serving /health and /metrics."""

        return _require_component(self._ctx.health_app, "health_app")

    @property
    def ws_manager(self) -> WebSocketManager | None:
        """Return the websocket manager if configured."""

        return self._ctx.websocket_manager

    @property
    def positions(self) -> PositionManager:
        """Return the position manager."""

        return _require_component(self._ctx.position_manager, "position_manager")

    def status_string(self) -> str:
        """Return a human friendly multi-line status string."""

        strategy_runner = _require_component(
            self._ctx.strategy_runner,
            "strategy_runner",
        )
        status = strategy_runner.get_status()
        running = "running" if status.get("running") else "stopped"
        if status.get("trading_paused"):
            running += " (paused)"
        active_symbols = status.get("active_symbols") or []
        symbol_line = ", ".join(active_symbols) if active_symbols else "none"

        position_manager = _require_component(
            self._ctx.position_manager,
            "position_manager",
        )
        positions = position_manager.get_all_positions()
        if not positions:
            position_lines = ["Positions: none"]
        else:
            summary = [
                (
                    f"{pos.symbol} {pos.side} qty={pos.quantity} "
                    f"pnl={pos.unrealized_pnl:.2f}"
                )
                for pos in positions[:5]
            ]
            more = len(positions) - len(summary)
            if more > 0:
                summary.append(f"(+{more} more)")
            position_lines = ["Positions:"] + summary

        lines = [
            "Nifty Scalper Bot",
            f"Core status: {running}",
            f"Active symbols: {symbol_line}",
            *position_lines,
        ]
        return "\n".join(lines)

    def simulate_disconnect(self) -> None:
        """Test helper forcing websocket reconnect."""

        streamer = getattr(self._ctx, "streamer", None)
        simulate = getattr(streamer, "simulate_disconnect", None)
        if callable(simulate):  # pragma: no branch - optional hook
            simulate()

    def is_connected(self) -> bool:
        """Return websocket connectivity state."""

        streamer = getattr(self._ctx, "streamer", None)
        if streamer is None:
            return True
        is_connected = getattr(streamer, "is_connected", None)
        if callable(is_connected):
            try:
                return bool(is_connected())
            except Exception:  # pragma: no cover - defensive
                return False
        return True

    def backlog_size(self) -> int:
        """Return queued tick backlog size."""

        streamer = getattr(self._ctx, "streamer", None)
        if streamer is None:
            return 0
        backlog_fn = getattr(streamer, "backlog_size", None)
        if callable(backlog_fn):
            try:
                return int(backlog_fn())
            except Exception:  # pragma: no cover - defensive
                return 0
        tracked = getattr(streamer, "tracked_tokens", None)
        if callable(tracked):
            try:
                return len(tracked())
            except Exception:  # pragma: no cover - defensive
                return 0
        return 0

    def rejection_count(self) -> int:
        """Return accumulated order rejection count."""

        safe_order_manager = _require_component(
            self._ctx.safe_order_manager,
            "safe_order_manager",
        )
        return safe_order_manager.rejection_count()

    async def start(self) -> None:
        """Start the trading stack and background health monitoring."""

        if self._running:
            LOGGER.info("NiftyScalperApp.start() ignored; already running")
            return
        await startup_sequence(self._ctx)
        self._running = True
        self._shutdown_event.clear()
        self._health_task = asyncio.create_task(
            self._health_loop(), name="core-health-monitor"
        )
        if self._ctx.selfchecker is not None:
            self._self_test_task = asyncio.create_task(
                self._self_test_loop(),
                name="core-runtime-selftest",
            )
        # ------------------------------------------------------------------
        # Telegram Service Initialization (Webhook vs Polling)
        # ------------------------------------------------------------------
        application = self._ctx.telegram_application
        controller = _HTTP_CONTROLLER
        
        # 1. Telegram App
        if application is not None:
            # Check if Webhook is actually configured
            if self.settings.notifications.webhook_enabled and self.settings.notifications.public_base_url:
                try:
                    await application.initialize()
                    await application.start()
                    if controller:
                        controller.notify_application_ready()
                    self._telegram_application_started = True
                    LOGGER.info("telegram_application_started (Webhook)")
                except Exception as exc:
                    LOGGER.exception("telegram_application_start_failed")
            
            # ✅ FIX: Fallback to Polling if Webhook is NOT enabled
            elif self._ctx.telegram_bot:
                LOGGER.info("🚀 Starting Telegram Polling (Background)...")
                await self._ctx.telegram_bot.start()

    async def stop(self) -> None:
        """Stop the trading stack gracefully."""

        if not self._running:
            return
        self._shutdown_event.set()
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None
        if self._self_test_task:
            self._self_test_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._self_test_task
            self._self_test_task = None
        if (
            self._ctx.telegram_application is not None
            and self._telegram_application_started
        ):
            controller = _HTTP_CONTROLLER
            if controller is not None:
                controller.notify_application_ready(ready=False)
            with suppress(Exception):
                await self._ctx.telegram_application.stop()
            with suppress(Exception):
                await self._ctx.telegram_application.shutdown()
            self._telegram_application_started = False
        if self._telegram_task and self._ctx.telegram_bot is not None:
            await self._ctx.telegram_bot.shutdown()
            self._telegram_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._telegram_task
            self._telegram_task = None
        await shutdown_sequence(self._ctx)
        self._running = False

    def close_all_positions(self, *, reason: str) -> None:
        """Close all known positions immediately."""

        _close_all_positions(self._ctx, reason=reason)

    async def _self_test_loop(self) -> None:
        """Execute periodic runtime self-checks and alert on failures.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        checker = self._ctx.selfchecker
        if checker is None:
            LOGGER.info(
                "Runtime self-test loop skipped: checker unavailable.",
                extra={"event": "runtime_self_test_missing"},
            )
            return
        interval = getattr(checker, "interval_seconds", self._self_test_interval)
        try:
            interval_value = max(float(interval), 60.0)
        except Exception:  # pragma: no cover - defensive parsing
            interval_value = self._self_test_interval
        LOGGER.debug(
            "Entered runtime self-test loop",
            extra={
                "event": "runtime_self_test_loop_enter",
                "interval": interval_value,
            },
        )
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=interval_value
                )
            except asyncio.TimeoutError:
                LOGGER.debug(
                    "Executing runtime self-test iteration",
                    extra={"event": "runtime_self_test_iteration"},
                )
                try:
                    results = checker.run_full_check()
                except Exception as exc:  # noqa: BLE001 - defensive
                    LOGGER.error(
                        "Failure in runtime self-test execution: %s",
                        exc,
                        extra={"event": "runtime_self_test_execute_error"},
                        exc_info=exc,
                    )
                    continue
                for name, result in results.items():
                    if not bool(result.get("ok")):
                        detail = str(result.get("detail", "unknown"))
                        meta_obj = result.get("meta")
                        LOGGER.error(
                            "Silent failure detected: %s check failed: %s",
                            name,
                            detail,
                            extra={
                                "event": "runtime_self_test_failure",
                                "check": name,
                                "detail": detail,
                                "meta": meta_obj,
                            },
                        )
                        meta_payload = (
                            meta_obj if isinstance(meta_obj, Mapping) else None
                        )
                        await self._send_self_test_alert(
                            name,
                            detail,
                            cast(Mapping[str, object] | None, meta_payload),
                        )
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - defensive loop guard
                LOGGER.error(
                    "Failure in runtime self-test loop: %s",
                    exc,
                    extra={"event": "runtime_self_test_loop_error"},
                    exc_info=exc,
                )
                await asyncio.sleep(5.0)
        LOGGER.debug(
            "Runtime self-test loop exiting",
            extra={"event": "runtime_self_test_loop_exit"},
        )

    async def _send_self_test_alert(
        self,
        check_name: str,
        detail: str,
        meta: Mapping[str, object] | None,
    ) -> None:
        """Send Telegram notifier alert for runtime self-test failures.

        Args:
            check_name: Identifier of the failing runtime check.
            detail: Description or reason for the failure.
            meta: Optional metadata describing the failure context.

        Returns:
            None.

        Raises:
            None.
        """

        notifier = self._ctx.telegram_notifier
        payload_meta: dict[str, object]
        if isinstance(meta, Mapping):
            payload_meta = dict(meta)
        else:
            payload_meta = {}
        payload = {
            "check": check_name,
            "detail": detail,
            "meta": payload_meta,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if notifier is None:
            LOGGER.info(
                "Runtime self-test alert skipped: notifier unavailable",
                extra={
                    "event": "runtime_self_test_alert_skipped",
                    "check": check_name,
                    "detail": detail,
                },
            )
            return
        try:
            await notifier.send_event("SILENT_FAILURE", payload)
        except Exception as exc:  # noqa: BLE001 - defensive notifier surface
            LOGGER.error(
                "Failure in runtime self-test alert send: %s",
                exc,
                extra={
                    "event": "runtime_self_test_alert_error",
                    "check": check_name,
                },
                exc_info=exc,
            )
        else:
            LOGGER.info(
                "Condition met: runtime self-test alert dispatched",
                extra={
                    "event": "runtime_self_test_alert_sent",
                    "check": check_name,
                },
            )

    async def _health_loop(self) -> None:
        interval = 60.0
        last_heavy = time_module.monotonic()
        heavy_interval = 60.0
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                now = time_module.monotonic()
                if now - last_heavy >= heavy_interval:
                    _health_check(self._ctx)
                    try:
                        await _reconcile_state(self._ctx)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning(
                            "Periodic state reconciliation failed",
                            extra={"event": "state_reconcile_failed_periodic", "error": str(exc)},
                            exc_info=True
                        )
                    try:
                        _alert_overnight_exposure(self._ctx)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning(
                            "Overnight exposure check failed",
                            extra={"event": "overnight_exposure_check_failed", "error": str(exc)},
                            exc_info=True
                        )
                    last_heavy = now
                continue
            break

# ----------------------------------------------------------------
# ✅ NEW HELPER: Background ATR Feed
# ----------------------------------------------------------------
async def _run_atr_feed_task(ctx: BotContext) -> None:
    """Periodically pushes fresh ATR data to the Bracket Manager."""
    LOGGER.info("🚀 Starting ATR Feed to BracketManager...")
    while True:
        try:
            if ctx.bracket_manager and ctx.indicator_engine:
                # Access protected member safely for internal core logic
                active_symbols = list(ctx.bracket_manager._symbol_map.keys())
                for symbol in active_symbols:
                    atr = ctx.indicator_engine.compute_atr(symbol, period=14)
                    if atr and atr > 0:
                        ctx.bracket_manager.update_market_stats(symbol, atr=atr)
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(5)


__all__ = [
    "NiftyScalperApp",
    "initialize_components",
    "startup_sequence",
    "shutdown_sequence",
    "get_http_app",
    "get_telegram_notifier",
    "get_nifty_expiry",
]

# Deployment Fix: Force Update 2026-01-11
