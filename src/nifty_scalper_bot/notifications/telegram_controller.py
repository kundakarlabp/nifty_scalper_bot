"""Telegram operator console: single-chat, production-grade commands.

Commands (core):
  /opshelp               – show commands
  /ping                  – round-trip latency
  /whoami                – broker identity
  /version               – bot version & build info
  /status                – overall health (core/ws/mdm/strategies)
  /session               – session guard state & overrides
  /mdm                   – market-data manager stats
  /ws_status             – websocket state, queue depth & heartbeat
  /ws_reconnect          – refresh session then reconnect
  /resolve <SYMBOL>      – instrument metadata & token resolution
  /broker_quote <SYMBOL> – inspect broker quote candidates
  /tick <SYMBOL|TOKEN>   – last cached tick
  /subscribe <TOKEN(s)>  – subscribe tokens (comma/space separated)
  /unsubscribe <TOKEN(s)> – unsubscribe tokens
  /quote <SYMBOL>        – REST quote (and cache update)
  /strategies            – list strategies & tracked symbols
  /regime                – latest market regime snapshot & guidance
  /positions             – open positions snapshot
  /orders                – recent orders snapshot
  /risk                  – current risk config snapshot
  /selftest              – run freshness + risk diagnostics (JSON summary)
  /assess <area>         – targeted checks (data|risk|orders|broker)
  /shadow [on|off]       – toggle shadow (dry-run) mode (if supported)
  /paper [section] [mode] – inspect or toggle paper mode per subsystem
  /errors                – last N errors from in-memory ring buffer

Optional SRE-grade:
  /config                – sanitized runtime config (secrets redacted)
  /env                   – important env toggles
  /metrics [filter]      – summarize top counters/gauges
  /ws_diag               – websocket diagnostics (versions, DNS, TCP)
  /gc                    – trigger GC and show deltas
  /profile [sec]         – cProfile for N seconds, return text top calls
  /dumpLogs [N]          – return last N log lines as a file
  /reload                – hot-reload strategies/config (if available)
                           [safe noop otherwise]
  /backtest <SYMBOL> [D] – quick offline backtest hook (noop stub unless implemented)
"""

# ruff: noqa: I001

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import gc
import html as _html
import importlib
import io
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time
import tracemalloc
import typing as t
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
import uvicorn

from nifty_scalper_bot.backtesting.backtest_engine import (
    BacktestSummary,
    run_quick_backtest,
)
from nifty_scalper_bot.core.market_regime import MarketRegimeDetector, RegimeSnapshot
from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager
from nifty_scalper_bot.core.performance_metrics import build_performance_snapshot
from nifty_scalper_bot.core.trading_switch import trading_switch
from nifty_scalper_bot.utils.pricing import canonical_price_source
from nifty_scalper_bot.data import (
    InstrumentUniverseStatus,
    ensure_sqlite,
    load_rows_for_resolver,
    refresh_from_csv,
)
from nifty_scalper_bot.data.assess_data import assess_datahub_fresh
from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.diagnostics.assess import assess_suite
from nifty_scalper_bot.infra.diagnostics import LOG_TAP
from nifty_scalper_bot.infra.metrics import METRICS, MetricsCollector
from nifty_scalper_bot.infra.ws_diag import WsDiag, run_diag
from nifty_scalper_bot.notifications.safe_messenger import SafeMessenger
from nifty_scalper_bot.utils.alerts import (
    AggregatedAlert,
    AlertDeduplicator,
    AlertLogHandler,
)
from nifty_scalper_bot.utils.env import coalesce_bool, get_bool
from nifty_scalper_bot.utils.errors import OrderPlacementError
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.reasons import SOFT, canonical
from nifty_scalper_bot.utils.response_builder import EMOJI, RB, ResponseBuilder
from telegram import Bot, Chat, InputFile, Message, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, NetworkError, RetryAfter, TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest  # ✅ NEW IMPORT

F = t.TypeVar("F", bound=t.Callable[..., t.Any])


if t.TYPE_CHECKING:
    from nifty_scalper_bot.core.unified_manager import UMPlan, UnifiedManager


def command_meta(cmd_usage: str, help_text: str) -> t.Callable[[F], F]:
    """Attach Telegram help metadata to a command handler."""

    def decorator(func: F) -> F:
        setattr(func, "__cmd__", cmd_usage)
        setattr(func, "__help__", help_text)
        return func

    return decorator


def register_regime_commands(bot_context: t.Any, application: Application) -> None:
    """Register lightweight market regime inspection commands.

    Args:
        bot_context: Container exposing ``market_regime_manager``.
        application: Telegram application used for handler registration.

    Returns:
        None.

    Raises:
        None.
    """

    log.debug(
        "Entered register_regime_commands",
        extra={"event": "telegram_register_regime_commands"},
    )

    async def regime_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send the current market regime and confidence to the chat.

        Args:
            update: Telegram update payload containing the chat message.
            context: Telegram context payload (unused).

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered register_regime_commands.regime_cmd",
            extra={"event": "telegram_regime_simple_enter"},
        )
        message = getattr(update, "message", None)
        if message is None:
            return
        try:
            manager = getattr(bot_context, "market_regime_manager", None)
            if manager is None:
                await message.reply_text("Regime manager unavailable.")
                return
            regime = manager.get_current_regime()
            confidence = manager.get_regime_confidence()
            await message.reply_text(
                f"Current Regime: {regime or 'UNKNOWN'}\nConfidence: {confidence:.2f}"
            )
            log.info(
                "Condition met: telegram_regime_simple_sent",
                extra={"event": "telegram_regime_simple_sent"},
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in regime_cmd: %s",
                exc,
                extra={"event": "telegram_regime_simple_error"},
            )
            try:
                await message.reply_text("Unable to fetch regime right now.")
            except Exception as notify_exc:  # pragma: no cover - defensive
                log.error(
                    "Failure sending regime error notice: %s",
                    notify_exc,
                    extra={"event": "telegram_regime_simple_notify_error"},
                )

    async def bypass_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Toggle the regime bypass flag and report the new state.

        Args:
            update: Telegram update payload containing the chat message.
            context: Telegram context payload (unused).

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered register_regime_commands.bypass_cmd",
            extra={"event": "telegram_regime_bypass_enter"},
        )
        message = getattr(update, "message", None)
        if message is None:
            return
        try:
            manager = getattr(bot_context, "market_regime_manager", None)
            if manager is None:
                await message.reply_text("Regime manager unavailable.")
                return
            state = manager.toggle_bypass()
            await message.reply_text(f"Regime filter bypass set to {state}")
            log.info(
                "Condition met: telegram_regime_bypass_toggled",
                extra={"event": "telegram_regime_bypass_toggled", "state": state},
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in bypass_cmd: %s",
                exc,
                extra={"event": "telegram_regime_bypass_error"},
            )
            try:
                await message.reply_text("Unable to toggle bypass.")
            except Exception as notify_exc:  # pragma: no cover - defensive
                log.error(
                    "Failure sending regime bypass error notice: %s",
                    notify_exc,
                    extra={"event": "telegram_regime_bypass_notify_error"},
                )

    async def diag_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Return a short diagnostic summary of recent regime changes.

        Args:
            update: Telegram update payload containing the chat message.
            context: Telegram context payload (unused).

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered register_regime_commands.diag_cmd",
            extra={"event": "telegram_regime_diag_enter"},
        )
        message = getattr(update, "message", None)
        if message is None:
            return
        try:
            manager = getattr(bot_context, "market_regime_manager", None)
            if manager is None:
                await message.reply_text("Regime manager unavailable.")
                return
            history = manager.get_history(limit=10)
            lines: list[str] = []
            for snapshot in history:
                timestamp = time.strftime("%H:%M", time.localtime(snapshot.updated_at))
                lines.append(
                    f"{timestamp} {snapshot.regime}"
                    f" (conf={float(snapshot.confidence):.2f})"
                )
            lines.append(f"Bypass: {manager.get_regime_filter_bypass()}")
            await message.reply_text(
                "\n".join(lines) if lines else "No history available."
            )
            log.info(
                "Condition met: telegram_regime_diag_sent",
                extra={"event": "telegram_regime_diag_sent", "entries": len(history)},
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in diag_cmd: %s",
                exc,
                extra={"event": "telegram_regime_diag_error"},
            )
            try:
                await message.reply_text("Unable to fetch regime diagnostics.")
            except Exception as notify_exc:  # pragma: no cover - defensive
                log.error(
                    "Failure sending regime diagnostics error notice: %s",
                    notify_exc,
                    extra={"event": "telegram_regime_diag_notify_error"},
                )

    try:
        application.add_handler(CommandHandler("regime", regime_cmd))
        application.add_handler(CommandHandler("regimebypass", bypass_cmd))
        application.add_handler(CommandHandler("regimediag", diag_cmd))
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Failure in register_regime_commands: %s",
            exc,
            extra={"event": "telegram_register_regime_error"},
        )


# --- Optional rate limiter support (the PTB extra is not always installed) ---
_AIORateLimiter: t.Any
try:  # requires: pip install "python-telegram-bot[rate-limiter]"
    _ptb_ext = importlib.import_module("telegram.ext")
    _AIORateLimiter = getattr(_ptb_ext, "AIORateLimiter")
    _HAS_RATE_LIMITER = True
except Exception:  # pragma: no cover
    _AIORateLimiter = None
    _HAS_RATE_LIMITER = False

log = get_logger(__name__)


# -----------------------------
# In-memory ring buffer logging
# -----------------------------
class _Ring:
    def __init__(self, maxlen: int = 2000) -> None:
        self.buf: deque[str] = deque(maxlen=maxlen)

    def add(self, line: str) -> None:
        # NOTE: all inputs already stringified; keep compact
        self.buf.append(line.rstrip())

    def tail(self, n: int = 200) -> list[str]:
        if n <= 0:
            return []
        return list(self.buf)[-n:]


RING = _Ring()


# Optional: integrate with your project logger elsewhere if you have a dedicated
# handler.
class RingLogHandler(logging.Handler):  # simple handler-like shim
    _PTB_WEBHOOK_TOKEN = "telegram_webhook_not_configured"

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record)
        # Suppress PTB's startup warning when webhook is intentionally disabled
        if self._PTB_WEBHOOK_TOKEN in msg and str(
            os.getenv("TELEGRAM__WEBHOOK_ENABLED", "")
        ).strip().lower() not in {"1", "true", "yes", "on"}:
            # Don't pollute /issues and /why with an expected condition
            return
        RING.add(
            f"{time.strftime('%H:%M:%S')} [{record.levelname}] {record.name}: {msg}"
        )


try:
    logging.getLogger().addHandler(RingLogHandler())  # best effort
except Exception as exc:  # noqa: BLE001
    log.error(
        "Failed to attach RingLogHandler: %s",
        exc,
        extra={"event": "telegram_ring_handler_error"},
    )


_RING_LINE_RE = re.compile(
    r"^(?P<time>\d{2}:\d{2}:\d{2}) \[(?P<level>[A-Z]+)\] (?P<rest>.*)$"
)
_EVENT_KEY_RE = re.compile(r"event=([\w.:\-]+)")


# -----------------------------
# Deps from the core application
# -----------------------------
@dataclass(slots=True)
class TelegramDeps:
    token: str
    chat_id: int
    app_version: str
    telegram_plain: bool | None = None
    webhook_url: str | None = None
    webhook_path: str = "/telegram_webhook"
    webhook_secret_token: str | None = None
    webhook_max_failures: int = 5
    enable_polling_fallback: bool = False
    polling_interval_seconds: float = 5.0
    webhook_listen_host: str = "0.0.0.0"
    webhook_listen_port: int = 8000
    # Core singletons (optional; handlers check for None safely)
    broker_client: t.Any | None = None
    websocket_manager: t.Any | None = None
    market_data_manager: t.Any | None = None
    market_regime: MarketRegimeDetector | None = None
    regime_manager: MarketRegimeManager | None = None
    strategy_manager: t.Any | None = None
    strategy_runner: t.Any | None = None
    position_manager: t.Any | None = None
    order_manager: t.Any | None = None
    safe_order_manager: t.Any | None = None
    risk_manager: t.Any | None = None
    instrument_resolver: t.Any | None = None
    resolver: t.Any | None = None
    instrument_universe: InstrumentUniverseStatus | None = None
    instrument_db_path: str | None = None
    instrument_csv_path: str | None = None
    streamer: t.Any | None = None
    stream_supervisor: t.Any | None = None
    websocket_enabled: bool = True
    # Metrics registry (optional)
    metrics: t.Any | None = None
    session_guard: t.Any | None = None
    rate_limiter: t.Any | None = None
    get_ws_token: t.Callable[[], str] | None = None
    get_ws_token_issued_at: t.Callable[[], float | None] | None = None
    ws_host: str | None = None
    # Shadow mode toggler (optional): Callable[[bool], bool] | None
    set_shadow_mode: t.Callable[[bool], bool] | None = None
    get_shadow_mode: t.Callable[[], bool] | None = None
    paper_mode_getters: dict[str, t.Callable[[], bool]] | None = None
    paper_mode_setters: dict[str, t.Callable[[bool], bool]] | None = None
    regime_gate: t.Any | None = None
    replay_harness: t.Any | None = None
    replay_settings: t.Any | None = None
    kpi_tracker: t.Any | None = None
    data_hub: DataHub | None = None
    unified_manager: t.Any | None = None
    # Optional hot-reload hook
    reload_hook: t.Callable[[], str] | None = None
    selfchecker: t.Any | None = None

    # NOTE: the attributes above are intentionally soft-typed; the handlers below
    # probe methods with getattr/with suppress() to avoid hard coupling with
    # internal classes. That way, if a component is missing in your build, the
    # command just replies with a helpful message instead of crashing the app.


def _mask_token(token: str) -> str:
    """Return *token* with the middle characters redacted."""

    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}…{token[-4:]}"


def _sanitize_webhook_url(url: str) -> str:
    """Return webhook URL without query or fragment components."""

    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


@dataclass(slots=True)
class TelegramRuntimeMetrics:
    """In-memory counters for webhook and polling activity."""

    webhook_updates: int = 0
    webhook_failures: int = 0
    webhook_secret_failures: int = 0
    fallback_activations: int = 0
    polling_updates: int = 0
    polling_errors: int = 0

    def snapshot(self) -> dict[str, int]:
        """Return a serialisable snapshot of the counters."""

        return {
            "webhook_updates": self.webhook_updates,
            "webhook_failures": self.webhook_failures,
            "webhook_secret_failures": self.webhook_secret_failures,
            "fallback_activations": self.fallback_activations,
            "polling_updates": self.polling_updates,
            "polling_errors": self.polling_errors,
        }


class TelegramBot:
    _CHECK_ORDER: tuple[str, ...] = (
        "broker",
        "ws",
        "mdm",
        "resolver",
        "risk",
        "orders",
        "positions",
        "strategies",
        "session",
    )
    _CHECK_GROUPS: dict[str, tuple[str, ...]] = {
        "core": ("session", "risk", "orders", "strategies"),
        "market": ("broker", "ws", "mdm", "resolver"),
        "execution": ("orders", "positions", "risk"),
        "connectivity": ("broker", "session", "ws"),
    }
    _CONFIRMATION_WINDOW: timedelta = timedelta(seconds=90)

    def __init__(self, deps: TelegramDeps) -> None:
        self.deps = deps
        self.deps.enable_polling_fallback = True
        self.deps.webhook_url = None
        log.info("Telegram running in polling mode (Railway compatible)")
        self._app: Application | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._fallback_lock: asyncio.Lock | None = None
        self._webhook_server: "TelegramWebhookServer" | None = None
        self._polling_task: asyncio.Task[None] | None = None
        self._fallback_active = False
        self._webhook_failure_count = 0
        self._metrics = TelegramRuntimeMetrics()
        self._messenger: SafeMessenger | None = None
        self._help_registry: dict[int, dict[str, t.Any]] = {}
        self._response_builder: ResponseBuilder = RB
        self._supervisor = deps.stream_supervisor
        plain_override = getattr(deps, "telegram_plain", None)
        if plain_override is None:
            plain_override = get_bool("TELEGRAM__PLAIN_TEXT", False)
        self._plain_mode = bool(plain_override)
        log.info("Telegram enabled for chat_id=%s", deps.chat_id)
        self._tail_rate: dict[int, float] = {}
        self._admin_allowlist = self._build_admin_allowlist()
        self._reconcile_state_lock = threading.Lock()
        self._reconcile_health: dict[str, t.Any] = {}
        self._reconcile_alert_failures = 0
        self._reconcile_last_failure_at: datetime | None = None
        self._reconcile_last_success_at: datetime | None = None
        self._reconcile_alert_thresholds: tuple[int, ...] = (1, 3, 5, 10)
        self._wire_position_manager_events()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._alert_queue: asyncio.Queue[tuple[str, str, str, bool]] | None = None
        self._aggregated_alerts: dict[str, AggregatedAlert] = {}
        self._alert_worker: asyncio.Task[None] | None = None
        batch_seconds = float(os.getenv("TELEGRAM__ALERT_BATCH_SECONDS", "900") or 900)
        self._aggregation_interval = max(60.0, batch_seconds)
        quiet_window = timedelta(seconds=self._aggregation_interval)
        self._alert_deduplicator = AlertDeduplicator(
            quiet_window,
            bucket_capacity=5,
            bucket_refill_seconds=max(self._aggregation_interval / 5.0, 60.0),
        )
        self._log_handler: logging.Handler | None = None
        self._allocation_history: deque[tuple[datetime, dict[str, float]]] = deque(
            maxlen=12
        )
        self._last_allocation_snapshot: dict[str, float] | None = None
        self._regime_last_state: dict[str, tuple[str, float]] = {}
        self._regime_transitions: deque[
            tuple[datetime, str, str | None, str, float]
        ] = deque(maxlen=12)
        self._pnl_tracker: dict[str, float] = {}
        self._strategy_pnl_cache: dict[str, float] = {}
        self._pending_confirmation: dict[str, t.Any] | None = None
        self._started_at: datetime = datetime.now(timezone.utc)
        self._um: t.Any | None = None
        self._message_debug_handler_registered = False

    # =========================================================================
    # ✅ CORRECTED STARTUP LOGIC (Single Source of Truth)
    # =========================================================================

    async def start(self) -> None:
        """
        Unified startup for Telegram Bot.
        Handles Initialization -> Start -> Polling in one safe flow.
        """
        if self._app is not None and self._app.running:
            log.warning("TelegramBot start requested but already running. Ignoring.")
            return

        try:
            if self._app is None:
                self._app = self.builder().build()
                self._wire_handlers(self._app)

            command_handlers = (
                ("status", self.cmd_status),
                ("positions", self.cmd_positions),
                ("signals", self.cmd_signals),
                ("risk", self.cmd_risk),
                ("logs", self.cmd_tail),
                ("test_trade", self.cmd_test_trade),
                ("engine_test", self.cmd_engine_test),
            )
            for cmd, handler in command_handlers:
                if not self._command_registered(self._app, cmd):
                    self._app.add_handler(CommandHandler(cmd, handler))

            await self._app.initialize()
            if not self._app.running:
                await self._app.start()

            await self._app.bot.delete_webhook(drop_pending_updates=True)
            try:
                self._polling_task = asyncio.create_task(
                    self._app.run_polling(
                        allowed_updates=Update.ALL_TYPES,
                        drop_pending_updates=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.error("Telegram polling failed: %s", exc, exc_info=True)
                raise
            log.info("Telegram polling started successfully")
            self._ensure_alert_worker()

        except Exception as exc:  # noqa: BLE001
            log.error("Telegram polling failed: %s", exc, exc_info=True)

    async def _start_polling_if_needed(self) -> None:
        """
        Starts polling with explicit initialization to fix RuntimeError.
        """
        if not self.deps.enable_polling_fallback:
            return
        if self._app is None or self._app.updater is None:
            return
        
        # GUARD: Stop if already running
        if self._app.updater.running:
            return

        try:
            self._mark_polling_started()
            
            # ✅ FIX 1: Explicit Cool-down
            await asyncio.sleep(2.0)
            
            log.info("📡 Starting Telegram Polling (Background)...")

            # ✅ FIX 2: Clear Webhook safely
            try:
                await self._app.bot.delete_webhook(drop_pending_updates=True)
            except Exception:
                pass

            # ✅ FIX 3: Initialize the Updater explicitly
            # This fixes 'RuntimeError: This Updater was not initialized via Updater.initialize!'
            if not self._app.updater._initialized:
                 await self._app.updater.initialize()

            # ✅ FIX 4: Start Polling
            await self._app.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES,
                poll_interval=2.0
            )
            
            log.info("✅ Telegram Polling Active.")

        except Exception as exc:
            import traceback
            log.error(f"❌ Polling CRASH: {exc}")
            log.error(traceback.format_exc())
            self._metrics.polling_errors += 1
            self._mark_polling_stopped()

  
    def _command_registered(self, app: Application, command: str) -> bool:
        """Return ``True`` when *command* is already wired on *app*.

        Args:
            app: Telegram application inspected for handlers.
            command: Command name without leading slash.

        Returns:
            ``True`` if a handler already exists for the command.

        Raises:
            None.
        """

        log.debug(
            "Entered _command_registered",
            extra={"event": "telegram_command_registered_enter", "command": command},
        )
        handlers = list(app.handlers.get(0, []))
        for handler in handlers:
            if isinstance(handler, CommandHandler):
                raw_commands = getattr(handler, "commands", None) or getattr(
                    handler, "command", None
                )
                if raw_commands is None:
                    continue
                commands: t.AbstractSet[str]
                if isinstance(raw_commands, str):
                    commands = {raw_commands}
                else:
                    commands = t.cast(t.AbstractSet[str], raw_commands)
                if command in commands:
                    log.info(
                        "Condition met: command already registered",
                        extra={
                            "event": "telegram_command_exists",
                            "command": command,
                        },
                    )
                    return True
        return False

    def _get_unified_manager(self) -> UnifiedManager:
        """Return a lazily constructed unified manager instance.

        Args:
            None.

        Returns:
            Unified manager bound to existing bot dependencies.

        Raises:
            RuntimeError: If the manager cannot be initialised.
        """

        log.debug(
            "Entered TelegramBot._get_unified_manager",
            extra={"event": "telegram_um_enter"},
        )
        if self._um is not None:
            return self._um
        existing = getattr(self.deps, "unified_manager", None)
        if existing is not None:
            self._um = existing
            log.info(
                "Condition met: telegram_um_from_deps",
                extra={"event": "telegram_um_from_deps"},
            )
            return self._um
        try:
            from nifty_scalper_bot.core.unified_manager import UnifiedManager

            self._um = UnifiedManager(
                broker=self.deps.broker_client,
                mdm=self.deps.market_data_manager,
                ws=getattr(self.deps, "websocket_manager", None),
                risk=getattr(self.deps, "risk_manager", None),
                streamer=getattr(self.deps, "streamer", None),
                data_hub=getattr(self.deps, "data_hub", None),
                orders=(
                    getattr(self.deps, "safe_order_manager", None)
                    or getattr(self.deps, "order_manager", None)
                ),
                strategies=getattr(self.deps, "strategy_manager", None),
                logger=log.getChild("um"),
            )
            self.deps.unified_manager = self._um
            with suppress(Exception):
                mdm = getattr(self.deps, "market_data_manager", None)
                if mdm is not None:
                    mdm.set_unified_manager(self._um)
            with suppress(Exception):
                risk = getattr(self.deps, "risk_manager", None)
                if risk is not None:
                    risk.set_unified_manager(self._um)
            log.info(
                "Condition met: telegram_um_ready",
                extra={"event": "telegram_um_ready"},
            )
            return self._um
        except Exception as exc:  # pragma: no cover - defensive safeguard
            log.error(
                "Failure in TelegramBot._get_unified_manager: %s",
                exc,
                extra={"event": "telegram_um_error"},
                exc_info=exc,
            )
            raise RuntimeError("UnifiedManager unavailable") from exc

    def _invoke_um_plan(
        self, symbol: str, side: str, quantity: int
    ) -> tuple[UMPlan | None, str | None]:
        """Safely compute a unified manager plan for ``symbol``.

        Args:
            symbol: Instrument identifier requested by the operator.
            side: Desired trade direction (``BUY``/``SELL``).
            quantity: Requested lot count (non-negative integer).

        Returns:
            Tuple of the resulting :class:`UMPlan` (if any) and an optional
            human-readable error message suitable for operator feedback.

        Raises:
            None.
        """

        log.debug(
            "Entered TelegramBot._invoke_um_plan",
            extra={
                "event": "telegram_um_plan_enter",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
            },
        )
        side_norm = str(side or "").strip().upper() or "BUY"
        try:
            qty_norm = max(1, int(quantity))
        except (TypeError, ValueError):
            qty_norm = 1
        try:
            manager = self._get_unified_manager()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in _invoke_um_plan manager: %s",
                exc,
                extra={"event": "telegram_um_plan_manager_error"},
                exc_info=exc,
            )
            return None, "Unified manager unavailable."
        try:
            plan = manager.plan(symbol, side_norm, qty_norm)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in _invoke_um_plan planner: %s",
                exc,
                extra={"event": "telegram_um_plan_error"},
                exc_info=exc,
            )
            return None, f"Planner failure: {self._short_reason(str(exc))}"
        log.info(
            "Condition met: telegram_um_plan_ready",
            extra={
                "event": "telegram_um_plan_ready",
                "symbol": symbol,
                "side": side_norm,
                "qty": qty_norm,
                "ok": plan.ok,
                "reason": plan.reason,
            },
        )
        return plan, None

    def _mark_polling_started(self) -> None:
        """Mark polling as active and bump activation counter once."""

        if not self._fallback_active:
            self._fallback_active = True
            self._metrics.fallback_activations += 1

    def _mark_polling_stopped(self) -> None:
        """Mark polling as inactive."""

        self._fallback_active = False

  
    # -----------------
    # Lifecycle control
    # -----------------
    async def _guard(self, update: Update) -> t.Any | None:
        """Return the authorized chat or ``None`` if access should be denied."""
        chat = update.effective_chat
        user = update.effective_user
        message = update.effective_message

        text = ""
        if message is not None:
            text = str(getattr(message, "text", "") or "").strip()
        if text.startswith("/"):
            command = text.split()[0]
            log.info("Telegram command received: %s", command)
        
        if chat is None:
            log.warning("Received Telegram update without chat context")
            return None
            
        chat_id = int(chat.id)
        user_id = int(user.id) if user else "Unknown"
        username = user.username if user else "Unknown"
        
        # 1. Resolve Allowed ID (Primary Owner)
        allowed_primary = int(self.deps.chat_id)
        
        # 2. Check Authorization (Primary OR Admin List)
        is_allowed = (chat_id == allowed_primary) or (chat_id in self._admin_allowlist)
        
        if not is_allowed:
            # --- DEBUG LOGGING (So you can see your ID) ---
            log.warning(
                f"⛔ Blocked Access: Chat ID {chat_id} (User: {username}, ID: {user_id}). "
                f"Allowed Primary: {allowed_primary}, Admins: {self._admin_allowlist}"
            )
            
            with suppress(Exception):
                # Reply to user so they know they are connected but blocked
                bot = getattr(chat, "bot", None) or (self._app.bot if self._app else None)
                messenger = self._ensure_messenger(None, chat, bot_override=bot)
                await messenger.send_text(
                    chat_id,
                    f"⛔ 403 Forbidden.\nYour ID: <code>{chat_id}</code>",
                    disable_web_page_preview=True,
                    parse_mode=ParseMode.HTML
                )
            return None

        # 3. Metrics & Wrapping
        if self._fallback_active:
            self._metrics.polling_updates += 1
            
        if isinstance(chat, Chat):
            outer = self

            class _ChatWrapper:
                def __init__(self, wrapped: Chat):
                    self._wrapped = wrapped

                async def send_message(self, text: str, **kwargs: t.Any) -> t.Any:
                    kwargs.setdefault("disable_web_page_preview", True)
                    messenger = outer._ensure_messenger(None, self._wrapped)
                    message_text = str(text)
                    parse_mode_arg = kwargs.pop("parse_mode", None)
                    if outer._plain_mode:
                        plain = outer._html_to_plain(message_text)
                        return await messenger.send_text(
                            int(self._wrapped.id),
                            plain,
                            html_ready=True,
                            parse_mode=None,
                            **kwargs,
                        )
                    parse_mode = parse_mode_arg or ParseMode.HTML
                    try:
                        return await messenger.send_html(
                            int(self._wrapped.id),
                            message_text,
                            parse_mode=parse_mode,
                            **kwargs,
                        )
                    except TelegramError:
                        plain = outer._html_to_plain(message_text)
                        return await messenger.send_text(
                            int(self._wrapped.id),
                            plain,
                            html_ready=True,
                            parse_mode=None,
                            **kwargs,
                        )

                def __getattr__(
                    self, name: str
                ) -> t.Any:  # pragma: no cover - passthrough
                    return getattr(self._wrapped, name)

            return _ChatWrapper(chat)
        return chat

    def _build_admin_allowlist(self) -> set[int]:
        allow: set[int] = {int(self.deps.chat_id)}
        raw = os.getenv("TELEGRAM__ADMIN_CHAT_ID", "").strip()
        if not raw:
            return allow
        for chunk in raw.replace(";", ",").split(","):
            part = chunk.strip()
            if not part:
                continue
            try:
                allow.add(int(part))
            except ValueError:
                log.warning("invalid_admin_chat_id", extra={"chat_id": part})
        return allow

    def _wire_position_manager_events(self) -> None:
        """Attach position reconciliation listener when available.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered _wire_position_manager_events",
            extra={"event": "telegram_wire_pm_events"},
        )
        position_manager = getattr(self.deps, "position_manager", None)
        if position_manager is None:
            return
        add_listener = getattr(position_manager, "add_reconcile_listener", None)
        if not callable(add_listener):
            return
        try:
            add_listener(self._handle_position_reconcile_event)
        except Exception as exc:  # noqa: BLE001 - defensive hook guard
            log.error(
                "Failure in _wire_position_manager_events: %s",
                exc,
                extra={"event": "telegram_reconcile_listener_error"},
            )
            return
        log.info(
            "Condition met: telegram_reconcile_listener_registered",
            extra={"event": "telegram_reconcile_listener_registered"},
        )

    def _handle_position_reconcile_event(
        self, event: str, payload: t.Mapping[str, t.Any]
    ) -> None:
        """Record reconciliation events and trigger Telegram alerts when needed.

        Args:
            event: Event name provided by the position manager.
            payload: Event payload metadata describing the outcome.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered _handle_position_reconcile_event",
            extra={"event": "telegram_reconcile_event", "event_name": event},
        )
        now = datetime.now(timezone.utc)
        payload_dict = dict(payload)
        reason_value = payload_dict.get("reason")
        if reason_value is not None:
            payload_dict["reason"] = canonical(str(reason_value))
        elif event == "position_reconcile_ok":
            payload_dict["reason"] = canonical("ok")
        snapshot = {
            "event": event,
            "timestamp": now.isoformat(),
            "payload": payload_dict,
        }
        should_alert = False
        notify_recovery = False
        health_update = False
        previous_notified = 0
        failures = int(payload_dict.get("failures") or 0)
        with self._reconcile_state_lock:
            previous_notified = self._reconcile_alert_failures
            if event == "position_reconcile_failed":
                should_alert = self._should_alert_reconcile_failure(
                    failures, previous_notified
                )
                health_update = failures > 0 and failures != previous_notified
                if should_alert or health_update:
                    self._reconcile_alert_failures = failures
                self._reconcile_last_failure_at = now
            else:
                notify_recovery = previous_notified > 0
                self._reconcile_alert_failures = 0
                self._reconcile_last_success_at = now
            self._reconcile_health = snapshot
        if should_alert or health_update:
            payload_dict["health_update"] = bool(health_update and not should_alert)
            self._trigger_reconcile_alert(
                event,
                payload_dict,
                recovery=False,
                previous_failures=None,
            )
        elif notify_recovery:
            previous_failures = int(
                payload_dict.get("previous_failures") or previous_notified
            )
            payload_dict["health_update"] = False
            self._trigger_reconcile_alert(
                event,
                payload_dict,
                recovery=True,
                previous_failures=previous_failures,
            )

    def _should_alert_reconcile_failure(self, failures: int, previous: int) -> bool:
        """Return ``True`` when a failure count warrants a Telegram alert.

        Args:
            failures: Current consecutive failure count.
            previous: Last failure count that generated an alert.

        Returns:
            ``True`` when a notification should be emitted, else ``False``.

        Raises:
            None.
        """

        if failures <= 0 or failures == previous:
            return False
        if failures in self._reconcile_alert_thresholds:
            return True
        if failures > max(self._reconcile_alert_thresholds) and failures % 5 == 0:
            return True
        return False

    def _trigger_reconcile_alert(
        self,
        event: str,
        payload: t.Mapping[str, t.Any],
        *,
        recovery: bool,
        previous_failures: int | None,
    ) -> None:
        """Schedule Telegram notification for reconciliation outcomes.

        Args:
            event: Event identifier emitted by the position manager.
            payload: Metadata describing the reconciliation attempt.
            recovery: ``True`` when the alert reflects a recovery.
            previous_failures: Prior failure streak for recovery messaging.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered _trigger_reconcile_alert",
            extra={
                "event": "telegram_reconcile_alert_schedule",
                "event_name": event,
                "recovery": recovery,
            },
        )
        message = self._format_reconcile_message(
            event,
            payload,
            recovery=recovery,
            previous_failures=previous_failures,
        )
        failures = int(payload.get("failures") or 0)
        health_update = bool(payload.get("health_update"))
        severity = "info" if recovery else ("warning" if failures < 3 else "critical")
        reason_token = self._short_reason(str(payload.get("reason", "unknown")))
        key = f"reconcile:{reason_token}"
        immediate = recovery or failures <= 1 or health_update
        self._queue_alert_event(
            key=key,
            message=message,
            severity=severity,
            immediate=immediate,
        )

    def _safe_create_task(
        self, coro: t.Coroutine[t.Any, t.Any, None], name: str
    ) -> None:
        """Schedule *coro* on the Telegram application's event loop if ready.

        Args:
            coro: Coroutine scheduled for execution.
            name: Human-readable task name for diagnostics.

        Returns:
            None.

        Raises:
            None.
        """

        app = self._app
        if app is None:
            log.debug(
                "Skipped task scheduling; application unavailable",
                extra={"event": "telegram_task_skipped", "task_name": name},
            )
            return
        try:
            app.create_task(coro, name=name)
        except Exception as exc:  # noqa: BLE001 - defensive scheduling guard
            log.error(
                "Failure in _safe_create_task: %s",
                exc,
                extra={"event": "telegram_task_schedule_failed", "task_name": name},
            )

    def _ensure_alert_worker(self) -> None:
        """Ensure the background alert aggregator task is active."""

        if self._alert_queue is None:
            self._alert_queue = asyncio.Queue(maxsize=256)
        if self._alert_worker is not None and not self._alert_worker.done():
            return
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._alert_worker = loop.create_task(
            self._alert_aggregator(),
            name="telegram-alert-aggregator",
        )
        self._ensure_log_alert_handler()

    def _ensure_log_alert_handler(self) -> None:
        """Attach log handler that forwards warnings and errors to alerts.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        if self._log_handler is not None:
            return
        try:
            handler = AlertLogHandler(
                self._handle_log_alert_payload,
                exclude={__name__, "nifty_scalper_bot.utils.alerts"},
            )
            logging.getLogger().addHandler(handler)
            self._log_handler = handler
            log.info(
                "Condition met: alert log handler registered",
                extra={"event": "telegram_alert_log_handler_enabled"},
            )
        except Exception as exc:  # noqa: BLE001 - defensive guard
            log.error(
                "Failure in _ensure_log_alert_handler: %s",
                exc,
                extra={"event": "telegram_alert_log_handler_error"},
            )

    def _handle_log_alert_payload(self, payload: t.Mapping[str, str]) -> None:
        """Queue alert events derived from application log records.

        Args:
            payload: Mapping containing ``key``, ``message``, and ``severity``.

        Returns:
            None.

        Raises:
            None.
        """

        try:
            key = str(payload.get("key", "log:unknown"))
            message = str(payload.get("message", ""))
            severity = str(payload.get("severity", "warning"))
        except Exception as exc:  # noqa: BLE001 - defensive guard
            log.error(
                "Failure in _handle_log_alert_payload decode: %s",
                exc,
                extra={"event": "telegram_alert_log_payload_error"},
            )
            return
        self._queue_alert_event(
            key=key,
            message=message,
            severity=severity,
            immediate=False,
        )


    def _get_nifty_spot_price(self) -> float | None:
        """Get NIFTY spot price with multi-tier fallback."""
        
        # Tier 1: Market data manager
        mdm = getattr(self.deps, "market_data_manager", None)
        if mdm:
            for symbol in ["NIFTY", "NSE:NIFTY 50", "NSE:NIFTY50"]:
                try:
                    tick = mdm.get_latest_tick(symbol)
                    if tick and isinstance(tick, dict):
                        ltp = tick.get("ltp") or tick.get("last_price")
                        if ltp and ltp > 0:
                            return float(ltp)
                except Exception:
                    continue
                    
        # Tier 2: Data hub
        hub = getattr(self.deps, "data_hub", None)
        if hub:
            try:
                snapshot = hub.snapshot("NIFTY")
                if snapshot and snapshot.get("ltp"):
                    return float(snapshot["ltp"])
            except Exception:
                pass
                
        return None

    def _parse_nifty_option_symbol(self, symbol: str) -> dict | None:
        """Parse NIFTY option symbol to extract components."""
        import re
        import calendar
        from datetime import datetime, timedelta
        
        symbol = symbol.replace("NFO:", "").strip()
        
        # Monthly Format: NIFTY25NOV25950CE
        monthly = re.match(r'NIFTY(\d{2})([A-Z]{3})(\d+)(CE|PE)', symbol)
        if monthly:
            year, month_str, strike, opt_type = monthly.groups()
            months = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            month = months.get(month_str)
            if month:
                year_full = 2000 + int(year)
                
                # Find the last Thursday of the month for expiry
                last_day = calendar.monthrange(year_full, month)[1]
                expiry = datetime(year_full, month, last_day)
                while expiry.weekday() != 3: # 3 is Thursday
                    expiry = expiry - timedelta(days=1)
                            
                days = max((expiry - datetime.now()).days, 0)
                return {
                    "strike": int(strike),
                    "expiry": expiry,
                    "days_to_expiry": days,
                    "option_type": opt_type,
                    "symbol_type": "Monthly"
                }
                
        # Weekly Format (Hypothetical): NIFTY25N2625950CE
        weekly = re.match(r'NIFTY(\d{2})([A-Z])(\d{2})(\d+)(CE|PE)', symbol)
        if weekly:
            year, month_code, day, strike, opt_type = weekly.groups()
            month_map = {
                '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
                '7': 7, '8': 8, '9': 9, 'O': 10, 'N': 11, 'D': 12
            }
            month = month_map.get(month_code)
            if month:
                expiry = datetime(2000 + int(year), month, int(day))
                days = max((expiry - datetime.now()).days, 0)
                return {
                    "strike": int(strike),
                    "expiry": expiry,
                    "days_to_expiry": days,
                    "option_type": opt_type,
                    "symbol_type": "Weekly"
                }
                
        return None

    def _calculate_greeks_simple(
        self,
        spot: float,
        strike: float,
        days_to_expiry: float,
        option_type: str,
    ) -> dict:
        """Simple Greeks approximation using Black-Scholes principles (20% IV assumption)."""
        import math
        
        if days_to_expiry <= 0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "days_to_expiry": 0,
                "moneyness": spot / strike,
                "error": "Expired"
            }
            
        t = days_to_expiry / 365.25
        moneyness = spot / strike
        
        # Delta Approximation
        if option_type.upper() in ["CE", "CALL"]:
            # Caps delta between 0.01 and 0.99
            delta = 0.5 + min(0.49, max(-0.49, (moneyness - 1.0) * 2.0))
        else:
            # Caps delta between -0.99 and -0.01
            delta = -0.5 - min(0.49, max(-0.49, (1.0 - moneyness) * 2.0))
            
        # Gamma Approximation (Higher Gamma near ATM, lower for short expiry)
        gamma = 0.01 / (abs(moneyness - 1) + 0.01) * math.sqrt(1 / max(t, 0.01))
        gamma = min(gamma, 0.05)
        
        # Theta Approximation (Uses fixed 20% IV)
        # Formula: -Spot * IV / (2 * sqrt(t)) / 365.25
        theta = -spot * 0.20 / (2 * math.sqrt(max(t, 0.01))) / 365.25
        
        # Vega Approximation (Uses fixed 20% IV)
        vega = spot * math.sqrt(max(t, 0.01)) * 0.01
        
        return {
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta": round(theta, 2),
            "vega": round(vega, 2),
            "days_to_expiry": round(days_to_expiry, 1),
            "moneyness": round(moneyness, 3)
        }
  
    def _remove_log_alert_handler(self) -> None:
        """Detach the log alert handler when shutting down.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        handler = self._log_handler
        if handler is None:
            return
        try:
            logging.getLogger().removeHandler(handler)
        except Exception as exc:  # noqa: BLE001 - defensive guard
            log.error(
                "Failure in _remove_log_alert_handler: %s",
                exc,
                extra={"event": "telegram_alert_log_handler_remove_error"},
            )
        finally:
            self._log_handler = None

    def _queue_alert_event(
        self,
        *,
        key: str,
        message: str,
        severity: str,
        immediate: bool,
    ) -> None:
        """Queue or dispatch an alert message with throttling support.

        Args:
            key: Aggregation key describing the alert source.
            message: Human-readable alert message body.
            severity: Severity string (info|warning|critical).
            immediate: Caller hint requesting immediate delivery.

        Returns:
            None.

        Raises:
            None.
        """

        dispatch_now = self._evaluate_alert_dispatch(
            key=key,
            severity=severity,
            hint_immediate=immediate,
        )
        queue = self._alert_queue
        if queue is None or self._loop is None:
            self._safe_create_task(
                self._dispatch_alert(message, severity),
                name="telegram-alert-direct",
            )
            return
        loop = self._loop

        def _put() -> None:
            try:
                queue.put_nowait((key, message, severity, dispatch_now))
            except asyncio.QueueFull:
                log.warning(
                    "Alert queue full; delivering immediately",
                    extra={"event": "telegram_alert_queue_full", "key": key},
                )
                loop.create_task(  # type: ignore[union-attr]
                    self._dispatch_alert(message, severity),
                    name="telegram-alert-overflow",
                )

        loop.call_soon_threadsafe(_put)

    def _evaluate_alert_dispatch(
        self,
        *,
        key: str,
        severity: str,
        hint_immediate: bool,
    ) -> bool:
        """Return ``True`` when an alert should bypass aggregation.

        Args:
            key: Aggregation key describing the alert source.
            severity: Severity string associated with the alert.
            hint_immediate: Caller preference for immediate dispatch.

        Returns:
            ``True`` if the alert should send immediately, else ``False``.

        Raises:
            None.
        """

        try:
            decision = self._alert_deduplicator.should_immediate(
                key,
                severity,
                hint_immediate=hint_immediate,
            )
            log.debug(
                "Entered _evaluate_alert_dispatch",
                extra={
                    "event": "telegram_alert_evaluate",
                    "key": key,
                    "severity": severity,
                    "hint_immediate": hint_immediate,
                    "decision": decision,
                },
            )
            return decision
        except Exception as exc:  # noqa: BLE001 - defensive guard
            log.error(
                "Failure in _evaluate_alert_dispatch: %s",
                exc,
                extra={"event": "telegram_alert_evaluate_error", "key": key},
            )
            return hint_immediate

    async def _alert_aggregator(self) -> None:
        """Aggregate queued alerts and flush on cadence or critical events."""

        if self._alert_queue is None:
            return
        queue = self._alert_queue
        deadline = time.monotonic() + self._aggregation_interval
        try:
            while True:
                now_monotonic = time.monotonic()
                if now_monotonic >= deadline:
                    await self._flush_aggregated_alerts()
                    deadline = time.monotonic() + self._aggregation_interval
                    continue
                timeout = max(0.5, deadline - now_monotonic)
                try:
                    key, message, severity, immediate = await asyncio.wait_for(
                        queue.get(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    await self._flush_aggregated_alerts()
                    deadline = time.monotonic() + self._aggregation_interval
                    continue
                if immediate:
                    await self._flush_aggregated_alerts()
                    await self._dispatch_alert(message, severity)
                    deadline = time.monotonic() + self._aggregation_interval
                    continue
                unique_key = key not in self._aggregated_alerts
                if unique_key and self._aggregated_alerts:
                    log.info(
                        "Condition met: unique alert triggered flush",
                        extra={
                            "event": "telegram_alert_unique_flush",
                            "key": key,
                        },
                    )
                    await self._flush_aggregated_alerts()
                    deadline = time.monotonic() + self._aggregation_interval
                self._update_aggregated_alert(key, message, severity)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - defensive guard
            log.error(
                "Failure in _alert_aggregator: %s",
                exc,
                extra={"event": "telegram_alert_aggregator_error"},
            )
        finally:
            await self._flush_aggregated_alerts()

    def _update_aggregated_alert(self, key: str, message: str, severity: str) -> None:
        """Update in-memory aggregation bucket for *key* with *message*."""

        now = datetime.now(timezone.utc)
        category = key.split(":", 1)[0].strip().lower() or "misc"
        entry = self._aggregated_alerts.get(key)
        if entry is None:
            entry = AggregatedAlert(
                key=key,
                severity=severity,
                first_seen=now,
                last_seen=now,
                category=category,
            )
            self._aggregated_alerts[key] = entry
        else:
            entry.category = category
        entry.count += 1
        entry.last_seen = now
        entry.severity = severity
        entry.messages.append(message[:500])
        if len(entry.messages) > 5:
            entry.messages = entry.messages[-5:]

    async def _flush_aggregated_alerts(self) -> None:
        """Flush aggregated alerts as a single Telegram message."""

        if not self._aggregated_alerts:
            return
        total_events = sum(item.count for item in self._aggregated_alerts.values())
        severity_rank = {"info": 0, "warning": 1, "critical": 2}
        severity = "info"
        first_seen = min(item.first_seen for item in self._aggregated_alerts.values())
        last_seen = max(item.last_seen for item in self._aggregated_alerts.values())
        window = f"{first_seen.strftime('%H:%M:%S')}Z–{last_seen.strftime('%H:%M:%S')}Z"
        category_counts: dict[str, int] = {}
        for entry in self._aggregated_alerts.values():
            category_counts[entry.category] = (
                category_counts.get(entry.category, 0) + entry.count
            )
        lines: list[str] = [
            f"Aggregated alerts ({total_events} events)",
            f"Window: {window}",
        ]
        if category_counts:
            breakdown = " | ".join(
                f"{cat.replace('_', ' ')}: {count}"
                for cat, count in sorted(category_counts.items())
            )
            lines.append(f"Breakdown: {breakdown}")
        for entry in sorted(
            self._aggregated_alerts.values(),
            key=lambda item: item.last_seen,
            reverse=True,
        ):
            if severity_rank.get(entry.severity, 0) > severity_rank.get(severity, 0):
                severity = entry.severity
            stamp = entry.last_seen.strftime("%H:%M:%S")
            lines.append(f"• {entry.key} – {entry.count} events (last {stamp}Z)")
            for msg in entry.messages[-3:]:
                lines.append(f"    • {msg}")
        message = "\n".join(lines)
        log.info(
            "Condition met: aggregated alerts flushed",
            extra={
                "event": "telegram_alerts_flushed",
                "total_events": total_events,
                "bucket_count": len(self._aggregated_alerts),
                "severity": severity,
            },
        )
        await self._dispatch_alert(message, severity)
        self._aggregated_alerts.clear()
        self._alert_deduplicator.prune(now=datetime.now(timezone.utc))

    async def _dispatch_alert(self, message: str, severity: str) -> None:
        """Send *message* to the configured chat with severity prefix."""

        app = self._app
        if app is None:
            log.debug(
                "Skipped alert dispatch; application unavailable",
                extra={"event": "telegram_alert_dispatch_skipped"},
            )
            return
        prefix_map = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
        prefix = prefix_map.get(severity, "ℹ️")
        try:
            messenger = self._ensure_messenger(None, None, bot_override=app.bot)
        except Exception as exc:  # noqa: BLE001 - defensive messenger guard
            log.error(
                "Failure in _dispatch_alert messenger: %s",
                exc,
                extra={"event": "telegram_alert_dispatch_messenger_failed"},
            )
            return
        try:
            await messenger.send_text(
                int(self.deps.chat_id),
                f"{prefix} {message}",
                disable_web_page_preview=True,
            )
        except Exception as exc:  # noqa: BLE001 - defensive send guard
            log.error(
                "Failure in _dispatch_alert send: %s",
                exc,
                extra={"event": "telegram_alert_dispatch_failed"},
            )

    async def _drain_alerts(self) -> None:
        """Flush any pending aggregated alerts before shutdown."""

        await self._flush_aggregated_alerts()

    async def _send_reconcile_alert(
        self,
        event: str,
        payload: t.Mapping[str, t.Any],
        *,
        recovery: bool,
        previous_failures: int | None,
    ) -> None:
        """Dispatch Telegram message describing reconciliation health.

        Args:
            event: Event identifier emitted by the position manager.
            payload: Metadata describing the reconciliation attempt.
            recovery: ``True`` when the alert reflects a recovery.
            previous_failures: Prior failure streak for recovery messaging.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered _send_reconcile_alert",
            extra={"event": "telegram_reconcile_alert_send", "event_name": event},
        )
        app = self._app
        if app is None:
            return
        try:
            messenger = self._ensure_messenger(None, None, bot_override=app.bot)
        except Exception as exc:  # noqa: BLE001 - defensive messenger guard
            log.error(
                "Failure in _send_reconcile_alert messenger: %s",
                exc,
                extra={"event": "telegram_reconcile_alert_messenger_failed"},
            )
            return
        message = self._format_reconcile_message(
            event,
            payload,
            recovery=recovery,
            previous_failures=previous_failures,
        )
        try:
            await messenger.send_text(
                int(self.deps.chat_id),
                message,
                disable_web_page_preview=True,
            )
        except Exception as exc:  # noqa: BLE001 - defensive send guard
            log.error(
                "Failure in _send_reconcile_alert send: %s",
                exc,
                extra={"event": "telegram_reconcile_alert_send_failed"},
            )

    def _format_reconcile_message(
        self,
        event: str,
        payload: t.Mapping[str, t.Any],
        *,
        recovery: bool,
        previous_failures: int | None,
    ) -> str:
        """Return a concise textual summary for reconciliation alerts.

        Args:
            event: Event identifier emitted by the position manager.
            payload: Metadata describing the reconciliation attempt.
            recovery: ``True`` when the alert reflects a recovery.
            previous_failures: Prior failure streak for recovery messaging.

        Returns:
            Human-readable alert message for Telegram delivery.

        Raises:
            None.
        """

        reason = self._short_reason(str(payload.get("reason", "unknown")))
        retry_val = payload.get("retry_sec")
        retry_txt = ""
        if isinstance(retry_val, (int, float)):
            retry_txt = f", retry in {float(retry_val):.1f}s"
        error_txt = payload.get("error")
        if recovery:
            count_val = payload.get("count")
            parts = ["✅ Position reconcile recovered"]
            if isinstance(count_val, (int, float)):
                parts.append(f"positions={int(count_val)}")
            if previous_failures:
                parts.append(f"after {int(previous_failures)} failure(s)")
            return " ".join(parts)
        message = (
            f"⚠️ Position reconcile FAILED (reason={reason}, failures="
            f"{int(payload.get('failures') or 0)}{retry_txt})"
        )
        if error_txt:
            message += f" – {self._short_reason(str(error_txt))}"
        return message

    def _reconcile_status_line(self) -> tuple[str | None, str | None]:
        """Build reconciliation status summary and optional operator hint.

        Args:
            None.

        Returns:
            Tuple of ``(line, hint)`` where ``line`` is formatted HTML and
            ``hint`` is a plain-text operator suggestion when available.

        Raises:
            None.
        """

        rb = self._response_builder
        with self._reconcile_state_lock:
            snapshot = dict(self._reconcile_health)
        event = snapshot.get("event")
        if not event:
            return None, None
        payload = t.cast(t.Mapping[str, t.Any], snapshot.get("payload", {}))
        timestamp = snapshot.get("timestamp")
        when = ""
        if isinstance(timestamp, str):
            when = self._format_timestamp(timestamp)
        suffix = f" @<i>{rb.esc(when)}</i>" if when else ""
        if event == "position_reconcile_failed":
            reason = self._short_reason(str(payload.get("reason", "unknown")))
            failures = int(payload.get("failures") or 0)
            retry_val = payload.get("retry_sec")
            retry_txt = ""
            if isinstance(retry_val, (int, float)):
                retry_txt = f" retry=<code>{retry_val:.1f}s</code>"
            err_text = payload.get("error")
            err_suffix = (
                f" err=<i>{rb.esc(self._short_reason(str(err_text)))}</i>"
                if err_text
                else ""
            )
            line = (
                f"{EMOJI['warn']} Reconcile: <b>FAILED</b> "
                f"reason=<code>{rb.esc(reason)}</code> "
                f"failures=<b>{failures}</b>{retry_txt}{err_suffix}{suffix}"
            )
            hint = "Position reconciliation failing; inspect broker connectivity."
            return line, hint
        count_val = payload.get("count")
        prev_failures = int(payload.get("previous_failures") or 0)
        count_txt = (
            f" count=<b>{int(count_val)}</b>"
            if isinstance(count_val, (int, float))
            else ""
        )
        recovered_txt = f" recovered=<b>{prev_failures}</b>" if prev_failures else ""
        line = f"{EMOJI['ok']} Reconcile: <b>OK</b>{count_txt}{recovered_txt}{suffix}"
        return line, None

    def _format_timestamp(self, value: str) -> str:
        """Return compact UTC timestamp for display in status outputs.

        Args:
            value: ISO-8601 timestamp string to format.

        Returns:
            Timestamp formatted as ``HH:MM:SSZ`` when possible, otherwise the
            original ``value``.

        Raises:
            None.
        """

        with suppress(Exception):
            when = datetime.fromisoformat(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            else:
                when = when.astimezone(timezone.utc)
            return when.strftime("%H:%M:%SZ")
        return value

    async def _guard_admin(self, update: Update) -> bool:
        chat = update.effective_chat
        chat_id = int(getattr(chat, "id", 0)) if chat is not None else 0
        if chat_id in self._admin_allowlist:
            return True
        if chat is not None:
            messenger = self._ensure_messenger(None, chat)
            await messenger.send_text(
                chat_id,
                "Admin command requires elevated access.",
                disable_web_page_preview=True,
            )
        log.warning(
            "telegram_admin_guard_rejected",
            extra={"event": "telegram_admin_guard_rejected", "chat_id": chat_id},
        )
        return False

    def _ensure_messenger(
        self,
        ctx: ContextTypes.DEFAULT_TYPE | None,
        chat: Chat | t.Any | None,
        *,
        bot_override: Bot | None = None,
    ) -> SafeMessenger:
        bot_obj: t.Any = bot_override
        if bot_obj is None and ctx is not None:
            bot_obj = getattr(ctx, "bot", None)
        if bot_obj is None and self._app is not None:
            bot_obj = self._app.bot
        if bot_obj is None and chat is not None:
            bot_obj = getattr(chat, "bot", None)

        if bot_obj is None and chat is not None:

            class _ChatProxy:
                def __init__(self, wrapped: t.Any) -> None:
                    self._wrapped = wrapped

                async def send_message(
                    self, chat_id: int, text: str, **kwargs: t.Any
                ) -> t.Any:
                    sender = getattr(self._wrapped, "send_message", None)
                    if sender is None:
                        raise RuntimeError("Chat stub missing send_message")
                    result = sender(text, **kwargs)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result

                async def send_document(
                    self,
                    chat_id: int,
                    document: t.Any,
                    **kwargs: t.Any,
                ) -> t.Any:
                    sender = getattr(self._wrapped, "send_document", None)
                    if sender is None:
                        raise BadRequest("send_document unsupported")
                    result = sender(document=document, **kwargs)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result

            bot_obj = _ChatProxy(chat)

        if bot_obj is None:
            raise RuntimeError("Telegram bot instance unavailable")

        if self._messenger is None or self._messenger.bot is not bot_obj:
            self._messenger = SafeMessenger(bot_obj)
        return self._messenger

    def _set_pending_confirmation(
        self,
        action: str,
        chat_id: int,
        *,
        metadata: dict[str, t.Any] | None = None,
    ) -> None:
        """Record *action* confirmation intent for ``chat_id``.

        Args:
            action: Canonical identifier for the pending action.
            chat_id: Telegram chat requesting the operation.
            metadata: Optional supplemental context persisted with the request.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered _set_pending_confirmation",
            extra={
                "event": "telegram_confirm_set",
                "action": action,
                "chat_id": chat_id,
            },
        )
        payload: dict[str, t.Any] = {
            "action": action,
            "requested_by": chat_id,
            "created_at": datetime.now(timezone.utc),
        }
        if metadata:
            payload.update(metadata)
        self._pending_confirmation = payload

    def _clear_pending_confirmation(self) -> None:
        """Remove any stored confirmation request.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered _clear_pending_confirmation",
            extra={"event": "telegram_confirm_clear"},
        )
        self._pending_confirmation = None

    def _pending_confirmation_expired(self, pending: dict[str, t.Any]) -> bool:
        """Return ``True`` when *pending* has exceeded the confirmation window.

        Args:
            pending: Stored confirmation metadata structure.

        Returns:
            ``True`` if the confirmation window elapsed or metadata is invalid.

        Raises:
            None.
        """

        try:
            created = pending.get("created_at")
            if not isinstance(created, datetime):
                return True
            age = datetime.now(timezone.utc) - created
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in _pending_confirmation_expired: %s",
                exc,
                extra={"event": "telegram_confirm_expired_error"},
            )
            return True
        return age > self._CONFIRMATION_WINDOW

    def _format_uptime(self) -> str:
        """Return a human-friendly uptime string for the bot instance.

        Args:
            None.

        Returns:
            Readable uptime string summarising elapsed time.

        Raises:
            None.
        """

        log.debug(
            "Entered _format_uptime",
            extra={"event": "telegram_format_uptime_enter"},
        )
        try:
            delta = datetime.now(timezone.utc) - self._started_at
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in _format_uptime: %s",
                exc,
                extra={"event": "telegram_format_uptime_error"},
            )
            return "unknown"
        total_seconds = max(int(delta.total_seconds()), 0)
        days, rem = divmod(total_seconds, 86_400)
        hours, rem = divmod(rem, 3_600)
        minutes, seconds = divmod(rem, 60)
        parts: list[str] = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if seconds or not parts:
            parts.append(f"{seconds}s")
        uptime = " ".join(parts)
        log.info(
            "Condition met: telegram_format_uptime_success",
            extra={"event": "telegram_format_uptime_success", "uptime": uptime},
        )
        return uptime

    def _mode_status_icon(self) -> str:
        """Return emoji summarising the current trading mode.

        Args:
            None.

        Returns:
            Emoji string representing the active trading mode.

        Raises:
            None.
        """

        log.debug(
            "Entered _mode_status_icon",
            extra={"event": "telegram_mode_icon_enter"},
        )
        try:
            get_shadow = getattr(self.deps, "get_shadow_mode", None)
            shadow_enabled = bool(get_shadow and get_shadow())
            icon = "🟡" if shadow_enabled else "🟢"
            log.info(
                "Condition met: telegram_mode_icon_success",
                extra={
                    "event": "telegram_mode_icon_success",
                    "shadow_mode": shadow_enabled,
                },
            )
            return icon
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in _mode_status_icon: %s",
                exc,
                extra={"event": "telegram_mode_icon_error"},
                exc_info=exc,
            )
            return "❔"

    def _mode_badge_text(self) -> str:
        """Return decorated trading mode label for Telegram responses.

        Args:
            None.

        Returns:
            str: Emoji-tagged trading mode label.

        Raises:
            None.
        """

        log.debug(
            "Entered _mode_badge_text",
            extra={"event": "telegram_mode_badge_enter"},
        )
        try:
            get_shadow = getattr(self.deps, "get_shadow_mode", None)
            shadow = bool(get_shadow and get_shadow())
            badge = f"{EMOJI['shadow']} SHADOW" if shadow else f"{EMOJI['live']} LIVE"
            log.info(
                "Condition met: telegram_mode_badge_success",
                extra={
                    "event": "telegram_mode_badge_success",
                    "shadow_mode": shadow,
                },
            )
            return badge
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in _mode_badge_text: %s",
                exc,
                extra={"event": "telegram_mode_badge_error"},
                exc_info=exc,
            )
            return "UNKNOWN"

    def _build_basic_kpi(self) -> str:
        """Construct fallback KPI summary when the tracker is unavailable.

        Args:
            None.

        Returns:
            Multi-line fallback KPI summary string.

        Raises:
            None.
        """

        log.debug(
            "Entered _build_basic_kpi",
            extra={"event": "telegram_kpi_fallback_enter"},
        )
        try:
            pm = getattr(self.deps, "position_manager", None)
            rm = getattr(self.deps, "risk_manager", None)
            if pm is None:
                log.info(
                    "Condition met: telegram_kpi_no_position_manager",
                    extra={"event": "telegram_kpi_no_position_manager"},
                )
                return (
                    "KPI tracker unavailable and position metrics are not accessible.\n"
                    "Configure the KPI tracker for full analytics."
                )
            open_positions = 0
            realized = 0.0
            unrealized = 0.0
            net_pnl = 0.0
            exposure = 0.0
            raw_positions: t.Iterable[t.Any] = []
            with suppress(Exception):
                raw_positions = getattr(pm, "get_open_positions", lambda: [])()
                open_positions = len(list(raw_positions))
            with suppress(Exception):
                realized_val = getattr(pm, "get_realized_pnl", lambda: 0.0)()
                realized = float(realized_val)
            with suppress(Exception):
                unrealized_val = getattr(pm, "get_unrealized_pnl", lambda: 0.0)()
                unrealized = float(unrealized_val)
            with suppress(Exception):
                net_val = getattr(pm, "get_net_pnl", lambda: 0.0)()
                net_pnl = float(net_val)
            with suppress(Exception):
                exposure_val = getattr(pm, "get_total_exposure", lambda: 0.0)()
                exposure = float(exposure_val)
            lines = [
                "📊 Basic KPIs (fallback)",
                f"Open positions: {open_positions}",
                f"Net P&L: ₹{net_pnl:.2f}",
                f"Realized: ₹{realized:.2f}",
                f"Unrealized: ₹{unrealized:.2f}",
                f"Exposure: ₹{exposure:.2f}",
            ]
            if rm is not None:
                try:
                    snapshot = rm.snapshot()
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure in _build_basic_kpi snapshot: %s",
                        exc,
                        extra={"event": "telegram_kpi_snapshot_error"},
                    )
                else:
                    day_loss = float(getattr(snapshot, "day_loss", 0.0))
                    limit = float(getattr(snapshot, "max_day_loss", 0.0))
                    losses = int(getattr(snapshot, "losses_in_row", 0))
                    lines.append(f"Drawdown: ₹{day_loss:.2f} / limit ₹{limit:.2f}")
                    lines.append(f"Loss streak: {losses}")
            lines.append("")
            lines.append(
                "💡 Full KPI tracking becomes available when the tracker is enabled."
            )
            message = "\n".join(lines)
            log.info(
                "Condition met: telegram_kpi_fallback_success",
                extra={"event": "telegram_kpi_fallback_success"},
            )
            return message
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in _build_basic_kpi: %s",
                exc,
                extra={"event": "telegram_kpi_fallback_error"},
                exc_info=exc,
            )
            return "KPI tracker unavailable and fallback metrics failed."

    def _humanize_margin_reason(self, reason: str | None) -> str:
        """Translate raw margin planner *reason* into user guidance.

        Args:
            reason: Raw reason string returned by the margin planner.

        Returns:
            Human readable explanation suitable for operator feedback.

        Raises:
            None.
        """

        log.debug(
            "Entered _humanize_margin_reason",
            extra={"event": "telegram_margin_reason_enter", "reason": reason},
        )
        if not reason:
            return "No blocking reason reported."
        normalized = reason.upper()
        if "NO_QTY_AFTER_RISK" in normalized:
            return (
                "Risk sizing reduced quantity to zero. Reduce order size or adjust "
                "risk caps."
            )
        if "MIS_WINDOW_CLOSED" in normalized:
            return "MIS window closed. Switch to NRML or wait for market open."
        if normalized.startswith("MARGIN NEEDED="):
            return (
                "Available margin is below requirement. Add funds or lower the "
                "quantity."
            )
        return reason

    async def _confirm_enable_live(
        self, chat: Chat, ctx: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Enable live trading mode after explicit confirmation.

        Args:
            chat: Telegram chat metadata for the requestor.
            ctx: Handler context used to send replies.

        Returns:
            ``True`` when live trading successfully switches on.

        Raises:
            None.
        """

        log.debug(
            "Entered _confirm_enable_live",
            extra={"event": "telegram_confirm_enable_live_enter"},
        )
        try:
            toggle = self.deps.set_shadow_mode
            if toggle is None:
                await self._reply(chat, ctx, "Shadow toggle unavailable.")
                return False
            try:
                ok = bool(toggle(False))
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in _confirm_enable_live toggle: %s",
                    exc,
                    extra={"event": "telegram_confirm_enable_live_error"},
                    exc_info=exc,
                )
                await self._reply(chat, ctx, "Failed to enable live trading.")
                return False
            if ok:
                override_active = False
                guard = getattr(self.deps, "session_guard", None)
                if guard is not None:
                    with suppress(Exception):
                        status_obj = guard.last_status()
                        override_active = bool(
                            getattr(status_obj, "override_out_of_hours", False)
                        )
                message = (
                    "shadow=false (override: market closed)"
                    if override_active
                    else "shadow=false (live enabled)"
                )
                await self._reply(chat, ctx, message)
                log.info(
                    "Condition met: telegram_confirm_enable_live_success",
                    extra={"event": "telegram_confirm_enable_live_success"},
                )
                return True
            status = self._session_guard_status() or {}
            reasons = status.get("reasons") or []
            message = "shadow=false blocked"
            if reasons:
                message += f": {', '.join(str(reason) for reason in reasons)}"
            await self._reply(chat, ctx, message)
            log.info(
                "Condition met: telegram_confirm_enable_live_blocked",
                extra={"event": "telegram_confirm_enable_live_blocked"},
            )
            return False
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in _confirm_enable_live outer: %s",
                exc,
                extra={"event": "telegram_confirm_enable_live_outer_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unexpected error enabling live trading.")
            return False

    async def _execute_emergency_flatten(
        self, chat: Chat, ctx: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Cancel orders and flatten positions as an emergency action.

        Args:
            chat: Telegram chat metadata for the requestor.
            ctx: Handler context used to send replies.

        Returns:
            ``True`` when cancellation and flattening succeed.

        Raises:
            None.
        """

        log.debug(
            "Entered _execute_emergency_flatten",
            extra={"event": "telegram_confirm_emergency_enter"},
        )
        try:
            order_manager = getattr(self.deps, "order_manager", None)
            position_manager = getattr(self.deps, "position_manager", None)
            if order_manager is None or position_manager is None:
                await self._reply(chat, ctx, "Order or position manager unavailable.")
                return False
            market_data_manager = getattr(self.deps, "market_data_manager", None)
            errors: list[str] = []
            cancelled: list[str] = []
            cancel_fn = getattr(order_manager, "cancel_pending_orders", None)
            if callable(cancel_fn):
                try:
                    cancelled = list(cancel_fn())
                    log.info(
                        "Condition met: go_flat cancelled orders",
                        extra={
                            "event": "cmd_go_flat_cancelled",
                            "count": len(cancelled),
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure in _execute_emergency_flatten cancel: %s",
                        exc,
                        extra={"event": "cmd_go_flat_cancel_error"},
                    )
                    errors.append(f"cancel:{self._short_reason(str(exc))}")
            else:
                log.info(
                    "Condition met: go_flat cancel unsupported",
                    extra={"event": "cmd_go_flat_cancel_unsupported"},
                )
            positions: list[t.Any] = []
            with suppress(Exception):
                positions = list(position_manager.get_open_positions())
            flattened: list[str] = []
            for position in positions:
                symbol = getattr(position, "symbol", None) or getattr(
                    position, "tradingsymbol", ""
                )
                quantity = getattr(position, "quantity", 0)
                side = "SELL" if quantity > 0 else "BUY"
                exit_qty = abs(int(quantity))
                if not symbol or exit_qty <= 0:
                    continue
                try:
                    if market_data_manager is not None and hasattr(
                        market_data_manager, "submit_market_exit"
                    ):
                        market_data_manager.submit_market_exit(  # type: ignore[attr-defined]
                            symbol=symbol,
                            side=side,
                            quantity=exit_qty,
                        )
                    else:
                        close_fn = getattr(order_manager, "close_position", None)
                        if callable(close_fn):
                            close_fn(symbol=symbol, quantity=exit_qty)
                        else:
                            place_fn = getattr(order_manager, "place_order", None)
                            if callable(place_fn):
                                place_fn(
                                    symbol=symbol,
                                    side=side,
                                    quantity=exit_qty,
                                    price=None,
                                )
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure in _execute_emergency_flatten close: %s",
                        exc,
                        extra={"event": "cmd_go_flat_close_error", "symbol": symbol},
                    )
                    errors.append(f"close:{symbol}:{self._short_reason(str(exc))}")
                else:
                    flattened.append(f"{symbol}x{exit_qty}")
            summary = [
                "🚨 Emergency flatten executed.",
                f"Cancelled orders: {len(cancelled)}",
                f"Closed legs: {', '.join(flattened) if flattened else 'none'}",
            ]
            if errors:
                summary.append("Issues: " + ", ".join(errors))
            await self._reply(chat, ctx, "\n".join(summary))
            log.info(
                "Condition met: telegram_confirm_emergency_success",
                extra={
                    "event": "telegram_confirm_emergency_success",
                    "cancelled": len(cancelled),
                    "closed": len(flattened),
                    "errors": len(errors),
                },
            )
            return True
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in _execute_emergency_flatten: %s",
                exc,
                extra={"event": "telegram_confirm_emergency_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Emergency flatten failed unexpectedly.")
            return False

    # -----------------
    # Transport control
    # -----------------
    async def _ensure_transport(self) -> None:
        """Ensure a Telegram transport is active when webhooks are disabled."""

        if self._app is None:
            return

        webhook_enabled_env = get_bool("TELEGRAM__WEBHOOK_ENABLED", False)
        webhook_configured = bool((self.deps.webhook_url or "").strip())
        webhook_enabled = bool(webhook_enabled_env and webhook_configured)

        if webhook_enabled:
            return

        await self._start_polling_if_needed()

    # ---------- Small parsing helpers ----------
    def _parse_order_args(
        self, args: t.Sequence[str] | None
    ) -> tuple[str | None, int, float | None]:
        """Parse manual order arguments.

        Args:
            args: Optional sequence containing ``SYMBOL [QTY] [PRICE]``.

        Returns:
            Tuple of ``(symbol, quantity, price)`` where price ``None`` implies market.
        """

        if not args:
            return None, 1, None
        sym = args[0].strip().upper()
        qty = 1
        price: float | None = None
        if len(args) >= 2:
            with suppress(Exception):
                qty = max(1, int(args[1]))
        if len(args) >= 3:
            with suppress(Exception):
                price = float(args[2])
        return sym, qty, price

    def _coerce_risk_snapshot(self) -> dict[str, t.Any] | None:
        rm = self.deps.risk_manager
        if rm is None:
            return None
        snapshot_obj: t.Any
        try:
            getter = getattr(rm, "snapshot", None)
            if callable(getter):
                snapshot_obj = getter()
            else:
                return None
        except Exception:  # pragma: no cover - defensive
            return None
        fields = {
            "daily_realized",
            "daily_loss_limit",
            "day_loss",
            "max_day_loss",
            "losses_in_row",
            "cooldown_remaining",
            "breaker_tripped",
            "breaker_reason",
            "shadow_forced",
            "per_trade_risk_pct",
            "last_rejection",
            "timestamp",
        }
        payload: dict[str, t.Any] = {}
        for field_name in fields:
            value = getattr(snapshot_obj, field_name, None)
            if isinstance(snapshot_obj, dict):
                value = snapshot_obj.get(field_name, value)
            if field_name == "timestamp" and isinstance(value, datetime):
                payload[field_name] = value.isoformat()
            else:
                payload[field_name] = value
        return payload

    def _session_guard_status(self) -> dict[str, t.Any] | None:
        """Return latest trading session guard status snapshot.

        Args:
            None.

        Returns:
            dict[str, t.Any] | None: Session guard payload when available.

        Raises:
            None.
        """

        log.debug(
            "Entered TelegramController._session_guard_status",
            extra={"event": "telegram_session_guard_status_start"},
        )
        guard = getattr(self.deps, "session_guard", None)
        if guard is None:
            log.debug(
                "telegram_session_guard_missing",
                extra={"event": "telegram_session_guard_missing"},
            )
            return None
        try:
            status = guard.evaluate()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
            log.error(
                "Failure in TelegramController._session_guard_status: %s",
                exc,
                extra={"event": "telegram_session_guard_evaluate_error"},
                exc_info=exc,
            )
            return None
        payload = status.as_dict()
        market_open_flag = bool(
            getattr(status, "market_open", payload.get("market_open", False))
        )
        override_flag = bool(
            getattr(
                status,
                "override_out_of_hours",
                payload.get("override_out_of_hours", False),
            )
        )
        reasons_source = getattr(status, "reasons", payload.get("reasons", []))
        if isinstance(reasons_source, list):
            reasons_list = [str(reason) for reason in reasons_source]
        elif isinstance(reasons_source, (tuple, set)):
            reasons_list = [str(reason) for reason in reasons_source]
        elif isinstance(reasons_source, str):
            reasons_list = [reasons_source]
        elif isinstance(reasons_source, Iterable):
            reasons_list = [str(reason) for reason in reasons_source]
        else:
            reasons_list = [str(reasons_source)]
        status_all_clear = False
        all_clear_fn = getattr(status, "all_clear", None)
        if callable(all_clear_fn):
            try:
                status_all_clear = bool(all_clear_fn())
            except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
                log.error(
                    "Failure in TelegramController._session_guard_status all_clear: %s",
                    exc,
                    extra={"event": "telegram_session_guard_all_clear_error"},
                    exc_info=exc,
                )
                status_all_clear = False
        else:
            status_all_clear = bool(
                payload.get("session_valid")
                and payload.get("rate_limits_ok")
                and payload.get("risk_green")
                and (market_open_flag or override_flag)
            )
        if not market_open_flag and override_flag and not status_all_clear:
            log.info(
                "Condition met: session_guard_override_blocked",
                extra={
                    "event": "session_guard_override_blocked",
                    "payload": payload,
                    "reasons": reasons_list,
                },
            )
        return payload

    def _coerce_float_value(self, value: t.Any, *, field: str) -> float | None:
        """Safely convert ``value`` (or callable) to ``float`` for telemetry.

        Args:
            value: Value or zero-argument callable returning a numeric candidate.
            field: Field identifier used for diagnostic logging.

        Returns:
            float | None: Converted float value when coercion succeeds.

        Raises:
            None.
        """

        try:
            candidate = value() if callable(value) else value
            if candidate is None:
                return None
            return float(candidate)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "Failure in _coerce_float_value",
                extra={
                    "event": "telegram_float_coerce_error",
                    "field": field,
                },
                exc_info=exc,
            )
            return None

    def _format_currency(self, value: float | None, *, signed: bool = False) -> str:
        """Return currency-formatted text for ``value`` (₹) or ``'n/a'``.

        Args:
            value: Numeric value representing rupee amount.
            signed: Whether to include an explicit sign in the formatting.

        Returns:
            str: Formatted currency string or ``'n/a'`` when unavailable.

        Raises:
            None.
        """

        if value is None:
            return "n/a"
        pattern = "₹{0:+,.2f}" if signed else "₹{0:,.2f}"
        try:
            return pattern.format(value)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "Failure in _format_currency",
                extra={"event": "telegram_currency_format_error"},
                exc_info=exc,
            )
            return "n/a"

    def _looks_like_option(self, sym: str) -> bool:
        """Return ``True`` when ``sym`` resembles an option trading symbol.

        Args:
            sym: Candidate trading symbol to inspect.

        Returns:
            bool: ``True`` for option-like symbols, ``False`` otherwise.

        Raises:
            None.
        """

        normalized = str(sym or "").upper()
        return (
            normalized.endswith(("CE", "PE"))
            or "CE" in normalized
            or "PE" in normalized
            or "FUT" in normalized
        )

    def _fallback_quote_keys(self, base: str) -> list[str]:
        """Return heuristic broker keys when MDM candidates are unavailable.

        Args:
            base: Base trading symbol used to derive broker keys.

        Returns:
            list[str]: Ordered keys suitable for REST quote seeding.

        Raises:
            None.
        """

        normalized = str(base or "").strip().upper()
        if not normalized:
            return []
        derived: list[str] = []
        if self._looks_like_option(normalized) or normalized.endswith("FUT"):
            derived.append(f"NFO:{normalized}")
        else:
            derived.append(f"NSE:{normalized}")
        derived.append(normalized)
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in derived:
            if candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)
        return deduped

    def _fmt_tick_row(
        self,
        sym: str,
        tick: dict[str, t.Any] | None,
        *,
        source: str,
        age: float | None,
        token: int | None,
        ws: bool,
        poll: bool,
    ) -> str:
        """Return formatted cache row summarising tick metadata.

        Args:
            sym: Symbol displayed in the row.
            tick: Tick payload sourced from the cache.
            source: Origin indicator for the cached tick.
            age: Age of the tick in seconds when available.
            token: Numeric token associated with the symbol.
            ws: ``True`` when websocket transport is connected.
            poll: ``True`` when REST polling is enabled.

        Returns:
            str: Human-readable row string for diagnostics tables.

        Raises:
            None.
        """

        ltp = tick.get("ltp") if isinstance(tick, dict) else None
        bid = tick.get("bid") if isinstance(tick, dict) else None
        ask = tick.get("ask") if isinstance(tick, dict) else None
        ltp_s = f"{float(ltp):,.2f}" if isinstance(ltp, (int, float)) else "—"
        bid_s = f"{float(bid):,.2f}" if isinstance(bid, (int, float)) else "—"
        ask_s = f"{float(ask):,.2f}" if isinstance(ask, (int, float)) else "—"
        age_s = f"{age:.1f}s" if isinstance(age, (int, float)) and age >= 0 else "n/a"
        tok_s = str(token) if token is not None else "n/a"
        ws_s = "yes" if ws else "no"
        poll_s = "yes" if poll else "no"
        return (
            f"{sym:<14} {ltp_s:>12} {bid_s:>10} {ask_s:>10}  {age_s:>6}  "
            f"{source:<6}  tok={tok_s:<8}  WS={ws_s:<3}  Poll={poll_s:<3}"
        )

    def _fmt_bool(self, ok: bool | None) -> str:
        """Return emoji indicator for tri-state boolean values.

        Args:
            ok: Boolean status or ``None`` when unavailable.

        Returns:
            str: Emoji checkmark, cross, or ``?`` for indeterminate values.

        Raises:
            None.
        """

        try:
            if ok is None:
                return "?"
            return "✅" if ok else "❌"
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in _fmt_bool: %s",
                exc,
                extra={"event": "telegram_fmt_bool_error"},
                exc_info=exc,
            )
            return "?"

    def _risk_balance_snapshot(
        self, *, force: bool = False, risk_manager: t.Any | None = None
    ) -> tuple[float | None, str]:
        """Return cached risk balance and source label for Telegram displays.

        Args:
            force: When ``True`` the risk manager refreshes its balance cache.
            risk_manager: Optional explicit risk manager instance to reuse.

        Returns:
            tuple[float | None, str]: Pair of balance figure and source label.

        Raises:
            None.
        """

        manager = (
            risk_manager
            if risk_manager is not None
            else getattr(self.deps, "risk_manager", None)
        )
        if manager is None:
            return None, "unknown"

        try:
            latest_balance = manager.refresh_account_balance(force=force)
        except Exception as refresh_exc:  # noqa: BLE001
            log.error(
                "Failure in _risk_balance_snapshot refresh: %s",
                refresh_exc,
                extra={"event": "telegram_balance_snapshot_refresh_error"},
                exc_info=refresh_exc,
            )
            latest_balance = getattr(manager, "account_balance", None)

        balance_value = self._coerce_float_value(
            latest_balance,
            field="risk.balance",
        )

        label = "unknown"
        try:
            label = str(manager.balance_source_label())
        except Exception as label_exc:  # noqa: BLE001
            log.error(
                "Failure in _risk_balance_snapshot label: %s",
                label_exc,
                extra={"event": "telegram_balance_snapshot_label_error"},
                exc_info=label_exc,
            )
            label = "unknown"
        return balance_value, label

    async def cmd_probe(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Trace Broker→MDM→DataHub→Risk pipeline for symbol diagnostics.

        Args:
            update: Telegram update payload containing the command context.
            ctx: Telegram context with parsed arguments and chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug("Entered cmd_probe", extra={"event": "telegram_cmd_probe_enter"})
        try:
            emoji_catalog = globals().get("EMOJI", {}) or {}
        except Exception:  # pragma: no cover
            emoji_catalog = {}

        def _emoji(name: str, default: str) -> str:
            """Return emoji for name with fallback.

            Args:
                name: Emoji key to look up.
                default: Fallback emoji when key is missing.

            Returns:
                Resolved emoji string from catalog or provided default.

            Raises:
                None.
            """

            try:
                value = emoji_catalog.get(name)
            except Exception as lookup_exc:  # pragma: no cover - defensive access
                log.debug(
                    "Failure in cmd_probe emoji lookup",
                    extra={
                        "event": "telegram_cmd_probe_emoji_lookup_error",
                        "emoji": name,
                    },
                )
                log.debug(
                    "Failure detail: %s",
                    lookup_exc,
                    extra={"event": "telegram_cmd_probe_emoji_lookup_detail"},
                )
                value = None
            return value if isinstance(value, str) and value else default

        chat = await self._guard(update)
        if chat is None:
            return

        try:
            args = list(ctx.args or [])
            if not args:
                await self._reply(chat, ctx, "Usage: /probe SYMBOL [QTY] [BUY|SELL]")
                return

            symbol = str(args[0]).strip().upper()
            qty = 1
            side = "BUY"
            if len(args) >= 2:
                with suppress(Exception):
                    qty = max(1, int(args[1]))
            if len(args) >= 3:
                candidate_side = str(args[2]).strip().upper()
                if candidate_side in {"BUY", "SELL"}:
                    side = candidate_side

            rb = self._response_builder
            broker = getattr(self.deps, "broker_client", None)
            mdm = getattr(self.deps, "market_data_manager", None)
            hub = getattr(self.deps, "data_hub", None)
            risk_manager = getattr(self.deps, "risk_manager", None)

            broker_ok: bool | None = None
            broker_ltp: t.Any = None
            broker_bid: t.Any = None
            broker_ask: t.Any = None
            broker_err: str | None = None
            try:
                if broker is not None and hasattr(broker, "get_quote"):
                    # [FIX] Run blocking network call in thread
                    quote = await asyncio.to_thread(broker.get_quote, symbol)

                    def _pick(candidate: t.Any, *keys: str) -> t.Any:
                        for key in keys:
                            value = (
                                candidate.get(key)
                                if isinstance(candidate, dict)
                                else getattr(candidate, key, None)
                            )
                            if value is not None:
                                return value
                        return None

                    broker_ltp = _pick(quote, "last_price", "ltp", "close", "LastPrice")
                    broker_bid = _pick(quote, "buy", "bid", "best_bid_price")
                    broker_ask = _pick(quote, "sell", "ask", "best_ask_price")
                    if broker_ltp is None and hasattr(broker, "get_ltp"):
                        ltp_response = broker.get_ltp(symbol)
                        broker_ltp = (
                            ltp_response.get("ltp")
                            if isinstance(ltp_response, dict)
                            else getattr(ltp_response, "ltp", None)
                        )
                    broker_ok = broker_ltp is not None and float(broker_ltp) > 0
                else:
                    broker_ok = False
                    broker_err = "broker client unavailable"
            except Exception as broker_exc:  # noqa: BLE001
                broker_ok = False
                broker_err = str(broker_exc)
                log.error(
                    "Failure in cmd_probe broker path: %s",
                    broker_exc,
                    extra={"event": "telegram_probe_broker_error"},
                    exc_info=broker_exc,
                )

            mdm_latest: t.Any = None
            mdm_pull: t.Any = None
            mdm_source: t.Any = None
            try:
                if mdm is not None:
                    mdm_latest = mdm.get_latest_tick(symbol)
                    with suppress(Exception):
                        mdm_source = (getattr(mdm, "_last_tick_source", {}) or {}).get(
                            symbol
                        )
                    with suppress(Exception):
                        mdm_pull = mdm.pull_quote(symbol)
            except Exception as mdm_exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_probe mdm path: %s",
                    mdm_exc,
                    extra={"event": "telegram_probe_mdm_error"},
                    exc_info=mdm_exc,
                )

            hub_snapshot: dict[str, t.Any] = {}
            try:
                if hub is not None and hasattr(hub, "get_account_snapshot"):
                    hub_snapshot = hub.get_account_snapshot(force=True) or {}
            except Exception as hub_exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_probe datahub path: %s",
                    hub_exc,
                    extra={"event": "telegram_probe_hub_error"},
                    exc_info=hub_exc,
                )

            balance_value, balance_source = self._risk_balance_snapshot(
                force=True, risk_manager=risk_manager
            )

            lot_size = 75
            ltp_for_estimate: float | None = None

            def _num(candidate: t.Any) -> float | None:
                try:
                    return float(candidate) if candidate is not None else None
                except Exception:
                    return None

            if isinstance(mdm_pull, dict):
                ltp_for_estimate = _num(mdm_pull.get("ltp"))
            if ltp_for_estimate is None and isinstance(mdm_latest, dict):
                ltp_for_estimate = _num(mdm_latest.get("ltp"))
            if ltp_for_estimate is None:
                ltp_for_estimate = _num(broker_ltp)
            estimated_required: float | None = None
            if ltp_for_estimate is not None and ltp_for_estimate > 0:
                estimated_required = (
                    float(ltp_for_estimate) * float(lot_size) * float(qty)
                )

            def cur(candidate: t.Any) -> str:
                if isinstance(candidate, (int, float)):
                    return self._format_currency(candidate)
                return "n/a"

            def fval(candidate: t.Any) -> str:
                try:
                    return f"{float(candidate):,.2f}"
                except Exception:
                    return "n/a"

            lines: list[str] = []
            header = (
                f"{rb.section('Pipeline Probe')} 🔎 {rb.esc(symbol)}  "
                f"qty={qty}  side={side}"
            )
            lines.append(header)
            broker_status = self._fmt_bool(broker_ok)
            lines.append(f"{_emoji('broker', '💼')} Broker")
            lines.append(
                "  "
                + f"{broker_status} quote.ltp={fval(broker_ltp)} "
                + f"bid={fval(broker_bid)} ask={fval(broker_ask)}"
            )
            if broker_err:
                lines.append(f"  ⚠️ {rb.esc(broker_err)}")

            latest_ltp = (
                (mdm_latest or {}).get("ltp") if isinstance(mdm_latest, dict) else None
            )
            pull_ltp = (
                (mdm_pull or {}).get("ltp") if isinstance(mdm_pull, dict) else None
            )
            lines.append(f"{_emoji('market_data', '📈')} MarketDataManager")
            lines.append(
                "  "
                + f"latest.ltp={fval(latest_ltp)} "
                + f"source={rb.esc(str(mdm_source or 'n/a'))}"
            )
            lines.append(f"  pull.ltp={fval(pull_ltp)}")

            lines.append(f"{_emoji('data', '🗂️')} DataHub")
            lines.append(
                "  "
                + "snapshot.available="
                + f"{fval((hub_snapshot or {}).get('available'))}  "
                + f"net={fval((hub_snapshot or {}).get('net'))}"
            )

            lines.append(f"{_emoji('risk', '🧯')} RiskManager")
            lines.append(
                f"  balance={cur(balance_value)} (source={rb.esc(balance_source)})"
            )

            lines.append(f"{_emoji('money', '💰')} Quick Estimate")
            lines.append(
                "  "
                + f"lot_size={lot_size}  ltp_for_est={fval(ltp_for_estimate)}  "
                + f"est_required={cur(estimated_required)}"
            )

            hints: list[str] = []
            if self._ws_disabled():
                hints.append("Websocket disabled; relying on REST/cached quotes.")
            if broker_ok and pull_ltp is None and latest_ltp is None:
                hints.append(
                    "Broker LTP OK but MDM has no tick → symbol not subscribed / "
                    "no REST pull."
                )
            if (hub_snapshot or {}).get("available") in (
                None,
                0,
                0.0,
            ) and balance_source in {"env", "cache"}:
                hints.append("DataHub snapshot empty; balance may be cached/env.")
            if balance_source in {"env", "unknown"}:
                hints.append("Balance uses fallback; verify broker margin API.")
            if estimated_required is None:
                hints.append(
                    "No price for estimate; try subscribing the strike "
                    "or enable polling."
                )

            try:
                text = self._compose([lines], hints=hints)
            except Exception as compose_exc:  # noqa: BLE001 - defensive compose
                log.error(
                    "Failure in cmd_probe compose: %s",
                    compose_exc,
                    extra={"event": "telegram_cmd_probe_compose_error"},
                    exc_info=compose_exc,
                )
                text = "\n".join(lines + (["", "Hints:"] + hints if hints else []))

            try:
                await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)
                log.info(
                    "Condition met: telegram_cmd_probe_dispatched",
                    extra={"event": "telegram_cmd_probe_dispatched"},
                )
            except Exception as reply_exc:  # noqa: BLE001 - defensive reply
                log.error(
                    "Failure in cmd_probe reply: %s",
                    reply_exc,
                    extra={"event": "telegram_cmd_probe_reply_error"},
                    exc_info=reply_exc,
                )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_probe: %s",
                exc,
                extra={"event": "telegram_cmd_probe_error"},
                exc_info=exc,
            )
            await self._reply(
                chat,
                ctx,
                "Probe failed due to internal error. Please inspect logs.",
            )

    def _format_percent(self, value: float | None, *, signed: bool = False) -> str:
        """Return percentage-formatted string for ``value`` or ``'n/a'``.

        Args:
            value: Numeric value expressed as fraction (e.g. ``0.12`` for 12%).
            signed: Whether to include sign in the representation.

        Returns:
            str: Percentage string or ``'n/a'`` if conversion fails.

        Raises:
            None.
        """

        if value is None:
            return "n/a"
        pattern = "{0:+.1f}%" if signed else "{0:.1f}%"
        try:
            return pattern.format(value * 100.0)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "Failure in _format_percent",
                extra={"event": "telegram_percent_format_error"},
                exc_info=exc,
            )
            return "n/a"

    def _trend_delta(
        self,
        current: float | None,
        previous: float | None,
        *,
        threshold: float = 0.05,
    ) -> tuple[str, float | None]:
        """Return visual trend indicator and delta between snapshots.

        Args:
            current: Current numeric value.
            previous: Previous numeric value for comparison.
            threshold: Minimum absolute delta required to mark movement.

        Returns:
            tuple[str, float | None]: Trend symbol and signed delta when available.

        Raises:
            None.
        """

        if current is None:
            return "—", None
        if previous is None:
            return "•", None
        delta = current - previous
        if abs(delta) <= threshold:
            return "→", delta
        if delta > 0:
            return "↗", delta
        return "↘", delta

    def _heat_bar(self, value: float | None, max_abs: float, *, width: int = 3) -> str:
        """Return a compact heat bar representing relative magnitude.

        Args:
            value: Numeric input representing magnitude (positive or negative).
            max_abs: Maximum absolute value in the sample for scaling.
            width: Number of characters used to render the bar (>=1).

        Returns:
            str: Heat bar string using unicode blocks.

        Raises:
            None.
        """

        if value is None or max_abs <= 0 or width <= 0:
            return "·" * max(1, width)
        ratio = min(abs(value) / max_abs, 1.0)
        steps = "▁▂▃▄▅▆▇█"
        index = min(int(round(ratio * (len(steps) - 1))), len(steps) - 1)
        block = steps[index]
        direction = "▲" if value >= 0 else "▼"
        return direction + block * (max(1, width) - 1)

    def _age_label(self, seconds: float | None) -> str:
        if seconds is None:
            return "n/a"
        try:
            value = float(seconds)
        except (TypeError, ValueError):
            return "n/a"
        label = f"{max(0.0, value):.1f}s"
        if value > 5.0:
            label += " ⚠"
        return label

    def _format_ws_error(self, err: t.Any) -> str | None:
        if err is None:
            return None
        code = getattr(err, "code", None)
        reason = getattr(err, "reason", None)
        parts: list[str] = []
        if code is not None:
            parts.append(str(code))
        if reason:
            parts.append(str(reason))
        if parts:
            return ":".join(parts)
        text = str(err)
        return text or None

    def _collect_flattened_symbols(
        self,
        position_manager: t.Any,
        order_manager: t.Any,
        market_data_manager: t.Any | None,
    ) -> list[str]:
        """Submit offsetting orders to flatten open positions.

        Args:
            position_manager: Manager providing open position snapshots.
            order_manager: Order execution interface.
            market_data_manager: Market data helper for price discovery.

        Returns:
            Ordered list of symbols flattened via market exits.

        Raises:
            None.
        """

        log.debug(
            "Entered _collect_flattened_symbols",
            extra={"event": "go_flat_collect_enter"},
        )
        flattened: list[str] = []
        try:
            getter = getattr(position_manager, "get_open_positions", None)
            positions = list(getter()) if callable(getter) else []
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error(
                "Failure in _collect_flattened_symbols: %s",
                exc,
                extra={"event": "go_flat_positions_error"},
            )
            return flattened
        for position in positions:
            symbol = str(getattr(position, "symbol", "") or "").strip().upper()
            quantity = int(getattr(position, "quantity", 0) or 0)
            if not symbol or quantity == 0:
                continue
            side_token = str(getattr(position, "side", "")).upper()
            exit_side = "SELL" if side_token == "LONG" or quantity > 0 else "BUY"
            exit_qty = abs(quantity)
            exit_price = None
            if market_data_manager is not None:
                with suppress(Exception):
                    exit_price = market_data_manager.get_latest_price(symbol)
            if exit_price is None:
                exit_price = getattr(position, "current_price", None) or getattr(
                    position, "entry_price", None
                )
            try:
                order_manager.place_order(
                    symbol=symbol,
                    side=exit_side,
                    quantity=exit_qty,
                    price=None,
                )
            except Exception as exc:  # noqa: BLE001 - defensive
                log.error(
                    "Failure in _collect_flattened_symbols place_order: %s",
                    exc,
                    extra={
                        "event": "go_flat_place_error",
                        "symbol": symbol,
                        "side": exit_side,
                        "quantity": exit_qty,
                    },
                )
                continue
            flattened.append(symbol)
            log.info(
                "Condition met: flattened position",
                extra={
                    "event": "go_flat_flattened",
                    "symbol": symbol,
                    "side": exit_side,
                    "quantity": exit_qty,
                },
            )
        return flattened

    def _build_price_diag(self, symbol: str) -> dict[str, t.Any]:
        """Return diagnostic payload describing price freshness and sources.

        Args:
            symbol: Trading symbol inspected for diagnostics.

        Returns:
            Mapping containing price source, freshness, and gate status.

        Raises:
            RuntimeError: If the data hub dependency is unavailable.
        """

        log.debug(
            "Entered _build_price_diag",
            extra={"event": "diag_price_build_enter", "symbol": symbol},
        )
        data_hub = getattr(self.deps, "data_hub", None)
        if data_hub is None:
            raise RuntimeError("data_hub_unavailable")
        normalized = data_hub.normalize(symbol)
        quote = data_hub.get_quote(normalized, allow_pull=False) or {}
        fresh_ok, fresh_meta = data_hub.is_fresh(normalized)
        market_data_manager = getattr(self.deps, "market_data_manager", None)
        mdm_status: dict[str, t.Any] = {}
        source = "unknown"
        last_age = None
        fallback_enabled = None
        heartbeat_age = None
        if market_data_manager is not None:
            try:
                status = market_data_manager.mdm_status()
            except Exception as exc:  # noqa: BLE001 - defensive
                log.error(
                    "Failure in _build_price_diag mdm_status: %s",
                    exc,
                    extra={"event": "diag_price_mdm_error", "symbol": normalized},
                )
            else:
                mdm_status = status
                try:
                    source = str(
                        status.get("last_tick_source", {}).get(normalized, source)
                    )
                except Exception:  # noqa: BLE001 - defensive
                    source = "unknown"
                try:
                    last_age = status.get("last_tick_age", {}).get(normalized)
                except Exception:  # noqa: BLE001 - defensive
                    last_age = None
                fallback_enabled = bool(status.get("fallback_enabled"))
                heartbeat_age = status.get("heartbeat_age")
        mid_price = None
        mid_age = None
        if market_data_manager is not None:
            with suppress(Exception):
                mid_price, mid_age = market_data_manager.cached_mid(normalized)

        def _coerce(value: t.Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        reference_prices = {
            "bid": _coerce(quote.get("bid")),
            "ask": _coerce(quote.get("ask")),
            "ltp": _coerce(quote.get("ltp")),
            "last_price": _coerce(quote.get("last_price")),
            "close": _coerce(quote.get("close")),
            "open": _coerce(quote.get("open")),
            "mid": mid_price,
            "mid_age_ms": mid_age,
        }
        risk_manager = getattr(self.deps, "risk_manager", None)
        gate_allowed = None
        gate_reasons: tuple[str, ...] | None = None
        if risk_manager is not None:
            try:
                gate_allowed, gate_reasons = risk_manager.risk_gate_should_trade()
            except Exception as exc:  # noqa: BLE001 - defensive
                log.error(
                    "Failure in _build_price_diag gate: %s",
                    exc,
                    extra={"event": "diag_price_gate_error", "symbol": normalized},
                )
        payload = {
            "symbol": normalized,
            "price_source": source,
            "last_tick_age_s": last_age,
            "fallback_enabled": fallback_enabled,
            "heartbeat_age": heartbeat_age,
            "fresh_ok": fresh_ok,
            "fresh_meta": fresh_meta,
            "reference_prices": reference_prices,
            "gate": {
                "allowed": gate_allowed,
                "reasons": list(gate_reasons or ()),
            },
            "mdm_status": mdm_status,
        }
        log.info(
            "Condition met: built price diagnostics",
            extra={"event": "diag_price_built", "symbol": normalized},
        )
        return payload

    def _build_risk_diag(self, symbol: str | None = None) -> dict[str, t.Any]:
        """Return diagnostic payload summarising risk gate signals.

        Args:
            symbol: Optional trading symbol for contextual metadata.

        Returns:
            Mapping containing gate status, thresholds, and cooldowns.

        Raises:
            RuntimeError: If the risk manager dependency is unavailable.
        """

        log.debug(
            "Entered _build_risk_diag",
            extra={"event": "diag_risk_build_enter", "symbol": symbol},
        )
        risk_manager = getattr(self.deps, "risk_manager", None)
        if risk_manager is None:
            raise RuntimeError("risk_manager_unavailable")
        gate_allowed, gate_reasons = risk_manager.risk_gate_should_trade()
        risk_state = getattr(risk_manager, "risk_state", None)
        state_payload: dict[str, t.Any] = {}
        if risk_state is not None:
            now_ns = time.time_ns()
            last_tick_ns = getattr(risk_state, "_last_tick_ns", -1)
            age_ms = None
            if last_tick_ns and last_tick_ns > 0:
                age_ms = max((now_ns - last_tick_ns) / 1_000_000, 0.0)
            state_payload = {
                "quote_stale_ms_threshold": getattr(
                    risk_state, "quote_stale_ms_threshold", None
                ),
                "spread_widen_mult": getattr(risk_state, "spread_widen_mult", None),
                "max_intraday_dd": getattr(risk_state, "max_intraday_dd", None),
                "max_consecutive_losses": getattr(
                    risk_state, "max_consecutive_losses", None
                ),
                "median_spread": getattr(risk_state, "_median_spread", None),
                "consecutive_losses": getattr(risk_state, "_consecutive_losses", None),
                "last_tick_age_ms": age_ms,
                "reasons": sorted(getattr(risk_state, "_reasons", set())),
                "data_symbol": getattr(risk_state, "_data_symbol", None),
            }
        switches = getattr(risk_manager, "_switches", None)
        switches_payload: dict[str, t.Any] = {}
        if switches is not None:
            try:
                snap = switches.snapshot()
                switches_payload = {
                    "day_loss": snap.day_loss,
                    "max_day_loss": snap.max_day_loss,
                    "consecutive_losses": snap.consecutive_losses,
                    "max_consecutive_losses": snap.max_consecutive_losses,
                    "cooldown_remaining": snap.cooldown_remaining,
                }
            except Exception as exc:  # noqa: BLE001 - defensive
                log.error(
                    "Failure in _build_risk_diag switches: %s",
                    exc,
                    extra={"event": "diag_risk_switch_error"},
                )
        snapshot = self._coerce_risk_snapshot() or {}
        payload = {
            "symbol": symbol,
            "gate": {
                "allowed": gate_allowed,
                "reasons": list(gate_reasons),
            },
            "risk_state": state_payload,
            "switches": switches_payload,
            "snapshot": snapshot,
        }
        log.info(
            "Condition met: built risk diagnostics",
            extra={"event": "diag_risk_built", "allowed": gate_allowed},
        )
        return payload

    def _categorize(self, line: str) -> str:
        lower = line.lower()
        if "webhook" in lower:
            return "webhook"
        if "telegram" in lower:
            return "telegram"
        if any(token in lower for token in ("ws", "websocket", "stream")):
            return "ws"
        if any(token in lower for token in ("mdm", "market data", "tick")):
            return "mdm"
        if any(token in lower for token in ("broker", "kite", "zerodha")):
            return "broker"
        if "risk" in lower or "guard" in lower:
            return "risk"
        if any(token in lower for token in ("order", "safeordermanager")):
            return "orders"
        if "strategy" in lower:
            return "strategy"
        if "resolver" in lower:
            return "resolver"
        return "other"

    def _parse_ring_line(self, line: str) -> tuple[str | None, str | None, str]:
        match = _RING_LINE_RE.match(line)
        if not match:
            return (None, None, line.strip())
        ts = match.group("time")
        level = match.group("level")
        rest = match.group("rest")
        message = rest
        if ":" in rest:
            message = rest.split(":", 1)[1].strip()
        return (ts, level, message)

    def _iso_from_time(self, hhmmss: str | None) -> str | None:
        if not isinstance(hhmmss, str):
            return None
        text = hhmmss.strip()
        if not text:
            return None
        try:
            return datetime.combine(
                datetime.now().date(),
                datetime.strptime(text, "%H:%M:%S").time(),
            ).isoformat()
        except Exception:
            return None

    def _last_error(self) -> tuple[str, str, str] | None:
        lines = RING.tail(300)
        today = datetime.now().date()
        for line in reversed(lines):
            ts, level, message = self._parse_ring_line(line)
            if level not in {"ERROR", "WARNING", "CRITICAL"}:
                continue
            when = self._iso_from_time(ts)
            if when is None:
                when = datetime.combine(today, datetime.now().time()).isoformat(
                    timespec="seconds"
                )
            category = self._categorize(line)
            return (category, self._short_reason(message), when)
        return None

    def _short_reason(self, text: str) -> str:
        if not text:
            return ""
        clean = text.strip().replace("\n", " ")
        if len(clean) <= 100:
            return clean
        return clean[:97].rstrip() + "…"

    def _hint_for(self, category: str, message: str) -> str:
        msg = message.lower()
        if "instrument" in msg or "resolve" in msg:
            return "Try /resolve SYMBOL and ensure instruments are warmed."
        if "token" in msg and "quote" in msg:
            return "Check /check resolver and verify exchange prefix for the symbol."
        if category == "ws":
            if "heartbeat" in msg or "hb" in msg:
                return "Use /ws_status and consider reconnecting if hbΔ remains high."
            return "Inspect /check ws for connection state and errors."
        if category == "mdm":
            return "Run /check mdm to confirm tick fan-out and subscriptions."
        if category == "broker":
            return "Check broker session via /check broker or renew login credentials."
        if category == "risk":
            return "Review /risk or /check risk for breakers and cooldowns."
        if category == "orders":
            return "Use /check orders to inspect recent statuses and throttles."
        if category == "strategy":
            return "Run /check strategies to verify runners and tracked symbols."
        if category == "telegram":
            return "See /health telemetry and webhook stats for delivery issues."
        if category == "webhook":
            return (
                "Verify webhook secrets and endpoints; fallback to polling if needed."
            )
        return "Review logs for more context or retry the previous command."

    def builder(self, *, bot: Bot | None = None) -> ApplicationBuilder:
        b = ApplicationBuilder()
        if bot is not None:
            # PTB disallows attaching a rate limiter when a Bot is injected.
            try:
                b = b.bot(bot)  # type: ignore[arg-type]
            except AttributeError:  # pragma: no cover - PTB API guard
                b = b.token(self.deps.token)
            return b

        b = b.token(self.deps.token)
        if _HAS_RATE_LIMITER and _AIORateLimiter is not None:
            try:
                limiter = t.cast(t.Any, _AIORateLimiter())
                b = b.rate_limiter(limiter)
                log.info("Telegram: AIORateLimiter enabled.")
            except Exception as exc:
                log.warning(
                    (
                        "Telegram: AIORateLimiter not attached (%s). Continuing "
                        "without it."
                    ),
                    exc,
                )
        return b

    def build_application(self, *, bot: Bot | None = None) -> Application:
        """Return a PTB :class:`Application` wired with all console handlers."""

        app = self.builder(bot=bot).build()  # type: ignore[arg-type]
        self._wire_handlers(app)
        self._app = app
        self._messenger = SafeMessenger(app.bot)
        return app

    async def process_webhook_payload(self, payload: dict[str, t.Any]) -> int | None:
        """Convert *payload* into an update and queue it for processing."""

        if self._app is None:
            raise RuntimeError("Telegram application is not initialised")
        update = Update.de_json(payload, self._app.bot)
        chat = update.effective_chat
        chat_id = int(chat.id) if chat else None
        await self._app.update_queue.put(update)
        self._metrics.webhook_updates += 1
        self.record_webhook_success()
        return chat_id

    def record_webhook_success(self) -> None:
        """Reset webhook failure counters after a successful update."""

        self._webhook_failure_count = 0

    def record_webhook_failure(self, *, reason: str, status_code: int) -> None:
        """Increment failure counters and trigger fallback when needed."""

        self._metrics.webhook_failures += 1
        self._webhook_failure_count += 1
        log.warning(
            "Telegram webhook failure.",
            extra={
                "telegram_webhook_failure": {
                    "reason": reason,
                    "status_code": status_code,
                    "failures": self._webhook_failure_count,
                    "threshold": self.deps.webhook_max_failures,
                }
            },
        )
        threshold = max(1, int(self.deps.webhook_max_failures))
        if (
            self.deps.enable_polling_fallback
            and not self._fallback_active
            and self._webhook_failure_count >= threshold
        ):
            if self._fallback_lock is None:
                return
            asyncio.create_task(self._activate_polling_fallback(reason))

    def record_webhook_secret_failure(self) -> None:
        """Track failed secret validation events."""

        self._metrics.webhook_secret_failures += 1

    @property
    def metrics(self) -> TelegramRuntimeMetrics:
        """Return runtime metrics for external inspection."""

        return self._metrics

    @property
    def is_polling_active(self) -> bool:
        """Return ``True`` when polling fallback is active."""

        return self._fallback_active

    async def _start_webhook_stack(self) -> bool:
        """Configure Telegram webhook and start the HTTP server."""

        if self._app is None or not self.deps.webhook_url:
            return False
        configured = await self._configure_webhook()
        if not configured:
            return False
        server = TelegramWebhookServer(
            bot=self,
            path=self.deps.webhook_path,
            secret_token=self.deps.webhook_secret_token,
            host=self.deps.webhook_listen_host,
            port=self.deps.webhook_listen_port,
        )
        try:
            await server.start()
        except (
            Exception
        ) as exc:  # pragma: no cover - uvicorn errors are environment specific
            log.exception("Failed to start Telegram webhook server: %s", exc)
            return False
        self._webhook_server = server
        log.info(
            "Telegram webhook listening.",
            extra={
                "telegram_webhook": {
                    "url": _sanitize_webhook_url(self.deps.webhook_url),
                    "path": self.deps.webhook_path,
                    "host": self.deps.webhook_listen_host,
                    "port": self.deps.webhook_listen_port,
                }
            },
        )
        return True

    async def _configure_webhook(self) -> bool:
        """Register webhook URL with Telegram."""

        assert self._app is not None
        url = t.cast(str, self.deps.webhook_url)
        with suppress(TelegramError):
            await self._app.bot.delete_webhook(drop_pending_updates=True)
        attempts = 0
        while attempts < 3:
            try:
                await self._app.bot.set_webhook(
                    url=url,
                    allowed_updates=Update.ALL_TYPES,
                    secret_token=self.deps.webhook_secret_token,
                    drop_pending_updates=True,
                )
                log.info(
                    "Telegram webhook configured.",
                    extra={
                        "telegram_webhook": {
                            "url": _sanitize_webhook_url(url),
                            "token": _mask_token(self.deps.token),
                        }
                    },
                )
                return True
            except RetryAfter as exc:
                attempts += 1
                delay = float(getattr(exc, "retry_after", 1.0) or 1.0)
                log.warning("Telegram webhook rate limited; retrying in %.1fs", delay)
                await asyncio.sleep(delay)
            except TelegramError as exc:
                log.error("Failed to set Telegram webhook: %s", exc)
                return False
        return False

    async def after_application_built(self) -> None:
        """Ensure a Telegram transport is active after PTB wiring."""

        try:
            await self._ensure_transport()
        except Exception as exc:  # pragma: no cover - defensive
            log.exception(
                "telegram_transport_bootstrap_failed",
                extra={
                    "event": "telegram_transport_bootstrap_failed",
                    "err": str(exc),
                },
            )

    async def _activate_polling_fallback(self, reason: str) -> None:
        """Switch to polling mode when webhook delivery repeatedly fails."""

        if self._fallback_lock is None:
            return
        async with self._fallback_lock:
            if self._fallback_active:
                return
            if self._app is None:
                return
            updater = self._app.updater
            if updater is None:
                log.error(
                    "PTB updater is not available; cannot start polling fallback."
                )
                return
            if self._webhook_server is not None:
                await self._webhook_server.stop()
                self._webhook_server = None
            with suppress(Exception):
                await self._app.bot.delete_webhook(drop_pending_updates=False)
            self._webhook_failure_count = 0
            self._mark_polling_started()
            log.error("Telegram switching to polling fallback (%s)", reason)
            try:
                self._polling_task = asyncio.create_task(
                    self._polling_loop(updater),
                    name="telegram-polling-fallback",
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._mark_polling_stopped()
                log.exception("Failed to spawn polling fallback task: %s", exc)

    async def _polling_loop(self, updater: t.Any) -> None:
        """Run polling loop with backoff handling."""

        task = asyncio.current_task()
        try:
            log.warning("Telegram polling fallback active.")
            while True:
                if self._shutdown_event is not None and self._shutdown_event.is_set():
                    break
                try:
                    await updater.start_polling(
                        allowed_updates=Update.ALL_TYPES,
                        timeout=30,
                        drop_pending_updates=False,
                    )
                    if self._shutdown_event is not None:
                        await self._shutdown_event.wait()
                    break
                except RetryAfter as exc:
                    self._metrics.polling_errors += 1
                    delay = float(
                        getattr(exc, "retry_after", self.deps.polling_interval_seconds)
                        or self.deps.polling_interval_seconds
                    )
                    log.warning("Telegram polling rate limited; sleeping %.1fs", delay)
                    await asyncio.sleep(delay)
                except NetworkError as exc:
                    self._metrics.polling_errors += 1
                    log.warning("Telegram polling network error: %s", exc)
                    await asyncio.sleep(self.deps.polling_interval_seconds)
                except Exception as exc:  # pragma: no cover - defensive logging
                    self._metrics.polling_errors += 1
                    log.exception("Telegram polling error: %s", exc)
                    await asyncio.sleep(self.deps.polling_interval_seconds)
                finally:
                    with suppress(Exception):
                        await updater.stop()
        except asyncio.CancelledError:
            raise
        finally:
            self._mark_polling_stopped()
            if task is not None and self._polling_task is task:
                self._polling_task = None
            log.info("Telegram polling fallback loop exiting.")

    async def run(self) -> None:
        """Run the Telegram console in webhook-first mode."""

        self._shutdown_event = asyncio.Event()
        self._fallback_lock = asyncio.Lock()
        try:
            self._app = self.builder().build()
            self._wire_handlers(self._app)
            await self._app.initialize()
            await self._app.start()
            self._loop = asyncio.get_running_loop()
            self._ensure_alert_worker()
            log.info(
                "Telegram application initialised.",
                extra={
                    "telegram_boot": {
                        "chat_id": self.deps.chat_id,
                        "webhook_primary": bool(self.deps.webhook_url),
                        "polling_fallback": self.deps.enable_polling_fallback,
                    }
                },
            )

            if self.deps.webhook_url:
                started = await self._start_webhook_stack()
                if not started and self.deps.enable_polling_fallback:
                    await self._activate_polling_fallback("webhook setup failed")
                elif not started:
                    log.error(
                        "Telegram webhook failed to start and polling fallback "
                        "disabled."
                    )
            elif self.deps.enable_polling_fallback:
                log.warning(
                    "Telegram webhook URL not configured; enabling polling fallback."
                )
                await self._activate_polling_fallback("webhook url missing")
            else:
                log.warning(
                    (
                        "Telegram webhook URL not configured and fallback disabled; "
                        "console idle."
                    )
                )

            if self._shutdown_event is not None:
                await self._shutdown_event.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            log.exception("Telegram run() failed: %s", exc)
            if self.deps.enable_polling_fallback and not self._fallback_active:
                await self._activate_polling_fallback("run failure")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        if self._polling_task is not None:
            self._polling_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._polling_task
            self._polling_task = None
        if self._alert_worker is not None:
            self._alert_worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._alert_worker
            self._alert_worker = None
        if self._alert_queue is not None:
            await self._drain_alerts()
            self._alert_queue = None
        self._remove_log_alert_handler()
        if self._webhook_server is not None:
            await self._webhook_server.stop()
            self._webhook_server = None
        if self._app is None:
            log.info(
                "Telegram shut down.",
                extra={"telegram_metrics": self._metrics.snapshot()},
            )
            return
        updater = self._app.updater
        if updater is not None:
            with suppress(Exception):
                await updater.stop()
        with suppress(Exception):
            await self._app.bot.delete_webhook(drop_pending_updates=True)
        with suppress(Exception):
            await self._app.stop()
        with suppress(Exception):
            await self._app.shutdown()
        self._app = None
        self._fallback_active = False
        self._loop = None
        log.info(
            "Telegram shut down.",
            extra={"telegram_metrics": self._metrics.snapshot()},
        )

    # --------------
    # Command wiring
    # --------------
    def _wire_handlers(self, app: Application) -> None:
        async def _debug_message(
            update: Update, context: ContextTypes.DEFAULT_TYPE
        ) -> None:
            """Log every incoming Telegram message for diagnostics.

            Args:
                update: Telegram update payload.
                context: Callback context from python-telegram-bot.

            Returns:
                None.

            Raises:
                None.
            """

            del context
            message = getattr(update, "message", None)
            text = getattr(message, "text", None)
            log.debug("Telegram message received: %s", text)

        if not self._message_debug_handler_registered:
            app.add_handler(MessageHandler(filters.ALL, _debug_message))
            self._message_debug_handler_registered = True

        def register(
            cmd: str,
            handler: t.Callable[[Update, t.Any], t.Coroutine[t.Any, t.Any, t.Any]],
        ) -> None:
            if self._command_registered(app, cmd):
                return
            app.add_handler(CommandHandler(cmd, handler))
            self._register_command_doc(cmd, handler)

        def register_with_aliases(
            primary: str,
            handler: t.Callable[[Update, t.Any], t.Coroutine[t.Any, t.Any, t.Any]],
            aliases: tuple[str, ...] = (),
        ) -> None:
            register(primary, handler)
            for alias in aliases:
                register(alias, handler)

        CommandHandlerType = t.Callable[
            [Update, t.Any],
            t.Coroutine[t.Any, t.Any, t.Any],
        ]
        CommandEntry = tuple[str, CommandHandlerType, tuple[str, ...]]
        CommandGroup = tuple[str, tuple[CommandEntry, ...]]

        command_groups: tuple[CommandGroup, ...] = (
            (
                "top_level_diagnostics",
                (
                    ("help", self.cmd_help, ()),
                    ("status", self.cmd_status, ()),
                    ("session", self.cmd_session, ()),
                    ("quick", self.cmd_quick, ()),
                    ("health", self.cmd_health, ()),
                    ("version", self.cmd_version, ()),
                    ("whoami", self.cmd_whoami, ()),
                ),
            ),
            (
                "core_functionality_state",
                (
                    ("start", self.cmd_start, ()),
                    ("stop", self.cmd_stop, ()),
                    ("mode", self.cmd_shadow, ("shadow",)),
                    ("plan", self.cmd_plan, ()),
                    ("limits", self.cmd_gate, ("gate",)),
                    ("save", self.cmd_dump_logs, ()),
                    ("reload", self.cmd_reload, ()),
                ),
            ),
            (
                "market_data_connectivity",
                (
                    ("mdm", self.cmd_mdm, ()),
                    ("ws_status", self.cmd_ws_status, ("ws",)),
                    ("ws_diag", self.cmd_ws_diag, ()),
                    ("ws_reconnect", self.cmd_ws_reconnect, ()),
                    ("poll_status", self.cmd_poll_status, ()),
                    ("subscribe", self.cmd_subscribe, ("sub",)),
                    ("unsubscribe", self.cmd_unsubscribe, ("unsub",)),
                    ("watch", self.cmd_watch, ()),
                    ("unwatch", self.cmd_unwatch, ()),
                    ("tracking", self.cmd_tracking, ()),
                    ("iv", self.cmd_iv, ()),
                    ("greeks", self.cmd_greeks, ()),
                    (
                        "quote",
                        self.cmd_quote,
                        ("book", "ohlc", "chain", "atm", "spot"),
                    ),
                    ("tick", self.cmd_tick, ()),
                    ("diag_price", self.cmd_diag_price, ()),
                    ("fresh", self.cmd_fresh, ()),
                    ("transport", self.cmd_transport, ()),
                    ("resolve", self.cmd_resolve, ()),
                    ("broker_quote", self.cmd_broker_quote, ()),
                    ("qprobe", self.cmd_qprobe, ()),
                    ("ingestprobe", self.cmd_ingestprobe, ("iprobe", "pipe")),
                    ("quote_probe", self.cmd_quote_probe, ()),
                    ("probe", self.cmd_probe, ()),
                ),
            ),
            (
                "unified_manager",
                (
                    ("um", self.cmd_um, ()),
                    ("umlearn", self.cmd_umlearn, ()),
                    ("umprobe", self.cmd_umprobe, ()),
                    ("umtransport", self.cmd_umtransport, ()),
                    ("umplan", self.cmd_umplan, ()),
                    ("umwatch", self.cmd_umwatch, ()),
                    ("uminstruments", self.cmd_uminstruments, ()),
                    ("umwarm", self.cmd_umwarm, ()),
                    ("umstats", self.cmd_umstats, ()),
                ),
            ),
            (
                "orders_positions_trades",
                (
                    ("positions", self.cmd_positions, ()),
                    ("orders", self.cmd_orders, ()),
                    ("trades", self.cmd_trades, ("fills",)),
                    ("buy", self.cmd_buy, ("entry",)),
                    ("sell", self.cmd_sell, ("exit",)),
                    ("cancel", self.cmd_cancel, ("flat",)),
                    ("net", self.cmd_ping, ("ping",)),
                ),
            ),
            (
                "risk_performance_regimes",
                (
                    ("risk", self.cmd_risk, ("riskstate",)),
                    ("diag_risk", self.cmd_diag_risk, ()),
                    ("regime", self.cmd_regime, ()),
                    ("diag", self.cmd_diag, ()),
                    ("balance", self.cmd_balance, ()),
                    ("balance_probe", self.cmd_balance_probe, ()),
                    ("margin", self.cmd_margin, ()),
                    ("pnl", self.cmd_pnl, ()),
                    ("heatmap", self.cmd_heatmap, ("metrics_heatmap",)),
                    ("report", self.cmd_report, ()),
                    ("kpi", self.cmd_kpi, ("score",)),
                    ("trail", self.cmd_guard, ()),
                    ("size", self.cmd_plan, ()),
                    ("journal", self.cmd_issues, ()),
                ),
            ),
            (
                "strategy_execution_management",
                (
                    ("strategies", self.cmd_strategies, ("sig",)),
                    ("signals", self.cmd_signals, ()),
                    (
                        "strategy_scores",
                        self.cmd_strategy_scores,
                        ("scores",),
                    ),
                    (
                        "strategy_allocate",
                        self.cmd_strategy_allocate,
                        ("strategy_alloc",),
                    ),
                    ("score", self.cmd_score, ()),
                    ("alloc", self.cmd_allocations, ()),
                    (
                        "strategy_disable",
                        self.cmd_strategy_disable,
                        ("strategy_off",),
                    ),
                    (
                        "strategy_enable",
                        self.cmd_strategy_enable,
                        ("strategy_on",),
                    ),
                    ("regime_bypass", self.cmd_regime_bypass, ()),
                    ("guard", self.cmd_guard, ()),
                    ("selftest", self.cmd_selftest, ()),
                    ("assess", self.cmd_assess, ()),
                    ("cooldown", self.cmd_cooldown, ()),
                    ("pause", self.cmd_pause, ()),
                    ("resume", self.cmd_resume, ()),
                    ("test_trade", self.cmd_test_trade, ()),
                    ("engine_test", self.cmd_engine_test, ()),
                    ("test_flow", self.cmd_test_flow, ()),
                    ("paper", self.cmd_paper, ()),
                    ("paper_on", self.cmd_paper_on, ()),
                    ("paper_off", self.cmd_paper_off, ()),
                    ("shadow_on", self.cmd_shadow_on, ()),
                    ("shadow_off", self.cmd_shadow_off, ()),
                ),
            ),
            (
                "logging_error_reporting",
                (
                    ("logs", self.cmd_tail, ("tail",)),
                    ("dumpLogs", self.cmd_dump_logs, ()),
                    ("errors", self.cmd_errors, ()),
                    ("issues", self.cmd_issues, ()),
                    ("stderror", self.cmd_stderr, ()),
                    ("check", self.cmd_check, ()),
                    ("check_core", self.cmd_check_core, ()),
                    ("check_market", self.cmd_check_market, ()),
                    ("check_execution", self.cmd_check_execution, ()),
                    ("check_connectivity", self.cmd_check_connectivity, ()),
                    ("why", self.cmd_why, ()),
                    ("flow", self.cmd_flow, ()),
                    ("profile", self.cmd_profile, ()),
                    ("metrics", self.cmd_metrics, ()),
                    ("config", self.cmd_config, ()),
                    ("env", self.cmd_env, ()),
                    ("gc", self.cmd_gc, ()),
                ),
            ),
            (
                "advanced_administrative",
                (
                    ("opshelp", self.cmd_opshelp, ()),
                    ("confirm", self.cmd_confirm, ()),
                    ("debug_on", self.cmd_debug_on, ()),
                    ("debug_off", self.cmd_debug_off, ()),
                    ("replay", self.cmd_replay, ("replay_day",)),
                    ("backtest", self.cmd_backtest, ()),
                    ("go_flat", self.cmd_go_flat, ()),
                ),
            ),
        )

        for _, entries in command_groups:
            for primary, handler, aliases in entries:
                register_with_aliases(primary, handler, aliases)

    # ------------------
    # Helper pretty print
    # ------------------
    def _code(self, s: str) -> str:
        s = s.strip()
        return f"<pre>{_escape(s)}</pre>"

    def _parse_token_args(
        self, args: t.Sequence[str] | None
    ) -> tuple[list[int], list[str]]:
        if not args:
            return ([], [])
        raw = " ".join(args)
        pieces = [
            part.strip() for part in raw.replace(",", " ").split() if part.strip()
        ]
        tokens: list[int] = []
        invalid: list[str] = []
        for piece in pieces:
            try:
                tokens.append(int(piece))
            except ValueError:
                invalid.append(piece)
        return tokens, invalid

    def _parse_symbol_token_args(
        self, args: t.Sequence[str] | None
    ) -> tuple[list[int], list[str], list[str]]:
        """Parse args into integer tokens, symbol strings, and invalid entries."""

        if not args:
            return ([], [], [])
        raw = " ".join(args)
        pieces = [
            part.strip() for part in raw.replace(",", " ").split() if part.strip()
        ]
        tokens: list[int] = []
        symbols: list[str] = []
        invalid: list[str] = []
        for piece in pieces:
            try:
                tokens.append(int(piece))
                continue
            except ValueError:
                cleaned = piece.strip().upper()
                if cleaned:
                    symbols.append(cleaned)
                else:
                    invalid.append(piece)
        return tokens, symbols, invalid

    def _polling_streamer(self) -> t.Any | None:
        return getattr(self.deps, "streamer", None)

    def _stream_supervisor(self) -> t.Any | None:
        if self._supervisor is not None:
            return self._supervisor
        return getattr(self.deps, "stream_supervisor", None)

    def _fmt_age(self, seconds: float | None) -> str:
        """Return a compact age string for transport diagnostics.

        Args:
            seconds: Age value measured in seconds.

        Returns:
            str: Human-readable representation of ``seconds``.

        Raises:
            None.
        """

        if seconds is None or not isinstance(seconds, (int, float)):
            return "—"
        value = float(seconds)
        if value < 0:
            value = 0.0
        if value < 60:
            return f"{value:.1f}s"
        minutes, remainder = divmod(value, 60.0)
        if minutes < 60:
            return f"{int(minutes)}m{int(remainder)}s"
        hours, rem_minutes = divmod(minutes, 60.0)
        return f"{int(hours)}h{int(rem_minutes)}m"

    def _resolve_symbol_token(self, symbol: str) -> int | None:
        """Resolve ``symbol`` to a token using resolver and cache fallbacks.

        Args:
            symbol: Trading symbol to resolve.

        Returns:
            int | None: Matching token identifier when available.

        Raises:
            None.
        """

        log.debug(
            "Entered _resolve_symbol_token",
            extra={"event": "telegram_transport_resolve_token", "symbol": symbol},
        )
        resolver = getattr(self.deps, "resolver", None)
        if resolver is not None and hasattr(resolver, "resolve_symbol_to_token"):
            try:
                token = resolver.resolve_symbol_to_token(symbol)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in _resolve_symbol_token resolver: %s",
                    exc,
                    extra={
                        "event": "telegram_transport_resolve_token_error",
                        "symbol": symbol,
                    },
                )
            else:
                try:
                    if token is not None:
                        return int(token)
                except (TypeError, ValueError):
                    log.debug(
                        "Resolver returned non-integer token",
                        extra={
                            "event": "telegram_transport_resolve_token_cast",
                            "symbol": symbol,
                            "token": token,
                        },
                    )
        mdm = getattr(self.deps, "market_data_manager", None)
        if mdm is None:
            return None
        mapping = getattr(mdm, "_token_by_symbol", None)
        if isinstance(mapping, dict):
            try:
                token = mapping.get(symbol)
            except Exception as exc:  # noqa: BLE001 - defensive dict access
                log.error(
                    "Failure in _resolve_symbol_token cache lookup: %s",
                    exc,
                    extra={
                        "event": "telegram_transport_resolve_token_cache_error",
                        "symbol": symbol,
                    },
                )
                return None
            if token is not None:
                try:
                    return int(token)
                except (TypeError, ValueError):
                    log.debug(
                        "Cache returned non-integer token",
                        extra={
                            "event": "telegram_transport_resolve_token_cache_cast",
                            "symbol": symbol,
                            "token": token,
                        },
                    )
        return None

    def _symbol_tracking_rows(
        self,
        *,
        limit: int = 10,
        filter_symbol: str | None = None,
    ) -> tuple[list[str], list[str]]:
        """Return table rows summarising tracked symbols and data freshness.

        Args:
            limit: Maximum number of symbols to include in the table.
            filter_symbol: Optional symbol to filter for a deep dive view.

        Returns:
            tuple[list[str], list[str]]: Rendered table rows and advisory hints.

        Raises:
            None.
        """

        log.debug(
            "Entered _symbol_tracking_rows",
            extra={
                "event": "telegram_transport_rows_enter",
                "limit": limit,
                "filter": filter_symbol,
            },
        )
        mdm = getattr(self.deps, "market_data_manager", None)
        ws = (
            None
            if self._ws_disabled()
            else getattr(self.deps, "websocket_manager", None)
        )
        rows: list[str] = []
        hints: list[str] = []

        latest: dict[str, t.Any] = {}
        last_src: dict[str, t.Any] = {}
        if mdm is not None:
            try:
                latest_raw = getattr(mdm, "_latest_ticks", {}) or {}
                if isinstance(latest_raw, dict):
                    latest = {str(key): value for key, value in latest_raw.items()}
                last_src_raw = getattr(mdm, "_last_tick_source", {}) or {}
                if isinstance(last_src_raw, dict):
                    last_src = {str(key): value for key, value in last_src_raw.items()}
            except Exception as exc:  # noqa: BLE001 - defensive cache read
                log.error(
                    "Failure reading market data cache: %s",
                    exc,
                    extra={"event": "telegram_transport_rows_cache_error"},
                )
                hints.append("Unable to read market data cache.")
                latest = {}
                last_src = {}

        ws_subs: set[int] = set()
        if ws is not None:
            try:
                subscription_tokens = getattr(ws, "_subscription_tokens", None)
                if isinstance(subscription_tokens, set):
                    ws_subs = {int(token) for token in subscription_tokens}
            except Exception as exc:  # noqa: BLE001 - websocket private attribute
                log.error(
                    "Failure reading websocket subscriptions: %s",
                    exc,
                    extra={"event": "telegram_transport_rows_ws_error"},
                )

        poll_watch: set[str] = set()
        if mdm is not None:
            try:
                raw_poll = getattr(mdm, "_rest_poll_symbols", None)
                if isinstance(raw_poll, (set, list, tuple)):
                    poll_watch = {str(item) for item in raw_poll}
            except Exception as exc:  # noqa: BLE001 - defensive poll access
                log.error(
                    "Failure reading polling watch list: %s",
                    exc,
                    extra={"event": "telegram_transport_rows_poll_error"},
                )

        header = (
            f"{'sym':<18}{'ltp':>12}{'bid':>12}{'ask':>12}"
            f"{'age':>8}{'src':>10}{'tok':>8}{'WS':>6}{'Poll':>7}{'Tracked':>9}"
        )
        rows.append(header)
        rows.append("—" * len(header))

        symbols = sorted(latest.keys())
        if filter_symbol:
            symbols = [
                symbol for symbol in symbols if symbol.upper() == filter_symbol.upper()
            ]
            if not symbols:
                hints.append(
                    "Symbol not present in cache; confirm subscription or polling.",
                )
        if limit > 0:
            limit_val = max(1, int(limit))
            symbols = symbols[:limit_val]

        now_mono = time.monotonic()

        def _format_number(value: t.Any) -> str:
            if isinstance(value, (int, float)):
                return f"{float(value):,.2f}"
            try:
                return f"{float(value):,.2f}"
            except Exception:  # noqa: BLE001 - fallback formatting
                return "—"

        for symbol in symbols:
            tick = latest.get(symbol) or {}
            ltp = tick.get("ltp")
            bid = tick.get("bid")
            ask = tick.get("ask")

            age_value: float | None = None
            ingest_mono = tick.get("ingest_mono_s") or tick.get("_ingest_mono")
            if isinstance(ingest_mono, (int, float)) and ingest_mono > 0:
                age_value = max(0.0, now_mono - float(ingest_mono))
            else:
                timestamp_raw = tick.get("timestamp") or tick.get("server_ts_s")
                if isinstance(timestamp_raw, (int, float)) and timestamp_raw > 0:
                    age_value = max(0.0, time.time() - float(timestamp_raw))

            source = last_src.get(symbol) or tick.get("_source") or "unknown"
            token = self._resolve_symbol_token(symbol)
            in_ws = bool(token is not None and token in ws_subs)
            in_poll = symbol in poll_watch if poll_watch else False
            tracked_flag = False
            if mdm is not None:
                try:
                    tracked_flag = bool(mdm.is_tracked(symbol))
                except Exception as exc:  # noqa: BLE001 - defensive tracking lookup
                    log.error(
                        "Failure reading tracking status: %s",
                        exc,
                        extra={
                            "event": "telegram_transport_rows_tracked_error",
                            "symbol": symbol,
                        },
                    )

            row = (
                f"{symbol:<18}"
                f"{_format_number(ltp):>12}"
                f"{_format_number(bid):>12}"
                f"{_format_number(ask):>12}"
                f"{self._fmt_age(age_value):>8}"
                f"{str(source)[:9]:>10}"
                f"{(str(token) if token is not None else '—'):>8}"
                f"{'yes' if in_ws else 'no':>6}"
                f"{'yes' if in_poll else 'no':>7}"
                f"{'yes' if tracked_flag else 'no':>9}"
            )
            rows.append(row)

        if not latest:
            hints.append(
                "Market data cache empty; trigger /quote or ensure feeds are active.",
            )
        return rows, hints

    def _ws_snapshot(self) -> dict[str, t.Any]:
        """Return websocket transport snapshot for diagnostics.

        Args:
            None.

        Returns:
            dict[str, t.Any]: Snapshot describing websocket availability.

        Raises:
            None.
        """

        disabled = self._ws_disabled()
        ws = None if disabled else getattr(self.deps, "websocket_manager", None)
        snapshot: dict[str, t.Any] = {
            "disabled": disabled,
            "available": ws is not None,
            "connected": False,
            "state": "disabled" if disabled else "missing",
            "heartbeat_age": None,
            "subscription_count": 0,
            "errors": [],
        }
        if ws is None:
            return snapshot
        try:
            state = ws.connection_state()
            snapshot["state"] = getattr(state, "name", str(state))
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure reading websocket state: %s",
                exc,
                extra={"event": "telegram_transport_ws_state_error"},
            )
            snapshot.setdefault("errors", []).append(str(exc))
        try:
            snapshot["connected"] = bool(ws.is_connected())
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure probing websocket connectivity: %s",
                exc,
                extra={"event": "telegram_transport_ws_connected_error"},
            )
            snapshot.setdefault("errors", []).append(str(exc))
        try:
            hb = float(ws.last_heartbeat_monotonic())
            if hb > 0:
                snapshot["heartbeat_age"] = max(0.0, time.monotonic() - hb)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure reading websocket heartbeat: %s",
                exc,
                extra={"event": "telegram_transport_ws_heartbeat_error"},
            )
            snapshot.setdefault("errors", []).append(str(exc))
        try:
            tokens = getattr(ws, "_subscription_tokens", None)
            if isinstance(tokens, set):
                snapshot["subscription_count"] = len(tokens)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure reading websocket subscriptions: %s",
                exc,
                extra={"event": "telegram_transport_ws_subscription_error"},
            )
            snapshot.setdefault("errors", []).append(str(exc))
        return snapshot

    def _polling_snapshot(self) -> dict[str, t.Any]:
        supervisor = self._stream_supervisor()
        if supervisor is not None:
            try:
                snapshot = supervisor.snapshot()
            except Exception as exc:  # pragma: no cover - defensive snapshot
                interval = float(
                    getattr(getattr(supervisor, "streamer", None), "_interval_s", 0.0)
                    or 0.0
                )
                batch = int(
                    getattr(getattr(supervisor, "streamer", None), "_batch_size", 0)
                    or 0
                )
                return {
                    "running": False,
                    "tokens": 0,
                    "interval_s": interval,
                    "batch_size": batch,
                    "status": f"error: {exc}",
                }
            status_line = ""
            with suppress(Exception):
                status_line = str(supervisor.status_line())
            if status_line:
                snapshot.setdefault("status", status_line)
            return snapshot

        streamer = self._polling_streamer()
        tokens = 0
        interval = 0.0
        batch = 0
        running = False
        if streamer is not None:
            tracked = getattr(streamer, "tracked_tokens", None)
            if callable(tracked):
                with suppress(Exception):
                    result = tracked()
                    if isinstance(result, Iterable):
                        tokens = len(list(result))
            with suppress(Exception):
                interval = float(getattr(streamer, "_interval_s", 0.0) or 0.0)
            with suppress(Exception):
                batch = int(getattr(streamer, "_batch_size", 0) or 0)
            running = self._polling_running()
        heart = "💓 running" if running else "⛔ stopped"
        status = f"tokens={tokens} @{interval:.2f}s {heart}"
        return {
            "running": running,
            "tokens": tokens,
            "interval_s": interval,
            "batch_size": batch,
            "status": status,
        }

    def _primary_symbol(self) -> str:
        """Return the most relevant trading symbol for diagnostics."""

        sm = getattr(self.deps, "strategy_manager", None)
        if sm is not None:
            with suppress(Exception):
                tracked_callable = getattr(sm, "tracked_symbols", None)
                if callable(tracked_callable):
                    tracked = list(tracked_callable())
                    for symbol in tracked:
                        if symbol:
                            return str(symbol)
            with suppress(Exception):
                strategies = getattr(sm, "strategies", None) or []
                for strategy in strategies:
                    candidate = getattr(strategy, "symbol", None)
                    if candidate:
                        return str(candidate)
        config = getattr(self.deps, "config", None)
        if config is not None:
            with suppress(Exception):
                symbols = getattr(config, "symbols", None)
                if isinstance(symbols, (list, tuple, set)):
                    for candidate in symbols:
                        if candidate:
                            return str(candidate)
                if isinstance(symbols, str) and symbols:
                    return symbols
        with suppress(Exception):
            spot_symbol = getattr(self.deps, "spot_symbol", None)
            if spot_symbol:
                return str(spot_symbol)
        return "NIFTY"

    def _poll_interval_seconds(self) -> float:
        """Return the polling interval used for adaptive freshness bounds."""

        snapshot = self._polling_snapshot()
        interval = snapshot.get("interval_s")
        if isinstance(interval, (int, float)) and interval > 0:
            return float(interval)
        streamer = self._polling_streamer()
        if streamer is not None:
            with suppress(Exception):
                raw = getattr(streamer, "_interval_s", None)
                if isinstance(raw, (int, float)) and raw > 0:
                    return float(raw)
        return 0.7

    def _adaptive_freshness_limit_ms(self) -> int:
        """Compute the adaptive freshness threshold in milliseconds."""

        poll_seconds = max(0.0, self._poll_interval_seconds())
        limit = int(poll_seconds * 1000.0 * 2.5) if poll_seconds else 0
        if limit <= 0:
            limit = 2000
        return max(2000, min(5000, limit))

    def _assess_risk_state(self) -> tuple[bool, str, dict[str, t.Any]]:
        """Evaluate soft/hard breakers from the risk snapshot."""

        snapshot = self._coerce_risk_snapshot()
        if snapshot is None:
            return False, "no_snapshot", {}
        breaker = bool(snapshot.get("breaker_tripped"))
        cooldown_raw = snapshot.get("cooldown_remaining")
        cooldown = 0.0
        if isinstance(cooldown_raw, (int, float)):
            cooldown = float(cooldown_raw)
        elif cooldown_raw is not None:
            with suppress(Exception):
                cooldown = float(t.cast(float | str, cooldown_raw))
        cooldown = max(0.0, cooldown)
        reason = snapshot.get("breaker_reason") or snapshot.get("last_rejection")
        meta: dict[str, t.Any] = {
            "breaker_tripped": breaker,
            "cooldown_remaining": cooldown,
        }
        if reason:
            meta["reason"] = str(reason)
        if breaker:
            return False, "breaker", meta
        if cooldown > 0.0:
            return False, "cooldown", meta
        meta["status"] = "ok"
        return True, "ok", meta

    def _streamer_token_count(self, streamer: t.Any) -> int:
        tracked = getattr(streamer, "tracked_tokens", None)
        if callable(tracked):
            try:
                result = tracked()
            except Exception:
                return 0
            if result is None:
                return 0
            if isinstance(result, list):
                return len(result)
            if isinstance(result, (set, tuple)):
                return len(result)
            try:
                iterable_result = t.cast(t.Iterable[t.Any], result)
                return len(list(iterable_result))
            except Exception:
                return 0
        tokens_attr = getattr(streamer, "_tokens", None)
        if isinstance(tokens_attr, (set, list, tuple)):
            return len(tokens_attr)
        if isinstance(tokens_attr, Iterable):
            try:
                iterable_tokens = t.cast(Iterable[t.Any], tokens_attr)
                return len(list(iterable_tokens))
            except Exception:
                return 0
        return 0

    def _streamer_subscribe(self, streamer: t.Any, tokens: t.Sequence[int]) -> None:
        handler = getattr(streamer, "subscribe_tokens", None)
        if callable(handler):
            handler(tokens)
            return
        handler = getattr(streamer, "subscribe", None)
        if callable(handler):
            handler(tokens)
            return
        raise RuntimeError("Polling streamer does not support subscribe")

    def _streamer_unsubscribe(self, streamer: t.Any, tokens: t.Sequence[int]) -> None:
        for name in ("unsubscribe_tokens", "unsubscribe"):
            handler = getattr(streamer, name, None)
            if callable(handler):
                handler(tokens)
                return
        raise RuntimeError("Polling streamer does not support unsubscribe")

    # ------------------------------------------------------------------
    # Consistent composer: render sections without multiplying <br> tags.
    # Each inner list is rendered with RB.bullets(); sections are joined
    # with RB.sep(). Optional hints appear as a "Hints" section last.
    # ------------------------------------------------------------------
    def _compose(
        self,
        sections: list[list[str]],
        *,
        hints: list[str] | None = None,
    ) -> str:
        rb = self._response_builder
        parts: list[str] = []
        for block in sections:
            block = [line for line in block if isinstance(line, str) and len(line) > 0]
            if not block:
                continue
            rendered = rb.bullets(block)
            if rendered:
                parts.append(rendered)
        text = rb.sep().join(parts) if parts else ""
        if hints:
            hint_lines = [rb.esc(h) for h in hints if h]
            if hint_lines:
                tail = rb.section_block("Hints", hint_lines)
                text = (text + rb.sep() + tail) if text else tail
        return text

    def _resolve_metrics(self) -> MetricsCollector | None:
        """Return the metrics collector bound to the bot instance.

        Args:
            None.

        Returns:
            MetricsCollector | None: Resolved collector or ``None`` when unavailable.

        Raises:
            None.
        """

        metrics = getattr(self.deps, "metrics", None)
        if metrics is not None:
            return metrics
        return METRICS

    def _fetch_metric_summary(self) -> dict[str, t.Any]:
        """Return cached metric aggregates for health and diagnostics.

        Args:
            None.

        Returns:
            dict[str, t.Any]: Snapshot of metric aggregates when available.

        Raises:
            None.
        """

        collector = self._resolve_metrics()
        if collector is None:
            return {}
        try:
            snapshot = collector.metric_summary()
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error(
                "Failure in _fetch_metric_summary: %s",
                exc,
                extra={"event": "metric_summary_error"},
                exc_info=exc,
            )
            return {}
        return dict(snapshot)

    def _snapshot_alerts(self) -> dict[str, dict[str, object]]:
        """Return alert state snapshot from the metrics collector.

        Args:
            None.

        Returns:
            dict[str, dict[str, object]]: Mapping of alert names to metadata.

        Raises:
            None.
        """

        collector = self._resolve_metrics()
        if collector is None:
            return {}
        try:
            alerts = collector.snapshot_alerts()
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error(
                "Failure in _snapshot_alerts: %s",
                exc,
                extra={"event": "metric_alerts_error"},
                exc_info=exc,
            )
            return {}
        return {name: dict(payload) for name, payload in alerts.items()}

    def _collect_recent_errors(
        self,
        summary: t.Mapping[str, t.Any] | None = None,
        *,
        limit: int = 5,
    ) -> dict[str, int]:
        """Return ranked error counts from metric summary data.

        Args:
            summary: Optional metric summary to reuse instead of refetching.
            limit: Maximum number of error reasons returned.

        Returns:
            dict[str, int]: Ordered mapping of error reasons to counts.

        Raises:
            None.
        """

        payload = summary or self._fetch_metric_summary()
        source = (
            payload.get("error_events_5m") if isinstance(payload, t.Mapping) else None
        )
        if not isinstance(source, t.Mapping):
            return {}
        items = [
            (str(key), int(value))
            for key, value in source.items()
            if isinstance(value, (int, float))
        ]
        items.sort(key=lambda entry: entry[1], reverse=True)
        limited = items[: max(0, int(limit))]
        return {name: count for name, count in limited}

    def _persistent_state_summary(
        self, summary: t.Mapping[str, t.Any] | None = None
    ) -> dict[str, t.Any]:
        """Return best-effort persistent state health hints.

        Args:
            summary: Optional metric summary used to enrich the output.

        Returns:
            dict[str, t.Any]: Metadata describing attachment and flush cadence.

        Raises:
            None.
        """

        runner = getattr(self.deps, "strategy_runner", None)
        attached = False
        if runner is not None:
            attached = getattr(runner, "_persistent_state", None) is not None
        manager = getattr(self.deps, "persistent_state", None)
        telemetry_snapshot: dict[str, t.Any] = {}
        if manager is not None:
            try:
                telemetry_snapshot = manager.telemetry()
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in _persistent_state_summary telemetry: %s",
                    exc,
                    extra={"event": "telegram_persistence_telemetry_error"},
                )
        metrics_summary = (
            summary if isinstance(summary, t.Mapping) else self._fetch_metric_summary()
        )
        flush_age = (
            metrics_summary.get("heartbeat_flush_age")
            if isinstance(metrics_summary, t.Mapping)
            else None
        )
        last_latency = (
            metrics_summary.get("heartbeat_last_latency")
            if isinstance(metrics_summary, t.Mapping)
            else None
        )
        flush_counts = (
            metrics_summary.get("heartbeat_flushes_5m")
            if isinstance(metrics_summary, t.Mapping)
            else None
        )
        payload: dict[str, t.Any] = {"attached": attached}
        if isinstance(flush_age, (int, float)):
            payload["flush_age_seconds"] = round(float(flush_age), 2)
        if isinstance(last_latency, (int, float)):
            payload["last_flush_latency"] = round(float(last_latency), 3)
        if isinstance(flush_counts, t.Mapping):
            payload["flush_counts"] = {
                str(key): int(value)
                for key, value in flush_counts.items()
                if isinstance(value, (int, float))
            }
        if telemetry_snapshot:
            health = telemetry_snapshot.get("health")
            if isinstance(health, str):
                payload["health"] = health
            last_write = telemetry_snapshot.get("last_write_at")
            if isinstance(last_write, str):
                payload["last_write_at"] = last_write
            last_flush = telemetry_snapshot.get("last_flush_at")
            if isinstance(last_flush, str):
                payload["last_flush_at"] = last_flush
            pending_depth = telemetry_snapshot.get("pending_queue_depth")
            if isinstance(pending_depth, (int, float)):
                payload["pending_queue_depth"] = int(pending_depth)
            mode = telemetry_snapshot.get("mode")
            if isinstance(mode, str):
                payload["mode"] = mode
            last_error = telemetry_snapshot.get("last_error")
            if isinstance(last_error, str) and last_error:
                payload["last_error"] = last_error
        return payload

    def _compute_allocation_delta(
        self, current: t.Mapping[str, float]
    ) -> dict[str, float]:
        """Return allocation delta versus the previous recorded snapshot.

        Args:
            current: Mapping of strategy name to current allocation fraction.

        Returns:
            dict[str, float]: Differences keyed by strategy (positive implies increase).

        Raises:
            None.
        """

        previous = self._last_allocation_snapshot or {}
        keys = set(previous) | {str(name) for name in current.keys()}
        delta: dict[str, float] = {}
        for key in sorted(keys):
            current_val = float(current.get(key, 0.0) or 0.0)
            previous_val = float(previous.get(key, 0.0) or 0.0)
            diff = round(current_val - previous_val, 4)
            if abs(diff) > 1e-9:
                delta[key] = diff
        return delta

    def _store_allocation_history(
        self,
        current: t.Mapping[str, float],
        delta: t.Mapping[str, float],
    ) -> None:
        """Persist allocation state and delta for future diagnostics.

        Args:
            current: Mapping of strategy names to allocations.
            delta: Mapping of deltas recently observed.

        Returns:
            None.

        Raises:
            None.
        """

        if delta:
            self._allocation_history.append(
                (datetime.now(timezone.utc), {k: float(v) for k, v in delta.items()})
            )
        self._last_allocation_snapshot = {
            str(name): float(value) for name, value in current.items()
        }

    def _update_regime_history(
        self,
        symbol: str,
        regime: str,
        confidence: float,
        updated_at: float,
    ) -> tuple[str | None, float | None, bool]:
        """Record regime transitions for diagnostics and return prior state.

        Args:
            symbol: Instrument symbol associated with the snapshot.
            regime: New regime name reported by the detector.
            confidence: Confidence score in the new regime.
            updated_at: Epoch timestamp when the snapshot was produced.

        Returns:
            tuple[str | None, float | None, bool]: Previous regime, timestamp, and
            a flag indicating if a transition occurred.

        Raises:
            None.
        """

        previous = self._regime_last_state.get(symbol)
        previous_regime = previous[0] if previous else None
        previous_ts = previous[1] if previous else None
        changed = False
        updated_after_previous = True
        if updated_at is not None and previous_ts is not None:
            updated_after_previous = updated_at >= previous_ts
        if previous is None or regime != previous_regime:
            if updated_after_previous:
                changed = True
                event_time = datetime.fromtimestamp(
                    updated_at or time.time(), timezone.utc
                )
                self._regime_transitions.append(
                    (event_time, symbol, previous_regime, regime, float(confidence))
                )
        if previous is None or updated_after_previous:
            self._regime_last_state[symbol] = (regime, float(updated_at or time.time()))
        return previous_regime, previous_ts, changed

    def _build_diag_summary(self) -> dict[str, list[str]]:
        """Construct diagnostic summary lines for the /diag command.

        Args:
            None.

        Returns:
            dict[str, list[str]]: Mapping of section names to human friendly lines.

        Raises:
            None.
        """

        summary = self._fetch_metric_summary()
        alerts_snapshot = self._snapshot_alerts()
        errors = self._collect_recent_errors(summary)
        persistence_meta = self._persistent_state_summary(summary)
        ok, label, risk_meta = self._assess_risk_state()

        alerts_lines: list[str] = []
        for name, payload in alerts_snapshot.items():
            if not isinstance(payload, dict):
                continue
            severity = str(payload.get("severity") or "").lower() or "info"
            active = "active" if payload.get("active") else "cleared"
            observed = payload.get("observed")
            threshold = payload.get("threshold")
            observed_str = (
                f" obs={float(observed):.2f}"
                if isinstance(observed, (int, float))
                else ""
            )
            threshold_str = (
                f" thr={float(threshold):.2f}"
                if isinstance(threshold, (int, float))
                else ""
            )
            alerts_lines.append(
                f"{name}: {severity} ({active}{observed_str}{threshold_str})"
            )

        error_lines = [f"{reason}×{count}" for reason, count in errors.items()]

        allocation_lines: list[str] = []
        for timestamp, delta in list(self._allocation_history)[-3:]:
            formatted = ", ".join(
                f"{name}:{change:+.2f}" for name, change in delta.items()
            )
            allocation_lines.append(f"{timestamp:%H:%M:%S}Z {formatted}")

        regime_lines: list[str] = []
        for event_time, symbol, previous, current, confidence in list(
            self._regime_transitions
        )[-5:]:
            prefix = f"{event_time:%H:%M:%S}Z {symbol}: "
            if previous is None:
                regime_lines.append(f"{prefix}{current} (conf={confidence:.2f})")
            else:
                regime_lines.append(
                    f"{prefix}{previous}→{current} (conf={confidence:.2f})"
                )

        state_lines: list[str] = []
        if risk_meta:
            detail_bits = [
                f"{key}={value}"
                for key, value in risk_meta.items()
                if key not in {"status"}
            ]
            details = f" ({', '.join(detail_bits)})" if detail_bits else ""
            state_lines.append(f"Risk: {label}{details}")
        if persistence_meta:
            parts: list[str] = []
            if persistence_meta.get("attached"):
                parts.append("attached")
            flush_age_val = persistence_meta.get("flush_age_seconds")
            if isinstance(flush_age_val, (int, float)):
                parts.append(f"age={float(flush_age_val):.1f}s")
            latency_val = persistence_meta.get("last_flush_latency")
            if isinstance(latency_val, (int, float)):
                parts.append(f"latency={float(latency_val):.3f}s")
            if parts:
                state_lines.append(f"Persistence: {', '.join(parts)}")

        hints: list[str] = []
        if not ok and risk_meta.get("reason"):
            hints.append(str(risk_meta["reason"]))

        return {
            "alerts": alerts_lines,
            "errors": error_lines,
            "allocations": allocation_lines,
            "regimes": regime_lines,
            "state": state_lines,
            "hints": hints,
        }

    def _diag_highlights(
        self,
        *,
        summary: dict[str, list[str]] | None = None,
        plain: bool = True,
        limit: int = 2,
    ) -> list[str]:
        """Return condensed diagnostic highlights for command responses.

        Args:
            summary: Optional precomputed diagnostic summary to reuse.
            plain: ``True`` to emit plain text, ``False`` for HTML-safe strings.
            limit: Maximum entries per diagnostic category.

        Returns:
            list[str]: Highlight strings covering errors, regimes, allocations, and
            persistent state.

        Raises:
            None.
        """

        try:
            diag = summary or self._build_diag_summary()
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error(
                "Failure in _diag_highlights: %s",
                exc,
                extra={"event": "diag_highlights_error"},
                exc_info=exc,
            )
            return []
        rb = self._response_builder

        def _fmt(values: list[str], *, reverse: bool = False) -> str:
            ordered = list(reversed(values)) if reverse else list(values)
            subset = ordered[: max(1, limit)]
            joined = " | ".join(subset)
            return rb.esc(joined) if not plain else joined

        lines: list[str] = []
        errors = diag.get("errors") or []
        if errors:
            snippet = ", ".join(errors[: max(1, limit)])
            lines.append(f"Errors 5m: {rb.esc(snippet) if not plain else snippet}")
        allocations = diag.get("allocations") or []
        if allocations:
            lines.append(f"AllocΔ: {_fmt(allocations, reverse=True)}")
        regimes = diag.get("regimes") or []
        if regimes:
            lines.append(f"Regime shifts: {_fmt(regimes, reverse=True)}")
        state = diag.get("state") or []
        if state:
            lines.append(f"State: {_fmt(state)}")
        return lines

    def _register_command_doc(
        self,
        cmd: str,
        handler: t.Callable[[Update, t.Any], t.Coroutine[t.Any, t.Any, t.Any]],
    ) -> None:
        registry = self._help_registry
        key = id(handler)
        usage = str(getattr(handler, "__cmd__", "")).strip()
        help_text = str(getattr(handler, "__help__", "")).strip()
        entry = registry.setdefault(
            key,
            {
                "usage": usage or f"/{cmd}",
                "help": help_text,
                "commands": [],
            },
        )
        if not entry["usage"]:
            entry["usage"] = f"/{cmd}"
        if help_text and not entry["help"]:
            entry["help"] = help_text
        entry["commands"].append(f"/{cmd}")

    def _log_limits(self) -> tuple[int, int]:
        env_default = os.getenv("LOG_MAX_LINES_REPLY", "")
        try:
            env_limit = int(env_default)
        except (TypeError, ValueError):
            env_limit = 30
        env_limit = max(1, env_limit)
        return env_limit, min(50, env_limit)

    def _extract_query_params(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> dict[str, str]:
        text = (update.effective_message.text or "") if update.effective_message else ""
        fragments: list[str] = []
        if "?" in text:
            fragments.append(text.split("?", 1)[1])
        for arg in ctx.args or []:
            cleaned = arg.lstrip("?")
            if "=" in cleaned:
                fragments.append(cleaned)
        params: dict[str, str] = {}
        for fragment in fragments:
            for token in fragment.split("&"):
                token = token.strip().lstrip("?")
                if not token or "=" not in token:
                    continue
                key, value = token.split("=", 1)
                params[key.lower()] = value
        return params

    def _pagination_args(
        self,
        update: Update,
        ctx: ContextTypes.DEFAULT_TYPE,
        *,
        default_limit: int | None = None,
    ) -> tuple[int, int]:
        env_cap, env_default = self._log_limits()
        limit = default_limit if default_limit is not None else env_default
        page = 1
        params = self._extract_query_params(update, ctx)
        page_val = params.get("page")
        if page_val:
            with suppress(Exception):
                page = max(1, int(page_val))
        limit_val = params.get("limit")
        if limit_val:
            with suppress(Exception):
                limit = max(1, min(int(limit_val), min(env_cap, 200)))
        else:
            limit = max(1, min(limit, env_cap))
        return page, limit

    # ------------------
    # Rendering helpers
    # ------------------
    _TAG_RE = re.compile(r"<[^>]+>")
    _BR_RE = re.compile(r"(?i)<br\s*/?>")
    _PRE_OPEN_RE = re.compile(r"(?i)<pre[^>]*>")
    _PRE_CLOSE_RE = re.compile(r"(?i)</pre>")

    _SECTION_HEADERS = (
        "Health",
        "Check",
        "Flow",
        "Status",
        "Bot Status",
        "Issues",
        "Last Error",
        "Hints",
    )
    _EMOJI_PREFIX = {
        "WS:": "🌐 ",
        "Poll:": "🔁 ",
        "Orders:": "🧾 ",
        "Positions:": "📊 ",
        "Telemetry:": "📥 ",
        "Guard:": "🕘 ",
    }

    def _beautify_plain(self, text: str) -> str:
        """Improve readability for plain Telegram messages."""
        if not text:
            return ""

        raw_lines = [ln.rstrip() for ln in text.split("\n")]
        cooked: list[str] = []

        def _is_section_header(line: str) -> bool:
            candidate = line.strip()
            if not candidate:
                return False
            if any(candidate.startswith(header) for header in self._SECTION_HEADERS):
                return True
            return bool(
                re.match(
                    r"^(Health|Check|Flow|Status|Bot Status|Issues|Last Error)\b",
                    candidate,
                )
            )

        def _prefix_emoji(line: str) -> str:
            trimmed = line.lstrip()
            for key, emoji in self._EMOJI_PREFIX.items():
                if trimmed.startswith(key) and not trimmed.startswith(emoji):
                    return f"{emoji}{trimmed}"
            return trimmed

        prev_blank = True
        for raw_line in raw_lines:
            line = raw_line.strip()
            if not line:
                if not prev_blank:
                    cooked.append("")
                prev_blank = True
                continue

            line = _prefix_emoji(line)
            cooked.append(line)
            prev_blank = False

            if _is_section_header(line):
                cooked.append("")
                prev_blank = True

        out: list[str] = []
        blank = False
        for cooked_line in cooked:
            if cooked_line == "":
                if not blank:
                    out.append("")
                blank = True
                continue

            out.append(cooked_line)
            blank = False

        return "\n".join(out).strip()

    def _html_to_plain(self, text: str) -> str:
        """Convert HTML-formatted Telegram message to clean plain text."""
        if not text:
            return ""
        content = str(text)
        content = self._PRE_OPEN_RE.sub("", content)
        content = self._PRE_CLOSE_RE.sub("", content)
        content = self._BR_RE.sub("\n", content)
        content = self._TAG_RE.sub("", content)
        content = _html.unescape(content)
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        content = re.sub(r"\n{3,}", "\n\n", content)
        return self._beautify_plain(content.strip())

    # ---------- status helpers ----------
    def _ws_disabled(self) -> bool:
        explicit = getattr(self.deps, "websocket_enabled", None)
        if explicit is not None:
            return not bool(explicit)
        return coalesce_bool("WEBSOCKET__DISABLED", "WEBSOCKET_DISABLED", default=False)

    def _polling_alive(self) -> bool:
        """Return True if the fallback polling task is active."""

        task = self._polling_task
        return bool(task is not None and not task.done())

    def _polling_running(self) -> bool:
        if self._polling_alive():
            return True
        supervisor = self._stream_supervisor()
        if supervisor is not None:
            get_health = getattr(supervisor, "get_health", None)
            if callable(get_health):
                with suppress(Exception):
                    health = get_health()
                    running = getattr(health, "running", None)
                    if running is not None:
                        return bool(running)
            try:
                return bool(supervisor.is_running())
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure probing polling supervisor state: %s",
                    exc,
                    extra={"event": "telegram_poll_supervisor_error"},
                )
                with suppress(Exception):
                    snapshot = supervisor.snapshot()
                    running = snapshot.get("running")
                    if isinstance(running, bool):
                        return running
                return False
        streamer = getattr(self.deps, "streamer", None)
        if streamer is None:
            return False
        is_running = getattr(streamer, "is_running", None)
        if callable(is_running):
            try:
                return bool(is_running())
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure probing streamer.is_running: %s",
                    exc,
                    extra={"event": "telegram_streamer_running_error"},
                )
        thread = getattr(streamer, "_thread", None)
        if thread is not None:
            is_alive = getattr(thread, "is_alive", None)
            if callable(is_alive):
                try:
                    return bool(is_alive())
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure probing streamer thread state: %s",
                        exc,
                        extra={"event": "telegram_streamer_thread_error"},
                    )
            return bool(thread)
        for attr in ("started", "_started"):
            if hasattr(streamer, attr):
                return bool(getattr(streamer, attr))
        return False

    # ------------------
    # Command handlers
    # ------------------
    async def _reply(
        self,
        chat: Chat,
        ctx: ContextTypes.DEFAULT_TYPE,
        text: str,
        **kwargs: t.Any,
    ) -> Message:
        kwargs.setdefault("disable_web_page_preview", True)
        messenger = self._ensure_messenger(ctx, chat)
        parse_mode_arg = kwargs.pop("parse_mode", None)
        message_text = str(text)
        if self._plain_mode:
            plain = self._html_to_plain(message_text)
            result = await messenger.send_text(
                int(chat.id),
                plain,
                html_ready=True,
                parse_mode=None,
                **kwargs,
            )
            if result is None:  # pragma: no cover - defensive
                raise RuntimeError("Failed to dispatch Telegram message")
            return result

        parse_mode = parse_mode_arg or ParseMode.HTML
        try:
            send_html = getattr(messenger, "send_html", None)
            if callable(send_html):
                result = await send_html(
                    int(chat.id),
                    message_text,
                    parse_mode=parse_mode,
                    **kwargs,
                )
            else:
                result = await messenger.send_text(
                    int(chat.id),
                    message_text,
                    html_ready=True,
                    parse_mode=parse_mode,
                    **kwargs,
                )
        except TelegramError:
            plain = self._html_to_plain(message_text)
            result = await messenger.send_text(
                int(chat.id),
                plain,
                html_ready=True,
                parse_mode=None,
                **kwargs,
            )
        if result is None:  # pragma: no cover - defensive
            raise RuntimeError("Failed to dispatch Telegram message")
        return result

    async def _send_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        rb = self._response_builder
        lines: list[str] = [f"{rb.section('Commands')} {EMOJI['info']}"]
        for entry in self._help_registry.values():
            commands = entry.get("commands") or []
            usage = str(entry.get("usage") or "").strip()
            if not usage.startswith("/"):
                prefix = commands[0] if commands else "/?"
                usage = f"{prefix} {usage}".strip()
            primary = usage.split()[0] if usage else (commands[0] if commands else "/?")
            alias_list = sorted({cmd for cmd in commands if cmd != primary})
            alias_html = ""
            if alias_list:
                alias_html = (
                    " <i>aka "
                    + ", ".join(f"<code>{rb.esc(alias)}</code>" for alias in alias_list)
                    + "</i>"
                )
            desc = str(entry.get("help") or "").strip() or "—"
            lines.append(f"<code>{rb.esc(usage)}</code>{alias_html} – {rb.esc(desc)}")
        if len(lines) == 1:
            lines.append("No commands registered.")
        lines.append(
            (
                "<i>Use ?page= and ?limit= for long outputs (e.g., "
                "/errors?page=2&limit=50).</i>"
            )
        )
        await self._reply(chat, ctx, "<br>".join(lines), parse_mode=ParseMode.HTML)
        return

    @command_meta("/opshelp", "Command reference (alias of /help).")
    async def cmd_opshelp(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await self._send_help(update, ctx)

    @command_meta("/help", "Command reference with descriptions.")
    async def cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await self._send_help(update, ctx)

    @command_meta("/ping", "Telegram connectivity test")
    async def cmd_ping(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Reply with pong to validate Telegram command routing.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        try:
            chat = await self._guard(update)
            if chat is None:
                return
            await self._reply(chat, ctx, "pong")
        except Exception as exc:  # noqa: BLE001
            log.error("Failure in cmd_ping: %s", exc, exc_info=exc)

    async def cmd_whoami(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        broker = self.deps.broker_client
        name = "unknown"
        with suppress(Exception):
            if broker is not None:
                prof = broker.get_profile()  # type: ignore[attr-defined]
                name = prof.get("user_name") or prof.get("user_name", "unknown")
        await self._reply(chat, ctx, f"👤 {name}")

    async def cmd_version(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Report build metadata, mode, and uptime in a compact summary.

        Args:
            update: Telegram update triggering the command.
            ctx: Handler context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_version",
            extra={"event": "telegram_cmd_version_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        try:
            version = getattr(self.deps, "app_version", None) or "dev"
            build_sha = os.getenv("GIT_SHA", "unknown")
            build_time = os.getenv("BUILD_TIME", "unknown")
            strategy_count = 0
            tracked_symbols: list[str] = []
            sm = getattr(self.deps, "strategy_manager", None)
            if sm is not None:
                with suppress(Exception):
                    strategies = getattr(sm, "strategies", None)
                    if strategies is not None:
                        strategy_count = len(list(strategies))
                    else:
                        getter = getattr(sm, "get_all_strategies", None)
                        if callable(getter):
                            strategy_count = len(list(getter()))
                with suppress(Exception):
                    tracker = getattr(sm, "get_tracked_symbols", None)
                    if callable(tracker):
                        tracked_symbols = sorted(
                            str(symbol) for symbol in list(tracker()) if symbol
                        )
            uptime = self._format_uptime()
            mode = self._mode_badge_text()
            python_version = platform.python_version()
            try:
                started_at = self._started_at.astimezone(timezone.utc)
                started_text = f"{started_at:%Y-%m-%d %H:%M:%S %Z}"
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_version start time: %s",
                    exc,
                    extra={"event": "telegram_cmd_version_started_error"},
                    exc_info=exc,
                )
                started_text = "unknown"
            lines = [
                "🤖 Nifty Scalper Bot",
                f"Version: {version}",
                f"Build: {build_sha}",
                f"Mode: {mode}",
                f"Uptime: {uptime}",
                f"Python: {python_version}",
                f"Started: {started_text}",
                f"Strategies: {strategy_count}",
                f"Build time: {build_time}",
            ]
            if tracked_symbols:
                lines.append(f"Symbols: {', '.join(tracked_symbols)}")
            await self._reply(chat, ctx, "\n".join(lines))
            log.info(
                "Condition met: telegram_cmd_version_success",
                extra={"event": "telegram_cmd_version_success"},
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_version: %s",
                exc,
                extra={"event": "telegram_cmd_version_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unable to resolve version metadata.")

    @command_meta(
        "/quick",
        "Compact summary of mode, P&L, positions, and risk alerts.",
    )
    async def cmd_quick(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Provide a concise status snapshot for mobile operators.

        Args:
            update: Telegram update triggering the command.
            ctx: Handler context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_quick",
            extra={"event": "telegram_cmd_quick_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        try:
            mode_icon = self._mode_status_icon()
            pm = getattr(self.deps, "position_manager", None)
            pnl_value: float | None = None
            open_positions: int | None = None
            if pm is not None:
                with suppress(Exception):
                    pnl_candidate = getattr(pm, "get_net_pnl", lambda: None)()
                    if isinstance(pnl_candidate, (int, float)):
                        pnl_value = float(pnl_candidate)
                raw_positions: t.Iterable[t.Any] = []
                with suppress(Exception):
                    raw_positions = getattr(pm, "get_open_positions", lambda: [])()
                    open_positions = len(list(raw_positions))
            pnl_text = f"₹{pnl_value:.0f}" if pnl_value is not None else "₹n/a"
            pos_text = str(open_positions) if open_positions is not None else "n/a"
            breaker_icon = "🟢"
            breaker_tripped = False
            cooldown_seconds = 0.0
            risk_suffix = ""
            with suppress(Exception):
                snapshot = self._coerce_risk_snapshot() or {}
                breaker_tripped = bool(snapshot.get("breaker_tripped"))
                cooldown_value = snapshot.get("cooldown_remaining")
                if isinstance(cooldown_value, (int, float)):
                    cooldown_seconds = max(float(cooldown_value), 0.0)
                if breaker_tripped:
                    breaker_icon = "🔴"
                    risk_suffix = " 🚨Breaker"
                elif cooldown_seconds > 0:
                    breaker_icon = "🟠"
                    risk_suffix = f" ⚠️Cooldown {int(round(cooldown_seconds))}s"
            message = (
                f"{mode_icon} {breaker_icon} | P&L: {pnl_text} "
                f"| Pos: {pos_text}{risk_suffix}"
            )
            await self._reply(chat, ctx, message)
            log.info(
                "Condition met: telegram_cmd_quick_success",
                extra={
                    "event": "telegram_cmd_quick_success",
                    "breaker_tripped": breaker_tripped,
                    "cooldown_seconds": cooldown_seconds,
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_quick: %s",
                exc,
                extra={"event": "telegram_cmd_quick_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unable to build quick status snapshot.")

    @command_meta(
        "/session",
        "Session guard readiness, overrides, and failure reasons.",
    )
    async def cmd_session(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Send trading session readiness snapshot to Telegram.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Handler context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_session",
            extra={"event": "telegram_cmd_session_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_session_guard_blocked",
                extra={"event": "telegram_cmd_session_guard_blocked"},
            )
            return
        try:
            rb = self._response_builder
            status = self._session_guard_status() or {}
            if not status:
                log.info(
                    "Condition met: telegram_cmd_session_no_status",
                    extra={"event": "telegram_cmd_session_no_status"},
                )
                await self._reply(chat, ctx, "Session guard unavailable.")
                return

            def _coerce_flag(raw: t.Any) -> bool | None:
                if raw is None:
                    return None
                if isinstance(raw, bool):
                    return raw
                if isinstance(raw, (int, float)):
                    return bool(raw)
                if isinstance(raw, str):
                    lowered = raw.strip().lower()
                    if lowered in {"1", "true", "yes", "on"}:
                        return True
                    if lowered in {"0", "false", "no", "off"}:
                        return False
                    return None
                return bool(raw)

            def _fmt_bool(flag: bool | None) -> str:
                if flag is None:
                    return "n/a"
                return "yes" if flag else "no"

            market_open_flag = _coerce_flag(status.get("market_open"))
            override_raw = _coerce_flag(status.get("override_out_of_hours"))
            override_flag = bool(override_raw)
            session_valid_flag = _coerce_flag(status.get("session_valid")) or False
            rate_ok_flag = _coerce_flag(status.get("rate_limits_ok"))
            risk_ok_flag = _coerce_flag(status.get("risk_green"))
            broker_flag = _coerce_flag(status.get("broker_session_valid"))
            fail_reason = status.get("session_fail_reason")
            timestamp = status.get("timestamp")
            reasons_raw = status.get("reasons") or []
            reasons = [
                self._short_reason(str(reason))
                for reason in reasons_raw
                if str(reason).strip()
            ]

            session_ok = (
                session_valid_flag
                and (rate_ok_flag if rate_ok_flag is not None else True)
                and (risk_ok_flag if risk_ok_flag is not None else True)
                and (bool(market_open_flag) or bool(override_flag))
            )
            mode_badge = self._mode_badge_text()
            state_label = "ACTIVE" if session_ok else "BLOCKED"
            state_line = f"{EMOJI['session']} {rb.badge(session_ok, state_label)}"
            detail_rows = [
                ("Market open", _fmt_bool(market_open_flag)),
                ("Override", _fmt_bool(override_raw)),
                (
                    "Rate limits",
                    (
                        "ok"
                        if rate_ok_flag is True
                        else ("n/a" if rate_ok_flag is None else "low")
                    ),
                ),
                (
                    "Risk",
                    (
                        "green"
                        if risk_ok_flag is True
                        else ("n/a" if risk_ok_flag is None else "alert")
                    ),
                ),
                (
                    "Broker session",
                    (
                        "valid"
                        if broker_flag is True
                        else ("n/a" if broker_flag is None else "stale")
                    ),
                ),
            ]
            detail_block = rb.grid(detail_rows)

            timestamp_line = ""
            if timestamp:
                timestamp_line = (
                    f"{EMOJI['info']} Updated: <code>{rb.esc(timestamp)}</code>"
                )

            reason_lines: list[str] = []
            if fail_reason:
                reason_lines.append(
                    f"{EMOJI['warn']} {rb.esc(self._short_reason(str(fail_reason)))}"
                )
            for reason in reasons:
                reason_lines.append(f"{EMOJI['warn']} {rb.esc(reason)}")
            reason_block = ""
            if reason_lines:
                reason_block = (
                    f"{rb.section('Reasons')}{rb.br()}" f"{'<br>'.join(reason_lines)}"
                )

            risk_line = ""
            risk_snapshot = self._coerce_risk_snapshot()
            if risk_snapshot is not None:
                breaker_flag = _coerce_flag(risk_snapshot.get("breaker_tripped"))
                breaker_label = "TRIPPED" if breaker_flag else "CLEAR"
                losses = risk_snapshot.get("losses_in_row")
                cooldown_raw = risk_snapshot.get("cooldown_remaining")
                cooldown: float | None = None
                if isinstance(cooldown_raw, (int, float)):
                    cooldown = max(float(cooldown_raw), 0.0)
                daily_limit = risk_snapshot.get("daily_loss_limit")
                daily_realized = risk_snapshot.get("daily_realized")
                remaining: float | None = None
                if isinstance(daily_limit, (int, float)) and isinstance(
                    daily_realized, (int, float)
                ):
                    remaining = float(daily_limit) - abs(float(daily_realized))
                details: list[str] = []
                if losses is not None:
                    details.append(f"losses={losses}")
                if cooldown is not None:
                    details.append(f"cooldown={cooldown:.0f}s")
                if remaining is not None:
                    details.append(f"buffer=₹{remaining:,.0f}")
                suffix = (
                    " " + " ".join(rb.esc(part) for part in details) if details else ""
                )
                risk_line = (
                    f"{EMOJI['risk']} Breaker: <b>{rb.esc(breaker_label)}</b>{suffix}"
                )

            parts: list[str] = [
                f"{rb.section('Session Overview')} {mode_badge}",
                state_line,
                detail_block,
            ]
            if timestamp_line:
                parts.append(timestamp_line)
            if risk_line:
                parts.append(risk_line)
            if reason_block:
                parts.append(reason_block)

            message = rb.br().join(parts)
            await self._reply(chat, ctx, message, parse_mode=ParseMode.HTML)
            log.info(
                "Condition met: telegram_cmd_session_success",
                extra={
                    "event": "telegram_cmd_session_success",
                    "session_ok": session_ok,
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_session: %s",
                exc,
                extra={"event": "telegram_cmd_session_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unable to fetch session status.")

    @command_meta(
        "/status",
        "Snapshot of mode, session, streams, risk, orders, and limits.",
    )
    async def cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Send the consolidated bot status snapshot to Telegram.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_status",
            extra={"event": "telegram_cmd_status_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_status_guard_blocked",
                extra={"event": "telegram_cmd_status_guard_blocked"},
            )
            return
        try:
            rb = self._response_builder
            risk_manager = getattr(self.deps, "risk_manager", None)
            balance_value, balance_source = self._risk_balance_snapshot(
                force=True, risk_manager=risk_manager
            )
            balance_text = self._format_currency(balance_value)
            balance_label = rb.esc(balance_source)
            if risk_manager is None:
                balance_line = (
                    f"{EMOJI['risk']} Balance: <i>{rb.esc(balance_text)}</i> "
                    f"({balance_label})"
                )
            else:
                balance_line = (
                    f"{EMOJI['risk']} Balance: <b>{rb.esc(balance_text)}</b> "
                    f"({balance_label})"
                )
            get_sm = self.deps.get_shadow_mode
            shadow = bool(get_sm and get_sm())
            mode_badge = (
                f"{EMOJI['shadow']} SHADOW" if shadow else f"{EMOJI['live']} LIVE"
            )

            guard_status = self._session_guard_status()
            guard_keys = (
                "session_valid",
                "rate_limits_ok",
                "market_open",
                "risk_green",
            )
            session_ok = guard_status is not None and all(
                bool(guard_status.get(key)) for key in guard_keys
            )
            session_label = (
                "PASS"
                if session_ok
                else ("FAIL" if guard_status is not None else "UNKNOWN")
            )
            market_open: bool | None = None
            override = False
            session_notes: list[str] = []
            session_hint = ""
            reason_line = ""
            reasons: list[str] = []
            if guard_status:
                raw_market_open = guard_status.get("market_open")
                if raw_market_open is not None:
                    market_open = bool(raw_market_open)
                override = bool(guard_status.get("override_out_of_hours"))
                session_notes.append(f"override={override}")
                if market_open is not None:
                    session_notes.append(f"market_open={market_open}")
                reasons = [str(reason) for reason in guard_status.get("reasons") or []]
                if reasons:
                    reason_line = (
                        f"{EMOJI['info']} Reasons: "
                        f"<i>{rb.esc('; '.join(reasons))}</i>"
                    )
                if market_open is False and not override:
                    session_hint = (
                        "Set SESSION_ALLOW_OUT_OF_HOURS=true for lab testing."
                    )

            if not session_ok:
                log.info(
                    "Condition met: telegram_cmd_status_guard_degraded",
                    extra={
                        "event": "telegram_cmd_status_guard_degraded",
                        "reasons": reasons,
                    },
                )

            session_line = f"{EMOJI['session']} Session: <b>{session_label}</b>"
            if session_notes:
                session_line += f" ({rb.esc('; '.join(session_notes))})"

            ws_disabled = self._ws_disabled()
            ws = None if ws_disabled else self.deps.websocket_manager
            streamer = self._polling_streamer()
            supervisor = self._stream_supervisor()
            ws_label = "disabled" if ws_disabled else "missing"
            ws_details: list[str] = []
            ws_tokens: int | None = None
            hb_age: float | None = None
            ws_ok = ws_disabled
            hints: list[str] = []
            if risk_manager is None:
                hints.append("Risk manager unavailable; broker balance not refreshed.")
            else:
                if balance_source in {"env", "unknown"}:
                    hints.append("Broker balance unavailable; using fallback capital.")
                elif balance_source == "cache":
                    hints.append(
                        "Broker balance sourced from cache; confirm live feed."
                    )
            if ws is not None:
                ws_state = "unknown"
                with suppress(Exception):
                    state = ws.connection_state()
                    ws_state = (
                        getattr(state, "name", str(state))
                        if state is not None
                        else "unknown"
                    )
                with suppress(Exception):
                    ws_ok = bool(ws.is_connected())
                with suppress(Exception):
                    hb = float(ws.last_heartbeat_monotonic())
                    if hb > 0:
                        hb_age = max(0.0, time.monotonic() - hb)
                with suppress(Exception):
                    tokens = getattr(ws, "_subscription_tokens", None)
                    if isinstance(tokens, set):
                        ws_tokens = len(tokens)
                ws_label = "ok" if ws_ok else ws_state.lower()
                if hb_age is not None:
                    ws_details.append(f"hbΔ={hb_age:.1f}s")
                    if hb_age > 5.0 and (market_open or market_open is None):
                        hints.append("Websocket heartbeat stale (>5s).")
                if ws_tokens is not None:
                    ws_details.append(f"tokens={ws_tokens}")
            ws_line = f"{EMOJI['ws']} WS: <b>{rb.esc(ws_label)}</b>"
            if ws_details:
                ws_line += f" <i>{rb.esc(', '.join(ws_details))}</i>"

            poll_available = supervisor is not None or streamer is not None
            poll_snapshot: dict[str, t.Any] | None = None
            if poll_available:
                poll_snapshot = self._polling_snapshot()
            poll_tokens = int(poll_snapshot.get("tokens", 0)) if poll_snapshot else 0
            poll_interval_raw = (
                poll_snapshot.get("interval_s") if poll_snapshot else None
            )
            poll_interval: float | None = None
            if isinstance(poll_interval_raw, (int, float)) and poll_interval_raw > 0:
                poll_interval = float(poll_interval_raw)
            poll_batch_raw = poll_snapshot.get("batch_size") if poll_snapshot else None
            poll_batch: int | None = None
            if isinstance(poll_batch_raw, (int, float)) and int(poll_batch_raw) > 0:
                poll_batch = int(poll_batch_raw)
            poll_label = f"tokens={poll_tokens}" if poll_available else "off"
            poll_line = f"{EMOJI['poll']} Poll: <b>{rb.esc(poll_label)}</b>"
            poll_details: list[str] = []
            if poll_interval:
                poll_details.append(f"@{poll_interval:.2f}s")
            if poll_batch:
                poll_details.append(f"batch={poll_batch}")
            if poll_details:
                poll_line += " " + " ".join(rb.esc(detail) for detail in poll_details)
            poll_alive = bool(poll_snapshot.get("running")) if poll_snapshot else False
            if not poll_alive:
                poll_alive = self._polling_alive()
            heart = "💓 running" if poll_alive else "⛔ stopped"
            poll_line += f" {heart}"
            if not ws_ok and not ws_disabled and poll_tokens == 0:
                hints.append("Enable polling or subscribe tokens.")

            risk_line = f"{EMOJI['risk']} Risk: <b>n/a</b>"
            risk_snapshot = self._coerce_risk_snapshot()
            if risk_snapshot is not None:
                breaker = bool(risk_snapshot.get("breaker_tripped"))
                losses = risk_snapshot.get("losses_in_row")
                cooldown = risk_snapshot.get("cooldown_remaining")
                cooldown_txt = "n/a"
                if isinstance(cooldown, (int, float)):
                    cooldown_txt = f"{float(cooldown):.0f}s"
                risk_line = (
                    f"{EMOJI['risk']} Risk: "
                    f"breaker=<b>{'ON' if breaker else 'OFF'}</b> "
                    f"losses={rb.esc(str(losses) if losses is not None else '0')} "
                    f"cooldown={rb.esc(cooldown_txt)}"
                )
                if breaker:
                    reason = risk_snapshot.get("breaker_reason") or risk_snapshot.get(
                        "last_rejection"
                    )
                    if reason:
                        hints.append(
                            f"Risk breaker reason: {self._short_reason(str(reason))}"
                        )

            orders_line = f"{EMOJI['orders']} Orders: <b>n/a</b>"
            om = self.deps.order_manager
            if om is not None:
                try:
                    recent_orders: list[t.Any] = om.recent_orders(limit=10)
                except Exception as exc:  # noqa: BLE001
                    orders_line = (
                        f"{EMOJI['orders']} Orders: <b>error</b> "
                        f"<i>{rb.esc(self._short_reason(str(exc)))}</i>"
                    )
                else:
                    pending = 0
                    for item in recent_orders:
                        status = ""
                        if isinstance(item, dict):
                            status = str(item.get("status", ""))
                        else:
                            status = str(item)
                        if "PEND" in status.upper():
                            pending += 1
                    orders_line = (
                        f"{EMOJI['orders']} Orders: recent=<b>{len(recent_orders)}</b> "
                        f"pending=<b>{pending}</b>"
                    )

            positions_line = f"{EMOJI['pos']} Positions: <b>n/a</b>"
            pm = self.deps.position_manager
            if pm is not None:
                open_count = None
                net_pnl: float | None = None
                with suppress(Exception):
                    positions: list[t.Any] = list(
                        getattr(pm, "get_open_positions", lambda: [])()
                    )
                    open_count = len(positions)
                with suppress(Exception):
                    pnl_value = getattr(pm, "get_net_pnl", lambda: None)()
                    if isinstance(pnl_value, (int, float)):
                        net_pnl = float(pnl_value)
                if open_count is not None:
                    positions_line = (
                        f"{EMOJI['pos']} Positions: open=<b>{open_count}</b>"
                    )
                else:
                    positions_line = f"{EMOJI['pos']} Positions: open=<b>n/a</b>"
                if net_pnl is not None:
                    positions_line += f" netPnL=<b>{net_pnl:.2f}</b>"

            rate_line = ""
            rate_limiter = getattr(self.deps, "rate_limiter", None)
            if rate_limiter is not None:
                with suppress(Exception):
                    snapshot = rate_limiter.snapshot()
                if snapshot:
                    rate_parts: list[str] = []
                    for name, bucket in snapshot.items():
                        if not isinstance(bucket, dict):
                            continue
                        tokens = bucket.get("tokens")
                        capacity = bucket.get("capacity")
                        if isinstance(tokens, (int, float)) and isinstance(
                            capacity, (int, float)
                        ):
                            rate_parts.append(f"{name}={tokens:.1f}/{capacity:.0f}")
                    if rate_parts:
                        rate_line = f"{EMOJI['info']} Rates: " + " ".join(
                            f"<code>{rb.esc(part)}</code>" for part in rate_parts
                        )

            switch_line = "⏸️ Trading: <b>n/a</b>"
            try:
                switch_snapshot = trading_switch().snapshot()
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_status trading switch: %s",
                    exc,
                    extra={"event": "telegram_status_switch_error"},
                )
            else:
                if switch_snapshot.can_trade:
                    switch_line = "▶️ Trading: <b>READY</b>"
                else:
                    detail = ""
                    if switch_snapshot.remaining_seconds > 0.5:
                        detail = f" ({int(round(switch_snapshot.remaining_seconds))}s)"
                    switch_line = f"⏸️ Trading: <b>PAUSED</b>{detail}"

            section_main = [
                f"{rb.section('Bot Status')}  {mode_badge}",
                session_line,
            ]
            if reason_line:
                section_main.append(reason_line)
            section_main.extend(
                [
                    ws_line,
                    poll_line,
                    risk_line,
                    balance_line,
                    switch_line,
                    orders_line,
                    positions_line,
                ]
            )
            reconcile_line, reconcile_hint = self._reconcile_status_line()
            if reconcile_line:
                section_main.append(reconcile_line)
            if rate_line:
                section_main.append(rate_line)
            if reconcile_hint:
                hints.append(reconcile_hint)
            if session_hint:
                hints.insert(0, session_hint)
            text = self._compose([section_main], hints=hints)

            await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)
            log.info(
                "Condition met: telegram_cmd_status_dispatched",
                extra={
                    "event": "telegram_cmd_status_dispatched",
                    "session_ok": session_ok,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "Failure in cmd_status: %s",
                exc,
                extra={"event": "telegram_cmd_status_error"},
                exc_info=exc,
            )
            with suppress(Exception):
                await self._reply(
                    chat,
                    ctx,
                    "Unable to build status snapshot. Inspect logs for details.",
                )

    @command_meta(
        "/health [minutes]",
        "Quick guard, stream, counts, telemetry, and recent alerts.",
    )
    async def cmd_health(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Send aggregated guard, stream, and telemetry health summary.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_health",
            extra={"event": "telegram_cmd_health_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_health_guard_blocked",
                extra={"event": "telegram_cmd_health_guard_blocked"},
            )
            return
        try:
            rb = self._response_builder
            lookback_minutes = 30.0
            if ctx.args:
                try:
                    lookback_minutes = max(5.0, float(ctx.args[0]))
                except (TypeError, ValueError):
                    await self._reply(chat, ctx, "Usage: /health [minutes]")
                    return
            get_sm = self.deps.get_shadow_mode
            shadow = bool(get_sm and get_sm())
            mode_badge = (
                f"{EMOJI['shadow']} SHADOW" if shadow else f"{EMOJI['live']} LIVE"
            )

            guard_status = self._session_guard_status()
            guard_keys = (
                "session_valid",
                "rate_limits_ok",
                "market_open",
                "risk_green",
            )
            guard_ok = guard_status is not None and all(
                bool(guard_status.get(key)) for key in guard_keys
            )
            guard_details: list[str] = []
            reasons: list[str] = []
            reason_line = ""
            market_open: bool | None = None
            session_hint = ""
            if guard_status:
                raw_market_open = guard_status.get("market_open")
                if raw_market_open is not None:
                    market_open = bool(raw_market_open)
                override = bool(guard_status.get("override_out_of_hours"))
                guard_details.append(f"override={override}")
                if market_open is not None:
                    guard_details.append(f"market_open={market_open}")
                reasons = [str(reason) for reason in guard_status.get("reasons") or []]
                if reasons:
                    reason_line = (
                        f"{EMOJI['info']} Reasons: "
                        f"<i>{rb.esc('; '.join(reasons))}</i>"
                    )
                if market_open is False and not override:
                    session_hint = (
                        "Outside trading window; set SESSION_ALLOW_OUT_OF_HOURS=true "
                        "for lab testing."
                    )
            if not guard_ok:
                log.info(
                    "Condition met: telegram_cmd_health_guard_degraded",
                    extra={
                        "event": "telegram_cmd_health_guard_degraded",
                        "reasons": reasons,
                    },
                )
            guard_line = (
                f"{EMOJI['session']} Guard: "
                f"{rb.badge(guard_ok, 'CLEAR' if guard_ok else 'BLOCKED')}"
            )
            if guard_details:
                guard_line += f" <i>{rb.esc('; '.join(guard_details))}</i>"

            ws_disabled = self._ws_disabled()
            ws = None if ws_disabled else self.deps.websocket_manager
            ws_label = "disabled" if ws_disabled else "missing"
            ws_details: list[str] = []
            ws_ok = ws_disabled
            hints: list[str] = []
            if ws is not None:
                ws_state = "unknown"
                with suppress(Exception):
                    state = ws.connection_state()
                    ws_state = (
                        getattr(state, "name", str(state))
                        if state is not None
                        else "unknown"
                    )
                with suppress(Exception):
                    ws_ok = bool(ws.is_connected())
                hb_age: float | None = None
                with suppress(Exception):
                    hb = float(ws.last_heartbeat_monotonic())
                    if hb > 0:
                        hb_age = max(0.0, time.monotonic() - hb)
                if hb_age is not None:
                    ws_details.append(f"hbΔ={hb_age:.1f}s")
                    if hb_age > 5.0 and (market_open or market_open is None):
                        hints.append("Websocket heartbeat stale (>5s).")
                with suppress(Exception):
                    tokens = getattr(ws, "_subscription_tokens", None)
                    if isinstance(tokens, set):
                        ws_details.append(f"tokens={len(tokens)}")
                ws_label = "ok" if ws_ok else ws_state.lower()
            ws_line = f"{EMOJI['ws']} WS: <b>{rb.esc(ws_label)}</b>"
            if ws_details:
                ws_line += f" <i>{rb.esc(', '.join(ws_details))}</i>"

            poll_snapshot = self._polling_snapshot()
            poll_status = str(poll_snapshot.get("status") or "off")
            poll_tokens = int(poll_snapshot.get("tokens", 0) or 0)
            poll_line = f"{EMOJI['poll']} Poll: <b>{rb.esc(poll_status)}</b>"
            if not ws_ok and not ws_disabled and poll_tokens == 0:
                hints.append("Enable polling or subscribe tokens.")

            om = self.deps.order_manager
            recent_orders: list[t.Any] = []
            pending = 0
            if om is not None:
                with suppress(Exception):
                    recent_orders = om.recent_orders(limit=10)
                for item in recent_orders:
                    status = ""
                    if isinstance(item, dict):
                        status = str(item.get("status", ""))
                    else:
                        status = str(item)
                    if "PEND" in status.upper():
                        pending += 1
            orders_line = (
                f"{EMOJI['orders']} Orders: "
                f"recent=<b>{len(recent_orders)}</b> pending=<b>{pending}</b>"
            )

            pm = self.deps.position_manager
            open_count = 0
            net_pnl: float | None = None
            if pm is not None:
                with suppress(Exception):
                    positions = list(getattr(pm, "get_open_positions", lambda: [])())
                    open_count = len(positions)
                with suppress(Exception):
                    pnl_value = getattr(pm, "get_net_pnl", lambda: None)()
                    if isinstance(pnl_value, (int, float)):
                        net_pnl = float(pnl_value)
            positions_line = f"{EMOJI['pos']} Positions: open=<b>{open_count}</b>"
            if net_pnl is not None:
                positions_line += f" netPnL=<b>{net_pnl:.2f}</b>"

            telemetry = self._metrics.snapshot()
            webhook_updates = telemetry.get("webhook_updates", 0)
            webhook_failures = telemetry.get("webhook_failures", 0)
            secret_failures = telemetry.get("webhook_secret_failures", 0)
            fallback = telemetry.get("fallback_activations", 0)
            polling_updates = telemetry.get("polling_updates", 0)
            polling_errors = telemetry.get("polling_errors", 0)
            tele_line = (
                f"{EMOJI['info']} Telemetry: "
                f"webhook={webhook_updates}/{webhook_failures} "
                f"secret={secret_failures} fallback={fallback} "
                f"poll={polling_updates}/{polling_errors}"
            )

            metric_summary = self._fetch_metric_summary()
            alerts_snapshot = self._snapshot_alerts()
            recent_errors = self._collect_recent_errors(metric_summary)
            persistence_meta = self._persistent_state_summary(metric_summary)
            regime_meta = (
                metric_summary.get("current_regime")
                if isinstance(metric_summary, dict)
                else None
            )
            extra_lines: list[str] = []

            active_alerts: list[str] = []
            for name, payload in alerts_snapshot.items():
                if not isinstance(payload, dict):
                    continue
                if payload.get("active"):
                    severity = str(payload.get("severity") or "").lower()
                    active_alerts.append(f"{name}:{severity}")
            if active_alerts:
                extra_lines.append(
                    f"Alerts active: {rb.esc(', '.join(sorted(active_alerts)))}"
                )

            if recent_errors:
                formatted_errors = ", ".join(
                    f"{reason}×{count}" for reason, count in recent_errors.items()
                )
                extra_lines.append(f"Errors 5m: {rb.esc(formatted_errors)}")

            if isinstance(regime_meta, dict):
                regime_name = regime_meta.get("regime") or "unknown"
                confidence_val = regime_meta.get("confidence")
                updated_at_val = regime_meta.get("updated_at")
                timestamp_hint = ""
                if isinstance(updated_at_val, (int, float)) and updated_at_val > 0:
                    ts = datetime.fromtimestamp(updated_at_val, timezone.utc)
                    timestamp_hint = f" @ {ts:%H:%M:%S}Z"
                extra_lines.append(
                    (
                        f"Regime: {rb.esc(str(regime_name))} "
                        f"({float(confidence_val or 0.0):.2f}){timestamp_hint}"
                    )
                )

            if persistence_meta:
                bits: list[str] = []
                if persistence_meta.get("attached"):
                    bits.append("attached")
                flush_age_val = persistence_meta.get("flush_age_seconds")
                if isinstance(flush_age_val, (int, float)):
                    bits.append(f"age={float(flush_age_val):.1f}s")
                latency_val = persistence_meta.get("last_flush_latency")
                if isinstance(latency_val, (int, float)):
                    bits.append(f"latency={float(latency_val):.3f}s")
                flush_counts = persistence_meta.get("flush_counts")
                if isinstance(flush_counts, dict) and flush_counts:
                    parts = ", ".join(
                        f"{key}:{value}" for key, value in sorted(flush_counts.items())
                    )
                    bits.append(f"recent={parts}")
                if bits:
                    extra_lines.append(f"State: {rb.esc('; '.join(bits))}")
            checker = getattr(self.deps, "selfchecker", None)
            last_self_test = "never"
            if checker is not None:
                last_run = getattr(checker, "last_run", None)
                if isinstance(last_run, datetime):
                    last_self_test = f"{last_run:%H:%M:%S}Z"
                elif last_run:
                    last_self_test = str(last_run)
            last_trip_time = getattr(self.deps.risk_manager, "_last_trip_time", None)
            if isinstance(last_trip_time, datetime):
                last_breaker_trip = f"{last_trip_time:%H:%M:%S}Z"
            elif last_trip_time:
                last_breaker_trip = str(last_trip_time)
            else:
                last_breaker_trip = "never"
            runtime_line = (
                f"Self-Test: {rb.esc(last_self_test)} | "
                f"Breaker: {rb.esc(last_breaker_trip)}"
            )
            extra_lines.append(runtime_line)
            alloc_summaries: list[str] = []
            for timestamp, delta in list(self._allocation_history)[-2:]:
                formatted = ", ".join(
                    f"{name}:{change:+.2f}" for name, change in sorted(delta.items())
                )
                alloc_summaries.append(f"{timestamp:%H:%M:%S}Z {formatted}")
            if alloc_summaries:
                extra_lines.append(f"AllocΔ: {rb.esc(' | '.join(alloc_summaries))}")

            section_main = [
                f"{rb.section('Health')}  {mode_badge}",
                guard_line,
            ]
            if reason_line:
                section_main.append(reason_line)
            section_main.extend(
                [
                    ws_line,
                    poll_line,
                    orders_line,
                    positions_line,
                    tele_line,
                    *extra_lines,
                ]
            )
            if lookback_minutes > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(
                    minutes=lookback_minutes
                )
                alert_summaries: list[str] = []
                for entry in sorted(
                    self._aggregated_alerts.values(),
                    key=lambda item: item.last_seen,
                    reverse=True,
                ):
                    if entry.last_seen < cutoff:
                        continue
                    alert_summaries.append(
                        (
                            f"{entry.key}×{entry.count}"
                            f" (last {entry.last_seen.strftime('%H:%M:%S')}Z)"
                        )
                    )
                if alert_summaries:
                    section_main.append(
                        f"Alerts {lookback_minutes:.0f}m: "
                        f"{rb.esc('; '.join(alert_summaries))}"
                    )
            if session_hint:
                hints.insert(0, session_hint)
            text = self._compose([section_main], hints=hints)

            await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)
            log.info(
                "Condition met: telegram_cmd_health_dispatched",
                extra={
                    "event": "telegram_cmd_health_dispatched",
                    "guard_ok": guard_ok,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "Failure in cmd_health: %s",
                exc,
                extra={"event": "telegram_cmd_health_error"},
                exc_info=exc,
            )
            with suppress(Exception):
                await self._reply(
                    chat,
                    ctx,
                    "Unable to fetch health telemetry. Inspect logs for details.",
                )

    @command_meta(
        "/diag",
        "Summarize alerts, recent errors, allocations, and regimes.",
    )
    async def cmd_diag(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Reply with compact diagnostics across alerts, errors, and state.

        Args:
            update: Telegram update payload.
            ctx: PTB context associated with the incoming update.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered TelegramBot.cmd_diag",
            extra={"event": "telegram_cmd_diag"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        try:
            summary = self._build_diag_summary()
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error(
                "Failure in cmd_diag: %s",
                exc,
                extra={"event": "telegram_cmd_diag_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Diagnostics unavailable.")
            return
        rb = self._response_builder
        sections: list[list[str]] = []
        section_map = (
            ("alerts", "Alerts"),
            ("errors", "Errors"),
            ("allocations", "Allocation changes"),
            ("regimes", "Regime shifts"),
            ("state", "State"),
        )
        for key, title in section_map:
            lines = summary.get(key) or []
            if not lines:
                continue
            block = [rb.section(title)]
            block.extend(rb.esc(line) for line in lines)
            sections.append(block)
        hints = summary.get("hints") or []
        if not sections:
            await self._reply(chat, ctx, "No diagnostics available yet.")
            return
        text = self._compose(sections, hints=hints)
        try:
            await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error(
                "Failure in cmd_diag reply: %s",
                exc,
                extra={"event": "telegram_cmd_diag_reply_error"},
                exc_info=exc,
            )
        else:
            log.info(
                "Condition met: telegram_diag_sent",
                extra={
                    "event": "telegram_diag_sent",
                    "sections": len(sections),
                },
            )

    def _collect_check_results(
        self, areas: t.Sequence[str]
    ) -> list[tuple[str, tuple[str, str]]]:
        """Collect health check results for the requested areas.

        Args:
            areas: Ordered collection of health check identifiers.

        Returns:
            Ordered pairs of area name and corresponding (status, detail).

        Raises:
            KeyError: If an unknown area identifier is supplied.
        """

        log.debug("Entered _collect_check_results areas=%s", areas)
        try:

            def check_broker() -> tuple[str, str]:
                """Evaluate broker connectivity state.

                Args:
                    None.

                Returns:
                    Tuple with PASS/FAIL status and broker detail string.

                Raises:
                    None.
                """

                broker = self.deps.broker_client
                if broker is None:
                    return ("FAIL", "broker client unavailable")
                try:
                    profile = broker.get_profile()
                except Exception as exc:
                    log.error("Failure in check_broker: %s", exc, exc_info=exc)
                    return ("FAIL", self._short_reason(str(exc)))
                if not isinstance(profile, dict):
                    return ("FAIL", "unexpected profile shape")
                user_id = profile.get("user_id") or profile.get("client_id")
                name = profile.get("user_name") or profile.get("name")
                masked = _mask_token(str(user_id)) if user_id else "n/a"
                friendly = str(name) if name else "n/a"
                return ("PASS", f"user={masked} name={friendly}")

            def check_ws() -> tuple[str, str]:
                """Evaluate websocket transport readiness.

                Args:
                    None.

                Returns:
                    Tuple with PASS/FAIL status and websocket detail string.

                Raises:
                    None.
                """

                if self._ws_disabled():
                    return ("PASS", "disabled via env toggle")
                ws = self.deps.websocket_manager
                if ws is None:
                    return ("FAIL", "ws manager unavailable")
                state_name = "unknown"
                connected = False
                hb_age: float | None = None
                last_error: str | None = None
                with suppress(Exception):
                    state = ws.connection_state()
                    state_name = (
                        getattr(state, "name", str(state))
                        if state is not None
                        else "n/a"
                    )
                with suppress(Exception):
                    connected = bool(ws.is_connected())
                with suppress(Exception):
                    hb = float(ws.last_heartbeat_monotonic())
                    if hb > 0:
                        hb_age = max(0.0, time.monotonic() - hb)
                with suppress(Exception):
                    err = ws.last_error()
                    if err:
                        formatted = self._format_ws_error(err)
                        if formatted:
                            last_error = formatted
                hb_label = self._age_label(hb_age)
                detail = f"state={state_name} connected={connected} hbΔ={hb_label}"
                if last_error:
                    detail += f" last_error={last_error}"
                status = (
                    "PASS"
                    if connected and state_name.upper() == "CONNECTED"
                    else "FAIL"
                )
                return (status, detail)

            def check_mdm() -> tuple[str, str]:
                """Evaluate market data manager health.

                Args:
                    None.

                Returns:
                    Tuple with PASS/FAIL status and market data detail string.

                Raises:
                    None.
                """

                mdm_raw = self.deps.market_data_manager
                if mdm_raw is None:
                    return ("FAIL", "mdm unavailable")
                try:
                    stats = t.cast(t.Any, mdm_raw).stats()
                except Exception as exc:
                    log.error("Failure in check_mdm: %s", exc, exc_info=exc)
                    return ("FAIL", self._short_reason(str(exc)))
                symbols = stats.get("symbols")
                ws_connected = stats.get("ws_connected")
                detail = f"symbols={symbols} ws_connected={ws_connected}"
                status = "PASS" if symbols not in (None, 0) else "FAIL"
                return (status, detail)

            def check_resolver() -> tuple[str, str]:
                """Evaluate instrument resolver availability.

                Args:
                    None.

                Returns:
                    Tuple with PASS/FAIL status and resolver detail string.

                Raises:
                    None.
                """

                resolver = self.deps.instrument_resolver
                if resolver is None:
                    return ("FAIL", "resolver unavailable")
                symbol = "NIFTY"
                token = None
                with suppress(Exception):
                    token = resolver.resolve(symbol)
                if token is None:
                    with suppress(Exception):
                        store = getattr(resolver, "_by_symbol", {})
                        for key in store:
                            token = resolver.resolve(key)
                            if token is not None:
                                symbol = key
                                break
                if token is None:
                    return ("FAIL", "resolution failed for sample symbol")
                return ("PASS", f"{symbol}->{token}")

            def check_risk() -> tuple[str, str]:
                """Evaluate risk guard configuration.

                Args:
                    None.

                Returns:
                    Tuple with PASS/FAIL status and risk detail string.

                Raises:
                    None.
                """

                snapshot = self._coerce_risk_snapshot()
                if snapshot is None:
                    return ("FAIL", "snapshot unavailable")
                breaker = bool(snapshot.get("breaker_tripped"))
                losses = snapshot.get("losses_in_row")
                cooldown = snapshot.get("cooldown_remaining")
                reason = snapshot.get("breaker_reason") or snapshot.get(
                    "last_rejection"
                )
                detail = (
                    "breaker="
                    f"{'ON' if breaker else 'OFF'} losses={losses} cooldown={cooldown}"
                )
                if reason:
                    detail += f" reason={self._short_reason(str(reason))}"
                status = "FAIL" if breaker else "PASS"
                return (status, detail)

            def check_orders() -> tuple[str, str]:
                """Evaluate order manager throughput.

                Args:
                    None.

                Returns:
                    Tuple with PASS/FAIL status and order detail string.

                Raises:
                    None.
                """

                om = self.deps.order_manager
                if om is None:
                    return ("FAIL", "order manager unavailable")
                try:
                    recent_list: list[t.Any] = getattr(
                        om, "recent_orders", lambda limit=5: []
                    )(limit=5)
                except Exception as exc:
                    log.error("Failure in check_orders: %s", exc, exc_info=exc)
                    return ("FAIL", self._short_reason(str(exc)))
                pending = 0
                last_status = "n/a"
                for item in recent_list:
                    status = (
                        str(item.get("status", "")).upper()
                        if isinstance(item, dict)
                        else ""
                    )
                    if "PEND" in status:
                        pending += 1
                    if status:
                        last_status = status
                detail = (
                    f"recent={len(recent_list)} pending={pending} last={last_status}"
                )
                return ("PASS", detail)

            def check_positions() -> tuple[str, str]:
                """Evaluate open positions exposure.

                Args:
                    None.

                Returns:
                    Tuple with PASS/FAIL status and position detail string.

                Raises:
                    None.
                """

                pm = self.deps.position_manager
                if pm is None:
                    return ("FAIL", "position manager unavailable")
                try:
                    positions = list(getattr(pm, "get_open_positions", lambda: [])())
                except Exception as exc:
                    log.error("Failure in check_positions: %s", exc, exc_info=exc)
                    return ("FAIL", self._short_reason(str(exc)))
                net_pnl: float | None = None
                try:
                    pnl_value = getattr(pm, "get_net_pnl", lambda: None)()
                    if isinstance(pnl_value, (int, float)):
                        net_pnl = float(pnl_value)
                except Exception as exc:
                    log.error("Failure in check_positions pnl: %s", exc, exc_info=exc)
                exposure_note = "n/a"
                with suppress(Exception):
                    exposure_note = str(
                        getattr(pm, "exposure_summary", lambda: "n/a")()
                    )
                detail = f"open={len(positions)}"
                if net_pnl is not None:
                    detail += f" netPnL={net_pnl:.2f}"
                if exposure_note and exposure_note != "n/a":
                    detail += f" exposure={exposure_note}"
                return ("PASS", detail)

            def check_strategies() -> tuple[str, str]:
                """Evaluate strategy manager readiness.

                Args:
                    None.

                Returns:
                    Tuple with PASS/FAIL status and strategy detail string.

                Raises:
                    None.
                """

                sm = self.deps.strategy_manager
                if sm is None:
                    return ("FAIL", "strategies unavailable")
                names: list[str] = []
                with suppress(Exception):
                    names = [type(s).__name__ for s in getattr(sm, "strategies", [])]
                tracked: list[str] = []
                with suppress(Exception):
                    tracked = list(getattr(sm, "tracked_symbols", lambda: [])())
                if not names:
                    return ("FAIL", "no strategies configured")
                detail = (
                    f"strategies={','.join(names)} tracked={','.join(tracked) or 'n/a'}"
                )
                return ("PASS", detail)

            def check_session() -> tuple[str, str]:
                """Evaluate trading session guard status.

                Args:
                    None.

                Returns:
                    Tuple with PASS/FAIL status and session detail string.

                Raises:
                    None.
                """

                status = self._session_guard_status()
                if status is None:
                    return ("FAIL", "session guard unavailable")
                ok = all(
                    bool(status.get(key))
                    for key in (
                        "session_valid",
                        "rate_limits_ok",
                        "market_open",
                        "risk_green",
                    )
                )
                reasons = status.get("reasons") or []
                detail = (
                    f"market_open={status.get('market_open')} "
                    f"override={status.get('override_out_of_hours')} "
                    f"reasons={'; '.join(reasons) or 'n/a'}"
                )
                return ("PASS" if ok else "FAIL", detail)

            checks: dict[str, t.Callable[[], tuple[str, str]]] = {
                "broker": check_broker,
                "ws": check_ws,
                "mdm": check_mdm,
                "resolver": check_resolver,
                "risk": check_risk,
                "orders": check_orders,
                "positions": check_positions,
                "strategies": check_strategies,
                "session": check_session,
            }

            normalized: list[str] = []
            for name in areas:
                if name not in checks:
                    raise KeyError(name)
                normalized.append(name)

            results: list[tuple[str, tuple[str, str]]] = []
            for name in normalized:
                fn = checks[name]
                try:
                    results.append((name, fn()))
                except Exception as exc:
                    log.error(
                        "Failure in _collect_check_results for %s: %s",
                        name,
                        exc,
                        exc_info=exc,
                    )
                    results.append((name, ("FAIL", self._short_reason(str(exc)))))
            return results
        except KeyError as exc:
            unknown = exc.args[0] if exc.args else "unknown"
            log.error(
                "Failure in _collect_check_results: unknown area %s",
                unknown,
                exc_info=exc,
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            log.error("Failure in _collect_check_results: %s", exc, exc_info=exc)
            raise

    def _render_check_summary(
        self,
        results: t.Sequence[tuple[str, tuple[str, str]]],
        label: str,
    ) -> str:
        """Render a compact PASS/FAIL summary for multiple areas.

        Args:
            results: Ordered list of area names with (status, detail) tuples.
            label: Uppercase label describing the rendered selection.

        Returns:
            Rendered HTML snippet suitable for Telegram replies.

        Raises:
            None.
        """

        log.debug(
            "Entered _render_check_summary label=%s area_count=%s",
            label,
            len(results),
        )
        try:
            rb = self._response_builder
            all_ok = all(status.upper() == "PASS" for _, (status, _) in results)
            rows: list[str] = []
            fail_notes: list[str] = []
            failures: list[str] = []
            for name, (status, detail) in results:
                ok = status.upper() == "PASS"
                piece = f"{'✅' if ok else '❌'} <b>{rb.esc(name)}</b>"
                if detail:
                    short = self._short_reason(detail)
                    if short:
                        piece += f" — <i>{rb.esc(short)}</i>"
                if not ok and detail:
                    fail_notes.append(f"{name}: {detail}")
                if not ok:
                    failures.append(name)
                rows.append(piece)
            header = f"{rb.section('Check')} " f"{rb.badge(all_ok, label.upper())}"
            if failures:
                failure_names = ", ".join(rb.esc(name.upper()) for name in failures)
                summary_line = f"❌ <b>{len(failures)}</b> failing: {failure_names}"
            else:
                summary_line = "✅ All systems operational."
            header_block = [header, summary_line]
            return self._compose([header_block, rows], hints=fail_notes or None)
        except Exception as exc:  # pragma: no cover - defensive guard
            log.error("Failure in _render_check_summary: %s", exc, exc_info=exc)
            raise

    async def _handle_check_group(
        self,
        update: Update,
        ctx: ContextTypes.DEFAULT_TYPE,
        group: str,
    ) -> None:
        """Send a rendered summary for the requested check group.

        Args:
            update: Telegram update initiating the command.
            ctx: Callback context provided by python-telegram-bot.
            group: Group identifier defined in ``_CHECK_GROUPS``.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug("Entered _handle_check_group group=%s", group)
        chat = await self._guard(update)
        if chat is None:
            return
        try:
            areas = list(self._CHECK_GROUPS[group])
            log.info("Condition met: executing group check group=%s", group)
            results = self._collect_check_results(areas)
            text = self._render_check_summary(results, group.upper())
            await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)
        except Exception as exc:  # pragma: no cover - defensive guard
            log.error("Failure in _handle_check_group: %s", exc, exc_info=exc)
            await self._reply(
                chat,
                ctx,
                f"Check group '{group}' failed. Inspect logs for details.",
            )

    @command_meta(
        "/check [area]",
        "PASS/FAIL board for broker, data, and risk subsystems.",
    )
    async def cmd_check(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /check command for targeted health checks.

        Args:
            update: Telegram update initiating the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug("Entered cmd_check")
        chat = await self._guard(update)
        if chat is None:
            return
        try:
            rb = self._response_builder
            area = ctx.args[0].lower() if ctx.args else "all"
            valid = set(self._CHECK_ORDER) | set(self._CHECK_GROUPS.keys()) | {"all"}
            if area not in valid:
                options = ", ".join(sorted(valid - {"all"}))
                log.info("Condition met: invalid check area requested area=%s", area)
                await self._reply(
                    chat,
                    ctx,
                    (
                        f"{rb.section('Check')}<br>Unknown area: "
                        f"<code>{rb.esc(area)}</code>. Use one of: "
                        f"{rb.esc(options)}, all."
                    ),
                    parse_mode=ParseMode.HTML,
                )
                return

            if area == "all":
                areas = list(self._CHECK_ORDER)
                label = "ALL"
            elif area in self._CHECK_GROUPS:
                areas = list(self._CHECK_GROUPS[area])
                label = area.upper()
            else:
                areas = [area]
                label = area.upper()

            results = self._collect_check_results(areas)

            if len(results) == 1:
                name, (status, detail) = results[0]
                ok = status.upper() == "PASS"
                log.info("Condition met: single area check area=%s ok=%s", name, ok)
                lines = [
                    f"{rb.section('Check')} {rb.badge(ok, name.upper())}",
                ]
                if detail:
                    lines.append(f"<code>{rb.esc(detail)}</code>")
                await self._reply(
                    chat,
                    ctx,
                    "<br>".join(lines),
                    parse_mode=ParseMode.HTML,
                )
                return

            log.info(
                "Condition met: multi-area check area=%s targets=%s",
                area,
                areas,
            )
            text = self._render_check_summary(results, label)
            await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)
        except Exception as exc:  # pragma: no cover - defensive guard
            log.error("Failure in cmd_check: %s", exc, exc_info=exc)
            if chat is not None:
                await self._reply(
                    chat,
                    ctx,
                    "Check command failed. Inspect logs for details.",
                )
        return

    @command_meta(
        "/check_core",
        "Core guardrail summary for session, risk, orders, and strategies.",
    )
    async def cmd_check_core(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /check_core command for core subsystems.

        Args:
            update: Telegram update initiating the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug("Entered cmd_check_core")
        try:
            await self._handle_check_group(update, ctx, "core")
        except Exception as exc:  # pragma: no cover - defensive guard
            log.error("Failure in cmd_check_core: %s", exc, exc_info=exc)

    @command_meta(
        "/check_market",
        "Market data and connectivity summary for broker, WS, MDM, resolver.",
    )
    async def cmd_check_market(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /check_market command for market plumbing.

        Args:
            update: Telegram update initiating the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug("Entered cmd_check_market")
        try:
            await self._handle_check_group(update, ctx, "market")
        except Exception as exc:  # pragma: no cover - defensive guard
            log.error("Failure in cmd_check_market: %s", exc, exc_info=exc)

    @command_meta(
        "/check_execution",
        "Execution path summary for orders, positions, and risk modules.",
    )
    async def cmd_check_execution(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /check_execution command for execution checks.

        Args:
            update: Telegram update initiating the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug("Entered cmd_check_execution")
        try:
            await self._handle_check_group(update, ctx, "execution")
        except Exception as exc:  # pragma: no cover - defensive guard
            log.error("Failure in cmd_check_execution: %s", exc, exc_info=exc)

    @command_meta(
        "/check_connectivity",
        "Connectivity summary for broker, session, and websocket subsystems.",
    )
    async def cmd_check_connectivity(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /check_connectivity command for transport checks.

        Args:
            update: Telegram update initiating the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug("Entered cmd_check_connectivity")
        try:
            await self._handle_check_group(update, ctx, "connectivity")
        except Exception as exc:  # pragma: no cover - defensive guard
            log.error("Failure in cmd_check_connectivity: %s", exc, exc_info=exc)

    async def cmd_mdm(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        mdm_raw = self.deps.market_data_manager
        if mdm_raw is None:
            await self._reply(chat, ctx, "MDM not available")
            return
        st = t.cast(t.Any, mdm_raw).stats()
        msg = self._code(_pp(st))
        await self._reply(chat, ctx, msg, parse_mode=ParseMode.HTML)

    async def cmd_ws_status(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        ws = self.deps.websocket_manager
        if ws is None:
            await self._reply(
                chat, ctx, "Polling mode active (no websocket). Use /poll_status."
            )
            return
        state_name = "unknown"
        connecting = False
        connected = False
        hb_age = 0.0
        counts: dict[str, int] = {"subs": 0, "unsubs": 0, "modes": 0}
        last_error_text: str | None = None

        with suppress(Exception):
            state = ws.connection_state()
            if state is not None:
                state_name = getattr(state, "name", str(state))
                connecting = getattr(state, "name", "").upper() == "CONNECTING"
        with suppress(Exception):
            connected = bool(ws.is_connected())
        with suppress(Exception):
            hb = float(ws.last_heartbeat_monotonic())
            if hb > 0:
                hb_age = max(0.0, time.monotonic() - hb)
        with suppress(Exception):
            counts = ws.queue_depths()
        with suppress(Exception):
            err = ws.last_error()
            last_error_text = self._format_ws_error(err)

        hb_label = self._age_label(hb_age)
        queue_parts = []
        try:
            queue_parts.append(f"subs={int(counts.get('subs', 0))}")
            queue_parts.append(f"unsubs={int(counts.get('unsubs', 0))}")
            queue_parts.append(f"modes={int(counts.get('modes', 0))}")
        except Exception:
            queue_parts = []
        parts = [f"state={state_name}"]
        parts.append(f"connected={'yes' if connected else 'no'}")
        if connecting:
            parts.append("phase=connecting")
        parts.append(f"hbΔ={hb_label}")
        if queue_parts:
            parts.append("queue=" + ", ".join(queue_parts))
        if last_error_text:
            parts.append(f"last_error={last_error_text}")
        await self._reply(chat, ctx, "WS " + " ".join(parts))

    async def cmd_ws_reconnect(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        ws = self.deps.websocket_manager
        if ws is None:
            await self._reply(
                chat, ctx, "Polling mode active (no websocket). Use /poll_status."
            )
            return
        message = await self._reply(chat, ctx, "Refreshing WS session…")
        refresh = getattr(ws, "refresh_session", None)
        try:
            if callable(refresh):
                await asyncio.to_thread(refresh)
                await message.edit_text("WS session refreshed. Rebuilding…")
            else:
                await message.edit_text("No refresh hook; reconnecting…")
            await asyncio.to_thread(ws.force_reconnect)
            err_text = None
            with suppress(Exception):
                err_text = self._format_ws_error(ws.last_error())
            final = "Reconnect requested. Monitor logs."
            if err_text:
                final += f" last_error={err_text}"
            await message.edit_text(final)
        except Exception as exc:
            await message.edit_text(f"Reconnect failed: {exc}")

    async def cmd_ws_diag(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        ws = self.deps.websocket_manager
        if ws is None:
            await self._reply(
                chat, ctx, "Polling mode active (no websocket). Use /poll_status."
            )
            return

        token_supplier = getattr(self.deps, "get_ws_token", None)
        issued_supplier = getattr(self.deps, "get_ws_token_issued_at", None)
        token = token_supplier() if callable(token_supplier) else None
        issued_at = issued_supplier() if callable(issued_supplier) else None

        host = self.deps.ws_host or ""
        if not host:
            try:
                raw_url = getattr(getattr(ws, "_client", None), "_url", "")
                if raw_url:
                    host = urlsplit(str(raw_url)).hostname or ""
            except Exception:
                host = ""

        diag: WsDiag = await asyncio.to_thread(
            run_diag, ws, host or None, token, issued_at
        )

        dns = ", ".join(diag.ws_dns) if diag.ws_dns else "n/a"
        age = "unknown"
        if diag.token_age_s is not None:
            age = f"{diag.token_age_s:.0f}s"
        tcp = "ok" if diag.tcp_ok else "fail"
        if diag.tcp_peer:
            tcp += f" peer={diag.tcp_peer}"
        next_retry = "n/a"
        if diag.next_retry_s is not None:
            next_retry = f"{diag.next_retry_s:.1f}s"

        lines = [
            f"host={host or 'n/a'}",
            f"kiteconnect={diag.lib_kite}",
            f"websocket-client={diag.lib_wsclient}",
            f"python={diag.python}",
            f"ssl={diag.ssl_version}",
            f"token={diag.token_hint or 'n/a'}",
            f"token_age={age}",
            f"dns={dns}",
            f"tcp={tcp}",
            f"state={diag.state}",
            f"last_error={diag.last_error or 'none'}",
            f"breaker_open={'yes' if diag.breaker_open else 'no'}",
            f"next_retry={next_retry}",
        ]
        msg = self._code("\n".join(lines))
        await self._reply(chat, ctx, msg, parse_mode=ParseMode.HTML)

    async def cmd_poll_status(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        supervisor = self._stream_supervisor()
        streamer = self._polling_streamer()
        if supervisor is None and streamer is None:
            await self._reply(chat, ctx, "Websocket mode. Use /ws_status.")
            return
        if supervisor is not None:
            status_line = ""
            with suppress(Exception):
                status_line = str(supervisor.status_line())
            if not status_line:
                snapshot = self._polling_snapshot()
                status_line = str(snapshot.get("status") or "n/a")
            await self._reply(chat, ctx, f"Poll: {status_line}")
            return
        snapshot = self._polling_snapshot()
        status_line = str(snapshot.get("status") or "n/a")
        await self._reply(chat, ctx, f"Poll: {status_line}")

    async def cmd_subscribe(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        tokens, symbols, invalid = self._parse_symbol_token_args(ctx.args)
        if not tokens and not symbols:
            await self._reply(chat, ctx, "Usage: /subscribe SYMBOL|TOKEN [MORE]")
            return
        ws = self.deps.websocket_manager
        if ws is None:
            supervisor = self._stream_supervisor()
            streamer = self._polling_streamer()
            if supervisor is None and streamer is None:
                await self._reply(chat, ctx, "Polling streamer not available")
                return
            try:
                lines: list[str] = []
                unresolved: list[str] = []
                resolved_pairs: list[str] = []
                added = 0
                combined_tokens = list(tokens)
                if supervisor is not None and symbols:
                    resolver = getattr(supervisor, "resolve_symbols", None)
                    if callable(resolver):
                        resolved_tokens, unresolved, mapping = resolver(symbols)
                        combined_tokens.extend(resolved_tokens)
                        resolved_pairs = [
                            f"{symbol}->{mapping[symbol]}" for symbol in mapping
                        ]
                    else:
                        unresolved = list(symbols)
                elif symbols:
                    unresolved = list(symbols)

                if supervisor is not None:
                    token_count = getattr(supervisor, "token_count", None)
                    if callable(token_count):
                        before = int(token_count())
                    else:
                        before = len(
                            getattr(supervisor, "tracked_tokens", lambda: [])()
                        )
                    unique_tokens = sorted({int(token) for token in combined_tokens})
                    after = supervisor.subscribe_tokens(unique_tokens)
                    added = max(0, int(after) - before)
                    with suppress(Exception):
                        supervisor.ensure_started()
                    status_line = ""
                    with suppress(Exception):
                        status_line = str(supervisor.status_line())
                    if added > 0:
                        lines.append(f"Added {added} token(s).")
                    elif unique_tokens:
                        lines.append("No new tokens added (already tracking).")
                    if resolved_pairs:
                        lines.append(f"Resolved: {', '.join(resolved_pairs)}")
                    if unresolved:
                        lines.append(f"Unresolved: {', '.join(unresolved)}")
                    if invalid:
                        lines.append(f"Ignored: {', '.join(invalid)}")
                    if not status_line:
                        status_line = str(
                            self._polling_snapshot().get("status") or "n/a"
                        )
                    lines.append(f"Poll: {status_line}")
                    await self._reply(chat, ctx, "\n".join(lines))
                    return
                self._streamer_subscribe(streamer, tokens)
                count = self._streamer_token_count(streamer)
            except Exception as exc:  # noqa: BLE001
                await self._reply(chat, ctx, f"Subscribe failed: {exc}")
                return
            message = f"Subscribed {len(tokens)} token(s). Now tracking {count}."
            if invalid:
                message += f" (ignored: {', '.join(invalid)})"
            if symbols:
                message += f" (symbols ignored: {', '.join(symbols)})"
            await self._reply(chat, ctx, message)
            return
        try:
            await asyncio.to_thread(ws.subscribe, tokens)
        except Exception as exc:
            await self._reply(chat, ctx, f"Subscribe failed: {exc}")
            return
        message = f"Queued subscribe for tokens: {', '.join(map(str, tokens))}"
        if invalid:
            message += f" (ignored: {', '.join(invalid)})"
        if symbols:
            message += f" (symbols ignored: {', '.join(symbols)})"
        await self._reply(chat, ctx, message)

    async def cmd_unsubscribe(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        tokens, invalid = self._parse_token_args(ctx.args)
        if not tokens:
            await self._reply(chat, ctx, "Usage: /unsubscribe TOKEN[,TOKEN]")
            return
        ws = self.deps.websocket_manager
        if ws is None:
            supervisor = self._stream_supervisor()
            streamer = self._polling_streamer()
            if supervisor is None and streamer is None:
                await self._reply(chat, ctx, "Polling streamer not available")
                return
            try:
                if supervisor is not None:
                    count = supervisor.unsubscribe_tokens(tokens)
                    status_line = ""
                    with suppress(Exception):
                        status_line = str(supervisor.status_line())
                    message_lines = [
                        f"Unsubscribed {len(tokens)} token(s). Now tracking {count}.",
                    ]
                    if invalid:
                        message_lines.append(f"Ignored: {', '.join(invalid)}")
                    if not status_line:
                        status_line = str(
                            self._polling_snapshot().get("status") or "n/a"
                        )
                    message_lines.append(f"Poll: {status_line}")
                    await self._reply(chat, ctx, "\n".join(message_lines))
                    return
                else:
                    self._streamer_unsubscribe(streamer, tokens)
                    count = self._streamer_token_count(streamer)
            except Exception as exc:  # noqa: BLE001
                await self._reply(chat, ctx, f"Unsubscribe failed: {exc}")
                return
            message = f"Unsubscribed {len(tokens)} token(s). Now tracking {count}."
            if invalid:
                message += f" (ignored: {', '.join(invalid)})"
            await self._reply(chat, ctx, message)
            return
        try:
            await asyncio.to_thread(ws.unsubscribe, tokens)
        except Exception as exc:
            await self._reply(chat, ctx, f"Unsubscribe failed: {exc}")
            return
        message = f"Queued unsubscribe for tokens: {', '.join(map(str, tokens))}"
        if invalid:
            message += f" (ignored: {', '.join(invalid)})"
        await self._reply(chat, ctx, message)

    @command_meta("/sub N[,N]", "Subscribe tokens (alias of /subscribe).")
    async def cmd_sub(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:  # type: ignore[override]
        return await self.cmd_subscribe(update, ctx)

    @command_meta("/unsub N[,N]", "Unsubscribe tokens (alias of /unsubscribe).")
    async def cmd_unsub(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:  # type: ignore[override]
        return await self.cmd_unsubscribe(update, ctx)

    @command_meta(
        "/umlearn SYMBOL",
        "Fetch a fresh quote via unified manager and persist resolver hints.",
    )
    async def cmd_umlearn(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Trigger unified manager quote learning for ``symbol``.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Telegram callback context containing chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_umlearn",
            extra={"event": "telegram_cmd_umlearn_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_umlearn_guard_blocked",
                extra={"event": "telegram_cmd_umlearn_guard_blocked"},
            )
            return
        raw = " ".join(ctx.args or []).strip()
        if not raw:
            await self._reply(chat, ctx, "Usage: /umlearn SYMBOL")
            return
        symbol = raw.upper()
        try:
            manager = self._get_unified_manager()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_umlearn manager: %s",
                exc,
                extra={"event": "telegram_cmd_umlearn_manager_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unified manager unavailable.")
            return
        try:
            price = manager.get_price(symbol)
            log.info(
                "Condition met: telegram_cmd_umlearn_price",
                extra={
                    "event": "telegram_cmd_umlearn_price",
                    "symbol": symbol,
                    "source": getattr(price, "source", "n/a"),
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_umlearn get_price: %s",
                exc,
                extra={"event": "telegram_cmd_umlearn_price_error", "symbol": symbol},
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"Unable to fetch price for {symbol}.")
            return
        instrument: t.Any | None = None
        try:
            instrument = manager.resolve(symbol)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_umlearn resolve: %s",
                exc,
                extra={"event": "telegram_cmd_umlearn_resolve_error", "symbol": symbol},
                exc_info=exc,
            )
        resolver = getattr(self.deps, "resolver", None) or getattr(
            self.deps, "instrument_resolver", None
        )
        if (
            resolver is not None
            and instrument is not None
            and getattr(instrument, "token", None) is not None
        ):
            try:
                upsert_fn = getattr(resolver, "upsert", None)
                if callable(upsert_fn):
                    upsert_fn(
                        getattr(instrument, "symbol", symbol),
                        int(instrument.token),
                        exchange=getattr(instrument, "exchange", None),
                    )
                    log.info(
                        "Condition met: telegram_cmd_umlearn_upsert",
                        extra={
                            "event": "telegram_cmd_umlearn_upsert",
                            "symbol": symbol,
                            "token": int(instrument.token),
                        },
                    )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_umlearn resolver upsert: %s",
                    exc,
                    extra={
                        "event": "telegram_cmd_umlearn_upsert_error",
                        "symbol": symbol,
                    },
                    exc_info=exc,
                )
        payload = {
            "token": getattr(instrument, "token", None),
            "exchange": getattr(instrument, "exchange", None),
            "lot": getattr(instrument, "lot_size", None),
        }
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        await self._reply(
            chat,
            ctx,
            self._code(serialized),
            parse_mode=ParseMode.HTML,
        )
        log.info(
            "Condition met: telegram_cmd_umlearn_complete",
            extra={
                "event": "telegram_cmd_umlearn_complete",
                "symbol": symbol,
                "token": payload.get("token"),
            },
        )

    @command_meta("/resolve SYMBOL", "Resolve instrument metadata and token.")
    async def cmd_resolve(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Resolve instrument metadata for ``symbol`` via unified manager.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Telegram callback context containing chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_resolve",
            extra={"event": "telegram_cmd_resolve_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_resolve_guard_blocked",
                extra={"event": "telegram_cmd_resolve_guard_blocked"},
            )
            return
        raw = " ".join(ctx.args or []).strip()
        if not raw:
            await self._reply(chat, ctx, "Usage: /resolve SYMBOL")
            return
        symbol = raw.upper()
        try:
            manager = self._get_unified_manager()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_resolve manager: %s",
                exc,
                extra={"event": "telegram_cmd_resolve_manager_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unified manager unavailable.")
            return
        instrument: t.Any | None = None
        try:
            instrument = manager.resolve(symbol)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_resolve resolve: %s",
                exc,
                extra={"event": "telegram_cmd_resolve_um_error", "symbol": symbol},
                exc_info=exc,
            )
        resolver = getattr(self.deps, "resolver", None) or getattr(
            self.deps, "instrument_resolver", None
        )
        resolver_meta: dict[str, t.Any] | None = None
        if resolver is not None:
            try:
                resolver_meta = resolver.lookup(symbol)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_resolve lookup: %s",
                    exc,
                    extra={
                        "event": "telegram_cmd_resolve_lookup_error",
                        "symbol": symbol,
                    },
                    exc_info=exc,
                )
        resolved_symbol = symbol
        exchange = None
        token_value: int | None = None
        lot_size: int | None = None
        expiry: str | None = None
        if instrument is not None:
            resolved_symbol = getattr(instrument, "symbol", symbol) or symbol
            exchange = getattr(instrument, "exchange", exchange)
            token_value = getattr(instrument, "token", token_value)
            lot_size = getattr(instrument, "lot_size", lot_size)
            expiry = getattr(instrument, "expiry", expiry)
        if resolver_meta:
            exchange = resolver_meta.get("exchange") or exchange
            if token_value is None:
                token_hint = resolver_meta.get("instrument_token")
                with suppress(Exception):
                    token_value = (
                        int(token_hint) if token_hint is not None else token_value
                    )
            meta_lot = resolver_meta.get("lot_size")
            if lot_size is None and isinstance(meta_lot, int):
                lot_size = meta_lot
        resolved_symbol = str(resolved_symbol or symbol).upper()
        payload = {
            "symbol": resolved_symbol,
            "exchange": str(exchange or "").upper() or None,
            "token": token_value,
            "lot_size": lot_size,
            "expiry": expiry,
        }
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        await self._reply(
            chat,
            ctx,
            self._code(serialized),
            parse_mode=ParseMode.HTML,
        )
        log.info(
            "Condition met: telegram_cmd_resolve_dispatched",
            extra={
                "event": "telegram_cmd_resolve_dispatched",
                "symbol": resolved_symbol,
                "token": token_value,
            },
        )

    @command_meta(
        "/broker_quote SYMBOL",
        "Try broker quote keys and report the first successful match.",
    )
    async def cmd_broker_quote(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        raw = " ".join(ctx.args or []).strip()
        if not raw:
            await self._reply(chat, ctx, "Usage: /broker_quote SYMBOL")
            return
        symbol = raw.strip().upper()
        mdm_raw = self.deps.market_data_manager
        if mdm_raw is None:
            await self._reply(chat, ctx, "MDM not available")
            return
        mdm = t.cast(t.Any, mdm_raw)
        candidates: list[t.Any] = []
        try:
            candidate_fn = getattr(mdm, "_candidate_quote_keys", None)
            if callable(candidate_fn):
                candidates = list(candidate_fn(symbol))
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_broker_quote: %s",
                exc,
                extra={
                    "event": "cmd_broker_quote_candidates_failed",
                    "symbol": symbol,
                    "error": str(exc),
                },
            )
            candidates = []
        quote_fn = getattr(mdm, "_broker_quote_any", None)
        hit_key: t.Any | None = None
        payload: dict[str, t.Any] | None = None
        fields: list[str] = []
        if callable(quote_fn):
            for candidate in candidates:
                try:
                    quote = quote_fn(candidate)
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure in cmd_broker_quote: %s",
                        exc,
                        extra={
                            "event": "cmd_broker_quote_fetch_failed",
                            "symbol": symbol,
                            "candidate": candidate,
                            "error": str(exc),
                        },
                    )
                    continue
                if isinstance(quote, dict) and quote:
                    hit_key = candidate
                    payload = dict(quote)
                    for key in ("bid", "ask", "ltp", "last_price", "price"):
                        value = quote.get(key)
                        if value not in (None, "") and key not in fields:
                            fields.append(key)
                    break
        else:
            log.info(
                "cmd_broker_quote_no_api",
                extra={"symbol": symbol},
            )
        lines = [f"symbol={symbol}", f"candidates={candidates}"]
        if hit_key is not None:
            lines.append(f"hit={hit_key}")
            if fields:
                lines.append(f"fields={fields}")
            if payload is not None:
                preview = {k: payload.get(k) for k in fields}
                lines.append(f"preview={preview}")
        else:
            lines.append("hit=None")
        await self._reply(
            chat,
            ctx,
            self._code("\n".join(lines)),
            parse_mode=ParseMode.HTML,
        )

    async def cmd_tick(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        if not ctx.args:
            await self._reply(chat, ctx, "Usage: /tick SYMBOL|TOKEN")
            return
        raw_arg = ctx.args[0].strip()
        token: int | None = None
        try:
            token = int(raw_arg)
        except ValueError:
            token = None

        mdm_raw = self.deps.market_data_manager
        if mdm_raw is None:
            await self._reply(chat, ctx, "MDM not available")
            return

        mdm = t.cast(t.Any, mdm_raw)
        symbol = raw_arg.upper()
        resolved_symbol: str | None = symbol
        if token is not None:
            resolved_symbol = None
            with suppress(Exception):
                mapping = getattr(mdm, "_symbol_by_token", {})
                if isinstance(mapping, dict):
                    resolved_symbol = mapping.get(token)
            if resolved_symbol is None:
                await self._reply(
                    chat,
                    ctx,
                    (
                        f"No symbol mapping for token {token}. Wait for ticks or "
                        "ensure subscription."
                    ),
                )
                return

        lookup_key = resolved_symbol if resolved_symbol is not None else symbol
        tick = mdm.get_latest_tick(lookup_key)
        if not tick:
            fallback_symbol = resolved_symbol or symbol
            rest_tick: t.Any | None = None
            with suppress(Exception):
                rest_tick = mdm.pull_quote(fallback_symbol)
            if rest_tick:
                await self._reply(
                    chat,
                    ctx,
                    self._code(_pp(rest_tick)),
                    parse_mode=ParseMode.HTML,
                )
                return
            broker = self.deps.broker_client
            get_ltp = getattr(broker, "get_ltp", None)
            if callable(get_ltp):
                try:
                    ltp_map = get_ltp([fallback_symbol])
                except Exception:
                    ltp_map = {}
                if isinstance(ltp_map, dict):
                    ltp_value = ltp_map.get(fallback_symbol)
                    if ltp_value is None and ltp_map:
                        try:
                            ltp_value = next(iter(ltp_map.values()))
                        except StopIteration:
                            ltp_value = None
                    if isinstance(ltp_value, (int, float)):
                        await self._reply(
                            chat,
                            ctx,
                            f"{fallback_symbol}: LTP {ltp_value} (REST)",
                        )
                        return
            await self._reply(chat, ctx, "No tick in cache")
            return
        await self._reply(chat, ctx, self._code(_pp(tick)), parse_mode=ParseMode.HTML)

    @command_meta(
        "/iv SYMBOL",
        "Return implied volatility or explain why data is unavailable.",
    )
    async def cmd_iv(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Fetch implied volatility for the requested symbol with guidance.

        Args:
            update: Telegram update triggering the command.
            ctx: Handler context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_iv",
            extra={"event": "telegram_cmd_iv_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        try:
            args = ctx.args or []
            if not args:
                await self._reply(chat, ctx, "Usage: /iv SYMBOL")
                log.info(
                    "Condition met: telegram_cmd_iv_missing_symbol",
                    extra={"event": "telegram_cmd_iv_missing_symbol"},
                )
                return
            raw_symbol = args[0].strip()
            symbol = DataHub.normalize(raw_symbol) or raw_symbol.upper()
            hub = getattr(self.deps, "data_hub", None)
            if hub is None or not hasattr(hub, "get_iv"):
                message = (
                    "📊 IV data not available\n\n"
                    "Reasons:\n"
                    "• IV feed disabled in current mode\n"
                    "• Data provider does not expose IV\n"
                    "• Market closed for derivatives\n\n"
                    "💡 Retry during market hours on a valid option symbol."
                )
                await self._reply(chat, ctx, message)
                log.info(
                    "Condition met: telegram_cmd_iv_unavailable",
                    extra={"event": "telegram_cmd_iv_unavailable"},
                )
                return
            try:
                iv_value = hub.get_iv(symbol)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_iv lookup: %s",
                    exc,
                    extra={"event": "telegram_cmd_iv_error", "symbol": symbol},
                    exc_info=exc,
                )
                await self._reply(
                    chat,
                    ctx,
                    f"IV lookup failed: {self._short_reason(str(exc))}",
                )
                return
            if iv_value is None:
                message = (
                    f"📊 IV data not available for {symbol}\n\n"
                    "Possible causes:\n"
                    "• Symbol not in options cache\n"
                    "• Incomplete quote data for the contract\n"
                    "• Market data feed still warming up"
                )
                await self._reply(chat, ctx, message)
                log.info(
                    "Condition met: telegram_cmd_iv_missing_data",
                    extra={"event": "telegram_cmd_iv_missing_data", "symbol": symbol},
                )
                return
            try:
                iv_pct = float(iv_value) * 100.0
            except (TypeError, ValueError):
                iv_pct = None
            if iv_pct is None:
                response = f"📊 IV for {symbol}: {iv_value}"
            else:
                response = f"📊 IV for {symbol}: {iv_pct:.2f}%"
            await self._reply(chat, ctx, response)
            log.info(
                "Condition met: telegram_cmd_iv_success",
                extra={"event": "telegram_cmd_iv_success", "symbol": symbol},
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_iv outer: %s",
                exc,
                extra={"event": "telegram_cmd_iv_outer_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unexpected IV lookup error.")

    @command_meta(
        "/greeks SYMBOL",
        "Return option Greeks with context-aware fallbacks.",
    )
    async def cmd_greeks(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Calculate and display option Greeks for a given NIFTY option symbol.

        Usage: /greeks NFO:NIFTY25NOV25950CE
        """
        from telegram.constants import ParseMode
        from nifty_scalper_bot.utils.logging import get_logger
        log = get_logger(__name__)

        # --- START FIX: Define HTML formatting helpers inline to bypass broken ResponseBuilder ---
        def h_strong(text): return f"<b>{text}</b>"
        def h_section(text): return f"<u>{text}</u>"
        def h_dim(text): return f"<i>{text}</i>"
        def h_br(): return "\n" # Use newline for join, ParseMode.HTML handles formatting
        # --- END FIX ---

        log.debug("Entered cmd_greeks", extra={"event": "telegram_cmd_greeks_enter"})

        chat = await self._guard(update)
        if chat is None:
            return

        try:
            args = list(ctx.args or [])
            if not args:
                await self._reply(
                    chat, ctx,
                    "Usage: /greeks SYMBOL\nExample: /greeks NFO:NIFTY25NOV25950CE"
                )
                return

            symbol = str(args[0]).strip().upper()

            # Parse the option symbol
            parsed = self._parse_nifty_option_symbol(symbol)
            if not parsed:
                await self._reply(chat, ctx, f"❌ Could not parse symbol: {symbol}")
                return

            # Get NIFTY spot price
            spot = self._get_nifty_spot_price()
            if not spot:
                await self._reply(
                    chat, ctx,
                    "❌ NIFTY spot price unavailable\n\n"
                    "**Fix:** Run `/watch NIFTY` then retry in a few seconds"
                )
                return

            # Get option LTP
            ltp = None
            mdm = getattr(self.deps, "market_data_manager", None)
            if mdm:
                tick = mdm.get_latest_tick(symbol)
                if tick and isinstance(tick, dict):
                    ltp = tick.get("ltp") or tick.get("last_price")

            # Calculate Greeks
            greeks = self._calculate_greeks_simple(
                spot=spot,
                strike=parsed["strike"],
                days_to_expiry=parsed["days_to_expiry"],
                option_type=parsed["option_type"]
            )

            # Build response using safe, inline formatting helpers
            lines = [
                f"{h_section('Greeks')} {symbol}", # Using h_section (underlined/bold)
                "",
                f"{h_strong('Underlying:')} NIFTY @ ₹{spot:,.2f}",
                f"{h_strong('Strike:')} ₹{parsed['strike']:,}",
                f"{h_strong('Type:')} {parsed['option_type']} ({parsed['symbol_type']})",
                f"{h_strong('Expiry:')} {parsed['expiry'].strftime('%d %b %Y')} ({greeks['days_to_expiry']:.0f} days)",
            ]

            if ltp:
                lines.append(f"{h_strong('LTP:')} ₹{ltp:.2f}")
            else:
                lines.append(f"{h_strong('LTP:')} n/a")

            lines.append(f"{h_strong('Moneyness:')} {greeks['moneyness']:.3f}")
            lines.append("")
            lines.append(h_strong("Greeks:"))
            lines.append(f"• {h_strong('Delta:')} {greeks['delta']:+.4f}")
            lines.append(f"• {h_strong('Gamma:')} {greeks['gamma']:.6f}")
            lines.append(f"• {h_strong('Theta:')} ₹{greeks['theta']:.2f}/day")
            lines.append(f"• {h_strong('Vega:')} ₹{greeks['vega']:.2f}")
            lines.append("")
            lines.append(h_dim("Calculated using 20% IV assumption (simple approximation)")) # Using h_dim (italics)

            # Join lines using the newline helper
            message = h_br().join(lines)
            await self._reply(chat, ctx, message, parse_mode=ParseMode.HTML)

            log.info(
                "Condition met: telegram_cmd_greeks_success",
                extra={"event": "telegram_cmd_greeks_success", "symbol": symbol},
            )

        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_greeks outer: %s",
                exc,
                extra={"event": "telegram_cmd_greeks_outer_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"❌ Greeks calculation error: {exc}")
    async def cmd_quote(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return

        raw = " ".join(ctx.args or []).strip()
        if not raw:
            await self._reply(chat, ctx, "Usage: /quote SYMBOL")
            return
        if (raw.startswith('"') and raw.endswith('"')) or (
            raw.startswith("'") and raw.endswith("'")
        ):
            raw = raw[1:-1]
        symbol = raw.strip()
        normalized = DataHub.normalize(symbol)

        hub = self.deps.data_hub
        mdm = self.deps.market_data_manager

        quote_data: dict[str, t.Any] | None = None
        if hub is not None:
            with suppress(Exception):
                cached = hub.get_quote(normalized, allow_pull=True)
                if isinstance(cached, t.Mapping):
                    quote_data = dict(cached)

        if quote_data is None and mdm is not None and hasattr(mdm, "pull_quote"):
            try:
                pulled = mdm.pull_quote(normalized)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                await self._reply(chat, ctx, f"Quote failed: {exc}")
                return
            if isinstance(pulled, t.Mapping):
                quote_data = dict(pulled)
                if hub is not None:
                    with suppress(Exception):
                        hub.ingest_tick(quote_data)

        if quote_data is None:
            await self._reply(chat, ctx, "No quote available.")
            return

        def _as_float(val: t.Any) -> float | None:
            try:
                if val is None:
                    return None
                return float(val)
            except (TypeError, ValueError):
                return None

        ltp_value = _as_float(
            quote_data.get("ltp")
            or quote_data.get("last_price")
            or quote_data.get("close")
        )
        bid_value = _as_float(quote_data.get("bid"))
        ask_value = _as_float(quote_data.get("ask"))
        ts_raw = (
            quote_data.get("timestamp")
            or quote_data.get("ts")
            or quote_data.get("ts_ms")
        )
        ts_value = _as_float(ts_raw) or time.time()
        if ts_value > 1_000_000_000_000:
            ts_value /= 1000.0
        ts_iso = datetime.fromtimestamp(ts_value).isoformat()

        def _fmt(value: float | None) -> str:
            if value is None:
                return "-"
            return f"{value:.2f}"

        summary = (
            f"{normalized}: ltp={_fmt(ltp_value)} | bid={_fmt(bid_value)} | "
            f"ask={_fmt(ask_value)} | ts={ts_iso}"
        )
        await self._reply(
            chat,
            ctx,
            f"{summary}\n{self._code(_pp(dict(quote_data)))}",
            parse_mode=ParseMode.HTML,
        )

    async def cmd_strategies(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Report strategy enablement and tracking metadata.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_strategies",
            extra={"event": "telegram_cmd_strategies_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_strategies_guard_blocked",
                extra={"event": "telegram_cmd_strategies_guard_blocked"},
            )
            return
        try:
            manager = getattr(self.deps, "strategy_manager", None)
            if manager is None:
                await self._reply(chat, ctx, "No strategies loaded")
                log.info(
                    "Condition met: telegram_cmd_strategies_manager_missing",
                    extra={"event": "telegram_cmd_strategies_manager_missing"},
                )
                return
            strategies: list[t.Any] = []
            with suppress(Exception):
                raw_strategies = getattr(manager, "strategies", [])
                strategies = list(raw_strategies) if raw_strategies is not None else []
            rb = self._response_builder
            lines: list[str] = [
                f"{rb.section('Strategies')} total=<b>{len(strategies)}</b>",
            ]
            for strategy in strategies:
                try:
                    name = getattr(strategy, "name", None) or type(strategy).__name__
                    enabled = bool(getattr(strategy, "enabled", True))
                    status = "🟢 ENABLED" if enabled else "🔴 DISABLED"
                    confidence_value = getattr(strategy, "confidence", None)
                    if callable(confidence_value):
                        confidence_value = confidence_value()
                    if isinstance(confidence_value, (int, float)):
                        confidence_text = f"{float(confidence_value):.2f}"
                    else:
                        confidence_text = "n/a"
                    lines.append(
                        (
                            f"{status} {rb.esc(str(name))} — "
                            f"conf={rb.esc(confidence_text)}"
                        )
                    )
                except Exception as strat_exc:  # noqa: BLE001
                    log.error(
                        "Failure in cmd_strategies detail build: %s",
                        strat_exc,
                        extra={"event": "telegram_cmd_strategies_detail_error"},
                        exc_info=strat_exc,
                    )
            tracked: list[str] = []
            with suppress(Exception):
                tracked = list(getattr(manager, "tracked_symbols", lambda: [])())
            if tracked:
                preview = ", ".join(sorted(tracked)[:20])
                suffix = "" if len(tracked) <= 20 else " …"
                lines.append(
                    (f"Tracked ({len(tracked)}): " f"{rb.esc(preview)}{suffix}")
                )
            message = self._compose([lines]) if lines else "No strategies registered."
            await self._reply(chat, ctx, message, parse_mode=ParseMode.HTML)
            log.info(
                "Condition met: telegram_cmd_strategies_dispatched",
                extra={
                    "event": "telegram_cmd_strategies_dispatched",
                    "strategy_count": len(strategies),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "Failure in cmd_strategies: %s",
                exc,
                extra={"event": "telegram_cmd_strategies_error"},
                exc_info=exc,
            )
            with suppress(Exception):
                await self._reply(
                    chat,
                    ctx,
                    "Unable to fetch strategies. Inspect logs for details.",
                )

    @command_meta(
        "/signals",
        "Show the most recent strategy signals from strategy manager/state.",
    )
    async def cmd_signals(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Return recent strategy signals. Args: update, ctx; Returns: None; Raises: None."""
        log.debug("Entered cmd_signals", extra={"event": "telegram_cmd_signals_enter"})
        chat = await self._guard(update)
        if chat is None:
            return
        try:
            manager = getattr(self.deps, "strategy_manager", None)
            signals: list[t.Any] = []
            for attr in ("recent_signals", "signal_history", "last_signals"):
                candidate = getattr(manager, attr, None) if manager is not None else None
                if callable(candidate):
                    with suppress(Exception):
                        candidate = candidate()
                if isinstance(candidate, list):
                    signals = candidate
                    break
            if not signals:
                await self._reply(chat, ctx, "No recent signals available.")
                return
            lines = ["Recent signals:"]
            for item in signals[-10:]:
                lines.append(f"- {item}")
            await self._reply(chat, ctx, "\n".join(lines))
        except Exception as e:  # noqa: BLE001
            log.error("Failure in cmd_signals: %s", e)
            await self._reply(chat, ctx, "Unable to fetch recent signals.")

    @command_meta("/test_trade", "Run a lightweight execution-path test order.")
    async def cmd_test_trade(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Place a single synthetic test trade. Args: update, ctx; Returns: None; Raises: None."""
        log.debug("Entered cmd_test_trade", extra={"event": "telegram_cmd_test_trade_enter"})
        chat = await self._guard(update)
        if chat is None:
            return
        try:
            symbol = "NFO:NIFTY"
            log.info("TEST TRADE TRIGGERED", extra={"event": "ENTRY_SIGNAL", "symbol": symbol})
            order_manager = getattr(self.deps, "order_manager", None)
            if order_manager is None:
                await self._reply(chat, ctx, "Order manager unavailable.")
                return
            order_manager.place_order(
                symbol=symbol,
                side="BUY",
                quantity=1,
                order_type="MARKET",
                strategy_name="telegram_test",
                tag="telegram_test_trade",
            )
            await self._reply(chat, ctx, "Test trade executed in shadow mode.")
        except Exception as e:  # noqa: BLE001
            log.error("Failure in cmd_test_trade: %s", e)
            await self._reply(chat, ctx, "Test trade failed.")

    @command_meta("/engine_test", "Run run_engine_tests.py and return summary output.")
    async def cmd_engine_test(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Run execution-engine tests command. Args: update, ctx; Returns: None; Raises: None."""
        log.debug("Entered cmd_engine_test", extra={"event": "telegram_cmd_engine_test_enter"})
        chat = await self._guard(update)
        if chat is None:
            return
        try:
            result = subprocess.run(
                ["python", "run_engine_tests.py", "--mode", "stress"],
                capture_output=True,
                text=True,
                check=False,
            )
            output = (result.stdout or result.stderr or "No output")[:4000]
            await self._reply(chat, ctx, output)
        except Exception as e:  # noqa: BLE001
            log.error("Failure in cmd_engine_test: %s", e)
            await self._reply(chat, ctx, "Engine test failed to execute.")

    @command_meta("/panic", "🚨 EMERGENCY: Cancel all orders and flatten positions.")
    async def cmd_panic(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """🚨 EMERGENCY: Cancel all orders and close all positions immediately."""
        if not await self._guard_admin(update): return

        await self._reply(update.effective_chat, context, "🚨 <b>PANIC SEQUENCE INITIATED...</b>")
        
        results = []
        try:
            # 1. Kill the Runner (Stop new trades)
            self._deps.strategy_runner.pause_trading()
            results.append("⛔ Strategy Paused")

            # 2. Cancel Orders (The Stage 3 logic)
            # Accessing the unified manager through dependencies
            um = self._deps.strategy_runner._strategy_manager.orchestrator._order_manager
            cancelled = um.cancel_all_open_orders() # Ensure this method exists in OrderManager
            results.append(f"🗑️ Cancelled {len(cancelled)} Orders")

            # 3. Flatten Positions (Market Exit)
            positions = um.position_manager.get_all_positions()
            for pos in positions:
                if pos.quantity != 0:
                    um.place_order(
                        symbol=pos.symbol,
                        side="SELL" if pos.quantity > 0 else "BUY",
                        quantity=abs(pos.quantity),
                        order_type="MARKET",
                        tag="PANIC_EXIT"
                    )
            results.append(f"📉 Closing {len(positions)} Positions")

        except Exception as e:
            results.append(f"❌ ERROR: {str(e)}")

        await self._reply(update.effective_chat, context, "\n".join(results))

    @command_meta(
        "/strategy_scores [limit]",
        "Show weighted strategy scores and allocations.",
    )
  
    async def cmd_strategy_scores(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Report strategy performance scores and allocations.

        Args:
            update: Telegram update payload for the command.
            ctx: Context provided by python-telegram-bot for handlers.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug("Entered TelegramBot.cmd_strategy_scores")
        chat = await self._guard(update)
        if chat is None:
            return
        manager = getattr(self.deps, "strategy_manager", None)
        if manager is None:
            await self._reply(chat, ctx, "Strategy manager unavailable")
            return
        limit_arg: int | None = None
        if ctx.args:
            try:
                limit_arg = max(1, min(int(ctx.args[0]), 20))
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_strategy_scores limit parse: %s",
                    exc,
                    exc_info=exc,
                )
                await self._reply(chat, ctx, "Invalid limit provided")
                return
        try:
            scores = manager.get_strategy_scores(limit_arg)
            allocations = manager.get_allocation_snapshot()
            regime_state = manager.refresh_regime_state()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_strategy_scores fetch: %s",
                exc,
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"Failed to fetch strategy scores: {exc}")
            return
        payload: list[dict[str, t.Any]] = []
        for entry in scores:
            try:
                allocation_fraction = float(allocations.get(entry.strategy, 0.0))
                regime_snapshot: dict[str, dict[str, float]] = {}
                try:
                    regime_snapshot = {
                        regime: {
                            "pnl": round(float(stats.get("pnl", 0.0)), 2),
                            "win_rate": round(float(stats.get("win_rate", 0.0)), 4),
                            "sharpe": round(float(stats.get("sharpe", 0.0)), 4),
                            "drawdown": round(float(stats.get("drawdown", 0.0)), 2),
                            "trades": round(float(stats.get("trades", 0.0)), 2),
                        }
                        for regime, stats in entry.regime_breakdown.items()
                    }
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure in cmd_strategy_scores regime snapshot build: %s",
                        exc,
                        exc_info=exc,
                    )
                    regime_snapshot = {}
                active_regime: dict[str, t.Any] = {}
                try:
                    active_regime = {
                        key: (
                            round(float(value), 4)
                            if isinstance(value, (int, float))
                            else value
                        )
                        for key, value in entry.active_regime_stats.items()
                    }
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure in cmd_strategy_scores active regime build: %s",
                        exc,
                        exc_info=exc,
                    )
                    active_regime = {}
                payload.append(
                    {
                        "strategy": entry.strategy,
                        "weight": round(float(entry.weight), 4),
                        "allocation": round(float(entry.allocation), 4),
                        "fraction": round(allocation_fraction, 4),
                        "score": round(float(entry.score), 4),
                        "regime_score": round(float(entry.regime_score), 4),
                        "pnl": round(float(entry.pnl), 2),
                        "rolling_pnl": round(float(entry.rolling_pnl), 2),
                        "sharpe": round(float(entry.sharpe), 4),
                        "win_rate": round(float(entry.win_rate), 4),
                        "win_loss": round(float(entry.win_loss_ratio), 4),
                        "drawdown": round(float(entry.drawdown), 2),
                        "manual": (
                            None
                            if entry.manual_allocation is None
                            else round(float(entry.manual_allocation), 4)
                        ),
                        "regime_bias": round(float(entry.regime_bias), 3),
                        "enabled": bool(entry.enabled),
                        "dynamic_disabled": bool(entry.dynamic_disabled),
                        "active_regime": active_regime,
                        "regimes": regime_snapshot,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_strategy_scores payload build: %s",
                    exc,
                    exc_info=exc,
                )
                continue
        log.info(
            "Condition met: telegram_strategy_scores",
            extra={
                "event": "telegram_strategy_scores",
                "entries": len(payload),
                "regime": getattr(regime_state, "regime", None),
                "confidence": getattr(regime_state, "confidence", None),
            },
        )
        message = {
            "scores": payload,
            "regime": getattr(regime_state, "regime", None),
            "confidence": getattr(regime_state, "confidence", None),
            "disabled": list(getattr(manager, "disabled_strategies", lambda: [])()),
        }
        await self._reply(
            chat,
            ctx,
            self._code(_pp(message)),
            parse_mode=ParseMode.HTML,
        )

    @command_meta("/elite_stats", "Show elite strategy diagnostics.")
    async def cmd_elite_stats(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Report telemetry for elite strategies.

        Args:
            update: Telegram update payload initiating the command.
            ctx: Context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug("Entered TelegramBot.cmd_elite_stats")
        chat = await self._guard(update)
        if chat is None:
            return
        manager = getattr(self.deps, "strategy_manager", None)
        if manager is None:
            await self._reply(chat, ctx, "Strategy manager unavailable.")
            return
        try:
            stats = manager.get_elite_strategy_stats()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_elite_stats fetch: %s",
                exc,
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Failed to collect elite stats.")
            return
        if not stats:
            await self._reply(chat, ctx, "No elite strategy stats available yet.")
            return
        await self._reply(
            chat,
            ctx,
            self._code(_pp(stats)),
            parse_mode=ParseMode.HTML,
        )

    @command_meta("/score [limit]", "Compact strategy score summary.")
    async def cmd_score(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Summarize strategy scores with current health signals.

        Args:
            update: Telegram update payload for the command.
            ctx: PTB context associated with the incoming update.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered TelegramBot.cmd_score",
            extra={"event": "telegram_cmd_score"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        manager = getattr(self.deps, "strategy_manager", None)
        if manager is None:
            await self._reply(chat, ctx, "Strategy manager unavailable.")
            return
        risk_manager = getattr(self.deps, "risk_manager", None)
        limit = 5
        if ctx.args:
            try:
                limit = max(1, min(int(ctx.args[0]), 20))
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_score limit parsing: %s",
                    exc,
                    exc_info=exc,
                )
                await self._reply(chat, ctx, "Invalid limit supplied.")
                return
        try:
            scores = manager.get_strategy_scores(limit)
            regime_state = manager.refresh_regime_state()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_score fetch: %s",
                exc,
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unable to fetch strategy scores.")
            return
        if not scores:
            await self._reply(chat, ctx, "No strategy scores available yet.")
            return
        balance_value, balance_source = self._risk_balance_snapshot(
            force=True, risk_manager=risk_manager
        )
        balance_text = self._format_currency(balance_value)
        balance_line = f"Balance: {balance_text} ({balance_source})"
        lines: list[str] = []
        current_allocations: dict[str, float] = {}
        for entry in scores:
            try:
                status = "✅"
                if not entry.enabled:
                    status = "⏸️" if entry.dynamic_disabled else "🚫"
                manual_flag = (
                    f" manual={float(entry.manual_allocation):.2f}"
                    if entry.manual_allocation is not None
                    else ""
                )
                regime_stats = entry.active_regime_stats or {}
                try:
                    regime_hit = float(regime_stats.get("hit_rate", 0.0))
                    regime_pnl = float(regime_stats.get("pnl", 0.0))
                except Exception as stat_exc:  # noqa: BLE001
                    log.error(
                        "Failure in cmd_score regime stat parse: %s",
                        stat_exc,
                        exc_info=stat_exc,
                    )
                    regime_hit = 0.0
                    regime_pnl = 0.0
                lines.append(
                    (
                        f"{status} {entry.strategy}: score={entry.score:.2f}/"
                        f"{entry.regime_score:.2f}, weight={entry.weight:.2f}, "
                        f"alloc={entry.allocation:.2f}, "
                        f"rollPnL={entry.rolling_pnl:.2f}, "
                        f"win/loss={entry.win_loss_ratio:.2f}, "
                        f"regimePnL={regime_pnl:.2f}, regimeHit={regime_hit:.2f}"
                        f"{manual_flag}"
                    )
                )
                current_allocations[entry.strategy] = float(entry.allocation)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_score line build: %s",
                    exc,
                    exc_info=exc,
                )
        allocation_delta = self._compute_allocation_delta(current_allocations)
        self._store_allocation_history(current_allocations, allocation_delta)
        header = (
            f"Regime: {getattr(regime_state, 'regime', 'unknown')} "
            f"({getattr(regime_state, 'confidence', 0.0):.2f})"
        )
        message_lines = [header, balance_line, *lines]
        if allocation_delta:
            delta_line = ", ".join(
                f"{name}:{change:+.2f}" for name, change in allocation_delta.items()
            )
            message_lines.append(f"Δalloc: {delta_line}")
        notes: list[str] = []
        if risk_manager is None:
            notes.append("⚠️ Risk manager unavailable; broker balance not refreshed.")
        else:
            if balance_source in {"env", "unknown"}:
                notes.append("⚠️ Broker balance unavailable; using fallback capital.")
            elif balance_source == "cache":
                notes.append("ℹ️ Using cached broker balance; confirm live broker feed.")
        if notes:
            message_lines.extend(notes)
        highlights = self._diag_highlights(plain=True)
        if highlights:
            message_lines.append("")
            message_lines.append("Diagnostics:")
            message_lines.extend(highlights)
        message = "\n".join(message_lines)
        try:
            await self._reply(chat, ctx, message)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_score reply: %s",
                exc,
                exc_info=exc,
            )
        else:
            log.info(
                "Condition met: telegram_score_sent",
                extra={
                    "event": "telegram_score_sent",
                    "entries": len(lines),
                    "regime": getattr(regime_state, "regime", None),
                    "allocation_delta": allocation_delta,
                },
            )

    @command_meta("/alloc", "Display active allocation weights per strategy.")
    async def cmd_allocations(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Report live allocation snapshot with status metadata.

        Args:
            update: Telegram update payload for the command.
            ctx: PTB context associated with the update.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered TelegramBot.cmd_allocations",
            extra={"event": "telegram_cmd_alloc"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        manager = getattr(self.deps, "strategy_manager", None)
        if manager is None:
            await self._reply(chat, ctx, "Strategy manager unavailable.")
            return
        snapshot_getter = getattr(manager, "get_allocation_snapshot", None)
        scores_getter = getattr(manager, "get_strategy_scores", None)
        if not callable(snapshot_getter) or not callable(scores_getter):
            await self._reply(
                chat,
                ctx,
                "Strategy allocations not available in current mode.",
            )
            log.info(
                "Condition met: telegram_cmd_allocations_unsupported",
                extra={"event": "telegram_cmd_allocations_unsupported"},
            )
            return
        try:
            snapshot = snapshot_getter()
            scores = scores_getter()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_allocations fetch: %s",
                exc,
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unable to fetch allocations.")
            return
        if not snapshot:
            await self._reply(chat, ctx, "No active allocations currently.")
            return
        score_lookup = {entry.strategy: entry for entry in scores}
        lines: list[str] = []
        for strategy, fraction in sorted(
            snapshot.items(), key=lambda item: item[1], reverse=True
        ):
            try:
                entry = score_lookup.get(strategy)
                status = "✅"
                weight = fraction
                score_value = 0.0
                regime_score = 0.0
                regime_hit = 0.0
                manual_flag = ""
                if entry is not None:
                    weight = entry.weight
                    score_value = entry.score
                    regime_score = float(entry.regime_score)
                    try:
                        regime_hit = float(
                            (entry.active_regime_stats or {}).get("hit_rate", 0.0)
                        )
                    except Exception as stat_exc:  # noqa: BLE001
                        log.error(
                            "Failure in cmd_allocations regime parse: %s",
                            stat_exc,
                            exc_info=stat_exc,
                        )
                        regime_hit = 0.0
                    if entry.enabled:
                        status = "✅"
                    elif entry.dynamic_disabled:
                        status = "⏸️"
                    else:
                        status = "🚫"
                    if entry.manual_allocation is not None:
                        manual_flag = f" manual={entry.manual_allocation:.2f}"
                lines.append(
                    (
                        f"{status} {strategy}: active={fraction:.2%}, "
                        f"weight={weight:.2f}, score={score_value:.2f}/"
                        f"{regime_score:.2f}, regimeHit={regime_hit:.2f}{manual_flag}"
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_allocations line build: %s",
                    exc,
                    exc_info=exc,
                )
        disabled_manual: tuple[str, ...] = ()
        try:
            disabled_manual = getattr(manager, "disabled_strategies", lambda: tuple())()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_allocations disabled fetch: %s",
                exc,
                exc_info=exc,
            )
        dynamic_suspended = [
            entry.strategy for entry in scores if entry.dynamic_disabled
        ]
        footer_parts: list[str] = []
        if disabled_manual:
            footer_parts.append("manual off: " + ", ".join(sorted(disabled_manual)))
        if dynamic_suspended:
            footer_parts.append(
                "dynamic off: " + ", ".join(sorted(set(dynamic_suspended)))
            )
        footer = "\n" + " | ".join(footer_parts) if footer_parts else ""
        message = "\n".join(lines) + footer
        try:
            await self._reply(chat, ctx, message)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_allocations reply: %s",
                exc,
                exc_info=exc,
            )
        else:
            log.info(
                "Condition met: telegram_allocations_sent",
                extra={
                    "event": "telegram_allocations_sent",
                    "entries": len(lines),
                    "manual_off": len(disabled_manual),
                    "dynamic_off": len(dynamic_suspended),
                },
            )

    def _resolve_strategy_for_command(
        self,
        manager: t.Any,
        scores: list[t.Any],
        raw_name: str,
    ) -> tuple[str | None, t.Any | None]:
        """Resolve a strategy identifier against scores and registry.

        Args:
            manager: Strategy manager used for fallback introspection.
            scores: Score snapshots fetched from the manager.
            raw_name: Raw strategy identifier supplied by the user.

        Returns:
            tuple[str | None, t.Any | None]: Resolved strategy name and score
            entry when available.

        Raises:
            None.
        """

        log.debug(
            "Entered TelegramBot._resolve_strategy_for_command",
            extra={
                "event": "telegram_strategy_resolve",
                "raw": raw_name,
            },
        )
        try:
            candidate = (raw_name or "").strip()
            if not candidate:
                return None, None
            lookup = {str(entry.strategy).lower(): entry for entry in scores}
            lowered = candidate.lower()
            if lowered in lookup:
                entry = lookup[lowered]
                return entry.strategy, entry
            strategies: Iterable[t.Any] = getattr(manager, "strategies", tuple())
            for strategy in strategies:
                name = str(getattr(strategy, "name", type(strategy).__name__))
                if name.lower() == lowered:
                    return name, lookup.get(name.lower())
            return None, None
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in TelegramBot._resolve_strategy_for_command: %s",
                exc,
                exc_info=exc,
            )
            return None, None

    @command_meta(
        "/strategy_allocate <name> <fraction|off>",
        "Override strategy allocation fraction or clear override.",
    )
    async def cmd_strategy_allocate(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Adjust manual capital allocation for a strategy.

        Args:
            update: Telegram update payload for the command.
            ctx: Context provided by python-telegram-bot for handlers.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug("Entered TelegramBot.cmd_strategy_allocate")
        chat = await self._guard(update)
        if chat is None:
            return
        manager = getattr(self.deps, "strategy_manager", None)
        if manager is None:
            await self._reply(chat, ctx, "Strategy manager unavailable")
            return
        args = ctx.args or []
        if len(args) < 2:
            await self._reply(
                chat,
                ctx,
                "Usage: /strategy_allocate <name> <fraction|off>",
            )
            return
        fraction_token = args[-1]
        name_tokens = args[:-1]
        raw_name = " ".join(name_tokens).replace("_", " ").strip()
        if not raw_name:
            await self._reply(chat, ctx, "Provide a strategy name")
            return
        strategies = getattr(manager, "strategies", [])
        lookup: dict[str, str] = {}
        for strategy in strategies:
            try:
                proper_name = str(getattr(strategy, "name", type(strategy).__name__))
            except Exception:  # noqa: BLE001 - defensive
                proper_name = type(strategy).__name__
            normalised = proper_name.lower().replace(" ", "")
            lookup[normalised] = proper_name
            lookup[type(strategy).__name__.lower()] = proper_name
        key = raw_name.lower().replace(" ", "")
        selected = lookup.get(key)
        if selected is None:
            await self._reply(chat, ctx, f"Unknown strategy: {raw_name}")
            return
        manual_fraction: float | None
        if fraction_token.lower() in {"off", "none", "clear"}:
            manual_fraction = None
        else:
            try:
                manual_fraction = float(fraction_token)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_strategy_allocate fraction parse: %s",
                    exc,
                    exc_info=exc,
                )
                await self._reply(chat, ctx, "Invalid fraction provided")
                return
        try:
            manager.set_manual_allocation(selected, manual_fraction)
            snapshot = manager.get_allocation_snapshot()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_strategy_allocate set: %s",
                exc,
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"Allocation update failed: {exc}")
            return
        log.info(
            "Condition met: telegram_strategy_allocation",
            extra={
                "event": "telegram_strategy_allocation",
                "strategy": selected,
                "fraction": manual_fraction,
            },
        )
        response = {
            "strategy": selected,
            "manual_fraction": manual_fraction,
            "allocations": snapshot,
        }
        await self._reply(
            chat,
            ctx,
            self._code(_pp(response)),
            parse_mode=ParseMode.HTML,
        )

    @command_meta(
        "/strategy_disable <name>",
        "Disable a strategy after reviewing the latest scores.",
    )
    async def cmd_strategy_disable(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Disable strategy execution at runtime.

        Args:
            update: Telegram update payload for the command.
            ctx: Context provided by python-telegram-bot for handlers.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug("Entered TelegramBot.cmd_strategy_disable")
        chat = await self._guard(update)
        if chat is None:
            return
        manager = getattr(self.deps, "strategy_manager", None)
        if manager is None:
            await self._reply(chat, ctx, "Strategy manager unavailable")
            return
        if not ctx.args:
            await self._reply(chat, ctx, "Usage: /strategy_disable <name>")
            return
        raw_name = " ".join(ctx.args)
        try:
            scores = manager.get_strategy_scores()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_strategy_disable fetch: %s",
                exc,
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"Failed to fetch scores: {exc}")
            return
        resolved, entry = self._resolve_strategy_for_command(manager, scores, raw_name)
        if not resolved:
            await self._reply(chat, ctx, f"Unknown strategy: {raw_name}")
            return
        try:
            disabled = manager.disable_strategy(resolved)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_strategy_disable toggle: %s",
                exc,
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"Disable failed: {exc}")
            return
        if not disabled:
            state = False
            try:
                state = not manager.is_strategy_enabled(resolved)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_strategy_disable state check: %s",
                    exc,
                    exc_info=exc,
                )
            message = (
                f"Strategy {resolved} already disabled"
                if state
                else f"Strategy {resolved} could not be disabled"
            )
            await self._reply(chat, ctx, message)
            return
        try:
            post_scores = manager.get_strategy_scores()
            _, post_entry = self._resolve_strategy_for_command(
                manager, post_scores, resolved
            )
            allocations = manager.get_allocation_snapshot()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_strategy_disable post snapshot: %s",
                exc,
                exc_info=exc,
            )
            post_entry = entry
            allocations = {}
        disabled_roster: Iterable[str] = getattr(
            manager, "disabled_strategies", lambda: tuple()
        )()
        payload = {
            "strategy": resolved,
            "disabled": True,
            "score": None if post_entry is None else round(float(post_entry.score), 4),
            "weight": (
                None if post_entry is None else round(float(post_entry.weight), 4)
            ),
            "allocation": round(float(allocations.get(resolved, 0.0)), 4),
            "disabled_strategies": list(disabled_roster),
        }
        log.info(
            "Condition met: telegram_strategy_disable",
            extra={
                "event": "telegram_strategy_disable",
                "strategy": resolved,
            },
        )
        await self._reply(
            chat,
            ctx,
            self._code(_pp(payload)),
            parse_mode=ParseMode.HTML,
        )

    @command_meta(
        "/strategy_enable <name>",
        "Enable a previously disabled strategy when scores recover.",
    )
    async def cmd_strategy_enable(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Enable strategy execution at runtime.

        Args:
            update: Telegram update payload for the command.
            ctx: Context provided by python-telegram-bot for handlers.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug("Entered TelegramBot.cmd_strategy_enable")
        chat = await self._guard(update)
        if chat is None:
            return
        manager = getattr(self.deps, "strategy_manager", None)
        if manager is None:
            await self._reply(chat, ctx, "Strategy manager unavailable")
            return
        if not ctx.args:
            await self._reply(chat, ctx, "Usage: /strategy_enable <name>")
            return
        raw_name = " ".join(ctx.args)
        try:
            scores = manager.get_strategy_scores()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_strategy_enable fetch: %s",
                exc,
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"Failed to fetch scores: {exc}")
            return
        resolved, entry = self._resolve_strategy_for_command(manager, scores, raw_name)
        if not resolved:
            await self._reply(chat, ctx, f"Unknown strategy: {raw_name}")
            return
        try:
            enabled = manager.enable_strategy(resolved)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_strategy_enable toggle: %s",
                exc,
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"Enable failed: {exc}")
            return
        if not enabled:
            state = True
            try:
                state = manager.is_strategy_enabled(resolved)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_strategy_enable state check: %s",
                    exc,
                    exc_info=exc,
                )
            message = (
                f"Strategy {resolved} already enabled"
                if state
                else f"Strategy {resolved} could not be enabled"
            )
            await self._reply(chat, ctx, message)
            return
        try:
            post_scores = manager.get_strategy_scores()
            _, post_entry = self._resolve_strategy_for_command(
                manager, post_scores, resolved
            )
            allocations = manager.get_allocation_snapshot()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_strategy_enable post snapshot: %s",
                exc,
                exc_info=exc,
            )
            post_entry = entry
            allocations = {}
        disabled_roster: Iterable[str] = getattr(
            manager, "disabled_strategies", lambda: tuple()
        )()
        payload = {
            "strategy": resolved,
            "disabled": False,
            "score": None if post_entry is None else round(float(post_entry.score), 4),
            "weight": (
                None if post_entry is None else round(float(post_entry.weight), 4)
            ),
            "allocation": round(float(allocations.get(resolved, 0.0)), 4),
            "disabled_strategies": list(disabled_roster),
        }
        log.info(
            "Condition met: telegram_strategy_enable",
            extra={
                "event": "telegram_strategy_enable",
                "strategy": resolved,
            },
        )
        await self._reply(
            chat,
            ctx,
            self._code(_pp(payload)),
            parse_mode=ParseMode.HTML,
        )

    @command_meta(
        "/regimediag",
        "Detailed regime diagnostics including filter history.",
    )
    async def cmd_regimediag(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Reply with structured diagnostics from the regime manager.

        Args:
            update: Telegram update payload for the command.
            ctx: Context provided by python-telegram-bot for handlers.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered TelegramBot.cmd_regimediag",
            extra={"event": "telegram_cmd_regimediag"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        manager = getattr(self.deps, "regime_manager", None)
        if manager is None:
            await self._reply(chat, ctx, "Regime manager unavailable.")
            return
        try:
            diagnostics = manager.build_diagnostics()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_regimediag diagnostics: %s",
                exc,
                extra={"event": "telegram_cmd_regimediag_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unable to build regime diagnostics.")
            return
        await self._reply(
            chat,
            ctx,
            self._code(_pp(diagnostics)),
            parse_mode=ParseMode.HTML,
        )

    @command_meta(
        "/regime_bypass [on|off|toggle]",
        "Inspect or toggle the regime filter bypass flag.",
    )
    async def cmd_regime_bypass(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Toggle or inspect the regime filter bypass state.

        Args:
            update: Telegram update payload for the command.
            ctx: Context provided by python-telegram-bot for handlers.

        Returns:
            None.

        Raises:
            None.
        """

        await self._execute_regime_bypass(
            update=update, ctx=ctx, command_label="regime_bypass"
        )

    @command_meta(
        "/regimebypass [on|off|toggle]",
        "Alias for /regime_bypass using the central regime manager.",
    )
    async def cmd_regimebypass(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Toggle or inspect the regime filter bypass via alias command.

        Args:
            update: Telegram update payload for the command.
            ctx: Context provided by python-telegram-bot for handlers.

        Returns:
            None.

        Raises:
            None.
        """

        await self._execute_regime_bypass(
            update=update, ctx=ctx, command_label="regimebypass"
        )

    async def _execute_regime_bypass(
        self,
        *,
        update: Update,
        ctx: ContextTypes.DEFAULT_TYPE,
        command_label: str,
    ) -> None:
        """Execute inspection or toggling of the regime bypass flag.

        Args:
            update: Telegram update payload for the command.
            ctx: Context provided by python-telegram-bot for handlers.
            command_label: Identifier used for logging context.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered TelegramBot._execute_regime_bypass",
            extra={
                "event": "telegram_regime_bypass_execute",
                "command": command_label,
            },
        )
        chat = await self._guard(update)
        if chat is None:
            return
        manager = getattr(self.deps, "regime_manager", None)
        if manager is None:
            await self._reply(chat, ctx, "Regime manager unavailable.")
            return
        try:
            current_state = bool(manager.get_regime_filter_bypass())
            stats = manager.get_regime_filter_stats()
            reasons = manager.get_filter_reasons()
            history = manager.get_decision_history(limit=1)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in _execute_regime_bypass inspect: %s",
                exc,
                extra={"event": "telegram_regime_bypass_inspect_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Failed to inspect regime filter state.")
            return
        decision = history[-1] if history else None
        desired_state = current_state
        args = list(getattr(ctx, "args", []) or [])
        if args:
            token = str(args[0]).strip().lower()
            if token in {"on", "enable", "true", "1"}:
                desired_state = True
            elif token in {"off", "disable", "false", "0"}:
                desired_state = False
            elif token == "toggle":
                desired_state = not current_state
            elif token not in {"status", "state"}:
                await self._reply(
                    chat,
                    ctx,
                    "Usage: /regime_bypass [on|off|toggle]",
                )
                return
        if desired_state != current_state:
            try:
                updated_state = manager.set_regime_filter_bypass(desired_state)
                current_state = bool(updated_state)
                stats = manager.get_regime_filter_stats()
                reasons = manager.get_filter_reasons()
                history = manager.get_decision_history(limit=1)
                decision = history[-1] if history else None
                log.info(
                    "Condition met: telegram_regime_bypass",
                    extra={
                        "event": "telegram_regime_bypass",
                        "enabled": current_state,
                        "command": command_label,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in _execute_regime_bypass toggle: %s",
                    exc,
                    extra={"event": "telegram_regime_bypass_toggle_error"},
                    exc_info=exc,
                )
                await self._reply(chat, ctx, f"Toggle failed: {exc}")
                return
        decision_payload: dict[str, object] | None = None
        if decision is not None:
            try:
                decision_payload = {
                    "allowed": decision.allowed,
                    "reasons": list(decision.reasons),
                    "bypassed": decision.bypassed,
                    "confidence": decision.confidence,
                    "decided_at": datetime.fromtimestamp(
                        float(decision.decided_at), timezone.utc
                    ).isoformat(),
                }
            except Exception:
                decision_payload = {
                    "allowed": decision.allowed,
                    "reasons": list(decision.reasons),
                    "bypassed": decision.bypassed,
                    "confidence": decision.confidence,
                    "decided_at": decision.decided_at,
                }
        payload = {
            "regime_filter_bypass": current_state,
            "filter_stats": stats,
            "last_reasons": list(reasons),
        }
        if decision_payload is not None:
            payload["last_decision"] = decision_payload
        await self._reply(
            chat,
            ctx,
            self._code(_pp(payload)),
            parse_mode=ParseMode.HTML,
        )

    async def cmd_positions(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        pm = self.deps.position_manager
        if pm is None:
            await self._reply(chat, ctx, "No position manager")
            return
        positions: list[t.Any] = getattr(pm, "get_open_positions", lambda: [])()
        await self._reply(
            chat,
            ctx,
            self._code(_pp({"open_positions": positions})),
            parse_mode=ParseMode.HTML,
        )

    async def cmd_orders(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        om = self.deps.order_manager
        if om is None:
            await self._reply(chat, ctx, "No order manager")
            return
        limit = 15
        with suppress(Exception):
            if ctx.args:
                limit = max(1, min(int(ctx.args[0]), 50))
        recent_orders: list[t.Any] = getattr(om, "recent_orders", lambda n=10: [])(
            limit
        )
        await self._reply(
            chat,
            ctx,
            self._code(_pp({"recent_orders": recent_orders})),
            parse_mode=ParseMode.HTML,
        )

    def _build_regime_report(
        self,
        *,
        manager: MarketRegimeManager | None,
        detector: MarketRegimeDetector | None,
        limit: int = 5,
    ) -> tuple[list[str], RegimeSnapshot | None]:
        """Construct formatted regime report lines for Telegram.

        Args:
            manager: Central regime manager instance when available.
            detector: Legacy regime detector used as a fallback.
            limit: Maximum number of snapshots to display.

        Returns:
            tuple[list[str], RegimeSnapshot | None]:
                Ordered report lines and the most recent snapshot.

        Raises:
            ValueError: If no regime data is available for display.
        """

        log.debug(
            "Entered TelegramBot._build_regime_report",
            extra={"event": "telegram_regime_build_report", "limit": limit},
        )
        snapshots: list[RegimeSnapshot] = []
        try:
            if manager is not None:
                snapshots = manager.get_history(limit=limit)
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error(
                "Failure in _build_regime_report manager history: %s",
                exc,
                extra={"event": "telegram_regime_manager_history_error"},
                exc_info=exc,
            )
        if not snapshots and detector is not None:
            try:
                snapshots = detector.latest(limit=limit)
            except Exception as exc:  # noqa: BLE001 - defensive
                log.error(
                    "Failure in _build_regime_report detector history: %s",
                    exc,
                    extra={"event": "telegram_regime_detector_history_error"},
                    exc_info=exc,
                )
                raise
        if not snapshots:
            raise ValueError("no_regime_data")
        display_map = {
            "trend": "trending",
            "range": "ranging/choppy",
            "volatile": "volatile",
            "event": "event-driven",
        }
        lines: list[str] = []
        for snapshot in snapshots:
            updated = datetime.fromtimestamp(snapshot.updated_at, timezone.utc)
            adjustments: t.Mapping[str, t.Any] = {}
            if isinstance(snapshot.adjustments, t.Mapping):
                adjustments = snapshot.adjustments
            bias = adjustments.get("bias")
            sizing = adjustments.get("sizing_multiplier")
            blocklist = adjustments.get("blocklist")
            display_regime = adjustments.get("display_regime")
            regime_label = (
                str(display_regime)
                if isinstance(display_regime, str)
                else display_map.get(snapshot.regime, snapshot.regime)
            )
            detail_parts: list[str] = []
            if isinstance(bias, str) and bias:
                detail_parts.append(f"bias={bias}")
            if isinstance(sizing, (int, float)):
                detail_parts.append(f"size×{float(sizing):.2f}")
            if isinstance(blocklist, (list, tuple)) and blocklist:
                detail_parts.append(
                    "block=" + ",".join(str(name) for name in blocklist)
                )
            detail = ", ".join(detail_parts) if detail_parts else "no adjustments"
            entry_text = (
                f"{snapshot.symbol}: {regime_label} "
                f"({snapshot.confidence:.2f})\n"
                f"  reason={snapshot.reason} | updated={updated:%H:%M:%S}Z\n"
                f"  {detail}"
            )
            previous_regime, _, changed = self._update_regime_history(
                snapshot.symbol,
                snapshot.regime,
                snapshot.confidence,
                snapshot.updated_at,
            )
            if changed and previous_regime:
                previous_label = display_map.get(previous_regime, previous_regime)
                entry_text += (
                    "\n  transition: "
                    f"{previous_label}→{regime_label} @ {updated:%H:%M:%S}Z "
                    f"conf={snapshot.confidence:.2f}"
                )
            lines.append(entry_text)
        if manager is not None:
            stats = manager.get_regime_filter_stats()
            bypass = manager.get_regime_filter_bypass()
            reasons = manager.get_filter_reasons()
            decision = None
            try:
                history = manager.get_decision_history(limit=1)
            except Exception as exc:  # noqa: BLE001 - defensive
                log.error(
                    "Failure in _build_regime_report decision history: %s",
                    exc,
                    extra={"event": "telegram_regime_decision_history_error"},
                    exc_info=exc,
                )
                history = []
            if history:
                decision = history[-1]
            lines.append("")
            lines.append("Regime filter:")
            lines.append(f"  bypass={'ON' if bypass else 'OFF'}")
            if decision is not None:
                status = "allow" if decision.allowed else "block"
                confidence = (
                    decision.confidence if decision.confidence is not None else 0.0
                )
                lines.append(f"  last_decision={status} ({confidence:.2f})")
            if reasons:
                lines.append(f"  reasons={', '.join(reasons)}")
            if stats:
                stats_line = ", ".join(
                    f"{key}={int(value)}" for key, value in sorted(stats.items())
                )
                lines.append(f"  stats={stats_line}")
        return lines, snapshots[-1]

    async def cmd_trades(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        om = self.deps.order_manager
        if om is None:
            await self._reply(chat, ctx, "No order manager")
            return
        history: list[t.Any] = getattr(om, "get_order_history", lambda **_: [])(
            limit=50
        )
        fills = [
            order
            for order in history
            if str(getattr(order, "status", "")).upper().endswith("FILLED")
        ]
        await self._reply(
            chat,
            ctx,
            self._code(_pp({"fills": fills[-20:]})),
            parse_mode=ParseMode.HTML,
        )

    async def cmd_cancel(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        if not ctx.args:
            await self._reply(chat, ctx, "Usage: /cancel <ORDER_ID>")
            return
        order_id = ctx.args[0].strip()
        om = self.deps.order_manager
        if om is None:
            await self._reply(chat, ctx, "No order manager")
            return
        try:
            cancelled = getattr(om, "cancel_order", lambda _oid: False)(order_id)
        except Exception as exc:  # pragma: no cover - defensive
            await self._reply(chat, ctx, f"Cancel failed: {exc}")
            return
        await self._reply(
            chat, ctx, "Cancelled ✅" if cancelled else "Not cancelled (already final?)"
        )

    async def _manual_order(
        self,
        side: str,
        args: t.Sequence[str] | None,
        chat: t.Any,
        ctx: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        om = self.deps.order_manager
        if om is None:
            await self._reply(chat, ctx, "No order manager")
            return
        symbol, quantity, price = self._parse_order_args(args)
        if not symbol:
            await self._reply(
                chat,
                ctx,
                "Usage: /buy SYMBOL [QTY] [PRICE]  or  /sell SYMBOL [QTY] [PRICE]",
            )
            return
        guard = self._session_guard_status()
        if guard and not all(
            bool(guard.get(key))
            for key in ("session_valid", "rate_limits_ok", "market_open", "risk_green")
        ):
            await self._reply(
                chat,
                ctx,
                "⚠ Live guard is blocking; order may be rejected by risk/session.",
            )
        order_type_cls: t.Any | None = None
        with suppress(Exception):
            module = __import__(
                "nifty_scalper_bot.execution.order_manager", fromlist=["OrderType"]
            )
            order_type_cls = getattr(module, "OrderType")
        try:
            kwargs: dict[str, t.Any] = {
                "symbol": symbol,
                "side": side.upper(),
                "quantity": quantity,
            }
            if price is not None:
                kwargs["price"] = price
                if order_type_cls is not None:
                    kwargs["order_type"] = order_type_cls.LIMIT
            elif order_type_cls is not None:
                kwargs["order_type"] = order_type_cls.MARKET
            loop = asyncio.get_running_loop()
            order_id = await loop.run_in_executor(
              None, 
              lambda: getattr(om, "place_order")(**kwargs)
            )
        except Exception as exc:  # pragma: no cover - defensive
            await self._reply(chat, ctx, f"Order failed: {exc}")
            return
        price_label = f"@{price}" if price is not None else "(MKT)"
        await self._reply(
            chat,
            ctx,
            (
                f"Submitted {side.upper()} {symbol} x{quantity} {price_label}\n"
                f"order_id={order_id}"
            ),
        )

    async def cmd_buy(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        await self._manual_order("BUY", ctx.args, chat, ctx)

    async def cmd_sell(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        await self._manual_order("SELL", ctx.args, chat, ctx)

    @command_meta(
        "/go_flat",
        "Emergency cancel + flatten of all positions (admin only).",
    )
    async def cmd_go_flat(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /go_flat command for emergency liquidation.

        Args:
            update: Telegram update initiating the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_go_flat",
            extra={"event": "cmd_go_flat_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        if not await self._guard_admin(update):
            return
        chat_id = int(getattr(chat, "id", 0))
        pending = self._pending_confirmation
        if pending and self._pending_confirmation_expired(pending):
            self._clear_pending_confirmation()
            pending = None
        if pending and pending.get("action") == "emergency_stop":
            await self._reply(
                chat,
                ctx,
                (
                    "Emergency flatten already pending. Use /confirm emergency_stop "
                    "or /confirm cancel."
                ),
            )
            log.info(
                "Condition met: cmd_go_flat_pending",
                extra={"event": "cmd_go_flat_pending"},
            )
            return
        self._set_pending_confirmation("emergency_stop", chat_id)
        await self._reply(
            chat,
            ctx,
            (
                "🚨 Confirm emergency flatten with /confirm emergency_stop within "
                f"{int(self._CONFIRMATION_WINDOW.total_seconds())}s."
            ),
        )
        log.info(
            "Condition met: cmd_go_flat_requested",
            extra={"event": "cmd_go_flat_requested"},
        )

    @command_meta("/risk", "Risk guard, drawdown, and margin telemetry.")
    async def cmd_risk(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Reply with live risk, drawdown, and margin utilisation metrics.

        Args:
            update: Telegram update envelope triggering the command.
            ctx: Callback context carrying parsed command arguments.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_risk",
            extra={"event": "telegram_cmd_risk_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_risk_guard_blocked",
                extra={"event": "telegram_cmd_risk_guard_blocked"},
            )
            return
        rm = getattr(self.deps, "risk_manager", None)
        if rm is None:
            await self._reply(chat, ctx, "No risk manager")
            log.info(
                "Condition met: telegram_cmd_risk_no_rm",
                extra={"event": "telegram_cmd_risk_no_rm"},
            )
            return
        pm = getattr(self.deps, "position_manager", None)
        rb = self._response_builder
        payload: dict[str, t.Any] = {}
        try:
            balance_value = self._coerce_float_value(
                getattr(rm, "account_balance", None), field="risk_account_balance"
            )
            exposure_value = (
                self._coerce_float_value(
                    getattr(pm, "get_total_exposure", None),
                    field="position_total_exposure",
                )
                if pm is not None
                else None
            )
            margin_used = exposure_value
            margin_available = None
            if balance_value is not None and exposure_value is not None:
                margin_available = max(balance_value - exposure_value, 0.0)
            risk_snapshot = self._coerce_risk_snapshot()
            risk_state = getattr(rm, "risk_state", None) or getattr(
                rm, "_risk_state", None
            )
            peak_equity = self._coerce_float_value(
                getattr(risk_state, "_peak_equity", None), field="risk_peak_equity"
            )
            current_equity = self._coerce_float_value(
                getattr(risk_state, "_current_equity", None),
                field="risk_current_equity",
            )
            drawdown_value: float | None = None
            drawdown_pct: float | None = None
            if peak_equity and current_equity is not None:
                drawdown_value = current_equity - peak_equity
                if peak_equity != 0:
                    drawdown_pct = drawdown_value / peak_equity
            drawdown_limit_pct = self._coerce_float_value(
                getattr(risk_state, "max_intraday_dd", None),
                field="risk_max_intraday_dd",
            )
            allowed, reasons = rm.risk_gate_should_trade()
            badge = rb.badge(allowed, "risk gate")
            reason_text = ",".join(reasons) if reasons else "ok"
            used_text = rb.esc(self._format_currency(margin_used, signed=True))
            avail_text = rb.esc(self._format_currency(margin_available, signed=True))
            summary_lines = [
                (
                    f"{rb.section('Margin')} used=<b>{used_text}</b> "
                    f"avail=<b>{avail_text}</b>"
                ),
            ]
            if drawdown_value is not None:
                dd_currency = self._format_currency(drawdown_value, signed=True)
                dd_percent = self._format_percent(drawdown_pct, signed=True)
                limit_percent = self._format_percent(
                    (drawdown_limit_pct / 100.0) if drawdown_limit_pct else None
                )
                summary_lines.append(
                    f"{rb.section('Drawdown')} current=<b>{rb.esc(dd_currency)}</b> "
                    f"({rb.esc(dd_percent)}) limit={rb.esc(limit_percent)}"
                )
            summary_lines.append(f"{badge} reasons={rb.esc(reason_text)}")
            if risk_snapshot is not None:
                cooldown_remaining = risk_snapshot.get("cooldown_remaining")
                if (
                    isinstance(cooldown_remaining, (int, float))
                    and cooldown_remaining > 0
                ):
                    cooldown_text = rb.esc(f"{cooldown_remaining:.1f}s")
                    summary_lines.append(
                        f"{rb.section('Cooldown')} remaining=<b>{cooldown_text}</b>"
                    )
                losses_in_row = risk_snapshot.get("losses_in_row")
                if isinstance(losses_in_row, int) and losses_in_row > 0:
                    summary_lines.append(
                        f"{rb.section('Loss Streak')} count=<b>{losses_in_row}</b>"
                    )
            message = rb.bullets(summary_lines)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_risk",
                exc,
                extra={"event": "telegram_cmd_risk_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unable to fetch risk telemetry.")
            return
        payload["margin"] = {
            "used": margin_used,
            "available_est": margin_available,
            "exposure": exposure_value,
            "balance": balance_value,
        }
        payload["drawdown"] = {
            "value": drawdown_value,
            "fraction": drawdown_pct,
            "limit_pct": drawdown_limit_pct,
        }
        if risk_snapshot is not None:
            payload["risk_snapshot"] = risk_snapshot
        with suppress(Exception):
            payload["risk_config"] = _sanitize_dict(
                _asdict_safe(getattr(rm, "risk_config", None))
            )
        with suppress(Exception):
            payload["current_balance"] = getattr(rm, "current_balance", None)
        guard_status = self._session_guard_status()
        if guard_status is not None:
            payload["session_guard"] = guard_status
        detail_block = self._code(_pp(payload))
        try:
            await self._reply(
                chat,
                ctx,
                f"{message}{rb.br()}{detail_block}",
                parse_mode=ParseMode.HTML,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_risk reply: %s",
                exc,
                extra={"event": "telegram_cmd_risk_reply_error"},
                exc_info=exc,
            )
        else:
            log.info(
                "Condition met: telegram_cmd_risk_dispatched",
                extra={
                    "event": "telegram_cmd_risk_dispatched",
                    "allowed": allowed,
                    "reasons": reason_text,
                    "margin_used": margin_used,
                },
            )

    @command_meta("/report", "Latest nightly replay automation summary.")
    async def cmd_report(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Return summary from the nightly replay automation.

        Args:
            update: Telegram update invoking the report command.
            ctx: Handler context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_report",
            extra={"event": "telegram_cmd_report_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        report_path = Path("data/nightly/reports/nightly_report.json")
        if not report_path.exists():
            log.info(
                "Condition met: nightly_report_missing",
                extra={"event": "telegram_cmd_report_missing"},
            )
            await self._reply(chat, ctx, "Nightly report not available.")
            return
        try:
            payload = json.loads(report_path.read_text())
        except Exception as exc:  # noqa: BLE001 - defensive file parsing
            log.error(
                "Failure in cmd_report load: %s",
                exc,
                extra={"event": "telegram_cmd_report_load_failed"},
            )
            await self._reply(chat, ctx, "Failed to load nightly report.")
            return
        generated = str(payload.get("generated_at", "unknown"))
        inputs = int(payload.get("inputs_processed", 0) or 0)
        pnl_total = float(payload.get("pnl_total", 0.0) or 0.0)
        mismatches = int(payload.get("mismatches", 0) or 0)
        latency = payload.get("latency_p95")
        errors = payload.get("errors") or []
        rb = self._response_builder
        lines = [
            f"{rb.section('Nightly Replay')} generated=<b>{rb.esc(generated)}</b>",
            (
                f"inputs=<b>{inputs}</b> "
                f"pnl=<b>{pnl_total:.2f}</b> "
                f"mismatches=<b>{mismatches}</b>"
            ),
        ]
        if isinstance(latency, (int, float)):
            lines.append(f"latency_p95=<b>{float(latency):.3f}s</b>")
        if errors:
            error_text = ", ".join(rb.esc(str(err)) for err in errors[-5:])
            lines.append(f"errors={error_text}")
        log.info(
            "Condition met: telegram_report_dispatched",
            extra={"event": "telegram_report_dispatched", "inputs": inputs},
        )
        await self._reply(chat, ctx, "<br>".join(lines), parse_mode=ParseMode.HTML)

    @command_meta(
        "/diag_price <SYMBOL>",
        "Detailed price source, freshness, and gate diagnostics.",
    )
    async def cmd_diag_price(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /diag_price command for price diagnostics.

        Args:
            update: Telegram update initiating the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_diag_price",
            extra={"event": "cmd_diag_price_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        args = ctx.args or []
        symbol = args[0].strip() if args else self._primary_symbol()
        if not symbol:
            symbol = "NIFTY"
        try:
            payload = self._build_price_diag(symbol)
        except RuntimeError as exc:
            reason = self._short_reason(str(exc))
            await self._reply(chat, ctx, f"diag_price unavailable: {reason}")
            return
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_diag_price: %s",
                exc,
                extra={"event": "cmd_diag_price_error", "symbol": symbol},
            )
            await self._reply(chat, ctx, "diag_price failed. Inspect logs.")
            return
        log.info(
            "Condition met: responding diag_price",
            extra={"event": "cmd_diag_price_reply", "symbol": payload.get("symbol")},
        )
        summary = json.dumps(
            payload, ensure_ascii=False, default=str, separators=(",", ":")
        )
        await self._reply(
            chat,
            ctx,
            f"diag_price {self._code(summary)}",
            parse_mode=ParseMode.HTML,
        )

    @command_meta(
        "/diag_risk [SYMBOL]",
        "Risk gate, breaker, and cooldown diagnostics.",
    )
    async def cmd_diag_risk(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /diag_risk command for risk diagnostics.

        Args:
            update: Telegram update initiating the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_diag_risk",
            extra={"event": "cmd_diag_risk_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        args = ctx.args or []
        symbol = args[0].strip() if args else self._primary_symbol()
        try:
            payload = self._build_risk_diag(symbol)
        except RuntimeError as exc:
            reason = self._short_reason(str(exc))
            await self._reply(chat, ctx, f"diag_risk unavailable: {reason}")
            return
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_diag_risk: %s",
                exc,
                extra={"event": "cmd_diag_risk_error"},
            )
            await self._reply(chat, ctx, "diag_risk failed. Inspect logs.")
            return
        log.info(
            "Condition met: responding diag_risk",
            extra={"event": "cmd_diag_risk_reply", "symbol": payload.get("symbol")},
        )
        summary = json.dumps(
            payload, ensure_ascii=False, default=str, separators=(",", ":")
        )
        await self._reply(
            chat,
            ctx,
            f"diag_risk {self._code(summary)}",
            parse_mode=ParseMode.HTML,
        )

    @command_meta("/fresh [SYMBOL]", "Show cached quote freshness metrics.")
    async def cmd_fresh(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        data_hub = getattr(self.deps, "data_hub", None)
        if data_hub is None:
            await self._reply(chat, ctx, "Data hub unavailable.")
            return
        args = ctx.args or []
        symbol = args[0].strip() if args else self._primary_symbol()
        if not symbol:
            symbol = "NIFTY"
        try:
            fresh, meta = data_hub.is_fresh(symbol)
        except Exception as exc:  # noqa: BLE001
            reason = self._short_reason(str(exc))
            await self._reply(chat, ctx, f"fresh error: {reason}")
            return
        payload = {
            "symbol": symbol,
            "fresh": bool(fresh),
            "effective_ms": meta.get("effective_ms"),
            "mono_age_ms": meta.get("mono_age_ms"),
            "server_age_ms": meta.get("server_age_ms"),
            "threshold_ms": meta.get("threshold_ms"),
            "reason": meta.get("reason"),
        }
        summary = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        await self._reply(
            chat,
            ctx,
            f"fresh {self._code(summary)}",
            parse_mode=ParseMode.HTML,
        )

    @command_meta("/watch SYMBOL", "Track symbol via REST polling and warm cache.")
    async def cmd_watch(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Track a symbol through MarketDataManager and seed the cache.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_watch",
            extra={"event": "telegram_cmd_watch_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_watch_guard_blocked",
                extra={"event": "telegram_cmd_watch_guard_blocked"},
            )
            return
        if not ctx.args:
            await self._reply(chat, ctx, "Usage: /watch SYMBOL")
            return
        symbol = ctx.args[0].strip().upper()
        inferred_display = symbol
        if any(tag in symbol for tag in ("CE", "PE", "FUT")) and not symbol.startswith(
            ("NFO:", "NSE:", "BSE:")
        ):
            inferred_display = f"NFO:{symbol}"
        try:
            mdm = self.deps.market_data_manager
            if mdm is None:
                await self._reply(chat, ctx, "MarketDataManager unavailable.")
                return
            tracked = mdm.ensure_tracking(symbol, seed=True)
            if tracked:
                log.info(
                    "Condition met: telegram_cmd_watch_tracked",
                    extra={
                        "event": "telegram_cmd_watch_tracked",
                        "symbol": symbol,
                        "display_symbol": inferred_display,
                        "polling": self._polling_enabled(),
                    },
                )
                poll_state = self._flag(self._polling_enabled())
                await self._reply(
                    chat,
                    ctx,
                    f"👀 Watching {inferred_display} (poll={poll_state}).",
                )
            else:
                await self._reply(chat, ctx, f"Failed to watch {symbol}.")
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_watch: %s",
                exc,
                extra={"event": "telegram_cmd_watch_error", "symbol": symbol},
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"Error watching {symbol}: {exc}")

    @command_meta("/unwatch SYMBOL", "Stop REST tracking for a symbol.")
    async def cmd_unwatch(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Stop tracking a symbol via the MarketDataManager REST poller.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_unwatch",
            extra={"event": "telegram_cmd_unwatch_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_unwatch_guard_blocked",
                extra={"event": "telegram_cmd_unwatch_guard_blocked"},
            )
            return
        if not ctx.args:
            await self._reply(chat, ctx, "Usage: /unwatch SYMBOL")
            return
        symbol = ctx.args[0].strip().upper()
        try:
            mdm = self.deps.market_data_manager
            if mdm is None:
                await self._reply(chat, ctx, "MarketDataManager unavailable.")
                return
            existed = mdm.untrack(symbol)
            if existed:
                log.info(
                    "Condition met: telegram_cmd_unwatch_removed",
                    extra={
                        "event": "telegram_cmd_unwatch_removed",
                        "symbol": symbol,
                    },
                )
                await self._reply(chat, ctx, f"🗑️ Unwatched {symbol}.")
            else:
                await self._reply(chat, ctx, f"{symbol} was not being watched.")
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_unwatch: %s",
                exc,
                extra={"event": "telegram_cmd_unwatch_error", "symbol": symbol},
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"Error unwatching {symbol}: {exc}")

    @command_meta("/tracking", "List symbols tracked by REST polling.")
    async def cmd_tracking(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """List tracked symbols maintained by the MarketDataManager.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_tracking",
            extra={"event": "telegram_cmd_tracking_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_tracking_guard_blocked",
                extra={"event": "telegram_cmd_tracking_guard_blocked"},
            )
            return
        try:
            mdm = self.deps.market_data_manager
            if mdm is None:
                await self._reply(chat, ctx, "MarketDataManager unavailable.")
                return
            rows = mdm.tracked_snapshot()
            if not rows:
                await self._reply(chat, ctx, "No symbols are currently tracked.")
                return
            await self._reply(chat, ctx, "Tracked symbols:\n" + ", ".join(rows))
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_tracking: %s",
                exc,
                extra={"event": "telegram_cmd_tracking_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"Error fetching tracking list: {exc}")

    @command_meta("/um", "Unified manager status snapshot.")
    async def cmd_um(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a concise unified manager status summary.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Telegram callback context containing chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_um",
            extra={"event": "telegram_cmd_um_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_um_guard_blocked",
                extra={"event": "telegram_cmd_um_guard_blocked"},
            )
            return
        try:
            manager = self._get_unified_manager()
            snapshot = manager.status()
            rb = self._response_builder
            polling_info: t.Mapping[str, t.Any] | None = None
            raw_polling = snapshot.get("polling")
            if isinstance(raw_polling, t.Mapping):
                polling_info = raw_polling
            poll_active = (
                bool(polling_info.get("on")) if polling_info else bool(raw_polling)
            )
            poll_tokens: str = "?"
            poll_interval: str = "?"
            if polling_info is not None:
                with suppress(Exception):
                    poll_tokens = str(int(polling_info.get("tokens", 0)))
                with suppress(Exception):
                    interval_val = polling_info.get("interval")
                    if interval_val is not None:
                        poll_interval = str(interval_val)
            polling_summary = "✅" if poll_active else "❌"
            if polling_info is not None:
                polling_summary = (
                    f"{'✅' if poll_active else '❌'} ({poll_tokens} @ {poll_interval})"
                )
            section = [
                f"{rb.section('Unified Manager')} 🧩",
                (
                    "WS: "
                    f"{'✅' if snapshot.get('ws_connected') else '❌'} | "
                    "Polling: "
                    f"{polling_summary} | "
                    f"Cache: {snapshot.get('symbols', 0)} symbols"
                ),
                (
                    "Balance: "
                    f"{self._format_currency(snapshot.get('balance'))} "
                    f"({snapshot.get('balance_source', 'n/a')})"
                ),
            ]
            text = self._compose([section])
            await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)
            log.info(
                "Condition met: telegram_cmd_um_dispatched",
                extra={"event": "telegram_cmd_um_dispatched"},
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_um: %s",
                exc,
                extra={"event": "telegram_cmd_um_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unable to fetch unified manager status.")

    @command_meta(
        "/umprobe SYMBOL",
        "Run resolver, pricing, and transport diagnostics for a symbol.",
    )
    async def cmd_umprobe(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Inspect unified manager resolver, pricing, and cache paths.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Telegram callback context containing chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_umprobe",
            extra={"event": "telegram_cmd_umprobe_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_umprobe_guard_blocked",
                extra={"event": "telegram_cmd_umprobe_guard_blocked"},
            )
            return
        if not ctx.args:
            await self._reply(chat, ctx, "Usage: /umprobe SYMBOL")
            return
        symbol = ctx.args[0].strip()
        try:
            manager = self._get_unified_manager()
            probe = manager.quote_probe(symbol)
            rb = self._response_builder
            resolver = probe.get("resolver", {})
            normalize = probe.get("normalize", {})
            transport = probe.get("transport", {})
            cache_rows = t.cast(list[dict[str, t.Any]], probe.get("cache") or [])

            lines: list[str] = [
                f"{rb.section('Ingestion Probe')} 🧪 {rb.esc(symbol)}",
                (
                    "Resolver: token="
                    f"{resolver.get('token')} exch={resolver.get('exchange')} "
                    f"seg={resolver.get('kind')}"
                ),
                f"Candidates: {resolver.get('candidates', [])}",
                (
                    "Normalize: ltp="
                    f"{normalize.get('ltp')} bid={normalize.get('bid')} "
                    f"ask={normalize.get('ask')} src={normalize.get('source')}"
                ),
                (
                    "Transport: WS="
                    f"{'connected' if transport.get('ws_connected') else 'down'} "
                    f"Poll={'on' if transport.get('polling') else 'off'} "
                    f"(interval={transport.get('poll', {}).get('interval')}s, "
                    f"max={transport.get('poll', {}).get('max_symbols')})"
                ),
            ]
            if cache_rows:
                lines.append("Cache (symbol ltp bid ask age src token)")
                for row in cache_rows:
                    age = row.get("age")
                    age_display = "n/a" if age is None else f"{age:.1f}s"
                    formatted_row = (
                        f"{row.get('symbol'):<16} {str(row.get('ltp') or '-'):>10} "
                        f"{str(row.get('bid') or '-'):>8} "
                        f"{str(row.get('ask') or '-'):>8} {age_display:>7} "
                        f"{str(row.get('src') or 'n/a'):>6} "
                        f"{str(row.get('token') or '-'):>6}"
                    )
                    lines.append(formatted_row)
            await self._reply(chat, ctx, "\n".join(lines))
            log.info(
                "Condition met: telegram_cmd_umprobe_dispatched",
                extra={"event": "telegram_cmd_umprobe_dispatched", "symbol": symbol},
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_umprobe: %s",
                exc,
                extra={"event": "telegram_cmd_umprobe_error", "symbol": symbol},
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"Unable to probe {symbol}: {exc}")

    @command_meta(
        "/umtransport [SYMBOL]",
        "Show websocket, polling, and cache state (optional symbol filter).",
    )
    async def cmd_umtransport(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Display unified manager transport and cache details.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Telegram callback context containing chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_umtransport",
            extra={"event": "telegram_cmd_umtransport_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_umtransport_guard_blocked",
                extra={"event": "telegram_cmd_umtransport_guard_blocked"},
            )
            return
        symbol = ctx.args[0].strip() if ctx.args else None
        try:
            manager = self._get_unified_manager()
            snapshot = manager.transport_status(symbol=symbol)
            rb = self._response_builder
            lines: list[str] = [
                f"{rb.section('Transport')} 🛰️",
                ("WS: " f"{'✅ connected' if snapshot.get('ws_connected') else '❌'}"),
                (
                    "Poll: "
                    f"{'✅ on' if snapshot.get('polling') else '❌ off'} "
                    f"(interval={snapshot.get('poll', {}).get('interval')}s, "
                    f"max={snapshot.get('poll', {}).get('max_symbols')})"
                ),
            ]
            if symbol:
                lines.append(f"Tracking for: {rb.esc(symbol)}")
            cache_rows = t.cast(list[dict[str, t.Any]], snapshot.get("cache") or [])
            if cache_rows:
                lines.append(
                    "sym                ltp        bid      ask    age   src   tok"
                )
                for row in cache_rows:
                    age = row.get("age")
                    age_display = "n/a" if age is None else f"{age:.1f}s"
                    formatted_row = (
                        f"{row.get('symbol'):<18} {str(row.get('ltp') or '-'):>10} "
                        f"{str(row.get('bid') or '-'):>8} "
                        f"{str(row.get('ask') or '-'):>8} {age_display:>6} "
                        f"{str(row.get('src') or 'n/a'):>5} "
                        f"{str(row.get('token') or '-'):>6}"
                    )
                    lines.append(formatted_row)
            await self._reply(chat, ctx, "\n".join(lines))
            log.info(
                "Condition met: telegram_cmd_umtransport_dispatched",
                extra={
                    "event": "telegram_cmd_umtransport_dispatched",
                    "symbol": symbol,
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_umtransport: %s",
                exc,
                extra={"event": "telegram_cmd_umtransport_error", "symbol": symbol},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unable to fetch transport snapshot.")

    @command_meta(
        "/umplan SYMBOL BUY|SELL QTY",
        "Size a trade through the unified manager planner.",
    )
    async def cmd_umplan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Compute sizing for a symbol via the unified manager.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Telegram callback context containing chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_umplan",
            extra={"event": "telegram_cmd_umplan_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_umplan_guard_blocked",
                extra={"event": "telegram_cmd_umplan_guard_blocked"},
            )
            return
        args = ctx.args or []
        if len(args) < 3:
            await self._reply(chat, ctx, "Usage: /umplan SYMBOL BUY|SELL QTY")
            return
        symbol = args[0].strip()
        side = args[1].strip().upper()
        try:
            qty = int(args[2])
        except Exception:  # noqa: BLE001
            await self._reply(chat, ctx, "Quantity must be an integer.")
            return
        plan, error_message = self._invoke_um_plan(symbol, side, qty)
        if plan is None:
            await self._reply(chat, ctx, error_message or "Planner unavailable.")
            return
        if not plan.ok:
            await self._reply(
                chat,
                ctx,
                (
                    f"❌ Plan blocked: {plan.reason or 'unknown'} "
                    f"| est=₹{plan.est_required:.2f}"
                ),
            )
            return
        meta = plan.meta or {}
        price_value = meta.get("ltp")
        price_source = canonical_price_source(meta.get("price_source"))
        lot_size = meta.get("lot_size")
        tradingsymbol = meta.get("tradingsymbol")
        details = [
            f"✅ Plan OK {symbol} {side} x{plan.qty}",
            (
                f"Est required: ₹{plan.est_required:.2f} | "
                f"price={price_value} ({price_source}) lot={lot_size}"
            ),
        ]
        if tradingsymbol:
            details.append(f"tradingsymbol={tradingsymbol}")
        if price_source in {"historical", "est"}:
            details.append(
                "⚠️ Using non-live price source; order will be re-priced on open."
            )
        await self._reply(chat, ctx, "\n".join(details))
        log.info(
            "Condition met: telegram_cmd_umplan_dispatched",
            extra={
                "event": "telegram_cmd_umplan_dispatched",
                "symbol": symbol,
                "qty": plan.qty,
            },
        )

    @command_meta("/umwatch SYMBOL", "Ensure unified manager tracking for a symbol.")
    async def cmd_umwatch(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Trigger unified manager tracking for a symbol.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Telegram callback context containing chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_umwatch",
            extra={"event": "telegram_cmd_umwatch_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_umwatch_guard_blocked",
                extra={"event": "telegram_cmd_umwatch_guard_blocked"},
            )
            return
        if not ctx.args:
            await self._reply(chat, ctx, "Usage: /umwatch SYMBOL")
            return
        symbol = ctx.args[0].strip()
        try:
            manager = self._get_unified_manager()
            result = manager.ensure_tracking(symbol)
            await self._reply(
                chat,
                ctx,
                (
                    f"👀 Watching {symbol} "
                    f"(ws={'on' if result.get('ws') else 'off'}, "
                    f"poll={'on' if result.get('poll') else 'off'}, "
                    f"token={result.get('token')})"
                ),
            )
            log.info(
                "Condition met: telegram_cmd_umwatch_dispatched",
                extra={
                    "event": "telegram_cmd_umwatch_dispatched",
                    "symbol": symbol,
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_umwatch: %s",
                exc,
                extra={"event": "telegram_cmd_umwatch_error", "symbol": symbol},
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"Unable to watch {symbol}: {exc}")

    @command_meta(
        "/uminstruments SYMBOL",
        "Show resolver metadata for a symbol via unified manager.",
    )
    async def cmd_uminstruments(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Return unified manager resolver metadata for a symbol.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Telegram callback context containing chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_uminstruments",
            extra={"event": "telegram_cmd_uminstruments_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_uminstruments_guard_blocked",
                extra={"event": "telegram_cmd_uminstruments_guard_blocked"},
            )
            return
        if not ctx.args:
            await self._reply(chat, ctx, "Usage: /uminstruments SYMBOL")
            return
        symbol = ctx.args[0].strip()
        try:
            manager = self._get_unified_manager()
            instrument = manager.resolve(symbol)
            await self._reply(
                chat,
                ctx,
                (
                    f"{symbol}: exch={instrument.exchange} token={instrument.token} "
                    f"lot={instrument.lot_size} kind={instrument.kind} "
                    f"expiry={instrument.expiry or '-'}"
                ),
            )
            log.info(
                "Condition met: telegram_cmd_uminstruments_dispatched",
                extra={
                    "event": "telegram_cmd_uminstruments_dispatched",
                    "symbol": symbol,
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_uminstruments: %s",
                exc,
                extra={"event": "telegram_cmd_uminstruments_error", "symbol": symbol},
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"Unable to resolve {symbol}: {exc}")

    @command_meta(
        "/umwarm",
        "Reload resolver cache from disk and CSV before warming the UM.",
    )
    async def cmd_umwarm(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Force-refresh the resolver cache from the configured CSV dump.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Telegram callback context containing chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_umwarm",
            extra={"event": "telegram_cmd_umwarm_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_umwarm_guard_blocked",
                extra={"event": "telegram_cmd_umwarm_guard_blocked"},
            )
            return
        csv_path = getattr(self.deps, "instrument_csv_path", None)
        if not csv_path:
            await self._reply(chat, ctx, "Instrument CSV path is not configured.")
            return
        resolver = getattr(self.deps, "instrument_resolver", None)
        if resolver is None:
            await self._reply(chat, ctx, "Instrument resolver unavailable.")
            return
        db_path = getattr(self.deps, "instrument_db_path", None) or str(csv_path)
        conn = None
        try:
            conn = ensure_sqlite(db_path)
            summary = refresh_from_csv(conn, str(csv_path))
            rows = load_rows_for_resolver(conn)
            resolver.warm_from_broker_dump(rows)
            tokens = len(rows)
            options = (
                int(summary.get("stored", tokens))
                if isinstance(summary, dict)
                else tokens
            )
            timestamp = datetime.now(timezone.utc)
            universe_state = getattr(self.deps, "instrument_universe", None)
            if isinstance(universe_state, InstrumentUniverseStatus):
                universe_state.record_refresh(
                    tokens=tokens,
                    options=options,
                    source="manual",
                    path=db_path,
                    timestamp=timestamp,
                )
            METRICS.record_resolver_tokens(source="manual", count=tokens)
            await self._reply(
                chat,
                ctx,
                (f"Resolver warmed from CSV. Tokens={tokens} " f"Options={options}"),
            )
            log.info(
                "Condition met: telegram_cmd_umwarm_success",
                extra={
                    "event": "telegram_cmd_umwarm_success",
                    "tokens": tokens,
                    "options": options,
                    "path": db_path,
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_umwarm: %s",
                exc,
                extra={"event": "telegram_cmd_umwarm_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, f"Resolver warm failed: {exc}")
        finally:
            if conn is not None:
                with suppress(Exception):
                    conn.close()

    @command_meta(
        "/umstats",
        "Show resolver cache counts and miss ratios.",
    )
    async def cmd_umstats(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Display resolver token counts, options, and miss ratio.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Telegram callback context containing chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_umstats",
            extra={"event": "telegram_cmd_umstats_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_umstats_guard_blocked",
                extra={"event": "telegram_cmd_umstats_guard_blocked"},
            )
            return
        tokens = 0
        options = 0
        last_refresh_display = "unknown"
        db_path = getattr(self.deps, "instrument_db_path", None)
        conn = None
        try:
            if db_path:
                conn = ensure_sqlite(db_path)
                row = conn.execute(
                    (
                        "SELECT COUNT(1) AS count, MAX(updated_at) AS refreshed "
                        "FROM instruments"
                    )
                ).fetchone()
                if row:
                    options = int(row["count"] or 0)
                    tokens = options
                    refreshed = row["refreshed"]
                    if refreshed:
                        try:
                            parsed = datetime.fromisoformat(str(refreshed))
                        except ValueError:
                            parsed = None
                        if parsed is not None:
                            if parsed.tzinfo is None:
                                parsed = parsed.replace(tzinfo=timezone.utc)
                            ist_time = parsed.astimezone(ZoneInfo("Asia/Kolkata"))
                            last_refresh_display = ist_time.strftime(
                                "%Y-%m-%d %H:%M:%S %Z"
                            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_umstats db read: %s",
                exc,
                extra={"event": "telegram_cmd_umstats_db_error"},
                exc_info=exc,
            )
        finally:
            if conn is not None:
                with suppress(Exception):
                    conn.close()
        universe_state = getattr(self.deps, "instrument_universe", None)
        if isinstance(universe_state, InstrumentUniverseStatus):
            if universe_state.tokens:
                tokens = max(tokens, universe_state.tokens)
            if universe_state.options:
                options = max(options, universe_state.options)
            if universe_state.last_refresh is not None:
                ts = universe_state.last_refresh
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                ist_time = ts.astimezone(ZoneInfo("Asia/Kolkata"))
                last_refresh_display = ist_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        resolver_summary = METRICS.resolver_summary()
        miss_total = int(resolver_summary.get("miss_total", 0))
        miss_symbols = int(resolver_summary.get("miss_symbols", 0))
        tracked_tokens = int(resolver_summary.get("tokens", 0))
        base_tokens = max(tokens, tracked_tokens, 1)
        miss_ratio = miss_total / float(base_tokens)
        message = (
            "Instrument universe stats:\n"
            f"Tokens: {base_tokens}\n"
            f"Options: {options}\n"
            f"Last refresh: {last_refresh_display}\n"
            f"Resolver misses: total={miss_total} unique={miss_symbols} "
            f"ratio={miss_ratio:.4f}"
        )
        try:
            await self._reply(chat, ctx, message)
            log.info(
                "Condition met: telegram_cmd_umstats_dispatched",
                extra={
                    "event": "telegram_cmd_umstats_dispatched",
                    "tokens": base_tokens,
                    "options": options,
                    "miss_total": miss_total,
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_umstats reply: %s",
                exc,
                extra={"event": "telegram_cmd_umstats_reply_error"},
                exc_info=exc,
            )

    def _polling_enabled(self) -> bool:
        """Return ``True`` if the MarketDataManager REST poller is active.

        Args:
            None.

        Returns:
            Boolean indicating REST polling status.

        Raises:
            None.
        """

        try:
            mdm = self.deps.market_data_manager
            if mdm is None:
                return False
            return bool(getattr(mdm, "_rest_poll_enabled", False))
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in _polling_enabled: %s",
                exc,
                extra={"event": "telegram_polling_enabled_error"},
                exc_info=exc,
            )
            return False

    def _flag(self, on: bool) -> str:
        """Return a lowercase flag representation for boolean states.

        Args:
            on: Boolean indicator to convert.

        Returns:
            String "on" when ``True`` else "off".

        Raises:
            None.
        """

        return "on" if on else "off"

    @command_meta(
        "/transport [SYMBOL] [LIMIT]",
        "Show transport status and tracked symbol snapshot.",
    )
    async def cmd_transport(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Render websocket and polling health with per-symbol tracking table.

        Args:
            update: Telegram update payload containing the command context.
            ctx: Telegram context with parsed arguments and chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_transport",
            extra={"event": "telegram_cmd_transport_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return

        hints: list[str] = []
        args = ctx.args or []
        symbol_filter: str | None = None
        list_limit = 10
        if args:
            first = (args[0] or "").strip()
            if first.isdigit():
                list_limit = max(1, min(int(first), 100))
            else:
                symbol_filter = first.upper()
                if len(args) >= 2:
                    try:
                        list_limit = max(1, min(int(args[1]), 100))
                    except (TypeError, ValueError) as exc:
                        log.error(
                            "Failure parsing transport limit: %s",
                            exc,
                            extra={"event": "telegram_cmd_transport_limit_error"},
                        )
                        hints.append("Limit must be an integer between 1 and 100.")
        if symbol_filter and list_limit == 10:
            list_limit = 15

        try:
            rb = self._response_builder
            ws_snapshot = self._ws_snapshot()
            poll_snapshot = self._polling_snapshot() or {}
            table_lines, table_hints = self._symbol_tracking_rows(
                limit=list_limit, filter_symbol=symbol_filter
            )
            hints.extend(table_hints)

            ws_disabled = bool(ws_snapshot.get("disabled"))
            ws_available = bool(ws_snapshot.get("available"))
            ws_connected = bool(ws_snapshot.get("connected"))
            if ws_disabled:
                ws_state = "disabled"
            elif not ws_available:
                ws_state = "missing"
            elif ws_connected:
                ws_state = "connected"
            else:
                ws_state = str(ws_snapshot.get("state") or "disconnected")
            ws_line = f"{EMOJI['ws']} WS: <b>{rb.esc(ws_state)}</b>"
            ws_details: list[str] = []
            hb_age = ws_snapshot.get("heartbeat_age")
            if isinstance(hb_age, (int, float)):
                ws_details.append(f"hbΔ={self._fmt_age(float(hb_age))}")
                if float(hb_age) > 5.0 and not ws_disabled:
                    hints.append("Websocket heartbeat stale (>5s).")
            subs_count = ws_snapshot.get("subscription_count")
            if isinstance(subs_count, int) and subs_count >= 0:
                ws_details.append(f"subs={subs_count}")
            if ws_details:
                ws_line += f" <i>{rb.esc(', '.join(ws_details))}</i>"
            for error_text in ws_snapshot.get("errors", []):
                if error_text:
                    hints.append(self._short_reason(str(error_text)))
            if not ws_disabled and ws_available and not ws_connected:
                hints.append("Websocket disconnected; check broker session.")

            poll_running = bool(poll_snapshot.get("running"))
            poll_tokens_raw = poll_snapshot.get("tokens")
            poll_tokens = 0
            if isinstance(poll_tokens_raw, (int, float)):
                poll_tokens = int(poll_tokens_raw)
            if poll_tokens == 0 and not poll_running:
                poll_label = "off"
            else:
                poll_label = f"tokens={poll_tokens}"
            poll_line = f"{EMOJI['poll']} Poll: <b>{rb.esc(poll_label)}</b>"
            poll_details: list[str] = []
            poll_interval = poll_snapshot.get("interval_s")
            if isinstance(poll_interval, (int, float)) and poll_interval > 0:
                poll_details.append(f"@{float(poll_interval):.2f}s")
            poll_batch = poll_snapshot.get("batch_size")
            if isinstance(poll_batch, (int, float)) and int(poll_batch) > 0:
                poll_details.append(f"batch={int(poll_batch)}")
            poll_status = poll_snapshot.get("status")
            if poll_status:
                poll_details.append(str(poll_status))
            if poll_details:
                poll_line += f" <i>{rb.esc(', '.join(poll_details))}</i>"
            if poll_tokens > 0 and not poll_running:
                hints.append("Polling configured but streamer not running.")

            mdm = getattr(self.deps, "market_data_manager", None)
            cache_count = 0
            if mdm is not None:
                latest_attr = getattr(mdm, "_latest_ticks", None)
                if isinstance(latest_attr, dict):
                    cache_count = len(latest_attr)
            cache_line = ""
            if cache_count:
                cache_line = f"🗃️ Cache: {cache_count} symbols"

            header_line = f"{rb.section('Transport')}  {EMOJI['transport']}"
            block: list[str] = [header_line, ws_line, poll_line]
            if cache_line:
                block.append(cache_line)
            if symbol_filter:
                block.append(
                    f"{EMOJI['list']} Tracking for: <b>{rb.esc(symbol_filter)}</b>"
                )
            table_block = [rb.mono("\n".join(table_lines))]
            text = self._compose([block, table_block], hints=hints)
            await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)
            log.info(
                "Condition met: cmd_transport_response",
                extra={
                    "event": "telegram_cmd_transport_response",
                    "symbols": max(len(table_lines) - 2, 0),
                    "limit": list_limit,
                    "filter": symbol_filter,
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_transport: %s",
                exc,
                extra={"event": "telegram_cmd_transport_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "transport failed. Inspect logs.")

    @command_meta("/qprobe SYMBOL", "Trace quote ingestion path end-to-end.")
    async def cmd_qprobe(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Execute an end-to-end quote probe for market data diagnostics.

        Args:
            update: Telegram update payload containing the command context.
            ctx: Telegram context with parsed arguments and chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_qprobe",
            extra={"event": "telegram_cmd_qprobe_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        args = ctx.args or []
        if not args:
            await self._reply(
                chat,
                ctx,
                "Usage: /qprobe SYMBOL (e.g., NIFTY25OCT25900CE)",
            )
            return
        symbol = args[0].strip().upper()
        mdm = getattr(self.deps, "market_data_manager", None)
        if mdm is None:
            await self._reply(chat, ctx, "Market data manager unavailable.")
            return
        try:
            report = mdm.probe_quote(symbol)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_qprobe: %s",
                exc,
                extra={"event": "telegram_cmd_qprobe_error", "symbol": symbol},
                exc_info=exc,
            )
            await self._reply(
                chat,
                ctx,
                "Probe failed due to internal error. Inspect logs for details.",
            )
            return

        rb = self._response_builder
        resolver_meta = report.get("resolver", {})
        cache_meta = report.get("cache", {})
        broker_meta = report.get("broker_raw", {})
        normalized_meta = report.get("normalized", {})
        transport_meta = report.get("transport", {})
        hints = report.get("hints") or []

        def _flag(ok: bool) -> str:
            return EMOJI["ok"] if ok else EMOJI["fail"]

        header = f"{rb.section('Pipeline Probe')} {EMOJI['probe']} {rb.esc(symbol)}"
        resolver_section = rb.bullets(
            [
                "Resolver",
                (
                    f"{_flag(bool(resolver_meta.get('ok')))} token="
                    f"{rb.esc(str(resolver_meta.get('token')))} "
                    f"exch={rb.esc(str(resolver_meta.get('exchange')))} "
                    f"seg={rb.esc(str(resolver_meta.get('segment')))}"
                ),
            ]
        )
        raw_keys = ", ".join(str(key) for key in broker_meta.get("keys") or [])
        broker_section = rb.bullets(
            [
                "Broker (REST)",
                f"{_flag(bool(broker_meta.get('ok')))} keys=[{rb.esc(raw_keys)}]",
            ]
        )
        normalized_section = rb.bullets(
            [
                "Normalize",
                (
                    f"{_flag(bool(normalized_meta.get('ok')))} ltp="
                    f"{rb.esc(str(normalized_meta.get('ltp')))} "
                    f"bid={rb.esc(str(normalized_meta.get('bid')))} "
                    f"ask={rb.esc(str(normalized_meta.get('ask')))}"
                ),
            ]
        )
        cache_lines = [
            "Cache (MDM)",
            (
                f"{_flag(bool(cache_meta.get('has_tick')))} ltp="
                f"{rb.esc(str(cache_meta.get('ltp')))} "
                f"bid={rb.esc(str(cache_meta.get('bid')))} "
                f"ask={rb.esc(str(cache_meta.get('ask')))}"
            ),
        ]
        age_value = cache_meta.get("age_s")
        source_value = cache_meta.get("source")
        extras: list[str] = []
        if isinstance(age_value, (int, float)):
            extras.append(f"age={float(age_value):.1f}s")
        if source_value:
            extras.append(f"src={rb.esc(str(source_value))}")
        if extras:
            cache_lines.append("(" + ", ".join(extras) + ")")
        cache_section = rb.bullets(cache_lines)
        transport_section = rb.bullets(
            [
                "Transport",
                (
                    "WS: "
                    f"{'enabled' if transport_meta.get('ws_enabled') else 'disabled'} "
                    f"({_flag(bool(transport_meta.get('ws_connected')))} connected)"
                ),
                (
                    "Poll: "
                    f"{'on' if transport_meta.get('poll_enabled') else 'off'} "
                    f"(interval={transport_meta.get('poll_interval')}s, "
                    f"max={transport_meta.get('poll_max')})"
                ),
            ]
        )

        sections = [
            header,
            resolver_section,
            broker_section,
            normalized_section,
            cache_section,
            transport_section,
        ]
        if hints:
            hint_lines = "<br>".join(f"- {rb.esc(str(hint))}" for hint in hints)
            sections.append(f"Hints{rb.br()}<code>{hint_lines}</code>")

        await self._reply(chat, ctx, rb.br().join(sections), parse_mode=ParseMode.HTML)
        log.info(
            "Condition met: cmd_qprobe_response",
            extra={
                "event": "telegram_cmd_qprobe_complete",
                "symbol": symbol,
                "hints": len(hints),
            },
        )

    @command_meta(
        "/ingestprobe SYMBOL",
        "Run resolver → broker → cache ingestion diagnostics.",
    )
    async def cmd_ingestprobe(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Execute ingestion pipeline probe for the provided symbol.

        Args:
            update: Telegram update payload containing the command context.
            ctx: Telegram context with parsed arguments and chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_ingestprobe",
            extra={"event": "telegram_cmd_ingestprobe_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_ingestprobe_guard_blocked",
                extra={"event": "telegram_cmd_ingestprobe_guard_blocked"},
            )
            return

        args = list(ctx.args or [])
        if not args:
            await self._reply(chat, ctx, "Usage: /ingestprobe SYMBOL")
            return

        symbol = args[0].strip().upper()
        mdm = getattr(self.deps, "market_data_manager", None)
        broker = getattr(self.deps, "broker_client", None)
        resolver = getattr(self.deps, "resolver", None)
        ws_manager = (
            None
            if self._ws_disabled()
            else getattr(self.deps, "websocket_manager", None)
        )
        rb = self._response_builder
        hints: list[str] = []

        try:
            poll_on = False
            if mdm is not None:
                try:
                    poll_on = bool(getattr(mdm, "_rest_poll_enabled", False))
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure reading poll flag in cmd_ingestprobe: %s",
                        exc,
                        extra={"event": "telegram_cmd_ingestprobe_poll_flag_error"},
                        exc_info=exc,
                    )

            looks_option = self._looks_like_option(symbol)
            token: int | None = None
            exchange: str | None = None
            if resolver is not None:
                with suppress(Exception):
                    resolved = resolver.resolve_symbol_to_token(symbol)  # type: ignore[attr-defined]
                    if resolved is not None:
                        token = int(resolved)
                with suppress(Exception):
                    exchange_candidate = resolver.resolve_exchange(symbol)  # type: ignore[attr-defined]
                    if exchange_candidate:
                        exchange = str(exchange_candidate)
            if exchange is None and looks_option:
                exchange = "NFO"
            segment = "derivative" if looks_option else "equity"

            header = f"{rb.section('Ingestion Probe')} 🧪 {rb.esc(symbol)}"
            resolver_block = [
                "Resolver",
                (
                    f"{'✅' if token is not None else '❌'} "
                    f"token={rb.esc(str(token) if token is not None else 'n/a')} "
                    f"exch={rb.esc(exchange or 'n/a')} seg={rb.esc(segment)}"
                ),
            ]

            candidate_keys: list[str] = []
            if mdm is not None:
                try:
                    candidate_fn = getattr(mdm, "_candidate_quote_keys", None)
                    if callable(candidate_fn):
                        candidate_keys = [str(key) for key in candidate_fn(symbol)]
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure computing candidate keys in cmd_ingestprobe: %s",
                        exc,
                        extra={
                            "event": "telegram_cmd_ingestprobe_candidate_error",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )
            if not candidate_keys:
                candidate_keys = self._fallback_quote_keys(symbol)

            broker_ok = False
            broker_keys: list[str] = []
            rest_tick: dict[str, t.Any] | None = None
            if not candidate_keys:
                broker_block = ["Broker (REST)", "❌ no candidate keys generated"]
                hints.append(
                    "No candidate broker keys derived; verify symbol formatting "
                    "or add exchange prefix."
                )
            elif broker is None or not hasattr(broker, "get_quote"):
                broker_block = ["Broker (REST)", "❌ broker unavailable"]
            else:
                try:
                    quotes = await asyncio.to_thread(broker.get_quote, candidate_keys)
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure fetching broker quote in cmd_ingestprobe: %s",
                        exc,
                        extra={
                            "event": "telegram_cmd_ingestprobe_broker_error",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )
                    broker_block = [
                        "Broker (REST)",
                        f"❌ error={rb.esc(str(exc))}",
                    ]
                else:
                    if isinstance(quotes, dict):
                        broker_keys = [str(key) for key in quotes.keys()]
                        chosen: dict[str, t.Any] | None = None
                        for candidate in candidate_keys:
                            payload = quotes.get(candidate)
                            if isinstance(payload, dict):
                                chosen = t.cast(dict[str, t.Any], payload)
                                break
                        if chosen is None and quotes:
                            first_value = next(iter(quotes.values()))
                            if isinstance(first_value, dict):
                                chosen = t.cast(dict[str, t.Any], first_value)
                        if chosen is not None:
                            rest_tick = chosen
                            price = chosen.get("last_price")
                            if not isinstance(price, (int, float)):
                                price = chosen.get("ltp")
                            broker_ok = isinstance(price, (int, float))
                    else:
                        hints.append(
                            "Broker returned unexpected payload structure; "
                            "inspect logs."
                        )

                    broker_block = [
                        "Broker (REST)",
                        (
                            f"{'✅' if broker_ok else '❌'} "
                            f"keys={rb.esc(str(broker_keys) if broker_keys else '[]')}"
                        ),
                    ]

            ltp_value: float | int | None = None
            bid_value: float | int | None = None
            ask_value: float | int | None = None
            if rest_tick is not None:
                ltp_candidate = rest_tick.get("last_price")
                if not isinstance(ltp_candidate, (int, float)):
                    ltp_candidate = rest_tick.get("ltp")
                if isinstance(ltp_candidate, (int, float)):
                    ltp_value = ltp_candidate
                bid_candidate = rest_tick.get("bid")
                if isinstance(bid_candidate, (int, float)):
                    bid_value = bid_candidate
                ask_candidate = rest_tick.get("ask")
                if isinstance(ask_candidate, (int, float)):
                    ask_value = ask_candidate
            normalize_ok = isinstance(ltp_value, (int, float))
            ltp_label = rb.esc(str(ltp_value) if ltp_value is not None else "None")
            bid_label = rb.esc(str(bid_value) if bid_value is not None else "None")
            ask_label = rb.esc(str(ask_value) if ask_value is not None else "None")
            normalize_block = [
                "Normalize",
                (
                    f"{'✅' if normalize_ok else '❌'} "
                    f"ltp={ltp_label} bid={bid_label} ask={ask_label}"
                ),
            ]

            if normalize_ok and mdm is not None:
                try:
                    now_ts = float(time.time())
                    ltp_numeric = t.cast(float | int, ltp_value)
                    tick_payload: dict[str, t.Any] = {
                        "symbol": symbol,
                        "ltp": float(ltp_numeric),
                        "bid": bid_value,
                        "ask": ask_value,
                        "timestamp": now_ts,
                    }
                    with suppress(Exception):
                        mdm._latest_ticks[symbol] = tick_payload  # type: ignore[attr-defined]
                    with suppress(Exception):
                        mdm._last_tick_source[symbol] = "rest"  # type: ignore[attr-defined]
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure updating cache snapshot in cmd_ingestprobe: %s",
                        exc,
                        extra={
                            "event": "telegram_cmd_ingestprobe_cache_seed_error",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )

            cache_tick_raw: dict[str, t.Any] | None = None
            cache_age: float | None = None
            cache_source = "n/a"
            ws_connected = False
            if mdm is not None:
                with suppress(Exception):
                    cache_tick_raw = mdm.get_latest_tick(symbol)
                with suppress(Exception):
                    cache_source = str(mdm._last_tick_source.get(symbol, "n/a"))  # type: ignore[attr-defined]
                if isinstance(cache_tick_raw, dict):
                    timestamp = cache_tick_raw.get("timestamp")
                    if isinstance(timestamp, (int, float)):
                        cache_age = max(0.0, float(time.time()) - float(timestamp))
                with suppress(Exception):
                    ws_connected = bool(ws_manager and ws_manager.is_connected())
            else:
                hints.append(
                    "Market data manager unavailable; cache inspection skipped."
                )

            cache_tick = cache_tick_raw if isinstance(cache_tick_raw, dict) else None
            cache_ok = cache_tick is not None
            cache_row = self._fmt_tick_row(
                symbol,
                cache_tick,
                source=cache_source,
                age=cache_age,
                token=token,
                ws=ws_connected,
                poll=poll_on,
            )
            cache_block = [
                "Cache (MDM)",
                f"{'✅' if cache_ok else '❌'} {rb.mono(cache_row)}",
            ]

            ws_label = (
                "disabled"
                if self._ws_disabled()
                else ("ok" if ws_connected else "not connected")
            )
            poll_interval = (
                getattr(mdm, "_rest_poll_interval", "n/a") if mdm is not None else "n/a"
            )
            poll_max = (
                getattr(mdm, "_rest_poll_max_symbols", "n/a")
                if mdm is not None
                else "n/a"
            )
            poll_state = "on" if poll_on else "off"
            poll_interval_label = rb.esc(str(poll_interval))
            poll_max_label = rb.esc(str(poll_max))
            transport_block = [
                "Transport",
                f"🌐 WS: {rb.esc(str(ws_label))}",
                (
                    f"🔁 Poll: {rb.esc(poll_state)} "
                    f"(interval={poll_interval_label}s, max={poll_max_label})"
                ),
            ]

            if not broker_ok:
                hints.append(
                    "Broker get_quote did not return a usable tick; verify credentials "
                    "and symbol format."
                )
            if not cache_ok:
                hints.append(
                    "Symbol not in cache; /watch SYMBOL to enable polling or ensure "
                    "websocket connectivity."
                )
            if poll_on and broker_ok and not cache_ok:
                hints.append(
                    "Polling is active; cache should fill shortly. Review poll batch "
                    "size if delays persist."
                )
            if looks_option and all(not key.startswith("NFO:") for key in broker_keys):
                hints.append(
                    "Option symbols typically require NFO: prefix; confirm broker "
                    "key support."
                )
            if token is None and broker_ok:
                hints.append(
                    "Resolver missing token; it will populate after the first "
                    "successful polling cycle."
                )

            sections = [
                [header],
                resolver_block,
                broker_block,
                normalize_block,
                cache_block,
                transport_block,
            ]
            message = self._compose(sections, hints=hints)
            await self._reply(chat, ctx, message, parse_mode=ParseMode.HTML)
            log.info(
                "Condition met: telegram_cmd_ingestprobe_complete",
                extra={
                    "event": "telegram_cmd_ingestprobe_complete",
                    "symbol": symbol,
                    "broker_ok": broker_ok,
                    "cache_ok": cache_ok,
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_ingestprobe: %s",
                exc,
                extra={
                    "event": "telegram_cmd_ingestprobe_error",
                    "symbol": symbol,
                },
                exc_info=exc,
            )
            await self._reply(chat, ctx, "ingestprobe failed. Inspect logs.")

    @command_meta(
        "/quote_probe SYMBOL",
        "Trace quote ingestion path end-to-end (alias of /qprobe).",
    )
    async def cmd_quote_probe(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Alias for :meth:`cmd_qprobe` keeping legacy command name.

        Args:
            update: Telegram update payload containing the command context.
            ctx: Telegram context with parsed arguments and chat metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_quote_probe",
            extra={"event": "telegram_cmd_quote_probe_enter"},
        )
        await self.cmd_qprobe(update, ctx)

    @command_meta("/gate", "Inspect micro risk gate status.")
    async def cmd_gate(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        rm = self.deps.risk_manager
        if rm is None:
            await self._reply(chat, ctx, "Risk manager unavailable.")
            return
        allowed, reasons = rm.risk_gate_should_trade()
        codes = sorted({canonical(reason) for reason in reasons if reason})
        soft_only = bool(codes) and set(codes).issubset(SOFT)
        payload = {
            "allowed": bool(allowed),
            "reasons": codes,
            "soft_only": soft_only,
        }
        summary = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        await self._reply(
            chat,
            ctx,
            f"gate {self._code(summary)}",
            parse_mode=ParseMode.HTML,
        )

    def _load_mdm_margin_entries(
        self, *, force: bool = False
    ) -> tuple[list[dict[str, float | str | None]], dict[str, t.Any] | None]:
        """Return normalized margin entries from the market data manager.

        Args:
            force: Force a refresh of the cached margin snapshot before reading.

        Returns:
            tuple[list[dict[str, float | None]], dict[str, t.Any] | None]:
            Margin entries and the raw snapshot metadata.

        Raises:
            None.
        """

        log.debug(
            "Entered _load_mdm_margin_entries",
            extra={"event": "telegram_margin_entries_enter", "force": force},
        )
        mdm = getattr(self.deps, "market_data_manager", None)
        if mdm is None:
            log.warning(
                "Condition met: telegram_margin_entries_no_mdm",
                extra={"event": "telegram_margin_entries_no_mdm"},
            )
            return [], None

        try:
            mdm.refresh_margin_snapshot(force=force)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in _load_mdm_margin_entries refresh: %s",
                exc,
                extra={"event": "telegram_margin_entries_refresh_error"},
                exc_info=exc,
            )

        snapshot: dict[str, t.Any] | None
        try:
            snapshot = mdm.get_margin_snapshot()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in _load_mdm_margin_entries snapshot: %s",
                exc,
                extra={"event": "telegram_margin_entries_snapshot_error"},
                exc_info=exc,
            )
            return [], None

        if snapshot is None:
            log.info(
                "Condition met: telegram_margin_entries_snapshot_missing",
                extra={"event": "telegram_margin_entries_snapshot_missing"},
            )
            return [], None

        entries_raw = snapshot.get("entries")
        if not isinstance(entries_raw, list):
            log.info(
                "Condition met: telegram_margin_entries_empty",
                extra={"event": "telegram_margin_entries_empty"},
            )
            return [], snapshot

        def _coerce(value: t.Any) -> float | None:
            if value is None:
                return None
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(number):
                return None
            if number < 0:
                return None
            return number

        default_segment = (
            str(snapshot.get("segment") or "equity").strip().upper() or "EQUITY"
        )
        normalized: list[dict[str, float | str | None]] = []
        for entry in entries_raw:
            if not isinstance(entry, t.Mapping):
                continue
            segment_text = str(entry.get("segment") or default_segment).strip()
            segment_label = segment_text.upper() or default_segment
            available_value = _coerce(entry.get("available"))
            used_value = _coerce(entry.get("used"))
            net_value = _coerce(entry.get("net"))
            if net_value is None:
                net_value = available_value
            normalized.append(
                {
                    "segment": segment_label,
                    "available": available_value,
                    "used": used_value,
                    "net": net_value,
                }
            )

        if not normalized:
            log.info(
                "Condition met: telegram_margin_entries_normalized_empty",
                extra={"event": "telegram_margin_entries_normalized_empty"},
            )
        return normalized, snapshot

    @command_meta(
        "/balance_probe",
        "Force refresh of broker, market, and risk balance sources.",
    )
    async def cmd_balance_probe(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Probe balance retrieval across broker, MDM, data hub, and risk layers.

        Args:
            update: Telegram update invoking the diagnostic probe.
            ctx: Callback context supplied by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_balance_probe",
            extra={"event": "telegram_cmd_balance_probe_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return

        rb = self._response_builder
        broker = getattr(self.deps, "broker_client", None)
        mdm = getattr(self.deps, "market_data_manager", None)
        hub = getattr(self.deps, "data_hub", None)
        risk = getattr(self.deps, "risk_manager", None)

        br_ok = False
        br_summary: dict[str, t.Any] | None = None
        br_available: float | None = None
        br_err: str | None = None
        try:
            # [FIX] Run blocking calls in thread
            if broker is not None and hasattr(broker, "get_margin_summary"):
                summary = await asyncio.to_thread(broker.get_margin_summary, segment="equity")
                if isinstance(summary, dict):
                    br_summary = summary
            if broker is not None and hasattr(broker, "get_available_balance"):
                raw_available = await asyncio.to_thread(broker.get_available_balance, segment="equity")
                br_available = self._coerce_float_value(
                    raw_available,
                    field="broker.available",
                )
            br_ok = True
        except Exception as exc:  # noqa: BLE001
            br_err = str(exc)
            log.error(
                "Failure in cmd_balance_probe broker: %s",
                exc,
                extra={"event": "telegram_cmd_balance_probe_broker_error"},
                exc_info=exc,
            )

        mdm_ok = False
        mdm_available: float | None = None
        mdm_key: str | None = None
        mdm_err: str | None = None
        mdm_snapshot: dict[str, t.Any] = {}
        try:
            if mdm is not None and hasattr(mdm, "get_available_balance"):
                mdm_available = self._coerce_float_value(
                    mdm.get_available_balance(force=True),
                    field="mdm.available",
                )
            if mdm is not None and hasattr(mdm, "get_account_snapshot"):
                snap_raw = mdm.get_account_snapshot(force=True) or {}
                if isinstance(snap_raw, dict):
                    mdm_snapshot = {
                        key: snap_raw.get(key)
                        for key in (
                            "available",
                            "available_cash",
                            "live_balance",
                            "cash",
                            "opening_balance",
                            "net",
                        )
                        if key in snap_raw
                    }
                    for key in (
                        "available",
                        "available_cash",
                        "live_balance",
                        "cash",
                        "opening_balance",
                        "net",
                    ):
                        value = snap_raw.get(key)
                        try:
                            if value is not None and float(value) > 0:
                                mdm_key = key
                                break
                        except Exception:  # noqa: BLE001
                            continue
            mdm_ok = True
        except Exception as exc:  # noqa: BLE001
            mdm_err = str(exc)
            log.error(
                "Failure in cmd_balance_probe mdm: %s",
                exc,
                extra={"event": "telegram_cmd_balance_probe_mdm_error"},
                exc_info=exc,
            )

        dh_ok = False
        dh_available: float | None = None
        dh_err: str | None = None
        dh_snapshot: dict[str, t.Any] = {}
        try:
            if hub is not None and hasattr(hub, "get_available_balance"):
                dh_available = self._coerce_float_value(
                    hub.get_available_balance(force=True),
                    field="data_hub.available",
                )
            if hub is not None and hasattr(hub, "get_account_snapshot"):
                dsnap_raw = hub.get_account_snapshot(force=True) or {}
                if isinstance(dsnap_raw, dict):
                    dh_snapshot = {
                        key: dsnap_raw.get(key)
                        for key in (
                            "available",
                            "available_cash",
                            "live_balance",
                            "cash",
                            "opening_balance",
                            "net",
                        )
                        if key in dsnap_raw
                    }
            dh_ok = True
        except Exception as exc:  # noqa: BLE001
            dh_err = str(exc)
            log.error(
                "Failure in cmd_balance_probe data_hub: %s",
                exc,
                extra={"event": "telegram_cmd_balance_probe_data_hub_error"},
                exc_info=exc,
            )

        rm_ok = False
        rm_balance: float | None = None
        rm_source = "unknown"
        rm_err: str | None = None
        try:
            if risk is not None:
                rm_balance = self._coerce_float_value(
                    risk.refresh_account_balance(force=True),
                    field="risk.balance",
                )
                try:
                    rm_source = str(risk.balance_source_label())
                except Exception as label_exc:  # noqa: BLE001
                    rm_source = "unknown"
                    log.error(
                        "Failure in cmd_balance_probe risk label: %s",
                        label_exc,
                        extra={"event": "telegram_cmd_balance_probe_risk_label_error"},
                        exc_info=label_exc,
                    )
            rm_ok = True
        except Exception as exc:  # noqa: BLE001
            rm_err = str(exc)
            log.error(
                "Failure in cmd_balance_probe risk: %s",
                exc,
                extra={"event": "telegram_cmd_balance_probe_risk_error"},
                exc_info=exc,
            )

        def _fmt(value: float | None) -> str:
            return self._format_currency(value)

        header = f"{rb.section('Balance Probe')} 🔎"
        lines: list[str] = [header]

        lines.append(rb.section("Broker (Zerodha)"))
        if br_ok:
            lines.append(f"available: <code>{_fmt(br_available)}</code>")
            if isinstance(br_summary, dict):
                segment_map = br_summary.get("equity")
                if not isinstance(segment_map, dict):
                    segment_map = br_summary
                compact: dict[str, t.Any] = {}
                if isinstance(segment_map, dict):
                    for key in (
                        "available",
                        "available_cash",
                        "cash",
                        "net",
                        "utilised",
                    ):
                        if key in segment_map:
                            compact[key] = segment_map.get(key)
                    available_map = segment_map.get("available")
                    if isinstance(available_map, dict):
                        for key in (
                            "live_balance",
                            "cash",
                            "available_cash",
                            "opening_balance",
                        ):
                            if key in available_map:
                                compact[f"available.{key}"] = available_map.get(key)
                if compact:
                    joined = ", ".join(
                        f"{rb.esc(str(key))}={rb.esc(str(val))}"
                        for key, val in compact.items()
                    )
                    lines.append(f"summary: {joined}")
        else:
            lines.append(f"error: {rb.esc(br_err or 'unknown')}")

        lines.append("")
        lines.append(rb.section("MarketDataManager"))
        lines.append(f"available: <code>{_fmt(mdm_available)}</code>")
        lines.append(f"key: <code>{rb.esc(mdm_key or 'n/a')}</code>")
        if mdm_snapshot:
            mdm_joined = ", ".join(
                f"{rb.esc(str(key))}={rb.esc(str(val))}"
                for key, val in mdm_snapshot.items()
            )
            lines.append(f"snapshot: {mdm_joined}")
        if not mdm_ok and mdm_err:
            lines.append(f"error: {rb.esc(mdm_err)}")

        lines.append("")
        lines.append(rb.section("DataHub"))
        lines.append(f"available: <code>{_fmt(dh_available)}</code>")
        if dh_snapshot:
            dh_joined = ", ".join(
                f"{rb.esc(str(key))}={rb.esc(str(val))}"
                for key, val in dh_snapshot.items()
            )
            lines.append(f"snapshot: {dh_joined}")
        if not dh_ok and dh_err:
            lines.append(f"error: {rb.esc(dh_err)}")

        lines.append("")
        lines.append(rb.section("RiskManager"))
        lines.append(
            f"balance: <code>{_fmt(rm_balance)}</code> "
            f"(source=<code>{rb.esc(rm_source)}</code>)"
        )
        if not rm_ok and rm_err:
            lines.append(f"error: {rb.esc(rm_err)}")

        probe_report = "\n".join(lines)
        try:
            await self._reply(chat, ctx, probe_report, parse_mode=ParseMode.HTML)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_balance_probe reply: %s",
                exc,
                extra={"event": "telegram_cmd_balance_probe_reply_error"},
                exc_info=exc,
            )
            return

        log.info(
            "Condition met: telegram_cmd_balance_probe_complete",
            extra={
                "event": "telegram_cmd_balance_probe_complete",
                "broker_available": br_available,
                "mdm_available": mdm_available,
                "dh_available": dh_available,
                "rm_balance": rm_balance,
                "rm_source": rm_source,
            },
        )

    @command_meta("/balance", "Summarize broker balances by segment.")
    async def cmd_balance(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Report available broker balances across configured segments.

        Args:
            update: Telegram update initiating the command.
            ctx: Handler context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_balance",
            extra={"event": "telegram_cmd_balance_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return

        rb = self._response_builder
        entries, snapshot = await asyncio.to_thread(self._load_mdm_margin_entries, force=True)
        if not entries:
            log.info(
                "Condition met: telegram_cmd_balance_no_entries",
                extra={"event": "telegram_cmd_balance_no_entries"},
            )
            await self._reply(chat, ctx, "❌ Margin data unavailable via market hub.")
            return

        def _fmt(value: float | None) -> str:
            if value is None:
                return "n/a"
            return f"₹{value:,.2f}"

        blocks: list[str] = [f"{EMOJI['pos']} {rb.section('Broker Balance Summary')}"]
        for entry in sorted(entries, key=lambda row: row.get("segment") or ""):
            segment_label = str(entry.get("segment") or "EQUITY").upper()
            available_value = t.cast(float | None, entry.get("available"))
            used_value = t.cast(float | None, entry.get("used"))
            net_source = (
                entry.get("net") if entry.get("net") is not None else available_value
            )
            net_value = t.cast(float | None, net_source)
            block = rb.section_block(
                segment_label,
                [
                    f"Available: <code>{_fmt(available_value)}</code>",
                    f"Used: <code>{_fmt(used_value)}</code>",
                    f"Net: <code>{_fmt(net_value)}</code>",
                ],
            )
            blocks.append(block)
            log.info(
                "Condition met: telegram_cmd_balance_segment",
                extra={
                    "event": "telegram_cmd_balance_segment",
                    "segment": segment_label,
                    "available": _fmt(available_value),
                    "used": _fmt(used_value),
                    "net": _fmt(net_value),
                },
            )

        fetched_at = None
        if snapshot is not None:
            fetched_raw = snapshot.get("fetched_at")
            if isinstance(fetched_raw, (int, float)):
                fetched_at = float(fetched_raw)
        if fetched_at is not None:
            age_seconds = max(time.time() - fetched_at, 0.0)
            blocks.append(
                rb.section_block(
                    "Snapshot",
                    [f"Age: <code>{age_seconds:.1f}s</code>"],
                )
            )

        reply = rb.br().join(blocks)
        await self._reply(chat, ctx, reply, parse_mode=ParseMode.HTML)

    @command_meta(
        "/margin [SYMBOL] [QTY]",
        "Show available margin snapshot and estimated requirement for BUY 1 lot.",
    )
    async def cmd_margin(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Summarize margin feasibility and guidance for a BUY order.

        Args:
            update: Telegram update triggering the lookup.
            ctx: Handler context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_margin",
            extra={"event": "telegram_cmd_margin_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        try:
            risk_manager = getattr(self.deps, "risk_manager", None)
            balance_value: float | None = None
            balance_source = "unknown"
            if risk_manager is not None:
                balance_value, balance_source = self._risk_balance_snapshot(
                    force=True,
                    risk_manager=risk_manager,
                )

            order_manager = getattr(self.deps, "order_manager", None)
            if order_manager is None:
                await self._reply(chat, ctx, "Order manager unavailable.")
                return
            args = ctx.args or []
            symbol = args[0].strip() if args else self._primary_symbol()
            if not symbol:
                symbol = "NIFTY"
            symbol = symbol.upper()
            quantity = 1
            if len(args) > 1:
                try:
                    quantity = max(1, int(args[1]))
                except (TypeError, ValueError):
                    quantity = 1
            planner = getattr(order_manager, "_pre_trade_decision", None)
            if not callable(planner):
                await self._reply(chat, ctx, "Margin planner unavailable.")
                return
            try:
                decision, context = planner(  # type: ignore[misc]
                    symbol=symbol,
                    side="BUY",
                    quantity=quantity,
                    product=None,
                )
            except OrderPlacementError as exc:
                log.info(
                    "Condition met: telegram_cmd_margin_blocked",
                    extra={"event": "telegram_cmd_margin_blocked"},
                )
                await self._reply(
                    chat,
                    ctx,
                    f"Margin blocked: {self._short_reason(str(exc))}",
                )
                return
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_margin planner: %s",
                    exc,
                    extra={"event": "telegram_cmd_margin_error"},
                    exc_info=exc,
                )
                await self._reply(chat, ctx, "Margin planner failed unexpectedly.")
                return
            available_source = context.get("available_source", "unknown")
            if balance_source != "unknown":
                available_source = balance_source
            est_needed = float(getattr(decision, "est_required", 0.0) or 0.0)
            available_context = float(getattr(decision, "available", 0.0) or 0.0)
            available = (
                float(balance_value) if balance_value is not None else available_context
            )
            price_source_label = canonical_price_source(context.get("price_source"))
            fallback_plan: t.Any | None = None
            fallback_error: str | None = None
            fallback_meta: dict[str, t.Any] = {}
            fallback_qty = 0
            fallback_attempted = est_needed <= 0
            if fallback_attempted:
                fallback_plan, fallback_error = self._invoke_um_plan(
                    symbol,
                    "BUY",
                    quantity,
                )
                if fallback_plan is not None and fallback_plan.est_required > 0:
                    est_needed = float(fallback_plan.est_required)
                    fallback_meta = fallback_plan.meta or {}
                    price_source_label = canonical_price_source(
                        fallback_meta.get("price_source")
                    )
                    fallback_qty = fallback_plan.qty or 0
                    log.info(
                        "Condition met: telegram_cmd_margin_um_fallback",
                        extra={
                            "event": "telegram_cmd_margin_um_fallback",
                            "symbol": symbol,
                            "qty": fallback_qty,
                            "source": price_source_label,
                        },
                    )
                    METRICS.record_telegram_margin_um_fallback(
                        source=price_source_label
                    )
            remaining = available - est_needed
            required_text = self._format_currency(est_needed)
            available_text = self._format_currency(available)
            remaining_text = self._format_currency(remaining)
            updated_ts = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
            planned_type = getattr(decision, "order_type", "NRML")
            raw_planned_qty = getattr(decision, "quantity", None)
            if raw_planned_qty is None:
                planned_qty = quantity
            else:
                try:
                    planned_qty = int(raw_planned_qty)
                except (TypeError, ValueError):
                    planned_qty = quantity
            if planned_qty <= 0 and fallback_qty > 0:
                planned_qty = fallback_qty
            status_lines = [
                f"💰 Margin for {symbol} BUY x{quantity}",
                f"Required: {required_text}",
                f"Available: {available_text} ({available_source})",
                f"Post-trade: {remaining_text}",
                f"Price source: {price_source_label}",
                f"Order type: {planned_type} x{planned_qty}",
            ]
            ref_price = fallback_meta.get("ltp")
            if isinstance(ref_price, (int, float)):
                status_lines.append(f"Ref price: ₹{float(ref_price):.2f}")
            status_lines.append(f"Updated: {updated_ts}")
            warnings: list[str] = []
            extra_warnings: list[str] = []
            if balance_value is None:
                warnings.append("Broker balance unavailable.")
            if available_source in {"env", "unknown"}:
                warnings.append("Fallback balance in use; broker data unavailable.")
            elif available_source == "cache":
                warnings.append("Using cached broker balance (recent).")
            if fallback_plan is not None and fallback_plan.est_required > 0:
                extra_warnings.append(
                    "Using unified manager estimate; broker margin snapshot stale."
                )
                if price_source_label in {"historical", "est"}:
                    extra_warnings.append(
                        "Estimated price will be refreshed when markets open."
                    )
            elif fallback_attempted and fallback_error:
                extra_warnings.append(
                    f"Unified planner fallback unavailable: {fallback_error}"
                )
            if decision.ok:
                status_lines.append("Status: ✅ SUFFICIENT")
                log.info(
                    "Condition met: telegram_cmd_margin_ok",
                    extra={"event": "telegram_cmd_margin_ok"},
                )
            else:
                reason = self._humanize_margin_reason(decision.reason)
                status_lines.append("Status: ❌ INSUFFICIENT")
                status_lines.append(f"Reason: {reason}")
                status_lines.append("💡 Reduce quantity or wait for margin to free up.")
                log.info(
                    "Condition met: telegram_cmd_margin_insufficient",
                    extra={
                        "event": "telegram_cmd_margin_insufficient",
                        "reason": decision.reason,
                    },
                )
            if extra_warnings:
                warnings.extend(extra_warnings)
            if warnings:
                status_lines.append("")
                status_lines.extend(f"ℹ️ {note}" for note in warnings)
            await self._reply(chat, ctx, "\n".join(status_lines))
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_margin outer: %s",
                exc,
                extra={"event": "telegram_cmd_margin_outer_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Margin lookup failed unexpectedly.")

    @command_meta("/margins", "Fetch live margin data from broker.")
    async def cmd_margins(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Fetch live broker margin snapshot and render a summary.

        Args:
            update: Telegram update payload triggering the command.
            ctx: Callback context supplied by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_margins",
            extra={"event": "telegram_cmd_margins_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return

        rb = self._response_builder
        entries, snapshot = await asyncio.to_thread(self._load_mdm_margin_entries, force=True)
        if not entries:
            log.info(
                "Condition met: telegram_cmd_margins_no_entries",
                extra={"event": "telegram_cmd_margins_no_entries"},
            )
            await self._reply(chat, ctx, "❌ Margin data unavailable via market hub.")
            return

        def _fmt(value: float | None) -> str:
            if value is None:
                return "n/a"
            return f"₹{value:,.2f}"

        blocks: list[str] = [f"{EMOJI['pos']} {rb.section('Broker Margins')}"]
        for entry in sorted(entries, key=lambda row: row.get("segment") or ""):
            segment_label = str(entry.get("segment") or "EQUITY").upper()
            available_value = t.cast(float | None, entry.get("available"))
            used_value = t.cast(float | None, entry.get("used"))
            net_source = (
                entry.get("net") if entry.get("net") is not None else available_value
            )
            net_value = t.cast(float | None, net_source)
            blocks.append(
                rb.section_block(
                    segment_label,
                    [
                        f"Available: <code>{_fmt(available_value)}</code>",
                        f"Used: <code>{_fmt(used_value)}</code>",
                        f"Net: <code>{_fmt(net_value)}</code>",
                    ],
                )
            )
            log.info(
                "Condition met: telegram_cmd_margins_segment",
                extra={
                    "event": "telegram_cmd_margins_segment",
                    "segment": segment_label,
                    "available": _fmt(available_value),
                    "used": _fmt(used_value),
                    "net": _fmt(net_value),
                },
            )

        fetched_at = None
        if snapshot is not None:
            fetched_raw = snapshot.get("fetched_at")
            if isinstance(fetched_raw, (int, float)):
                fetched_at = float(fetched_raw)
        if fetched_at is not None:
            age_seconds = max(time.time() - fetched_at, 0.0)
            blocks.append(
                rb.section_block(
                    "Snapshot",
                    [f"Age: <code>{age_seconds:.1f}s</code>"],
                )
            )

        text = rb.br().join(blocks)
        await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)

    @command_meta(
        "/plan [SYMBOL] [SIDE] [QTY]",
        "Unified manager-backed plan; may use cached estimates off-hours.",
    )
    async def cmd_plan(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Proxy /plan to the unified manager planner for consistency.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Telegram callback context providing argument metadata.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_plan",
            extra={"event": "telegram_cmd_plan_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        args = ctx.args or []
        symbol = args[0].strip() if args else self._primary_symbol()
        if not symbol:
            symbol = "NIFTY"
        side = args[1].strip().upper() if len(args) > 1 else "BUY"
        if side not in {"BUY", "SELL"}:
            side = "BUY"
        quantity = 1
        if len(args) > 2:
            try:
                quantity = max(1, int(args[2]))
            except (TypeError, ValueError):
                await self._reply(chat, ctx, "Quantity must be an integer >= 1.")
                return
        plan, error_message = self._invoke_um_plan(symbol, side, quantity)
        if plan is None:
            await self._reply(chat, ctx, error_message or "Planner unavailable.")
            return

        meta = plan.meta or {}
        price_source = canonical_price_source(meta.get("price_source"))
        ltp = meta.get("ltp")
        lot_size = meta.get("lot_size")
        exchange = meta.get("exchange")
        tradingsymbol = meta.get("tradingsymbol")
        plan_qty = plan.qty if plan.qty > 0 else quantity

        header = f"/plan {symbol.upper()} {side} {quantity}"
        status = "ok" if plan.ok else "blocked"
        summary_line = (
            f"status={status} qty={plan_qty} "
            f"est=₹{plan.est_required:.2f} source={price_source}"
        )
        details = [header, summary_line]
        price_line = f"price={ltp} lot={lot_size}"
        if exchange:
            price_line += f" exch={exchange}"
        if tradingsymbol:
            price_line += f" ts={tradingsymbol}"
        details.append(price_line)
        if not plan.ok:
            reason = plan.reason or "unknown"
            details.append(f"reason={reason}")
            log.info(
                "Condition met: telegram_cmd_plan_blocked",
                extra={
                    "event": "telegram_cmd_plan_blocked",
                    "symbol": symbol,
                    "reason": reason,
                },
            )
        else:
            log.info(
                "Condition met: telegram_cmd_plan_ok",
                extra={
                    "event": "telegram_cmd_plan_ok",
                    "symbol": symbol,
                    "qty": plan_qty,
                    "source": price_source,
                },
            )
        if price_source in {"historical", "est"}:
            details.append("⚠️ Using estimated price; order will be re-priced on open.")
        await self._reply(chat, ctx, "\n".join(details))

    @command_meta(
        "/guard <SYMBOL> [SIDE] [QTY]",
        "Dry-run the trading guard for a symbol.",
    )
    async def cmd_guard(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        order_manager = self.deps.order_manager
        if order_manager is None:
            await self._reply(chat, ctx, "Order manager unavailable.")
            return
        args = ctx.args or []
        symbol = args[0].strip() if args else self._primary_symbol()
        if not symbol:
            symbol = "NIFTY"
        side = args[1].upper() if len(args) > 1 else "BUY"
        quantity = 1
        if len(args) > 2:
            try:
                quantity = max(1, int(args[2]))
            except (TypeError, ValueError):
                quantity = 1
        rm = self.deps.risk_manager
        allowed = True
        codes: list[str] = []
        soft_only = False
        if rm is not None:
            allowed, reasons = rm.risk_gate_should_trade()
            codes = sorted({canonical(reason) for reason in reasons if reason})
            soft_only = bool(codes) and set(codes).issubset(SOFT)
        result = "pass"
        error: str | None = None
        guard_allowed: bool | None = None
        try:
            guard_allowed = order_manager._ensure_trading_allowed(
                symbol=symbol,
                side=side,
                quantity=quantity,
            )
        except Exception as exc:  # noqa: BLE001
            error = self._short_reason(str(exc))
            result = f"hard_block:{error}"
        else:
            if guard_allowed is False:
                result = f"soft_block:{','.join(codes) or 'soft'}"
            elif not allowed and soft_only:
                result = f"soft_block:{','.join(codes) or 'soft'}"
            elif not allowed:
                result = f"blocked:{','.join(codes) or 'risk'}"
                error = error or "risk guard"
        payload = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "result": result,
            "allowed": bool(allowed),
            "reasons": codes,
            "soft_only": soft_only,
        }
        if guard_allowed is not None:
            payload["guard_allowed"] = guard_allowed
        if error:
            payload["error"] = error
        summary = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        await self._reply(
            chat,
            ctx,
            f"guard {self._code(summary)}",
            parse_mode=ParseMode.HTML,
        )

    @command_meta("/selftest", "Run freshness and risk diagnostics.")
    async def cmd_selftest(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return

        symbol = self._primary_symbol()
        limit_ms = self._adaptive_freshness_limit_ms()
        data_hub = getattr(self.deps, "data_hub", None)

        checks: list[tuple[str, t.Callable[[], tuple[bool, str, dict[str, t.Any]]]]] = (
            []
        )

        if data_hub is None:
            checks.append(
                (
                    "datahub_available",
                    lambda: (False, "missing_datahub", {"symbol": symbol}),
                )
            )
        else:
            hub_ref: DataHub = t.cast(DataHub, data_hub)

            def _data_check(
                hub: DataHub = hub_ref,
                sym: str = symbol,
                limit: int = limit_ms,
            ) -> tuple[bool, str, dict[str, t.Any]]:
                return assess_datahub_fresh(hub, sym, limit)

            checks.append(("datahub_fresh", _data_check))

        checks.append(("risk_state", self._assess_risk_state))

        results = assess_suite(checks)
        payload = {
            "ok": all(result.ok for result in results),
            "results": [
                {
                    "name": result.name,
                    "ok": result.ok,
                    "detail": result.detail,
                    "meta": result.meta,
                }
                for result in results
            ],
        }
        summary = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        icon = "✅" if payload["ok"] else "❌"
        message = f"{icon} selftest\n{self._code(summary)}"
        await self._reply(chat, ctx, message, parse_mode=ParseMode.HTML)

    def _assess_symbol_overview(self, symbol: str) -> tuple[bool, str]:
        lines: list[str] = []
        overall_ok = True
        data_hub = getattr(self.deps, "data_hub", None)
        fresh_ok = True
        if data_hub is None:
            fresh_ok = False
            lines.append("fresh: data_hub_unavailable")
        else:
            try:
                fresh_ok, meta = data_hub.is_fresh(symbol)
            except Exception as exc:  # noqa: BLE001
                fresh_ok = False
                lines.append(f"fresh: error={self._short_reason(str(exc))}")
            else:
                effective = meta.get("effective_ms")
                threshold = meta.get("threshold_ms")
                reason = meta.get("reason")
                eff_label = "n/a" if effective is None else f"{float(effective):.0f}"
                limit_label = "n/a" if threshold is None else f"{float(threshold):.0f}"
                reason_label = reason or "-"
                freshness = (
                    f"fresh: {bool(fresh_ok)} eff_ms={eff_label} "
                    f"limit_ms={limit_label} reason={reason_label}"
                )
                lines.append(freshness)
        rm = self.deps.risk_manager
        gate_allowed = True
        gate_codes: list[str] = []
        gate_soft = False
        if rm is None:
            gate_allowed = False
            lines.append("gate: risk_manager_unavailable")
        else:
            gate_allowed, gate_reasons = rm.risk_gate_should_trade()
            gate_codes = sorted(
                {canonical(reason) for reason in gate_reasons if reason}
            )
            gate_soft = bool(gate_codes) and set(gate_codes).issubset(SOFT)
            gate_summary = (
                f"gate: allowed={bool(gate_allowed)} "
                f"reasons={','.join(gate_codes) or '-'} soft_only={gate_soft}"
            )
            lines.append(gate_summary)
        om = self.deps.order_manager
        guard_ok = True
        if om is None:
            guard_ok = False
            lines.append("guard: order_manager_unavailable")
        else:
            guard_allowed: bool | None = None
            try:
                guard_allowed = om._ensure_trading_allowed(
                    symbol=symbol, side="BUY", quantity=1
                )
            except Exception as exc:  # noqa: BLE001
                guard_ok = False
                lines.append(f"guard: hard_block error={self._short_reason(str(exc))}")
            else:
                if guard_allowed is False:
                    lines.append(
                        "guard: soft_block reasons=" f"{','.join(gate_codes) or 'soft'}"
                    )
                elif not gate_allowed and gate_soft:
                    lines.append(
                        "guard: soft_block reasons=" f"{','.join(gate_codes) or 'soft'}"
                    )
                elif not gate_allowed:
                    guard_ok = False
                    lines.append(
                        "guard: blocked reasons=" f"{','.join(gate_codes) or 'risk'}"
                    )
                else:
                    lines.append("guard: pass")
        overall_ok = bool(fresh_ok and (gate_allowed or gate_soft) and guard_ok)
        return overall_ok, self._code("\n".join(lines))

    def _format_assess_symbol(self, symbol: str) -> str:
        """Render a compact assess summary for *symbol*."""

        data_hub = getattr(self.deps, "data_hub", None)
        fresh_ok = False
        mono_ms: t.Any = None
        server_ms: t.Any = None
        threshold_ms: t.Any = None
        reason_label: str = "-"
        if data_hub is None:
            reason_label = "data_hub_unavailable"
        else:
            try:
                fresh_ok, meta = data_hub.is_fresh(symbol)
            except Exception as exc:  # noqa: BLE001
                fresh_ok = False
                reason_label = f"error:{self._short_reason(str(exc))}"
            else:
                meta_map: dict[str, t.Any]
                if isinstance(meta, dict):
                    meta_map = dict(meta)
                else:
                    try:
                        meta_map = dict(t.cast(t.Mapping[str, t.Any], meta))
                    except Exception:  # pragma: no cover - defensive
                        meta_map = {}
                mono_ms = meta_map.get("mono_age_ms")
                server_ms = meta_map.get("server_age_ms")
                threshold_ms = meta_map.get("threshold_ms")
                raw_reason = meta_map.get("reason")
                reason_label = str(raw_reason) if raw_reason else "-"

        risk_manager = getattr(self.deps, "risk_manager", None)
        allowed = True
        reason_tuple: tuple[str, ...] = ()
        if risk_manager is None:
            allowed = False
            reason_tuple = ("risk_manager_unavailable",)
        else:
            try:
                allowed, reason_tuple = risk_manager.risk_gate_should_trade()
            except Exception as exc:  # noqa: BLE001
                allowed = False
                reason_tuple = (f"error:{self._short_reason(str(exc))}",)

        def _fmt(value: t.Any) -> str:
            if isinstance(value, (int, float)):
                return f"{float(value):.0f}"
            if value is None:
                return "None"
            return str(value)

        reason_text = ",".join(reason_tuple) if reason_tuple else "-"
        order_manager = getattr(self.deps, "order_manager", None)
        margin_line = "margin: planner_unavailable"
        if order_manager is None:
            margin_line = "margin: order_manager_missing"
        else:
            planner = getattr(order_manager, "_pre_trade_decision", None)
            if callable(planner):
                try:
                    decision, context = planner(  # type: ignore[misc]
                        symbol=symbol,
                        side="BUY",
                        quantity=1,
                        product=None,
                    )
                except Exception as exc:  # noqa: BLE001
                    margin_line = f"margin: error={self._short_reason(str(exc))}"
                else:
                    avail_src = context.get("available_source", "unknown")
                    margin_line = (
                        "margin: "
                        f"ok={decision.ok} reason={decision.reason or '-'} "
                        f"type={decision.order_type} qty={decision.quantity} "
                        f"est={_fmt(decision.est_required)} "
                        f"avail={_fmt(decision.available)} src={avail_src}"
                    )
            else:
                margin_line = "margin: planner_unavailable"
        text = (
            f"Assess {symbol}\n"
            f"fresh={fresh_ok} ages(ms): mono={_fmt(mono_ms)} "
            f"server={_fmt(server_ms)} limit={_fmt(threshold_ms)} "
            f"reason={reason_label}\n"
            f"micro_gate: allowed={allowed} reasons={reason_text}\n"
            f"{margin_line}"
        )
        return text

    @command_meta(
        "/assess <area>",
        "Targeted checks for data, risk, orders, or broker subsystems.",
    )
    async def cmd_assess(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        args = ctx.args or []
        if not args:
            default_symbol = self._primary_symbol() or "NIFTY"
            ctx.args = [default_symbol, "BUY", "1"]
            await self.cmd_plan(update, ctx)
            return
        area = args[0].strip().lower()
        rb = self._response_builder

        known = {"data", "risk", "orders", "broker"}
        if area not in known:
            symbol = args[0].strip()
            if not symbol:
                primary = self._primary_symbol()
                symbol = primary if primary else "NIFTY"
            symbol = symbol.upper()
            summary = self._format_assess_symbol(symbol)
            await self._reply(chat, ctx, summary)
            return

        if area == "data":
            data_hub = getattr(self.deps, "data_hub", None)
            symbol = self._primary_symbol()
            limit_ms = self._adaptive_freshness_limit_ms()
            if data_hub is None:
                ok = False
                detail = "datahub missing"
            else:
                try:
                    ok, detail, meta = assess_datahub_fresh(data_hub, symbol, limit_ms)
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    detail = f"error: {self._short_reason(str(exc))}"
                    meta = {"symbol": symbol}
                age_ms = meta.get("age_ms") if isinstance(meta, dict) else None
                age_label = "n/a"
                if isinstance(age_ms, (int, float)):
                    age_label = f"{float(age_ms):.0f}"
                detail = (
                    f"{detail} symbol={symbol} age={age_label}ms limit={limit_ms}ms"
                )
        elif area == "risk":
            ok, detail, meta = self._assess_risk_state()
            reason = ""
            if isinstance(meta, dict):
                cooldown = meta.get("cooldown_remaining")
                if isinstance(cooldown, (int, float)):
                    detail += f" cooldown={float(cooldown):.0f}s"
                if meta.get("breaker_tripped"):
                    detail += " breaker=ON"
                else:
                    detail += " breaker=OFF"
                reason = str(meta.get("reason", "")) if meta.get("reason") else ""
            if reason:
                detail += f" reason={reason}"
        elif area == "orders":
            om = self.deps.order_manager
            if om is None:
                ok = False
                detail = "order manager missing"
            else:
                try:
                    recent_orders: list[t.Any] = om.recent_orders(limit=10)
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    detail = f"error: {self._short_reason(str(exc))}"
                else:
                    pending = 0
                    for item in recent_orders:
                        status_text = ""
                        if isinstance(item, dict):
                            status_text = str(item.get("status", ""))
                        else:
                            status_text = str(item)
                        if "PEND" in status_text.upper():
                            pending += 1
                    ok = pending == 0
                    detail = f"recent={len(recent_orders)} pending={pending}" + (
                        " ok" if ok else " action_required"
                    )
        elif area == "broker":
            status = self._session_guard_status()
            if status is None:
                ok = False
                detail = "session guard unavailable"
            else:
                broker_ok = bool(status.get("broker_session_valid"))
                rate_ok = bool(status.get("rate_limits_ok"))
                risk_green = bool(status.get("risk_green"))
                market_open = status.get("market_open")
                reasons = status.get("reasons") or []
                ok = broker_ok and rate_ok and risk_green
                detail = f"broker={broker_ok} rate_limits={rate_ok} risk={risk_green}"
                if market_open is not None:
                    detail += f" market_open={market_open}"
                if reasons:
                    joined = "; ".join(str(reason) for reason in reasons)
                    detail += f" reasons={joined}"
        else:
            await self._reply(
                chat,
                ctx,
                "Unknown area. Use one of: data, risk, orders, broker.",
            )
            return

        icon = "✅" if ok else "❌"
        await self._reply(
            chat,
            ctx,
            f"{icon} {rb.esc(area)}: {rb.esc(detail)}",
            parse_mode=ParseMode.HTML,
        )

    @command_meta(
        "/confirm [ACTION]",
        "Confirm pending high-risk operations (enable_live, emergency_stop).",
    )
    async def cmd_confirm(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Execute or cancel pending confirmation-gated operations.

        Args:
            update: Telegram update invoking the confirmation.
            ctx: Handler context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_confirm",
            extra={"event": "telegram_cmd_confirm_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        try:
            args = ctx.args or []
            action_arg = args[0].lower() if args else ""
            if action_arg == "cancel":
                if self._pending_confirmation is None:
                    await self._reply(chat, ctx, "No pending confirmation to cancel.")
                    log.info(
                        "Condition met: telegram_cmd_confirm_cancel_empty",
                        extra={"event": "telegram_cmd_confirm_cancel_empty"},
                    )
                else:
                    self._clear_pending_confirmation()
                    await self._reply(chat, ctx, "Pending confirmation cleared.")
                    log.info(
                        "Condition met: telegram_cmd_confirm_cancelled",
                        extra={"event": "telegram_cmd_confirm_cancelled"},
                    )
                return
            pending = self._pending_confirmation
            if pending is None:
                await self._reply(chat, ctx, "No pending confirmation required.")
                log.info(
                    "Condition met: telegram_cmd_confirm_none",
                    extra={"event": "telegram_cmd_confirm_none"},
                )
                return
            if self._pending_confirmation_expired(pending):
                self._clear_pending_confirmation()
                await self._reply(
                    chat,
                    ctx,
                    "Confirmation window expired. Re-issue the original command.",
                )
                log.info(
                    "Condition met: telegram_cmd_confirm_expired",
                    extra={"event": "telegram_cmd_confirm_expired"},
                )
                return
            pending_action = str(pending.get("action", "")).lower()
            if action_arg and pending_action and action_arg != pending_action:
                await self._reply(
                    chat,
                    ctx,
                    (
                        f"Pending action is '{pending_action}'. "
                        f"Use /confirm {pending_action}."
                    ),
                )
                log.info(
                    "Condition met: telegram_cmd_confirm_mismatch",
                    extra={
                        "event": "telegram_cmd_confirm_mismatch",
                        "pending_action": pending_action,
                        "provided_action": action_arg,
                    },
                )
                return
            requester = int(pending.get("requested_by", 0))
            chat_id = int(getattr(chat, "id", 0))
            if requester and requester != chat_id:
                await self._reply(
                    chat,
                    ctx,
                    "Confirmation reserved for the requesting chat.",
                )
                log.warning(
                    "telegram_cmd_confirm_wrong_chat",
                    extra={
                        "event": "telegram_cmd_confirm_wrong_chat",
                        "expected": requester,
                        "actual": chat_id,
                    },
                )
                return
            if pending_action == "enable_live":
                if await self._confirm_enable_live(chat, ctx):
                    self._clear_pending_confirmation()
                return
            if pending_action == "emergency_stop":
                if not await self._guard_admin(update):
                    return
                if await self._execute_emergency_flatten(chat, ctx):
                    self._clear_pending_confirmation()
                return
            await self._reply(
                chat,
                ctx,
                "Unknown pending action. Use /confirm cancel to reset.",
            )
            log.info(
                "Condition met: telegram_cmd_confirm_unknown",
                extra={
                    "event": "telegram_cmd_confirm_unknown",
                    "action": pending_action,
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_confirm: %s",
                exc,
                extra={"event": "telegram_cmd_confirm_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Confirmation failed unexpectedly.")

    @command_meta(
        "/debug_on <logger> [LEVEL]",
        "Enable log tap for a logger at the desired level.",
    )
    async def cmd_debug_on(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        if not await self._guard_admin(update):
            return
        args = ctx.args or []
        if not args:
            await self._reply(chat, ctx, "Usage: /debug_on <logger> [LEVEL]")
            return
        logger_name = args[0].strip()
        level = args[1] if len(args) > 1 else "DEBUG"
        try:
            LOG_TAP.enable(logger_name, level)
        except Exception as exc:  # noqa: BLE001
            await self._reply(chat, ctx, f"Failed to enable tap: {exc}")
            return
        target = logger_name or "root"
        await self._reply(chat, ctx, f"Log tap enabled for {target} at {level}")

    @command_meta("/debug_off <logger>", "Disable log tap for a logger.")
    async def cmd_debug_off(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        if not await self._guard_admin(update):
            return
        args = ctx.args or []
        if not args:
            await self._reply(chat, ctx, "Usage: /debug_off <logger>")
            return
        logger_name = args[0].strip()
        try:
            LOG_TAP.disable(logger_name)
        except Exception as exc:  # noqa: BLE001
            await self._reply(chat, ctx, f"Failed to disable tap: {exc}")
            return
        target = logger_name or "root"
        await self._reply(chat, ctx, f"Log tap disabled for {target}")

    @command_meta(
        "/tail [logger] [N]",
        "Show the most recent log lines for a logger (default root).",
    )
    async def cmd_tail(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Return recent log lines with optional level filtering.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_tail",
            extra={"event": "telegram_cmd_tail_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_tail_guard_blocked",
                extra={"event": "telegram_cmd_tail_guard_blocked"},
            )
            return
        if not await self._guard_admin(update):
            log.info(
                "Condition met: telegram_cmd_tail_admin_denied",
                extra={"event": "telegram_cmd_tail_admin_denied"},
            )
            return
        try:
            args = list(ctx.args or [])
            level_aliases = {"DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"}
            level_filter: str | None = None
            target = ""
            count_arg: str | None = None
            remaining = [arg.strip() for arg in args if arg]
            if remaining:
                first = remaining[0].upper()
                if first in level_aliases:
                    level_filter = "WARNING" if first == "WARN" else first
                    remaining = remaining[1:]
            if remaining:
                candidate = remaining[0]
                if candidate and candidate.lstrip("-+").isdigit():
                    count_arg = candidate
                    remaining = remaining[1:]
                else:
                    target = candidate
                    remaining = remaining[1:]
                    if remaining:
                        extra = remaining[0]
                        if extra and extra.lstrip("-+").isdigit():
                            count_arg = extra
            max_lines = max(1, int(os.getenv("LOG_MAX_LINES_REPLY", "30")))
            try:
                requested = int(count_arg) if count_arg else max_lines
            except (TypeError, ValueError):
                requested = max_lines
            line_count = max(1, min(requested, max_lines))
            chat_id = int(getattr(chat, "id", self.deps.chat_id))
            now = time.time()
            last = self._tail_rate.get(chat_id, 0.0)
            if now - last < 3.0:
                wait = max(0.0, 3.0 - (now - last))
                await self._reply(
                    chat, ctx, f"Tail throttled. Try again in {wait:.1f}s."
                )
                return
            self._tail_rate[chat_id] = now
            entries = LOG_TAP.tail(target, line_count)
            if level_filter is not None:
                needle = f" {level_filter} "
                filtered = [line for line in entries if needle in line.upper()]
                entries = filtered[-line_count:]
            if not entries:
                label = target or "root"
                suffix = f" at level {level_filter}" if level_filter else ""
                await self._reply(
                    chat,
                    ctx,
                    f"No log lines recorded for {label}{suffix}.",
                )
                return
            text = "\n".join(entries)
            if len(text) > 3400:
                text = text[-3400:]
            await self._reply(chat, ctx, self._code(text), parse_mode=ParseMode.HTML)
            log.info(
                "Condition met: telegram_cmd_tail_dispatched",
                extra={
                    "event": "telegram_cmd_tail_dispatched",
                    "level": level_filter,
                    "target": target or "root",
                    "lines": len(entries),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "Failure in cmd_tail: %s",
                exc,
                extra={"event": "telegram_cmd_tail_error"},
                exc_info=exc,
            )
            with suppress(Exception):
                await self._reply(
                    chat,
                    ctx,
                    "Unable to fetch log tail. Inspect logs for details.",
                )

    @command_meta(
        "/paper [section] [shadow|live]",
        "Inspect or toggle paper trading state per subsystem.",
    )
    async def cmd_paper(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        getters = dict(self.deps.paper_mode_getters or {})
        if not getters:
            await self._reply(chat, ctx, "Paper controls not configured.")
            return
        rb = self._response_builder
        args = ctx.args or []
        if not args:
            rows: list[str] = []
            for section in sorted(getters):
                state = False
                with suppress(Exception):
                    state = bool(getters[section]())
                emoji = EMOJI["shadow"] if state else EMOJI["live"]
                mode_label = "shadow" if state else "live"
                rows.append(
                    (
                        f"{emoji} <b>{rb.esc(section.upper())}</b>: "
                        f"<code>{rb.esc(mode_label)}</code>"
                    )
                )
            text = rb.section_block("Paper Modes", rows)
            await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)
            return

        section = args[0].strip().lower()
        if section not in getters:
            available = ", ".join(sorted(getters))
            await self._reply(
                chat,
                ctx,
                f"Unknown section {section!r}. Available: {available}",
            )
            return

        if len(args) == 1:
            state = False
            with suppress(Exception):
                state = bool(getters[section]())
            await self._reply(
                chat,
                ctx,
                f"{section}={'shadow' if state else 'live'}",
            )
            return

        desired_raw = args[1].strip().lower()
        truthy = {"shadow", "on", "paper", "true", "1"}
        falsy = {"live", "off", "false", "0"}
        if desired_raw not in truthy | falsy:
            await self._reply(
                chat,
                ctx,
                "Usage: /paper <section> <shadow|live>",
            )
            return

        desired = desired_raw in truthy
        setters = self.deps.paper_mode_setters or {}
        setter = setters.get(section)
        if setter is None:
            await self._reply(chat, ctx, "Section not toggleable in this build.")
            return

        success = False
        with suppress(Exception):
            success = bool(setter(desired))

        state = False
        with suppress(Exception):
            state = bool(getters[section]())

        message = f"{section}={'shadow' if state else 'live'}"
        if not success:
            message += " (blocked)"
        await self._reply(chat, ctx, message)

    async def _toggle_orders_paper(
        self,
        chat: t.Any,
        ctx: ContextTypes.DEFAULT_TYPE,
        enable: bool,
    ) -> None:
        getters = self.deps.paper_mode_getters or {}
        setters = self.deps.paper_mode_setters or {}
        setter = setters.get("orders")
        getter = getters.get("orders")
        if setter is None:
            await self._reply(chat, ctx, "Orders paper toggle not available.")
            return
        success = False
        with suppress(Exception):
            success = bool(setter(enable))
        state = enable
        if getter is not None:
            with suppress(Exception):
                state = bool(getter())
        mode = "shadow" if state else "live"
        msg = f"orders={mode}"
        if not success:
            msg += " (blocked)"
        await self._reply(chat, ctx, msg)

    @command_meta("/paper_on", "Enable paper routing for orders.")
    async def cmd_paper_on(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        await self._toggle_orders_paper(chat, ctx, True)

    @command_meta("/paper_off", "Disable paper routing for orders.")
    async def cmd_paper_off(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        await self._toggle_orders_paper(chat, ctx, False)

    async def cmd_shadow(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        get_sm = self.deps.get_shadow_mode
        set_sm = self.deps.set_shadow_mode
        if get_sm is None or set_sm is None:
            await self._reply(chat, ctx, "Shadow mode not toggleable in this build.")
            return
        result = True
        if ctx.args:
            arg = ctx.args[0].strip().lower()
            if arg in ("on", "true", "1"):
                result = set_sm(True)
            elif arg in ("off", "false", "0"):
                result = set_sm(False)
        status = self._session_guard_status()
        state = bool(get_sm())
        override = bool(status.get("override_out_of_hours") if status else False)
        response = f"shadow={state}"
        if override and not state:
            response += " override=true"
        if not result and status is not None:
            reasons = status.get("reasons") or []
            if reasons:
                response += f" blocked: {', '.join(reasons)}"
        await self._reply(chat, ctx, response)

    async def _toggle_strategies(self, enable: bool) -> tuple[str, dict[str, t.Any]]:
        snapshot: dict[str, t.Any] = {}
        runner = getattr(self.deps, "strategy_runner", None)
        if runner is not None:
            action = "not available"
            with suppress(Exception):
                status = runner.get_status()
                if isinstance(status, dict):
                    snapshot = dict(status)
            with suppress(Exception):
                if enable and hasattr(runner, "resume_trading"):
                    runner.resume_trading()
                    action = "resumed"
                elif not enable and hasattr(runner, "pause_trading"):
                    runner.pause_trading()
                    action = "paused"
            with suppress(Exception):
                status = runner.get_status()
                if isinstance(status, dict):
                    snapshot = dict(status)
            if action != "not available":
                return action, snapshot

        manager = getattr(self.deps, "strategy_manager", None)
        if manager is not None:
            action = "not available"
            with suppress(Exception):
                if enable and hasattr(manager, "start"):
                    manager.start()
                    action = "started"
                elif not enable and hasattr(manager, "stop"):
                    manager.stop()
                    action = "stopped"
            return action, snapshot

        return "not available", snapshot

    async def cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        result, snapshot = await self._toggle_strategies(True)
        if result == "not available":
            await self._reply(chat, ctx, "start: runner unavailable")
            return
        get_sm = self.deps.get_shadow_mode
        mode = "shadow" if get_sm and get_sm() else "live"
        active = []
        if isinstance(snapshot.get("active_symbols"), list):
            active = [str(symbol) for symbol in snapshot.get("active_symbols", [])]
        guard_status = self._session_guard_status() or {}
        override = guard_status.get("override_out_of_hours") is True
        paused_state = snapshot.get("trading_paused")
        parts = [f"mode={mode}"]
        parts.append(f"symbols={', '.join(active) if active else 'none'}")
        if paused_state is not None:
            parts.append(f"paused={bool(paused_state)}")
        if override:
            parts.append("override=true")
        await self._reply(chat, ctx, f"trading {result}: " + " ".join(parts))

    async def cmd_stop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        result, snapshot = await self._toggle_strategies(False)
        if result == "not available":
            await self._reply(chat, ctx, "stop: runner unavailable")
            return
        get_sm = self.deps.get_shadow_mode
        mode = "shadow" if get_sm and get_sm() else "live"
        active = []
        if isinstance(snapshot.get("active_symbols"), list):
            active = [str(symbol) for symbol in snapshot.get("active_symbols", [])]
        guard_status = self._session_guard_status() or {}
        override = guard_status.get("override_out_of_hours") is True
        parts = [f"mode={mode}"]
        parts.append(f"symbols={', '.join(active) if active else 'none'}")
        trading_paused = snapshot.get("trading_paused")
        if trading_paused is not None:
            parts.append(f"paused={bool(trading_paused)}")
        if override:
            parts.append("override=true")
        await self._reply(chat, ctx, f"trading {result}: " + " ".join(parts))

    @command_meta("/pause", "Pause live trading until resumed.")
    async def cmd_pause(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        log.debug(
            "Entered cmd_pause",
            extra={"event": "telegram_cmd_pause_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        try:
            switch = trading_switch()
            switch.pause()
            snapshot = switch.snapshot()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_pause: %s",
                exc,
                extra={"event": "telegram_cmd_pause_error"},
            )
            await self._reply(chat, ctx, "⚠️ Unable to pause trading.")
            return
        log.info(
            "Condition met: trading paused via Telegram",
            extra={
                "event": "telegram_cmd_pause_applied",
                "remaining": round(snapshot.remaining_seconds, 3),
            },
        )
        await self._reply(chat, ctx, "⏸️ Trading paused (manual).")

    @command_meta("/resume", "Resume trading immediately.")
    async def cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        log.debug(
            "Entered cmd_resume",
            extra={"event": "telegram_cmd_resume_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        try:
            switch = trading_switch()
            switch.resume()
            snapshot = switch.snapshot()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_resume: %s",
                exc,
                extra={"event": "telegram_cmd_resume_error"},
            )
            await self._reply(chat, ctx, "⚠️ Unable to resume trading.")
            return
        log.info(
            "Condition met: trading resumed via Telegram",
            extra={
                "event": "telegram_cmd_resume_applied",
                "remaining": round(snapshot.remaining_seconds, 3),
            },
        )
        await self._reply(chat, ctx, "▶️ Trading resumed.")

    @command_meta("/cooldown <seconds>", "Pause trading for N seconds.")
    async def cmd_cooldown(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        log.debug(
            "Entered cmd_cooldown",
            extra={"event": "telegram_cmd_cooldown_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        args = list(getattr(ctx, "args", []) or [])
        if not args:
            await self._reply(chat, ctx, "Usage: /cooldown <seconds>")
            return
        try:
            seconds = max(float(args[0]), 0.0)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_cooldown: %s",
                exc,
                extra={"event": "telegram_cmd_cooldown_parse_error"},
            )
            await self._reply(chat, ctx, "Usage: /cooldown <seconds>")
            return
        seconds = max(seconds, 1.0)
        try:
            switch = trading_switch()
            switch.cooldown(seconds)
            snapshot = switch.snapshot()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_cooldown: %s",
                exc,
                extra={"event": "telegram_cmd_cooldown_error"},
            )
            await self._reply(chat, ctx, "⚠️ Unable to apply cooldown.")
            return
        log.info(
            "Condition met: trading cooldown via Telegram",
            extra={
                "event": "telegram_cmd_cooldown_applied",
                "seconds": seconds,
                "remaining": round(snapshot.remaining_seconds, 3),
            },
        )
        await self._reply(
            chat,
            ctx,
            f"⏲️ Trading paused for {int(round(seconds))}s.",
        )

    async def cmd_shadow_on(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        toggle = self.deps.set_shadow_mode
        if toggle is None:
            await self._reply(chat, ctx, "shadow=true (forced)")
            return
        outcome = False
        with suppress(Exception):
            outcome = bool(toggle(True))
        await self._reply(
            chat, ctx, "shadow=true" if outcome else "shadow=true (forced)"
        )

    async def cmd_shadow_off(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Request confirmation before disabling shadow (live) mode.

        Args:
            update: Telegram update triggering the command.
            ctx: Handler context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_shadow_off",
            extra={"event": "telegram_cmd_shadow_off_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        toggle = self.deps.set_shadow_mode
        if toggle is None:
            await self._reply(chat, ctx, "shadow=false blocked: toggle unavailable")
            log.info(
                "Condition met: telegram_cmd_shadow_off_no_toggle",
                extra={"event": "telegram_cmd_shadow_off_no_toggle"},
            )
            return
        guard_status = self._session_guard_status() or {}
        guard_keys = ("session_valid", "rate_limits_ok", "risk_green")
        session_ok = all(bool(guard_status.get(key)) for key in guard_keys)
        market_open = bool(guard_status.get("market_open", True))
        override_active = bool(guard_status.get("override_out_of_hours"))
        reasons = [str(reason) for reason in guard_status.get("reasons", [])]
        if guard_status and (
            not session_ok or (not market_open and not override_active)
        ):
            message = "shadow=false blocked"
            if reasons:
                message += f": {', '.join(reasons)}"
            await self._reply(chat, ctx, message)
            log.info(
                "Condition met: telegram_cmd_shadow_off_blocked",
                extra={"event": "telegram_cmd_shadow_off_blocked"},
            )
            return
        chat_id = int(getattr(chat, "id", 0))
        pending = self._pending_confirmation
        if pending and self._pending_confirmation_expired(pending):
            self._clear_pending_confirmation()
            pending = None
        if pending and pending.get("action") == "enable_live":
            await self._reply(
                chat,
                ctx,
                (
                    "Live enable already pending. "
                    "Use /confirm enable_live or /confirm cancel."
                ),
            )
            log.info(
                "Condition met: telegram_cmd_shadow_off_pending",
                extra={"event": "telegram_cmd_shadow_off_pending"},
            )
            return
        self._set_pending_confirmation("enable_live", chat_id)
        hint_tokens: list[str] = []
        if override_active and not market_open:
            hint_tokens.append("override active")
        if reasons:
            hint_tokens.append("reasons: " + "; ".join(reasons))
        hint_suffix = f" ({'; '.join(hint_tokens)})" if hint_tokens else ""
        await self._reply(
            chat,
            ctx,
            (
                "⚠️ Confirm enabling LIVE trading with /confirm enable_live within "
                f"{int(self._CONFIRMATION_WINDOW.total_seconds())}s.{hint_suffix}"
            ),
        )
        log.info(
            "Condition met: telegram_cmd_shadow_off_requested",
            extra={"event": "telegram_cmd_shadow_off_requested"},
        )

    @command_meta(
        "/test_flow <SYMBOL>",
        "Simulate a paper trade and return hub snapshot for verification.",
    )
    async def cmd_test_flow(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        args = ctx.args or []
        if not args:
            await self._reply(chat, ctx, "Usage: /test_flow SYMBOL")
            return
        hub = self.deps.data_hub
        safe_manager = self.deps.safe_order_manager
        if hub is None or safe_manager is None:
            await self._reply(chat, ctx, "Paper environment unavailable.")
            return
        normalized = DataHub.normalize(args[0])
        if not normalized:
            await self._reply(chat, ctx, "Invalid symbol")
            return
        getters = self.deps.paper_mode_getters or {}
        orders_getter = getters.get("orders")
        if orders_getter is not None:
            with suppress(Exception):
                if not bool(orders_getter()):
                    await self._reply(chat, ctx, "Orders are not in paper mode.")
                    return
        resolver = self.deps.instrument_resolver
        lot_size = 75
        if resolver is not None and hasattr(resolver, "lot_size_for_symbol"):
            with suppress(Exception):
                candidate = int(resolver.lot_size_for_symbol(normalized))
                if candidate > 0:
                    lot_size = candidate
        loop = asyncio.get_running_loop()
        try:
            order_id = await loop.run_in_executor(
                None,
                lambda: safe_manager.place_order(  # type: ignore[arg-type]
                    symbol=normalized,
                    side="BUY",
                    quantity=lot_size,
                    tag="test_flow",
                ),
            )
        except Exception as exc:  # noqa: BLE001
            await self._reply(chat, ctx, f"Order failed: {exc}")
            return
        base_quote = hub.get_quote(normalized, allow_pull=True)
        base_price = 0.0
        if isinstance(base_quote, t.Mapping):
            with suppress(Exception):
                raw_price = (
                    base_quote.get("ltp")
                    or base_quote.get("last_price")
                    or base_quote.get("close")
                )
                if raw_price is not None:
                    base_price = float(raw_price)
        tick = {
            "symbol": normalized,
            "ltp": base_price + 50.0,
            "source": "telegram_test_flow",
            "timestamp": time.time(),
        }
        with suppress(Exception):
            hub.ingest_tick(tick)
        snapshot = hub.as_dict()
        payload = {
            "order_id": order_id,
            "orders": snapshot.get("orders", [])[-5:],
            "positions": snapshot.get("positions", []),
        }
        await self._reply(
            chat,
            ctx,
            self._code(_pp(payload)),
            parse_mode=ParseMode.HTML,
        )

    @command_meta("/pnl", "Realized/unrealized PnL with drawdown context.")
    async def cmd_pnl(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Reply with current profit and loss telemetry for the session.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered TelegramBot.cmd_pnl",
            extra={"event": "telegram_cmd_pnl_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_pnl_guard_blocked",
                extra={"event": "telegram_cmd_pnl_guard_blocked"},
            )
            return
        pm = getattr(self.deps, "position_manager", None)
        if pm is None:
            log.info(
                "Condition met: telegram_cmd_pnl_no_pm",
                extra={"event": "telegram_cmd_pnl_no_pm"},
            )
            await self._reply(chat, ctx, "No position manager")
            return
        rm = getattr(self.deps, "risk_manager", None)
        rb = self._response_builder
        payload: dict[str, t.Any] = {}
        try:
            realized = self._coerce_float_value(
                getattr(pm, "get_realized_pnl", lambda: None), field="realized_pnl"
            )
            unrealized = self._coerce_float_value(
                getattr(pm, "get_unrealized_pnl", lambda: None), field="unrealized_pnl"
            )
            net = self._coerce_float_value(
                getattr(pm, "get_net_pnl", lambda: None), field="net_pnl"
            )
            tracker = self._pnl_tracker
            previous_net = tracker.get("last")
            trend_symbol, delta_value = self._trend_delta(net, previous_net)
            if net is not None:
                current_peak = tracker.get("peak", net)
                current_trough = tracker.get("trough", net)
                tracker["peak"] = max(current_peak, net)
                tracker["trough"] = min(current_trough, net)
                tracker["last"] = net
            peak_attr = None
            trough_attr = None
            for candidate in ("get_pnl_peak", "pnl_peak"):
                peak_attr = getattr(pm, candidate, None)
                if peak_attr is not None:
                    break
            for candidate in ("get_pnl_trough", "pnl_trough"):
                trough_attr = getattr(pm, candidate, None)
                if trough_attr is not None:
                    break
            peak_value = (
                self._coerce_float_value(peak_attr, field="pnl_peak")
                if peak_attr
                else tracker.get("peak")
            )
            trough_value = (
                self._coerce_float_value(trough_attr, field="pnl_trough")
                if trough_attr
                else tracker.get("trough")
            )
            drawdown_value: float | None = None
            drawdown_pct: float | None = None
            if rm is not None:
                risk_state = getattr(rm, "risk_state", None) or getattr(
                    rm, "_risk_state", None
                )
                peak_equity = self._coerce_float_value(
                    getattr(risk_state, "_peak_equity", None), field="risk_peak_equity"
                )
                current_equity = self._coerce_float_value(
                    getattr(risk_state, "_current_equity", None),
                    field="risk_current_equity",
                )
                if peak_equity and current_equity is not None:
                    drawdown_value = current_equity - peak_equity
                    if peak_equity != 0:
                        drawdown_pct = (drawdown_value / peak_equity) * 100.0
            delta_display = self._format_currency(delta_value, signed=True)
            realized_text = rb.esc(self._format_currency(realized, signed=True))
            unrealized_text = rb.esc(self._format_currency(unrealized, signed=True))
            lines = [
                (
                    f"{rb.section('PnL')} "
                    f"realized=<b>{realized_text}</b> "
                    f"unrealized=<b>{unrealized_text}</b>"
                ),
            ]
            net_line = (
                f"net=<b>{rb.esc(self._format_currency(net, signed=True))}</b> "
                f"trend={rb.esc(trend_symbol)} {rb.esc(delta_display)}"
            )
            if peak_value is not None or trough_value is not None:
                peak_text = self._format_currency(peak_value, signed=True)
                trough_text = self._format_currency(trough_value, signed=True)
                net_line += f" | peak={rb.esc(peak_text)} trough={rb.esc(trough_text)}"
            lines.append(net_line)
            if drawdown_value is not None:
                dd_text = self._format_currency(drawdown_value, signed=True)
                pct_text = self._format_percent(
                    drawdown_pct / 100.0 if drawdown_pct is not None else None
                )
                lines.append(f"drawdown=<b>{rb.esc(dd_text)}</b> ({rb.esc(pct_text)})")
            message = rb.bullets(lines)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_pnl",
                exc,
                extra={"event": "telegram_cmd_pnl_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unable to fetch PnL telemetry.")
            return
        payload["realized"] = realized
        payload["unrealized"] = unrealized
        payload["net"] = net
        if peak_value is not None:
            payload["peak"] = peak_value
        if trough_value is not None:
            payload["trough"] = trough_value
        if drawdown_value is not None:
            payload["drawdown_value"] = drawdown_value
        if drawdown_pct is not None:
            payload["drawdown_pct"] = drawdown_pct
        detail_block = self._code(_pp(payload))
        try:
            await self._reply(
                chat,
                ctx,
                f"{message}{rb.br()}{detail_block}",
                parse_mode=ParseMode.HTML,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_pnl reply: %s",
                exc,
                extra={"event": "telegram_cmd_pnl_reply_error"},
                exc_info=exc,
            )
        else:
            log.info(
                "Condition met: telegram_cmd_pnl_dispatched",
                extra={
                    "event": "telegram_cmd_pnl_dispatched",
                    "net": net,
                    "trend": trend_symbol,
                    "delta": delta_value,
                },
            )

    @command_meta("/regime", "Current regime classification and adjustments.")
    async def cmd_regime(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Reply with the latest market regime snapshots.

        Args:
            update: Telegram update payload from PTB.
            ctx: Context carrying bot/application state.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered TelegramBot.cmd_regime",
            extra={"event": "telegram_cmd_regime"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        manager = getattr(self.deps, "regime_manager", None)
        detector = getattr(self.deps, "market_regime", None)
        if manager is None and detector is None:
            log.info(
                "Condition met: telegram_regime_detector_missing",
                extra={"event": "telegram_regime_detector_missing"},
            )
            await self._reply(chat, ctx, "Market regime services unavailable.")
            return
        try:
            report_lines, latest_snapshot = self._build_regime_report(
                manager=manager,
                detector=detector,
                limit=5,
            )
        except ValueError:
            log.info(
                "Condition met: telegram_regime_no_data",
                extra={"event": "telegram_regime_no_data"},
            )
            await self._reply(chat, ctx, "No regime data yet.")
            return
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error(
                "Failure in cmd_regime: %s",
                exc,
                extra={"event": "telegram_cmd_regime_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unable to fetch regime information.")
            return
        gate = getattr(self.deps, "regime_gate", None)
        if gate is not None:
            allow_fn = getattr(gate, "allow_signal", None)
            if callable(allow_fn):
                now = datetime.now(timezone.utc)
                trend_ok = bool(allow_fn(now, bias="trend"))
                range_ok = bool(allow_fn(now, bias="mean_revert"))
                report_lines.append(
                    (
                        f"Gating — trend: {'✅' if trend_ok else '🚫'}, "
                        f"mean-revert: {'✅' if range_ok else '🚫'}"
                    )
                )
        highlights = self._diag_highlights(plain=True)
        if highlights:
            report_lines.append("")
            report_lines.append("Diagnostics:")
            report_lines.extend(highlights)
        message = "\n".join(report_lines)
        try:
            await self._reply(chat, ctx, message)
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error(
                "Failure in cmd_regime reply: %s",
                exc,
                extra={"event": "telegram_cmd_regime_reply_error"},
                exc_info=exc,
            )
        else:
            if latest_snapshot is not None:
                log.info(
                    "Condition met: telegram_regime_sent",
                    extra={
                        "event": "telegram_regime_sent",
                        "symbol": latest_snapshot.symbol,
                        "regime": latest_snapshot.regime,
                        "confidence": latest_snapshot.confidence,
                    },
                )

    @command_meta(
        "/replay DAY", "Run minute replay for YYYYMMDD from configured directory."
    )
    async def cmd_replay(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        harness = getattr(self.deps, "replay_harness", None)
        settings = getattr(self.deps, "replay_settings", None)
        if harness is None or settings is None:
            await self._reply(chat, ctx, "Replay harness not configured.")
            return
        if not ctx.args:
            await self._reply(chat, ctx, "Usage: /replay YYYYMMDD")
            return
        day = ctx.args[0]
        try:
            result = await asyncio.to_thread(harness.run_day, settings.data_dir, day)
        except FileNotFoundError as exc:
            await self._reply(chat, ctx, f"Replay data missing: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("Replay run failed", exc_info=exc)
            await self._reply(chat, ctx, f"Replay failed: {exc}")
            return
            await self._reply(chat, ctx, result.format_summary())

    @command_meta("/heatmap", "Visualise per-strategy PnL, win rate, and allocation.")
    async def cmd_heatmap(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Reply with a compact strategy performance heatmap.

        Args:
            update: Telegram update payload invoking the command.
            ctx: Callback context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_heatmap",
            extra={"event": "telegram_cmd_heatmap_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: telegram_cmd_heatmap_guard_blocked",
                extra={"event": "telegram_cmd_heatmap_guard_blocked"},
            )
            return
        manager = getattr(self.deps, "strategy_manager", None)
        if manager is None:
            await self._reply(chat, ctx, "Strategy manager unavailable.")
            log.info(
                "Condition met: telegram_cmd_heatmap_no_manager",
                extra={"event": "telegram_cmd_heatmap_no_manager"},
            )
            return
        scores_getter = getattr(manager, "get_strategy_scores", None)
        if not callable(scores_getter):
            await self._reply(
                chat,
                ctx,
                "Strategy metrics not available in current mode.",
            )
            log.info(
                "Condition met: telegram_cmd_heatmap_unsupported",
                extra={"event": "telegram_cmd_heatmap_unsupported"},
            )
            return
        rb = self._response_builder
        try:
            scores = scores_getter()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_heatmap scores: %s",
                exc,
                extra={"event": "telegram_cmd_heatmap_scores_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unable to fetch strategy metrics.")
            return
        if not scores:
            await self._reply(chat, ctx, "No strategy metrics available yet.")
            log.info(
                "Condition met: telegram_cmd_heatmap_no_scores",
                extra={"event": "telegram_cmd_heatmap_no_scores"},
            )
            return
        try:
            performance_snapshot = manager.get_performance_snapshot()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_heatmap snapshot: %s",
                exc,
                extra={"event": "telegram_cmd_heatmap_snapshot_error"},
                exc_info=exc,
            )
            performance_snapshot = {}
        entries: list[dict[str, t.Any]] = []
        max_abs_pnl = 0.0
        seen: set[str] = set()
        for score in scores:
            name = str(getattr(score, "strategy", "unknown"))
            seen.add(name)
            aggregate_data = {}
            perf_entry = (
                performance_snapshot.get(name)
                if isinstance(performance_snapshot, dict)
                else {}
            )
            if isinstance(perf_entry, dict):
                aggregate_data = perf_entry.get("aggregate", {})
            pnl_value = self._coerce_float_value(
                aggregate_data.get("pnl"), field=f"{name}.pnl"
            )
            if pnl_value is None:
                pnl_value = self._coerce_float_value(
                    getattr(score, "pnl", None), field=f"{name}.score_pnl"
                )
            win_rate = self._coerce_float_value(
                aggregate_data.get("win_rate"), field=f"{name}.win_rate"
            )
            trades_raw = (
                aggregate_data.get("trades")
                if isinstance(aggregate_data, dict)
                else None
            )
            try:
                trades_count = int(trades_raw) if trades_raw is not None else 0
            except Exception:  # noqa: BLE001
                trades_count = 0
            allocation_value = self._coerce_float_value(
                getattr(score, "allocation", None), field=f"{name}.allocation"
            )
            previous_pnl = self._strategy_pnl_cache.get(name)
            trend_symbol, delta_value = self._trend_delta(
                pnl_value, previous_pnl, threshold=0.5
            )
            entries.append(
                {
                    "name": name,
                    "pnl": pnl_value,
                    "win_rate": win_rate,
                    "trades": trades_count,
                    "allocation": allocation_value,
                    "trend_symbol": trend_symbol,
                    "trend_delta": delta_value,
                }
            )
            if pnl_value is not None:
                max_abs_pnl = max(max_abs_pnl, abs(pnl_value))
                self._strategy_pnl_cache[name] = pnl_value
        for stale in set(self._strategy_pnl_cache) - seen:
            self._strategy_pnl_cache.pop(stale, None)
        header = "Strategy     |      P&L | Win%  | Trades | Alloc | Trend       | Heat"
        separator = "-" * len(header)
        rows: list[str] = []
        for entry in entries:
            pnl_display = self._format_currency(entry["pnl"], signed=True)
            win_display = self._format_percent(entry["win_rate"], signed=False)
            alloc_display = self._format_percent(entry["allocation"], signed=False)
            trades_display = str(entry["trades"]).rjust(6)
            trend_symbol = entry["trend_symbol"]
            trend_delta_display = self._format_currency(
                entry["trend_delta"], signed=True
            )
            heat = self._heat_bar(entry["pnl"], max_abs_pnl)
            row = (
                f"{entry['name']:<12} | {pnl_display:>8} | {win_display:>5} | "
                f"{trades_display} | {alloc_display:>5} | {trend_symbol:<2} "
                f"{trend_delta_display:>9} | {heat}"
            )
            rows.append(row)
        table_text = "\n".join([header, separator, *rows])
        message = (
            f"{rb.section('Strategy Heatmap')}{rb.br()}"
            f"{rb.mono(table_text, max_lines=40)}"
        )
        try:
            await self._reply(chat, ctx, message, parse_mode=ParseMode.HTML)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_heatmap reply: %s",
                exc,
                extra={"event": "telegram_cmd_heatmap_reply_error"},
                exc_info=exc,
            )
        else:
            log.info(
                "Condition met: telegram_cmd_heatmap_dispatched",
                extra={
                    "event": "telegram_cmd_heatmap_dispatched",
                    "strategies": len(entries),
                    "max_abs_pnl": max_abs_pnl,
                },
            )

    @command_meta("/kpi", "Summarize session KPIs captured by the tracker.")
    async def cmd_kpi(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Reply with KPI tracker output or a fallback snapshot.

        Args:
            update: Telegram update triggering the command.
            ctx: Handler context provided by python-telegram-bot.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_kpi",
            extra={"event": "telegram_cmd_kpi_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        try:
            tracker = getattr(self.deps, "kpi_tracker", None)
            if tracker is None:
                fallback = self._build_basic_kpi()
                await self._reply(chat, ctx, fallback)
                log.info(
                    "Condition met: telegram_cmd_kpi_fallback",
                    extra={"event": "telegram_cmd_kpi_fallback"},
                )
                return
            if (
                ctx.args
                and ctx.args[0].lower() == "reset"
                and hasattr(tracker, "reset")
            ):
                try:
                    tracker.reset()
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure in cmd_kpi reset: %s",
                        exc,
                        extra={"event": "telegram_cmd_kpi_reset_error"},
                        exc_info=exc,
                    )
                    await self._reply(
                        chat,
                        ctx,
                        f"KPI reset failed: {self._short_reason(str(exc))}",
                    )
                    return
                await self._reply(chat, ctx, "KPI tracker reset.")
                log.info(
                    "Condition met: telegram_cmd_kpi_reset",
                    extra={"event": "telegram_cmd_kpi_reset"},
                )
                return
            try:
                report_text = (
                    tracker.format_report()
                    if hasattr(tracker, "format_report")
                    else str(tracker)
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_kpi format: %s",
                    exc,
                    extra={"event": "telegram_cmd_kpi_format_error"},
                    exc_info=exc,
                )
                await self._reply(
                    chat,
                    ctx,
                    f"KPI render failed: {self._short_reason(str(exc))}",
                )
                return
            await self._reply(chat, ctx, str(report_text))
            log.info(
                "Condition met: telegram_cmd_kpi_success",
                extra={"event": "telegram_cmd_kpi_success"},
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in cmd_kpi: %s",
                exc,
                extra={"event": "telegram_cmd_kpi_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Unexpected KPI error.")

    @command_meta(
        "/errors[?page=1&limit=50]",
        "Recent error log lines with pagination.",
    )
    async def cmd_errors(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        rb = self._response_builder
        env_cap, default_limit = self._log_limits()
        page, limit = self._pagination_args(update, ctx, default_limit=default_limit)
        window = min(max(env_cap * 10, 200), 2000)
        entries = list(reversed(RING.tail(window)))
        total = len(entries)
        if total == 0:
            text = self._compose(
                [
                    [
                        f"{rb.section('Errors')} {EMOJI['warn']}",
                        "<i>No recent errors or log lines.</i>",
                    ]
                ]
            )
            await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)
            return
        total_pages = max(1, (total + limit - 1) // limit)
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * limit
        chunk = entries[start_idx : start_idx + limit]
        head = f"{rb.section('Errors')} {EMOJI['warn']} ({total} lines)"
        body = (
            rb.mono("\n".join(chunk), max_lines=limit)
            if chunk
            else "<i>No entries for this page.</i>"
        )
        footer = rb.footer(page, total_pages)
        text = self._compose([[head, body]])
        if footer:
            text = text + rb.sep() + footer
        if total_pages > 1 and page < total_pages:
            text = (
                text
                + rb.br()
                + f"<i>Next page: /errors?page={page + 1}&limit={limit}</i>"
            )
        await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)

    @command_meta(
        "/why",
        "Explain risk breaker state and the last recorded error.",
    )
    async def cmd_why(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        rb = self._response_builder
        snap = self._coerce_risk_snapshot()
        info = self._last_error()
        if info is not None:
            category, message, when = info
            hint = self._hint_for(category, message)
            err_rows = [
                ("category", category or "n/a"),
                ("message", message or "n/a"),
                ("hint", hint or "—"),
                ("when", when or "n/a"),
            ]
            last_error_html = (
                f"{rb.section('Last Error')} {EMOJI['warn']}<br>{rb.grid(err_rows)}"
            )
        else:
            last_error_html = f"{rb.section('Last Error')}<br><i>No recent errors.</i>"

        risk_html = ""
        if snap:
            losses_val = snap.get("losses_in_row")
            cooldown_val = snap.get("cooldown_remaining")
            cooldown_text = (
                f"{float(cooldown_val):.0f}s"
                if isinstance(cooldown_val, (int, float))
                else "n/a"
            )
            risk_rows = [
                ("breaker", "ON" if snap.get("breaker_tripped") else "OFF"),
                ("losses", str(losses_val) if losses_val is not None else "n/a"),
                ("cooldown", cooldown_text),
                (
                    "reason",
                    self._short_reason(
                        str(
                            snap.get("breaker_reason")
                            or snap.get("last_rejection")
                            or ""
                        )
                    )
                    or "—",
                ),
            ]
            risk_html = (
                f"{EMOJI['risk']} {rb.section('Risk Snapshot')}<br>{rb.grid(risk_rows)}"
            )

        text = self._compose([[last_error_html]] + ([[risk_html]] if risk_html else []))

        await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)
        return

    @command_meta(
        "/issues[?page=1&limit=30]",
        "Group recent warnings and errors by category with pagination.",
    )
    async def cmd_issues(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        rb = self._response_builder
        env_cap, default_limit = self._log_limits()
        page, limit = self._pagination_args(update, ctx, default_limit=default_limit)
        limit = max(1, min(limit, 50))
        entries = RING.tail(min(2000, env_cap * 20))
        buckets: dict[str, dict[str, t.Any]] = {}
        for line in entries:
            ts, level, message = self._parse_ring_line(line)
            if level not in {"ERROR", "WARNING", "CRITICAL"}:
                continue
            category = self._categorize(line)
            bucket = buckets.setdefault(
                category,
                {"count": 0, "latest": None, "messages": [], "seen": set()},
            )
            bucket["count"] += 1
            iso = self._iso_from_time(ts)
            if iso:
                bucket["latest"] = iso
            short = self._short_reason(message)
            if short and short not in bucket["seen"]:
                bucket["messages"].append(short)
                bucket["seen"].add(short)
                if len(bucket["messages"]) > 3:
                    bucket["messages"] = bucket["messages"][:3]
        if not buckets:
            text = self._compose(
                [
                    [
                        f"{rb.section('Issues')} {EMOJI['warn']}",
                        "<i>No recent warnings or errors.</i>",
                    ]
                ]
            )
            await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)
            return
        category_order = [
            "webhook",
            "telegram",
            "ws",
            "mdm",
            "broker",
            "risk",
            "orders",
            "strategy",
            "resolver",
            "session",
            "other",
        ]
        ordered: list[tuple[str, dict[str, t.Any]]] = []
        for category in category_order:
            data = buckets.pop(category, None)
            if data:
                ordered.append((category, data))
        ordered.extend(sorted(buckets.items()))
        total = len(ordered)
        total_pages = max(1, (total + limit - 1) // limit)
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * limit
        slice_rows = ordered[start_idx : start_idx + limit]
        header = f"{rb.section('Issues')} {EMOJI['warn']} ({total} categories)"
        formatted: list[str] = []
        for category, data in slice_rows:
            latest = data.get("latest") or "n/a"
            count = data.get("count", 0)
            samples = data.get("messages", []) or []
            entry = (
                f"{EMOJI['warn']} <b>{rb.esc(category)}</b>: "
                f"{rb.esc(str(count))} <i>(latest {rb.esc(latest)})</i>"
            )
            if samples:
                entry += "<br><code>" + rb.esc(" | ".join(samples)) + "</code>"
            formatted.append(entry)
        footer = rb.footer(page, total_pages)
        body = rb.bullets(formatted) if formatted else "<i>No issues.</i>"
        text = self._compose([[header, body]])
        if footer:
            text = text + rb.sep() + footer
        if total_pages > 1 and page < total_pages:
            text = (
                text
                + rb.br()
                + f"<i>Next page: /issues?page={page + 1}&limit={limit}</i>"
            )
        await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)

    @command_meta(
        "/flow",
        "Visualise data, strategy, and Telegram pipeline health.",
    )
    async def cmd_flow(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        log.debug(
            "Entered TelegramController.cmd_flow",
            extra={"event": "telegram_cmd_flow_start"},
        )
        rb = self._response_builder
        ws = self.deps.websocket_manager
        streamer = getattr(self.deps, "streamer", None)
        mdm = self.deps.market_data_manager
        indicator = None
        strategy_manager = self.deps.strategy_manager
        if strategy_manager is not None:
            with suppress(Exception):
                indicator = getattr(strategy_manager, "_indicator_engine", None)
        safe_order = getattr(self.deps, "safe_order_manager", None)
        order_manager = self.deps.order_manager
        if (
            safe_order is None
            and order_manager is not None
            and order_manager.__class__.__name__ == "SafeOrderManager"
        ):
            safe_order = order_manager
            order_manager = getattr(order_manager, "order_manager", order_manager)
        broker = self.deps.broker_client
        ws_disabled = self._ws_disabled()
        risk_manager = getattr(self.deps, "risk_manager", None)

        balance = None
        balance_source = "unknown"
        max_cap_pct = None
        risk_pct = None
        min_lots: int | None = None
        max_lots: int | None = None
        if risk_manager is not None:
            try:
                latest_balance = risk_manager.current_balance
            except Exception as refresh_exc:  # noqa: BLE001
                log.error(
                    "Failure in cmd_flow balance refresh: %s",
                    refresh_exc,
                    extra={"event": "telegram_cmd_flow_balance_error"},
                    exc_info=refresh_exc,
                )
                latest_balance = getattr(risk_manager, "account_balance", None)
            balance = self._coerce_float_value(
                latest_balance,
                field="risk.balance",
            )
            try:
                balance_source = str(risk_manager.balance_source_label())
            except Exception as label_exc:  # noqa: BLE001
                balance_source = "unknown"
                log.error(
                    "Failure in cmd_flow balance label: %s",
                    label_exc,
                    extra={"event": "telegram_cmd_flow_label_error"},
                    exc_info=label_exc,
                )
            settings = getattr(risk_manager, "settings", None)
            if settings is not None:
                risk_pct = self._coerce_float_value(
                    getattr(settings, "per_trade_risk_pct", None),
                    field="risk.per_trade_risk_pct",
                )
                max_cap_pct = self._coerce_float_value(
                    getattr(settings, "per_trade_cap_pct", None),
                    field="risk.per_trade_cap_pct",
                )
                min_lots_value = self._coerce_float_value(
                    getattr(settings, "min_lots_per_trade", None),
                    field="risk.min_lots_per_trade",
                )
                if min_lots_value is not None:
                    min_lots = int(min_lots_value)
                max_lots_value = self._coerce_float_value(
                    getattr(settings, "max_lots_per_trade", None),
                    field="risk.max_lots_per_trade",
                )
                if max_lots_value is not None:
                    max_lots = int(max_lots_value)

        def _fmt_pct(value: float | None) -> str:
            return f"{value:.1f}%" if value is not None else "n/a"

        def _fmt_currency(value: float | None) -> str:
            return self._format_currency(value)

        def _mark(ok: bool) -> str:
            return "✅" if ok else "⚠️"

        max_capital_value = (
            balance * (max_cap_pct / 100.0)
            if balance is not None and max_cap_pct is not None
            else None
        )
        risk_amount = (
            balance * (risk_pct / 100.0)
            if balance is not None and risk_pct is not None
            else None
        )
        if min_lots is not None and max_lots is not None:
            lot_label = f"{min_lots} – {max_lots}"
        elif min_lots is not None:
            lot_label = str(min_lots)
        elif max_lots is not None:
            lot_label = str(max_lots)
        else:
            lot_label = "n/a"

        core_chain = [
            (
                "WS (disabled)" if ws_disabled else "WS",
                True if ws_disabled else bool(ws),
            ),
            ("Streamer", bool(streamer)),
            ("MarketData", bool(mdm)),
            ("Indicator", bool(indicator)),
            ("Strategy", bool(strategy_manager)),
            ("SafeOrder", bool(safe_order)),
            ("OrderMgr", bool(order_manager)),
            ("Broker", bool(broker)),
        ]
        core_line = " → ".join(
            f"{'✅' if component else '❌'} {rb.esc(name)}"
            for name, component in core_chain
        )
        flow_diagram = rb.mono(
            "\n".join(
                [
                    f"{_mark(not ws_disabled and bool(ws))} Websocket",
                    "    ↓",
                    f"{_mark(bool(streamer))} Streamer",
                    "    ↓",
                    f"{_mark(bool(mdm))} Market Data",
                    "    ↓",
                    f"{_mark(bool(indicator))} Indicators",
                    "    ↓",
                    f"{_mark(bool(strategy_manager))} Strategies",
                    "    ↓",
                    "✅ Signal Aggregation",
                    "    ↓",
                    f"{_mark(bool(risk_manager))} Risk Manager",
                    "    ↓",
                    f"{_mark(bool(order_manager))} Order Manager",
                    "    ↓",
                    f"{_mark(bool(broker))} Broker",
                ]
            )
        )

        webhook_enabled = bool(
            self._webhook_server or getattr(self.deps, "webhook_url", None)
        )
        polling_running = (
            self._polling_alive() or self._polling_running() or self._fallback_active
        )
        telegram_chain = [
            ("Webhook", webhook_enabled),
            ("Polling", polling_running),
            ("Controller", True),
            ("Application", bool(self._app)),
        ]
        telegram_line = " → ".join(
            f"{'✅' if component else '❌'} {rb.esc(name)}"
            for name, component in telegram_chain
        )

        is_env_balance = balance_source == "env"
        broker_has_balance = (
            balance is not None and balance > 0 and balance_source in {"live", "cache"}
        )

        warnings: list[str] = []
        if mdm is not None:
            linked_ws = None
            with suppress(Exception):
                linked_ws = getattr(mdm, "_ws", None)
            if ws is not None and linked_ws is not ws:
                warnings.append("MarketDataManager not wired to current WS manager")
        if strategy_manager is not None and indicator is None:
            warnings.append("StrategyManager missing indicator engine link")
        if safe_order is None:
            warnings.append("SafeOrderManager not exposed via Telegram deps")
        if risk_manager is None:
            warnings.append("RiskManager not exposed via Telegram deps")
        elif is_env_balance or not broker_has_balance:
            warnings.append("Broker balance unavailable; using fallback capital")
        if self._app is not None:
            handlers = None
            with suppress(Exception):
                handlers = getattr(self._app, "handlers", None)
            if handlers is not None and not any(handlers.values()):
                warnings.append("Application has no registered handlers")

        header = f"{rb.section('Flow')} {EMOJI['info']}"
        section_core = [header, core_line, flow_diagram]
        if risk_manager is None:
            account_section = [
                f"{rb.section('Account')} {EMOJI['risk']}",
                "<i>Risk manager unavailable</i>",
            ]
        else:
            account_rows = [
                ("Balance", f"{_fmt_currency(balance)} ({balance_source})"),
                (
                    "Max per trade",
                    f"{_fmt_currency(max_capital_value)} ({_fmt_pct(max_cap_pct)})",
                ),
                (
                    "Risk per trade",
                    f"{_fmt_currency(risk_amount)} ({_fmt_pct(risk_pct)})",
                ),
                ("Lot range", f"{lot_label} lots"),
            ]
            account_section = [
                f"{rb.section('Account')} {EMOJI['risk']}",
                rb.grid(account_rows),
            ]
        section_tel = [telegram_line]
        section_warn = [
            f"{EMOJI['warn']} <i>{rb.esc(warning)}</i>" for warning in warnings
        ]
        sections = [section_core, account_section, section_tel]
        if section_warn:
            sections.append(section_warn)
        text = self._compose(sections)

        log.info(
            "telegram_cmd_flow_summary",
            extra={
                "event": "telegram_cmd_flow_summary",
                "balance": round(balance, 2) if balance is not None else None,
                "max_capital": (
                    round(max_capital_value, 2)
                    if max_capital_value is not None
                    else None
                ),
                "risk_amount": (
                    round(risk_amount, 2) if risk_amount is not None else None
                ),
                "per_trade_cap_pct": max_cap_pct,
                "per_trade_risk_pct": risk_pct,
            },
        )

        await self._reply(chat, ctx, text, parse_mode=ParseMode.HTML)

    async def cmd_stderr(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        entries = RING.tail(400)
        items: list[dict[str, t.Any]] = []
        for line in reversed(entries):
            ts, level, message = self._parse_ring_line(line)
            if level not in {"ERROR", "WARNING", "CRITICAL"}:
                continue
            category = self._categorize(line)
            when = self._iso_from_time(ts) or datetime.now().isoformat(
                timespec="seconds"
            )
            event_match = _EVENT_KEY_RE.search(line)
            event_key = event_match.group(1) if event_match else None
            items.append(
                {
                    "when": when,
                    "area": category,
                    "event": event_key,
                    "message": self._short_reason(message),
                    "hint": self._hint_for(category, message),
                }
            )
            if len(items) >= 5:
                break
        if not items:
            await self._reply(chat, ctx, "No recent errors.")
            return
        await self._reply(chat, ctx, self._code(_pp(items)), parse_mode=ParseMode.HTML)

    # --- SRE-grade helpers ---
    async def cmd_config(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        # Expect your AppConfig accessible via deps; otherwise noop
        cfg = getattr(
            sys.modules.get("nifty_scalper_bot.config.env"), "CURRENT_CONFIG", None
        )
        data = _sanitize_dict(_asdict_safe(cfg))
        await self._reply(chat, ctx, self._code(_pp(data)), parse_mode=ParseMode.HTML)

    async def cmd_env(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        keys = ["NODE_ENV", "ENV", "PYTHONHASHSEED", "TZ"]
        data = {k: os.getenv(k) for k in keys}
        await self._reply(chat, ctx, self._code(_pp(data)), parse_mode=ParseMode.HTML)

    async def cmd_metrics(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Return aggregated Prometheus metrics or filtered raw samples.

        Args:
            update: Telegram update envelope triggering the command.
            ctx: Callback context carrying command arguments.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered cmd_metrics",
            extra={"event": "telegram_cmd_metrics_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            log.info(
                "Condition met: cmd_metrics_guard_blocked",
                extra={"event": "telegram_cmd_metrics_guard_blocked"},
            )
            return
        args = list(ctx.args or [])
        mode = args[0].lower() if args else ""
        try:
            if mode == "raw":
                filt = " ".join(args[1:]) if len(args) > 1 else None
                registry = getattr(self.deps, "metrics", None) or METRICS
                summary = _metrics_summary(registry, filt)
                await self._reply(
                    chat,
                    ctx,
                    self._code(summary or "no metrics"),
                    parse_mode=ParseMode.HTML,
                )
                log.info(
                    "Condition met: cmd_metrics_raw_dispatched",
                    extra={"event": "telegram_cmd_metrics_raw", "filter": filt or ""},
                )
                return
            metrics_source = getattr(self.deps, "metrics", None)
            collector_override: MetricsCollector | None = None
            metric_method = getattr(metrics_source, "metric_summary", None)
            if callable(metric_method):
                collector_override = t.cast(MetricsCollector, metrics_source)
            order_manager = getattr(self.deps, "order_manager", None)
            snapshot = build_performance_snapshot(
                order_manager=order_manager,
                metrics=collector_override,
            )
            summary_data = dict(snapshot.metrics)
            fill_stats = snapshot.fill_stats.as_dict()

            def _fmt_seconds(value: object) -> str:
                try:
                    numeric = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return "n/a"
                if numeric <= 0:
                    return "0.000s"
                return f"{numeric:.3f}s"

            def _fmt_float(value: object) -> str:
                try:
                    numeric = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return "n/a"
                return f"{numeric:.2f}"

            def _fmt_percent(value: object) -> str:
                try:
                    numeric = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return "n/a"
                return f"{numeric * 100:.1f}%"

            rb = self._response_builder
            tick_p95 = rb.esc(_fmt_seconds(summary_data.get("tick_latency_p95")))
            tick_line = f"Tick p95: <b>{tick_p95}</b>"
            order_p95 = rb.esc(_fmt_seconds(summary_data.get("order_latency_p95")))
            order_line = f"Order p95: <b>{order_p95}</b>"
            latency_line = f"{rb.section('Latency')} {tick_line} | {order_line}"
            staleness_value = rb.esc(
                _fmt_seconds(summary_data.get("max_tick_staleness"))
            )
            staleness_line = f"Max tick staleness: <b>{staleness_value}</b>"
            queue_avg = _fmt_float(summary_data.get("queue_depth_avg"))
            queue_max = _fmt_float(summary_data.get("queue_depth_max"))
            queue_line = (
                "Queue depth avg=<b>"
                f"{rb.esc(queue_avg)}</b> max=<b>{rb.esc(queue_max)}</b>"
            )
            rejections = summary_data.get("order_rejections_5m", {})
            rejection_line = "No rejections in 5m"
            if isinstance(rejections, t.Mapping) and rejections:
                pairs = ", ".join(
                    f"{rb.esc(str(reason))}:{int(count)}"
                    for reason, count in sorted(rejections.items())
                )
                rejection_line = f"Rejections 5m: <b>{pairs}</b>"
            errors = summary_data.get("error_events_per_minute", {})
            error_line = "Errors/min: <b>n/a</b>"
            if isinstance(errors, t.Mapping) and errors:
                ordered = ", ".join(
                    f"{rb.esc(str(minute))}:{int(count)}"
                    for minute, count in sorted(errors.items())
                )
                error_line = f"Errors/min: <b>{ordered}</b>"
            pnl_breakdown = summary_data.get("pnl_breakdown", {})
            pnl_line = "PnL: <b>n/a</b>"
            if isinstance(pnl_breakdown, t.Mapping) and pnl_breakdown:
                segments: list[str] = []
                for book, values in sorted(pnl_breakdown.items()):
                    realized = (
                        _fmt_float(values.get("realized"))
                        if isinstance(values, t.Mapping)
                        else "n/a"
                    )
                    unrealized = (
                        _fmt_float(values.get("unrealized"))
                        if isinstance(values, t.Mapping)
                        else "n/a"
                    )
                    entry = (
                        f"{rb.esc(str(book))}:R={rb.esc(realized)} "
                        f"U={rb.esc(unrealized)}"
                    )
                    segments.append(entry)
                pnl_line = "PnL: <b>" + ", ".join(segments) + "</b>"
            errors_by_type = summary_data.get("error_events_5m", {})
            type_line = "Errors 5m: <b>n/a</b>"
            if isinstance(errors_by_type, t.Mapping) and errors_by_type:
                type_pairs = ", ".join(
                    f"{rb.esc(str(reason))}:{int(count)}"
                    for reason, count in sorted(errors_by_type.items())
                )
                type_line = f"Errors 5m: <b>{type_pairs}</b>"
            fill_map = fill_stats if isinstance(fill_stats, t.Mapping) else {}
            fills_line = "Fills: <b>n/a</b>"
            if fill_map:
                fill_rate = _fmt_percent(fill_map.get("fill_rate"))
                partial_rate = _fmt_percent(fill_map.get("partial_fill_rate"))
                average_ratio = _fmt_percent(fill_map.get("average_fill_ratio"))
                try:
                    total_orders = int(fill_map.get("total_orders", 0))
                except (TypeError, ValueError):
                    total_orders = 0
                fills_line = (
                    "Fills: <b>"
                    f"rate={rb.esc(fill_rate)} partial={rb.esc(partial_rate)} "
                    f"avg_ratio={rb.esc(average_ratio)} total={total_orders}</b>"
                )
            lines = [
                latency_line,
                staleness_line,
                queue_line,
                rejection_line,
                type_line,
                error_line,
                pnl_line,
                fills_line,
            ]
            await self._reply(chat, ctx, "<br>".join(lines), parse_mode=ParseMode.HTML)
            log.info(
                "Condition met: cmd_metrics_summary_dispatched",
                extra={
                    "event": "telegram_cmd_metrics_summary",
                    "fill_rate": fill_map.get("fill_rate") if fill_map else None,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.error(
                "Failure in cmd_metrics: %s",
                exc,
                extra={"event": "telegram_cmd_metrics_error"},
            )
            with suppress(Exception):
                await self._reply(chat, ctx, "⚠️ metrics unavailable.")

    async def cmd_gc(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        tracemalloc.start()
        a0, b0 = tracemalloc.get_traced_memory()
        before = gc.get_stats()[-1] if hasattr(gc, "get_stats") else {}
        gc.collect()
        a1, b1 = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        after = gc.get_stats()[-1] if hasattr(gc, "get_stats") else {}
        msg = {
            "alloc_cur": a1,
            "alloc_peak": b1,
            "gc_before": before,
            "gc_after": after,
        }
        await self._reply(chat, ctx, self._code(_pp(msg)), parse_mode=ParseMode.HTML)

    async def cmd_profile(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        seconds = 5
        with suppress(Exception):
            if ctx.args:
                seconds = max(1, min(int(ctx.args[0]), 60))
        import cProfile  # noqa
        import pstats

        pr = cProfile.Profile()
        pr.enable()
        await self._reply(chat, ctx, f"Profiling CPU for {seconds}s…")
        await asyncio.sleep(seconds)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(25)
        await self._reply(
            chat, ctx, self._code(s.getvalue()[:3500]), parse_mode=ParseMode.HTML
        )

    async def cmd_dump_logs(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        n = 500
        with suppress(Exception):
            if ctx.args:
                n = max(10, min(int(ctx.args[0]), 5000))
        text = "\n".join(RING.tail(n))
        bio = io.BytesIO(text.encode("utf-8"))
        bio.name = "logs.txt"
        await chat.send_document(InputFile(bio))

    async def cmd_reload(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat = await self._guard(update)
        if chat is None:
            return
        hook = self.deps.reload_hook
        if hook is None:
            await self._reply(chat, ctx, "Reload not supported in this build.")
            return
        out = "reload ok"
        with suppress(Exception):
            out = hook()
        await self._reply(chat, ctx, out)

    @command_meta(
        "/backtest <SYMBOL> [DAYS]",
        "Run the historical options backtest for SYMBOL over DAYS (default 5).",
    )
    async def cmd_backtest(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /backtest command and deliver summary metrics.

        Args:
            update: Telegram update payload.
            ctx: Callback context provided by PTB.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered TelegramBot.cmd_backtest",
            extra={"event": "telegram_cmd_backtest_enter"},
        )
        chat = await self._guard(update)
        if chat is None:
            return
        if not ctx.args:
            await self._reply(chat, ctx, "Usage: /backtest SYMBOL [DAYS]")
            return
        symbol = ctx.args[0].upper()
        try:
            days = int(ctx.args[1]) if len(ctx.args) > 1 else 5
        except (TypeError, ValueError):
            await self._reply(chat, ctx, "Days must be an integer between 1 and 120.")
            return
        days = max(1, min(days, 120))
        try:
            loop = asyncio.get_running_loop()
            await self._reply(
                chat,
                ctx,
                f"Running backtest for {symbol} across {days} day(s)…",
            )
            summary: BacktestSummary = await loop.run_in_executor(
                None, run_quick_backtest, symbol, days
            )
        except FileNotFoundError as exc:
            log.error(
                "Failure in TelegramBot.cmd_backtest: %s",
                exc,
                extra={"event": "telegram_cmd_backtest_data_missing"},
            )
            await self._reply(
                chat,
                ctx,
                f"Backtest data missing for {symbol}: {exc}",
            )
            return
        except Exception as exc:  # noqa: BLE001 - defensive logging
            log.error(
                "Failure in TelegramBot.cmd_backtest: %s",
                exc,
                extra={"event": "telegram_cmd_backtest_error"},
                exc_info=exc,
            )
            await self._reply(chat, ctx, "Backtest failed due to an unexpected error.")
            return
        log.info(
            "Condition met: telegram_cmd_backtest_complete",
            extra={
                "event": "telegram_cmd_backtest_complete",
                "symbol": symbol,
                "days": days,
                "pnl": summary.total_pnl,
                "trades": summary.trade_count,
            },
        )
        await self._reply(
            chat,
            ctx,
            summary.format_markdown(),
            parse_mode=ParseMode.MARKDOWN,
        )


class TelegramWebhookServer:
    """Minimal FastAPI server exposing the Telegram webhook endpoint."""

    def __init__(
        self,
        *,
        bot: TelegramBot,
        path: str,
        secret_token: str | None,
        host: str,
        port: int,
    ) -> None:
        self._bot = bot
        self._path = path if path.startswith("/") else f"/{path}"
        self._secret_token = secret_token
        self._host = host
        self._port = port
        self._app = FastAPI()
        self._app.add_api_route(self._path, self._handle_update, methods=["POST"])
        self._app.add_api_route("/healthz", self._health, methods=["GET"])
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def app(self) -> FastAPI:
        """Expose the underlying FastAPI application (useful for tests)."""

        return self._app

    async def start(self) -> None:
        """Start the bot polling loop properly."""
        if self._running:
            return

        LOGGER.info("Starting Telegram Controller...")
        try:
            # 1. Initialize & Start App
            if not self.application.running:
                await self.application.initialize()
                await self.application.start()

            # 2. FORCE START POLLING (Critical Fix)
            # This ensures the bot actually listens to /commands
            if self.application.updater:
                await self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
                LOGGER.info("✅ Telegram Polling Started Successfully")
            else:
                LOGGER.error("❌ Telegram Updater missing! Commands will not work.")

            # 3. Register Menu
            await self._set_commands()
            self._running = True
            
        except Exception as e:
            LOGGER.error(f"Failed to start Telegram Controller: {e}", exc_info=True)

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.should_exit = True
        if self._task is not None:
            with suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._server = None

    async def _handle_update(self, request: Request) -> Response:
        start = time.perf_counter()
        status_code = status.HTTP_200_OK
        error: str | None = None
        chat_id: int | None = None

        if self._secret_token is not None:
            secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
            if secret != self._secret_token:
                self._bot.record_webhook_secret_failure()
                error = "secret_mismatch"
                status_code = status.HTTP_401_UNAUTHORIZED
            else:
                error = None

        if status_code == status.HTTP_200_OK:
            try:
                payload = await request.json()
                chat_id = await self._bot.process_webhook_payload(payload)
            except json.JSONDecodeError:
                status_code = status.HTTP_400_BAD_REQUEST
                error = "invalid_json"
                self._bot.record_webhook_failure(
                    reason=error,
                    status_code=status_code,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                error = exc.__class__.__name__
                self._bot.record_webhook_failure(
                    reason=error,
                    status_code=status_code,
                )

        latency_ms = (time.perf_counter() - start) * 1000
        log_fn = log.info
        if status_code >= 500:
            log_fn = log.error
        elif status_code >= 400:
            log_fn = log.warning
        log_fn(
            "telegram.webhook_request",
            extra={
                "telegram_webhook": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": round(latency_ms, 2),
                    "chat_id": chat_id,
                    "error": error,
                    "polling_fallback": self._bot.is_polling_active,
                }
            },
        )
        return Response(status_code=status_code)

    async def _health(self) -> JSONResponse:
        payload: dict[str, t.Any] = {
            "ok": True,
            "mode": "polling" if self._bot.is_polling_active else "webhook",
            "metrics": self._bot.metrics.snapshot(),
        }
        guard = self._bot._session_guard_status()
        if guard is not None:
            payload["guard"] = "ok" if guard.get("session_valid") else "blocked"
            payload["guard_details"] = guard
            if "reasons" in guard:
                payload["reasons"] = list(guard.get("reasons", []))
        return JSONResponse(payload)


# -----------------
# Pretty utilities
# -----------------
def _pp(obj: t.Any) -> str:
    try:
        import json

        return json.dumps(obj, indent=2, sort_keys=True, default=str)
    except Exception:
        return repr(obj)


def _asdict_safe(obj: t.Any) -> dict[str, t.Any]:
    if obj is None:
        return {}
    with suppress(Exception):
        from dataclasses import asdict, is_dataclass

        if is_dataclass(obj) and not isinstance(obj, type):
            return t.cast(dict, asdict(obj))
    with suppress(Exception):
        return dict(obj)
    return {"_repr": repr(obj)}


def _sanitize_dict(d: dict[str, t.Any]) -> dict[str, t.Any]:
    if not isinstance(d, dict):
        return d  # type: ignore[return-value]
    redacted = set(["api_key", "api_secret", "access_token", "token", "password"])
    out: dict[str, t.Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _sanitize_dict(v)
        elif any(key in k.lower() for key in redacted):
            out[k] = "****"
        else:
            out[k] = v
    return out


def _metrics_summary(reg: t.Any, filt: str | None) -> str:
    try:
        if reg is None:
            return ""
        # Accept either Prometheus client or custom counters with .collect()
        families: t.Iterable[t.Any] = getattr(reg, "collect", lambda: [])()
        lines: list[str] = []
        for fam in families:
            name = getattr(fam, "name", "")
            if filt and filt.lower() not in name.lower():
                continue
            for s in getattr(fam, "samples", []):
                lines.append(f"{s.name}{s.labels} = {s.value}")
        return "\n".join(lines[:200])
    except Exception:
        return ""


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
