"""
nifty_scalper_bot.main

Production Entrypoint
- Preserves existing bot logic
- Prevents zombie / false-alive state
- Fails fast on core app crash
- ✅ FIX: Data directory permission handling
"""

import asyncio
import logging
import os
import socket
import sys
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from nifty_scalper_bot.config.env_utils import normalise_live_env_defaults
from nifty_scalper_bot.config.paths import get_data_dir
from nifty_scalper_bot.utils.async_helpers import safe_task
from nifty_scalper_bot.utils.metrics import ensure_multiproc_dir

# -------------------------------------------------------
# BASIC PROCESS SETUP - PRODUCTION-GRADE ENV LOADING
# -------------------------------------------------------

sys.stdout.reconfigure(line_buffering=True)


def _load_env_file() -> None:
    """Load .env defaults only. Args: none. Returns: None. Raises: none."""
    search_paths = [
        Path.cwd() / ".env",
        Path("/app/.env"),
        Path(__file__).resolve().parents[3] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]

    loaded_from: Path | None = None
    for env_path in search_paths:
        if env_path.exists() and env_path.is_file():
            load_dotenv(dotenv_path=str(env_path), override=False)
            loaded_from = env_path
            break

    normalise_live_env_defaults()

    if loaded_from is not None:
        print(f"✅ ENV FILE FOUND (defaults only): {loaded_from}", flush=True)
    else:
        print("⚠️ WARNING: No .env file found! Using system env vars only.", flush=True)

    effective_keys = [
        "ENABLE_LIVE",
        "ENABLE_LIVE_TRADING",
        "EXECUTION_MODE",
        "ORDERS__ENABLE_LIVE",
        "PAPER__ENABLED",
        "PAPER_MODE",
        "SHADOW_MODE",
        "DATA_DIR",
        "PORT",
    ]
    for key in effective_keys:
        print(f"   🔧 {key} = {os.getenv(key, 'NOT_SET')}", flush=True)


def _ensure_data_directory() -> None:
    """Ensure canonical DATA_DIR exists. Args: none. Returns: None. Raises: OSError."""
    data_dir = get_data_dir()
    os.environ["DATA_DIR"] = str(data_dir)
    print(f"✅ DATA DIRECTORY: {data_dir}", flush=True)


_load_env_file()
_ensure_data_directory()

# Prepare the Prometheus multiprocess directory before any child process or
# metric collector reads the env var.  ``ensure_multiproc_dir`` is idempotent
# and tolerates read-only filesystems — it falls back to a PID-scoped dir or
# disables multiprocess mode entirely without raising.
try:
    _PROM_DIR = ensure_multiproc_dir(clear_stale=True)
    if _PROM_DIR is not None:
        print(f"✅ PROMETHEUS_MULTIPROC_DIR={_PROM_DIR}", flush=True)
    else:
        print(
            "⚠️ PROMETHEUS_MULTIPROC_DIR unavailable — metrics in-process only",
            flush=True,
        )
except Exception as _prom_exc:  # pragma: no cover - defensive
    print(f"⚠️ Prometheus dir setup failed: {_prom_exc}", flush=True)

logging.basicConfig(level="INFO", stream=sys.stdout)
LOG = logging.getLogger("nifty_scalper_bot.main")
STARTUP_INSTANCE_ID = os.getenv("STARTUP_INSTANCE_ID") or uuid4().hex
STARTUP_HOSTNAME = socket.gethostname()
STARTUP_BUILD_SHA = (
    os.getenv("RAILWAY_GIT_COMMIT_SHA")
    or os.getenv("GIT_COMMIT_SHA")
    or os.getenv("SOURCE_VERSION")
    or "unknown"
)
_BOT_START_GUARD = False
print(
    "🚀 PYTHON START: Initializing "
    f"startup_instance={STARTUP_INSTANCE_ID} "
    f"pid={os.getpid()} hostname={STARTUP_HOSTNAME} "
    f"build_sha={STARTUP_BUILD_SHA}",
    flush=True,
)


# -------------------------------------------------------
# FASTAPI LIFESPAN (SUPERVISOR ONLY)
# -------------------------------------------------------


def _try_acquire_bot_start_guard() -> bool:
    """Acquire process-local trading-engine startup guard if available."""

    global _BOT_START_GUARD
    if _BOT_START_GUARD:
        return False
    _BOT_START_GUARD = True
    return True


def _release_bot_start_guard() -> None:
    """Release process-local trading-engine startup guard."""

    global _BOT_START_GUARD
    _BOT_START_GUARD = False
_EVENT_LOOP_LAG_MS = 0.0


async def _event_loop_lag_monitor() -> None:
    global _EVENT_LOOP_LAG_MS
    threshold_ms = float(os.getenv("EVENT_LOOP_LAG_WARN_MS", "500") or 500)
    interval_s = float(os.getenv("EVENT_LOOP_LAG_INTERVAL_SECONDS", "0.5") or 0.5)
    last_warn = 0.0
    loop = asyncio.get_running_loop()
    expected = loop.time() + interval_s
    while True:
        await asyncio.sleep(interval_s)
        now = loop.time()
        lag_ms = max(0.0, (now - expected) * 1000.0)
        _EVENT_LOOP_LAG_MS = lag_ms
        expected = now + interval_s
        if lag_ms >= threshold_ms and now - last_warn >= 30.0:
            last_warn = now
            pending = None
            drain_state = None
            quote_inflight = None
            persistence_state = None
            try:
                ctx = _latest_context()
                mdm = getattr(ctx, "market_data_manager", None) if ctx is not None else None
                stats_fn = getattr(mdm, "get_tick_pressure_stats", None)
                if callable(stats_fn):
                    stats = stats_fn() or {}
                    pending = stats.get("pending_ticks")
                    drain_state = {
                        "scheduled": stats.get("drain_scheduled"),
                        "active": stats.get("active_drains"),
                    }
                datahub = getattr(ctx, "data_hub", None) if ctx is not None else None
                inflight_fn = getattr(datahub, "quote_checkpoint_inflight", None)
                if callable(inflight_fn):
                    quote_inflight = inflight_fn()
                status_fn = getattr(datahub, "persistence_status", None)
                if callable(status_fn):
                    persistence_state = status_fn()
            except Exception:
                pass
            LOG.warning("EVENT_LOOP_LAG_HIGH lag_ms=%.1f tick_pending=%s drain_state=%s quote_checkpoint_inflight=%s persistence_state=%s", lag_ms, pending, drain_state, quote_inflight, persistence_state)


async def _run_bot_background(
    app: FastAPI,
    *,
    startup_delay: float = 5.0,
    app_factory=None,
) -> None:
    """Start the trading engine once for a FastAPI process lifespan."""

    if not _try_acquire_bot_start_guard():
        LOG.error(
            "Duplicate trading-engine startup blocked "
            "startup_instance=%s pid=%s hostname=%s build_sha=%s",
            STARTUP_INSTANCE_ID,
            os.getpid(),
            STARTUP_HOSTNAME,
            STARTUP_BUILD_SHA,
        )
        return

    try:
        print("⏳ BACKGROUND: Waiting 5s for Server Port Bind...", flush=True)
        await asyncio.sleep(startup_delay)
        app.state.bot_starting = True

        print("📦 BACKGROUND: Importing Trading Engine...", flush=True)
        if app_factory is None:
            from nifty_scalper_bot.core.app import NiftyScalperApp

            app_factory = NiftyScalperApp

        print("🤖 BACKGROUND: Initializing Bot...", flush=True)
        bot = app_factory()
        app.state.bot = bot

        print("▶️ BACKGROUND: Starting Trading Loop...", flush=True)
        await bot.start()
        app.state.bot_started = True
        app.state.bot_starting = False

        print("🟢 BACKGROUND: Bot fully operational", flush=True)

    except asyncio.CancelledError:
        print("🛑 BACKGROUND: Task Cancelled", flush=True)
        raise

    except Exception as exc:
        app.state.bot_error = str(exc)
        app.state.bot_started = False
        app.state.bot_starting = False
        print(f"⚠️ BOT STARTUP WARNING: {exc}", flush=True)
        warmup_tokens = ("WARMING_UP", "DATA_WARMUP", "HISTORICAL_READY")
        is_warmup_like = any(token in str(exc).upper() for token in warmup_tokens)
        if is_warmup_like:
            LOG.info(
                "run_bot_background warmup continuation: %s",
                exc,
            )
            LOG.warning("Condition met: bot entered warmup mode after startup delay")
        else:
            LOG.error("Failure in run_bot_background: %s", exc, exc_info=exc)
            LOG.warning(
                "Condition met: bot entered degraded mode after startup failure"
            )

    finally:
        _release_bot_start_guard()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bot = None
    app.state.bot_error = None
    app.state.bot_started = False
    app.state.bot_starting = False
    app.state.startup_instance_id = STARTUP_INSTANCE_ID
    app.state.startup_pid = os.getpid()
    app.state.startup_hostname = STARTUP_HOSTNAME
    app.state.startup_build_sha = STARTUP_BUILD_SHA

    task = safe_task(_run_bot_background(app))
    lag_task = safe_task(_event_loop_lag_monitor())
    yield

    try:
        for running_task in (task, lag_task):
            if not running_task.done():
                running_task.cancel()
                with suppress(asyncio.CancelledError):
                    await running_task
        if app.state.bot:
            await app.state.bot.stop()
    except Exception:
        __import__("logging").getLogger(__name__).exception(
            "[CRITICAL] unhandled exception", exc_info=True
        )


# -------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------

app = FastAPI(lifespan=lifespan)
app.state.bot = None
app.state.bot_error = None
app.state.bot_started = False
app.state.bot_starting = False

# Browser admin dashboard (credentials, logs, daily token, restart) for
# non-technical operation on a plain VM. Routes are password-protected.
try:
    from nifty_scalper_bot.admin_dashboard import router as _admin_router

    app.include_router(_admin_router)
except Exception as _admin_exc:  # noqa: BLE001
    import logging as _logging

    _logging.getLogger(__name__).warning("admin dashboard not mounted: %s", _admin_exc)


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Nifty Scalper Bot",
        "startup_instance_id": STARTUP_INSTANCE_ID,
        "pid": os.getpid(),
        "hostname": STARTUP_HOSTNAME,
        "build_sha": STARTUP_BUILD_SHA,
    }


@app.get("/health")
def health():
    return readyz()


@app.get("/livez")
def livez():
    return {
        "status": "alive",
        "bot_loaded": app.state.bot is not None,
        "engine_http_responsive": True,
        "event_loop_lag_ms": round(float(_EVENT_LOOP_LAG_MS), 3),
    }


def _latest_context():
    bot = getattr(app.state, "bot", None)
    ctx = getattr(bot, "_ctx", None)
    if ctx is not None:
        return ctx
    try:
        from nifty_scalper_bot.core.app import get_latest_bot_context

        return get_latest_bot_context()
    except Exception:
        return None


def _context_blockers(ctx) -> list[str]:  # noqa: ANN001
    blockers: list[str] = []
    decision = getattr(ctx, "readiness_decision", None)
    decision_blockers = list(getattr(decision, "blocker_list", ()) or ())
    blockers.extend(str(item) for item in decision_blockers if item)
    if bool(getattr(ctx, "broker_auth_invalid", False)):
        blockers.append("broker_auth_invalid")
    if not bool(getattr(ctx, "broker_balance_valid", False)):
        blockers.append("broker_balance_unavailable")
    if bool(getattr(ctx, "position_reconciliation_failed", False)):
        blockers.append("position_reconciliation_failed")
    if not bool(getattr(ctx, "position_reconciliation_completed", False)):
        blockers.append("position_reconciliation_incomplete")
    if bool(getattr(ctx, "unprotected_broker_positions", set())):
        blockers.append("unprotected_broker_position")
    live_block_reason = getattr(ctx, "live_block_reason", None)
    if live_block_reason:
        reason = str(live_block_reason).split(":", 1)[-1]
        if reason:
            blockers.append(reason)
    return list(dict.fromkeys(blockers))


_NON_OPERATIONAL_READYZ_BLOCKERS = {
    "market_closed",
    "exchange_holiday",
    "outside_session",
}


def _symbol_bar_counts(ctx, symbol):  # noqa: ANN001
    if not symbol:
        return {"mdm": 0, "runner": 0, "indicator": 0}
    mdm = getattr(ctx, "market_data_manager", None)
    runner = getattr(ctx, "strategy_runner", None)
    try:
        mdm_count = int(mdm.ohlc_count(symbol)) if callable(getattr(mdm, "ohlc_count", None)) else len(mdm.get_ohlc_bars(symbol) or [])
    except Exception:
        mdm_count = 0
    try:
        runner_count = int(runner.runner_history_count(symbol)) if callable(getattr(runner, "runner_history_count", None)) else 0
    except Exception:
        runner_count = 0
    try:
        indicator_count = int(runner.indicator_history_count(symbol)) if callable(getattr(runner, "indicator_history_count", None)) else 0
    except Exception:
        indicator_count = 0
    return {"mdm": mdm_count, "runner": runner_count, "indicator": indicator_count}


def _structured_runtime_status(ctx):  # noqa: ANN001
    basket = getattr(ctx, "active_contract_basket", None) or {}
    def _get(key):
        return basket.get(key) if isinstance(basket, dict) else getattr(basket, key, None)
    selected_ce = getattr(ctx, "selected_ce", None) or _get("selected_ce")
    selected_pe = getattr(ctx, "selected_pe", None) or _get("selected_pe")
    selected = {"atm": getattr(ctx, "atm_strike", None) or _get("atm_strike"), "ce": selected_ce, "pe": selected_pe}
    required = int(os.getenv("READINESS_OPTION_EXEC_MIN_BARS", os.getenv("OPTION_EXECUTION_MIN_BARS", "30")) or 30)
    mdm = getattr(ctx, "market_data_manager", None)
    stats_fn = getattr(mdm, "get_tick_pressure_stats", None)
    tick_pressure = stats_fn() if callable(stats_fn) else {}
    auth_state = "unknown"
    if bool(getattr(ctx, "broker_auth_verified", False) or getattr(ctx, "broker_authenticated", False)):
        auth_state = "authenticated"
    if bool(getattr(ctx, "broker_auth_invalid", False) or getattr(ctx, "broker_session_invalid", False)):
        auth_state = "invalid"
    return {
        "selected": selected,
        "history": {
            "required_execution_bars": required,
            "ce": _symbol_bar_counts(ctx, selected_ce),
            "pe": _symbol_bar_counts(ctx, selected_pe),
        },
        "state": {
            "startup_ready": bool(getattr(ctx, "startup_ready", getattr(ctx, "data_hard_ready", False))),
            "data_hard_ready": bool(getattr(ctx, "data_hard_ready", False)),
            "evaluation_ready": bool(getattr(ctx, "evaluation_ready", False)),
            "live_orders_armed": bool(getattr(ctx, "live_orders_armed", False)),
        },
        "broker_authentication": auth_state,
        "event_loop_lag_ms": round(float(_EVENT_LOOP_LAG_MS), 3),
        "tick_pressure": tick_pressure,
        "build_sha": STARTUP_BUILD_SHA,
    }


@app.get("/readyz")
def readyz():
    if app.state.bot_error:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "ready": False,
                "primary_blocker": "startup_failed",
                "error": app.state.bot_error,
            },
        )

    ctx = _latest_context()
    if not bool(getattr(app.state, "bot_started", False)) or ctx is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "starting",
                "ready": False,
                "primary_blocker": "startup_incomplete",
            },
        )

    blockers = _context_blockers(ctx)
    operational_blockers = [
        blocker
        for blocker in blockers
        if blocker not in _NON_OPERATIONAL_READYZ_BLOCKERS
    ]
    ready = (
        not bool(getattr(ctx, "broker_auth_invalid", False))
        and bool(getattr(ctx, "broker_balance_valid", False))
        and bool(getattr(ctx, "position_reconciliation_completed", False))
        and not bool(getattr(ctx, "position_reconciliation_failed", False))
        and not operational_blockers
    )
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "blocked",
            "ready": ready,
            "primary_blocker": (
                operational_blockers[0]
                if operational_blockers
                else blockers[0]
                if blockers
                else None
            ),
            "blockers": blockers,
        },
    )


@app.get("/health/trading")
def health_trading():
    ctx = _latest_context()
    if ctx is None:
        return JSONResponse(
            status_code=200,
            content={
                "status": "starting",
                "ready": False,
                "live_orders_armed": False,
                "primary_blocker": "startup_incomplete",
                "blockers": ["startup_incomplete"],
            },
        )
    decision = getattr(ctx, "readiness_decision", None)
    blockers = _context_blockers(ctx)
    live_orders_armed = bool(
        getattr(decision, "live_orders_armed", getattr(ctx, "live_orders_armed", False))
    )
    execution_ready = bool(
        getattr(decision, "execution_ready", getattr(ctx, "execution_armed", False))
    )
    primary = getattr(decision, "primary_blocker", None) or (blockers[0] if blockers else None)
    if not live_orders_armed and not primary:
        primary = "startup_pipeline_incomplete"
        blockers = blockers or [primary]
    return JSONResponse(
        status_code=200,
        content={
            "status": "armed" if live_orders_armed else "blocked",
            "ready": execution_ready and not blockers,
            "live_orders_armed": live_orders_armed and not blockers,
            "primary_blocker": primary,
            "blockers": [b for b in blockers if b],
            **_structured_runtime_status(ctx),
            "broker": {
                "ready": bool(getattr(ctx, "broker_ready", False)),
                "authenticated": not bool(getattr(ctx, "broker_auth_invalid", False)),
                "auth_invalid": bool(getattr(ctx, "broker_auth_invalid", False)),
                "balance_valid": bool(getattr(ctx, "broker_balance_valid", False)),
                "balance": getattr(ctx, "last_valid_broker_balance", None),
                "balance_error": getattr(ctx, "broker_balance_error", None),
            },
            "reconciliation": {
                "started": bool(getattr(ctx, "position_reconciliation_started", False)),
                "completed": bool(
                    getattr(ctx, "position_reconciliation_completed", False)
                ),
                "failed": bool(getattr(ctx, "position_reconciliation_failed", False)),
                "error": getattr(ctx, "position_reconciliation_error", None),
                "unprotected_positions": sorted(
                    getattr(ctx, "unprotected_broker_positions", set()) or []
                ),
            },
        },
    )


@app.get("/debug/env")
def debug_env():
    """Expose minimal safe env diagnostics when explicitly enabled."""
    if os.getenv("ALLOW_DEBUG_ENV", "false").lower() not in {"1", "true", "yes", "on"}:
        raise HTTPException(status_code=404, detail="Not found")

    return {
        "ENABLE_LIVE": os.getenv("ENABLE_LIVE", "NOT_SET"),
        "EXECUTION_MODE": os.getenv("EXECUTION_MODE", "NOT_SET"),
        "FORCE_SIGNAL": os.getenv("FORCE_SIGNAL", "NOT_SET"),
        "DATA_DIR": os.getenv("DATA_DIR", "NOT_SET"),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    """Prometheus scrape endpoint.

    Exposes every metric registered in-process plus any sibling process
    metrics when ``PROMETHEUS_MULTIPROC_DIR`` is honoured.  Guarded for the
    rare case where ``prometheus_client`` is not importable so that a missing
    dependency never breaks the HTTP server itself.
    """

    try:
        from prometheus_client import (  # type: ignore[attr-defined]
            CONTENT_TYPE_LATEST,
            CollectorRegistry,
            generate_latest,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        LOG.error("prometheus_client unavailable: %s", exc)
        return PlainTextResponse(
            "# prometheus_client_unavailable\n",
            media_type="text/plain; charset=utf-8",
        )

    registry: CollectorRegistry | None = None
    try:
        from prometheus_client import multiprocess  # type: ignore[attr-defined]

        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - defensive
        LOG.debug("multiprocess registry unavailable: %s", exc)
        registry = None

    payload = generate_latest(registry) if registry is not None else generate_latest()
    return PlainTextResponse(
        payload.decode("utf-8", errors="replace"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/trading/status")
def trading_status():
    """Check if bot is configured for LIVE trading."""
    enable_live = os.getenv("ENABLE_LIVE", "false").lower() == "true"
    exec_mode = os.getenv("EXECUTION_MODE", "SHADOW")

    ctx = _latest_context()
    structured = _structured_runtime_status(ctx) if ctx is not None else {}
    return {
        "enable_live": enable_live,
        "execution_mode": exec_mode,
        "will_trade": enable_live and exec_mode.upper() == "LIVE",
        "engine_http_responsive": True,
        "bot_loaded": app.state.bot is not None,
        "event_loop_lag_ms": round(float(_EVENT_LOOP_LAG_MS), 3),
        **structured,
        "bot_status": (
            "running"
            if app.state.bot
            else ("degraded" if app.state.bot_error else "starting")
        ),
        "warning": (
            None
            if enable_live
            else "⚠️ ENABLE_LIVE is not 'true' - bot will NOT execute real trades!"
        ),
    }
