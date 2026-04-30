"""
nifty_scalper_bot.main

Production Entrypoint
- Preserves existing bot logic
- Prevents zombie / false-alive state
- Fails fast on core app crash
- ✅ FIX: Data directory permission handling
"""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from nifty_scalper_bot.config.paths import get_data_dir
from nifty_scalper_bot.utils.async_helpers import safe_task
from nifty_scalper_bot.utils.metrics import ensure_multiproc_dir

# -------------------------------------------------------
# BASIC PROCESS SETUP - PRODUCTION-GRADE ENV LOADING
# -------------------------------------------------------

sys.stdout.reconfigure(line_buffering=True)


def _load_env_file() -> None:
    """Load .env file from multiple possible locations.

    Search order:
    1. Current working directory
    2. /app/.env (Docker/Railway standard)
    3. Project root (relative to this file)
    4. Parent directories up to 3 levels
    """
    search_paths = [
        Path.cwd() / ".env",
        Path("/app/.env"),
        Path(__file__).resolve().parent.parent.parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]

    workdir = os.getenv("WORKDIR") or os.getenv("APP_DIR")
    if workdir:
        search_paths.insert(0, Path(workdir) / ".env")

    env_loaded = False
    for env_path in search_paths:
        if env_path.exists() and env_path.is_file():
            print(f"✅ ENV FILE FOUND: {env_path}", flush=True)
            load_dotenv(dotenv_path=str(env_path), override=True)
            env_loaded = True

            enable_live = os.getenv("ENABLE_LIVE", "NOT_SET")
            exec_mode = os.getenv("EXECUTION_MODE", "NOT_SET")
            print(f"   📋 ENABLE_LIVE={enable_live}", flush=True)
            print(f"   📋 EXECUTION_MODE={exec_mode}", flush=True)
            break

    if not env_loaded:
        print(
            "⚠️ WARNING: No .env file found! Using Railway/system env vars only.",
            flush=True,
        )
        print(f"   Searched paths: {[str(p) for p in search_paths]}", flush=True)

    load_dotenv(override=False)


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
        print("⚠️ PROMETHEUS_MULTIPROC_DIR unavailable — metrics in-process only", flush=True)
except Exception as _prom_exc:  # pragma: no cover - defensive
    print(f"⚠️ Prometheus dir setup failed: {_prom_exc}", flush=True)

logging.basicConfig(level="INFO", stream=sys.stdout)
LOG = logging.getLogger("nifty_scalper_bot.main")

print("🚀 PYTHON START: Initializing...", flush=True)
print(f"   🔧 ENABLE_LIVE = {os.getenv('ENABLE_LIVE', 'NOT_SET')}", flush=True)
print(f"   🔧 EXECUTION_MODE = {os.getenv('EXECUTION_MODE', 'NOT_SET')}", flush=True)
print(f"   🔧 FORCE_SIGNAL = {os.getenv('FORCE_SIGNAL', 'NOT_SET')}", flush=True)
print(f"   🔧 DATA_DIR = {os.getenv('DATA_DIR', 'NOT_SET')}", flush=True)


# -------------------------------------------------------
# FASTAPI LIFESPAN (SUPERVISOR ONLY)
# -------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bot = None
    app.state.bot_error = None

    async def run_bot_background():
        try:
            print("⏳ BACKGROUND: Waiting 5s for Server Port Bind...", flush=True)
            await asyncio.sleep(5)

            print("📦 BACKGROUND: Importing Trading Engine...", flush=True)
            from nifty_scalper_bot.core.app import NiftyScalperApp

            print("🤖 BACKGROUND: Initializing Bot...", flush=True)
            bot = NiftyScalperApp()
            app.state.bot = bot

            print("▶️ BACKGROUND: Starting Trading Loop...", flush=True)
            await bot.start()

            print("🟢 BACKGROUND: Bot fully operational", flush=True)

        except asyncio.CancelledError:
            print("🛑 BACKGROUND: Task Cancelled", flush=True)
            raise

        except Exception as exc:
            app.state.bot_error = str(exc)
            print(f"⚠️ BOT STARTUP WARNING: {exc}", flush=True)
            LOG.error("Failure in run_bot_background: %s", exc, exc_info=exc)
            LOG.warning(
                "Condition met: bot entered degraded mode after startup failure"
            )

    task = safe_task(run_bot_background())
    yield

    try:
        if not task.done():
            task.cancel()
        if app.state.bot:
            await app.state.bot.stop()
    except Exception as e:
        __import__("logging").getLogger(__name__).exception(
            "[CRITICAL] unhandled exception", exc_info=True
        )
        raise


# -------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------

app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Nifty Scalper Bot",
    }


@app.get("/health")
def health():
    if app.state.bot_error:
        return {
            "status": "degraded",
            "error": app.state.bot_error,
        }

    return {
        "status": "running" if app.state.bot else "starting",
        "bot_loaded": app.state.bot is not None,
    }


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

    return {
        "enable_live": enable_live,
        "execution_mode": exec_mode,
        "will_trade": enable_live and exec_mode.upper() == "LIVE",
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
