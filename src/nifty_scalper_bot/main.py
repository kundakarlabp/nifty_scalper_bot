from __future__ import annotations
import os, sys
from dotenv import load_dotenv

print("PYTHON STARTED", flush=True)
load_dotenv(override=True)

REQUIRED_VARS = [
    "KITE_API_KEY",
    "KITE_API_SECRET",
    "KITE_ACCESS_TOKEN",
]
missing = [v for v in REQUIRED_VARS if v not in os.environ]
if missing:
    print("⚠️ WARNING - MISSING ENV VARS:", missing, flush=True)
    # Temporarily allow startup without credentials for health checks
    print("Continuing anyway for health checks...", flush=True)
    # sys.exit(1)  # ← Comment this out for testing


import asyncio
import logging
import signal
from functools import partial
from typing import Any, Callable

import sentry_sdk
import uvicorn
from fastapi import FastAPI
from sentry_sdk.integrations.logging import LoggingIntegration

# ✅ CRITICAL FIX: Create a STUB ASGI app for Railway health checks
# This does NOT initialize the bot - just returns a minimal response
app = FastAPI(title="nifty-scalper-bot-health")

@app.get("/health")
async def health_check():
    """Minimal health endpoint - Railway calls this"""
    return {"status": "ok", "ready": False, "message": "Bot starting..."}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"status": "running"}

# ✅ The REAL bot initialization happens in main(), NOT at import time
LOG = logging.getLogger("nifty_scalper_bot.main")

sentry_logging = LoggingIntegration(
    level=logging.INFO,
    event_level=logging.ERROR,
)
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN", ""),
    integrations=[sentry_logging],
    traces_sample_rate=0.1,
    environment=os.getenv("ENV", "production"),
)

os.environ.setdefault("TZ", "Asia/Kolkata")


def _env_flag(name: str, *, default: bool) -> bool:
    """Return boolean flag from environment respecting common truthy values."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _should_start_http_server() -> bool:
    """Determine whether the embedded uvicorn server should be started."""
    return _env_flag("ENABLE_EMBEDDED_HTTP_SERVER", default=True)


async def _run() -> None:
    """Main bot startup - called ONLY when python -m is executed"""
    # NOW we can import the heavy stuff
    from nifty_scalper_bot.core.app import NiftyScalperApp
    from nifty_scalper_bot.utils.logging import (
        silence_third_party_loggers,
        enable_business_logic_logging,
    )

    print("INITIALIZING BOT CORE...", flush=True)
    
    app_core = NiftyScalperApp()
    silence_third_party_loggers()
    enable_business_logic_logging()

    uv_server: uvicorn.Server | None = None
    http_task: asyncio.Task[None] | None = None
    http_enabled = _should_start_http_server()

    if http_enabled:
        port = int(os.environ.get("PORT", "8000"))
        uv_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            lifespan="on",
            log_config=None,
        )
        uv_server = uvicorn.Server(uv_config)
    else:
        LOG.info("Embedded HTTP server disabled via ENABLE_EMBEDDED_HTTP_SERVER=false.")

    loop = asyncio.get_running_loop()
    stop_future: asyncio.Future[None] = loop.create_future()
    shutting_down = asyncio.Event()

    async def _shutdown(reason: str) -> None:
        nonlocal http_task
        if shutting_down.is_set():
            return
        shutting_down.set()
        LOG.info("Shutting down (%s)...", reason)
        try:
            await app_core.stop()
        except Exception as exc:
            LOG.warning("Shutdown step raised: %s", exc)
        if http_task is not None:
            if uv_server is not None and not uv_server.should_exit:
                uv_server.should_exit = True
            try:
                await http_task
            except Exception as exc:
                LOG.warning("HTTP server shutdown raised: %s", exc)
        LOG.info("Shutdown complete.")
        if not stop_future.done():
            stop_future.set_result(None)

    def _schedule_shutdown(signum: int) -> None:
        try:
            sig = signal.Signals(signum)
            label = f"signal {sig.name}"
        except ValueError:
            label = f"signal {signum}"
        if shutting_down.is_set():
            return
        LOG.info("Received %s", label)
        asyncio.create_task(_shutdown(label))

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, partial(_schedule_shutdown, sig.value))
        except (NotImplementedError, AttributeError):
            pass

    if http_enabled and uv_server is not None:
        def _http_task_done(task: asyncio.Task[None]) -> None:
            if shutting_down.is_set():
                return
            try:
                task.result()
            except Exception as exc:
                LOG.error("HTTP server stopped unexpectedly: %s", exc)
                asyncio.create_task(_shutdown("http server error"))
            else:
                LOG.info("HTTP server stopped unexpectedly.")
                asyncio.create_task(_shutdown("http server stopped"))

        http_task = asyncio.create_task(uv_server.serve(), name="uvicorn-server")
        http_task.add_done_callback(_http_task_done)

    try:
        await app_core.start()
        LOG.info("Core ready, strategies active.")
        await stop_future
    except Exception as exc:
        LOG.exception("Fatal error: %s", exc)
        await _shutdown("fatal error")
        LOG.error("Startup failed, entering idle mode instead of exiting.")
        await asyncio.sleep(3600)
        return
    finally:
        await _shutdown("finalize")


def main() -> None:
    """Entry point used by scripts and ``python -m`` invocations."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_run())
    else:
        loop.create_task(_run())


if __name__ == "__main__":
    main()
