"""
nifty_scalper_bot.main

Production-safe entry point for Railway deployment.
Build-safe, import-safe, runtime-validated.
"""

from __future__ import annotations

# ============================================================================
# IMPORT-SAFE BOOTSTRAP (NO RUNTIME ASSUMPTIONS)
# ============================================================================

import asyncio
import logging
import os
import signal
import sys
from typing import Any

from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv

# ============================================================================
# BASIC BOOTSTRAP (SAFE DURING BUILD)
# ============================================================================

print("🚀 nifty-scalper-bot module imported", flush=True)

load_dotenv(override=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
PORT = int(os.getenv("PORT", "8000"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

LOG = logging.getLogger("nifty_scalper_bot.main")

# ============================================================================
# FASTAPI APP (IMPORT-SAFE)
# ============================================================================

app = FastAPI(
    title="nifty-scalper-bot",
    docs_url=None,
    redoc_url=None,
)

_bot_ready = False
_bot_error: str | None = None


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "running"}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if _bot_error is None else "error",
        "ready": _bot_ready,
        "error": _bot_error,
    }


# ============================================================================
# RUNTIME ENV VALIDATION (NEVER AT IMPORT TIME)
# ============================================================================

REQUIRED_ENV_VARS = [
    "KITE_API_KEY",
    "KITE_API_SECRET",
    "KITE_ACCESS_TOKEN",
]


def validate_runtime_env() -> None:
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")


# ============================================================================
# CORE BOT TASK
# ============================================================================

async def run_bot() -> None:
    """
    Runs the trading engine in background.
    This function is NEVER executed during Docker build.
    """
    global _bot_ready, _bot_error

    try:
        LOG.info("🔍 Validating runtime environment")
        validate_runtime_env()

        LOG.info("📦 Importing core trading engine")
        from nifty_scalper_bot.core.app import NiftyScalperApp

        bot = NiftyScalperApp()

        LOG.info("🤖 Starting trading engine")
        await bot.start()

        _bot_ready = True
        LOG.info("✅ Bot is live and trading")

        # Keep running until cancelled
        await asyncio.Event().wait()

    except asyncio.CancelledError:
        LOG.info("🛑 Bot cancellation requested")

    except Exception as exc:
        _bot_error = str(exc)
        LOG.exception("❌ Fatal bot error")

    finally:
        try:
            LOG.info("🔻 Shutting down trading engine")
            await bot.stop()
        except Exception:
            LOG.exception("⚠️ Error during bot shutdown")


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

def start_event_loop() -> None:
    """
    Owns the asyncio event loop and bot lifecycle.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    bot_task = loop.create_task(run_bot(), name="bot-task")

    def shutdown() -> None:
        LOG.info("📡 Shutdown signal received")
        bot_task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(bot_task)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


# ============================================================================
# ENTRYPOINT (RAILWAY EXPECTS THIS)
# ============================================================================

if __name__ == "__main__":
    LOG.info(f"🌐 Starting HTTP server on port {PORT}")

    # Start bot lifecycle in background thread
    import threading
    threading.Thread(
        target=start_event_loop,
        name="bot-runtime",
        daemon=True,
    ).start()

    # Start HTTP server (PID 1 stays alive)
    uvicorn.run(
        "nifty_scalper_bot.main:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        access_log=False,
    )
