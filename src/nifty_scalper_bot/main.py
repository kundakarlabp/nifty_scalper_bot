"""
nifty_scalper_bot.main

Production-safe entry point for Railway deployment.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

# =============================================================================
# EARLY BOOTSTRAP (NO HEAVY IMPORTS)
# =============================================================================

print("🚀 nifty-scalper-bot starting", flush=True)

load_dotenv(override=True)

REQUIRED_VARS = [
    "KITE_API_KEY",
    "KITE_API_SECRET",
    "KITE_ACCESS_TOKEN",
]

missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    print(f"❌ Missing required env vars: {missing}", flush=True)
    sys.exit(1)

PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

LOG = logging.getLogger("nifty_scalper_bot.main")

# =============================================================================
# HEALTH SERVER (LIGHTWEIGHT, ALWAYS UP)
# =============================================================================

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


# =============================================================================
# CORE BOT RUNNER
# =============================================================================

async def run_bot() -> None:
    global _bot_ready, _bot_error

    try:
        LOG.info("📦 Importing core application")
        from nifty_scalper_bot.core.app import NiftyScalperApp

        bot = NiftyScalperApp()

        LOG.info("🤖 Starting trading engine")
        await bot.start()

        _bot_ready = True
        LOG.info("✅ Bot fully started")

        # Block forever until cancelled
        await asyncio.Event().wait()

    except asyncio.CancelledError:
        LOG.info("🛑 Bot cancellation requested")

    except Exception as exc:
        _bot_error = str(exc)
        LOG.exception("❌ Fatal bot error")

    finally:
        try:
            LOG.info("🔻 Shutting down bot")
            await bot.stop()
        except Exception:
            LOG.exception("⚠️ Error during shutdown")


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    bot_task = loop.create_task(run_bot())

    def _shutdown() -> None:
        LOG.info("📡 Shutdown signal received")
        bot_task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(bot_task)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


# =============================================================================
# UVICORN LAUNCH (RAILWAY REQUIRES THIS)
# =============================================================================

if __name__ == "__main__":
    LOG.info(f"🌐 Starting HTTP server on port {PORT}")

    uvicorn.run(
        "nifty_scalper_bot.main:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )
