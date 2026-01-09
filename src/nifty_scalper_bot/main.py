"""
nifty_scalper_bot.main
Railway-compatible production entrypoint.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# SAFE BOOTSTRAP
# -----------------------------------------------------------------------------

print("🚀 nifty-scalper-bot imported", flush=True)

load_dotenv(override=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

LOG = logging.getLogger("nifty_scalper_bot.main")

REQUIRED_ENV_VARS = [
    "KITE_API_KEY",
    "KITE_API_SECRET",
    "KITE_ACCESS_TOKEN",
]


def validate_runtime_env() -> None:
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")


# -----------------------------------------------------------------------------
# FASTAPI LIFESPAN (NON-BLOCKING — CRITICAL)
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    DO NOT BLOCK STARTUP.
    Railway requires port to bind quickly.
    """
    validate_runtime_env()

    from nifty_scalper_bot.core.app import NiftyScalperApp

    bot = NiftyScalperApp()
    app.state.bot_ready = False
    app.state.bot_error = None

    async def start_bot():
        try:
            LOG.info("🤖 Starting trading engine (background)")
            await bot.start()
            app.state.bot_ready = True
            LOG.info("✅ Trading engine running")
        except Exception as exc:
            app.state.bot_error = str(exc)
            LOG.exception("❌ Bot crashed")

    # 🔑 START BOT IN BACKGROUND — DO NOT AWAIT
    asyncio.create_task(start_bot())

    yield  # 🚦 SERVER IS LIVE, PORT IS OPEN

    LOG.info("🛑 Shutting down trading engine")
    try:
        await bot.stop()
    except Exception:
        LOG.exception("⚠️ Error during shutdown")


# -----------------------------------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------------------------------

app = FastAPI(
    title="nifty-scalper-bot",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "running"}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if app.state.bot_error is None else "error",
        "ready": app.state.bot_ready,
        "error": app.state.bot_error,
    }
