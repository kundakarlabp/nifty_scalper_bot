"""
nifty_scalper_bot.main
Railway-safe entrypoint.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# BOOTSTRAP (FAST, SAFE)
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
# FASTAPI LIFESPAN — ZERO BLOCKING
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    MUST yield immediately.
    Railway will kill the container otherwise.
    """
    app.state.bot_ready = False
    app.state.bot_error = None

    async def boot_bot():
        try:
            # EVERYTHING HEAVY HAPPENS HERE (AFTER PORT IS OPEN)
            missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
            if missing:
                raise RuntimeError(f"Missing env vars: {missing}")

            LOG.info("📦 Importing trading engine")
            from nifty_scalper_bot.core.app import NiftyScalperApp

            bot = NiftyScalperApp()
            app.state.bot = bot

            LOG.info("▶️ Starting trading engine")
            await bot.start()

            app.state.bot_ready = True
            LOG.info("✅ Bot is live")

        except Exception as exc:
            app.state.bot_error = str(exc)
            LOG.exception("❌ Bot startup failed")

    # 🔑 START IN BACKGROUND
    asyncio.create_task(boot_bot())

    # 🔑 IMMEDIATELY RELEASE UVICORN
    yield

    LOG.info("🛑 Shutdown initiated")
    bot = getattr(app.state, "bot", None)
    if bot:
        try:
            await bot.stop()
        except Exception:
            LOG.exception("⚠️ Error during shutdown")

# ---------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------

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
        "ready": app.state.bot_ready,
        "error": app.state.bot_error,
    }
