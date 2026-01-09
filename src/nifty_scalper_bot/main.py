"""
nifty_scalper_bot.main

Railway-compatible production entrypoint.
Uses FastAPI lifespan for clean startup/shutdown.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv

# =============================================================================
# SAFE BOOTSTRAP (IMPORT-TIME SAFE — BUILD WILL NOT FAIL)
# =============================================================================

print("🚀 nifty-scalper-bot module imported", flush=True)

load_dotenv(override=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

LOG = logging.getLogger("nifty_scalper_bot.main")

# =============================================================================
# RUNTIME ENV VALIDATION (NOT AT IMPORT TIME)
# =============================================================================

REQUIRED_ENV_VARS = [
    "KITE_API_KEY",
    "KITE_API_SECRET",
    "KITE_ACCESS_TOKEN",
]


def validate_runtime_env() -> None:
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {missing}")


# =============================================================================
# FASTAPI LIFESPAN — THIS IS THE KEY FIX
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Railway-safe lifecycle:
    - Runs only at container runtime
    - No threads
    - No manual event loop ownership
    """
    bot = None
    try:
        LOG.info("🔍 Validating runtime environment")
        validate_runtime_env()

        LOG.info("📦 Importing trading engine")
        from nifty_scalper_bot.core.app import NiftyScalperApp

        LOG.info("🤖 Initializing trading engine")
        bot = NiftyScalperApp()

        LOG.info("▶️ Starting trading engine")
        await bot.start()

        app.state.bot_ready = True
        LOG.info("✅ Bot is live")

        yield  # 🚦 Application is now RUNNING

    except Exception as exc:
        app.state.bot_error = str(exc)
        LOG.exception("❌ Fatal startup error")
        raise

    finally:
        if bot is not None:
            try:
                LOG.info("🛑 Stopping trading engine")
                await bot.stop()
                LOG.info("✅ Bot stopped cleanly")
            except Exception:
                LOG.exception("⚠️ Error during shutdown")


# =============================================================================
# FASTAPI APP (RAILWAY EXPECTS THIS)
# =============================================================================

app = FastAPI(
    title="nifty-scalper-bot",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

app.state.bot_ready = False
app.state.bot_error: str | None = None


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
