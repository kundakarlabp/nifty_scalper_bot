"""
nifty_scalper_bot.main
Railway-compatible production entrypoint.
Handles non-blocking startup and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. BOOTSTRAP: FAST & SAFE
# -----------------------------------------------------------------------------

# Immediate feedback for logs
print("🚀 nifty-scalper-bot initializing...", flush=True)

load_dotenv(override=True)

# Configure logging to stdout
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

LOG = logging.getLogger("nifty_scalper_bot.main")

REQUIRED_ENV_VARS = [
    "KITE_API_KEY",
    "KITE_API_SECRET",
    "KITE_ACCESS_TOKEN",
]

# -----------------------------------------------------------------------------
# 2. LIFESPAN MANAGER (NON-BLOCKING)
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle.
    CRITICAL: Yields immediately to allow Uvicorn to bind the port.
    The trading bot starts in a background task.
    """
    # Initialize state
    app.state.bot_ready = False
    app.state.bot_error = None
    app.state.bot = None

    async def boot_trading_engine():
        """Internal task to boot the heavy trading logic."""
        try:
            LOG.info("🔒 Validating environment...")
            missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
            if missing:
                raise RuntimeError(f"Missing required env vars: {missing}")

            LOG.info("📦 Importing trading core...")
            # Import inside function to avoid module-level import errors blocking startup
            from nifty_scalper_bot.core.app import NiftyScalperApp

            LOG.info("🤖 Initializing NiftyScalperApp...")
            bot = NiftyScalperApp()
            app.state.bot = bot  # Store ref for shutdown

            LOG.info("▶️ Starting trading loop (Background)...")
            # This will run forever until stop() is called
            await bot.start()
            
            # If we get here (unlikely for infinite loop), mark ready
            app.state.bot_ready = True 

        except asyncio.CancelledError:
            LOG.info("🛑 Boot task cancelled")
        except Exception as exc:
            app.state.bot_error = str(exc)
            LOG.critical(f"❌ FATAL: Bot startup failed: {exc}", exc_info=True)
            # We do NOT exit here, so the web server stays alive to serve logs/health

    # --- STARTUP PHASE ---
    # Create the background task
    boot_task = asyncio.create_task(boot_trading_engine())
    
    # ⚡ YIELD IMMEDIATELY: This tells Railway "We are live!"
    yield

    # --- SHUTDOWN PHASE ---
    LOG.info("🛑 Shutdown signal received. Cleaning up...")
    
    # Cancel boot task if it's still stuck starting
    if not boot_task.done():
        boot_task.cancel()
        try:
            await boot_task
        except asyncio.CancelledError:
            pass

    # Stop the bot gracefully
    if app.state.bot:
        try:
            await app.state.bot.stop()
            LOG.info("✅ Bot shutdown complete.")
        except Exception as exc:
            LOG.error(f"⚠️ Error during bot shutdown: {exc}", exc_info=True)

# -----------------------------------------------------------------------------
# 3. FASTAPI APP DEFINITION
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Nifty Scalper Bot",
    description="Algorithmic Trading Bot for Nifty Options",
    version="2.0.0",
    docs_url=None,   # Disable docs for security in prod
    redoc_url=None,
    lifespan=lifespan,
)

@app.get("/")
async def root() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "running", "service": "nifty-scalper-bot"}

@app.get("/health")
async def health() -> dict[str, Any]:
    """
    Health check endpoint.
    Railway can poll this to check if the TRADING LOGIC is actually running.
    """
    # If the background task crashed, expose the error here
    status = "healthy"
    if app.state.bot_error:
        status = "crashed"
    elif not app.state.bot:
        status = "initializing"
        
    return {
        "status": status,
        "bot_ready": app.state.bot_ready,
        "last_error": app.state.bot_error,
    }
