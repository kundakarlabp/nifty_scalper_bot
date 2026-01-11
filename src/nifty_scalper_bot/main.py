"""
nifty_scalper_bot.main
Railway-compatible entrypoint with Startup Delay.
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
# 1. SETUP LOGGING (UNBUFFERED)
# -----------------------------------------------------------------------------
# Ensure logs flush immediately so you can see them in Railway
print("🚀 SYSTEM BOOT: Initializing...", flush=True)

load_dotenv(override=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
LOG = logging.getLogger("nifty_scalper_bot.main")

# -----------------------------------------------------------------------------
# 2. LIFESPAN (WITH SAFETY DELAY)
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bot_ready = False
    app.state.bot_error = None
    app.state.bot = None

    async def boot_sequence():
        """
        Delayed boot to ensure Uvicorn binds the port first.
        """
        try:
            # 🛑 CRITICAL DELAY: Lets Railway mark the deploy 'Healthy' first
            LOG.info("⏳ Waiting 10s for Server Port Binding...")
            await asyncio.sleep(10) 

            LOG.info("🏁 Initiating Bot Startup Sequence...")
            
            # Check Env Vars
            required_vars = ["KITE_API_KEY", "KITE_API_SECRET", "KITE_ACCESS_TOKEN"]
            missing = [v for v in required_vars if not os.getenv(v)]
            if missing:
                raise RuntimeError(f"Missing env vars: {missing}")

            # Import Core (Heavy Operation)
            LOG.info("📦 Importing NiftyScalperApp...")
            from nifty_scalper_bot.core.app import NiftyScalperApp
            
            # Initialize (Heavy Operation)
            LOG.info("🤖 Instantiating Bot...")
            bot = NiftyScalperApp()
            app.state.bot = bot

            # Start Loop
            LOG.info("▶️ Starting Trading Engine...")
            await bot.start()
            
            app.state.bot_ready = True
            LOG.info("✅ Bot is Live and Trading")

        except asyncio.CancelledError:
            LOG.info("🛑 Boot sequence cancelled")
        except Exception as exc:
            # This catches the crash and prints it, keeping the server alive!
            app.state.bot_error = str(exc)
            LOG.critical(f"❌ FATAL BOT CRASH: {exc}", exc_info=True)

    # Launch in background
    task = asyncio.create_task(boot_sequence())
    
    # ⚡ YIELD INSTANTLY -> Railway sees open port -> Deployment turns Green
    yield 

    # Cleanup
    if not task.done():
        task.cancel()
    if app.state.bot:
        with asyncio.suppress(Exception):
            await app.state.bot.stop()

# -----------------------------------------------------------------------------
# 3. API
# -----------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "online", "message": "Check /health for bot status"}

@app.get("/health")
async def health():
    return {
        "status": "crashed" if app.state.bot_error else "running",
        "ready": app.state.bot_ready,
        "last_error": app.state.bot_error
    }
