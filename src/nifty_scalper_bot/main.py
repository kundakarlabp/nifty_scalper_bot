"""
nifty_scalper_bot.main
Production Entrypoint: Decouples Server Startup from Trading Logic.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv

# --- 1. System Setup (Fail-Safe) ---
# Flush output immediately so logs appear in Railway
print("🚀 SYSTEM BOOT: Initializing...", flush=True)

load_dotenv(override=True)

# Configure logging to stdout
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
LOG = logging.getLogger("nifty_scalper_bot.main")

# --- 2. Background Bot Runner ---
async def run_trading_bot(app: FastAPI):
    """
    Runs the trading bot in the background. 
    Catches errors so the web server doesn't crash.
    """
    try:
        # Artificial delay to ensure Uvicorn binds the port FIRST
        LOG.info("⏳ Waiting 5s for Server Port Binding...")
        await asyncio.sleep(5) 
        
        LOG.info("📦 Importing Trading Engine...")
        # Lazy import: Prevents import-time crashes from killing the server
        from nifty_scalper_bot.core.app import NiftyScalperApp
        
        LOG.info("🤖 initializing Bot...")
        bot = NiftyScalperApp()
        app.state.bot = bot # Save ref for shutdown

        LOG.info("▶️ Starting Trading Loop...")
        # This will run forever
        await bot.start()
        
    except asyncio.CancelledError:
        LOG.info("🛑 Bot task cancelled.")
    except Exception as exc:
        # CRITICAL: Capture the crash reason but keep server alive!
        app.state.bot_error = str(exc)
        LOG.critical(f"❌ FATAL BOT CRASH: {exc}", exc_info=True)

# --- 3. Lifecycle Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize State
    app.state.bot = None
    app.state.bot_error = None
    
    # Start Bot in Background Task
    task = asyncio.create_task(run_trading_bot(app))
    
    # ⚡ YIELD IMMEDIATELY: This tells Railway "We are live!"
    yield 
    
    # Cleanup on Shutdown
    if not task.done():
        task.cancel()
    if app.state.bot:
        try:
            await app.state.bot.stop()
        except Exception:
            pass

# --- 4. Web Application ---
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "online", "service": "Nifty Scalper Bot"}

@app.get("/health")
async def health():
    """
    Railway checks this. If bot crashed, we report it here.
    """
    return {
        "status": "crashed" if app.state.bot_error else "running",
        "error": app.state.bot_error
    }
