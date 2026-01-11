"""
nifty_scalper_bot.main
Production Entrypoint: Non-blocking startup with Crash Capture.
"""
import sys
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv

# Force logs to flush immediately
sys.stdout.reconfigure(line_buffering=True)

# 1. SETUP
load_dotenv(override=True)
logging.basicConfig(level="INFO", stream=sys.stdout)
LOG = logging.getLogger("nifty_scalper_bot.main")

print("🚀 PYTHON START: Initializing...", flush=True)

# 2. LIFESPAN MANAGER
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bot = None
    app.state.bot_error = None

    async def run_bot_background():
        """Boot the bot safely in the background."""
        try:
            print("⏳ BACKGROUND: Waiting 5s for Server Port Bind...", flush=True)
            await asyncio.sleep(5)
            
            print("📦 BACKGROUND: Importing Trading Engine...", flush=True)
            # Lazy import avoids top-level crashes
            from nifty_scalper_bot.core.app import NiftyScalperApp
            
            print("🤖 BACKGROUND: Initializing Bot...", flush=True)
            bot = NiftyScalperApp()
            app.state.bot = bot
            
            print("▶️ BACKGROUND: Starting Trading Loop...", flush=True)
            await bot.start()
            
        except asyncio.CancelledError:
            print("🛑 BACKGROUND: Task Cancelled", flush=True)
        except Exception as exc:
            # CRITICAL: Capture crash, don't kill container
            app.state.bot_error = str(exc)
            print(f"❌ FATAL BOT CRASH: {exc}", flush=True)
            LOG.critical(f"Bot Crash: {exc}", exc_info=True)

    # Start Task
    task = asyncio.create_task(run_bot_background())
    
    # ⚡ YIELD INSTANTLY -> RAILWAY SEES GREEN DEPLOY
    yield 
    
    # Cleanup
    if not task.done():
        task.cancel()
    if app.state.bot:
        try:
            await app.state.bot.stop()
        except Exception:
            pass

# 3. APP DEFINITION
app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "online", "service": "Nifty Scalper Bot"}

@app.get("/health")
def health():
    """Check if the background bot is alive or crashed."""
    status = "crashed" if app.state.bot_error else "running"
    return {
        "status": status,
        "error": app.state.bot_error,
        "bot_loaded": app.state.bot is not None
    }
