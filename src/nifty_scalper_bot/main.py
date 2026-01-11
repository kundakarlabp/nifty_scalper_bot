"""
nifty_scalper_bot.main

Production Entrypoint
- Preserves existing bot logic
- Prevents zombie / false-alive state
- Fails fast on core app crash
"""

import sys
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv

# -------------------------------------------------------
# BASIC PROCESS SETUP (UNCHANGED BEHAVIOR)
# -------------------------------------------------------

sys.stdout.reconfigure(line_buffering=True)
load_dotenv(override=True)

logging.basicConfig(level="INFO", stream=sys.stdout)
LOG = logging.getLogger("nifty_scalper_bot.main")

print("🚀 PYTHON START: Initializing...", flush=True)

# -------------------------------------------------------
# HARD EXIT (CRITICAL FOR TRADING SAFETY)
# -------------------------------------------------------

def _fatal_exit(reason: str, exc: Exception | None = None) -> None:
    LOG.critical(f"❌ FATAL BOT EXIT: {reason}", exc_info=exc)
    os._exit(1)   # DO NOT allow partial survival


# -------------------------------------------------------
# FASTAPI LIFESPAN (SUPERVISOR ONLY)
# -------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bot = None
    app.state.bot_error = None

    async def run_bot_background():
        try:
            # Preserve your existing startup delay
            print("⏳ BACKGROUND: Waiting 5s for Server Port Bind...", flush=True)
            await asyncio.sleep(5)

            print("📦 BACKGROUND: Importing Trading Engine...", flush=True)
            from nifty_scalper_bot.core.app import NiftyScalperApp

            print("🤖 BACKGROUND: Initializing Bot...", flush=True)
            bot = NiftyScalperApp()
            app.state.bot = bot

            print("▶️ BACKGROUND: Starting Trading Loop...", flush=True)
            await bot.start()

            print("🟢 BACKGROUND: Bot fully operational", flush=True)

        except asyncio.CancelledError:
            print("🛑 BACKGROUND: Task Cancelled", flush=True)
            raise

        except Exception as exc:
            # THIS IS THE KEY FIX
            app.state.bot_error = str(exc)
            print(f"❌ FATAL BOT CRASH: {exc}", flush=True)
            LOG.critical("Bot crash during startup", exc_info=True)

            # HARD EXIT – NO ZOMBIE STATE
            _fatal_exit("Core bot crashed during startup", exc)

    task = asyncio.create_task(run_bot_background())

    # Yield immediately so FastAPI can bind ports (Railway-safe)
    yield

    # Shutdown path
    try:
        if not task.done():
            task.cancel()
        if app.state.bot:
            await app.state.bot.stop()
    except Exception:
        pass


# -------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Nifty Scalper Bot",
    }

@app.get("/health")
def health():
    """
    Accurate health:
    - crashed → bot_error present
    - running → bot exists
    """
    if app.state.bot_error:
        return {
            "status": "crashed",
            "error": app.state.bot_error,
        }

    return {
        "status": "running" if app.state.bot else "starting",
        "bot_loaded": app.state.bot is not None,
    }
