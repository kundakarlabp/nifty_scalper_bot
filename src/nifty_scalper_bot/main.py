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
from pathlib import Path
from fastapi import FastAPI
from dotenv import load_dotenv

# -------------------------------------------------------
# BASIC PROCESS SETUP - PRODUCTION-GRADE ENV LOADING
# -------------------------------------------------------

sys.stdout.reconfigure(line_buffering=True)

# ✅ CRITICAL FIX: Explicitly find and load .env file
def _load_env_file() -> None:
    """Load .env file from multiple possible locations.
    
    Search order:
    1. Current working directory
    2. /app/.env (Docker/Railway standard)
    3. Project root (relative to this file)
    4. Parent directories up to 3 levels
    """
    search_paths = [
        Path.cwd() / ".env",                          # Current directory
        Path("/app/.env"),                            # Docker/Railway standard
        Path(__file__).resolve().parent.parent.parent.parent / ".env",  # Project root from src/
        Path(__file__).resolve().parent.parent.parent / ".env",         # One level up
    ]
    
    # Also check WORKDIR environment variable if set
    workdir = os.getenv("WORKDIR") or os.getenv("APP_DIR")
    if workdir:
        search_paths.insert(0, Path(workdir) / ".env")
    
    env_loaded = False
    for env_path in search_paths:
        if env_path.exists() and env_path.is_file():
            print(f"✅ ENV FILE FOUND: {env_path}", flush=True)
            load_dotenv(dotenv_path=str(env_path), override=True)
            env_loaded = True
            
            # Debug: Print critical env vars to verify loading
            enable_live = os.getenv("ENABLE_LIVE", "NOT_SET")
            exec_mode = os.getenv("EXECUTION_MODE", "NOT_SET")
            print(f"   📋 ENABLE_LIVE={enable_live}", flush=True)
            print(f"   📋 EXECUTION_MODE={exec_mode}", flush=True)
            break
    
    if not env_loaded:
        print("⚠️ WARNING: No .env file found! Using Railway/system env vars only.", flush=True)
        print(f"   Searched paths: {[str(p) for p in search_paths]}", flush=True)
    
    # Always call load_dotenv() as fallback (handles Railway env vars)
    load_dotenv(override=False)  # Don't override what we already loaded

_load_env_file()

logging.basicConfig(level="INFO", stream=sys.stdout)
LOG = logging.getLogger("nifty_scalper_bot.main")

# ✅ CRITICAL: Log the actual env values for debugging
print("🚀 PYTHON START: Initializing...", flush=True)
print(f"   🔧 ENABLE_LIVE = {os.getenv('ENABLE_LIVE', 'NOT_SET')}", flush=True)
print(f"   🔧 EXECUTION_MODE = {os.getenv('EXECUTION_MODE', 'NOT_SET')}", flush=True)
print(f"   🔧 FORCE_SIGNAL = {os.getenv('FORCE_SIGNAL', 'NOT_SET')}", flush=True)

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
