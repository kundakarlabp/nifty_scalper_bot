"""
nifty_scalper_bot.main

Production Entrypoint
- Preserves existing bot logic
- Prevents zombie / false-alive state
- Fails fast on core app crash
- ✅ FIX: Data directory permission handling
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


def _load_env_file() -> None:
    """Load .env file from multiple possible locations.
    
    Search order:
    1. Current working directory
    2. /app/.env (Docker/Railway standard)
    3. Project root (relative to this file)
    4. Parent directories up to 3 levels
    """
    search_paths = [
        Path.cwd() / ".env",
        Path("/app/.env"),
        Path(__file__).resolve().parent.parent.parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
    
    workdir = os.getenv("WORKDIR") or os.getenv("APP_DIR")
    if workdir:
        search_paths.insert(0, Path(workdir) / ".env")
    
    env_loaded = False
    for env_path in search_paths:
        if env_path.exists() and env_path.is_file():
            print(f"✅ ENV FILE FOUND: {env_path}", flush=True)
            load_dotenv(dotenv_path=str(env_path), override=True)
            env_loaded = True
            
            enable_live = os.getenv("ENABLE_LIVE", "NOT_SET")
            exec_mode = os.getenv("EXECUTION_MODE", "NOT_SET")
            print(f"   📋 ENABLE_LIVE={enable_live}", flush=True)
            print(f"   📋 EXECUTION_MODE={exec_mode}", flush=True)
            break
    
    if not env_loaded:
        print("⚠️ WARNING: No .env file found! Using Railway/system env vars only.", flush=True)
        print(f"   Searched paths: {[str(p) for p in search_paths]}", flush=True)
    
    load_dotenv(override=False)


def _ensure_data_directory() -> None:
    """Ensure data directory exists and is writable.
    
    ✅ FIX: Prevents '[Errno 13] Permission denied: /app/data/trades.json'
    """
    data_dirs = [
        Path("/app/data"),
        Path.cwd() / "data",
    ]
    
    for data_dir in data_dirs:
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            test_file = data_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            os.environ["DATA_DIR"] = str(data_dir)
            print(f"✅ DATA DIRECTORY: {data_dir} (writable)", flush=True)
            return
        except (PermissionError, OSError) as e:
            print(f"⚠️ Cannot use {data_dir}: {e}", flush=True)
            continue
    
    fallback = Path.cwd() / "data"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["DATA_DIR"] = str(fallback)
    print(f"⚠️ Using fallback data directory: {fallback}", flush=True)


_load_env_file()
_ensure_data_directory()

logging.basicConfig(level="INFO", stream=sys.stdout)
LOG = logging.getLogger("nifty_scalper_bot.main")

print("🚀 PYTHON START: Initializing...", flush=True)
print(f"   🔧 ENABLE_LIVE = {os.getenv('ENABLE_LIVE', 'NOT_SET')}", flush=True)
print(f"   🔧 EXECUTION_MODE = {os.getenv('EXECUTION_MODE', 'NOT_SET')}", flush=True)
print(f"   🔧 FORCE_SIGNAL = {os.getenv('FORCE_SIGNAL', 'NOT_SET')}", flush=True)
print(f"   🔧 DATA_DIR = {os.getenv('DATA_DIR', 'NOT_SET')}", flush=True)


# -------------------------------------------------------
# FASTAPI LIFESPAN (SUPERVISOR ONLY)
# -------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bot = None
    app.state.bot_error = None

    async def run_bot_background():
        try:
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
            app.state.bot_error = str(exc)
            print(f"⚠️ BOT STARTUP WARNING: {exc}", flush=True)
            LOG.error("Failure in run_bot_background: %s", exc, exc_info=exc)
            LOG.warning(
                "Condition met: bot entered degraded mode after startup failure"
            )

    task = asyncio.create_task(run_bot_background())
    yield

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
    if app.state.bot_error:
        return {
            "status": "degraded",
            "error": app.state.bot_error,
        }

    return {
        "status": "running" if app.state.bot else "starting",
        "bot_loaded": app.state.bot is not None,
    }


@app.get("/debug/env")
def debug_env():
    """Debug endpoint to verify environment variables are loaded correctly."""
    return {
        "ENABLE_LIVE": os.getenv("ENABLE_LIVE", "NOT_SET"),
        "EXECUTION_MODE": os.getenv("EXECUTION_MODE", "NOT_SET"),
        "FORCE_SIGNAL": os.getenv("FORCE_SIGNAL", "NOT_SET"),
        "GLOBAL_MIN_SIGNAL_CONFIDENCE": os.getenv("GLOBAL_MIN_SIGNAL_CONFIDENCE", "NOT_SET"),
        "MIN_INDICATOR_BARS": os.getenv("MIN_INDICATOR_BARS", "NOT_SET"),
        "ELITE_STRATEGIES_ENABLED": os.getenv("ELITE_STRATEGIES_ENABLED", "NOT_SET"),
        "SMC_ENABLED": os.getenv("SMC_ENABLED", "NOT_SET"),
        "WEBSOCKET__DISABLED": os.getenv("WEBSOCKET__DISABLED", "NOT_SET"),
        "DATA_DIR": os.getenv("DATA_DIR", "NOT_SET"),
        "cwd": os.getcwd(),
        "env_file_exists_cwd": os.path.exists(".env"),
        "env_file_exists_app": os.path.exists("/app/.env"),
    }


@app.get("/trading/status")
def trading_status():
    """Check if bot is configured for LIVE trading."""
    enable_live = os.getenv("ENABLE_LIVE", "false").lower() == "true"
    exec_mode = os.getenv("EXECUTION_MODE", "SHADOW")
    
    return {
        "enable_live": enable_live,
        "execution_mode": exec_mode,
        "will_trade": enable_live and exec_mode.upper() == "LIVE",
        "bot_status": "running" if app.state.bot else ("degraded" if app.state.bot_error else "starting"),
        "warning": None if enable_live else "⚠️ ENABLE_LIVE is not 'true' - bot will NOT execute real trades!",
    }
