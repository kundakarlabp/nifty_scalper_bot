"""
nifty_scalper_bot.main

Production-grade entry point for Railway.app deployment with proper
error handling, environment validation, and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from functools import partial
from typing import Any, Callable

import sentry_sdk
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from sentry_sdk.integrations.logging import LoggingIntegration

# ============================================================================
# PHASE 1: INITIALIZATION & ENV VALIDATION (Runs at import time)
# ============================================================================

print("🚀 PYTHON STARTED - Initializing bot...", flush=True)

# Load environment variables with Railway override support
load_dotenv(override=True)

# ============================================================================
# PHASE 1A: STRICT ENVIRONMENT VARIABLE VALIDATION
# ============================================================================

REQUIRED_VARS = [
    "KITE_API_KEY",
    "KITE_API_SECRET",
    "KITE_ACCESS_TOKEN",
]

OPTIONAL_VARS = {
    "SENTRY_DSN": "",
    "ENV": "production",
    "PORT": "8000",
    "ENABLE_EMBEDDED_HTTP_SERVER": "true",
    "LOG_LEVEL": "INFO",
}

def validate_env_vars() -> tuple[bool, list[str]]:
    """Validate that all required environment variables are present."""
    missing = [v for v in REQUIRED_VARS if v not in os.environ]
    return len(missing) == 0, missing

def setup_optional_vars() -> None:
    """Set up optional environment variables with sensible defaults."""
    for var, default_value in OPTIONAL_VARS.items():
        os.environ.setdefault(var, default_value)

# Validate required variables BEFORE creating FastAPI app
is_valid, missing_vars = validate_env_vars()

if not is_valid:
    error_msg = f"❌ CRITICAL: Missing required environment variables: {missing_vars}"
    print(error_msg, flush=True)
    # Log to stderr for Railway visibility
    sys.stderr.write(f"{error_msg}\n")
    sys.stderr.flush()
    # Exit immediately - do not allow partial startup
    sys.exit(1)

# Set up optional variables after validation
setup_optional_vars()

# ============================================================================
# PHASE 2: HEALTH CHECK API (Lightweight, no dependencies)
# ============================================================================
# This FastAPI app serves the /health endpoint that Railway uses
# It's intentionally minimal to avoid import overhead

app = FastAPI(
    title="nifty-scalper-bot-health",
    docs_url=None,  # Disable Swagger docs to reduce memory
    redoc_url=None,  # Disable ReDoc to reduce memory
)

# Track bot initialization status
_bot_initialized = False
_bot_ready = False
_initialization_error: str | None = None

@app.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Railway health check endpoint.
    
    Returns:
        dict with status, ready flag, and optional error message
    """
    global _bot_ready, _initialization_error
    
    status = "error" if _initialization_error else "running"
    message = _initialization_error or "Bot starting..."
    
    return {
        "status": status,
        "ready": _bot_ready,
        "message": message,
    }

@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint for basic connectivity check."""
    return {"status": "running"}

# ============================================================================
# PHASE 3: LOGGING & MONITORING SETUP
# ============================================================================

LOG = logging.getLogger("nifty_scalper_bot.main")

def setup_logging() -> None:
    """Configure logging with appropriate handlers and formatters."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# Configure Sentry for error tracking
sentry_dsn = os.getenv("SENTRY_DSN", "")
if sentry_dsn:
    sentry_logging = LoggingIntegration(
        level=logging.INFO,
        event_level=logging.ERROR,
    )
    
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[sentry_logging],
        traces_sample_rate=0.1,
        environment=os.getenv("ENV", "production"),
        attach_stacktrace=True,
    )
    LOG.info("✅ Sentry initialized for error tracking")
else:
    LOG.debug("⚠️  Sentry DSN not configured - error tracking disabled")

# Set system timezone
os.environ.setdefault("TZ", "Asia/Kolkata")

# ============================================================================
# PHASE 4: ENVIRONMENT FLAG UTILITIES
# ============================================================================

def _env_flag(name: str, *, default: bool = False) -> bool:
    """
    Parse environment variable as boolean.
    
    Recognizes: 1, true, yes, on, y (case-insensitive)
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}

def _should_start_http_server() -> bool:
    """Determine if embedded HTTP server should be started."""
    return _env_flag("ENABLE_EMBEDDED_HTTP_SERVER", default=True)

# ============================================================================
# PHASE 5: MAIN BOT RUNTIME
# ============================================================================

async def _run() -> None:
    """
    Main bot startup routine.
    
    This async function:
    1. Imports heavy modules (trading logic, market data handlers)
    2. Initializes the core trading engine
    3. Starts the HTTP server for health checks & metrics
    4. Manages graceful shutdown on signals
    """
    global _bot_initialized, _bot_ready, _initialization_error
    
    try:
        # NOW we import the heavy stuff (lazy loading)
        LOG.info("📦 Importing core trading modules...")
        from nifty_scalper_bot.core.app import NiftyScalperApp
        from nifty_scalper_bot.utils.logging import (
            silence_third_party_loggers,
            enable_business_logic_logging,
        )
        
        LOG.info("🔧 Initializing NiftyScalperApp core...")
        app_core = NiftyScalperApp()
        
        LOG.info("📝 Setting up logging configuration...")
        silence_third_party_loggers()
        enable_business_logic_logging()
        
        _bot_initialized = True
        
        # ====================================================================
        # HTTP Server Setup
        # ====================================================================
        
        uv_server: uvicorn.Server | None = None
        http_task: asyncio.Task[None] | None = None
        http_enabled = _should_start_http_server()
        
        if http_enabled:
            port = int(os.environ.get("PORT", "8000"))
            LOG.info(f"🌐 Starting embedded HTTP server on port {port}")
            
            uv_config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=port,
                lifespan="on",
                log_config=None,  # Use existing logging config
                access_log=False,  # Reduce noise in logs
            )
            uv_server = uvicorn.Server(uv_config)
        else:
            LOG.info("⏭️  Embedded HTTP server disabled (ENABLE_EMBEDDED_HTTP_SERVER=false)")
        
        # ====================================================================
        # Shutdown Handler Setup
        # ====================================================================
        
        loop = asyncio.get_running_loop()
        stop_future: asyncio.Future[None] = loop.create_future()
        shutting_down = asyncio.Event()
        
        async def _shutdown(reason: str) -> None:
            """Gracefully shut down all services."""
            nonlocal http_task
            
            if shutting_down.is_set():
                return  # Already shutting down
            
            shutting_down.set()
            LOG.info(f"🛑 Shutting down ({reason})...")
            
            try:
                LOG.info("Stopping core trading engine...")
                await app_core.stop()
            except Exception as exc:
                LOG.warning(f"⚠️  Core shutdown raised: {exc}", exc_info=True)
            
            if http_task is not None:
                if uv_server is not None and not uv_server.should_exit:
                    uv_server.should_exit = True
                
                try:
                    LOG.info("Waiting for HTTP server to shutdown...")
                    await asyncio.wait_for(http_task, timeout=5.0)
                except asyncio.TimeoutError:
                    LOG.warning("⏱️  HTTP server shutdown timeout")
                except Exception as exc:
                    LOG.warning(f"⚠️  HTTP server shutdown raised: {exc}")
            
            LOG.info("✅ Shutdown complete")
            if not stop_future.done():
                stop_future.set_result(None)
        
        def _schedule_shutdown(signum: int) -> None:
            """Schedule shutdown on receiving signals."""
            try:
                sig = signal.Signals(signum)
                label = f"signal {sig.name}"
            except ValueError:
                label = f"signal {signum}"
            
            if shutting_down.is_set():
                return
            
            LOG.info(f"📡 Received {label}")
            asyncio.create_task(_shutdown(label))
        
        # Register signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, partial(_schedule_shutdown, sig.value))
            except (NotImplementedError, AttributeError):
                # Windows or unsupported platform
                pass
        
        # ====================================================================
        # Start HTTP Server (if enabled)
        # ====================================================================
        
        if http_enabled and uv_server is not None:
            def _http_task_done(task: asyncio.Task[None]) -> None:
                """Handle HTTP server unexpected termination."""
                if shutting_down.is_set():
                    return
                
                try:
                    task.result()
                except Exception as exc:
                    LOG.error(f"❌ HTTP server stopped unexpectedly: {exc}")
                    asyncio.create_task(_shutdown("http server error"))
                else:
                    LOG.warning("⚠️  HTTP server stopped unexpectedly")
                    asyncio.create_task(_shutdown("http server stopped"))
            
            http_task = asyncio.create_task(
                uv_server.serve(),
                name="uvicorn-server"
            )
            http_task.add_done_callback(_http_task_done)
        
        # ====================================================================
        # Start Core Trading Engine
        # ====================================================================
        
        try:
            LOG.info("🤖 Starting core trading engine...")
            await app_core.start()
            
            _bot_ready = True
            LOG.info("✅ Core ready - strategies active and trading!")
            
            # Wait for shutdown signal
            await stop_future
            
        except Exception as exc:
            _initialization_error = str(exc)
            LOG.exception(f"❌ Fatal error in core: {exc}")
            await _shutdown("fatal error")
            
            # Attempt graceful degradation - stay alive for health checks
            LOG.error("⚠️  Entering idle mode instead of exiting")
            await asyncio.sleep(3600)
            return
        
        finally:
            await _shutdown("finalize")
    
    except Exception as exc:
        _initialization_error = str(exc)
        error_details = f"Startup failed: {exc}"
        LOG.exception(error_details)
        print(f"❌ {error_details}", flush=True)
        sys.exit(1)

def main() -> None:
    """
    Entry point for python -m nifty_scalper_bot.main invocation.
    
    Handles both fresh event loop creation and existing event loops.
    """
    try:
        # Check if event loop already running (shouldn't be at module level)
