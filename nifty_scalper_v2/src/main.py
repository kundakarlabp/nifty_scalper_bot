"""NiftyScalperBot v2 — FastAPI app factory + uvicorn entrypoint."""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

# Global bot instance
_bot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: start bot on startup, clean shutdown on exit."""
    global _bot
    from .config.settings import get_settings
    from .utils.logging import configure_logging

    settings = get_settings()
    configure_logging(
        log_level=settings.infra.log_level.value,
        log_format=settings.infra.log_format.value,
    )

    from .bot import NiftyScalperBot
    _bot = NiftyScalperBot()

    # Start bot as background task
    bot_task = asyncio.create_task(_bot.run(), name="nifty_scalper_main_loop")

    try:
        yield
    finally:
        log.info("FastAPI lifespan shutting down...")
        if _bot and _bot.is_running:
            await _bot.shutdown()
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    """Factory function — creates and configures the FastAPI app."""
    app = FastAPI(
        title="NiftyScalperBot v2",
        version="2.0.0",
        description="Autonomous Nifty intraday options scalping bot",
        lifespan=lifespan,
    )

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {"status": "ok", "version": "2.0.0", "service": "NiftyScalperBot"}

    @app.get("/health")
    async def health() -> JSONResponse:
        from .infra.health import HealthStatus
        if _bot is None:
            return JSONResponse(
                {"status": "starting", "version": "2.0.0"},
                status_code=503,
            )
        status = HealthStatus(
            status="healthy" if _bot.is_running else "degraded",
            uptime_seconds=_bot.uptime_seconds,
            mode=_bot.mode,
            open_positions=0,
            daily_pnl=0.0,
            last_tick_at=None,
            ws_connected=False,
            warmup_complete=(_bot._tick_hub.warmup_complete if _bot._tick_hub else False),
        )
        return JSONResponse(status.to_dict())

    @app.get("/api/status")
    async def api_status() -> dict[str, Any]:
        if _bot is None:
            return {"status": "starting"}
        return {
            "status": "running" if _bot.is_running else "stopped",
            "mode": _bot.mode,
            "uptime_seconds": _bot.uptime_seconds,
            "version": NiftyScalperBot.VERSION,
        }

    return app


def main() -> None:
    """Entrypoint for `python -m src.main` or `nifty-scalper` CLI."""
    import uvicorn
    from .config.settings import get_settings

    settings = get_settings()
    app = create_app()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.infra.api_port,
        log_level=settings.infra.log_level.value.lower(),
        access_log=False,
    )


# Needed for `python -m src.main`
from .bot import NiftyScalperBot  # noqa: E402 (import after function defs)

if __name__ == "__main__":
    main()
