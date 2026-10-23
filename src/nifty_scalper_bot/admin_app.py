"""Purpose: Serve the isolated Lightsail admin-control ASGI process.

Responsibilities:
- Expose admin routes independently from the engine event loop.
- Offer a minimal health endpoint for its systemd service.
- Route operator requests to the existing admin-dashboard API handlers.

Boundaries: No strategy loop, market feed or independent broker orders.
Safety: A working admin page is not proof of trading readiness.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from nifty_scalper_bot.admin_dashboard import router as _admin_router

app = FastAPI(title="Nifty Scalper Bot — Admin", docs_url=None, redoc_url=None)
app.include_router(_admin_router)


@app.get("/")
def _root() -> RedirectResponse:
    return RedirectResponse("/admin")


@app.get("/healthz")
def _healthz() -> dict[str, str]:
    # Liveness for the admin process itself (independent of the engine).
    return {"status": "ok"}
