"""Standalone admin dashboard ASGI app.

The dashboard and the trading engine previously ran in ONE uvicorn process,
sharing a single asyncio event loop (the engine runs as a lifespan task on the
same loop). On the memory-tight host, engine CPU/loop load and swap pressure
gradually starved the shared loop until the dashboard stopped responding —
"works for a while, then hangs".

This entrypoint serves ONLY the admin router, which is filesystem-only — it
touches ``.env``, journald, git and systemctl, and imports no engine objects.
Run it as its own service so the dashboard is fully isolated and stays
responsive no matter how busy the engine is:

    uvicorn nifty_scalper_bot.admin_app:app --host 0.0.0.0 --port 8081

Token entry / mode changes are written to ``.env``; the engine (a separate
service) reads them on its next restart, which the dashboard's Restart/Update
buttons trigger.
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
