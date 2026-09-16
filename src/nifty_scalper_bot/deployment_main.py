"""Production ASGI wrapper with release verification and low-resource off-hours mode.

Release freshness is always enforced before either the full trading application
or the lightweight admin-only application binds the production port.
"""
from __future__ import annotations

from fastapi import FastAPI

from nifty_scalper_bot.core.release_guard import enforce_release_freshness, start_release_watchdog_thread
from nifty_scalper_bot.operating_control import engine_should_run, operating_mode, router as operating_router, start_transition_watchdog

_RELEASE = enforce_release_freshness()
_RELEASE_WATCHDOG = start_release_watchdog_thread(_RELEASE)
_ENGINE_EXPECTED = engine_should_run()

if _ENGINE_EXPECTED:
    # Import trading code only when the configured operating profile requires it.
    from nifty_scalper_bot.main import app  # noqa: E402
else:
    # Keep a tiny management plane reachable while the expensive trading stack is asleep.
    from nifty_scalper_bot.admin_dashboard import router as admin_router

    app = FastAPI(title="Nifty Scalper Quiet Control")
    app.include_router(admin_router)

    @app.get("/")
    def quiet_root() -> dict[str, object]:
        return {"status": "online", "operating_mode": operating_mode(), "engine_loaded": False, "quiet": True}

    @app.get("/livez")
    def quiet_livez() -> dict[str, object]:
        return {"status": "alive", "bot_loaded": False, "engine_http_responsive": True, "operating_mode": operating_mode(), "quiet": True}

app.include_router(operating_router)
app.state.release = _RELEASE.as_dict()
app.state.release_watchdog_started = _RELEASE_WATCHDOG is not None
app.state.operating_mode = operating_mode()
app.state.engine_expected = _ENGINE_EXPECTED
_OPERATING_WATCHDOG = start_transition_watchdog(_ENGINE_EXPECTED)


@app.get("/releasez")
def releasez() -> dict[str, object]:
    return {
        "status": "fresh" if _RELEASE.fresh or not _RELEASE.strict else "blocked",
        "watchdog_started": _RELEASE_WATCHDOG is not None,
        "operating_mode": operating_mode(),
        "engine_expected": _ENGINE_EXPECTED,
        **_RELEASE.as_dict(),
    }


__all__ = ["app", "releasez"]
