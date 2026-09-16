"""Production ASGI wrapper with release verification and off-hours quiet mode.

File purpose:
    Select either the canonical trading application or a lightweight admin-only
    application after release verification.

Key responsibilities:
    - Enforce release freshness before binding the production port.
    - Load the trading stack only when the operating policy requires it.
    - Keep admin and operating-mode controls reachable while the engine is quiet.

Operational constraints:
    - Trading code must not be imported in QUIET/off-hours AUTO mode.
    - Release verification remains mandatory in every operating mode.
    - Operating-mode transitions restart the service rather than mutating a live
      trading engine in place.
"""

from __future__ import annotations

from fastapi import FastAPI

from nifty_scalper_bot.core.release_guard import (
    enforce_release_freshness,
    start_release_watchdog_thread,
)
from nifty_scalper_bot.operating_control import (
    engine_should_run,
    install_admin_power_card,
    operating_mode,
    start_transition_watchdog,
)
from nifty_scalper_bot.operating_control import router as operating_router

_RELEASE = enforce_release_freshness()
_RELEASE_WATCHDOG = start_release_watchdog_thread(_RELEASE)
_ENGINE_EXPECTED = engine_should_run()

if _ENGINE_EXPECTED:
    # Import trading code only when the configured operating profile requires it.
    from nifty_scalper_bot.main import app  # noqa: E402
else:
    # Keep the management plane reachable while the trading stack is asleep.
    from nifty_scalper_bot.admin_dashboard import router as admin_router

    app = FastAPI(title="Nifty Scalper Quiet Control")
    app.include_router(admin_router)

    @app.get("/")
    def quiet_root() -> dict[str, object]:
        return {
            "status": "online",
            "operating_mode": operating_mode(),
            "engine_loaded": False,
            "quiet": True,
        }

    @app.get("/livez")
    def quiet_livez() -> dict[str, object]:
        return {
            "status": "alive",
            "bot_loaded": False,
            "engine_http_responsive": True,
            "operating_mode": operating_mode(),
            "quiet": True,
        }


# The same card is visible in the canonical /admin page in both ACTIVE and QUIET.
install_admin_power_card()
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
