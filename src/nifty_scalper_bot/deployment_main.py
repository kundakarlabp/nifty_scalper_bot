"""File purpose:
    Start the ASGI trading application only after verifying the deployed release.

Key responsibilities:
    - Enforce embedded-versus-runtime commit identity before importing the app.
    - Start the stale-release watchdog and expose the ``/releasez`` endpoint.

Operational constraints:
    - Trading code must not be imported before release verification succeeds.
    - A stale or mismatched image must never bind the service port or arm orders.
"""

from __future__ import annotations

from nifty_scalper_bot.core.release_guard import (
    enforce_release_freshness,
    start_release_watchdog_thread,
)

_RELEASE = enforce_release_freshness()
_RELEASE_WATCHDOG = start_release_watchdog_thread(_RELEASE)

# Import only after release verification. The imported app retains its complete
# lifespan, health, admin and trading routes.
from nifty_scalper_bot.main import app  # noqa: E402

try:  # noqa: SIM105 - keep dashboard optional so release boot never fails on ops UI.
    from nifty_scalper_bot.admin_ops_dashboard import router as _admin_ops_router  # noqa: E402

    app.include_router(_admin_ops_router)
except Exception as _admin_ops_exc:  # noqa: BLE001
    import logging as _logging  # noqa: E402

    _logging.getLogger(__name__).warning("admin ops dashboard not mounted: %s", _admin_ops_exc)

app.state.release = _RELEASE.as_dict()
app.state.release_watchdog_started = _RELEASE_WATCHDOG is not None


@app.get("/releasez")
def releasez() -> dict[str, object]:
    return {
        "status": "fresh" if _RELEASE.fresh or not _RELEASE.strict else "blocked",
        "watchdog_started": _RELEASE_WATCHDOG is not None,
        **_RELEASE.as_dict(),
    }


__all__ = ["app", "releasez"]
