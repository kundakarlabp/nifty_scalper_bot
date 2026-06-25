"""Release-verified ASGI entrypoint.

The release identity is verified before importing the trading application. This
prevents a cached or mismatched image from binding the service port and arming
orders. A daemon then stops an instance when GitHub ``main`` is confirmed to be
newer than the running commit.
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
