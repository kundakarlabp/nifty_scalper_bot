"""Retired compatibility shell for the former standalone lifecycle manager.

The canonical :class:`BracketManager` owns TP1/TP2, stop-loss, trailing,
confirmed-fill accounting and closure.  This class remains only because the
startup context still exposes the historical field; it never subscribes to
market data or mutates a position.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class LifecycleManager:
    """No-op compatibility shell; not an execution or bracket authority."""

    def __init__(
        self,
        *,
        data_hub: Any | None,
        state_tracker: Any | None = None,
        config_path: Any | None = None,
        clock: Any | None = None,
    ) -> None:
        del data_hub, state_tracker, config_path, clock
        self.retired = True
        LOGGER.info(
            "LIFECYCLE_MANAGER_RETIRED canonical_owner=BracketManager",
            extra={
                "event": "LIFECYCLE_MANAGER_RETIRED",
                "canonical_owner": "BracketManager",
            },
        )

    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    def on_fill(self, **_kwargs: Any) -> None:
        LOGGER.warning(
            "RETIRED_LIFECYCLE_ON_FILL_IGNORED canonical_owner=BracketManager",
            extra={"event": "RETIRED_LIFECYCLE_ON_FILL_IGNORED"},
        )

    def update_all_positions(self, current_time: datetime) -> None:
        del current_time
        return None


__all__ = ["LifecycleManager"]
