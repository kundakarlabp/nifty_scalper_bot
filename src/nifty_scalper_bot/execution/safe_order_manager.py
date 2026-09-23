"""File purpose:
    Preserve the historical ``SafeOrderManager`` constructor shape for startup code.

Key responsibilities:
    - Delegate monitoring, counters, skip reasons and order placement to ``OrderManager``.
    - Keep operator-facing compatibility without owning execution state.

Operational constraints:
    - This adapter must not retry, throttle, reprice or submit an order independently.
    - The canonical ``OrderManager`` remains the only execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from nifty_scalper_bot.config.settings import OrderSettings
from nifty_scalper_bot.execution.order_manager import OrderManager


@dataclass(slots=True)
class SafeOrderManager:
    """Thin compatibility adapter with no independent execution state."""

    order_manager: OrderManager
    settings: OrderSettings
    on_order_rejected: Callable[[str, str], None] | None = None
    post_order_hook: Callable[[str, str, int, float | None], None] | None = None
    regime_manager: Any | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.order_manager, name)

    def set_live_enabled(self, enabled: bool) -> None:
        self.settings.enable_live = bool(enabled)

    def start_monitoring(self) -> None:
        callback = getattr(self.order_manager, "start_monitoring", None)
        if callable(callback):
            callback()

    def stop_monitoring(self) -> None:
        callback = getattr(self.order_manager, "stop_monitoring", None)
        if callable(callback):
            callback()

    def throttled_count(self) -> int:
        getter = getattr(self.order_manager, "throttled_count", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                return 0
        return int(getattr(self.order_manager, "_throttled", 0) or 0)

    def rejection_count(self) -> int:
        getter = getattr(self.order_manager, "rejection_count", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                return 0
        return int(getattr(self.order_manager, "_rejections", 0) or 0)

    def consume_skip_reason(self) -> str | None:
        callback = getattr(self.order_manager, "consume_skip_reason", None)
        return callback() if callable(callback) else None

    def place_order(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate directly; this adapter never retries or mutates the request."""

        return self.order_manager.place_order(*args, **kwargs)


__all__ = ["SafeOrderManager"]
