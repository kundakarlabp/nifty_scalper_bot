"""Prevent active option-basket mutations outside the safe entry window."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Iterable

from nifty_scalper_bot.core.active_basket import active_contract_selection_from_basket
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.market_hours import (
    MarketState,
    get_market_state,
    is_safe_entry_window,
)

LOGGER = get_logger(__name__)


def _rotation_allowed() -> tuple[bool, MarketState]:
    """Return whether active basket mutation is allowed right now."""
    state = get_market_state()
    return state == MarketState.OPEN and is_safe_entry_window(), state


def apply_patches() -> None:
    """Freeze existing membership outside entry hours; allow initial seeding."""
    from nifty_scalper_bot.core.universe_controller import UniverseController

    if getattr(UniverseController, "_off_market_basket_safety_installed", False):
        return

    original: Callable[..., tuple[set[str], set[str]]] = UniverseController.update

    @wraps(original)
    def update(self: Any, new_universe: Iterable[str]) -> tuple[set[str], set[str]]:
        requested = {
            str(symbol) for symbol in new_universe if str(symbol or "").strip()
        }
        rotation_allowed, state = _rotation_allowed()
        current = set(getattr(self, "current_universe", set()) or set())
        if current and requested != current and not rotation_allowed:
            LOGGER.info(
                "UNIVERSE_REFRESH_DEFERRED_OUTSIDE_ENTRY_WINDOW state=%s added=%s removed=%s",
                state.value,
                sorted(requested - current),
                sorted(current - requested),
                extra={
                    "event": "UNIVERSE_REFRESH_DEFERRED_OUTSIDE_ENTRY_WINDOW",
                    "market_state": state.value,
                    "safe_entry_window": False,
                    "added": sorted(requested - current),
                    "removed": sorted(current - requested),
                },
            )
            return set(), set()
        return original(self, requested)

    UniverseController._off_market_basket_safety_original_update = original
    UniverseController.update = update
    UniverseController._off_market_basket_safety_installed = True


def apply_app_patch(app_module: Any) -> None:
    """Preserve the selected pair outside entry hours once a basket exists."""
    if getattr(app_module, "_off_market_basket_commit_safety_installed", False):
        return
    original = getattr(app_module, "_commit_active_dynamic_basket", None)
    if not callable(original):
        raise RuntimeError("active basket commit function unavailable")

    @wraps(original)
    def commit(ctx: Any, **kwargs: Any) -> tuple[str | None, str | None]:
        existing = getattr(ctx, "active_contract_basket", None) or getattr(
            ctx, "active_trading_universe", None
        )
        rotation_allowed, state = _rotation_allowed()
        if existing and not rotation_allowed:
            selection = active_contract_selection_from_basket(existing)
            selected_ce = selection.selected_ce or getattr(ctx, "selected_ce", None)
            selected_pe = selection.selected_pe or getattr(ctx, "selected_pe", None)
            if selected_ce and selected_pe:
                LOGGER.info(
                    "ACTIVE_BASKET_COMMIT_DEFERRED_OUTSIDE_ENTRY_WINDOW selected_ce=%s selected_pe=%s",
                    selected_ce,
                    selected_pe,
                    extra={
                        "event": "ACTIVE_BASKET_COMMIT_DEFERRED_OUTSIDE_ENTRY_WINDOW",
                        "selected_ce": selected_ce,
                        "selected_pe": selected_pe,
                        "market_state": state.value,
                        "safe_entry_window": False,
                    },
                )
                return str(selected_ce), str(selected_pe)
        return original(ctx, **kwargs)

    app_module._off_market_basket_commit_original = original
    app_module._commit_active_dynamic_basket = commit
    app_module._off_market_basket_commit_safety_installed = True
    apply_patches()


__all__ = ["apply_app_patch", "apply_patches"]
