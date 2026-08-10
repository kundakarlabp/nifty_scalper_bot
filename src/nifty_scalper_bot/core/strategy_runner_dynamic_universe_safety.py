"""Keep StrategyRunner's dynamic universe and live evaluation state safe."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any, Callable

from nifty_scalper_bot.utils.symbols import normalize_symbol


def apply_patches() -> None:
    """Install the dynamic-universe and selected-option evaluation fixes once."""
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    if getattr(StrategyRunner, "_dynamic_universe_safety_installed", False):
        return

    original_validate: Callable[..., bool] = StrategyRunner._validate_symbol_for_cycle
    original_sync = StrategyRunner._sync_active_selection_from_basket
    original_mark_live = StrategyRunner._mark_live
    original_on_tick = StrategyRunner._on_tick

    @wraps(original_validate)
    def validate_symbol_for_cycle(self: Any, symbol: str) -> bool:
        normalized = normalize_symbol(str(symbol or ""))
        admitted = False
        if normalized and bool(getattr(self, "_universe_dynamic_mode", False)):
            lock = getattr(self, "_lock", None)
            if lock is not None:
                with lock:
                    active = normalized in getattr(self, "_active_symbols", set())
                    frozen = getattr(self, "_frozen_universe", None)
                    if active and isinstance(frozen, set) and normalized not in frozen:
                        frozen.add(normalized)
                        admitted = True
        if admitted:
            self._logger.info(
                "DYNAMIC_ACTIVE_SYMBOL_ADMITTED symbol=%s reason=active_universe_authority",
                normalized,
                extra={
                    "event": "DYNAMIC_ACTIVE_SYMBOL_ADMITTED",
                    "symbol": normalized,
                    "reason": "active_universe_authority",
                },
            )
        return original_validate(self, normalized or symbol)

    @wraps(original_sync)
    def sync_active_selection_from_basket(self: Any, selection: Any) -> None:
        """Do not promote a newly selected CE/PE pair before its indicator history is warm."""
        new_ce = normalize_symbol(str(getattr(selection, "selected_ce", None) or "")) or None
        new_pe = normalize_symbol(str(getattr(selection, "selected_pe", None) or "")) or None
        runtime_ready = hasattr(self, "_option_required_bars") and hasattr(
            self, "_indicator_engine"
        )
        if new_ce and new_pe and runtime_ready:
            required = int(getattr(self, "_option_required_bars", 1) or 1)
            try:
                pair_ready = all(
                    self._history_count_for_symbol(symbol) >= required
                    for symbol in (new_ce, new_pe)
                )
            except Exception:  # noqa: BLE001 - readiness must fail closed
                pair_ready = False
            if not pair_ready:
                self.set_active_option_context(
                    selected_ce=new_ce,
                    selected_pe=new_pe,
                    atm_strike=getattr(selection, "atm_strike", None),
                    option_symbols=getattr(selection, "option_symbols", None),
                )
                return
        original_sync(self, selection)

    @wraps(original_mark_live)
    def mark_live(self: Any, symbol: str) -> Any:
        """Record authoritative live evidence even when phase was already set LIVE."""
        normalized = normalize_symbol(str(symbol or "")) or symbol
        result = original_mark_live(self, normalized)
        live_seen = getattr(self, "_live_bar_seen", None)
        if isinstance(live_seen, set) and normalized:
            live_seen.add(normalized)
        return result

    @wraps(original_on_tick)
    def on_tick(self: Any, symbol: str, tick: Mapping[str, Any]) -> Any:
        """Keep quote versions and hydration-only bar state out of candle gating."""
        normalized = normalize_symbol(str(symbol or "")) or symbol
        selected = {
            normalize_symbol(str(item or ""))
            for item in (
                getattr(self, "_active_selected_ce", None),
                getattr(self, "_active_selected_pe", None),
            )
            if item
        }
        live_seen = getattr(self, "_live_bar_seen", set())
        if normalized in selected and normalized not in live_seen:
            state = (getattr(self, "_symbol_state", {}) or {}).get(normalized)
            if state is not None:
                state._last_eval_bar_ts = None

        clean_tick = tick
        if isinstance(tick, Mapping) and ("version" in tick or "data_version" in tick):
            clean_tick = dict(tick)
            clean_tick.pop("version", None)
            clean_tick.pop("data_version", None)
        return original_on_tick(self, normalized, clean_tick)

    StrategyRunner._dynamic_universe_safety_original_validate = original_validate
    StrategyRunner._dynamic_universe_safety_original_sync = original_sync
    StrategyRunner._dynamic_universe_safety_original_mark_live = original_mark_live
    StrategyRunner._dynamic_universe_safety_original_on_tick = original_on_tick
    StrategyRunner._validate_symbol_for_cycle = validate_symbol_for_cycle
    StrategyRunner._sync_active_selection_from_basket = sync_active_selection_from_basket
    StrategyRunner._mark_live = mark_live
    StrategyRunner._on_tick = on_tick
    StrategyRunner._dynamic_universe_safety_installed = True


__all__ = ["apply_patches"]
