"""Keep StrategyRunner's dynamic universe and live evaluation state safe."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
import sys
import time
from typing import Any, Callable

from nifty_scalper_bot.utils.symbols import normalize_symbol


def _active_selected_pair(runner: Any) -> tuple[str | None, str | None]:
    """Return the authoritative active CE/PE pair."""
    ce = normalize_symbol(str(getattr(runner, "_active_selected_ce", None) or "")) or None
    pe = normalize_symbol(str(getattr(runner, "_active_selected_pe", None) or "")) or None
    return ce, pe


def _record_selected_pair_transition(
    runner: Any,
    before: tuple[str | None, str | None],
    after: tuple[str | None, str | None],
    *,
    now: float | None = None,
) -> bool:
    """Record only a real active-pair rotation, never initial selection."""
    if before == after or not all(before) or not all(after):
        return False
    started = time.monotonic() if now is None else float(now)
    setattr(runner, "_selected_entry_eval_epoch_pair", after)
    setattr(runner, "_selected_entry_eval_epoch_started_at", started)
    return True


def _apply_selected_pair_transition_liveness(
    runner: Any,
    state: Mapping[str, Any],
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Ignore previous-pair completion age only inside a recorded rotation epoch."""
    adjusted = dict(state)
    current_pair = _active_selected_pair(runner)
    epoch_pair = getattr(runner, "_selected_entry_eval_epoch_pair", None)
    epoch_started = float(
        getattr(runner, "_selected_entry_eval_epoch_started_at", 0.0) or 0.0
    )
    if epoch_pair != current_pair or epoch_started <= 0.0 or not all(current_pair):
        return adjusted

    resolved_now = time.monotonic() if now is None else float(now)
    selected_eval_at = float(
        getattr(runner, "_last_selected_candidate_eval_completed_ts", 0.0) or 0.0
    )
    adjusted["selected_pair_epoch_age_s"] = round(
        max(0.0, resolved_now - epoch_started), 1
    )
    adjusted["selected_pair_epoch_current"] = True

    # Once this pair has produced its own selected-candidate completion, the
    # canonical runner liveness calculation is authoritative again.
    if selected_eval_at >= epoch_started:
        adjusted["selected_eval_in_current_epoch"] = True
        return adjusted

    adjusted["selected_eval_in_current_epoch"] = False
    epoch_age = max(0.0, resolved_now - epoch_started)
    selected_tick_at = float(
        getattr(runner, "_last_selected_option_tick_ts", 0.0) or 0.0
    )
    tick_in_epoch = selected_tick_at >= epoch_started
    tick_age = (
        max(0.0, resolved_now - selected_tick_at) if tick_in_epoch else None
    )
    dispatch_stall_s = float(
        getattr(runner, "_entry_eval_dispatch_stall_s", 120.0) or 120.0
    )
    work_outstanding = bool(adjusted.get("work_outstanding"))
    drain_active = bool(adjusted.get("drain_active"))
    drain_active_age = float(adjusted.get("drain_active_age_s") or 0.0)

    dispatch_stalled = bool(
        not work_outstanding
        and tick_age is not None
        and tick_age <= 5.0
        and epoch_age >= dispatch_stall_s
    )
    worker_stalled = bool(
        work_outstanding
        and (
            drain_active_age >= 90.0 if drain_active else epoch_age >= 90.0
        )
    ) or dispatch_stalled

    adjusted["tick_age_s"] = round(tick_age, 1) if tick_age is not None else None
    adjusted["dispatch_stalled"] = dispatch_stalled
    adjusted["last_progress_age_s"] = round(epoch_age, 1)
    adjusted["selected_eval_age_s"] = None
    adjusted["evaluation_alive"] = epoch_age < dispatch_stall_s
    adjusted["worker_stalled"] = worker_stalled
    return adjusted


def apply_patches() -> None:
    """Install the dynamic-universe and selected-option evaluation fixes once."""
    from nifty_scalper_bot.core.runtime_history_event_loop_hardening import (
        apply_app_patch as _apply_runtime_history_patch,
    )
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    # core.app is fully loaded when the production runtime-hardening installer
    # calls this function. Patch the app function before the class idempotency
    # return so an app-module reload cannot lose the history deferral adapter.
    app_module = sys.modules.get("nifty_scalper_bot.core.app")
    if app_module is not None:
        _apply_runtime_history_patch(app_module)

    if getattr(StrategyRunner, "_dynamic_universe_safety_installed", False):
        return

    original_validate: Callable[..., bool] = StrategyRunner._validate_symbol_for_cycle
    original_sync = StrategyRunner._sync_active_selection_from_basket
    original_mark_live = StrategyRunner._mark_live
    original_on_tick = StrategyRunner._on_tick
    original_liveness = StrategyRunner._entry_eval_liveness_snapshot

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
        before_pair = _active_selected_pair(self)
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
                _record_selected_pair_transition(
                    self, before_pair, _active_selected_pair(self)
                )
                return
        original_sync(self, selection)
        _record_selected_pair_transition(self, before_pair, _active_selected_pair(self))

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
        stored_version = int(
            (getattr(self, "_candle_versions", {}) or {}).get(normalized, 0) or 0
        )
        try:
            incoming_version = int(tick.get("candle_version") or 0)
        except (TypeError, ValueError):
            incoming_version = 0
        current_version = max(stored_version, incoming_version)
        last_version = int(
            (getattr(self, "_last_strategy_versions", {}) or {}).get(normalized, 0)
            or 0
        )
        if (
            normalized in selected
            and normalized not in live_seen
            and current_version > last_version
        ):
            symbol_state = (getattr(self, "_symbol_state", {}) or {}).get(normalized)
            if symbol_state is not None:
                symbol_state._last_eval_bar_ts = None

        clean_tick = tick
        if isinstance(tick, Mapping) and ("version" in tick or "data_version" in tick):
            clean_tick = dict(tick)
            clean_tick.pop("version", None)
            clean_tick.pop("data_version", None)
        return original_on_tick(self, normalized, clean_tick)

    @wraps(original_liveness)
    def entry_eval_liveness_snapshot(
        self: Any, now: float | None = None
    ) -> dict[str, Any]:
        resolved_now = time.monotonic() if now is None else float(now)
        canonical = original_liveness(self, resolved_now)
        return _apply_selected_pair_transition_liveness(
            self, canonical, now=resolved_now
        )

    StrategyRunner._dynamic_universe_safety_original_validate = original_validate
    StrategyRunner._dynamic_universe_safety_original_sync = original_sync
    StrategyRunner._dynamic_universe_safety_original_mark_live = original_mark_live
    StrategyRunner._dynamic_universe_safety_original_on_tick = original_on_tick
    StrategyRunner._dynamic_universe_safety_original_liveness = original_liveness
    StrategyRunner._validate_symbol_for_cycle = validate_symbol_for_cycle
    StrategyRunner._sync_active_selection_from_basket = sync_active_selection_from_basket
    StrategyRunner._mark_live = mark_live
    StrategyRunner._on_tick = on_tick
    StrategyRunner._entry_eval_liveness_snapshot = entry_eval_liveness_snapshot
    StrategyRunner._dynamic_universe_safety_installed = True


__all__ = [
    "_apply_selected_pair_transition_liveness",
    "_record_selected_pair_transition",
    "apply_patches",
]
