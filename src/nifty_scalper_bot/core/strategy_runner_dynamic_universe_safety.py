"""Keep StrategyRunner's dynamic universe and live evaluation state safe."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
import os
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


def _selected_pair_eval_stall_seconds(runner: Any) -> float:
    """Return the short fail-closed window for a newly rotated selected pair."""
    dispatch_limit = float(
        getattr(runner, "_entry_eval_dispatch_stall_s", 120.0) or 120.0
    )
    raw = os.getenv("SELECTED_PAIR_EVAL_STALL_SECONDS", "15")
    try:
        configured = float(raw or 15.0)
    except (TypeError, ValueError):
        configured = 15.0
    return min(dispatch_limit, max(5.0, configured))


def _schedule_selected_pair_evaluation(
    runner: Any,
    pair: tuple[str | None, str | None],
    *,
    now: float | None = None,
) -> bool:
    """Force the newly selected CE/PE pair onto the entry-eval worker.

    A real ATM rotation must not inherit a stale ``drain_scheduled`` latch from
    the previous pair. Production showed the exact stranded state
    ``pending + drain_scheduled + not drain_active`` for several minutes. On a
    pair transition we therefore coalesce both selected symbols, reset the
    scheduler latch when no drain is active, and explicitly schedule one drain.
    The normal single-worker generation contract still owns evaluation and
    prevents duplicate order paths.
    """
    if not all(pair):
        return False
    lock = getattr(runner, "_eval_gate_lock", None)
    scheduler = getattr(runner, "_schedule_entry_eval_drain", None)
    pending = getattr(runner, "_pending_entry_eval_symbols", None)
    generations = getattr(runner, "_entry_eval_generation_by_symbol", None)
    if lock is None or not callable(scheduler) or not isinstance(pending, set):
        return False
    if generations is None or not hasattr(generations, "get"):
        return False

    resolved_now = time.monotonic() if now is None else float(now)
    should_schedule = False
    with lock:
        if bool(getattr(runner, "_entry_eval_shutdown", False)):
            return False
        for symbol in pair:
            assert symbol is not None
            generations[symbol] = int(generations.get(symbol, 0) or 0) + 1
            pending.add(symbol)
        setattr(runner, "_entry_eval_last_progress_ts", resolved_now)
        setattr(runner, "_last_entry_eval_enqueued_at", resolved_now)
        if not bool(getattr(runner, "_entry_eval_active", False)):
            # A stale True latch with no active drain was the production stall.
            setattr(runner, "_entry_eval_drain_scheduled", True)
            should_schedule = True

    if not should_schedule:
        return True
    if bool(scheduler()):
        logger = getattr(runner, "_logger", None)
        if logger is not None:
            logger.info(
                "SELECTED_PAIR_ENTRY_EVAL_RESCHEDULED selected_ce=%s selected_pe=%s",
                pair[0],
                pair[1],
                extra={
                    "event": "SELECTED_PAIR_ENTRY_EVAL_RESCHEDULED",
                    "selected_ce": pair[0],
                    "selected_pe": pair[1],
                },
            )
        return True

    # If a real runtime pair cannot schedule its sole evaluator, fail closed
    # immediately instead of advertising LIVE/armed until the watchdog timeout.
    with lock:
        setattr(runner, "_entry_eval_drain_scheduled", False)
    setattr(runner, "_entry_eval_stall_disarmed", True)
    setattr(runner, "_runtime_live_orders_armed", False)
    setattr(runner, "_runtime_readiness_reason", "strategy_evaluation_stalled")
    logger = getattr(runner, "_logger", None)
    if logger is not None:
        logger.warning(
            "SELECTED_PAIR_ENTRY_EVAL_SCHEDULE_FAILED selected_ce=%s selected_pe=%s",
            pair[0],
            pair[1],
            extra={
                "event": "SELECTED_PAIR_ENTRY_EVAL_SCHEDULE_FAILED",
                "selected_ce": pair[0],
                "selected_pe": pair[1],
            },
        )
    return False


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
    selected_pair_stall_s = _selected_pair_eval_stall_seconds(runner)
    work_outstanding = bool(adjusted.get("work_outstanding"))
    drain_active = bool(adjusted.get("drain_active"))
    drain_active_age = float(adjusted.get("drain_active_age_s") or 0.0)
    selected_pair_stalled = bool(
        tick_age is not None
        and tick_age <= 5.0
        and epoch_age >= selected_pair_stall_s
    )

    dispatch_stalled = bool(not work_outstanding and selected_pair_stalled)
    worker_stalled = bool(
        work_outstanding
        and (
            drain_active_age >= 90.0
            if drain_active
            else selected_pair_stalled or epoch_age >= 90.0
        )
    ) or dispatch_stalled

    adjusted["tick_age_s"] = round(tick_age, 1) if tick_age is not None else None
    adjusted["dispatch_stalled"] = dispatch_stalled
    adjusted["last_progress_age_s"] = round(epoch_age, 1)
    adjusted["selected_eval_age_s"] = None
    adjusted["selected_pair_eval_stall_s"] = selected_pair_stall_s
    adjusted["evaluation_alive"] = bool(
        not selected_pair_stalled and epoch_age < dispatch_stall_s
    )
    adjusted["worker_stalled"] = worker_stalled
    return adjusted


def _live_ws_option_tick_fresh(
    runner: Any,
    symbol: str,
    *,
    max_age_s: float | None,
) -> bool | None:
    """Return LIVE WS freshness when that authority is available.

    ``None`` means the runner is not in LIVE mode or its MDM does not expose
    genuine WebSocket age, so the caller may preserve the existing fallback.
    In LIVE mode, an exposed MDM probe is authoritative and fails closed when
    no genuine WS tick exists.
    """

    try:
        live_mode = bool(runner._resolve_execution_mode_snapshot().is_live_mode)
    except Exception:  # noqa: BLE001 - legacy/test runners keep old fallback
        live_mode = False
    if not live_mode:
        return None

    mdm = getattr(runner, "_market_data", None)
    time_since_live_ws = getattr(mdm, "time_since_last_live_ws_tick", None)
    if not callable(time_since_live_ws):
        return None

    normalized = normalize_symbol(str(symbol or "")) or symbol
    try:
        age = time_since_live_ws(normalized)
    except (TypeError, ValueError, RuntimeError):
        return False
    if age is None:
        return False
    try:
        age_s = max(0.0, float(age))
        limit_s = 60.0 if max_age_s is None else max(0.0, float(max_age_s))
    except (TypeError, ValueError):
        return False
    return age_s <= limit_s






def apply_patches() -> None:
    """Install the dynamic-universe and selected-option evaluation fixes once."""
    from nifty_scalper_bot.core.runtime_history_event_loop_hardening import (
        apply_app_patch as _apply_runtime_history_patch,
    )
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    # core.app is fully loaded when the production runtime-hardening installer
    # calls this function. Patch app policy + runtime history orchestration before
    # the class idempotency return so module reloads cannot lose either adapter.
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
    original_option_tick_fresh = StrategyRunner._is_option_symbol_tick_fresh

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
                after_pair = _active_selected_pair(self)
                if _record_selected_pair_transition(self, before_pair, after_pair):
                    _schedule_selected_pair_evaluation(self, after_pair)
                return
        original_sync(self, selection)
        after_pair = _active_selected_pair(self)
        if _record_selected_pair_transition(self, before_pair, after_pair):
            _schedule_selected_pair_evaluation(self, after_pair)

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

    @wraps(original_option_tick_fresh)
    def option_symbol_tick_fresh(
        self: Any, symbol: str, *, max_age_s: float | None = None
    ) -> bool:
        live_ws_fresh = _live_ws_option_tick_fresh(
            self, symbol, max_age_s=max_age_s
        )
        if live_ws_fresh is not None:
            return live_ws_fresh
        return bool(original_option_tick_fresh(self, symbol, max_age_s=max_age_s))


    StrategyRunner._dynamic_universe_safety_original_validate = original_validate
    StrategyRunner._dynamic_universe_safety_original_sync = original_sync
    StrategyRunner._dynamic_universe_safety_original_mark_live = original_mark_live
    StrategyRunner._dynamic_universe_safety_original_on_tick = original_on_tick
    StrategyRunner._dynamic_universe_safety_original_liveness = original_liveness
    StrategyRunner._dynamic_universe_safety_original_option_tick_fresh = (
        original_option_tick_fresh
    )
    StrategyRunner._validate_symbol_for_cycle = validate_symbol_for_cycle
    StrategyRunner._sync_active_selection_from_basket = sync_active_selection_from_basket
    StrategyRunner._mark_live = mark_live
    StrategyRunner._on_tick = on_tick
    StrategyRunner._entry_eval_liveness_snapshot = entry_eval_liveness_snapshot
    StrategyRunner._is_option_symbol_tick_fresh = option_symbol_tick_fresh
    StrategyRunner._dynamic_universe_safety_installed = True


__all__ = [
    "_apply_selected_pair_transition_liveness",
    "_live_ws_option_tick_fresh",
    "_record_selected_pair_transition",
    "_schedule_selected_pair_evaluation",
    "apply_patches",
]
