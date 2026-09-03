"""Keep underlying completed-bar history continuous from MDM to strategies.

Ownership contract:
- MarketDataManager owns authoritative completed candle/history state.
- StrategyRunner mirrors that completed state into its local history and the
  shared IndicatorEngine through the existing canonical sync function.
- DataHub remains quote/context transport and is not a second history owner.
- StrategyManager/strategies consume the shared IndicatorEngine.
- OrderManager never hydrates or owns market history.

This module deliberately does not relax readiness, strategy, risk, or order
checks. Deeper ORB history is a structural target, not a new execution minimum.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from functools import wraps
import os
import time
from typing import Any, Callable

from nifty_scalper_bot.execution.readiness import HistoryReadinessPolicy
from nifty_scalper_bot.utils.symbols import normalize_symbol

_CONTEXT_SESSION_HISTORY_BARS = 400
_CONTEXT_PROBE_INTERVAL_SECONDS = 5.0
_CONTEXT_TARGET_REQUEST_INTERVAL_SECONDS = 30.0
_TRUE_VALUES = {"1", "true", "yes", "y", "on", "enable", "enabled"}
_CONTEXT_ROLES = {"spot_context", "futures_context"}


def _env_enabled(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in _TRUE_VALUES


def _safe_positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(float(os.getenv(name, str(default)) or default)))
    except (TypeError, ValueError):
        return max(1, int(default))


def _structural_context_target() -> int:
    """Return bounded underlying history target needed by enabled strategies."""
    try:
        smc_min = max(1, int(HistoryReadinessPolicy.from_env().smc_min_bars))
    except Exception:  # noqa: BLE001 - deterministic safety fallback
        smc_min = _safe_positive_int("SMC_MIN_BARS_REQUIRED", 30)
    target = smc_min
    if _env_enabled("ORB_ENABLED", True):
        target = max(target, _CONTEXT_SESSION_HISTORY_BARS)
    try:
        safety_max = int(float(os.getenv("HYDRATION_MAX_BARS", "0") or 0))
    except (TypeError, ValueError):
        safety_max = 0
    if safety_max > 0:
        target = min(target, safety_max)
    return max(1, target)


def _context_minimum(runner: Any) -> int:
    configured = max(1, int(getattr(runner, "_context_required_bars", 1) or 1))
    try:
        smc_min = max(1, int(HistoryReadinessPolicy.from_env().smc_min_bars))
    except Exception:  # noqa: BLE001
        smc_min = _safe_positive_int("SMC_MIN_BARS_REQUIRED", 30)
    return max(configured, smc_min)


def _context_role(runner: Any, symbol: str) -> str | None:
    normalized = normalize_symbol(str(symbol or "")) or str(symbol or "")
    resolver = getattr(runner, "_history_role_for_symbol", None)
    if callable(resolver):
        try:
            resolved = str(resolver(normalized) or "")
        except Exception:  # noqa: BLE001 - deterministic identity fallback below
            resolved = ""
        if resolved in _CONTEXT_ROLES:
            return resolved
    active_future = normalize_symbol(
        str(getattr(runner, "_active_futures_symbol", None) or "")
    )
    spot_symbol = normalize_symbol(
        str(getattr(runner, "_spot_symbol", None) or "NSE:NIFTY")
    )
    if active_future and normalized == active_future:
        return "futures_context"
    if spot_symbol and normalized == spot_symbol:
        return "spot_context"
    return None


def _raw_bar_timestamp(row: Any) -> Any:
    if isinstance(row, Mapping):
        return (
            row.get("timestamp")
            or row.get("start")
            or row.get("date")
            or row.get("time")
        )
    return (
        getattr(row, "timestamp", None)
        or getattr(row, "start", None)
        or getattr(row, "date", None)
        or getattr(row, "time", None)
    )


def _bar_epoch(row: Any) -> float | None:
    value = _raw_bar_timestamp(row)
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, (int, float)):
            raw = float(value)
            if abs(raw) >= 1e17:
                raw /= 1_000_000_000.0
            elif abs(raw) >= 1e14:
                raw /= 1_000_000.0
            elif abs(raw) >= 1e11:
                raw /= 1_000.0
            return raw
        else:
            dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _completed_rows(rows: Sequence[Any] | None) -> list[Any]:
    completed: list[Any] = []
    for row in rows or ():
        if isinstance(row, Mapping):
            if row.get("is_provisional") is True or row.get("is_complete") is False:
                continue
        else:
            if getattr(row, "is_provisional", False) is True:
                continue
            if getattr(row, "is_complete", True) is False:
                continue
        if _bar_epoch(row) is not None:
            completed.append(row)
    return completed


def _latest_epoch(rows: Sequence[Any] | None) -> float | None:
    values = [_bar_epoch(row) for row in _completed_rows(rows)]
    valid = [value for value in values if value is not None]
    return max(valid) if valid else None


def _indicator_rows(runner: Any, symbol: str) -> list[Any]:
    getter = getattr(getattr(runner, "_indicator_engine", None), "get_history", None)
    if not callable(getter):
        return []
    try:
        return list(getter(symbol, field="bars") or [])
    except TypeError:
        try:
            return list(getter(symbol) or [])
        except Exception:  # noqa: BLE001
            return []
    except Exception:  # noqa: BLE001
        return []


def _runner_rows(runner: Any, symbol: str) -> list[Any]:
    history = getattr(runner, "_symbol_history", None)
    if not isinstance(history, Mapping):
        return []
    try:
        return list(history.get(symbol, []) or [])
    except Exception:  # noqa: BLE001
        return []


def _mdm_rows(runner: Any, symbol: str, target: int) -> list[Any]:
    getter = getattr(runner, "_get_mdm_bars", None)
    if not callable(getter):
        return []
    try:
        return list(getter(symbol, target) or [])
    except Exception:  # noqa: BLE001
        return []


def _needs_completed_bar_sync(
    *,
    mdm_rows: Sequence[Any],
    runner_rows: Sequence[Any],
    indicator_rows: Sequence[Any],
) -> bool:
    """Return True only when MDM has completed state not visible downstream."""
    mdm_completed = _completed_rows(mdm_rows)
    if not mdm_completed:
        return False
    runner_completed = _completed_rows(runner_rows)
    indicator_completed = _completed_rows(indicator_rows)
    mdm_latest = _latest_epoch(mdm_completed)
    runner_latest = _latest_epoch(runner_completed)
    indicator_latest = _latest_epoch(indicator_completed)
    if len(runner_completed) < len(mdm_completed) or len(indicator_completed) < len(
        mdm_completed
    ):
        return True
    if mdm_latest is None:
        return False
    return bool(
        runner_latest is None
        or indicator_latest is None
        or runner_latest < mdm_latest
        or indicator_latest < mdm_latest
    )


def _schedule_structural_target(
    runner: Any,
    *,
    symbol: str,
    role: str,
    minimum: int,
    target: int,
    source: str,
) -> None:
    if target <= minimum:
        return
    state = getattr(runner, "_context_structural_request_at", None)
    if not isinstance(state, dict):
        state = {}
        setattr(runner, "_context_structural_request_at", state)
    now = time.monotonic()
    last = float(state.get(symbol, 0.0) or 0.0)
    if now - last < _CONTEXT_TARGET_REQUEST_INTERVAL_SECONDS:
        return
    scheduler = getattr(runner, "_schedule_runtime_history_ensure", None)
    if not callable(scheduler):
        return
    state[symbol] = now
    try:
        scheduler(
            symbol,
            role=role,
            phase="runner_sync",
            reason=source,
            required_bars=minimum,
            target_bars=target,
        )
    except Exception:  # noqa: BLE001 - canonical runner remains fail-closed
        state.pop(symbol, None)


def _sync_context_completed_state(runner: Any, *, source: str) -> None:
    """Mirror newly completed MDM context bars and request structural depth."""
    symbols_getter = getattr(runner, "_active_context_symbols_for_history", None)
    if not callable(symbols_getter):
        return
    try:
        symbols = list(symbols_getter() or [])
    except Exception:  # noqa: BLE001
        return
    minimum = _context_minimum(runner)
    target = max(minimum, _structural_context_target())
    sync = getattr(runner, "_sync_history_from_mdm_cache", None)

    for raw_symbol in symbols:
        symbol = normalize_symbol(str(raw_symbol or "")) or str(raw_symbol or "")
        role = _context_role(runner, symbol)
        if not symbol or role is None:
            continue
        indicator_rows = _indicator_rows(runner, symbol)
        runner_rows = _runner_rows(runner, symbol)
        if not runner_rows:
            runner_rows = indicator_rows
        existing_depth = max(len(indicator_rows), len(runner_rows), minimum)
        # Preserve an already-warm window when ORB is disabled, while ORB keeps
        # the larger session structural target. This avoids shrinking a 100-bar
        # context to the 30-bar SMC minimum during incremental propagation.
        read_target = max(target, existing_depth)
        mdm_rows = _mdm_rows(runner, symbol, read_target)
        mdm_completed = _completed_rows(mdm_rows)

        if _needs_completed_bar_sync(
            mdm_rows=mdm_rows,
            runner_rows=runner_rows,
            indicator_rows=indicator_rows,
        ) and callable(sync) and len(mdm_completed) >= minimum:
            # Ask canonical sync to preserve the current warm depth, but never
            # demand more than MDM currently has. The dynamic cached-read adapter
            # still returns the full ORB window when enabled, so every available
            # completed bar is mirrored without turning the 400-bar target into
            # a false execution-readiness blocker.
            sync_required = max(
                minimum,
                min(len(mdm_completed), existing_depth),
            )
            try:
                sync(
                    symbol,
                    required_bars=sync_required,
                    source=source,
                    request_if_short=False,
                )
            except Exception:  # noqa: BLE001 - existing readiness remains authoritative
                pass

        if len(mdm_completed) < target:
            _schedule_structural_target(
                runner,
                symbol=symbol,
                role=role,
                minimum=minimum,
                target=target,
                source=source,
            )


def _apply_context_history_policy(app_module: Any) -> bool:
    """Permit a 400-bar ORB target without raising the readiness minimum."""
    original = getattr(app_module, "resolve_history_policy", None)
    if not callable(original):
        return False
    if bool(getattr(original, "_context_structural_policy_adapted", False)):
        return True

    @wraps(original)
    def resolve_history_policy(
        ctx: Any,
        symbol: str,
        *,
        role: str,
        phase: str,
        reason: str,
    ) -> Any:
        decision = original(
            ctx,
            symbol,
            role=role,
            phase=phase,
            reason=reason,
        )
        if str(role or "") not in _CONTEXT_ROLES or not _env_enabled(
            "ORB_ENABLED", True
        ):
            return decision
        structural = max(int(decision.required_bars), _structural_context_target())
        role_cap = max(int(decision.role_cap), structural)
        deep_cap = max(int(decision.deep_cap), structural)
        target = int(decision.target_bars)
        # Canonical startup should hydrate the opening-range anchor immediately.
        # Later callers keep their normal target unless they explicitly request
        # the deeper target through the runtime ensurer.
        if str(phase or "") == "startup" and str(reason or "") == "startup_hydration":
            target = max(target, structural)
        return replace(
            decision,
            target_bars=min(target, role_cap),
            role_cap=role_cap,
            deep_cap=deep_cap,
        )

    setattr(resolve_history_policy, "_context_structural_policy_adapted", True)
    setattr(resolve_history_policy, "_context_structural_policy_original", original)
    app_module.resolve_history_policy = resolve_history_policy
    return True


def apply_patches(app_module: Any | None = None) -> bool:
    """Install continuous context-history propagation on production runtime."""
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    if app_module is not None:
        _apply_context_history_policy(app_module)

    if bool(getattr(StrategyRunner, "_context_history_continuity_installed", False)):
        return True

    original_sync_context: Callable[..., Any] = StrategyRunner._sync_context_history_if_cold
    original_on_tick: Callable[..., Any] = StrategyRunner._on_tick

    @wraps(original_sync_context)
    def sync_context_history_if_cold(
        self: Any, *, source: str = "context_history_sync"
    ) -> None:
        # Preserve all existing cold-history behavior and diagnostics first.
        original_sync_context(self, source=source)
        # Then enforce freshness/structural depth even when the symbol is warm
        # by count. This is a local MDM-cache read plus canonical Runner sync;
        # any broker request is scheduled via the existing runtime ensurer.
        _sync_context_completed_state(self, source=source)

    @wraps(original_on_tick)
    def on_tick(self: Any, symbol: str, tick: Mapping[str, Any]) -> Any:
        result = original_on_tick(self, symbol, tick)
        normalized = normalize_symbol(str(symbol or "")) or str(symbol or "")
        if _context_role(self, normalized) is None:
            return result
        state = getattr(self, "_context_history_probe_at", None)
        if not isinstance(state, dict):
            state = {}
            setattr(self, "_context_history_probe_at", state)
        now = time.monotonic()
        last = float(state.get(normalized, 0.0) or 0.0)
        if now - last < _CONTEXT_PROBE_INTERVAL_SECONDS:
            return result
        state[normalized] = now
        try:
            self._sync_context_history_if_cold(source="context_tick_bar_sync")
        except Exception:  # noqa: BLE001 - strategy path remains fail-closed
            pass
        return result

    StrategyRunner._context_history_continuity_original_sync_context = original_sync_context
    StrategyRunner._context_history_continuity_original_on_tick = original_on_tick
    StrategyRunner._sync_context_history_if_cold = sync_context_history_if_cold
    StrategyRunner._on_tick = on_tick
    StrategyRunner._context_history_continuity_installed = True
    return True


__all__ = [
    "_apply_context_history_policy",
    "_needs_completed_bar_sync",
    "_structural_context_target",
    "_sync_context_completed_state",
    "apply_patches",
]
