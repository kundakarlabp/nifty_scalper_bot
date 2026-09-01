"""Keep non-gating dynamic option history work off the ATM commit path.

Only cold, far option-context additions are deferred. Selected-option roles,
near-ATM candidates, overrides/deep history, and any uncertain classification
continue through the canonical synchronous/fail-closed orchestration.
"""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Mapping

from nifty_scalper_bot.core.active_basket import extract_symbol_strike
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.symbols import normalize_symbol

_LOG = get_logger(__name__)
_PATCH_ATTR = "_dynamic_context_history_deferral_installed"
_TASKS: dict[tuple[int, str], asyncio.Task[Any]] = {}


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _current_spot(ctx: Any) -> float | None:
    mdm = getattr(ctx, "market_data_manager", None)
    getter = getattr(mdm, "get_symbol_snapshot", None)
    if not callable(getter):
        return None
    try:
        snapshot = getter("NSE:NIFTY")
    except Exception:
        return None
    if isinstance(snapshot, Mapping):
        return _positive_float(
            snapshot.get("ltp")
            or snapshot.get("last_price")
            or snapshot.get("price")
        )
    return _positive_float(
        getattr(snapshot, "ltp", None)
        or getattr(snapshot, "last_price", None)
        or getattr(snapshot, "price", None)
    )


def _strike_step(ctx: Any) -> int:
    settings = getattr(ctx, "settings", None)
    option_universe = getattr(settings, "option_universe", None)
    raw = getattr(option_universe, "strike_step", 50)
    try:
        step = int(float(raw))
    except (TypeError, ValueError):
        step = 50
    return max(step, 1)


def _already_selected(ctx: Any, symbol: str) -> bool:
    normalized = normalize_symbol(symbol)
    candidates = {
        normalize_symbol(str(getattr(ctx, "selected_ce", None) or "")),
        normalize_symbol(str(getattr(ctx, "selected_pe", None) or "")),
    }
    runner = getattr(ctx, "strategy_runner", None)
    candidates.update(
        {
            normalize_symbol(str(getattr(runner, "_active_selected_ce", None) or "")),
            normalize_symbol(str(getattr(runner, "_active_selected_pe", None) or "")),
        }
    )
    return normalized in {item for item in candidates if item}


def _safe_far_context_candidate(ctx: Any, symbol: str) -> tuple[bool, float | None, int | None]:
    """Return deferral eligibility only with strong proof the symbol is non-ATM."""
    if _already_selected(ctx, symbol):
        return False, None, None
    spot = _current_spot(ctx)
    strike = extract_symbol_strike(symbol)
    if spot is None or strike is None:
        return False, spot, strike
    step = _strike_step(ctx)
    atm = round(float(spot) / float(step)) * step
    # Keep ATM and the first wing synchronous. A genuinely new selected pair or
    # nearest-contract fallback therefore remains fail-closed even on a large
    # ATM jump. The rolling ±3-strike universe adds its cold edge at >=2 steps.
    return abs(float(strike) - float(atm)) >= float(2 * step), spot, int(atm)


def _current_result(
    app_module: Any,
    ctx: Any,
    symbol: str,
    *,
    role: str,
    phase: str,
    reason: str,
) -> Any:
    policy = app_module.resolve_history_policy(
        ctx, symbol, role=role, phase=phase, reason=reason
    )
    mdm = getattr(ctx, "market_data_manager", None)
    runner = getattr(ctx, "strategy_runner", None)
    try:
        mdm_bars = len(mdm.get_ohlc_bars(symbol) or []) if mdm is not None else 0
    except Exception:
        mdm_bars = 0
    try:
        runner_bars = len((getattr(runner, "_symbol_history", {}) or {}).get(symbol, []) or [])
    except Exception:
        runner_bars = 0
    try:
        counter = getattr(runner, "_history_count_for_symbol", None)
        if callable(counter):
            indicator_bars = int(counter(symbol) or 0)
        else:
            history_getter = getattr(getattr(runner, "_indicator_engine", None), "get_history", None)
            indicator_bars = len(history_getter(symbol) or []) if callable(history_getter) else 0
    except Exception:
        indicator_bars = 0
    readiness = app_module.compute_history_readiness(
        symbol=symbol,
        role=policy.role,
        required_bars=policy.required_bars,
        mdm_bars=mdm_bars,
        runner_bars=runner_bars,
        indicator_bars=indicator_bars,
    )
    failure_reason = None if readiness.minimum_ready else "dynamic_context_hydration_deferred"
    return app_module.RuntimeHistoryResult(
        symbol=symbol,
        role=policy.role,
        phase=policy.phase,
        reason=reason,
        required_bars=policy.required_bars,
        target_bars=policy.target_bars,
        mdm_bars=mdm_bars,
        runner_bars=runner_bars,
        indicator_bars=indicator_bars,
        minimum_ready=readiness.minimum_ready,
        target_ready=bool(
            failure_reason is None
            and mdm_bars >= policy.target_bars
            and runner_bars >= policy.required_bars
            and indicator_bars >= policy.required_bars
        ),
        sync_success=readiness.minimum_ready,
        hydration=None,
        failure_reason=failure_reason,
    )


def apply_app_patch(app_module: Any) -> bool:
    """Defer only cold far-context history while preserving canonical ownership."""
    if bool(getattr(app_module, _PATCH_ATTR, False)):
        return True
    original = getattr(app_module, "ensure_symbol_runtime_history", None)
    if not callable(original):
        raise RuntimeError("ensure_symbol_runtime_history_missing")

    @wraps(original)
    async def ensure_symbol_runtime_history(
        ctx: Any,
        symbol: str,
        *,
        role: str,
        phase: str,
        reason: str,
        required_bars: int | None = None,
        target_bars: int | None = None,
        deep_history: bool = False,
    ) -> Any:
        normalized = normalize_symbol(str(symbol or ""))
        eligible = bool(
            normalized
            and role == "option_context"
            and phase == "dynamic_update"
            and reason == "dynamic_option_universe"
            and required_bars is None
            and target_bars is None
            and not deep_history
        )
        far_context = False
        spot = None
        atm = None
        if eligible:
            far_context, spot, atm = _safe_far_context_candidate(ctx, normalized)
        if not far_context:
            return await original(
                ctx,
                symbol,
                role=role,
                phase=phase,
                reason=reason,
                required_bars=required_bars,
                target_bars=target_bars,
                deep_history=deep_history,
            )

        key = (id(ctx), normalized)
        existing = _TASKS.get(key)
        if existing is None or existing.done():
            task = asyncio.create_task(
                original(
                    ctx,
                    normalized,
                    role=role,
                    phase=phase,
                    reason=reason,
                    required_bars=required_bars,
                    target_bars=target_bars,
                    deep_history=deep_history,
                ),
                name=f"dynamic-context-history-{normalized}",
            )
            _TASKS[key] = task

            def _done(completed: asyncio.Task[Any], *, task_key: tuple[int, str] = key) -> None:
                _TASKS.pop(task_key, None)
                if completed.cancelled():
                    return
                try:
                    completed.result()
                except Exception as exc:  # noqa: BLE001 - background context is non-gating
                    _LOG.warning(
                        "DYNAMIC_CONTEXT_HISTORY_BACKGROUND_FAILED symbol=%s error_type=%s error=%s",
                        normalized,
                        type(exc).__name__,
                        exc,
                        extra={
                            "event": "DYNAMIC_CONTEXT_HISTORY_BACKGROUND_FAILED",
                            "symbol": normalized,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )

            task.add_done_callback(_done)

        current = _current_result(
            app_module,
            ctx,
            normalized,
            role=role,
            phase=phase,
            reason=reason,
        )
        _LOG.info(
            "DYNAMIC_CONTEXT_HISTORY_DEFERRED symbol=%s spot=%s atm=%s strike=%s required_bars=%s current_mdm_bars=%s current_runner_bars=%s current_indicator_bars=%s",
            normalized,
            spot,
            atm,
            extract_symbol_strike(normalized),
            current.required_bars,
            current.mdm_bars,
            current.runner_bars,
            current.indicator_bars,
            extra={
                "event": "DYNAMIC_CONTEXT_HISTORY_DEFERRED",
                "symbol": normalized,
                "spot": spot,
                "atm": atm,
                "strike": extract_symbol_strike(normalized),
                "required_bars": current.required_bars,
                "current_mdm_bars": current.mdm_bars,
                "current_runner_bars": current.runner_bars,
                "current_indicator_bars": current.indicator_bars,
            },
        )
        return current

    app_module.ensure_symbol_runtime_history = ensure_symbol_runtime_history
    setattr(app_module, _PATCH_ATTR, True)
    return True


__all__ = ["apply_app_patch"]
