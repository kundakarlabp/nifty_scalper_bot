"""Canonical history-readiness and hydration computation (SSOT).

Runtime role:
- Pure computation layer for symbol hydration status and history readiness.
- Owns the canonical readiness functions consumed by app.py's live-arming gate
  and by the runner: build_symbol_hydration_status, _hydration_status_map,
  compute_history_readiness, compute_selected_option_history_readiness,
  resolve_history_policy, and the ctx-adapter resolve_symbol_history_role.

Position in the pipeline:
    core/history_roles.py (pure role SSOT) + execution/readiness.py (contracts)
    -> THIS MODULE (history_readiness.py)
    -> core/app.py readiness/arming gate + strategies/runner.py

Owns / does NOT own:
- Owns: the readiness/hydration *decision* logic (extracted verbatim from app.py
  to keep app.py focused on orchestration).
- Does NOT own: history storage (MarketDataManager), contract selection
  (InstrumentManager), or broker fetching. These functions are pure: they read
  cached bar counts and quotes and decide readiness; they never fetch history.

Safe-edit notes:
- SSOT: readiness gates on mdm/runner/indicator bar counts only — never DataHub
  bars (DataHub owns no history). Keep all four call paths consistent.
- ActiveContractBasket is the selected CE/PE source of truth. Legacy ctx.selected_*
  and runner pending fields are synchronized/read only as fallbacks; they must not
  override a newer active basket.
- compute_selected_option_history_readiness must stay pure (no ensure_history /
  historical_data / reseed / replace_history / ingest calls). An architecture
  test enforces this.
- ensure_bot_context_runtime_fields and _symbol_history_requirement live in
  app.py and are imported lazily here to avoid a circular import.
"""

from __future__ import annotations

import os
from nifty_scalper_bot.config.defaults import DEFAULT_OPTION_EXEC_MIN_BARS as _DEFAULT_OPT_MIN_BARS
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from nifty_scalper_bot.core.active_basket import pick_atm_option_symbols_from_basket
from nifty_scalper_bot.core.history_roles import (
    resolve_symbol_history_role as _shared_resolve_symbol_history_role,
)
from nifty_scalper_bot.execution.readiness import (
    HistoryReadinessPolicy,
    HydrationStatus,
    evaluate_quote_readiness,
)
from nifty_scalper_bot.utils.market_hours import get_runtime_market_mode
from nifty_scalper_bot.utils.logging import get_logger

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nifty_scalper_bot.core.app import BotContext

LOGGER = get_logger("nifty_scalper_bot.core.history_readiness")


def _count_bars_from_provider(provider: Any, symbol: str, *, limit: int = 500) -> int:
    """Count OHLC bars from a provider without fetching history."""
    if provider is None or not symbol:
        return 0
    fn = getattr(provider, "get_ohlc_bars", None)
    if not callable(fn):
        return 0
    try:
        return len(list(fn(symbol, limit=limit) or []))
    except TypeError:
        try:
            return len(list(fn(symbol) or []))
        except Exception:
            return 0
    except Exception:
        return 0


def _get_bars_from_provider(provider: Any, symbol: str, *, limit: int = 500) -> list[Any]:
    """Read cached OHLC bars from a provider without triggering historical fetch."""
    if provider is None or not symbol:
        return []
    fn = getattr(provider, "get_ohlc_bars", None)
    if not callable(fn):
        return []
    try:
        return list(fn(symbol, limit=limit) or [])
    except TypeError:
        try:
            return list(fn(symbol) or [])
        except Exception:
            return []
    except Exception:
        return []


def _bar_timestamp(row: Any) -> datetime | None:
    """Extract a UTC timestamp from a cached bar-like object."""
    value = None
    if isinstance(row, Mapping):
        value = row.get("timestamp") or row.get("start") or row.get("date") or row.get("time")
    else:
        value = getattr(row, "timestamp", None) or getattr(row, "start", None)
    try:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if value:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None
    return None


def _get_cached_quote(ctx: "BotContext", symbol: str) -> Mapping[str, Any]:
    """Return cached quote/tick data without pulling broker APIs."""
    for provider in (getattr(ctx, "data_hub", None), getattr(ctx, "datahub", None)):
        if provider is None:
            continue
        fn = getattr(provider, "get_quote", None)
        if callable(fn):
            try:
                quote = fn(symbol, allow_pull=False)
            except TypeError:
                try:
                    quote = fn(symbol)
                except Exception:
                    quote = None
            except Exception:
                quote = None
            if isinstance(quote, Mapping):
                return quote
    mdm = getattr(ctx, "market_data_manager", None)
    for name in ("get_quote", "get_latest_tick", "get_last_tick"):
        fn = getattr(mdm, name, None)
        if callable(fn):
            try:
                quote = fn(symbol)
            except Exception:
                quote = None
            if isinstance(quote, Mapping):
                return quote
    snap_fn = getattr(mdm, "get_symbol_snapshot", None)
    if callable(snap_fn):
        try:
            snap = snap_fn(symbol)
            if snap is not None:
                return {
                    "ltp": getattr(snap, "ltp", None),
                    "bid": getattr(snap, "bid", None),
                    "ask": getattr(snap, "ask", None),
                    "spread_pct": getattr(snap, "spread_pct", None),
                    "depth_available": getattr(snap, "depth_available", None),
                    "depth": getattr(snap, "depth", None),
                    "tradable_quote": getattr(snap, "tradable_quote", None),
                    "tick_age_s": getattr(snap, "tick_age_s", None),
                }
        except Exception:
            return {}
    return {}


def _basket_get(basket: Any, key: str, default: Any = None) -> Any:
    if basket is None:
        return default
    if isinstance(basket, Mapping):
        return basket.get(key, default)
    return getattr(basket, key, default)


def _basket_option_symbols(basket: Any) -> list[str]:
    raw = _basket_get(basket, "option_symbols", None) or _basket_get(basket, "symbols", ()) or ()
    return [str(sym) for sym in raw if str(sym).endswith(("CE", "PE"))]


def _active_selection_symbols(ctx: "BotContext") -> tuple[str | None, str | None]:
    """Read selected CE/PE directly from active basket without mutating context."""
    basket = getattr(ctx, "active_contract_basket", None) or getattr(ctx, "active_trading_universe", None)
    ce = str(_basket_get(basket, "selected_ce", None) or _basket_get(basket, "atm_ce", None) or "") or None
    pe = str(_basket_get(basket, "selected_pe", None) or _basket_get(basket, "atm_pe", None) or "") or None
    return ce, pe


def build_symbol_hydration_status(
    ctx: "BotContext",
    symbol: str | None,
    role: str,
    required_bars: int,
) -> HydrationStatus:
    """Build the canonical startup hydration status for one symbol.

    This reads only existing MDM/DataHub/Runner/Indicator caches; historical
    fetching remains in scheduled hydration/backfill paths, never tick paths.
    """
    normalized = str(symbol or "").strip()
    required = max(0, int(required_bars or 0))
    mdm = getattr(ctx, "market_data_manager", None)
    datahub = getattr(ctx, "data_hub", None) or getattr(ctx, "datahub", None)
    runner = getattr(ctx, "strategy_runner", None)
    token: int | None = None
    for source in (
        getattr(ctx, "active_symbol_tokens", None),
        getattr(mdm, "_token_by_symbol", None),
        getattr(mdm, "_symbol_to_token", None),
        getattr(datahub, "_token_by_symbol", None),
    ):
        if isinstance(source, Mapping) and normalized in source:
            try:
                token = int(source[normalized])
                break
            except (TypeError, ValueError):
                token = None
    if token is None:
        for provider in (datahub, mdm):
            fn = getattr(provider, "resolve_symbol_token", None)
            if callable(fn):
                try:
                    resolved = fn(normalized)
                    if resolved:
                        token = int(resolved)
                        break
                except (TypeError, ValueError):
                    token = None
                except Exception:
                    token = None

    mdm_bars = _count_bars_from_provider(mdm, normalized)
    datahub_bars = _count_bars_from_provider(datahub, normalized) if datahub is not None else mdm_bars
    runner_bars = 0
    indicator_bars = 0
    if runner is not None:
        history = getattr(runner, "_symbol_history", None)
        if isinstance(history, Mapping):
            runner_bars = len(list(history.get(normalized, []) or []))
        indicator = getattr(runner, "_indicator_engine", None)
        get_history = getattr(indicator, "get_history", None)
        if callable(get_history):
            try:
                indicator_bars = len(list(get_history(normalized) or []))
            except Exception:
                indicator_bars = 0
        if runner_bars == 0 and indicator_bars > 0:
            runner_bars = indicator_bars
    bars_for_ts = _get_bars_from_provider(mdm, normalized) or _get_bars_from_provider(datahub, normalized)
    timestamps = [ts for ts in (_bar_timestamp(row) for row in bars_for_ts) if ts is not None]
    quote = _get_cached_quote(ctx, normalized)
    max_quote_age_s = float(os.getenv("HYDRATION_LIVE_TICK_MAX_AGE_MS", "60000") or 60000) / 1000.0
    max_spread = float(os.getenv("HYDRATION_MAX_SPREAD_PCT", os.getenv("MAX_OPTION_SPREAD_PCT", "12")) or 12)
    quote_ready = evaluate_quote_readiness(
        normalized,
        dict(quote) if isinstance(quote, Mapping) else quote,
        max_spread_pct=max_spread,
        require_fresh=True,
        max_age_s=max_quote_age_s,
    )
    bid = quote_ready.bid
    ask = quote_ready.ask
    spread_pct = quote_ready.spread_pct
    # Top-level bid/ask and depth top-of-book are both canonical quote proof;
    # do not require a separate depth flag unless final order preflight does so.
    depth_available = bool(quote_ready.depth_available or quote_ready.bid_ask_available)
    tradable_quote = bool(quote_ready.tradable_quote_ready)
    live_tick_fresh = quote_ready.reason not in {"quote_age_unknown", "quote_stale", "timestamp_quality_unusable", "quote_missing"}
    exchange = normalized.split(":", 1)[0] if ":" in normalized else None
    tradingsymbol = normalized.split(":", 1)[1] if ":" in normalized else normalized or None
    gating_counts = [mdm_bars, runner_bars, indicator_bars]
    blockers: list[str] = []
    if not normalized:
        blockers.append(f"{role}_symbol_missing")
    if token is None and role in {"selected_ce", "selected_pe", "option_context", "futures_context"}:
        blockers.append("option_token_missing" if role.startswith("selected_") or role == "option_context" else "futures_token_missing")
    if required > 0 and any(count < required for count in gating_counts):
        blockers.append(f"{role}_history_cold")
    selected_role = role in {"selected_ce", "selected_pe"}
    if selected_role and quote_ready.reason != "ready":
        blockers.append(quote_ready.reason)
    ready_for_evaluation = bool(normalized and required >= 0 and all(count >= required for count in gating_counts))
    ready_for_execution = bool(
        ready_for_evaluation
        and (
            not selected_role
            or (
                token is not None
                and tradable_quote
                and quote_ready.reason == "ready"
            )
        )
    )
    status = HydrationStatus(
        symbol=normalized,
        role=role,
        token=token,
        tradingsymbol=tradingsymbol,
        exchange=exchange,
        required_bars=required,
        historical_rows_returned=mdm_bars,
        historical_rows_accepted=mdm_bars,
        mdm_bars=mdm_bars,
        datahub_bars=datahub_bars,
        runner_bars=runner_bars,
        indicator_bars=indicator_bars,
        live_tick_fresh=live_tick_fresh,
        tradable_quote=tradable_quote,
        depth_available=depth_available,
        bid=bid,
        ask=ask,
        spread_pct=spread_pct,
        ready_for_evaluation=ready_for_evaluation,
        ready_for_execution=ready_for_execution,
        blocker_reasons=list(dict.fromkeys(blockers)),
        first_bar_ts=min(timestamps) if timestamps else None,
        last_bar_ts=max(timestamps) if timestamps else None,
        live_merge_applied=bool(mdm_bars and datahub_bars and (mdm_bars == datahub_bars)),
    )
    LOGGER.debug(
        "HYDRATION_PROPAGATION_RESULT symbol=%s role=%s required_bars=%s mdm_bars=%s datahub_bars=%s runner_bars=%s indicator_bars=%s tradable_quote=%s depth_available=%s ready_for_evaluation=%s ready_for_execution=%s blockers=%s",
        status.symbol,
        status.role,
        status.required_bars,
        status.mdm_bars,
        status.datahub_bars,
        status.runner_bars,
        status.indicator_bars,
        status.tradable_quote,
        status.depth_available,
        status.ready_for_evaluation,
        status.ready_for_execution,
        status.blocker_reasons,
        extra={"event": "HYDRATION_PROPAGATION_RESULT", **status.to_dict()},
    )
    return status


def _hydration_status_map(ctx: "BotContext", *, required_option_bars: int, required_context_bars: int) -> dict[str, HydrationStatus]:
    """Build hydration statuses using ActiveContractBasket as selected-symbol SSOT.

    Production safety invariant: a newly selected active basket must beat stale
    legacy ctx.selected_* or runner pending symbols. If the basket rolls ATM from
    24350 to 24400, readiness must immediately evaluate 24400 and block until
    24400 has enough bars/quote/depth; it must never arm stale 24350.
    """
    from nifty_scalper_bot.core.app import ensure_bot_context_runtime_fields, get_active_contract_selection

    ensure_bot_context_runtime_fields(ctx)
    runner = getattr(ctx, "strategy_runner", None)
    basket_obj = getattr(ctx, "active_contract_basket", None) or getattr(ctx, "active_trading_universe", None) or {}
    selection = get_active_contract_selection(ctx)

    selected_ce = str(selection.selected_ce or _basket_get(basket_obj, "selected_ce", None) or _basket_get(basket_obj, "atm_ce", None) or getattr(runner, "_pending_selected_ce", None) or getattr(ctx, "selected_ce", None) or "")
    selected_pe = str(selection.selected_pe or _basket_get(basket_obj, "selected_pe", None) or _basket_get(basket_obj, "atm_pe", None) or getattr(runner, "_pending_selected_pe", None) or getattr(ctx, "selected_pe", None) or "")
    futures_symbol = str(selection.futures_symbol or _basket_get(basket_obj, "futures_symbol", None) or _basket_get(basket_obj, "future_symbol", None) or getattr(ctx, "active_futures_symbol", None) or "")
    spot_symbol = str(_basket_get(basket_obj, "spot_symbol", None) or getattr(ctx, "nifty_symbol", None) or "NSE:NIFTY")
    option_symbols = list(selection.option_symbols or _basket_option_symbols(basket_obj))

    old_ce = getattr(ctx, "selected_ce", None)
    old_pe = getattr(ctx, "selected_pe", None)
    if (selection.selected_ce and old_ce and str(old_ce) != str(selection.selected_ce)) or (selection.selected_pe and old_pe and str(old_pe) != str(selection.selected_pe)):
        LOGGER.warning(
            "ACTIVE_SELECTION_DRIFT_BLOCKED old_ce=%s old_pe=%s new_ce=%s new_pe=%s source=history_readiness",
            old_ce,
            old_pe,
            selection.selected_ce,
            selection.selected_pe,
            extra={
                "event": "ACTIVE_SELECTION_DRIFT_BLOCKED",
                "old_ce": old_ce,
                "old_pe": old_pe,
                "new_ce": selection.selected_ce,
                "new_pe": selection.selected_pe,
                "source": "history_readiness",
            },
        )

    role_by_symbol: dict[str, str] = {spot_symbol: "spot"}
    if futures_symbol:
        role_by_symbol[futures_symbol] = "futures_context"
    if selected_ce:
        role_by_symbol[selected_ce] = "selected_ce"
    if selected_pe:
        role_by_symbol[selected_pe] = "selected_pe"
    for sym in option_symbols:
        role_by_symbol.setdefault(sym, "option_context")
    statuses: dict[str, HydrationStatus] = {}
    for sym, role in role_by_symbol.items():
        if not sym:
            continue
        required = required_context_bars if role in {"spot", "futures_context"} else required_option_bars
        statuses[sym] = build_symbol_hydration_status(ctx, sym, role, required)
    ctx.hydration_status_by_symbol = {sym: status.to_dict() for sym, status in statuses.items()}
    ctx.last_hydration_status_at = datetime.now(timezone.utc)
    return statuses


def _status_for_role(statuses: Mapping[str, HydrationStatus], role: str) -> HydrationStatus | None:
    for status in statuses.values():
        if status.role == role:
            return status
    return None


def _count_symbol_bars(ctx: "BotContext", symbol: str | None) -> int:
    """Count hydrated bars for symbol. Args: ctx/symbol. Returns: bars count. Raises: none."""
    if not symbol or ctx.market_data_manager is None:
        return 0
    try:
        return len(list(ctx.market_data_manager.get_ohlc_bars(symbol, limit=500) or []))
    except Exception:
        return 0


def _pick_atm_option_symbols_from_basket(basket: dict[str, object]) -> tuple[str | None, str | None]:
    """Compatibility wrapper for basket symbol selection. Args: basket. Returns: ce/pe. Raises: none."""
    return pick_atm_option_symbols_from_basket(basket)


@dataclass(frozen=True, slots=True)
class HistoryPolicyDecision:
    """Role/phase history requirements for runtime hydration."""

    role: str
    phase: str
    required_bars: int
    target_bars: int
    allow_broker_fetch: bool
    sync_runner: bool
    priority: int
    minimum_only: bool
    role_cap: int = 75
    deep_cap: int = 300


@dataclass(frozen=True, slots=True)
class RuntimeHistoryResult:
    """Canonical runtime history orchestration result."""

    symbol: str
    role: str
    phase: str
    reason: str
    required_bars: int
    target_bars: int
    mdm_bars: int
    runner_bars: int
    indicator_bars: int
    minimum_ready: bool
    target_ready: bool
    sync_success: bool
    hydration: Any | None
    failure_reason: str | None

    @property
    def ready(self) -> bool:
        return self.minimum_ready and self.sync_success


@dataclass(frozen=True, slots=True)
class HistoryReadiness:
    symbol: str
    role: str
    required_bars: int
    mdm_bars: int
    runner_bars: int
    indicator_bars: int
    minimum_ready: bool


@dataclass(frozen=True, slots=True)
class SelectedOptionHistoryReadiness:
    selected_ce: str | None
    selected_pe: str | None
    ce: HistoryReadiness | None
    pe: HistoryReadiness | None
    both_ready: bool
    blocker: str | None


def compute_history_readiness(
    *,
    symbol: str,
    role: str,
    required_bars: int,
    mdm_bars: int,
    runner_bars: int,
    indicator_bars: int,
) -> HistoryReadiness:
    """Compute current-state history readiness without mutation."""
    minimum_ready = bool(mdm_bars >= required_bars and runner_bars >= required_bars and indicator_bars >= required_bars)
    return HistoryReadiness(symbol, role, required_bars, mdm_bars, runner_bars, indicator_bars, minimum_ready)


def resolve_symbol_history_role(ctx: "BotContext", symbol: str) -> str:
    """Args: ctx, symbol. Returns: canonical history role. Resolution order:
    active basket selected CE/PE -> Runner selected CE/PE -> spot -> active
    future -> open/recovery positions -> option_context fallback. Symbols are
    normalized before comparison; not every CE/PE is 'selected'. Raises: none.
    """
    runner = getattr(ctx, "strategy_runner", None)
    basket = getattr(ctx, "active_contract_basket", None)

    def _attr(obj: Any, name: str) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(name)
        return getattr(obj, name, None)

    manager = getattr(ctx, "position_manager", None) or getattr(runner, "_position_manager", None)
    open_symbols: list[str] = []
    try:
        if manager is not None and callable(getattr(manager, "get_open_positions", None)):
            open_symbols = [str(getattr(p, "symbol", p)) for p in manager.get_open_positions()]
    except Exception:
        open_symbols = []
    return _shared_resolve_symbol_history_role(
        symbol=symbol,
        selected_ce=_attr(basket, "selected_ce") or getattr(runner, "_active_selected_ce", None),
        selected_pe=_attr(basket, "selected_pe") or getattr(runner, "_active_selected_pe", None),
        spot_symbol=getattr(ctx, "spot_symbol", None) or getattr(runner, "_spot_symbol", None) or "NSE:NIFTY",
        futures_symbol=getattr(runner, "_active_futures_symbol", None) or getattr(ctx, "active_futures_symbol", None),
        open_position_symbols=open_symbols,
    )


def compute_selected_option_history_readiness(
    ctx: "BotContext",
    selected_ce: str | None,
    selected_pe: str | None,
) -> SelectedOptionHistoryReadiness:
    """Args: ctx, selected CE/PE. Returns: canonical selected-option history
    readiness built fresh from current MDM/Runner/Indicator counts — never
    carries forward a stale blocker, never hydrates or mutates. This is the one
    function every path computing selected_option_history_cold must use. Raises:
    none.
    """
    active_ce, active_pe = _active_selection_symbols(ctx)
    if active_ce and active_pe and (selected_ce != active_ce or selected_pe != active_pe):
        LOGGER.warning(
            "SELECTED_OPTION_HISTORY_DRIFT_CORRECTED old_ce=%s old_pe=%s new_ce=%s new_pe=%s source=active_contract_basket",
            selected_ce,
            selected_pe,
            active_ce,
            active_pe,
            extra={
                "event": "SELECTED_OPTION_HISTORY_DRIFT_CORRECTED",
                "old_ce": selected_ce,
                "old_pe": selected_pe,
                "new_ce": active_ce,
                "new_pe": active_pe,
                "source": "active_contract_basket",
            },
        )
        selected_ce, selected_pe = active_ce, active_pe

    mdm = getattr(ctx, "market_data_manager", None)
    runner = getattr(ctx, "strategy_runner", None)
    option_min = int(os.getenv("READINESS_OPTION_EXEC_MIN_BARS", os.getenv("OPTION_EXECUTION_MIN_BARS", str(_DEFAULT_OPT_MIN_BARS))) or _DEFAULT_OPT_MIN_BARS)
    required = max(option_min, int(getattr(runner, "_option_required_bars", 0) or 0))

    def _readiness(sym: str | None) -> HistoryReadiness | None:
        if not sym:
            return None
        try:
            mdm_bars = len(mdm.get_ohlc_bars(sym) or []) if mdm is not None else 0
        except Exception:
            mdm_bars = 0
        runner_bars = int(runner.runner_history_count(sym)) if runner is not None and callable(getattr(runner, "runner_history_count", None)) else 0
        if runner is not None and callable(getattr(runner, "indicator_history_count", None)):
            indicator_bars = int(runner.indicator_history_count(sym))
        elif runner is not None and callable(getattr(runner, "_history_count_for_symbol", None)):
            indicator_bars = int(runner._history_count_for_symbol(sym))
        else:
            indicator_bars = 0
        return compute_history_readiness(symbol=sym, role="selected_option", required_bars=required, mdm_bars=mdm_bars, runner_bars=runner_bars, indicator_bars=indicator_bars)

    ce = _readiness(selected_ce)
    pe = _readiness(selected_pe)
    both_ready = bool(ce and pe and ce.minimum_ready and pe.minimum_ready)
    blocker: str | None = None
    if not both_ready:
        if selected_ce and selected_pe:
            ce_ready = bool(ce and ce.minimum_ready)
            pe_ready = bool(pe and pe.minimum_ready)
            if not ce_ready and pe_ready:
                blocker = "selected_ce_history_cold"
            elif ce_ready and not pe_ready:
                blocker = "selected_pe_history_cold"
            else:
                blocker = "selected_option_history_cold"
        else:
            blocker = "selected_option_not_set"
    return SelectedOptionHistoryReadiness(selected_ce, selected_pe, ce, pe, both_ready, blocker)


def resolve_history_policy(
    ctx: "BotContext",
    symbol: str,
    *,
    role: str,
    phase: str,
    reason: str,
) -> HistoryPolicyDecision:
    """Resolve role-aware history policy without changing configured thresholds."""
    runner = getattr(ctx, "strategy_runner", None)
    role = str(role or "option_context")
    phase = str(phase or "dynamic_update")
    option_min = int(os.getenv("READINESS_OPTION_EXEC_MIN_BARS", os.getenv("OPTION_EXECUTION_MIN_BARS", str(_DEFAULT_OPT_MIN_BARS))) or _DEFAULT_OPT_MIN_BARS)
    context_env = int(os.getenv("READINESS_CONTEXT_MIN_BARS", os.getenv("CONTEXT_EXECUTION_MIN_BARS", "20")) or 20)
    context_min = max(context_env, int(getattr(runner, "_context_required_bars", 0) or 0))
    from nifty_scalper_bot.core.app import _symbol_history_requirement

    generic_required = _symbol_history_requirement(ctx)
    if role == "selected_option":
        required = max(option_min, int(getattr(runner, "_option_required_bars", 0) or 0))
        target = max(required, generic_required)
        priority = 10
    elif role in {"spot_context", "futures_context"}:
        required = max(context_min, 1)
        target = max(required, generic_required)
        priority = 5 if role == "spot_context" else 4
    elif role == "recovery_or_open_position":
        required = max(option_min, generic_required)
        target = required
        priority = 9
    else:
        required = max(option_min, 1)
        target = max(required, generic_required)
        priority = 1
    # Only explicit closed sessions block option-context broker fetch.
    # A transient UNKNOWN mode (startup, clock lag, calendar hiccup near open)
    # must NOT keep context strategies cold: UNKNOWN != "OPEN" previously
    # evaluated as closed and suppressed SMC/context votes.
    market_closed_context = (
        role == "option_context"
        and get_runtime_market_mode() in {"PRE_MARKET", "POST_MARKET", "HOLIDAY"}
        and phase != "recovery"
    )
    _role_caps = {
        "selected_option": int(os.getenv("HYDRATION_CAP_SELECTED_OPTION", "75") or 75),
        "option_context": int(os.getenv("HYDRATION_CAP_OPTION_CONTEXT", "50") or 50),
        "spot_context": int(os.getenv("HYDRATION_CAP_SPOT_CONTEXT", "100") or 100),
        "futures_context": int(os.getenv("HYDRATION_CAP_FUTURES_CONTEXT", "100") or 100),
        "recovery_or_open_position": int(os.getenv("HYDRATION_CAP_RECOVERY", "100") or 100),
    }
    role_cap = _role_caps.get(role, int(os.getenv("HYDRATION_CAP_DEFAULT", "75") or 75))
    _deep_caps = {
        "selected_option": int(os.getenv("HYDRATION_DEEP_SELECTED_OPTION", "300") or 300),
        "option_context": int(os.getenv("HYDRATION_DEEP_OPTION_CONTEXT", str(role_cap)) or role_cap),
        "spot_context": int(os.getenv("HYDRATION_DEEP_SPOT_CONTEXT", "300") or 300),
        "futures_context": int(os.getenv("HYDRATION_DEEP_FUTURES_CONTEXT", "300") or 300),
        "recovery_or_open_position": int(os.getenv("HYDRATION_DEEP_RECOVERY", "300") or 300),
    }
    deep_cap = max(role_cap, _deep_caps.get(role, role_cap))
    safety_max = int(os.getenv("HYDRATION_MAX_BARS", "0") or 0)
    if safety_max > 0:
        role_cap = min(role_cap, safety_max)
        deep_cap = min(deep_cap, safety_max)
    target = max(required, min(target, role_cap))
    return HistoryPolicyDecision(
        role=role,
        phase=phase,
        required_bars=required,
        target_bars=target,
        allow_broker_fetch=not market_closed_context,
        sync_runner=True,
        priority=priority,
        minimum_only=False,
        role_cap=role_cap,
        deep_cap=deep_cap,
    )
