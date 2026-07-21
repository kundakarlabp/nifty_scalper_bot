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

import math
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import pandas as pd

from nifty_scalper_bot.config.defaults import (
    DEFAULT_OPTION_EXEC_MIN_BARS as _DEFAULT_OPT_MIN_BARS,
)
from nifty_scalper_bot.config.env_utils import parse_float_env
from nifty_scalper_bot.core.active_basket import pick_atm_option_symbols_from_basket
from nifty_scalper_bot.core.history_roles import (
    resolve_symbol_history_role as _shared_resolve_symbol_history_role,
)
from nifty_scalper_bot.data.time_contract import coerce_market_timestamp
from nifty_scalper_bot.execution.readiness import (
    HydrationStatus,
    evaluate_quote_readiness,
    resolve_max_quote_age_seconds,
)
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.market_hours import IST, get_runtime_market_mode
from nifty_scalper_bot.utils.smart_symbol import is_nse_trading_day
from nifty_scalper_bot.utils.symbols import canonical

if TYPE_CHECKING:
    from nifty_scalper_bot.core.app import BotContext

LOGGER = get_logger("nifty_scalper_bot.core.history_readiness")


@dataclass(frozen=True, slots=True)
class HistoryProviderRead:
    bars: tuple[Any, ...]
    count: int
    error_type: str | None
    error_category: str | None

    @property
    def error(self) -> str | None:
        return self.error_category


@dataclass(frozen=True, slots=True)
class _HistoryQuality:
    first_ts_utc: datetime | None
    last_ts_utc: datetime | None
    expected_latest_ts_utc: datetime | None
    market_date_ist: date | None
    latest_bar_age_seconds: float | None
    latest_bar_fresh: bool
    recent_window_contiguous: bool
    missing_minute_count: int
    largest_gap_minutes: int
    invalid_timestamp_count: int
    blockers: tuple[str, ...]


def _read_bars_from_provider(
    provider: Any, symbol: str, *, limit: int = 500
) -> HistoryProviderRead:
    """Read cached OHLC bars from a provider without triggering historical fetch."""
    name = type(provider).__name__ if provider is not None else "missing"
    if provider is None or not symbol:
        return HistoryProviderRead((), 0, None, None)
    fn = getattr(provider, "get_ohlc_bars", None)
    if not callable(fn):
        return HistoryProviderRead((), 0, None, None)
    try:
        bars = tuple(fn(symbol, limit=limit) or ())
        return HistoryProviderRead(bars, len(bars), None, None)
    except TypeError:
        try:
            bars = tuple(fn(symbol) or ())
            return HistoryProviderRead(bars, len(bars), None, None)
        except Exception as exc:  # noqa: BLE001
            error_type = type(exc).__name__
            error_category = "provider_read_failed"
    except Exception as exc:  # noqa: BLE001
        error_type = type(exc).__name__
        error_category = "provider_read_failed"
    LOGGER.warning(
        "HISTORY_PROVIDER_READ_FAILED symbol=%s provider=%s exception_type=%s",
        symbol,
        name,
        error_type,
        extra={
            "event": "HISTORY_PROVIDER_READ_FAILED",
            "symbol": symbol,
            "provider": name,
            "exception_type": error_type,
            "category": error_category,
        },
    )
    return HistoryProviderRead((), 0, error_type, error_category)


def _count_bars_from_provider(provider: Any, symbol: str, *, limit: int = 500) -> int:
    return _read_bars_from_provider(provider, symbol, limit=limit).count


def _get_bars_from_provider(
    provider: Any, symbol: str, *, limit: int = 500
) -> list[Any]:
    return list(_read_bars_from_provider(provider, symbol, limit=limit).bars)


def _clamped_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(float(str(raw).strip())) if raw not in {None, ""} else default
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _coerce_history_timestamp_utc(
    value: Any,
    *,
    now_utc: datetime,
    future_grace_seconds: float = 300.0,
) -> datetime | None:
    """Normalize history timestamps to UTC without using local machine time."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, (int, float)):
            raw = float(value)
            unit = "s"
            if abs(raw) >= 1e17:
                unit = "ns"
            elif abs(raw) >= 1e14:
                unit = "us"
            elif abs(raw) >= 1e11:
                unit = "ms"
            ts = pd.to_datetime(raw, unit=unit, utc=True, errors="coerce")
        else:
            ts = coerce_market_timestamp(value).tz_convert("UTC")
    except Exception:
        return None
    try:
        if pd.isna(ts):
            return None
        out = pd.Timestamp(ts).to_pydatetime().astimezone(timezone.utc)
    except Exception:
        return None
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    now = now_utc.astimezone(timezone.utc)
    if out < datetime(2000, 1, 1, tzinfo=timezone.utc):
        return None
    if out > now + timedelta(seconds=max(float(future_grace_seconds), 0.0)):
        return None
    return out


def _previous_nse_trading_day(day: date) -> date:
    cur = day
    for _ in range(370):
        if is_nse_trading_day(cur):
            return cur
        cur -= timedelta(days=1)
    return day


def _expected_latest_bar_start_utc(
    now_utc: datetime, *, publication_grace_seconds: float
) -> datetime | None:
    """Return expected latest finalized one-minute bar start in UTC."""
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    now = now_utc.astimezone(timezone.utc)
    now_ist = now.astimezone(IST)
    market_open = time(9, 15)
    last_start = time(15, 29)
    grace = timedelta(seconds=max(float(publication_grace_seconds), 0.0))

    def close_for(day: date) -> datetime:
        return datetime.combine(day, last_start, tzinfo=IST).astimezone(timezone.utc)

    if not is_nse_trading_day(now_ist.date()):
        return close_for(_previous_nse_trading_day(now_ist.date() - timedelta(days=1)))
    open_dt = datetime.combine(now_ist.date(), market_open, tzinfo=IST)
    final_dt = datetime.combine(now_ist.date(), last_start, tzinfo=IST)
    effective = now_ist - grace
    if effective < open_dt + timedelta(minutes=1):
        return close_for(_previous_nse_trading_day(now_ist.date() - timedelta(days=1)))
    expected = effective.replace(second=0, microsecond=0) - timedelta(minutes=1)
    if expected < open_dt:
        expected = open_dt
    if expected > final_dt:
        expected = final_dt
    return expected.astimezone(timezone.utc)


def _extract_history_ts(row: Any) -> Any:
    if isinstance(row, Mapping):
        return (
            row.get("timestamp")
            or row.get("start")
            or row.get("date")
            or row.get("time")
        )
    return getattr(row, "timestamp", None) or getattr(row, "start", None)


def _role_stale_blocker(role: str) -> str:
    return {
        "spot": "spot_history_stale",
        "spot_context": "spot_history_stale",
        "futures_context": "futures_context_history_stale",
        "selected_ce": "selected_ce_history_stale",
        "selected_pe": "selected_pe_history_stale",
    }.get(role, f"{role}_history_stale")


def _role_gap_blocker(role: str) -> str:
    return {
        "spot": "spot_history_gap_detected",
        "spot_context": "spot_history_gap_detected",
        "futures_context": "futures_context_history_gap_detected",
        "selected_ce": "selected_ce_history_gap_detected",
        "selected_pe": "selected_pe_history_gap_detected",
    }.get(role, f"{role}_history_gap_detected")


def _evaluate_recent_history_quality(
    bars: Sequence[Any],
    *,
    role: str,
    required_bars: int,
    now_utc: datetime,
    publication_grace_seconds: float,
    max_lag_minutes: int,
    continuity_window_bars: int,
    allowed_missing_minutes: int,
    provider_error: str | None,
) -> _HistoryQuality:
    valid: list[datetime] = []
    invalid = 0
    for row in bars or ():
        ts = _coerce_history_timestamp_utc(_extract_history_ts(row), now_utc=now_utc)
        if ts is None:
            invalid += 1
        else:
            valid.append(ts.replace(second=0, microsecond=0))
    ordered = sorted(set(valid))
    expected = _expected_latest_bar_start_utc(
        now_utc, publication_grace_seconds=publication_grace_seconds
    )
    last = ordered[-1] if ordered else None
    latest_age = (
        (now_utc.astimezone(timezone.utc) - last).total_seconds() if last else None
    )
    latest_fresh = bool(
        last is not None
        and expected is not None
        and last >= expected - timedelta(minutes=max_lag_minutes)
    )
    latest_session_date = ordered[-1].astimezone(IST).date() if ordered else None
    session_bars = [
        ts
        for ts in ordered
        if latest_session_date is not None
        and ts.astimezone(IST).date() == latest_session_date
        and time(9, 15) <= ts.astimezone(IST).time() <= time(15, 29)
    ]
    window_size = max(int(continuity_window_bars), 2)
    window = session_bars[-window_size:]
    missing = 0
    largest_gap = 0
    for left, right in zip(window, window[1:]):
        gap = int((right - left).total_seconds() // 60)
        if gap > 1:
            missing += gap - 1
            largest_gap = max(largest_gap, gap)
    contiguous = missing <= allowed_missing_minutes
    blockers: list[str] = []
    if invalid:
        blockers.append("history_timestamp_invalid")
    if provider_error:
        blockers.append("history_provider_error")
    if not latest_fresh:
        blockers.append(_role_stale_blocker(role))
    if not contiguous:
        blockers.append(_role_gap_blocker(role))
    return _HistoryQuality(
        first_ts_utc=ordered[0] if ordered else None,
        last_ts_utc=last,
        expected_latest_ts_utc=expected,
        market_date_ist=now_utc.astimezone(IST).date(),
        latest_bar_age_seconds=latest_age,
        latest_bar_fresh=latest_fresh,
        recent_window_contiguous=contiguous,
        missing_minute_count=missing,
        largest_gap_minutes=largest_gap,
        invalid_timestamp_count=invalid,
        blockers=tuple(dict.fromkeys(blockers)),
    )


def _bar_timestamp(row: Any) -> datetime | None:
    """Extract a UTC timestamp from a cached bar-like object."""
    value = None
    if isinstance(row, Mapping):
        value = (
            row.get("timestamp")
            or row.get("start")
            or row.get("date")
            or row.get("time")
        )
    else:
        value = getattr(row, "timestamp", None) or getattr(row, "start", None)
    try:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if value:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
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
    raw = (
        _basket_get(basket, "option_symbols", None)
        or _basket_get(basket, "symbols", ())
        or ()
    )
    return [str(sym) for sym in raw if str(sym).endswith(("CE", "PE"))]


def _active_selection_symbols(ctx: "BotContext") -> tuple[str | None, str | None]:
    """Read selected CE/PE directly from active basket without mutating context."""
    basket = getattr(ctx, "active_contract_basket", None) or getattr(
        ctx, "active_trading_universe", None
    )
    ce = (
        str(
            _basket_get(basket, "selected_ce", None)
            or _basket_get(basket, "atm_ce", None)
            or ""
        )
        or None
    )
    pe = (
        str(
            _basket_get(basket, "selected_pe", None)
            or _basket_get(basket, "atm_pe", None)
            or ""
        )
        or None
    )
    return ce, pe


def build_symbol_hydration_status(
    ctx: "BotContext",
    symbol: str | None,
    role: str,
    required_bars: int,
    *,
    now_utc: datetime | None = None,
) -> HydrationStatus:
    """Build the canonical startup hydration status for one symbol.

    This reads only existing MDM/DataHub/Runner/Indicator caches; historical
    fetching remains in scheduled hydration/backfill paths, never tick paths.
    """
    normalized = canonical(str(symbol or "").strip())
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

    mdm_read = _read_bars_from_provider(mdm, normalized)
    datahub_read = (
        _read_bars_from_provider(datahub, normalized)
        if datahub is not None
        else HistoryProviderRead(mdm_read.bars, mdm_read.count, None, None)
    )
    mdm_bars = mdm_read.count
    datahub_bars = datahub_read.count
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
    evaluated_at_utc = now_utc or datetime.now(timezone.utc)
    if evaluated_at_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    evaluated_at_utc = evaluated_at_utc.astimezone(timezone.utc)
    grace = float(_clamped_int_env("HISTORY_PUBLICATION_GRACE_SECONDS", 90, 0, 300))
    max_lag = _clamped_int_env("HISTORY_LATEST_BAR_MAX_LAG_MINUTES", 2, 0, 5)
    continuity_window = _clamped_int_env(
        "HISTORY_CONTINUITY_WINDOW_BARS", 50, required, 500
    )
    allowed_missing = _clamped_int_env(
        "HISTORY_ALLOWED_RECENT_MISSING_MINUTES", 0, 0, 5
    )
    quality = _evaluate_recent_history_quality(
        mdm_read.bars,
        role=role,
        required_bars=required,
        now_utc=evaluated_at_utc,
        publication_grace_seconds=grace,
        max_lag_minutes=max_lag,
        continuity_window_bars=continuity_window,
        allowed_missing_minutes=allowed_missing,
        provider_error=mdm_read.error_category,
    )
    hydration_getter = getattr(mdm, "get_last_hydration_result", None)
    hydration_result = (
        hydration_getter(normalized)
        if mdm is not None and callable(hydration_getter)
        else None
    )
    import_getter = getattr(mdm, "get_last_history_import_result", None)
    import_result = (
        import_getter(normalized)
        if mdm is not None and callable(import_getter)
        else None
    )
    quote = _get_cached_quote(ctx, normalized)
    max_quote_age_s = resolve_max_quote_age_seconds(
        "HYDRATION_LIVE_TICK_MAX_AGE_SECONDS",
        "HYDRATION_LIVE_TICK_MAX_AGE_MS",
        default_seconds=60.0,
    )
    raw_spread = os.getenv("HYDRATION_MAX_SPREAD_PCT")
    if raw_spread is None or not raw_spread.strip():
        raw_spread = os.getenv("MAX_OPTION_SPREAD_PCT")
    max_spread = max(
        0.0,
        parse_float_env(raw_spread, 12.0),
    )
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
    live_tick_fresh = quote_ready.reason not in {
        "quote_age_unknown",
        "quote_stale",
        "timestamp_quality_unusable",
        "quote_missing",
    }
    exchange = normalized.split(":", 1)[0] if ":" in normalized else None
    tradingsymbol = (
        normalized.split(":", 1)[1] if ":" in normalized else normalized or None
    )
    gating_counts = [mdm_bars, runner_bars, indicator_bars]
    blockers: list[str] = []
    if not normalized:
        blockers.append(f"{role}_symbol_missing")
    if token is None and role in {
        "selected_ce",
        "selected_pe",
        "option_context",
        "futures_context",
    }:
        blockers.append(
            "option_token_missing"
            if role.startswith("selected_") or role == "option_context"
            else "futures_token_missing"
        )
    if required > 0 and any(count < required for count in gating_counts):
        blockers.append(f"{role}_history_cold")
    if required > 0:
        blockers.extend(quality.blockers)
    selected_role = role in {"selected_ce", "selected_pe"}
    if selected_role and quote_ready.reason != "ready":
        blockers.append(quote_ready.reason)
    ready_for_evaluation = bool(
        normalized
        and required >= 0
        and all(count >= required for count in gating_counts)
        and (
            required <= 0
            or (
                quality.latest_bar_fresh
                and quality.recent_window_contiguous
                and not mdm_read.error_category
                and quality.invalid_timestamp_count == 0
            )
        )
    )
    ready_for_execution = bool(
        ready_for_evaluation
        and (
            not selected_role
            or (token is not None and tradable_quote and quote_ready.reason == "ready")
        )
    )
    status = HydrationStatus(
        symbol=normalized,
        role=role,
        token=token,
        tradingsymbol=tradingsymbol,
        exchange=exchange,
        required_bars=required,
        historical_rows_returned=int(getattr(hydration_result, "fetched_rows", 0) or 0),
        historical_rows_accepted=int(getattr(import_result, "accepted_rows", 0) or 0),
        fetch_returned_rows=int(getattr(hydration_result, "fetched_rows", 0) or 0),
        import_accepted_new_rows=int(getattr(import_result, "accepted_rows", 0) or 0),
        import_idempotent_rows=int(getattr(import_result, "idempotent_rows", 0) or 0),
        validation_rejected_rows=int(
            getattr(import_result, "validation_rejected_rows", 0) or 0
        ),
        final_cache_rows=mdm_bars,
        latest_import_status=getattr(import_result, "status", None),
        latest_import_reason=str(getattr(import_result, "reason", "") or "") or None,
        latest_import_error=getattr(import_result, "error", None),
        latest_import_at=getattr(import_result, "imported_at", None),
        history_provider_error=mdm_read.error_category,
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
        first_bar_ts=quality.first_ts_utc,
        last_bar_ts=quality.last_ts_utc,
        expected_latest_closed_ts=quality.expected_latest_ts_utc,
        latest_bar_age_seconds=quality.latest_bar_age_seconds,
        latest_bar_fresh=quality.latest_bar_fresh,
        recent_window_contiguous=quality.recent_window_contiguous,
        missing_expected_minute_count=quality.missing_minute_count,
        largest_intraday_gap_minutes=quality.largest_gap_minutes,
        propagation_consistent=bool(
            mdm_bars >= required
            and runner_bars >= required
            and indicator_bars >= required
        ),
        live_merge_applied=False,
    )
    LOGGER.debug(
        "HISTORY_QUALITY_RESULT symbol=%s role=%s latest_bar_fresh=%s "
        "recent_window_contiguous=%s blockers=%s",
        normalized,
        role,
        quality.latest_bar_fresh,
        quality.recent_window_contiguous,
        quality.blockers,
        extra={
            "event": "HISTORY_QUALITY_RESULT",
            "symbol": normalized,
            "role": role,
            "first_bar_ts_utc": (
                quality.first_ts_utc.isoformat() if quality.first_ts_utc else None
            ),
            "last_bar_ts_utc": (
                quality.last_ts_utc.isoformat() if quality.last_ts_utc else None
            ),
            "expected_latest_bar_ts_utc": (
                quality.expected_latest_ts_utc.isoformat()
                if quality.expected_latest_ts_utc
                else None
            ),
            "market_date_ist": (
                quality.market_date_ist.isoformat() if quality.market_date_ist else None
            ),
            "latest_bar_fresh": quality.latest_bar_fresh,
            "recent_window_contiguous": quality.recent_window_contiguous,
            "blocker_reasons": quality.blockers,
        },
    )
    LOGGER.debug(
        "HYDRATION_PROPAGATION_RESULT symbol=%s role=%s required_bars=%s "
        "mdm_bars=%s datahub_bars=%s runner_bars=%s indicator_bars=%s "
        "tradable_quote=%s depth_available=%s ready_for_evaluation=%s "
        "ready_for_execution=%s blockers=%s",
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


def _hydration_status_map(
    ctx: "BotContext", *, required_option_bars: int, required_context_bars: int
) -> dict[str, HydrationStatus]:
    """Build hydration statuses using ActiveContractBasket as selected-symbol SSOT.

    Production safety invariant: a newly selected active basket must beat stale
    legacy ctx.selected_* or runner pending symbols. If the basket rolls ATM from
    24350 to 24400, readiness must immediately evaluate 24400 and block until
    24400 has enough bars/quote/depth; it must never arm stale 24350.
    """
    from nifty_scalper_bot.core.app import (
        ensure_bot_context_runtime_fields,
        get_active_contract_selection,
    )

    ensure_bot_context_runtime_fields(ctx)
    runner = getattr(ctx, "strategy_runner", None)
    basket_obj = (
        getattr(ctx, "active_contract_basket", None)
        or getattr(ctx, "active_trading_universe", None)
        or {}
    )
    selection = get_active_contract_selection(ctx)

    selected_ce = str(
        selection.selected_ce
        or _basket_get(basket_obj, "selected_ce", None)
        or _basket_get(basket_obj, "atm_ce", None)
        or getattr(runner, "_pending_selected_ce", None)
        or getattr(ctx, "selected_ce", None)
        or ""
    )
    selected_pe = str(
        selection.selected_pe
        or _basket_get(basket_obj, "selected_pe", None)
        or _basket_get(basket_obj, "atm_pe", None)
        or getattr(runner, "_pending_selected_pe", None)
        or getattr(ctx, "selected_pe", None)
        or ""
    )
    futures_symbol = str(
        selection.futures_symbol
        or _basket_get(basket_obj, "futures_symbol", None)
        or _basket_get(basket_obj, "future_symbol", None)
        or getattr(ctx, "active_futures_symbol", None)
        or ""
    )
    spot_symbol = str(
        _basket_get(basket_obj, "spot_symbol", None)
        or getattr(ctx, "nifty_symbol", None)
        or "NSE:NIFTY"
    )
    option_symbols = list(
        selection.option_symbols or _basket_option_symbols(basket_obj)
    )

    old_ce = getattr(ctx, "selected_ce", None)
    old_pe = getattr(ctx, "selected_pe", None)
    if (
        selection.selected_ce and old_ce and str(old_ce) != str(selection.selected_ce)
    ) or (
        selection.selected_pe and old_pe and str(old_pe) != str(selection.selected_pe)
    ):
        LOGGER.warning(
            "ACTIVE_SELECTION_DRIFT_BLOCKED old_ce=%s old_pe=%s new_ce=%s "
            "new_pe=%s source=history_readiness",
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
        required = (
            required_context_bars
            if role in {"spot", "futures_context"}
            else required_option_bars
        )
        statuses[sym] = build_symbol_hydration_status(ctx, sym, role, required)
    ctx.hydration_status_by_symbol = {
        sym: status.to_dict() for sym, status in statuses.items()
    }
    ctx.last_hydration_status_at = datetime.now(timezone.utc)
    return statuses


def _status_for_role(
    statuses: Mapping[str, HydrationStatus], role: str
) -> HydrationStatus | None:
    for status in statuses.values():
        if status.role == role:
            return status
    return None


def _count_symbol_bars(ctx: "BotContext", symbol: str | None) -> int:
    """Count hydrated bars for symbol."""
    if not symbol or ctx.market_data_manager is None:
        return 0
    try:
        return len(list(ctx.market_data_manager.get_ohlc_bars(symbol, limit=500) or []))
    except Exception:
        return 0


def _pick_atm_option_symbols_from_basket(
    basket: dict[str, object],
) -> tuple[str | None, str | None]:
    """Compatibility wrapper for basket symbol selection."""
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
    minimum_ready = bool(
        mdm_bars >= required_bars
        and runner_bars >= required_bars
        and indicator_bars >= required_bars
    )
    return HistoryReadiness(
        symbol,
        role,
        required_bars,
        mdm_bars,
        runner_bars,
        indicator_bars,
        minimum_ready,
    )


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

    manager = getattr(ctx, "position_manager", None) or getattr(
        runner, "_position_manager", None
    )
    open_symbols: list[str] = []
    try:
        if manager is not None and callable(
            getattr(manager, "get_open_positions", None)
        ):
            open_symbols = [
                str(getattr(p, "symbol", p)) for p in manager.get_open_positions()
            ]
    except Exception:
        open_symbols = []
    return _shared_resolve_symbol_history_role(
        symbol=symbol,
        selected_ce=_attr(basket, "selected_ce")
        or getattr(runner, "_active_selected_ce", None),
        selected_pe=_attr(basket, "selected_pe")
        or getattr(runner, "_active_selected_pe", None),
        spot_symbol=getattr(ctx, "spot_symbol", None)
        or getattr(runner, "_spot_symbol", None)
        or "NSE:NIFTY",
        futures_symbol=getattr(runner, "_active_futures_symbol", None)
        or getattr(ctx, "active_futures_symbol", None),
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
    if (
        active_ce
        and active_pe
        and (selected_ce != active_ce or selected_pe != active_pe)
    ):
        LOGGER.warning(
            "SELECTED_OPTION_HISTORY_DRIFT_CORRECTED old_ce=%s old_pe=%s "
            "new_ce=%s new_pe=%s source=active_contract_basket",
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
    option_min = int(
        os.getenv(
            "READINESS_OPTION_EXEC_MIN_BARS",
            os.getenv("OPTION_EXECUTION_MIN_BARS", str(_DEFAULT_OPT_MIN_BARS)),
        )
        or _DEFAULT_OPT_MIN_BARS
    )
    required = max(option_min, int(getattr(runner, "_option_required_bars", 0) or 0))

    def _readiness(sym: str | None) -> HistoryReadiness | None:
        if not sym:
            return None
        try:
            mdm_bars = len(mdm.get_ohlc_bars(sym) or []) if mdm is not None else 0
        except Exception:
            mdm_bars = 0
        runner_bars = (
            int(runner.runner_history_count(sym))
            if runner is not None
            and callable(getattr(runner, "runner_history_count", None))
            else 0
        )
        if runner is not None and callable(
            getattr(runner, "indicator_history_count", None)
        ):
            indicator_bars = int(runner.indicator_history_count(sym))
        elif runner is not None and callable(
            getattr(runner, "_history_count_for_symbol", None)
        ):
            indicator_bars = int(runner._history_count_for_symbol(sym))
        else:
            indicator_bars = 0
        return compute_history_readiness(
            symbol=sym,
            role="selected_option",
            required_bars=required,
            mdm_bars=mdm_bars,
            runner_bars=runner_bars,
            indicator_bars=indicator_bars,
        )

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
    return SelectedOptionHistoryReadiness(
        selected_ce, selected_pe, ce, pe, both_ready, blocker
    )


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
    option_min = int(
        os.getenv(
            "READINESS_OPTION_EXEC_MIN_BARS",
            os.getenv("OPTION_EXECUTION_MIN_BARS", str(_DEFAULT_OPT_MIN_BARS)),
        )
        or _DEFAULT_OPT_MIN_BARS
    )
    context_env = int(
        os.getenv(
            "READINESS_CONTEXT_MIN_BARS", os.getenv("CONTEXT_EXECUTION_MIN_BARS", "20")
        )
        or 20
    )
    context_min = max(
        context_env, int(getattr(runner, "_context_required_bars", 0) or 0)
    )
    from nifty_scalper_bot.core.app import _symbol_history_requirement

    generic_required = _symbol_history_requirement(ctx)
    if role == "selected_option":
        required = max(
            option_min, int(getattr(runner, "_option_required_bars", 0) or 0)
        )
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
        "futures_context": int(
            os.getenv("HYDRATION_CAP_FUTURES_CONTEXT", "100") or 100
        ),
        "recovery_or_open_position": int(
            os.getenv("HYDRATION_CAP_RECOVERY", "100") or 100
        ),
    }
    role_cap = _role_caps.get(role, int(os.getenv("HYDRATION_CAP_DEFAULT", "75") or 75))
    _deep_caps = {
        "selected_option": int(
            os.getenv("HYDRATION_DEEP_SELECTED_OPTION", "300") or 300
        ),
        "option_context": int(
            os.getenv("HYDRATION_DEEP_OPTION_CONTEXT", str(role_cap)) or role_cap
        ),
        "spot_context": int(os.getenv("HYDRATION_DEEP_SPOT_CONTEXT", "300") or 300),
        "futures_context": int(
            os.getenv("HYDRATION_DEEP_FUTURES_CONTEXT", "300") or 300
        ),
        "recovery_or_open_position": int(
            os.getenv("HYDRATION_DEEP_RECOVERY", "300") or 300
        ),
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
