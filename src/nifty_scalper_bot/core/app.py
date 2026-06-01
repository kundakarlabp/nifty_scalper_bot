"""Core orchestration for the Nifty scalper trading bot.

Runtime role:
- Commits ActiveContractBasket from InstrumentManager.
- Passes basket to MDM, DataHub, runner, StrategyManager.
- Must not generate futures/options symbols.
"""

# ruff: noqa: I001

from __future__ import annotations

import asyncio  # Required for startup reconciliation and background tasks
from collections import OrderedDict
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from importlib import import_module
from importlib import metadata as importlib_metadata
import inspect
import hashlib
import logging
import math
import os
from pathlib import Path
import random
import sqlite3
import threading
import time as time_module
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Iterable,
    Literal,
    Sequence,
    Mapping,
    TypedDict,
    TypeVar,
    cast,
)

import pytz
from nifty_scalper_bot.journal.trade_journal import TradeJournal

from nifty_scalper_bot.config.paths import get_data_dir
from nifty_scalper_bot.core.active_basket import (
    normalize_active_basket_schema,
    pick_atm_option_symbols_from_basket,
)

from nifty_scalper_bot.data.robust_provider import (
    CircuitBreakerConfig,
    RobustDataProvider,
)
from nifty_scalper_bot.infra.watchdog import start_watchdog
from nifty_scalper_bot.instruments.active_contracts import canonical_nifty_future_symbol

LOGGER = logging.getLogger("nifty_scalper_bot.core.app")
SYNC_LOCK = threading.Lock()
instrument_cache_ready = threading.Event()


def _as_bool(value: object, default: bool = False) -> bool:
    """Coerce mixed readiness values to bool. Args: value/default. Returns: bool. Raises: none."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "live", "ready"}
    return default

async def _maybe_await(value: Any) -> Any:
    """Await possibly-awaitable values. Args: value. Returns: resolved value. Raises: none."""
    if inspect.isawaitable(value):
        return await value
    return value



def _build_startup_fingerprint() -> dict[str, str]:
    """Build startup fingerprint metadata. Args: none. Returns: metadata map. Raises: none."""

    version = "unknown"
    try:
        version = importlib_metadata.version("nifty_scalper_bot")
    except Exception:
        version = str(os.getenv("APP_VERSION", "unknown")).strip() or "unknown"

    release_id = str(
        os.getenv("RAILWAY_GIT_COMMIT_SHA")
        or os.getenv("GIT_SHA")
        or os.getenv("RELEASE_ID")
        or ""
    ).strip()
    if not release_id:
        git_head = Path(".git/HEAD")
        if git_head.exists():
            try:
                head_value = git_head.read_text(encoding="utf-8").strip()
                if head_value.startswith("ref:"):
                    ref_name = head_value.split(" ", 1)[1].strip()
                    ref_path = Path(".git") / ref_name
                    if ref_path.exists():
                        release_id = ref_path.read_text(encoding="utf-8").strip()[:12]
                elif head_value:
                    release_id = head_value[:12]
            except Exception:
                release_id = ""
    if not release_id:
        digest = hashlib.sha1(version.encode("utf-8"), usedforsecurity=False).hexdigest()
        release_id = f"cfg-{digest[:12]}"

    return {"version": version, "release": release_id}


def _polling_fallback_degraded(
    *,
    ws_ok: bool,
    lagging: bool,
    futures_fresh: bool,
    options_fresh: bool,
) -> bool:
    """Evaluate fallback degrade state using trading-critical feed health."""

    if not ws_ok:
        return True
    if lagging:
        return True
    if not futures_fresh:
        return True
    if not options_fresh:
        return True
    return False


def _safe_supervisor_call(
    name: str,
    candidate: Any,
    *args: Any,
    default: Any = None,
    **kwargs: Any,
) -> Any:
    """Call supervisor dependencies safely without letting bad attributes spam the loop."""

    if not callable(candidate):
        log_throttled(
            LOGGER,
            f"polling_supervisor_noncallable:{name}",
            "POLLING_SUPERVISOR_NONCALLABLE name=%s type=%s repr=%s"
            % (name, type(candidate).__name__, repr(candidate)[:160]),
            interval_sec=60.0,
            level=logging.WARNING,
            extra={
                "event": "POLLING_SUPERVISOR_NONCALLABLE",
                "callable_name": name,
                "candidate_type": type(candidate).__name__,
                "candidate_repr": repr(candidate)[:160],
            },
        )
        return default
    try:
        return candidate(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - supervisor dependencies must degrade safely
        log_throttled(
            LOGGER,
            f"polling_supervisor_call_failed:{name}:{type(exc).__name__}",
            "POLLING_SUPERVISOR_CALL_FAILED name=%s error_type=%s error=%s"
            % (name, type(exc).__name__, str(exc)),
            interval_sec=60.0,
            level=logging.WARNING,
            extra={
                "event": "POLLING_SUPERVISOR_CALL_FAILED",
                "callable_name": name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return default


async def _polling_failover_supervisor_iteration(
    ctx: Any,
    polling_fallback: Any,
    *,
    quote_stale_ms: int,
    degraded_since: float | None,
    recovered_since: float | None,
    activate_after: float = 3.0,
    recover_cooldown: float = 10.0,
) -> tuple[float | None, float | None]:
    """Run one polling failover supervisor pass. Args: ctx/fallback/state. Returns: state."""

    market_open = bool(_safe_supervisor_call("is_market_open_now", is_market_open_now, default=False))
    spot_symbol = "NSE:NIFTY"
    spot_age_ms = None
    now_mono = time_module.monotonic()
    if not market_open:
        # Off-market: never aggressively activate the REST polling fallback
        # just because the last quote crossed the open-market threshold.
        log_throttled(
            LOGGER,
            f"polling_fallback_skipped:{spot_symbol}:market_closed",
            "POLLING_FALLBACK_SKIPPED reason=market_closed age_ms=%s"
            % spot_age_ms,
            interval_sec=60.0,
            level=logging.DEBUG,
            extra={
                "event": "POLLING_FALLBACK_SKIPPED",
                "reason": "market_closed",
                "age_ms": spot_age_ms,
            },
        )
        degraded_since = None
        if polling_fallback.is_running():
            polling_fallback.set_websocket_mode(True)
            polling_fallback.stop()
        return degraded_since, recovered_since

    ws_ok = bool(_safe_supervisor_call(
        "websocket_manager.is_connected",
        getattr(ctx.websocket_manager, "is_connected", None),
        default=False,
    ))
    feed_health = _safe_supervisor_call(
        "market_data_manager.trading_feed_health",
        getattr(ctx.market_data_manager, "trading_feed_health", None),
        max_age_ms=quote_stale_ms,
        default={},
    )
    if not isinstance(feed_health, Mapping):
        feed_health = {}
    futures_fresh = bool(feed_health.get("futures_fresh"))
    options_fresh = bool(feed_health.get("options_fresh"))
    spot_fresh = bool(feed_health.get("spot_fresh"))
    spot_symbol = str(feed_health.get("spot_symbol") or "NSE:NIFTY")
    spot_age_ms = feed_health.get("spot_age_ms")
    auth_tick_age_raw = _safe_supervisor_call(
        "market_data_manager.data_age_ms",
        getattr(ctx.market_data_manager, "data_age_ms", None),
        default=10**9,
    )
    try:
        auth_tick_age_ms = float(auth_tick_age_raw)
    except (TypeError, ValueError):
        auth_tick_age_ms = float(10**9)
    lagging = auth_tick_age_ms > quote_stale_ms
    degraded = _polling_fallback_degraded(
        ws_ok=ws_ok,
        lagging=lagging,
        futures_fresh=futures_fresh,
        options_fresh=options_fresh,
    )
    if degraded:
        recovered_since = None
        degraded_since = degraded_since or now_mono
        if (
            now_mono - degraded_since >= activate_after
            and not polling_fallback.is_running()
        ):
            if (
                spot_age_ms is not None
                and float(spot_age_ms) <= float(quote_stale_ms)
                and ws_ok
            ):
                log_throttled(
                    LOGGER,
                    f"polling_fallback_skipped:{spot_symbol}:within_spot_stale_threshold",
                    "POLLING_FALLBACK_SKIPPED reason=within_spot_stale_threshold age_ms=%s threshold_ms=%s ws_ok=%s"
                    % (spot_age_ms, quote_stale_ms, ws_ok),
                    interval_sec=60.0,
                    level=logging.INFO,
                    extra={
                        "event": "POLLING_FALLBACK_SKIPPED",
                        "reason": "within_spot_stale_threshold",
                        "age_ms": spot_age_ms,
                        "threshold_ms": quote_stale_ms,
                        "ws_ok": ws_ok,
                    },
                )
                return degraded_since, recovered_since
            LOGGER.warning(
                "POLLING_FALLBACK_ACTIVATE reason=spot_stale age_ms=%s threshold_ms=%s ws_ok=%s lagging=%s",
                spot_age_ms,
                quote_stale_ms,
                ws_ok,
                lagging,
                extra={
                    "event": "POLLING_FALLBACK_ACTIVATE",
                    "reason": (
                        "ws_disconnected"
                        if not ws_ok
                        else "tick_lag"
                        if lagging
                        else "futures_stale"
                        if not futures_fresh
                        else "options_stale"
                    ),
                    "lagging": lagging,
                    "futures_fresh": futures_fresh,
                    "options_fresh": options_fresh,
                    "authoritative_age_ms": auth_tick_age_ms,
                },
            )
            polling_fallback.set_websocket_mode(False)
            polling_fallback.start()
    else:
        if not spot_fresh:
            ctx.market_data_manager.ensure_spot_reference_fresh(
                symbol=spot_symbol,
                stale_after_ms=quote_stale_ms,
            )
            LOGGER.info(
                "poll_fallback_skipped_spot_only_stale symbol=%s age_ms=%s",
                spot_symbol,
                spot_age_ms,
            )
        degraded_since = None
        recovered_since = recovered_since or now_mono
        if (
            now_mono - recovered_since >= recover_cooldown
            and polling_fallback.is_running()
        ):
            LOGGER.info(
                "Polling fallback deactivate (supervisor) after ws recovery cooldown",
                extra={"event": "polling_fallback_deactivated"},
            )
            polling_fallback.set_websocket_mode(True)
            polling_fallback.stop()
    return degraded_since, recovered_since


def _run_sync_locked(operation: Callable[[], Any]) -> Any:
    """Run synchronization-critical broker operations under a process-wide lock."""
    with SYNC_LOCK:
        return operation()


def _symbol_history_requirement(ctx: Any) -> int:
    """Compute per-symbol history requirement. Args: ctx. Returns: bars. Raises: none."""

    reqs = [20, 50]
    strategy_runner = getattr(ctx, "strategy_runner", None)
    market_data_manager = getattr(ctx, "market_data_manager", None)
    if strategy_runner is not None:
        reqs.append(int(getattr(strategy_runner, "_required_candles", 0) or 0))
    if market_data_manager is not None:
        reqs.append(int(getattr(market_data_manager, "_min_required_bars", 0) or 0))
    return max(r for r in reqs if r > 0)


def _history_lookback_minutes(required_bars: int) -> int:
    """Derive hydration lookback window. Args: required bars. Returns: minutes. Raises: none."""

    required = max(1, int(required_bars))
    buffer_bars = max(20, required // 2)
    return max(180, required + buffer_bars)


def _history_lookback_days(required_bars: int) -> int:
    """Convert required minute bars into safe historical lookback days."""
    minutes = _history_lookback_minutes(required_bars)
    return max(2, int(math.ceil(minutes / 375.0)) + 1)


def select_active_option_symbols(
    option_symbols: Sequence[str],
    atm: int | float | str | None = None,
    max_active: int = 6,
) -> list[str]:
    """Select balanced CE/PE symbols near ATM. Args: option_symbols/atm/max_active. Returns: selected symbols. Raises: none."""

    safe_limit = max(1, int(max_active or 1))
    unique_symbols = [str(sym) for sym in dict.fromkeys(option_symbols or ()) if sym]
    if not unique_symbols:
        return []

    try:
        atm_value = int(float(atm)) if atm is not None else None
    except Exception:
        atm_value = None

    parsed: list[tuple[int, int, str, str]] = []
    fallback: list[str] = []
    for rank, symbol in enumerate(unique_symbols):
        try:
            head, tail = symbol.rsplit(":", 1)
            _ = head
            side = "CE" if tail.endswith("CE") else "PE" if tail.endswith("PE") else ""
            if not side:
                fallback.append(symbol)
                continue
            strike_digits = ""
            for char in reversed(tail[:-2]):
                if char.isdigit():
                    strike_digits = char + strike_digits
                elif strike_digits:
                    break
            if not strike_digits:
                fallback.append(symbol)
                continue
            strike = int(strike_digits)
            distance = abs(strike - atm_value) if atm_value is not None else 0
            parsed.append((distance, rank, side, symbol))
        except Exception:
            fallback.append(symbol)

    if not parsed:
        return unique_symbols[:safe_limit]

    ce_sorted = [item for item in sorted(parsed) if item[2] == "CE"]
    pe_sorted = [item for item in sorted(parsed) if item[2] == "PE"]
    selected: list[str] = []
    while (ce_sorted or pe_sorted) and len(selected) < safe_limit:
        if ce_sorted and len(selected) < safe_limit:
            selected.append(ce_sorted.pop(0)[3])
        if pe_sorted and len(selected) < safe_limit:
            selected.append(pe_sorted.pop(0)[3])

    for symbol in fallback + unique_symbols:
        if len(selected) >= safe_limit:
            break
        if symbol not in selected:
            selected.append(symbol)
    return selected[:safe_limit]

def nearest_available_strike(spot: float, strikes: Sequence[float]) -> int:
    """Pick nearest strike from available list. Args: spot, strikes. Returns: strike. Raises: ValueError."""
    valid = sorted({int(float(s)) for s in strikes if float(s) > 0})
    if not valid:
        raise ValueError("No valid strikes available")
    return min(valid, key=lambda strike: (abs(strike - float(spot)), strike))


def _is_selected_trade_symbol(ctx: Any, symbol: str) -> bool:
    """Check if symbol is one of the selected execution symbols. Args: ctx/symbol. Returns: bool. Raises: none."""
    normalized_symbol = canonical(symbol)
    selected_symbols = {
        canonical(value)
        for value in (
            getattr(ctx, "selected_ce", None),
            getattr(ctx, "selected_pe", None),
            getattr(ctx, "atm_ce_symbol", None),
            getattr(ctx, "atm_pe_symbol", None),
        )
        if value
    }
    return normalized_symbol in selected_symbols




def build_active_trading_basket_symbols(ctx: Any, basket: Mapping[str, object]) -> list[str]:
    """Build deterministic active basket. Args: ctx,basket. Returns: ordered symbols. Raises: none."""
    max_active_options = max(2, int(os.getenv("MAX_ACTIVE_OPTION_SYMBOLS", "6") or 6))
    spot = str(basket.get("spot_symbol") or "NSE:NIFTY")
    fut = str(basket.get("futures_symbol") or "")
    selected_ce = str(basket.get("selected_ce") or basket.get("atm_ce") or "")
    selected_pe = str(basket.get("selected_pe") or basket.get("atm_pe") or "")
    atm_strike = basket.get("atm_strike")
    option_symbols = [
        str(s)
        for s in list(basket.get("option_symbols") or basket.get("symbols") or [])
        if str(s).endswith(("CE", "PE"))
    ]
    option_symbols = list(dict.fromkeys(option_symbols))
    near_options = select_active_option_symbols(option_symbols, atm=atm_strike, max_active=max_active_options)
    selected_options: list[str] = []
    for symbol in (selected_ce, selected_pe, *near_options):
        if symbol and symbol not in selected_options:
            selected_options.append(symbol)
        if len(selected_options) >= max_active_options:
            break
    ordered = [s for s in (spot, fut, *selected_options) if s]
    out = list(dict.fromkeys(ordered))
    LOGGER.info(
        "ACTIVE_TRADING_BASKET_SELECTED selected_ce=%s selected_pe=%s atm_strike=%s option_count=%d symbols=%s",
        selected_ce or None,
        selected_pe or None,
        atm_strike,
        len(selected_options),
        out,
    )
    return out


def prioritize_startup_hydration_symbols(
    symbols: Sequence[str],
    atm_ce: str | None,
    atm_pe: str | None,
    futures_symbol: str | None,
) -> list[str]:
    """Prioritize startup symbols. Args: symbols/atm/futures. Returns: ordered symbols. Raises: none."""
    basket = {
        "spot_symbol": "NSE:NIFTY",
        "futures_symbol": futures_symbol or "",
        "selected_ce": atm_ce or "",
        "selected_pe": atm_pe or "",
        "option_symbols": [s for s in symbols if s.endswith(("CE", "PE"))],
        "symbols": list(symbols),
    }
    return build_active_trading_basket_symbols(None, basket)




def sync_symbol_history_to_runner(
    ctx: Any,
    symbol: str,
    required_bars: int,
    reason: str,
) -> dict[str, Any]:
    """Sync MDM history into runner cache. Args: ctx/symbol/required_bars/reason. Returns: sync status map. Raises: none."""

    logger = LOGGER
    mdm = getattr(ctx, 'market_data_manager', None)
    runner = getattr(ctx, 'strategy_runner', None)
    mdm_before = 0
    runner_before = 0
    runner_after = 0
    ingested = 0
    ready = False
    safe_required_bars = max(1, int(required_bars or 1))
    try:
        if mdm is None or runner is None:
            logger.info(
                'SELECTED_OPTION_HYDRATION_SYNC_RESULT symbol=%s mdm_before=%s '
                'runner_before=%s runner_after=%s required=%s ready=%s reason=%s',
                symbol,
                0,
                0,
                0,
                safe_required_bars,
                False,
                reason,
            )
            return {
                'symbol': symbol,
                'mdm_before': 0,
                'runner_before': 0,
                'runner_after': 0,
                'required': safe_required_bars,
                'ready': False,
                'ingested': 0,
            }
        bars = list(mdm.get_ohlc_bars(symbol) or [])
        mdm_before = len(bars)
        engine = getattr(runner, '_indicator_engine', None)
        if engine is not None:
            runner_before = len(engine.get_history(symbol) or [])
        if runner_before < safe_required_bars and mdm_before >= safe_required_bars:
            for bar in bars[-safe_required_bars:]:
                try:
                    runner.ingest_historical_bar({**dict(bar), 'symbol': symbol})
                    ingested += 1
                except Exception as exc:
                    logger.error(
                        'Failure in sync_symbol_history_to_runner: %s',
                        exc,
                        exc_info=True,
                    )
        if engine is not None:
            runner_after = len(engine.get_history(symbol) or [])
        ready = runner_after >= safe_required_bars
        logger.info(
            'RUNNER_HISTORY_INGESTED symbol=%s token=%s bars_ingested=%s source=%s '
            'runner_history_count=%s mdm_history_count=%s',
            symbol,
            None,
            ingested,
            reason,
            runner_after,
            mdm_before,
        )
        logger.info(
            'SELECTED_OPTION_HYDRATION_SYNC_RESULT symbol=%s mdm_before=%s '
            'runner_before=%s runner_after=%s required=%s ready=%s',
            symbol,
            mdm_before,
            runner_before,
            runner_after,
            safe_required_bars,
            ready,
        )
    except Exception as exc:
        logger.error('Failure in sync_symbol_history_to_runner: %s', exc, exc_info=True)
    return {
        'symbol': symbol,
        'mdm_before': mdm_before,
        'runner_before': runner_before,
        'runner_after': runner_after,
        'required': safe_required_bars,
        'ready': ready,
        'ingested': ingested,
    }


def get_runner_history_count(ctx: Any, symbol: str) -> int:
    """Read runner history count for symbol. Args: ctx/symbol. Returns: bars. Raises: none."""
    runner = getattr(ctx, "strategy_runner", None)
    if runner is None:
        return 0
    engine = getattr(runner, "_indicator_engine", None)
    if engine is None:
        return 0
    try:
        return len(engine.get_history(symbol) or [])
    except Exception:
        return 0


def resolve_active_basket_tokens(
    ctx: Any,
    active_basket_symbols: Sequence[str],
    selected_ce: str | None,
    selected_pe: str | None,
) -> dict[str, int]:
    """Resolve tokens for active symbols. Args: ctx/symbols/selected. Returns: symbol token map. Raises: none."""
    token_map: dict[str, int] = {}
    instrument_manager = getattr(ctx, "instrument_manager", None)
    broker_client = getattr(ctx, "broker_client", None)
    for symbol in active_basket_symbols:
        token: int | None = None
        try:
            if str(symbol).startswith("NFO:") and instrument_manager is not None:
                token = int(instrument_manager.get_token(symbol))
            elif broker_client is not None:
                token = int(broker_client.get_instrument_token(symbol))
        except Exception:
            token = None
        if token is None:
            fatal = symbol in {selected_ce, selected_pe}
            LOGGER.error("ACTIVE_BASKET_TOKEN_MISSING symbol=%s fatal_for_live=%s", symbol, fatal)
            continue
        token_map[symbol] = int(token)
        LOGGER.info("ACTIVE_BASKET_TOKEN_RESOLVED symbol=%s token=%s", symbol, int(token))
    LOGGER.info(
        "ACTIVE_BASKET_TOKEN_MAP_READY count=%d selected_ce_token=%s selected_pe_token=%s",
        len(token_map),
        token_map.get(selected_ce or ""),
        token_map.get(selected_pe or ""),
    )
    return token_map


def _gate_runner_symbol_add(
    ctx: Any,
    symbol: str,
    pending_runner_symbols: set[str],
    *,
    token: int | None = None,
    source: str = "startup",
) -> bool:
    """Gate StrategyRunner add by bar readiness. Args: ctx/symbol/pending. Returns: added flag. Raises: none."""

    if not ctx.strategy_runner or not ctx.market_data_manager:
        return False
    runner_bars = 0
    try:
        runner_engine = getattr(ctx.strategy_runner, "_indicator_engine", None)
        if runner_engine is not None:
            runner_bars = len(runner_engine.get_history(symbol) or [])
    except Exception:
        runner_bars = 0
    mdm_bars = len(ctx.market_data_manager.get_ohlc_bars(symbol) or [])
    required = max(20, _symbol_history_requirement(ctx))
    quote_ready = False
    try:
        quote_ready = bool(ctx.market_data_manager.get_symbol_snapshot(symbol))
    except Exception:
        quote_ready = False
    if runner_bars < required and mdm_bars >= required:
        sync_symbol_history_to_runner(ctx, symbol, required, source)
        hydrate_fn = getattr(ctx.strategy_runner, "_hydrate_from_mdm_cache", None)
        if callable(hydrate_fn):
            LOGGER.info(
                "RUNNER_HYDRATION_SYNC_ATTEMPT symbol=%s mdm_bars=%d before_runner_bars=%d",
                symbol,
                mdm_bars,
                runner_bars,
            )
            try:
                hydrate_fn(symbol)
            except Exception:
                LOGGER.exception("RUNNER_HYDRATION_SYNC_EXCEPTION symbol=%s", symbol)
            try:
                runner_engine = getattr(ctx.strategy_runner, "_indicator_engine", None)
                if runner_engine is not None:
                    runner_bars = len(runner_engine.get_history(symbol) or [])
            except Exception:
                runner_bars = 0
            LOGGER.info(
                "RUNNER_HYDRATION_SYNC_RESULT symbol=%s after_runner_bars=%d required=%d history_ready=%s",
                symbol,
                runner_bars,
                required,
                runner_bars >= required,
            )
    effective_bars = min(mdm_bars, runner_bars)
    is_option = symbol.endswith(("CE", "PE"))
    history_ready = effective_bars >= required
    add_ready = bool(history_ready if is_option else (quote_ready or mdm_bars >= 1))
    if not add_ready:
        pending_runner_symbols.add(symbol)
        LOGGER.info("RUNNER_SYMBOL_DEFERRED_UNTIL_READY symbol=%s token=%s runner_bars=%d mdm_bars=%d required_bars=%d source=%s", symbol, token, runner_bars, mdm_bars, required, source)
        return False
    ctx.strategy_runner.add_symbol(symbol)
    if ctx.data_hub is not None and ctx.strategy_runner is not None:
        canonical_symbol = ctx.data_hub._canonical_quote_symbol(symbol)
        if hasattr(ctx.strategy_runner, "has_datahub_subscription") and ctx.strategy_runner.has_datahub_subscription(symbol, token):
            ctx.datahub_runner_subscriptions.add(f"{canonical_symbol}|{token or ''}")
            if token is not None:
                ctx.datahub_runner_subscriptions.add(f"TOKEN:{int(token)}")
    pending_runner_symbols.discard(symbol)
    LOGGER.info(
        "RUNNER_SYMBOL_STATUS symbol=%s token=%s added_to_runner=%s runner_bars=%d mdm_bars=%d required_bars=%d history_ready=%s source=%s reason=%s",
        symbol,
        token,
        True,
        runner_bars,
        mdm_bars,
        required,
        history_ready,
        source,
        "history_ready" if history_ready else ("quote_ready_history_pending" if quote_ready else "hydration_failed"),
        extra={
            "event": "RUNNER_SYMBOL_STATUS",
            "symbol": symbol,
            "token": token,
            "added_to_runner": True,
            "runner_bars": runner_bars,
            "mdm_bars": mdm_bars,
            "required_bars": required,
            "history_ready": history_ready,
            "source": source,
            "reason": "history_ready" if history_ready else ("quote_ready_history_pending" if quote_ready else "hydration_failed"),
        },
    )
    try:
        _emit_option_symbol_pipeline_status(
            ctx,
            symbol=symbol,
            token=token,
            selected=_is_selected_trade_symbol(ctx, symbol),
            hydrated_bars=runner_bars,
            runner_added=True,
            source=source,
            reason="gate_add",
        )
    except Exception:  # pragma: no cover - observability must never raise
        pass
    LOGGER.info("RUNNER_SYMBOL_ADDED_AFTER_READY symbol=%s token=%s source=%s", symbol, token, source)
    return True


def _emit_option_symbol_pipeline_status(
    ctx: Any,
    *,
    symbol: str,
    token: int | None,
    selected: bool,
    hydrated_bars: int | None,
    runner_added: bool,
    source: str,
    reason: str,
) -> None:
    """Emit OPTION_SYMBOL_PIPELINE_STATUS for one symbol across the full pipeline."""
    selected_set = {
        str(getattr(ctx, "selected_ce", "") or ""),
        str(getattr(ctx, "selected_pe", "") or ""),
        str((getattr(ctx, "active_trading_universe", {}) or {}).get("selected_ce") or ""),
        str((getattr(ctx, "active_trading_universe", {}) or {}).get("selected_pe") or ""),
    }
    selected_set.discard("")
    selected = symbol in selected_set or selected
    try:
        import time as _time_module

        trace_id = f"{symbol}-{_time_module.monotonic_ns()}"
    except Exception:  # noqa: BLE001
        trace_id = f"{symbol}-pipeline"
    dh = getattr(ctx, "data_hub", None)
    mdm = getattr(ctx, "market_data_manager", None)
    runner = getattr(ctx, "strategy_runner", None)
    computed_runner_added = bool(runner_added)
    runner_history_count = 0
    datahub_callback_registered = False
    datahub_quote_present = False
    datahub_runner_subscription_registered = False
    datahub_token_callback_registered = False
    datahub_token_quote_present = False
    datahub_mdm_delegate_subscribed = False
    mdm_tracked = False
    mdm_has_subscriber = False
    mdm_active_subscribed = False
    broker_ws_token_requested = False
    broker_ws_token_active = False
    live_tick_seen = False
    last_tick_age_s: float | None = None
    if runner is not None:
        try:
            computed_runner_added = computed_runner_added or (
                symbol in getattr(runner, "_active_symbols", set())
            )
            runner_engine = getattr(runner, "_indicator_engine", None)
            if runner_engine is not None:
                runner_history_count = len(runner_engine.get_history(symbol) or [])
        except Exception:
            pass
    if dh is not None:
        try:
            canonical = dh._canonical_quote_symbol(symbol)
            token_key = int(token) if token is not None else None
            debug_status = None
            if hasattr(dh, "debug_subscription_status"):
                try:
                    debug_status = dh.debug_subscription_status(symbol, token)
                except Exception:
                    debug_status = None
            symbol_callbacks = bool(getattr(dh, "_tick_subscribers", {}).get(canonical))
            token_callbacks = token_key is not None and bool(
                getattr(dh, "_tick_subscribers_by_token", {}).get(token_key)
            )
            if debug_status:
                datahub_callback_registered = bool(
                    debug_status["symbol_callbacks"] > 0
                    or debug_status["token_callbacks"] > 0
                )
                datahub_token_callback_registered = bool(
                    debug_status["token_callbacks"] > 0
                )
                datahub_quote_present = bool(
                    debug_status["quote_present"]
                    or debug_status["token_quote_present"]
                )
                datahub_token_quote_present = bool(
                    debug_status["token_quote_present"]
                )
            else:
                datahub_callback_registered = symbol_callbacks or token_callbacks
                datahub_quote_present = (
                    dh.get_quote(symbol, allow_pull=False) is not None
                    or (
                        token_key is not None
                        and getattr(dh, "get_tick_by_token", lambda _t: None)(token_key)
                        is not None
                    )
                )
                datahub_token_callback_registered = token_callbacks
                datahub_token_quote_present = (
                    token_key is not None
                    and getattr(dh, "get_tick_by_token", lambda _t: None)(token_key)
                    is not None
                )
            datahub_mdm_delegate_subscribed = canonical in getattr(
                dh, "_mdm_subscribed_symbols", set()
            )
            datahub_runner_subscription_registered = (
                canonical in getattr(ctx, "datahub_runner_subscriptions", set())
                or f"{canonical}|{token or ''}"
                in getattr(ctx, "datahub_runner_subscriptions", set())
                or (
                    token is not None
                    and f"TOKEN:{int(token)}"
                    in getattr(ctx, "datahub_runner_subscriptions", set())
                )
            )
        except Exception:
            pass
    if mdm is not None:
        try:
            mdm_tracked = symbol in getattr(mdm, "_tracked_symbols", set())
            mdm_has_subscriber = symbol in getattr(mdm, "_subscribers", {})
            mdm_active_subscribed = symbol in getattr(
                mdm, "_active_subscribed_symbols", set()
            )
            last_tick_map = getattr(mdm, "_last_tick_time", {}) or {}
            last_tick = last_tick_map.get(symbol)
            if isinstance(last_tick, (int, float)) and last_tick > 0:
                live_tick_seen = True
                last_tick_age_s = max(0.0, _time_module.time() - float(last_tick))
            token_candidates = (
                getattr(mdm, "_subscribed_tokens", set()),
                getattr(mdm, "_requested_tokens", set()),
                getattr(mdm, "_active_tokens", set()),
                getattr(mdm, "_desired_tokens", set()),
            )
            if token is not None:
                broker_ws_token_requested = any(token in s for s in token_candidates)
                broker_ws_token_active = any(
                    token in s
                    for s in (
                        getattr(mdm, "_subscribed_tokens", set()),
                        getattr(mdm, "_active_tokens", set()),
                    )
                )
        except Exception:
            pass
    effective_datahub_delivery = (
        datahub_callback_registered
        or datahub_token_callback_registered
        or datahub_runner_subscription_registered
    )
    effective_quote_present = (
        datahub_quote_present
        or datahub_token_quote_present
    )
    LOGGER.info(
        "OPTION_SYMBOL_PIPELINE_STATUS symbol=%s token=%s selected=%s hydrated_bars=%s runner_added=%s runner_history_count=%s datahub_callback_registered=%s datahub_mdm_delegate_subscribed=%s datahub_quote_present=%s datahub_runner_subscription_registered=%s datahub_token_callback_registered=%s datahub_token_quote_present=%s mdm_tracked=%s mdm_has_subscriber=%s mdm_active_subscribed=%s broker_ws_token_requested=%s broker_ws_token_active=%s live_tick_seen=%s last_tick_age_s=%s effective_datahub_delivery=%s effective_quote_present=%s",
        symbol,
        token,
        selected,
        hydrated_bars,
        computed_runner_added,
        runner_history_count,
        datahub_callback_registered,
        datahub_mdm_delegate_subscribed,
        datahub_quote_present,
        datahub_runner_subscription_registered,
        datahub_token_callback_registered,
        datahub_token_quote_present,
        mdm_tracked,
        mdm_has_subscriber,
        mdm_active_subscribed,
        broker_ws_token_requested,
        broker_ws_token_active,
        live_tick_seen,
        last_tick_age_s,
        effective_datahub_delivery,
        effective_quote_present,
        extra={
            "event": "OPTION_SYMBOL_PIPELINE_STATUS",
            "symbol": symbol,
            "token": token,
            "trace_id": trace_id,
            "selected": selected,
            "hydrated_bars": hydrated_bars,
            "runner_added": computed_runner_added,
            "runner_history_count": runner_history_count,
            "datahub_callback_registered": datahub_callback_registered,
            "datahub_mdm_delegate_subscribed": datahub_mdm_delegate_subscribed,
            "datahub_quote_present": datahub_quote_present,
            "datahub_runner_subscription_registered": datahub_runner_subscription_registered,
            "datahub_token_callback_registered": datahub_token_callback_registered,
            "datahub_token_quote_present": datahub_token_quote_present,
            "effective_datahub_delivery": effective_datahub_delivery,
            "effective_quote_present": effective_quote_present,
            "datahub_debug_status": debug_status if "debug_status" in locals() else None,
            "message_bus_tick_owner": getattr(ctx, "message_bus_tick_owner", "data_hub"),
            "mdm_tracked": mdm_tracked,
            "mdm_has_subscriber": mdm_has_subscriber,
            "mdm_active_subscribed": mdm_active_subscribed,
            "broker_ws_token_requested": broker_ws_token_requested,
            "broker_ws_token_active": broker_ws_token_active,
            "live_tick_seen": live_tick_seen,
            "last_tick_age_s": last_tick_age_s,
            "source": source,
            "reason": reason,
        },
    )


def _emit_trading_universe_summary(
    ctx: Any,
    *,
    startup_symbols: Iterable[str] = (),
    phase: str = "startup",
) -> None:
    """Emit TRADING_UNIVERSE_SUMMARY snapshot across runner, DataHub, and MDM."""
    runner = getattr(ctx, "strategy_runner", None)
    mdm = getattr(ctx, "market_data_manager", None)
    dh = getattr(ctx, "data_hub", None)
    startup_list = list(startup_symbols) if startup_symbols else []
    option_symbols = [s for s in startup_list if any(x in s for x in ("CE", "PE"))]
    runner_active_count = 0
    evaluation_seen_count = 0
    try:
        if runner is not None:
            runner_active_count = len(getattr(runner, "_active_symbols", set()))
            evaluation_seen_count = len(
                getattr(runner, "_warmup_complete_logged", set())
            )
    except Exception:  # pragma: no cover - defensive
        pass
    datahub_pending = 0
    datahub_subscribed = 0
    try:
        if dh is not None:
            datahub_pending = len(getattr(dh, "_pending_live_symbols", set()))
            datahub_subscribed = len(
                getattr(dh, "_mdm_subscribed_symbols", set())
            )
    except Exception:  # pragma: no cover - defensive
        pass
    mdm_registered = 0
    live_tick_seen = 0
    try:
        if mdm is not None:
            mdm_registered = len(getattr(mdm, "_token_by_symbol", {}))
            last_tick_map = getattr(mdm, "_last_tick_time", {}) or {}
            live_tick_seen = sum(
                1 for v in last_tick_map.values() if isinstance(v, (int, float)) and v > 0
            )
    except Exception:  # pragma: no cover - defensive
        pass
    LOGGER.info(
        "TRADING_UNIVERSE_SUMMARY phase=%s startup_symbols=%d option_symbols=%d runner_active=%d datahub_pending=%d datahub_subscribed=%d mdm_registered=%d live_tick_seen=%d evaluation_seen=%d",
        phase,
        len(startup_list),
        len(option_symbols),
        runner_active_count,
        datahub_pending,
        datahub_subscribed,
        mdm_registered,
        live_tick_seen,
        evaluation_seen_count,
        extra={
            "event": "TRADING_UNIVERSE_SUMMARY",
            "phase": phase,
            "startup_symbols_count": len(startup_list),
            "option_symbols_count": len(option_symbols),
            "runner_active_symbols_count": runner_active_count,
            "datahub_pending_symbols_count": datahub_pending,
            "datahub_subscribed_symbols_count": datahub_subscribed,
            "mdm_registered_symbols_count": mdm_registered,
            "live_tick_seen_count": live_tick_seen,
            "evaluation_seen_count": evaluation_seen_count,
        },
    )


from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

from nifty_scalper_bot.config.base import AppConfig
from nifty_scalper_bot.config.paths import get_data_dir
from nifty_scalper_bot.config.settings import Settings, get_settings
from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager
from nifty_scalper_bot.core.message_bus import Message, MessageBus, MessageType
from nifty_scalper_bot.core.option_universe import OptionUniverseManager
from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.core.unified_manager import UnifiedManager
from nifty_scalper_bot.core.universe_controller import UniverseController
from nifty_scalper_bot.data import (
    InstrumentUniverseStatus,
)
from nifty_scalper_bot.data.assess_data import assess_datahub_fresh
from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.data.market_data_policy import MarketDataPolicy
from nifty_scalper_bot.data.market_regime import MarketRegimeDetector
from nifty_scalper_bot.data.persistent_state import PersistentStateManager
from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.execution.bracket_manager import (
    BracketManager,
    SupportsCancelOrder,
)

from nifty_scalper_bot.execution.lifecycle_manager import LifecycleManager
from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType
from nifty_scalper_bot.execution.paper_fill_engine import PaperFillEngine
from nifty_scalper_bot.execution.position_manager import ActiveContract, PositionManager
from nifty_scalper_bot.execution.post_fill_monitor import PostFillMonitor
from nifty_scalper_bot.execution.readiness import compute_live_readiness
from nifty_scalper_bot.execution.safe_order_manager import SafeOrderManager
from nifty_scalper_bot.execution.state_tracker import StateTracker
from nifty_scalper_bot.infra.cron_refresh import schedule_instrument_refresh
from nifty_scalper_bot.infra.health import HealthState, create_health_app
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.infra.scheduled_tasks import start_background_tasks
from nifty_scalper_bot.infra.structured_logger import (
    emit_diag,
    setup_structured_logging,
)
from nifty_scalper_bot.notifications.telegram_commands import (
    Services as TelegramCommandServices,
    register_telegram_commands,
)
from nifty_scalper_bot.core.instrument_manager import ActiveContractBasket, InstrumentManager
from nifty_scalper_bot.options.contracts import OptionsContractStore
from nifty_scalper_bot.core.contract_selector import get_atm_contracts
from nifty_scalper_bot.options.strike_selector import StrikeSelector
from nifty_scalper_bot.risk import RiskManager, RiskSnapshot, RiskState
from nifty_scalper_bot.risk.session_gate import build_session_guard
from nifty_scalper_bot.server import selftest_router
from nifty_scalper_bot.shadow.shadow_paper import ShadowPaperTrader
from nifty_scalper_bot.storage import HubStore
from nifty_scalper_bot.strategies.elite_strategies.builder import (
    build_elite_strategies,
    get_strategy_tags as elite_strategy_tags,
)
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.orchestrator import StrategyOrchestrator
from nifty_scalper_bot.strategies.runner import StrategyRunner, StrategyRunnerConfig
from nifty_scalper_bot.streaming import (
    PollingStreamer,
    StreamSupervisor,
)
from nifty_scalper_bot.streaming.websocket_manager import WebSocketManager
from nifty_scalper_bot.utils.config_validation import validate_execution_config
from nifty_scalper_bot.utils.metrics import critical_errors_total
from nifty_scalper_bot.utils.env import (
    coalesce_bool,
    coalesce_float,
    coalesce_int,
    coalesce_str,
    get_bool,
    get_csv,
    get_str,
    normalize_path,
)
from nifty_scalper_bot.utils.errors import ConfigurationError
from nifty_scalper_bot.utils.logging import get_logger, log_throttled, setup_logging
from nifty_scalper_bot.utils.market_hours import (
    MarketState,
    allow_offhours_testing_safe,
    get_market_session_state,
    get_market_state,
    is_market_open_session,
    is_market_open_now,
)
from nifty_scalper_bot.utils.metrics import ensure_multiproc_dir
from nifty_scalper_bot.utils.rate_limiter import RateLimiter
from nifty_scalper_bot.utils.reasons import SOFT, canonical as canonical_reason
from nifty_scalper_bot.utils.symbols import canonical, unique_normalized_symbols

if TYPE_CHECKING:
    from nifty_scalper_bot.notifications.telegram_enhanced import TelegramEnhancedNotifier
    from nifty_scalper_bot.notifications.telegram_controller import TelegramBot
    from nifty_scalper_bot.notifications.telegram_webhook_enhanced import (
        TelegramWebhookController,
    )
    from telegram.ext import Application
else:
    TelegramEnhancedNotifier = Any
    TelegramWebhookController = Any

LOGGER = logging.getLogger("nifty_scalper_bot.core.app")


def _resolve_hydration_market_open_state(logger: logging.Logger = LOGGER) -> bool:
    """Resolve open-session state for hydration/subscription tracking with safe fallback."""
    market_open_now = False
    try:
        market_open_now = bool(is_market_open_session())
    except Exception as exc:  # noqa: BLE001
        market_open_now = False
        logger.warning(
            "MARKET_OPEN_CHECK_FAILED defaulting_closed error=%s",
            exc,
            exc_info=True,
        )
    return market_open_now


def _load_telegram_enhanced_notifier() -> type[Any] | None:
    """Load Telegram notifier lazily. Args: none. Returns: notifier class or None. Raises: none."""
    try:
        from nifty_scalper_bot.notifications.telegram_enhanced import TelegramEnhancedNotifier

        return TelegramEnhancedNotifier
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("TELEGRAM_ENHANCED_IMPORT_UNAVAILABLE reason=%s", exc)
        return None


_ComponentT = TypeVar("_ComponentT")


def _safe_ws_token_count(ctx: BotContext) -> int | str:
    """Resolve WS token diagnostics safely. Args: ctx. Returns: count or unknown. Raises: none."""
    for obj_name, attr_name in [
        ("market_data_manager", "_subscribed_tokens"),
        ("market_data_manager", "_desired_tokens"),
        ("websocket_manager", "_subscribed_tokens"),
        ("websocket_manager", "_tokens"),
        ("broker_client", "ws_tokens"),
    ]:
        obj = getattr(ctx, obj_name, None)
        value = getattr(obj, attr_name, None) if obj is not None else None
        if value is not None:
            try:
                return len(value)
            except Exception:
                continue
    return "unknown"


def _get_current_nifty_futures_symbol() -> str:
    """Deprecated test-only calendar helper; live runtime must use InstrumentManager."""
    if os.getenv("EXECUTION_MODE", os.getenv("MODE", "")).strip().upper() == "LIVE":
        raise RuntimeError("calendar futures generation disabled in LIVE; use InstrumentManager ActiveContractBasket")
    import calendar
    from datetime import datetime, timedelta

    now = datetime.now()
    year = now.year
    month = now.month

    # FIX S13: NIFTY expiry day is Tuesday (not Thursday).
    # Find last Tuesday of current month for futures rollover.
    last_day = calendar.monthrange(year, month)[1]
    expiry_date = datetime(year, month, last_day)
    while expiry_date.weekday() != 1:  # Tuesday = 1
        expiry_date -= timedelta(days=1)

    # If we're past expiry, roll to next month
    if now.date() > expiry_date.date():
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    # Deprecated compatibility formatting for non-live tests only.
    y_str = str(year)[-2:]
    months = [
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    ]
    m_str = months[month - 1]

    return "NFO:" + "NIFTY" + y_str + m_str + "FUT"


def _require_component(component: _ComponentT | None, name: str) -> _ComponentT:
    """Return *component* when present, otherwise raise ``RuntimeError``.

    Args:
        component: Optional component instance to validate.
        name: Human-readable component name for diagnostics.

    Returns:
        _ComponentT: The validated component instance.

    Raises:
        RuntimeError: If ``component`` is ``None``.
    """

    LOGGER.debug(
        "Entered _require_component",
        extra={"event": "require_component", "component": name},
    )
    try:
        if component is None:
            raise RuntimeError(f"{name} is not configured")
        return component
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _require_component: %s",
            exc,
            extra={"event": "require_component_failure", "component": name},
            exc_info=exc,
        )
        raise


_HTTP_APP: FastAPI | None = None
_HTTP_NOTIFIER: TelegramEnhancedNotifier | None = None
_HTTP_CONTROLLER: TelegramWebhookController | None = None
_LATEST_CTX: "BotContext | None" = None


def _resolve_active_futures_for_basket(ctx: BotContext, requested: object | None) -> str:
    """Return authoritative active NIFTY futures symbol for runtime basket."""
    canonical_requested = canonical_nifty_future_symbol(requested)
    if canonical_requested:
        LOGGER.info(
            "ACTIVE_BASKET_FUTURES_REQUESTED_ACCEPTED symbol=%s source=instrument_manager_basket",
            canonical_requested,
            extra={
                "event": "ACTIVE_BASKET_FUTURES_REQUESTED_ACCEPTED",
                "symbol": canonical_requested,
                "source": "instrument_manager_basket",
            },
        )
        return canonical_requested
    mdm = getattr(ctx, "market_data_manager", None)
    for method_name in ("get_active_nifty_future_symbol_cached", "resolve_active_nifty_future_symbol"):
        method = getattr(mdm, method_name, None)
        if not callable(method):
            continue
        try:
            resolved = method()
        except TypeError:
            try:
                resolved = method(now=None)
            except Exception:
                resolved = None
        except Exception:
            resolved = None
        canonical = canonical_nifty_future_symbol(resolved)
        if canonical:
            return canonical

    active_universe = getattr(ctx, "active_trading_universe", {}) or {}
    if isinstance(active_universe, Mapping):
        canonical = canonical_nifty_future_symbol(
            active_universe.get("futures_symbol") or active_universe.get("future_symbol")
        )
        if canonical:
            return canonical

    runner = getattr(ctx, "strategy_runner", None)
    canonical = canonical_nifty_future_symbol(getattr(runner, "_active_futures_symbol", None))
    if canonical:
        return canonical

    strategy_manager = getattr(ctx, "strategy_manager", None)
    canonical = canonical_nifty_future_symbol(getattr(strategy_manager, "_futures_symbol", None))
    if canonical:
        return canonical

    LOGGER.warning(
        "FUTURES_CONTEXT_UNAVAILABLE requested=%s reason=active_future_unresolved",
        requested,
        extra={"event": "FUTURES_CONTEXT_UNAVAILABLE", "requested_symbol": str(requested or ""), "reason": "active_future_unresolved"},
    )
    return ""


class _LifecycleTrackerAdapter:
    """Adapter exposing tracker hooks required by the lifecycle manager."""

    def __init__(self, tracker: StateTracker) -> None:
        """Store state tracker reference for delegation.

        Args:
            tracker: Concrete state tracker implementation.

        Returns:
            None.

        Raises:
            None.
        """

        self._tracker = tracker

    def record_lifecycle_event(
        self, symbol: str, event_type: str, payload: Mapping[str, Any] | None = None
    ) -> None:
        """Forward lifecycle events to the underlying tracker.

        Args:
            symbol: Trading symbol associated with the event.
            event_type: Lifecycle event type identifier.
            payload: Optional metadata describing the event.

        Returns:
            None.

        Raises:
            None. Errors are logged to preserve observability.
        """

        try:
            details = dict(payload or {})
            self._tracker.record_lifecycle_event(symbol, event_type, details)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleTrackerAdapter.record_lifecycle_event: %s",
                exc,
                extra={"event": "lifecycle_adapter_record_error", "symbol": symbol},
                exc_info=exc,
            )

    def get_open_positions(self) -> Iterable[Mapping[str, Any]]:
        """Return open positions from the underlying tracker.

        Args:
            None.

        Returns:
            Iterable[Mapping[str, Any]]: Snapshot of open positions.

        Raises:
            None. Errors are logged and an empty list returned.
        """

        try:
            return list(self._tracker.get_open_positions())
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in LifecycleTrackerAdapter.get_open_positions: %s",
                exc,
                extra={"event": "lifecycle_adapter_positions_error"},
                exc_info=exc,
            )
            return []


try:  # pragma: no cover - optional dependency guard
    import prometheus_client  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - defensive fallback
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    generate_latest = None  # type: ignore[assignment]
    multiprocess = None  # type: ignore[assignment]
    CollectorRegistryRef: type[Any] | None = None
else:
    CONTENT_TYPE_LATEST = getattr(
        prometheus_client,
        "CONTENT_TYPE_LATEST",
        "text/plain; version=0.0.4; charset=utf-8",
    )
    generate_latest = getattr(prometheus_client, "generate_latest", None)
    CollectorRegistryRef = cast(
        type[Any] | None, getattr(prometheus_client, "CollectorRegistry", None)
    )
    try:  # pragma: no cover - optional dependency guard
        multiprocess = import_module("prometheus_client.multiprocess")
    except Exception:  # pragma: no cover - defensive fallback
        multiprocess = None  # type: ignore[assignment]


def _render_prometheus_metrics() -> tuple[str, str]:
    """Render Prometheus exposition payload and media type.

    Args:
        None.

    Returns:
        tuple[str, str]: Tuple containing payload text and media type string.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered _render_prometheus_metrics",
        extra={"event": "render_prometheus_metrics_enter"},
    )
    if generate_latest is None:
        LOGGER.error(
            "Failure in _render_prometheus_metrics: prometheus_client missing",
            extra={"event": "render_prometheus_metrics_missing_client"},
        )
        return "# prometheus_client_unavailable\n", "text/plain; charset=utf-8"
    try:
        registry_obj: Any | None = None
        if CollectorRegistryRef is not None:
            registry_obj = CollectorRegistryRef()
            if multiprocess is not None and os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
                try:
                    multiprocess.MultiProcessCollector(registry_obj)  # type: ignore[attr-defined]
                except Exception as exc:  # noqa: BLE001 - defensive multiprocess init
                    LOGGER.error(
                        "Failure in multiprocess collector setup: %s",
                        exc,
                        extra={"event": "render_prometheus_metrics_multiprocess_error"},
                        exc_info=exc,
                    )
                    registry_obj = None
        payload_bytes = (
            generate_latest(registry_obj)
            if registry_obj is not None
            else generate_latest()
        )
        media_type = CONTENT_TYPE_LATEST or "text/plain; version=0.0.4"
        return payload_bytes.decode("utf-8"), media_type
    except Exception as exc:  # noqa: BLE001 - defensive fallback
        LOGGER.error(
            "Failure in _render_prometheus_metrics: %s",
            exc,
            extra={"event": "render_prometheus_metrics_error"},
            exc_info=exc,
        )
        return "# prometheus_metrics_error\n", "text/plain; charset=utf-8"


def _normalize_broker_positions(snapshot: Any) -> list[Mapping[str, object]]:
    """Normalize broker position snapshot into mappings.

    Args:
        snapshot: Raw payload returned from the broker `get_positions` call.

    Returns:
        A list containing mapping entries for each broker position.

    Raises:
        ValueError: If the payload cannot be interpreted as position mappings.
    """

    LOGGER.debug("Entered _normalize_broker_positions")
    try:
        if snapshot is None:
            LOGGER.info(
                "Condition met: broker_position_snapshot_empty",
                extra={"event": "broker_position_snapshot_empty"},
            )
            return []
        if isinstance(snapshot, Mapping):
            LOGGER.info(
                "Condition met: broker_position_single_entry",
                extra={"event": "broker_position_single_entry"},
            )
            return [cast(Mapping[str, object], snapshot)]
        if isinstance(snapshot, Iterable) and not isinstance(snapshot, (str, bytes)):
            normalized: list[Mapping[str, object]] = []
            for item in snapshot:
                if isinstance(item, Mapping):
                    normalized.append(cast(Mapping[str, object], item))
            LOGGER.info(
                "Condition met: broker_position_snapshot_normalized",
                extra={
                    "event": "broker_position_snapshot_normalized",
                    "entries": len(normalized),
                },
            )
            return normalized
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _normalize_broker_positions: %s",
            exc,
            exc_info=exc,
            extra={"event": "broker_position_snapshot_normalize_failed"},
        )
        raise
    raise ValueError("Unsupported broker position payload shape")


def _fetch_positions_with_retry(
    broker_client: Any,
    *,
    max_attempts: int,
    backoff_min: float,
    backoff_max: float,
    backoff_multiplier: float,
    jitter_fraction: float,
    total_timeout_sec: float = 60.0,
) -> list[Mapping[str, object]]:
    LOGGER.debug("Entered _fetch_positions_with_retry")
    if broker_client is None:
        LOGGER.error(
            "Failure in _fetch_positions_with_retry: broker client missing",
            extra={"event": "broker_position_sync_missing_client"},
        )
        raise ValueError("broker_client is required for position sync")

    get_positions = getattr(broker_client, "get_positions", None)
    if not callable(get_positions):
        LOGGER.error(
            "Failure in _fetch_positions_with_retry: get_positions unavailable",
            extra={"event": "broker_position_sync_missing_getter"},
        )
        raise ValueError("broker_client.get_positions is unavailable")

    # [FIX] Variables defined at correct indentation level
    attempt = 0
    delay = max(backoff_min, 0.0)
    last_error: Exception | None = None
    start_time = time_module.monotonic()

    while (
        attempt < max_attempts
        and (time_module.monotonic() - start_time) < total_timeout_sec
    ):
        attempt += 1
        try:
            # Startup/scheduler/stream loops share this lock to prevent parallel broker sync.
            snapshot = _run_sync_locked(get_positions)
            positions = _normalize_broker_positions(snapshot)
            LOGGER.info(
                "Condition met: broker_position_sync_success",
                extra={
                    "event": "broker_position_sync_success",
                    "attempt": attempt,
                    "positions": len(positions),
                },
            )
            return positions
        except Exception as exc:
            last_error = exc
            LOGGER.error(
                "Failure in _fetch_positions_with_retry: %s",
                exc,
                exc_info=exc,
                extra={
                    "event": "broker_position_sync_attempt_failed",
                    "attempt": attempt,
                },
            )

            if (time_module.monotonic() - start_time) + delay > total_timeout_sec:
                LOGGER.error("Broker position sync timed out")
                break

            sleep_window = min(backoff_max, max(delay, backoff_min))
            jitter_amplitude = max(0.0, jitter_fraction) * sleep_window
            if jitter_amplitude > 0.0:
                sleep_window += random.uniform(-jitter_amplitude, jitter_amplitude)
                sleep_window = max(backoff_min, sleep_window)

            LOGGER.info(
                "Condition met: broker_position_sync_retry_scheduled",
                extra={
                    "event": "broker_position_sync_retry_scheduled",
                    "attempt": attempt + 1,
                    "delay_sec": round(sleep_window, 3),
                },
            )
            time_module.sleep(sleep_window)
            delay = min(backoff_max, max(backoff_min, delay * backoff_multiplier))
            continue

    if last_error is not None:
        raise last_error
    return []


def _sync_data_hub_positions(
    data_hub: Any | None,
    position_manager: Any | None,
    *,
    logger: Any = LOGGER,
) -> None:
    if data_hub is None or position_manager is None:
        return
    try:
        rows = [
            {
                "symbol": pos.symbol,
                "quantity": pos.quantity if pos.side == "LONG" else -pos.quantity,
                "average_price": pos.entry_price,
            }
            for pos in position_manager.get_open_positions()
        ]
        data_hub.replace_positions(rows)
    except Exception as exc:  # noqa: BLE001
        log_throttled(
            logger,
            "data_hub_position_sync_failed",
            "data_hub_position_sync_failed error=%r" % exc,
            interval_sec=30.0,
            level=logging.WARNING,
        )


def _hydrate_positions(
    *,
    position_manager: Any,
    persistent_state: Any,
    broker_client: Any,
    data_hub: Any | None,
    logger: Any = LOGGER,
    max_attempts: int,
    backoff_min: float,
    backoff_max: float,
    backoff_multiplier: float,
    jitter_fraction: float,
    total_timeout_sec: float,
) -> list[Mapping[str, object]] | None:
    persisted_positions = persistent_state.load_positions()
    broker_positions: list[Mapping[str, object]] | None
    try:
        broker_positions = _fetch_positions_with_retry(
            broker_client,
            max_attempts=max_attempts,
            backoff_min=backoff_min,
            backoff_max=backoff_max,
            backoff_multiplier=backoff_multiplier,
            jitter_fraction=jitter_fraction,
            total_timeout_sec=total_timeout_sec,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "broker_position_sync_failed",
            extra={"event": "broker_position_sync_failed", "error": str(exc)},
        )
        broker_positions = None

    if broker_positions is None:
        position_manager.restore_positions(persisted_positions)
    else:
        position_manager.synchronize_with_broker(broker_positions)

    _sync_data_hub_positions(data_hub, position_manager, logger=logger)
    return broker_positions


def get_latest_bot_context() -> "BotContext | None":
    """Return the most recently initialized bot context, if any."""

    return _LATEST_CTX


class PollTick(TypedDict, total=False):
    """Shape of normalized poll ticks consumed by the app."""

    instrument_token: int
    token: int | float | str
    source: str
    symbol: str
    ltp: float
    last_price: float
    close: float


@dataclass(slots=True)
class TradingSessionStatus:
    """Compact view of trading-session readiness."""

    session_valid: bool
    rate_limits_ok: bool
    market_open: bool
    risk_green: bool
    reasons: list[str]
    timestamp: datetime
    override_out_of_hours: bool = False
    fail_reason: str | None = None

    def all_clear(self) -> bool:
        return (
            self.session_valid
            and self.rate_limits_ok
            and self.risk_green
            and (self.market_open or self.override_out_of_hours)
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "market_open": self.market_open,
            "override_out_of_hours": self.override_out_of_hours,
            "session_valid": self.session_valid,
            "rate_limits_ok": self.rate_limits_ok,
            "risk_green": self.risk_green,
            "broker_session_valid": self.session_valid,
            "reasons": list(self.reasons),
            "timestamp": self.timestamp.isoformat(),
        }
        if self.fail_reason:
            payload["session_fail_reason"] = self.fail_reason
        return payload


class TradingSessionGuard:
    """Evaluate whether live trading can be safely enabled."""

    def __init__(
        self,
        *,
        rate_limiter: RateLimiter,
        risk_manager: RiskManager,
        market_open: time = time(9, 15),
        market_close: time = time(15, 30),
        session_max_age_hours: float = 22.0,
        timezone_name: str = "Asia/Kolkata",
        allow_out_of_hours: bool = False,
    ) -> None:
        self._rate_limiter = rate_limiter
        self._risk_manager = risk_manager
        self._market_open = market_open
        self._market_close = market_close
        self._session_max_age = timedelta(hours=session_max_age_hours)
        self._tz = ZoneInfo(timezone_name)
        self._session_validated_at: datetime | None = None
        self._last_status: TradingSessionStatus | None = None
        self._allow_out_of_hours = bool(allow_out_of_hours)
        # Throttle: timestamp of last RISK BLOCK warning outside market hours
        self._risk_block_last_warned: float = 0.0

    def mark_session_valid(self) -> None:
        self._session_validated_at = datetime.now(timezone.utc)

    def reset_session_validation(self) -> None:
        """Invalidate previously marked broker sessions."""

        self._session_validated_at = None
        self._last_status = None

    def evaluate(self) -> TradingSessionStatus:
        now = datetime.now(timezone.utc)
        base_guard = build_session_guard(
            now=now,
            override=self._allow_out_of_hours,
            market_open=self._market_open,
            market_close=self._market_close,
        )
        raw_reasons = base_guard.get("reasons", [])
        reasons: list[str] = []
        if isinstance(raw_reasons, Iterable):
            reasons = [str(reason) for reason in raw_reasons]

        broker_session_valid = False
        if self._session_validated_at is None:
            reasons.append("Broker session not validated")
        else:
            broker_session_valid = (
                now - self._session_validated_at < self._session_max_age
            )

            # [FIX] Auto-refresh stale session (Throttled to prevent Event Loop block)
            if not broker_session_valid:
                _now_ts = time_module.monotonic()
                if not hasattr(self, "_last_profile_refresh") or _now_ts - getattr(self, "_last_profile_refresh", 0.0) > 60.0:
                    self._last_profile_refresh = _now_ts
                    LOGGER.warning(
                        f"⚠️ Broker session stale (Age: {now - self._session_validated_at}). Attempting auto-refresh..."
                    )
                    try:
                        # [FIX] Safely locate the broker client through the MDM or OrderManager
                        ctx = get_latest_bot_context()
                        if ctx:
                            # Try to extract the raw broker client from known context locations
                            raw_client = None
                            if hasattr(ctx, "market_data_manager") and ctx.market_data_manager:
                                raw_client = getattr(ctx.market_data_manager, "_provider", None)
                            elif hasattr(ctx, "order_manager") and ctx.order_manager:
                                raw_client = getattr(ctx.order_manager, "_broker", None)

                            # Unwrap if it's a RobustDataProvider or similar wrapper
                            client = getattr(raw_client, "client", raw_client)
                            
                            # Validate and execute
                            if client and hasattr(client, "get_profile"):
                                client.get_profile()  # Will raise if failed
                                self.mark_session_valid()
                                broker_session_valid = True
                                LOGGER.info("✅ Session auto-refreshed successfully.")
                    except Exception as e:
                        LOGGER.error(f"❌ Session auto-refresh failed: {e}")
                if not broker_session_valid:
                    reasons.append("Broker session stale")

        budgets_ok = True
        snapshot = self._rate_limiter.snapshot()
        for name, bucket in snapshot.items():
            tokens = float(bucket.get("tokens", 0.0))
            capacity = max(float(bucket.get("capacity", 1.0)), 1.0)
            if tokens <= 0.1:
                budgets_ok = False
                reasons.append(f"Rate limit depleted: {name}")
                break
            if tokens < max(1.0, 0.1 * capacity):
                budgets_ok = False
                reasons.append(f"Rate limit low: {name}")
                break

        market_ok = bool(base_guard.get("market_open", False))
        override_active = bool(base_guard.get("override_out_of_hours", False))

        risk_ok = True
        risk_snapshot: RiskSnapshot | None = None
        risk_fail_reason: str | None = None

        if os.getenv("SKIP_APP_RISK", "").lower() == "true":
            risk_ok = True
            LOGGER.warning("⚠️ APP-LEVEL RISK CHECK BYPASSED via SKIP_APP_RISK")
        else:
            try:
                risk_ok = self._risk_manager.is_green()
                if not risk_ok:
                    # ── FIX: single snapshot() call — is_green() already calls
                    # snapshot() internally.  Previously app.py called it 2 more
                    # times (once for log, once for reason), each triggering
                    # _refresh_realized_pnl() and potential broker API calls.
                    risk_snapshot = self._risk_manager.snapshot()

                    # ── FIX: show actual triggered condition via last_rejection
                    # instead of always showing day_loss/max_day_loss which is
                    # often NOT the check that fired (e.g. daily_realized limit
                    # or consecutive_losses are frequently the real trigger).
                    rejection = (
                        risk_snapshot.last_rejection
                        if risk_snapshot is not None
                        else None
                    ) or "unknown"
                    snap_for_log = risk_snapshot

                    # ── FIX: throttle post-market RISK BLOCK spam.
                    # During market hours: log every occurrence (actionable).
                    # Outside market hours: log at most once per hour (market is
                    # closed — repeated identical warnings are pure noise).
                    _now_ts = time_module.monotonic()
                    _should_log = market_ok or (
                        _now_ts - self._risk_block_last_warned >= 3600.0
                    )
                    if _should_log and snap_for_log is not None:
                        self._risk_block_last_warned = _now_ts
                        LOGGER.warning(
                            "⛔ RISK BLOCK [%s]: Breaker=%s | "
                            "DayLoss=%s/%s | Realized=%s/-%s | "
                            "Streak=%s | Cooldown=%ss",
                            rejection,
                            snap_for_log.breaker_tripped,
                            f"{snap_for_log.day_loss:.2f}",
                            f"{snap_for_log.max_day_loss:.2f}",
                            f"{snap_for_log.daily_realized:.2f}",
                            f"{snap_for_log.daily_loss_limit:.2f}",
                            snap_for_log.losses_in_row,
                            f"{snap_for_log.cooldown_remaining:.1f}",
                        )
            except Exception:
                risk_ok = False
                reasons.append("Risk manager unavailable")

        if not risk_ok and "Risk manager unavailable" not in reasons:
            # ── FIX: risk_snapshot already set in is_green() block above —
            # no need for a third snapshot() call here.  Fall back to a fresh
            # call only when is_green() raised an exception (risk_snapshot=None).
            if risk_snapshot is None:
                try:
                    risk_snapshot = self._risk_manager.snapshot()
                except Exception:  # pragma: no cover - defensive
                    risk_snapshot = None
            if risk_snapshot is not None:
                if risk_snapshot.breaker_tripped:
                    risk_fail_reason = "BREAKER"
                elif risk_snapshot.cooldown_remaining > 0:
                    risk_fail_reason = "COOLDOWN"
                else:
                    source = (
                        risk_snapshot.last_rejection
                        or risk_snapshot.breaker_reason
                        or ""
                    )
                    risk_fail_reason = canonical_reason(str(source))
                    if risk_fail_reason == "OK":
                        risk_fail_reason = "RISK_CHECK_FAILED"
            if risk_fail_reason is None:
                risk_fail_reason = "RISK_CHECK_FAILED"
            reasons.append(risk_fail_reason)

        status = TradingSessionStatus(
            session_valid=broker_session_valid,
            rate_limits_ok=budgets_ok,
            market_open=market_ok,
            risk_green=risk_ok,
            reasons=reasons,
            timestamp=now.astimezone(self._tz),
            override_out_of_hours=override_active,
            fail_reason=risk_fail_reason,
        )
        self._last_status = status
        return status

    def snapshot(self) -> dict[str, Any]:
        """Evaluate and return the guard payload as a dictionary."""

        status = self.evaluate()
        payload = status.as_dict()
        payload["rate_limits_ok"] = status.rate_limits_ok
        payload["risk_green"] = status.risk_green
        payload["broker_session_valid"] = status.session_valid
        session_ok = status.all_clear()
        fail_reason = "ok"
        if not session_ok:
            try:
                snapshot = self._risk_manager.snapshot()
            except Exception:  # pragma: no cover - defensive
                snapshot = None
            if snapshot is not None and snapshot.breaker_tripped:
                fail_reason = "BREAKER"
            elif snapshot is not None and snapshot.cooldown_remaining > 0:
                fail_reason = "COOLDOWN"
            elif status.fail_reason:
                fail_reason = status.fail_reason
            else:
                fail_reason = canonical_reason(",".join(status.reasons))
                if fail_reason == "OK":
                    fail_reason = "unknown"
        payload["session_fail_reason"] = fail_reason
        return payload

    def _is_market_open(self, now_utc: datetime) -> bool:
        guard = build_session_guard(
            now=now_utc,
            override=self._allow_out_of_hours,
            market_open=self._market_open,
            market_close=self._market_close,
        )
        return bool(guard.get("market_open", False))

    def allow_live(self) -> tuple[bool, TradingSessionStatus]:
        status = self.evaluate()
        return status.all_clear(), status

    def last_status(self) -> TradingSessionStatus | None:
        return self._last_status

    def allow_out_of_hours(self) -> bool:
        return self._allow_out_of_hours

    def set_allow_out_of_hours(self, allow: bool) -> None:
        """Enable or disable trading outside configured hours."""

        self._allow_out_of_hours = bool(allow)

    def set_trading_window(self, start_hhmm: str, end_hhmm: str) -> None:
        """Update the trading window from HH:MM formatted strings."""

        self._market_open = self._parse_hhmm(start_hhmm, self._market_open)
        self._market_close = self._parse_hhmm(end_hhmm, self._market_close)

    def get_trading_window(self) -> tuple[time, time]:
        """Return the current trading window as ``(open, close)`` times."""

        return self._market_open, self._market_close

    @staticmethod
    def _parse_hhmm(value: str, fallback: time) -> time:
        try:
            clean = (value or "").strip()
            if not clean:
                return fallback

            if ":" not in clean:
                raise ValueError(f"Invalid time format (missing colon): {clean}")

            hour_str, minute_str = clean.split(":", 1)
            hour = int(hour_str)
            minute = int(minute_str)

            if not (0 <= hour <= 23):
                raise ValueError(f"Hour must be 0-23, got {hour}")
            if not (0 <= minute <= 59):
                raise ValueError(f"Minute must be 0-59, got {minute}")

            return time(hour, minute)
        except ValueError as exc:
            LOGGER.error(
                f"Invalid time format '{value}': {exc}",
                extra={"event": "parse_hhmm_invalid", "value": value},
            )
            raise ConfigurationError(f"Invalid time format '{value}': {exc}")
        except Exception as exc:
            LOGGER.warning(
                f"Unexpected error parsing time '{value}': {exc}",
                extra={"event": "parse_hhmm_error", "value": value},
            )
            return fallback


def _resolve_session_reason(
    status: TradingSessionStatus, snapshot: RiskSnapshot | None
) -> tuple[str, bool]:
    """Return canonical session reason and soft-override eligibility."""

    base_reason = canonical(status.fail_reason or ",".join(status.reasons))
    if snapshot is None:
        reason = (
            base_reason if base_reason != "OK" else canonical(",".join(status.reasons))
        )
        return reason, False

    if snapshot.breaker_tripped:
        return "BREAKER", False

    if snapshot.cooldown_remaining > 0:
        reason = "COOLDOWN"
    else:
        source = status.fail_reason or snapshot.last_rejection or ""
        reason = canonical(source)
        if reason == "OK" and snapshot.last_rejection:
            reason = canonical(snapshot.last_rejection)
        if reason == "OK" and status.reasons:
            reason = canonical(",".join(status.reasons))
    if reason == "OK":
        reason = base_reason

    soft_override = (
        status.session_valid
        and status.rate_limits_ok
        and status.market_open
        and reason in SOFT
    )
    return reason if reason else "OK", soft_override


def _telegram_webhook_env_requested() -> bool:
    """Return whether webhook transport was explicitly enabled via environment variables."""

    raw_value = os.getenv("TELEGRAM__WEBHOOK_ENABLED")
    if raw_value is None:
        raw_value = os.getenv("TELEGRAM_WEBHOOK_ENABLED")
    return str(raw_value or "").strip().lower() == "true"


def _telegram_transport_mode(settings: Settings) -> str:
    """Return the configured transport mode for Telegram notifications."""

    notifications = settings.notifications
    if (
        notifications.enabled
        and notifications.webhook_enabled
        and bool((notifications.public_base_url or "").strip())
    ):
        return "webhook"
    return "polling"


def _telegram_requires_http_controller(settings: Settings) -> bool:
    """Return whether Telegram delivery for this process needs HTTP controller wiring."""

    return _telegram_transport_mode(settings) == "webhook"


def get_http_app() -> FastAPI:
    """Return the FastAPI application exposing inbound Telegram webhook."""
    global _HTTP_APP, _HTTP_NOTIFIER, _HTTP_CONTROLLER

    # Thread-safe singleton pattern
    if _HTTP_APP is not None:
        return _HTTP_APP

    settings = get_settings()

    telemetry_logger = get_logger("telegram.bootstrap")

    webhook_env_requested = _telegram_webhook_env_requested()

    if not webhook_env_requested and settings.notifications.webhook_enabled:
        settings.notifications.webhook_enabled = False

    app = FastAPI()
    app.state.ctx_getter = get_latest_bot_context
    _HTTP_APP = app

    try:
        from nifty_scalper_bot.notifications.telegram_enhanced import (
            TelegramEnhancedNotifier as _TelegramEnhancedNotifier,
        )
        from nifty_scalper_bot.notifications.telegram_webhook_enhanced import (
            TelegramWebhookController as _TelegramWebhookController,
            register_webhook as _register_webhook,
        )
    except Exception as exc:
        LOGGER.warning("TELEGRAM_DEPENDENCY_UNAVAILABLE error=%s", exc)
        _HTTP_NOTIFIER = None
        return app

    notifier = _TelegramEnhancedNotifier.from_settings(settings.notifications)
    _HTTP_NOTIFIER = notifier

    _HTTP_CONTROLLER = None
    controller: TelegramWebhookController | None = None
    if settings.notifications.enabled:
        if notifier is None:
            telemetry_logger.warning(
                "telegram_controller_skipped",
                extra={"event": "controller_skipped", "reason": "no_notifier"},
            )
        elif _telegram_requires_http_controller(settings):
            controller = _TelegramWebhookController(
                bot=notifier.bot,
                settings=settings.notifications,
            )
            app.include_router(controller.router)
            _HTTP_CONTROLLER = controller
        else:
            telemetry_logger.info(
                "telegram_http_controller_not_required",
                extra={
                    "event": "telegram_http_controller_not_required",
                    "mode": "polling",
                },
            )
    else:
        telemetry_logger.info(
            "telegram_disabled",
            extra={"event": "telegram_disabled", "reason": "notifications_disabled"},
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics() -> PlainTextResponse:
        """Serve Prometheus metrics payload for observability scrapes.

        Args:
            None.

        Returns:
            PlainTextResponse: Response containing Prometheus metrics payload.

        Raises:
            None.
        """

        telemetry_logger.debug(
            "Entered prometheus_metrics",
            extra={"event": "http_metrics_enter"},
        )
        try:
            payload, media_type = _render_prometheus_metrics()
        except Exception as exc:  # noqa: BLE001 - defensive
            telemetry_logger.error(
                "Failure in prometheus_metrics: %s",
                exc,
                extra={"event": "http_metrics_render_error"},
                exc_info=exc,
            )
            payload = "# prometheus_metrics_render_error\n"
            media_type = "text/plain; charset=utf-8"
        return PlainTextResponse(payload, media_type=media_type)

    @app.get("/health/dataflow", response_class=JSONResponse)
    async def health_dataflow() -> JSONResponse:
        """Check market data flow and strategy execution status."""
        ctx = get_latest_bot_context()
        if not ctx:
            return JSONResponse({"status": "error", "reason": "No active BotContext"})

        mdm = ctx.market_data_manager
        runner = ctx.strategy_runner

        mdm_status = mdm.transport_status if mdm else {}
        runner_ready = runner.is_ready() if runner else False

        symbols = []
        if mdm:
            symbols = list(getattr(mdm, "_symbol_by_token", {}).values())

        return JSONResponse({
            "status": "ok" if mdm_status.get("ws_connected") and runner_ready else "degraded",
            "market_data": {
                "ws_connected": mdm_status.get("ws_connected", False),
                "active_symbols_count": len(symbols),
                "active_symbols": symbols[:20],
                "last_tick_age": mdm_status.get("last_tick_age", -1),
            },
            "strategy": {
                "ready": runner_ready,
                "required_candles": getattr(runner, "_required_candles", 0),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    @app.get("/health", response_class=JSONResponse)
    async def http_health() -> JSONResponse:
        ctx = get_latest_bot_context()

        # ✅ FIX 1: Handle Startup Gracefully (Return 200, not 503)
        # This prevents Railway from killing the bot while it initializes.
        # ✅ FIX: Return 200 during startup
        if not ctx:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "starting",
                    "ready": False,
                    "reason": "Context initializing...",
                },
            )

        # ✅ FIX 2: Comprehensive Component Checks
        checks = {
            "broker": ctx.broker_client is not None
            and ctx.broker_client.is_connected(),
            "position_manager": ctx.position_manager is not None,
            "risk_manager": ctx.risk_manager is not None,
            "data_hub": ctx.data_hub is not None,
        }

        # Optional: Check Risk Breaker if available
        if ctx.risk_manager:
            try:
                # Assuming snapshot() or similar property exists
                checks["risk_breaker_ok"] = not getattr(
                    ctx.risk_manager, "breaker_tripped", False
                )
            except Exception:
                checks["risk_breaker_ok"] = False

        # ✅ FIX 3: Determine Status
        all_healthy = all(checks.values())
        status = "healthy" if all_healthy else "degraded"

        # We return 200 even if degraded so we can see the status JSON.
        # Only return 503 if something catastrophic (like broker down) happens in Production mode.
        # For now, 200 is safer for stability.
        return JSONResponse(
            status_code=200,
            content={
                "status": status,
                "ready": all_healthy,
                "checks": checks,
                "uptime_seconds": int(time_module.monotonic() - _START_TIME),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    @app.on_event("startup")
    async def _startup_webhook() -> None:
        telegram_logger = get_logger("telegram")
        notif_settings = settings.notifications
        if controller is None:
            telegram_logger.info(
                "telegram_webhook_startup_skipped",
                extra={"event": "telegram_webhook_startup_skipped", "mode": "polling"},
            )
            return
        if notif_settings.webhook_enabled and notif_settings.public_base_url:
            registered = await _register_webhook(
                controller.bot,
                notif_settings.public_base_url,
                logger=telegram_logger,
            )
            if registered:
                telegram_logger.info(
                    "telegram_webhook_ready",
                    extra={"event": "webhook_ready", "webhook_ready": True},
                )
            else:
                if notif_settings.allow_poll_fallback:
                    await controller.activate_polling_fallback("webhook setup failed")
                else:
                    telegram_logger.warning(
                        "telegram_webhook_registration_failed",
                        extra={
                            "event": "webhook_failed",
                            "public_url": notif_settings.public_base_url,
                            "fallback_enabled": False,
                        },
                    )
        else:
            if notif_settings.allow_poll_fallback:
                await controller.activate_polling_fallback("webhook url missing")
                telegram_logger.info(
                    "telegram_polling_started_no_webhook",
                    extra={"event": "polling_started", "reason": "no_public_url"},
                )
            else:
                disabled_via_env = (
                    not webhook_env_requested or not notif_settings.webhook_enabled
                )
                webhook_configured = bool(
                    (notif_settings.public_base_url or "").strip()
                )
                extra_payload = {
                    "event": "webhook_not_configured",
                    "public_url_set": webhook_configured,
                    "controller_ready": bool(controller),
                    "enabled": notif_settings.enabled,
                    "fallback_enabled": notif_settings.allow_poll_fallback,
                    "disabled_via_env": disabled_via_env,
                }

                if not notif_settings.webhook_enabled:
                    telegram_logger.info(
                        "Telegram running in polling mode (webhook disabled)",
                        extra=extra_payload,
                    )
                elif not webhook_configured:
                    if webhook_env_requested:
                        telegram_logger.warning(
                            "telegram_webhook_not_configured",
                            extra=extra_payload,
                        )
                    else:
                        telegram_logger.debug(
                            "telegram_webhook_not_configured",
                            extra=extra_payload,
                        )
                else:
                    telegram_logger.debug(
                        "telegram_webhook_configuration_state",
                        extra=extra_payload,
                    )

    # Telegram lifecycle is now managed in startup_sequence and NiftyScalperApp.stop()

    # ----------------------------------------------------------------

    app.include_router(selftest_router)
    return app


def get_telegram_notifier() -> TelegramEnhancedNotifier | None:
    """Return the notifier created for webhook delivery, if any."""

    if _HTTP_NOTIFIER is None and _HTTP_APP is None:
        get_http_app()
    return _HTTP_NOTIFIER


@dataclass(slots=True)
class BotContext:
    """Container for all bot components."""

    settings: Settings
    config: AppConfig
    rate_limiter: RateLimiter
    broker_client: ZerodhaKiteClient
    websocket_client: Any | None
    websocket_manager: WebSocketManager | None
    streamer: Any
    stream_supervisor: StreamSupervisor | None
    polling_fallback_streamer: PollingStreamer | None
    message_bus: MessageBus
    data_hub: DataHub | None = None
    market_data_manager: MarketDataManager | None = None
    market_regime: MarketRegimeDetector | None = None
    market_regime_manager: MarketRegimeManager | None = None
    indicator_engine: IndicatorEngine | None = None
    position_manager: PositionManager | None = None
    risk_manager: RiskManager | None = None
    persistent_state: PersistentStateManager | None = None
    order_manager: OrderManager | None = None
    trade_journal: TradeJournal | None = None
    bracket_manager: Any | None = None
    paper_engine: PaperFillEngine | None = None
    safe_order_manager: SafeOrderManager | None = None
    state_tracker: StateTracker | None = None
    lifecycle_manager: LifecycleManager | None = None
    post_fill_monitor: PostFillMonitor | None = None
    strategy_manager: StrategyManager | None = None
    strategy_runner: StrategyRunner | None = None
    unified_manager: UnifiedManager | None = None
    instrument_manager: InstrumentManager | None = None
    options_store: OptionsContractStore | None = None
    instrument_db: sqlite3.Connection | None = None
    instrument_universe: InstrumentUniverseStatus | None = None
    instrument_refresh_task: asyncio.Task[Any] | None = None
    websocket_enabled: bool = True
    shadow_mode_enabled: bool = False
    shadow_trader: ShadowPaperTrader | None = None
    out_of_hours_override: bool = False
    telegram_bot: "TelegramBot | None" = None
    telegram_application: "Application | None" = None
    telegram_notifier: TelegramEnhancedNotifier | None = None
    health_app: FastAPI | None = None
    session_guard: TradingSessionGuard | None = None
    selfchecker: "RuntimeSelfChecker | None" = None
    startup_spot_refresh_done: bool = False
    startup_spot_listener_registered: bool = False
    underlying_spot_prices: OrderedDict[str, float] = field(
        default_factory=lambda: OrderedDict()
    )
    option_universe: OptionUniverseManager | None = None
    subsystems_started: bool = False
    stream_supervisor_started: bool = False
    margin_engine_data_hub_attached: bool = False
    risk_manager_data_hub_attached: bool = False
    bracket_manager_attached: bool = False
    telegram_wired: bool = False
    background_tasks_started: bool = False
    data_hub_listeners_registered: bool = False
    live_orders_armed: bool = False
    trading_ready: bool = False
    data_observation_ready: bool = False
    data_pipeline_ready: bool = False
    data_hard_ready: bool = False
    mdm_strict_hard_ready: bool = False
    spot_ready: bool = False
    evaluation_ready: bool = False
    execution_ready_by_symbol: dict[str, bool] = field(default_factory=dict)
    selected_ce_exec_ready: bool = False
    selected_pe_exec_ready: bool = False
    context_exec_ready: bool = False
    broker_ready: bool = False
    runner_task: asyncio.Task[Any] | None = None
    readiness_mode: str = "SHADOW"
    effective_mode: str = "SHADOW"
    started_mono: float = field(default_factory=time_module.monotonic)
    live_block_reason: str | None = None
    market_session_state: str | None = None
    quote_api_available: bool = True
    quote_api_error: str | None = None
    # Authoritative live trading universe selected by the WS-spot-first
    # startup pipeline.  When populated, downstream legacy hydration paths
    # must reuse this rather than rebuild with their own (potentially
    # synthetic) spot price.
    selected_ce: str | None = None
    selected_pe: str | None = None
    atm_ce_symbol: str | None = None
    atm_pe_symbol: str | None = None
    active_trading_universe: dict[str, Any] = field(default_factory=dict)
    active_contract_basket: dict[str, object] | None = None
    active_basket_hydration: dict[str, object] | None = None
    active_symbol_tokens: dict[str, int] = field(default_factory=dict)
    message_bus_tick_subscribed: bool = False
    datahub_runner_subscriptions: set[str] = field(default_factory=set)
    execution_locked_symbols: set[str] = field(default_factory=set)
    execution_lock_timestamps: dict[str, datetime] = field(default_factory=dict)
    message_bus_running: bool = False
    deferred_basket_retry_started: bool = False
    deferred_basket_retry_task: asyncio.Task[Any] | None = None
    last_deferred_basket_retry_ts: float = 0.0
    basket_build_lock: asyncio.Lock | None = None
    basket_build_task: asyncio.Task[Any] | None = None
    basket_build_in_progress: bool = False
    basket_build_last_started_mono: float = 0.0
    basket_build_last_completed_mono: float = 0.0
    basket_build_last_error: str | None = None
    startup_phase: str = "created"
    startup_failed: bool = False
    startup_failure_reason: str | None = None
    startup_failure_exception: str | None = None

    def update_spot_price(
        self, underlying: str, price: float, max_size: int = 100
    ) -> None:
        """Update spot price with LRU eviction."""
        self.underlying_spot_prices[underlying] = price
        # Evict oldest entry if exceeds limit
        while len(self.underlying_spot_prices) > max_size:
            self.underlying_spot_prices.popitem(last=False)


class PersistentHeartbeatFlusher:
    """Flush :class:`PersistentStateManager` data on heartbeat cadence.

    Args:
        manager: Persistent state manager to flush.
        interval_sec: Minimum seconds between consecutive flushes.

    Returns:
        None.

    Raises:
        ValueError: If ``interval_sec`` is non-positive.
    """

    def __init__(
        self, manager: PersistentStateManager, interval_sec: float = 5.0
    ) -> None:
        self._logger = LOGGER
        self._manager = manager
        self._interval = max(float(interval_sec), 0.0)
        if self._interval <= 0.0:
            raise ValueError("interval_sec must be positive")
        self._last_flush = 0.0

    def handle_heartbeat(self, timestamp: float | None) -> None:
        """Flush persistent state when the heartbeat advances enough.

        Args:
            timestamp: Monotonic timestamp captured at the heartbeat.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PersistentHeartbeatFlusher.handle_heartbeat",
            extra={"event": "persistent_heartbeat_handle"},
        )
        now = float(timestamp) if timestamp is not None else time_module.monotonic()
        if self._last_flush > 0.0 and (now - self._last_flush) < self._interval:
            return
        flush_started = time_module.monotonic()
        try:
            self._manager.flush()
        except Exception as exc:  # noqa: BLE001
            flush_latency = time_module.monotonic() - flush_started
            try:
                METRICS.record_heartbeat_flush(
                    success=False,
                    latency_seconds=flush_latency,
                    now=time_module.time(),
                )
            except Exception as metrics_exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in PersistentHeartbeatFlusher.flush metrics: %s",
                    metrics_exc,
                    extra={"event": "persistent_heartbeat_flush_metric_error"},
                    exc_info=metrics_exc,
                )
            emit_diag(
                self._logger,
                "persistent_heartbeat_flush_failure",
                reason="flush_error",
                severity="critical",
                alert=True,
                latency_seconds=flush_latency,
            )
            self._logger.error(
                "Failure in PersistentHeartbeatFlusher.handle_heartbeat: %s",
                exc,
                exc_info=exc,
            )
            return
        flush_latency = time_module.monotonic() - flush_started
        self._last_flush = now
        try:
            METRICS.record_heartbeat_flush(
                success=True,
                latency_seconds=flush_latency,
                now=time_module.time(),
            )
        except Exception as metrics_exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PersistentHeartbeatFlusher.flush metrics: %s",
                metrics_exc,
                extra={"event": "persistent_heartbeat_flush_metric_error"},
                exc_info=metrics_exc,
            )
        emit_diag(
            self._logger,
            "persistent_heartbeat_flush",
            reason="ok",
            severity="info",
            interval_sec=self._interval,
            timestamp=now,
            latency_seconds=flush_latency,
        )


class RuntimeSelfChecker:
    """Run runtime self-tests to detect silent subsystem failures."""

    def __init__(self, context: BotContext, interval_seconds: float = 600.0) -> None:
        """Initialize the runtime self-check helper.

        Args:
            context: Live bot context exposing subsystem references.
            interval_seconds: Desired cadence for periodic checks in seconds.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger = LOGGER
        self._context = context
        self.interval_seconds = max(float(interval_seconds), 60.0)
        self.last_run: datetime | None = None
        self.last_results: dict[str, dict[str, object]] = {}

    def run_full_check(self) -> dict[str, dict[str, object]]:
        """Execute all configured runtime self-checks.

        Args:
            None.

        Returns:
            Mapping of check names to result dictionaries.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered RuntimeSelfChecker.run_full_check",
            extra={"event": "runtime_self_check_enter"},
        )
        results: dict[str, dict[str, object]] = {}
        for name, check in self._collect_checks().items():
            try:
                ok, detail, meta = check()
            except Exception as exc:  # noqa: BLE001 - defensive surface
                self._logger.error(
                    "Failure in RuntimeSelfChecker.run_full_check: %s",
                    exc,
                    extra={"event": "runtime_self_check_error", "check": name},
                    exc_info=exc,
                )
                ok = False
                detail = f"exception:{exc}"[:256]
                meta = {}
            meta_payload = dict(meta or {})
            meta_payload.setdefault("check", name)
            results[name] = {"ok": bool(ok), "detail": detail, "meta": meta_payload}
            if not ok:
                self._logger.error(
                    "Runtime self-test detected failure",
                    extra={
                        "event": "runtime_self_check_failed",
                        "check": name,
                        "detail": detail,
                        "meta": meta_payload,
                    },
                )
        self.last_run = datetime.now(timezone.utc)
        self.last_results = results
        return results

    def _collect_checks(
        self,
    ) -> dict[str, Callable[[], tuple[bool, str, dict[str, object]]]]:
        """Return mapping of check names to callable probes.

        Args:
            None.

        Returns:
            Dictionary mapping check identifiers to callables.

        Raises:
            None.
        """

        return {
            "data_freshness": self._check_data_freshness,
            "streamer": self._check_streamer,
            "risk_breaker": self._check_risk_breaker,
            "session_guard": self._check_session_guard,
        }

    def _check_data_freshness(self) -> tuple[bool, str, dict[str, object]]:
        """Check data freshness for active symbols. Args: None. Returns: status tuple. Raises: None."""
        self._logger.debug(
            "Entered RuntimeSelfChecker._check_data_freshness",
            extra={"event": "runtime_self_check_entry", "check": "data_freshness"},
        )
        try:
            hub = self._context.data_hub
            if hub is None:
                return True, "no_data_hub", {}

            market_state = get_market_state()
            if market_state != MarketState.OPEN:
                return True, "market_closed", {"market_state": market_state.value}

            # [FIX] Use actually tracked symbols from DataHub to prevent false negatives
            # Accessing protected member _quotes is necessary here for introspection
            symbols = list(getattr(hub, "_quotes", {}).keys())

            # During startup / hydration, do NOT block trading if no quotes yet
            if not symbols:
                return True, "no_symbols_yet", {}

            interval = getattr(self._context.streamer, "_interval_s", 0.7) or 0.7
            adaptive_ms = max(2000, min(5000, int(float(interval) * 1000.0 * 2.5)))
            hard_ready_fn = getattr(self._context.market_data_manager, "hard_ready", None)
            hard_ready = bool(hard_ready_fn()) if callable(hard_ready_fn) else False

            def _threshold_for_symbol(symbol_name: str) -> float:
                upper = str(symbol_name or "").upper()
                if upper.startswith("NFO:"):
                    return 60000.0
                if upper == "NSE:NIFTY":
                    return 30000.0
                return float(adaptive_ms)

            per_symbol: list[tuple[str, bool, dict[str, object], float]] = []
            for s in symbols:
                threshold_ms = _threshold_for_symbol(s)
                ok_s, meta_s = hub.is_fresh(s, threshold_ms=threshold_ms)
                per_symbol.append((s, bool(ok_s), dict(meta_s or {}), threshold_ms))

            fresh = [item for item in per_symbol if item[1]]
            stale = [item for item in per_symbol if not item[1]]
            critical = [
                item
                for item in per_symbol
                if str(item[0]).upper().startswith("NFO:") or str(item[0]).upper() == "NSE:NIFTY"
            ] or per_symbol
            all_critical_stale = not any(item[1] for item in critical)
            if hard_ready and symbols and stale and not all_critical_stale:
                return True, "partial_stale_ignored", {
                    "fresh_symbols": len(fresh),
                    "stale_symbols": len(stale),
                    "live_symbols": len(symbols),
                }

            symbol, ok, meta, symbol_threshold_ms = fresh[0] if fresh else per_symbol[0]

            if hasattr(hub, "is_fresh"):
                try:
                    ok, meta = hub.is_fresh(symbol, threshold_ms=float(symbol_threshold_ms))
                except Exception as exc:
                    self._logger.error(
                        "Failure in RuntimeSelfChecker._check_data_freshness: %s",
                        exc,
                        extra={
                            "event": "runtime_self_check_error",
                            "check": "data_freshness",
                            "symbol": symbol,
                        },
                        exc_info=exc,
                    )
                else:
                    detail = cast(str, meta.get("reason") or "ok")
                    payload = cast(dict[str, object], dict(meta or {}))
                    payload["symbol_checked"] = canonical(symbol)
                    payload["adaptive_ms"] = symbol_threshold_ms
                    self._logger.info(
                        "Condition met: runtime_self_check_data_freshness",
                        extra={
                            "event": "runtime_self_check_data_freshness",
                            "symbol": symbol,
                            "symbol_checked": symbol,
                            "adaptive_ms": symbol_threshold_ms,
                            "detail": detail,
                            "detail_code": detail,
                        },
                    )
                    if not ok:
                        backoff_seconds = max(0.0, float(symbol_threshold_ms) * 2.0 / 1000.0)
                        runner = getattr(self._context, "strategy_runner", None)
                        if runner is not None:
                            try:
                                runner.set_data_freshness_backoff(
                                    backoff_seconds,
                                    detail_code=detail,
                                    symbol=symbol,
                                )
                            except Exception as exc:
                                self._logger.error(
                                    "Failure in RuntimeSelfChecker data freshness backoff: %s",
                                    exc,
                                    extra={
                                        "event": "runtime_self_check_backoff_error",
                                        "check": "data_freshness",
                                        "symbol": symbol,
                                    },
                                    exc_info=exc,
                                )
                    return ok, detail, payload

            ok, detail, meta = assess_datahub_fresh(
                hub,
                symbol,
                freshness_ms=adaptive_ms,
            )

            payload = cast(dict[str, object], dict(meta or {}))
            payload["symbol_checked"] = canonical(symbol)
            payload["adaptive_ms"] = adaptive_ms

            self._logger.info(
                "Condition met: runtime_self_check_data_freshness",
                extra={
                    "event": "runtime_self_check_data_freshness",
                    "symbol": symbol,
                    "symbol_checked": symbol,
                    "adaptive_ms": adaptive_ms,
                    "detail": detail,
                    "detail_code": detail,
                },
            )
            if not ok:
                backoff_seconds = max(0.0, float(adaptive_ms) * 2.0 / 1000.0)
                runner = getattr(self._context, "strategy_runner", None)
                if runner is not None:
                    try:
                        runner.set_data_freshness_backoff(
                            backoff_seconds,
                            detail_code=detail,
                            symbol=symbol,
                        )
                    except Exception as exc:
                        self._logger.error(
                            "Failure in RuntimeSelfChecker data freshness backoff: %s",
                            exc,
                            extra={
                                "event": "runtime_self_check_backoff_error",
                                "check": "data_freshness",
                                "symbol": symbol,
                            },
                            exc_info=exc,
                        )

            return ok, detail, payload
        except Exception as exc:
            self._logger.error(
                "Failure in RuntimeSelfChecker._check_data_freshness: %s",
                exc,
                extra={"event": "runtime_self_check_error", "check": "data_freshness"},
                exc_info=exc,
            )
            return False, f"exception:{exc}"[:256], {"error": str(exc)}

    def _check_streamer(self) -> tuple[bool, str, dict[str, object]]:
        """Assess market data streamer connectivity and backlog state.

        Args:
            None.

        Returns:
            Tuple containing success flag, detail token, and metadata dictionary.

        Raises:
            None.
        """
        # ✅ FIX: Streamer being disconnected outside trading hours is expected
        from datetime import datetime, time as dt_time
        from zoneinfo import ZoneInfo

        ist = ZoneInfo("Asia/Kolkata")
        now_ist = datetime.now(ist)
        is_trading_window = now_ist.weekday() < 5 and dt_time(
            9, 0
        ) <= now_ist.time() <= dt_time(15, 45)
        if not is_trading_window:
            return (
                True,
                "market_closed",
                cast(dict[str, object], {"connected": False, "market_closed": True}),
            )

        streamer = self._context.streamer
        if streamer is None:
            return True, "no_streamer", {}
        connected = True
        backlog = 0
        detail = "ok"
        is_connected_fn = getattr(streamer, "is_connected", None)
        if callable(is_connected_fn):
            with suppress(Exception):
                connected = bool(is_connected_fn())
        backlog_fn = getattr(streamer, "backlog_size", None)
        if callable(backlog_fn):
            with suppress(Exception):
                backlog = int(backlog_fn())
        detail = "disconnected" if not connected else detail
        if backlog > 1000:
            detail = "backlog_high"
            self._logger.debug(
                "Condition met: runtime_streamer_backlog_high",
                extra={"event": "runtime_streamer_backlog_high", "backlog": backlog},
            )
        recent_tick_age = 0.0
        last_tick_ts = getattr(streamer, "last_tick_ts", None)
        if last_tick_ts is not None:
            with suppress(Exception):
                recent_tick_age = max(0.0, time.time() - float(last_tick_ts))
            if recent_tick_age > 10.0:
                connected = False
                detail = "no_recent_ticks"
        payload = cast(
            dict[str, object],
            {
                "connected": connected,
                "backlog": backlog,
                "recent_tick_age": round(recent_tick_age, 3),
            },
        )
        return connected, detail, payload

    def _check_risk_breaker(self) -> tuple[bool, str, dict[str, object]]:
        """Confirm the risk circuit breaker is not engaged.

        Args:
            None.

        Returns:
            Tuple with boolean status, detail string, and metadata.

        Raises:
            None.
        """

        risk_manager = self._context.risk_manager
        if risk_manager is None:
            return False, "risk_manager_missing", {}
        try:
            tripped, reason = risk_manager.is_circuit_breaker_tripped()
        except Exception as exc:  # noqa: BLE001 - defensive
            return False, f"exception:{exc}"[:256], {}
        return (
            (not tripped),
            (reason or "ok") if tripped else "ok",
            {
                "breaker_tripped": tripped,
                "reason": reason or "",
            },
        )

    def _check_session_guard(self) -> tuple[bool, str, dict[str, object]]:
        """Validate trading session guard status remains healthy.

        Args:
            None.

        Returns:
            Tuple indicating success, detail, and metadata dictionary.

        Raises:
            None.
        """

        guard = self._context.session_guard
        if guard is None:
            return True, "no_guard", {}
        try:
            status = guard.evaluate()
        except Exception as exc:  # noqa: BLE001 - guard errors surface
            return False, f"exception:{exc}"[:256], {}
        if status is None:
            return False, "no_status", {}
        as_dict = status.as_dict() if hasattr(status, "as_dict") else {}
        session_valid = bool(getattr(status, "session_valid", False))
        return session_valid, "ok" if session_valid else "blocked", dict(as_dict)

    def _resolve_symbol(self) -> str:
        """Return primary symbol used for runtime data checks."""
        symbols = getattr(self._context.config, "symbols", None)
        if isinstance(symbols, Iterable) and not isinstance(symbols, (str, bytes)):
            for candidate in symbols:
                candidate_str = str(candidate).strip()
                if candidate_str:
                    return candidate_str
        if isinstance(symbols, (str, bytes)) and symbols:
            return str(symbols)

        # ✅ FIX: Return the standard Zerodha format
        return "256265"


def _configure_rate_limiter(cfg: Any) -> RateLimiter:
    """Configure the rate limiter from nested configuration."""

    limiter = RateLimiter()
    limiter.configure_bucket(
        "orders",
        capacity=cfg.orders.capacity,
        refill_rate_per_sec=cfg.orders.refill_rate_per_sec,
    )
    limiter.configure_bucket(
        "rest",
        capacity=cfg.rest.capacity,
        refill_rate_per_sec=cfg.rest.refill_rate_per_sec,
    )
    limiter.configure_bucket(
        "hist",
        capacity=cfg.hist.capacity,
        refill_rate_per_sec=cfg.hist.refill_rate_per_sec,
    )
    return limiter


def get_nifty_expiry() -> str:
    """Return the current month's Nifty expiry code (e.g., 25NOV)."""
    from datetime import datetime

    now = datetime.now()
    # Get 2-digit year and upper-case short month (e.g., 25NOV)
    return now.strftime("%y%b").upper()


def get_nifty_atm_strike(nifty_spot):
    """Round to nearest 50 or 100, as in your option chain tokens."""
    return round(nifty_spot / 50) * 50


def _find_existing_nifty_option_symbol(
    expiry: str, strike: int, opt_type: str = "CE"
) -> str | None:
    """
    Return a best-effort matching tradingsymbol (without exchange prefix) present
    in the InstrumentManager. Checks both NFO-prefixed and unprefixed keys.
    """
    # Try to locate InstrumentManager from bot context
    ctx = get_latest_bot_context()
    resolver = getattr(ctx, "instrument_manager", None) if ctx else None

    # nothing to validate against
    if resolver is None:
        return None

    want_exp = (expiry or "").upper()
    want_str = str(int(strike))
    want_ot = (opt_type or "CE").upper()

    # Candidate formats to try (without exchange and with possible variations):
    candidates_to_try = [
        f"NIFTY{want_exp}{want_str}{want_ot}",
        f"NIFTY{want_exp}{want_str:0>2}{want_ot}",
        f"NIFTY{want_exp}{want_str}{want_ot}".upper(),
    ]

    # Helper to check resolver mapping safely
    def resolver_lookup(key: str):
        try:
            # Many InstrumentResolver implementations expose lookup(symbol) or dict-like maps
            lookup_fn = getattr(resolver, "lookup", None)
            if callable(lookup_fn):
                return lookup_fn(key)
            # some use dict-like accessors with exchange prefix "NFO:"
            for attr in (
                "_by_symbol",
                "symbols",
                "symbol_map",
                "_symbol_map",
                "_symbol_by_token",
            ):
                m = getattr(resolver, attr, None)
                if isinstance(m, dict) and key in m:
                    return m[key]
            # also try simple get()
            get_fn = getattr(resolver, "get", None)
            if callable(get_fn):
                return get_fn(key)
        except Exception:
            return None
        return None

    for cand in candidates_to_try:
        # try both unprefixed and exchange-prefixed forms
        for key in (cand, f"NFO:{cand}"):
            meta = resolver_lookup(key)
            if meta:
                # return the canonical trading symbol without exchange prefix
                ts = meta.get("tradingsymbol") or meta.get("symbol") or cand
                return ts
    # fallback: scan resolver keys for approximate match (safely)
    try:
        keys = None
        for attr in ("_by_symbol", "symbols", "keys"):
            m = getattr(resolver, attr, None)
            if isinstance(m, dict):
                keys = m.keys()
                break
        if keys:
            for s in keys:
                s_up = s.upper()
                if not s_up.startswith("NIFTY"):
                    continue
                if not s_up.endswith(want_ot):
                    continue
                if want_exp in s_up and want_str in s_up:
                    return (
                        s_up if not s_up.startswith("NFO:") else s_up.split(":", 1)[-1]
                    )
    except Exception as e:
        LOGGER.exception(
            "[CRITICAL] unhandled exception", exc_info=True
        )
        raise

    return None


def _option_symbols_from_active_basket(basket: Mapping[str, Any] | Any | None) -> list[str]:
    """Extract authoritative option symbols from an ActiveContractBasket."""
    if basket is None:
        return []

    def _get(key: str, default: Any = None) -> Any:
        if isinstance(basket, Mapping):
            return basket.get(key, default)
        return getattr(basket, key, default)

    sources: list[Any] = []
    option_symbols = _get("option_symbols")
    if option_symbols:
        sources.append(option_symbols)
    else:
        ce_symbols = _get("ce_symbols") or []
        pe_symbols = _get("pe_symbols") or []
        if ce_symbols or pe_symbols:
            sources.extend([ce_symbols, pe_symbols])
        else:
            sources.append([_get("selected_ce"), _get("selected_pe")])

    symbols: list[str] = []
    for source in sources:
        if isinstance(source, (str, bytes)):
            iterable = [source]
        else:
            iterable = list(source or [])
        for sym in iterable:
            text = str(sym).strip() if sym is not None else ""
            if text and text.upper().endswith(("CE", "PE")):
                symbols.append(text)
    return list(dict.fromkeys(symbols))


def _active_basket_for_symbol_resolution(
    explicit_basket: Mapping[str, Any] | Any | None,
    market_data_manager: MarketDataManager | None,
) -> Mapping[str, Any] | Any | None:
    """Find the already-committed ActiveContractBasket without invoking selectors."""
    if explicit_basket is not None:
        return explicit_basket
    ctx = _LATEST_CTX
    if ctx is not None:
        basket = getattr(ctx, "active_contract_basket", None) or getattr(ctx, "active_trading_universe", None)
        if basket:
            return basket
        hub = getattr(ctx, "data_hub", None)
        hub_get = getattr(hub, "get_active_contract_basket", None)
        if callable(hub_get):
            basket = hub_get()
            if basket:
                return basket
        mdm = getattr(ctx, "market_data_manager", None)
        mdm_get = getattr(mdm, "get_active_contract_basket", None)
        if callable(mdm_get):
            basket = mdm_get()
            if basket:
                return basket
    mdm_get = getattr(market_data_manager, "get_active_contract_basket", None)
    if callable(mdm_get):
        return mdm_get()
    return None


def _get_symbols(
    config: AppConfig,
    resolver: Any | None = None,
    broker: Any | None = None,
    option_universe: OptionUniverseManager | None = None,
    market_data_manager: MarketDataManager | None = None,
    active_contract_basket: Mapping[str, Any] | Any | None = None,
) -> list[str]:
    """Return validated option symbols for trading."""
    LOGGER.info("=" * 60)
    LOGGER.info("🔍 _get_symbols() STARTING - Symbol Resolution")

    basket = _active_basket_for_symbol_resolution(active_contract_basket, market_data_manager)
    basket_symbols = _option_symbols_from_active_basket(basket)
    if basket_symbols:
        if option_universe is not None and hasattr(option_universe, "set_active_contract_basket"):
            option_universe.set_active_contract_basket(basket)
        LOGGER.info(
            "SYMBOL_RESOLUTION_ACTIVE_BASKET_USED count=%d selected_ce=%s selected_pe=%s",
            len(basket_symbols),
            getattr(basket, "selected_ce", None) if not isinstance(basket, Mapping) else basket.get("selected_ce"),
            getattr(basket, "selected_pe", None) if not isinstance(basket, Mapping) else basket.get("selected_pe"),
            extra={"event": "SYMBOL_RESOLUTION_ACTIVE_BASKET_USED", "count": len(basket_symbols)},
        )
        LOGGER.info("🎯 FINAL SYMBOLS TO TRADE: %s", basket_symbols)
        LOGGER.info("=" * 60)
        return basket_symbols

    execution_mode = str(os.getenv("EXECUTION_MODE") or os.getenv("MODE") or getattr(config, "execution_mode", "") or "").upper()
    if execution_mode == "LIVE":
        LOGGER.warning(
            "SYMBOL_RESOLUTION_BLOCKED reason=ACTIVE_BASKET_MISSING stage=_get_symbols",
            extra={"event": "SYMBOL_RESOLUTION_BLOCKED", "reason": "ACTIVE_BASKET_MISSING", "stage": "_get_symbols"},
        )
        return []

    symbols = getattr(config, "symbols", None)
    if symbols:
        if isinstance(symbols, Iterable) and not isinstance(symbols, (str, bytes)):
            result = [str(s).strip() for s in symbols if str(s).strip()]
            LOGGER.info("✅ Using configured SYMBOLS env: %s", result)
            LOGGER.info("=" * 60)
            return result
        result = [str(symbols)]
        LOGGER.info("✅ Using configured SYMBOLS env: %s", result)
        LOGGER.info("=" * 60)
        return result

    def _wait_for_first_tick(
        symbol: str, timeout: float = 15.0
    ) -> dict[str, Any] | None:
        """Wait for first live tick. Args: symbol, timeout. Returns: Tick payload or None. Raises: None."""
        import time

        if market_data_manager is None:
            return None

        start = time_module.monotonic()
        while time_module.monotonic() - start < timeout:
            tick = market_data_manager.get_latest_tick(symbol)
            if tick:
                return tick
            time.sleep(0.25)
        return None

    ltp: float = 0.0
    spot_symbol = "NSE:NIFTY"

    _allow_offhours = allow_offhours_testing_safe()
    _wait_timeout = 0.5 if _allow_offhours else 15.0
    first_tick = _wait_for_first_tick(spot_symbol, timeout=_wait_timeout)
    if first_tick:
        ltp_raw = first_tick.get("ltp") or first_tick.get("last_price")
        try:
            ltp = float(ltp_raw) if ltp_raw is not None else 0.0
        except (TypeError, ValueError):
            ltp = 0.0
    elif market_data_manager is not None and broker is None:
        LOGGER.warning(
            "spot_unavailable_after_wait",
            extra={"event": "spot_unavailable_after_wait", "symbol": spot_symbol},
        )

    if broker:
        try:
            token_candidates = [256265]
            str_candidates = [spot_symbol]
            inner = getattr(broker, "client", getattr(broker, "_broker", broker))

            def parse_price(data: Any) -> float:
                if not data:
                    return 0.0
                if isinstance(data, (int, float)):
                    return float(data)
                if isinstance(data, dict):
                    for key in ("last_price", "ltp", "close"):
                        val = data.get(key)
                        if val:
                            try:
                                return float(val)
                            except (ValueError, TypeError):
                                continue
                return 0.0

            if ltp == 0 and hasattr(inner, "get_ltp_bulk"):
                response = inner.get_ltp_bulk(token_candidates)
                if response:
                    for t in token_candidates:
                        val = response.get(t) or response.get(str(t))
                        price = parse_price(val)
                        if price > 0:
                            ltp = price
                            break

            if ltp == 0 and hasattr(inner, "get_ltp"):
                for candidate in str_candidates:
                    try:
                        get_ltp_response = inner.get_ltp([candidate])
                        response_payload: Any = get_ltp_response
                        if isinstance(get_ltp_response, dict):
                            response_payload = (
                                get_ltp_response.get(candidate)
                                or get_ltp_response.get(spot_symbol)
                                or next(iter(get_ltp_response.values()), 0.0)
                            )
                        price = parse_price(response_payload)
                        if price > 0:
                            ltp = price
                            break
                    except Exception:
                        continue

            if ltp == 0 and hasattr(inner, "ltp"):
                try:
                    q = inner.ltp(str_candidates)
                    for candidate in (spot_symbol, *str_candidates):
                        payload = q.get(candidate) if isinstance(q, dict) else None
                        price = parse_price(payload)
                        if price > 0:
                            ltp = price
                            break

                    if ltp == 0 and isinstance(q, dict):
                        for payload in q.values():
                            price = parse_price(payload)
                            if price > 0:
                                ltp = price
                                break
                except Exception as e:
                    LOGGER.exception(
                        "[CRITICAL] unhandled exception", exc_info=True
                    )
                    raise
        except Exception as exc:
            LOGGER.error("Error fetching live price: %s", exc, exc_info=True)

    if ltp <= 0 and market_data_manager is not None:
        LOGGER.warning(
            "spot_unavailable_after_wait",
            extra={"event": "spot_unavailable_after_wait", "symbol": spot_symbol},
        )

    if ltp <= 0:
        LOGGER.warning("ATM_SELECTION_BLOCKED reason=spot_unavailable_or_stale age_sec=%s", -1)
        return []

    global _LATEST_CTX
    universe = option_universe
    if universe is None:
        settings = getattr(_LATEST_CTX, "settings", None)
        universe_config = getattr(settings, "option_universe", {})
        universe = OptionUniverseManager(universe_config)

    universe.update_underlying(float(ltp))
    final_symbols = universe.get_filtered_universe(float(ltp))

    if resolver is None:
        LOGGER.error("Strategy skipped — instrument resolver unavailable")
        return []

    contracts = resolver.get_option_contracts(underlying="NIFTY")
    today = datetime.now().date()
    valid_contracts = [c for c in contracts if c.get("expiry") >= today]
    if not valid_contracts:
        raise RuntimeError("No valid option contracts resolved. Check instrument dump.")

    nearest_expiry = min(c["expiry"] for c in valid_contracts)
    filtered = [c for c in valid_contracts if c["expiry"] == nearest_expiry]
    unique_strikes = sorted({c["strike"] for c in filtered})
    if len(unique_strikes) < 2:
        raise RuntimeError("No valid option contracts resolved. Check instrument dump.")
    strike_step = unique_strikes[1] - unique_strikes[0]
    if strike_step <= 0:
        raise RuntimeError("No valid option contracts resolved. Check instrument dump.")

    atm = nearest_available_strike(float(ltp), unique_strikes)
    selected = [c for c in filtered if abs(c["strike"] - atm) <= 200]
    final_symbols = [c["tradingsymbol"] for c in selected]
    final_symbols = list(dict.fromkeys(sym for sym in final_symbols if sym))

    if not final_symbols:
        raise RuntimeError("No valid option contracts resolved. Check instrument dump.")

    LOGGER.info("ATM_SELECTED spot=%s atm=%s source=fresh_spot_tick strike_source=instrument_dump expiry=%s", float(ltp), atm, nearest_expiry)
    LOGGER.info(
        "RESOLVED_CONTRACTS count=%d expiry=%s",
        len(final_symbols),
        nearest_expiry,
    )
    LOGGER.debug("OptionUniv: Universe refreshed -> %s", final_symbols)

    if _LATEST_CTX:
        _LATEST_CTX.update_spot_price("NIFTY", float(ltp))

    LOGGER.info("🎯 FINAL SYMBOLS TO TRADE: %s", final_symbols)
    LOGGER.info("=" * 60)
    return final_symbols


def _data_ready(mdm: MarketDataManager | None) -> bool:
    """Check live tick readiness. Args: mdm. Returns: bool. Raises: None."""
    if mdm is None:
        return False
    # Only require the spot index tick — futures symbol rolls monthly
    # and a hardcoded value causes perpetual False returns after expiry.
    required = ["NSE:NIFTY"]
    for sym in required:
        if not mdm.get_latest_tick(sym):
            return False
    return True


def _compute_indicator_readiness(ctx: BotContext) -> bool:
    """Compute indicator readiness from runner/indicator histories."""
    runner = getattr(ctx, "strategy_runner", None)
    indicator_engine = getattr(ctx, "indicator_engine", None)
    if runner is None or indicator_engine is None:
        return False
    required = int(getattr(runner, "_required_candles", 20) or 20)
    active_symbols = list(getattr(runner, "_active_symbols", set()) or [])
    tradeable_symbols = [
        symbol
        for symbol in active_symbols
        if str(symbol).upper().startswith("NFO:")
        and str(symbol).upper().endswith(("CE", "PE"))
    ]
    if not tradeable_symbols:
        return False
    for symbol in tradeable_symbols:
        try:
            if hasattr(indicator_engine, "has_min_bars") and indicator_engine.has_min_bars(
                symbol, required
            ):
                return True
            history = (
                indicator_engine.get_history(symbol)
                if hasattr(indicator_engine, "get_history")
                else []
            )
            if history is not None and len(history) >= required:
                return True
        except Exception:
            continue
    return False


def _build_canonical_active_basket(
    *,
    instrument_manager: InstrumentManager | None,
    spot_token_resolver: Callable[[str], int] | None,
    spot_ltp: float,
    futures_symbol: str | None = None,
    strike_step: int = 50,
    strikes_around_atm: int = 3,
) -> dict[str, Any]:
    """Build canonical live basket by delegating contract identity to InstrumentManager."""
    if instrument_manager is None:
        raise RuntimeError("instrument manager unavailable for basket resolution")
    basket_obj = instrument_manager.get_active_nifty_contracts(
        float(spot_ltp),
        strikes_around_atm=int(strikes_around_atm),
        strike_step=int(strike_step),
        include_future=True,
    )
    basket = asdict(basket_obj)
    # Compatibility keys consumed by existing readiness/runner code. Values all
    # come from ActiveContractBasket; no symbols are generated here.
    option_symbols = list(basket_obj.option_symbols)
    basket["symbols"] = list(basket_obj.all_symbols)
    basket["all_symbols"] = list(basket_obj.all_symbols)
    basket["all_tokens"] = list(basket_obj.all_tokens)
    basket["option_symbols"] = option_symbols
    basket["option_tokens"] = list(basket_obj.option_tokens)
    basket["ce_symbols"] = [sym for sym in option_symbols if str(sym).endswith("CE")]
    basket["pe_symbols"] = [sym for sym in option_symbols if str(sym).endswith("PE")]
    basket["atm_ce"] = basket_obj.selected_ce
    basket["atm_pe"] = basket_obj.selected_pe
    return basket


def _get_strategy_config(config: AppConfig) -> StrategyRunnerConfig:
    """Build strategy runner configuration with environment overrides.

    Environment Variables:
        MIN_INDICATOR_BARS: Number of bars required before signal generation (default: 10)
        SIGNAL_COOLDOWN_SECONDS: Cooldown between signals (default: 3.0)
        TRADE_COOLDOWN_SECONDS: Cooldown between trades (default: 10.0)
    """
    cfg = getattr(config, "strategy_config", None)
    if isinstance(cfg, StrategyRunnerConfig):
        return cfg

    # Allow environment override for warmup bars (CRITICAL for faster startup)
    warmup_bars_default = 20  # Changed from 50 to 10 for faster signal generation
    warmup_bars = int(os.getenv("MIN_INDICATOR_BARS", str(warmup_bars_default)))

    signal_cooldown = float(os.getenv("SIGNAL_COOLDOWN_SECONDS", "3.0"))
    trade_cooldown = float(os.getenv("TRADE_COOLDOWN_SECONDS", "10.0"))

    LOGGER.info(
        f"Strategy config: warmup_bars={warmup_bars}, signal_cooldown={signal_cooldown}s, "
        f"trade_cooldown={trade_cooldown}s"
    )

    return StrategyRunnerConfig(
        signal_cooldown_seconds=float(
            getattr(cfg, "signal_cooldown_seconds", signal_cooldown) or signal_cooldown
        ),
        trade_cooldown_seconds=float(
            getattr(cfg, "trade_cooldown_seconds", trade_cooldown) or trade_cooldown
        ),
        min_indicator_bars=int(
            getattr(cfg, "min_indicator_bars", warmup_bars) or warmup_bars
        ),
        max_trade_history=int(getattr(cfg, "max_trade_history", 100) or 100),
    )


def _bind_ws_mdm(ctx: BotContext) -> None:
    """Wire WebSocket connectivity events into the market data manager."""
    ws = getattr(ctx, "websocket_manager", None)
    mdm = getattr(ctx, "market_data_manager", None)
    if ws is None or mdm is None:
        return

    def _on_connect() -> None:
        try:
            mdm.set_ws_connected(True)
            mdm.bump_heartbeat()
            # Rehydrate bracket runtime state after reconnect to keep trailing/exit tracking continuous.
            runner = getattr(ctx, "strategy_runner", None)
            bracket_manager = getattr(runner, "_bracket_manager", None)
            position_manager = getattr(runner, "_position_manager", None)
            if (
                bracket_manager
                and position_manager
                and hasattr(position_manager, "get_all_positions")
            ):
                for pos in position_manager.get_all_positions():
                    symbol = getattr(pos, "symbol", "")
                    bracket = getattr(bracket_manager, "active_brackets", {}).get(
                        symbol
                    )
                    if bracket and hasattr(bracket, "rehydrate_state_from_position"):
                        bracket.rehydrate_state_from_position(pos)
        except Exception as exc:
            LOGGER.warning(
                f"Failed to set WS connected state: {exc}",
                extra={"event": "ws_mdm_connect_failed"},
            )

    def _on_disconnect() -> None:
        try:
            mdm.set_ws_connected(False)
        except Exception as exc:
            LOGGER.warning(
                f"Failed to set WS disconnected state: {exc}",
                extra={"event": "ws_mdm_disconnect_failed"},
            )

    try:
        ws.set_callbacks(on_connect=_on_connect, on_disconnect=_on_disconnect)
    except Exception as exc:
        LOGGER.error(
            f"Failed to bind WS callbacks: {exc}",
            extra={"event": "ws_mdm_bind_failed"},
            exc_info=True,
        )


async def reconcile_with_broker(
    broker_client: Any,
    bracket_manager: Any,
    order_manager: Any,
    logger: Any,
) -> None:
    """Phase 8: Ghost-order reconciliation on startup.

    Fetches live positions and orders from broker.  Any open position that has
    no bracket registered (ghost/orphan) gets a safety bracket attached using
    default risk parameters.  Any open order not tracked locally is logged for
    visibility.

    Args:
        broker_client: Live broker API client.
        bracket_manager: BracketManager instance to check/register brackets.
        order_manager: OrderManager for placing protective brackets.
        logger: Structured logger instance.

    Returns:
        None. All errors are caught and logged — startup must not be blocked.
    """
    logger.info("RECONCILE_START: checking broker for ghost orders and orphan positions")

    # ── 1. Fetch live orders ─────────────────────────────────────────────────
    try:
        if inspect.iscoroutinefunction(getattr(broker_client, "get_orders", None)):
            raw_orders = await broker_client.get_orders()
        else:
            raw_orders = await asyncio.to_thread(_run_sync_locked, broker_client.get_orders)
            if asyncio.iscoroutine(raw_orders):
                raw_orders = await raw_orders
                
        open_orders = [o for o in (raw_orders or []) if isinstance(o, dict) and str(o.get("status", "")).upper() in {"OPEN", "TRIGGER PENDING"}]
        logger.info("RECONCILE_ORDERS: found %d open orders from broker", len(open_orders))
        for o in open_orders:
            oid = o.get("order_id") or o.get("id", "")
            sym = o.get("tradingsymbol") or o.get("symbol", "")
            status = o.get("status", "")
            logger.info("BROKER_OPEN_ORDER order_id=%s symbol=%s status=%s", oid, sym, status)
    except Exception as exc:
        logger.warning("RECONCILE_ORDERS: failed to fetch broker orders: %s", exc)

    # ── 2. Fetch live positions and attach safety brackets for orphans ────────
    try:
        if inspect.iscoroutinefunction(getattr(broker_client, "get_positions", None)):
            raw_positions = await broker_client.get_positions()
        else:
            raw_positions = await asyncio.to_thread(_run_sync_locked, broker_client.get_positions)
            if asyncio.iscoroutine(raw_positions):
                raw_positions = await raw_positions
        positions = [p for p in (raw_positions or []) if isinstance(p, dict)]
        logger.info("RECONCILE_POSITIONS: found %d positions from broker", len(positions))

        for pos in positions:
            try:
                sym = str(
                    pos.get("tradingsymbol") or pos.get("symbol") or ""
                ).strip().upper()
                if not sym:
                    continue
                qty = int(pos.get("quantity") or pos.get("net_quantity") or 0)
                if qty == 0:
                    continue

                avg_price = float(
                    pos.get("average_price") or pos.get("buy_price") or pos.get("sell_price") or 0.0
                )

                # Check if bracket_manager already tracks this symbol
                is_managed = False
                if bracket_manager:
                    is_managed = bool(
                        getattr(bracket_manager, "is_symbol_managed", lambda s: False)(sym)
                    )

                if is_managed:
                    logger.debug("RECONCILE: %s already managed by BracketManager", sym)
                    continue

                logger.warning(
                    "GHOST_POSITION detected: symbol=%s qty=%d avg=%.2f — attaching safety bracket",
                    sym, qty, avg_price,
                )

                # Attach orphan position with default ATR-based SL (2% fallback)
                if bracket_manager and avg_price > 0:
                    try:
                        sl_default = round(avg_price * 0.98, 2)  # 2% below entry
                        tp_default = round(avg_price * 1.04, 2)  # 4% above entry
                        side = "BUY" if qty > 0 else "SELL"

                        attach_fn = getattr(bracket_manager, "attach_orphan_position", None)
                        if attach_fn:
                            attach_fn(
                                symbol=sym,
                                side=side,
                                qty=abs(qty),
                                entry_price=avg_price,
                                sl=sl_default,
                                tp=tp_default,
                            )
                            logger.warning(
                                "SAFETY_BRACKET_ATTACHED: symbol=%s sl=%.2f tp=%.2f",
                                sym, sl_default, tp_default,
                            )
                        else:
                            logger.warning(
                                "GHOST_POSITION: bracket_manager has no attach_orphan_position for %s", sym
                            )
                    except Exception as bracket_exc:
                        logger.error(
                            "SAFETY_BRACKET_FAILED: symbol=%s error=%s", sym, bracket_exc
                        )
            except Exception as pos_exc:
                logger.warning("RECONCILE_POSITION_ITEM_ERROR: %s", pos_exc)

    except Exception as exc:
        logger.warning("RECONCILE_POSITIONS: failed to fetch broker positions: %s", exc)

    logger.info("RECONCILE_COMPLETE")


async def reconcile_positions_on_startup(
    broker_client: Any,
    position_manager: Any,
    order_manager: Any,
    logger: Any,
) -> None:
    """Reconcile local positions with broker state on startup.

    Args:
        broker_client: Broker API client used to fetch live positions.
        position_manager: Local position manager maintaining state.
        order_manager: Order manager reference for diagnostic context.
        logger: Structured logger used for observability.

    Returns:
        None.

    Raises:
        None. Exceptions are logged and re-raised for upstream handling.
    """

    logger.debug(
        "Entered reconcile_positions_on_startup",
        extra={
            "event": "reconcile.start.enter",
            "order_manager": getattr(
                order_manager, "__class__", type(order_manager)
            ).__name__,
        },
    )
    logger.info(
        "Starting position reconciliation",
        extra={"event": "reconcile.start"},
    )

    try:
        broker_snapshot: list[Mapping[str, Any]] = []
        raw_positions = await asyncio.to_thread(
            _run_sync_locked,
            broker_client.get_positions,
        )
        for entry in raw_positions:
            if isinstance(entry, Mapping):
                broker_snapshot.append(entry)

        local_positions = position_manager.get_all_positions()

        def _normalize_symbol(payload: Mapping[str, Any]) -> str:
            raw_symbol = (
                payload.get("tradingsymbol")
                or payload.get("symbol")
                or payload.get("instrument")
                or ""
            )
            symbol = str(raw_symbol).strip().upper()
            if ":" in symbol:
                symbol = symbol.split(":", maxsplit=1)[-1].strip().upper()
            return symbol

        def _extract_quantity(payload: Mapping[str, Any]) -> int:
            """Return the integer quantity from a broker payload.

            Args:
                payload: Raw broker position payload.

            Returns:
                Normalised signed quantity as an integer.

            Raises:
                None.
            """

            quantity_candidate = (
                payload.get("net_qty")
                or payload.get("net_quantity")
                or payload.get("netQuantity")
                or payload.get("quantity")
                or payload.get("net")
            )
            if quantity_candidate is None:
                return 0
            try:
                numeric_quantity = float(quantity_candidate)
            except (TypeError, ValueError):
                return 0
            return int(numeric_quantity)

        def _extract_average_price(payload: Mapping[str, Any]) -> float:
            """Return the average price from a broker payload.

            Args:
                payload: Raw broker position payload.

            Returns:
                Average price when available, otherwise ``0.0``.

            Raises:
                None.
            """

            price_candidate = (
                payload.get("average_price")
                or payload.get("avg_price")
                or payload.get("price")
                or payload.get("buy_price")
                or payload.get("sell_price")
            )
            if price_candidate is None:
                return 0.0
            try:
                return float(price_candidate)
            except (TypeError, ValueError):
                return 0.0

        broker_symbols: dict[str, dict[str, Any]] = {}
        for payload in broker_snapshot:
            symbol = _normalize_symbol(payload)
            if not symbol:
                continue
            quantity = _extract_quantity(payload)
            if quantity == 0:
                continue
            broker_symbols[symbol] = {
                "quantity": quantity,
                "average_price": _extract_average_price(payload),
                "raw": payload,
            }

        local_symbols = {pos.symbol: pos for pos in local_positions}
        orphaned = set(broker_symbols) - set(local_symbols)

        if orphaned:
            logger.warning(
                "Found orphaned positions in broker",
                extra={
                    "event": "reconcile.orphaned",
                    "symbols": sorted(orphaned),
                    "count": len(orphaned),
                },
            )
            for symbol in sorted(orphaned):
                broker_position = broker_symbols[symbol]
                logger.info(
                    "Imported orphaned position",
                    extra={
                        "event": "reconcile.import",
                        "symbol": symbol,
                        "quantity": broker_position["quantity"],
                    },
                )

        mismatch_symbols: list[str] = []
        for symbol, local_position in local_symbols.items():
            broker_pos: dict[str, Any] | None = broker_symbols.get(symbol)
            if broker_pos is None:
                continue
            broker_qty_raw = int(broker_pos.get("quantity", 0))
            broker_qty = abs(broker_qty_raw)
            broker_side = "LONG" if broker_qty_raw > 0 else "SHORT"
            local_qty = int(getattr(local_position, "quantity", 0))
            local_side = str(getattr(local_position, "side", "LONG")).upper()
            if broker_qty != local_qty or broker_side != local_side:
                mismatch_symbols.append(symbol)
                logger.warning(
                    "Position quantity mismatch",
                    extra={
                        "event": "reconcile.mismatch",
                        "symbol": symbol,
                        "broker_qty": broker_qty,
                        "broker_side": broker_side,
                        "local_qty": local_qty,
                        "local_side": local_side,
                    },
                )

        if broker_snapshot:
            position_manager.synchronize_with_broker(broker_snapshot)

        logger.info(
            "Reconciliation complete",
            extra={
                "event": "reconcile.complete",
                "orphaned_count": len(orphaned),
                "mismatch_count": len(mismatch_symbols),
            },
        )

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Position reconciliation failed",
            extra={"event": "reconcile.failed", "error": str(exc)},
            exc_info=exc,
        )
        raise


def parse_nifty_option_symbol(symbol: str) -> dict | None:
    """
    Parse NIFTY option symbol to extract strike, expiry, and option type.
    """
    import calendar
    from datetime import datetime, timedelta, timezone  # Ensure timezone is imported
    import re

    symbol = symbol.replace("NFO:", "").strip()

    # Monthly/Far Weekly Pattern: NIFTY25NOV25950CE
    monthly_match = re.match(r"NIFTY(\d{2})([A-Z]{3})(\d+)(CE|PE)", symbol)
    if monthly_match:
        year, month_str, strike, opt_type = monthly_match.groups()
        month_names = {
            "JAN": 1,
            "FEB": 2,
            "MAR": 3,
            "APR": 4,
            "MAY": 5,
            "JUN": 6,
            "JUL": 7,
            "AUG": 8,
            "SEP": 9,
            "OCT": 10,
            "NOV": 11,
            "DEC": 12,
        }
        month = month_names.get(month_str)
        if month:
            year_full = 2000 + int(year)

            # FIX S13: NIFTY expiry is Tuesday (not Thursday).
            last_day = calendar.monthrange(year_full, month)[1]
            expiry = datetime(year_full, month, last_day)
            while expiry.weekday() != 1:  # Tuesday = 1
                expiry = expiry - timedelta(days=1)

            # Use total_seconds for float days_to_expiry
            days_to_expiry = (
                expiry - datetime.now(timezone.utc).replace(tzinfo=None)
            ).total_seconds() / 86400.0

            return {
                "strike": int(strike),
                "expiry": expiry,
                "days_to_expiry": max(days_to_expiry, 0.001),
                "option_type": opt_type,
                "symbol_type": "Monthly",
            }

    # Simple pattern match for weekly/other (can be expanded)
    weekly_match = re.match(r"NIFTY(\d{2})([A-Z])(\d{2})(\d+)(CE|PE)", symbol)
    if weekly_match:
        # Placeholder logic, needs proper date mapping for weeks
        return {
            "strike": int(weekly_match.groups()[3]),
            "expiry": datetime.now(),
            "days_to_expiry": 3.0,
            "option_type": weekly_match.groups()[4],
            "symbol_type": "Weekly (Approx)",
        }

    return None


def calculate_greeks_simple(
    spot: float,
    strike: float,
    days_to_expiry: float,
    option_type: str,
    volatility: float = 0.20,  # 20% IV assumption
) -> dict:
    """
    Simple Greeks approximation (using Black-Scholes principles).
    """
    import math

    if days_to_expiry <= 0.0:
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "days_to_expiry": 0.0,
            "moneyness": spot / strike,
        }

    t = days_to_expiry / 365.25  # Time in years
    moneyness = spot / strike

    # Simple delta approximation (ATMs are 0.5/-0.5)
    delta = 0.5
    if option_type.upper() in ["CE", "CALL"]:
        delta = 0.5 + min(0.49, max(0, moneyness - 1.0))
    else:  # PUT
        delta = -0.5 - min(0.49, max(0, 1.0 - moneyness))

    # Gamma (peaks at ATM)
    gamma = 0.01 / (abs(moneyness - 1) + 0.01) * math.sqrt(1 / max(t, 0.01))
    gamma = min(gamma, 0.05)

    # Theta (time decay)
    theta_base = spot * volatility / (2 * math.sqrt(max(t, 0.01))) / 365.25
    theta = -1.0 * theta_base

    # Vega (volatility sensitivity)
    vega = spot * math.sqrt(max(t, 0.01)) * 0.01

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 2),
        "vega": round(vega, 2),
        "days_to_expiry": round(days_to_expiry, 1),
        "moneyness": round(moneyness, 3),
    }


def _setup_telegram(ctx: BotContext) -> None:
    """Wire the Telegram controller with full access to bot components."""
    settings = ctx.settings.notifications

    # 1. Extract Credentials
    bot_token = settings.token
    # Handle Set[int] -> Single ID conversion safely
    chat_id = (
        next(iter(settings.whitelist_chat_ids)) if settings.whitelist_chat_ids else None
    )

    # 2. Validation with Explicit Logging
    if not bot_token or not chat_id:
        LOGGER.warning(
            f"⚠️ Telegram DISABLED: Missing Credentials. Token={'OK' if bot_token else 'MISSING'}, ChatID={'OK' if chat_id else 'MISSING'}"
        )
        return

    try:
        from nifty_scalper_bot.notifications.telegram_controller import (
            TelegramBot,
            TelegramDeps,
        )

        # 3. Initialize with extracted values
        deps = TelegramDeps(
            token=str(bot_token),  # Ensure string format
            chat_id=str(chat_id),  # Ensure string format
            app_version="1.0.0",
            risk_manager=ctx.risk_manager,
            order_manager=ctx.order_manager,
            position_manager=ctx.position_manager,
            strategy_runner=ctx.strategy_runner,
            market_data_manager=ctx.market_data_manager,
            unified_manager=ctx.unified_manager,
            stream_supervisor=getattr(ctx, "stream_supervisor", None),
            data_hub=ctx.data_hub,
            instrument_resolver=getattr(ctx, "instrument_manager", None),
            enable_polling_fallback=True,
            bot_context=ctx,
        )

        ctx.telegram_bot = TelegramBot(deps)
        LOGGER.info("✅ Telegram Controller wired successfully.")

    except Exception as e:
        LOGGER.error(f"❌ Telegram setup failed: {e}", exc_info=True)


def initialize_components(settings: Settings | None = None) -> BotContext:
    """Initialize all components in correct order."""

    import threading

    ensure_multiproc_dir(clear_stale=True)
    settings = settings or get_settings()
    fingerprint = _build_startup_fingerprint()
    LOGGER.info(
        "startup_fingerprint version=%s release=%s",
        fingerprint["version"],
        fingerprint["release"],
        extra={"event": "startup_fingerprint", **fingerprint},
    )
    live_enabled = _compute_live_execution_enabled()
    LOGGER.info(
        "BOT_INSTANCE_FINGERPRINT deployment_id=%s replica_id=%s commit_sha=%s live_execution=%s",
        str(os.getenv("RAILWAY_DEPLOYMENT_ID", "")).strip() or "unknown",
        str(os.getenv("RAILWAY_REPLICA_ID", "")).strip() or "unknown",
        str(os.getenv("RAILWAY_GIT_COMMIT_SHA", "")).strip() or fingerprint["release"],
        bool(live_enabled),
        extra={
            "event": "BOT_INSTANCE_FINGERPRINT",
            "deployment_id": str(os.getenv("RAILWAY_DEPLOYMENT_ID", "")).strip() or "unknown",
            "replica_id": str(os.getenv("RAILWAY_REPLICA_ID", "")).strip() or "unknown",
            "commit_sha": str(os.getenv("RAILWAY_GIT_COMMIT_SHA", "")).strip() or fingerprint["release"],
            "live_execution": bool(live_enabled),
        },
    )
    _enforce_live_single_replica_safety(is_live_execution=bool(live_enabled))
    config = settings.app
    raw_ws_disabled = os.getenv("WEBSOCKET__DISABLED")
    if raw_ws_disabled is None:
        raw_ws_disabled = os.getenv("WEBSOCKET_DISABLED")
    websocket_disabled_env = str(raw_ws_disabled or "false").strip().lower() == "true"

    raw_webhook_env = os.getenv("TELEGRAM__WEBHOOK_ENABLED")
    if raw_webhook_env is None:
        raw_webhook_env = os.getenv("TELEGRAM_WEBHOOK_ENABLED")
    telegram_webhook_env_enabled = (
        str(raw_webhook_env or "false").strip().lower() == "true"
    )

    notif_settings = settings.notifications
    if not telegram_webhook_env_enabled and notif_settings.webhook_enabled:
        notif_settings.webhook_enabled = False
    ws_host = urlsplit(str(config.broker.websocket_url)).hostname or ""

    rate_limiter = _configure_rate_limiter(config.ratelimit)
    message_bus = MessageBus()

    from nifty_scalper_bot.data.robust_provider import (
        CircuitBreakerConfig,
        RobustDataProvider,
    )

    broker_client = ZerodhaKiteClient(
        api_key=config.broker.api_key,
        api_secret=config.broker.api_secret,
        access_token=config.broker.access_token,
    )

    async def _provider_notify(event: str, data: dict[str, Any]) -> None:
        """Forward provider notifications to telegram notifier. Args: event/data. Returns: None. Raises: none."""
        if notifier is not None:
            await notifier.send_event(event, data)

    robust_provider = RobustDataProvider(
        broker_client=broker_client,
        circuit_config=CircuitBreakerConfig(failure_threshold=5, timeout_seconds=60.0),
        notifier=_provider_notify,
    )

    broker_client.preload_instruments()

    # ── InstrumentManager: token-first single source of truth ────────────────
    # Created here (not loaded yet — load() is deferred to startup_sequence
    # so the broker auth token is valid and market is open).
    instrument_manager = InstrumentManager(broker_client)

    margin_segment_env = os.getenv("BROKER_MARGIN_SEGMENT", "equity") or "equity"
    margin_segment = margin_segment_env.strip().lower()
    if margin_segment not in {"equity", "commodity"}:
        margin_segment = "equity"
    try:
        margin_summary = broker_client.get_margin_summary(segment=margin_segment)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "broker_margin_summary_failed",
            extra={
                "event": "broker_margin_summary_failed",
                "segment": margin_segment,
                "error": str(exc),
            },
            exc_info=exc,
        )
    else:
        LOGGER.info(
            "broker_margin_summary_loaded",
            extra={
                "event": "broker_margin_summary_loaded",
                "segment": margin_segment,
                "available": margin_summary.get("available"),
                "used": margin_summary.get("used"),
                "net": margin_summary.get("net"),
            },
        )

    websocket_enabled = bool(getattr(settings, "websocket_enabled", True))
    if websocket_disabled_env:
        websocket_enabled = False
    stream_mode_raw = coalesce_str(
        "STREAM__MODE",
        "STREAMING_MODE",
        default="websocket",
    )
    streaming_mode = stream_mode_raw.strip().lower() or "websocket"
    poll_enabled = coalesce_bool("POLLING__ENABLED", default=True)
    poll_interval_sec = coalesce_float("POLLING__INTERVAL_SEC", default=0.0)
    if poll_interval_sec > 10.0:
        poll_interval_sec = poll_interval_sec / 1000.0
    poll_interval_ms_fallback = coalesce_int(
        "POLL_INTERVAL_MS",
        "MICRO_QUOTE_POLL_MS",
        default=700,
    )
    if poll_interval_sec <= 0.0:
        poll_interval_sec = float(poll_interval_ms_fallback) / 1000.0
    poll_interval_sec = max(0.2, float(poll_interval_sec))
    poll_interval_ms = int(poll_interval_sec * 1000.0)
    poll_batch_size = max(
        1,
        coalesce_int("POLLING__BATCH_SIZE", "POLL_BATCH_SIZE", default=50),
    )
    poll_require_depth = coalesce_bool(
        "POLLING__REQUIRE_DEPTH",
        "POLL_REQUIRE_DEPTH",
        "EXECUTOR__REQUIRE_DEPTH",
        default=False,
    )
    poll_warn_rate_limit = coalesce_bool(
        "POLLING__WARN_ON_RATE_LIMIT",
        "POLL_WARN_RATE_LIMIT",
        default=True,
    )
    # Normalize symbols (quotes tolerated) and default fallback
    raw_syms = get_csv("POLLING__SYMBOLS")
    if raw_syms:
        poll_symbols = unique_normalized_symbols(raw_syms)
    else:
        poll_symbols = ["NSE:NIFTY"]

    # InstrumentManager is the single source of truth - no resolver needed
    # Broker client will use instrument_manager directly via unified_manager

    websocket_client: Any | None = None
    websocket_manager: WebSocketManager | None = None
    streamer: Any
    polling_fallback_streamer: PollingStreamer | None = None
    stream_supervisor: StreamSupervisor | None = None
    risk_manager_ref: dict[str, RiskManager | None] = {"instance": None}
    stream_supervisor_started = False
    risk_manager_data_hub_attached = False
    margin_engine_data_hub_attached = False
    bracket_manager_attached = False
    data_hub: DataHub | None = None
    hub_store: HubStore | None = None
    try:
        hub_store = HubStore()
    except Exception:  # pragma: no cover - defensive fallback
        LOGGER.exception("hub_store_init_failed")
        hub_store = None

    def _resolve_ws_token() -> str:
        return ""

    def _refresh_ws_session() -> None:  # pragma: no cover - polling default
        return None

    def _ws_token_issued_at() -> float | None:
        return None

    ws_mode_requested = streaming_mode in {"websocket", "ws"}
    use_polling = (not websocket_enabled) or streaming_mode in {"polling", "poll"}
    if websocket_enabled and ws_mode_requested:
        use_polling = False
        poll_enabled = False
    if not poll_enabled and not websocket_enabled:
        raise ConfigurationError(
            "Polling disabled while websocket transport is disabled; "
            "no streamer available"
        )
    if not poll_enabled and use_polling:
        use_polling = False  # explicit disable beats default
    market_data_mode = "polling" if use_polling else "websocket"
    LOGGER.info(
        "Market data streamer starting in %s mode",
        market_data_mode,
        extra={
            "event": "market_data_mode",
            "mode": market_data_mode,
            "websocket_disabled_env": websocket_disabled_env,
            "streaming_mode": streaming_mode,
        },
    )
    if use_polling:
        market_data_manager = MarketDataManager(
            broker_client,
            None,
            resolver=instrument_manager,
        )

        # [FIX] Start Health Monitor (Watchdog) for Polling Mode
        # This prevents "Zombie Mode" by killing the process if data stops for 3 mins
        def _monitor_data_health():
            import logging
            import os
            import threading
            import time

            logger = logging.getLogger("nifty_scalper_bot.watchdog")
            logger.info("✅ Data Health Monitor Started (Polling Mode)")

            while True:
                time.sleep(60)  # Check every minute

                # Check if data is flowing
                last_tick = getattr(market_data_manager, "last_tick_time", 0)

                # If we have received data before (>0) but it's now stale (>180s)
                if last_tick > 0 and (time.time() - last_tick > 180):
                    logger.critical(
                        f"🚨 FATAL: No data for {int(time.time() - last_tick)}s. Zombie Mode detected. Exiting."
                    )
                    os._exit(
                        1
                    )  # Kill process -> Railway auto-restarts -> Connection restored

        health_thread = threading.Thread(target=_monitor_data_health, daemon=True)
        health_thread.start()
        try:
            start_watchdog(market_data_manager)
            LOGGER.info("✅ Data Health Monitor (Watchdog) started")
        except Exception as exc:
            LOGGER.warning(f"Failed to start watchdog: {exc}")

        data_hub = DataHub(
            market_data_manager,
            options_only=True,
            store=hub_store,
            event_bus=message_bus,
        )
        # Explicitly mark WS disconnected in polling mode so health reflects polling
        market_data_manager.set_ws_connected(False)

    # ------------------------------------------------------------------
    # [FIX] Polling Tick Handler & Throttling Helper
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Streamer Selection Logic (Polling vs WebSocket)
    # ------------------------------------------------------------------

    def _on_poll_tick(tick: dict[str, Any]) -> None:
        if not tick or type(tick) is not dict:
            return
        token_raw = tick.get("instrument_token", tick.get("token"))
        try:
            token = int(token_raw) if token_raw is not None else None
        except (TypeError, ValueError):
            token = None
        if token is None:
            return

        symbol: str | None = None
        try:
            symbol = instrument_manager.get_symbol(token)
        except Exception:
            symbol = None
        if not symbol and market_data_manager is not None:
            try:
                symbol_lookup = market_data_manager.get_latest_tick(token)
                symbol = (
                    str(symbol_lookup.get("symbol"))
                    if isinstance(symbol_lookup, dict) and symbol_lookup.get("symbol")
                    else None
                )
            except Exception:
                symbol = None
            if not symbol:
                symbol = market_data_manager._symbol_by_token.get(token)  # noqa: SLF001
        if not symbol:
            log_throttled(
                LOGGER,
                f"poll_unresolved_token_{token}",
                f"Dropping polling tick with unresolved token={token}",
                interval_sec=60.0,
                level=logging.WARNING,
            )
            return

        if market_data_manager is None:
            return
        try:
            canonical_symbol = market_data_manager._canonical_symbol(symbol)  # noqa: SLF001
        except Exception:
            log_throttled(
                LOGGER,
                f"poll_invalid_symbol_{token}",
                f"Dropping polling tick with malformed symbol token={token} symbol={symbol}",
                interval_sec=60.0,
                level=logging.WARNING,
            )
            return

        t = tick.copy()
        t["source"] = "poll"
        t["symbol"] = canonical_symbol
        t["instrument_token"] = token

        if "average_price" in t:
            avg_price = t["average_price"]
            try:
                t["vwap"] = float(avg_price) if avg_price else 0.0
            except (ValueError, TypeError):
                t["vwap"] = 0.0

        if market_data_manager is not None:
            try:
                market_data_manager._enqueue_tick_threadsafe(t)
            except Exception as exc:
                LOGGER.error("Tick enqueue failed: %s", exc)      
                    
    use_websockets = not use_polling
    if use_websockets:
        LOGGER.info("Initializing WebSocket Streamer...")

        def _sanitize_ws_token(value: str | None) -> str:
            token = (value or "").strip()
            if ":" in token:
                token = token.split(":", 1)[-1].strip()
            return token

        initial_token = _sanitize_ws_token(cast(str | None, config.broker.access_token))
        _ws_token_state: dict[str, str] = {"token": initial_token}
        _ws_token_timestamp: dict[str, float] = {
            "ts": time_module.time() if initial_token else 0.0,
        }

        def _resolve_ws_token() -> str:  # type: ignore[redefined-outer-name]
            """Resolve websocket token from Railway and legacy env aliases safely."""

            try:
                candidates = [
                    os.getenv("KITE_ACCESS_TOKEN"),
                    os.getenv("ZERODHA_ACCESS_TOKEN"),
                    os.getenv("BROKER_ACCESS_TOKEN"),
                    cast(str | None, getattr(config.broker, "access_token", None)),
                    _ws_token_state.get("token", ""),
                ]
                for candidate in candidates:
                    sanitized = _sanitize_ws_token(candidate)
                    if sanitized:
                        previous = _ws_token_state.get("token")
                        _ws_token_state["token"] = sanitized
                        if (
                            sanitized != previous
                            or float(_ws_token_timestamp.get("ts", 0.0)) <= 0.0
                        ):
                            _ws_token_timestamp["ts"] = time_module.time()
                        return sanitized
                return _ws_token_state.get("token", "")
            except Exception as e:
                LOGGER.error("Failure in _resolve_ws_token: %s", e)
                return _ws_token_state.get("token", "")

        def _resolve_ws_api_key() -> str:
            """Resolve websocket API key from Railway and legacy env aliases safely."""

            try:
                candidates = [
                    os.getenv("KITE_API_KEY"),
                    os.getenv("ZERODHA_API_KEY"),
                    os.getenv("BROKER_API_KEY"),
                    cast(str | None, getattr(config.broker, "api_key", None)),
                ]
                for candidate in candidates:
                    value = (candidate or "").strip()
                    if value:
                        return value
                return ""
            except Exception as e:
                LOGGER.error("Failure in _resolve_ws_api_key: %s", e)
                return ""

        websocket_manager = WebSocketManager(
            _resolve_ws_api_key(),
            _resolve_ws_token(),
            on_error=lambda err: LOGGER.error("WebSocket manager error: %s", err),
            backoff_min_sec=2.0,
            backoff_max_sec=30.0,
            stale_threshold_seconds=30.0,
            heartbeat_interval_seconds=20.0,
            heartbeat_timeout_seconds=10.0,
        )

        # WebSocketManager is the primary streamer in WS mode.
        streamer = websocket_manager

        # Wire up WebSocket handlers
        market_data_manager = MarketDataManager(
            broker_client,
            websocket_manager,
            settings=settings,
            resolver=instrument_manager,
        )

        # ── INJECT MDM REFERENCE INTO WEBSOCKET MANAGER ──────────────────────
        # WebSocketManager._on_ticks calls:
        #   getattr(self, "_market_data_manager", None).process_ticks(ticks)
        #   getattr(self, "_market_data_manager", None).update_authoritative_ticks()
        # Without this injection both branches are dead — update_authoritative_ticks
        # never fires, so is_data_stale() always reads stale wall-clock (last_tick_time=0).
        websocket_manager._market_data_manager = market_data_manager
        LOGGER.info("✅ WS: MDM reference injected into WebSocketManager")
        data_hub = DataHub(
            market_data_manager,
            options_only=True,
            store=hub_store,
            event_bus=message_bus,
        )

        polling_fallback_streamer = PollingStreamer(
            broker_client=broker_client,
            on_tick=_on_poll_tick,
            instrument_resolver=instrument_manager,
            data_hub=data_hub,
            poll_interval_ms=int(poll_interval_sec * 1000),
            batch_size=poll_batch_size,
            require_depth=poll_require_depth,
            warn_on_rate_limit=poll_warn_rate_limit,
        )
        market_data_manager.disable_rest_polling(reason="polling_streamer_fallback")
        market_data_manager.set_polling_fallback_streamer(polling_fallback_streamer)
        polling_fallback_streamer.set_websocket_mode(True)

        def _activate_polling_fallback() -> None:
            polling_fallback_streamer.set_websocket_mode(False)
            if not polling_fallback_streamer.is_running():
                LOGGER.warning(
                    "PollingStreamer fallback activated due to websocket degradation",
                    extra={"event": "polling_fallback_activated"},
                )
                polling_fallback_streamer.start()

        def _deactivate_polling_fallback() -> None:
            polling_fallback_streamer.set_websocket_mode(True)
            if polling_fallback_streamer.is_running():
                LOGGER.info(
                    "PollingStreamer fallback deactivated after websocket recovery",
                    extra={"event": "polling_fallback_deactivated"},
                )
                polling_fallback_streamer.stop()

        websocket_manager.set_fallback_callbacks(
            on_start=_activate_polling_fallback,
            on_stop=_deactivate_polling_fallback,
        )
        LOGGER.info(
            "PollingStreamer standby fallback armed (inactive until websocket degrades)",
            extra={"event": "polling_fallback_standby"},
        )

    else:
        LOGGER.info("Initializing Polling Streamer...")

        # ✅ FIX: Initialize Managers for Polling Mode FIRST
        # Note: WebSocketManager is None in polling mode
        market_data_manager = MarketDataManager(
            broker_client,
            None,
            settings=settings,
            resolver=instrument_manager,
        )

        # ✅ FIX: Create DataHub BEFORE PollingStreamer so it's not None
        data_hub = DataHub(
            market_data_manager,
            options_only=True,
            store=hub_store,
            event_bus=message_bus,
        )
        LOGGER.info(f"✅ DataHub created: {data_hub is not None}")

        # ✅ FIX: NOW Initialize Polling Streamer with valid data_hub
        streamer = PollingStreamer(
            broker_client=broker_client,
            on_tick=_on_poll_tick,
            instrument_resolver=instrument_manager,
            data_hub=data_hub,  # data_hub is now valid!
            poll_interval_ms=int(poll_interval_sec * 1000),
            batch_size=poll_batch_size,
            require_depth=poll_require_depth,
            warn_on_rate_limit=poll_warn_rate_limit,
        )
        LOGGER.info(f"✅ PollingStreamer initialized with DataHub wired")

    # ------------------------------------------------------------------
    # Common Supervisor & Wiring Logic
    # ------------------------------------------------------------------
    if data_hub is None:
        raise ConfigurationError("Data hub initialisation failed")

    LOGGER.info("DataHub initialized. Snapshot deferred to startup sequence.")

    # Initialize Supervisor
    def _risk_halt_active() -> bool:
        """Return whether the circuit breaker currently requests a stream halt.

        Args:
            None.

        Returns:
            bool: True when the circuit breaker is tripped, False otherwise.

        Raises:
            None.
        """

        manager = risk_manager_ref.get("instance")
        if manager is None:
            LOGGER.warning(
                "Condition met: risk manager unavailable during stream halt check",
                extra={"event": "stream_risk_halt_check_skipped"},
            )
            return False
        try:
            return bool(manager.is_circuit_breaker_tripped()[0])
        except Exception as exc:  # noqa: BLE001 - defensive stream guard
            LOGGER.error(
                "Failure in _risk_halt_active: %s",
                exc,
                extra={"event": "stream_risk_halt_check_failed"},
                exc_info=exc,
            )
            return False

    if use_polling:
        stream_supervisor = StreamSupervisor(
            streamer=streamer,
            resolver=instrument_manager,
            market_data_manager=market_data_manager,
            default_symbols=list(poll_symbols or ["NSE:NIFTY"]),
            autostart=True,
            monitor_interval_s=1.0,
            # Keep stream supervisor passive during breaker halts to avoid restart churn.
            risk_halt_getter=_risk_halt_active,
        )
        stream_supervisor.bootstrap()
        stream_supervisor.ensure_started()
        stream_supervisor_started = True
    else:
        # WebSocket mode: StreamSupervisor (which drives the *active* polling
        # streamer) is intentionally not created.  The PollingStreamer built
        # earlier at line ~3113 is armed as a **standby fallback** and is
        # promoted to active only if the WebSocket degrades (see
        # ``_activate_polling_fallback`` / ``WebSocketManager.set_fallback_callbacks``).
        # Log the state explicitly so operators understand fallback remains
        # available — the legacy ``polling_supervisor_disabled`` wording read
        # as a regression but was actually normal for WS mode.
        stream_supervisor = None
        stream_supervisor_started = False
        fallback_ready = polling_fallback_streamer is not None
        LOGGER.info(
            "polling_supervisor_standby mode=websocket fallback_armed=%s",
            fallback_ready,
            extra={
                "event": "polling_supervisor_standby",
                "mode": "websocket",
                "fallback_armed": fallback_ready,
            },
        )

    # Initialize Indicators & Regime
    indicator_engine = IndicatorEngine()
    for env_key, env_default in (
        ("REGIME_MIN_CONFIDENCE", "0.40"),
        ("REGIME_STALE_AFTER_SEC", "300"),
        ("REGIME_BLOCK_EVENT", "0.80"),
        ("REGIME_BLOCK_VOLATILE", "0.95"),
        ("REGIME_FAIL_CLOSED", "0"),
        ("STRATEGY_ENFORCE_BLOCKLIST", "0"),
    ):
        os.environ.setdefault(env_key, env_default)

    regime_symbol = "NIFTY"
    if poll_symbols:
        regime_symbol = poll_symbols[0]

    market_regime_detector = MarketRegimeDetector()
    market_regime_manager = MarketRegimeManager(
        market_regime_detector,
        datahub=data_hub,
        indicators=indicator_engine,
        regime_settings={
            "symbol": regime_symbol,
            "update_interval_sec": 60,
            "atr_trend_threshold": 1.5,
            "vol_threshold": 25.0,
            "history_length": 1440,
        },
    )

    # Wire Regime Detector to Stream
    try:
        if hasattr(data_hub, "subscribe_ticks") and hasattr(
            market_regime_detector, "ingest_tick"
        ):
            callback = cast(
                Callable[[dict[str, Any]], None],
                market_regime_detector.ingest_tick,
            )
            data_hub.subscribe_ticks(regime_symbol, callback)
            LOGGER.debug(f"Regime detector subscribed via DataHub to {regime_symbol}")

        elif hasattr(streamer, "register_handler") and hasattr(
            market_regime_detector, "ingest_tick"
        ):
            streamer.register_handler(market_regime_detector.ingest_tick)
            LOGGER.debug(f"Regime detector subscribed via Streamer to {regime_symbol}")

    except Exception as exc:
        LOGGER.error(
            "Regime detector tick subscription failed",
            extra={"event": "regime_subscription_failed", "symbol": regime_symbol},
            exc_info=exc,
        )
    data_dir = os.getenv("DATA_DIR", "data")
    persistent_state = PersistentStateManager(base_path=Path(data_dir))

    heartbeat_interval = max(
        1.0,
        coalesce_float(
            "PERSISTENCE__HEARTBEAT_FLUSH_SEC",
            "PERSISTENCE_HEARTBEAT_FLUSH_SEC",
            default=5.0,
        ),
    )
    heartbeat_flusher = PersistentHeartbeatFlusher(
        persistent_state, interval_sec=heartbeat_interval
    )
    market_data_manager.register_heartbeat_callback(heartbeat_flusher.handle_heartbeat)

    position_state_path = Path(data_dir) / "positions.json"
    position_manager = PositionManager(state_file=str(position_state_path))
    position_manager.attach_persistent_state(persistent_state)

    broker_sync_attempts = max(
        1,
        coalesce_int(
            "BROKER_SYNC_MAX_ATTEMPTS",
            "BROKER_SYNC_RETRY_ATTEMPTS",
            default=5,
        ),
    )
    broker_sync_backoff_min = max(
        0.25,
        coalesce_float(
            "BROKER_SYNC_BACKOFF_MIN_SEC",
            default=1.0,
        ),
    )
    broker_sync_backoff_max = max(
        broker_sync_backoff_min,
        coalesce_float(
            "BROKER_SYNC_BACKOFF_MAX_SEC",
            default=15.0,
        ),
    )
    broker_sync_backoff_multiplier = max(
        1.0,
        coalesce_float(
            "BROKER_SYNC_BACKOFF_MULT",
            default=2.0,
        ),
    )
    broker_sync_jitter = max(
        0.0,
        min(
            0.5,
            coalesce_float(
                "BROKER_SYNC_BACKOFF_JITTER",
                default=0.2,
            ),
        ),
    )

    _hydrate_positions(
        position_manager=position_manager,
        persistent_state=persistent_state,
        broker_client=broker_client,
        data_hub=data_hub,
        logger=LOGGER,
        max_attempts=broker_sync_attempts,
        backoff_min=broker_sync_backoff_min,
        backoff_max=broker_sync_backoff_max,
        backoff_multiplier=broker_sync_backoff_multiplier,
        jitter_fraction=broker_sync_jitter,
        total_timeout_sec=60.0,
    )

    initial_balance = float(
        getattr(config, "initial_balance", 1_000_000.0) or 1_000_000.0
    )

    # 1. Initialize Risk Manager
    risk_manager = RiskManager(
        settings=settings.risk,
        position_manager=position_manager,
        account_balance=initial_balance,
    )
    risk_manager_ref["instance"] = risk_manager

    # 2. Attach Broker (Safe Try/Except)
    try:
        risk_manager.set_broker_client(broker_client)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("risk_manager_attach_broker_failed: %s", exc)

    # 3. Attach Market Data (Safe Try/Except)
    try:
        risk_manager.set_market_data_manager(market_data_manager)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("risk_manager_attach_mdm_failed: %s", exc)

    if not risk_manager_data_hub_attached:
        try:
            risk_manager.attach_data_hub(data_hub)
            risk_manager_data_hub_attached = True
            LOGGER.info("✅ Wired DataHub to Risk Manager")
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("risk_manager_attach_data_hub_failed: %s", exc)

    # 5. Wire Lot Size Provider using InstrumentManager as source of truth.
    def _lot_size_lookup(symbol: str) -> int:
        try:
            lot = instrument_manager.get_lot_size(symbol)
            if lot and lot > 0:
                if "NIFTY" in symbol.upper():
                    LOGGER.info(
                        "LOT_SIZE_RESOLVED underlying=NIFTY lot_size=%s source=instrument_dump",
                        int(lot),
                        extra={"event": "LOT_SIZE_RESOLVED", "underlying": "NIFTY", "lot_size": int(lot), "source": "instrument_dump"},
                    )
                return int(lot)
            # Fallback defaults
            sym_upper = symbol.upper()
            if "NIFTY" in sym_upper:
                env_lot = int(getattr(settings.risk, "contract_lot_size", 65) or 65)
                source = "env"
                if env_lot <= 0:
                    env_lot = 65
                    source = "fallback"
                LOGGER.info(
                    "LOT_SIZE_RESOLVED underlying=NIFTY lot_size=%s source=%s",
                    env_lot,
                    source,
                    extra={"event": "LOT_SIZE_RESOLVED", "underlying": "NIFTY", "lot_size": env_lot, "source": source},
                )
                return env_lot
            if "BANKNIFTY" in sym_upper:
                return 15
            return 1
        except Exception:
            if "NIFTY" in symbol.upper():
                LOGGER.info(
                    "LOT_SIZE_RESOLVED underlying=NIFTY lot_size=65 source=fallback",
                    extra={"event": "LOT_SIZE_RESOLVED", "underlying": "NIFTY", "lot_size": 65, "source": "fallback"},
                )
                return 65
            return 1

    # Always attach the provider
    risk_manager.set_lot_size_provider(_lot_size_lookup)
    LOGGER.info("✅ Wired Lot Size Provider to Risk Manager")

    risk_state: RiskState | None = None
    try:
        spread_mult = max(
            coalesce_float(
                "RISK_STATE_SPREAD_MULT",
                "RISK_STATE_SPREAD_WIDEN_MULT",
                default=3.0,
            ),
            1.0,
        )
        dd_limit = coalesce_float(
            "RISK_STATE_MAX_INTRADAY_DD",
            "RISK_STATE_MAX_DRAWDOWN",
            default=0.0,
        )
        if abs(dd_limit) <= 0.0:
            pct_cap = float(config.risk.max_drawdown_pct or 0.0) / 100.0
            dd_limit = -abs(initial_balance * pct_cap) if pct_cap > 0 else 0.0
        else:
            dd_limit = -abs(dd_limit)
        loss_cap = max(
            coalesce_int(
                "RISK_STATE_MAX_CONSECUTIVE_LOSSES",
                default=settings.risk.max_consecutive_losses,
            ),
            1,
        )
        risk_state = RiskState(
            quote_stale_ms_threshold=int(config.quote_stale_threshold_ms),
            spread_widen_mult=float(spread_mult),
            max_intraday_dd=float(dd_limit),
            max_consecutive_losses=int(loss_cap),
        )
        risk_manager.bind_risk_state(risk_state)
    except Exception:  # pragma: no cover - defensive wiring
        LOGGER.exception("risk_state_init_failed")
        risk_state = None

    paper_engine = PaperFillEngine(data_hub, instrument_manager)
    live_toggle_env = coalesce_bool(
        "ENABLE_LIVE_TRADING",
        "ENABLE_LIVE",
        default=settings.enable_live,
    )
    live_possible = bool(live_toggle_env and settings.orders.enable_live)
    paper_toggle_env = coalesce_bool("PAPER__ENABLED", default=not live_possible)
    shadow_mode_env = get_bool("SHADOW_MODE", default=not live_possible)
    paper_initial = bool((not live_possible) or paper_toggle_env or shadow_mode_env)
    broker_backend = robust_provider if not paper_initial else paper_engine

    # 9. Initialize Execution
    # [FIX] We inject indicator_engine here to enable Volatility-Adaptive Trailing
    trade_journal = TradeJournal(db_path=str(get_data_dir() / "trades.db"))
    trade_journal.start()
    order_manager = OrderManager(
        broker_client=broker_client,
        position_manager=position_manager,
        rate_limiter=rate_limiter,
        instrument_resolver=instrument_manager,
        indicator_engine=indicator_engine,  # <--- THIS IS THE CRITICAL ADDITION
        trade_journal=trade_journal,
    )
    order_manager.set_market_data_manager(market_data_manager)
    if not margin_engine_data_hub_attached:
        # OrderManager owns MarginEngine wiring; attach once to avoid duplicate callbacks.
        try:
            order_manager.attach_data_hub(data_hub)
            margin_engine_data_hub_attached = True
            LOGGER.info("✅ Wired DataHub to Margin Engine")
        except Exception as exc:
            LOGGER.error("margin_engine_attach_data_hub_failed: %s", exc)
    else:
        LOGGER.info("ℹ️ Margin Engine already attached to DataHub")
    order_manager.set_instrument_resolver(instrument_manager)
    order_manager.set_risk_manager(risk_manager)
    order_manager.attach_persistent_state(persistent_state)

    bracket_manager: BracketManager | None = None
    # ----------------------------------------------------------------
    # 1. Initialize BracketManager (Clean)
    # ----------------------------------------------------------------
    if settings.execution.enable_bracket_manager:
        try:
            LOGGER.debug("Initializing BracketManager...")

            bracket_manager = BracketManager(
                order_manager=order_manager,
                indicator_engine=indicator_engine,
                # Pass DataHub (SSOT) as the market-data facade; bracket manager
                # only needs subscribe/unsubscribe + quote access, which DataHub
                # exposes and transparently delegates to MDM.
                market_data=data_hub if 'data_hub' in locals() and data_hub else market_data_manager,
                trade_journal=trade_journal,
            )

            # Configure
            bracket_manager._auto_reduce_sl = settings.execution.bracket_auto_reduce_sl
            bracket_manager._stale_cleanup_age = getattr(
                settings.execution, "bracket_stale_cleanup_seconds", 86400
            )

            # Attach once; duplicate attachment can duplicate bracket listeners.
            if not bracket_manager_attached:
                order_manager.set_bracket_manager(bracket_manager=bracket_manager)
                bracket_manager_attached = True
                LOGGER.info("✅ BracketManager initialized and attached.")

        except Exception as exc:
            LOGGER.error(f"Failed to initialize BracketManager: {exc}")

    if risk_state is not None:

        def _sync_risk_state_pnl() -> None:
            """Synchronize risk state and Prometheus PnL metrics.

            Args:
                None.

            Returns:
                None.

            Raises:
                None.
            """

            try:
                realized = float(position_manager.get_realized_pnl())
            except Exception:  # pragma: no cover - defensive
                realized = 0.0
            try:
                unrealized = float(position_manager.get_unrealized_pnl())
            except Exception:  # pragma: no cover - defensive
                unrealized = 0.0
            try:
                METRICS.set_pnl_breakdown(
                    book="primary", realized=realized, unrealized=unrealized
                )
            except Exception:  # pragma: no cover - optional metrics
                LOGGER.debug("Unable to sync pnl breakdown", exc_info=True)
            risk_state.on_trade_update(realized_pnl=realized, unrealized_pnl=unrealized)

        def _first_float(payload: Mapping[str, Any], *keys: str) -> float | None:
            for key in keys:
                value = payload.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return None

        def _risk_state_tick_listener(tick: Mapping[str, Any]) -> None:
            bid = _first_float(
                tick,
                "best_bid",
                "best_bid_price",
                "bid",
                "buy_price",
            )
            ask = _first_float(
                tick,
                "best_ask",
                "best_ask_price",
                "ask",
                "sell_price",
            )
            if bid is None or ask is None:
                return
            ts_ns = time_module.time_ns()
            risk_state.on_tick(bid, ask, ts_ns=ts_ns)
            _sync_risk_state_pnl()

        def _risk_state_order_listener(_order: Mapping[str, Any]) -> None:
            _sync_risk_state_pnl()

        _sync_risk_state_pnl()
        risk_symbol = coalesce_str(
            "RISK_STATE_SYMBOL",
            "RISK_STATE__SYMBOL",
            default="NSE:NIFTY",
        )
        attach = getattr(risk_state, "attach_data_hub", None)
        if callable(attach):
            try:
                attach(data_hub, symbol=risk_symbol)
            except Exception:  # pragma: no cover - defensive
                LOGGER.debug("risk_state_attach_data_hub_failed", exc_info=True)
        if hasattr(data_hub, "subscribe_ticks") and risk_symbol:
            _subscribe_ticks_force_live_compat(data_hub, risk_symbol, _risk_state_tick_listener)
        if hasattr(data_hub, "subscribe_orders"):
            data_hub.subscribe_orders(_risk_state_order_listener)

    safe_order_manager = SafeOrderManager(
        order_manager=order_manager,
        settings=settings.orders,
        regime_manager=market_regime_manager,
    )

    session_guard = TradingSessionGuard(
        rate_limiter=rate_limiter,
        risk_manager=risk_manager,
        allow_out_of_hours=allow_offhours_testing_safe(),
    )
    session_allow_override = allow_offhours_testing_safe()
    session_guard.set_allow_out_of_hours(session_allow_override)
    settings.session_allow_out_of_hours = session_allow_override
    current_open, current_close = session_guard.get_trading_window()
    default_open = f"{current_open.hour:02d}:{current_open.minute:02d}"
    default_close = f"{current_close.hour:02d}:{current_close.minute:02d}"
    # ---- Fix/normalize PNL path if provided ----
    pnl_path_raw = get_str("PNL_PERSIST_PATH")
    pnl_path = normalize_path(pnl_path_raw)
    if pnl_path:
        os.environ["PNL_PERSIST_PATH"] = pnl_path

    trading_window_start = (
        get_str("DATA__TIME_FILTER_START", default_open) or default_open
    )
    trading_window_end = (
        get_str("DATA__TIME_FILTER_END", default_close) or default_close
    )
    session_guard.set_trading_window(trading_window_start, trading_window_end)

    # ----------------------------------------------------------------------
    # 6. Build Elite Strategies (Corrected)
    # ----------------------------------------------------------------------
    elite_strategies: list[Any] = []
    try:
        # ✅ FIX: Pass 'indicator_engine' to the builder
        elite_strategies = build_elite_strategies(
            settings.elite,  # Verify if this is settings.elite or settings.strategies in your config
            indicator_engine,
        )

        # Inject DataHub into strategies that need it (e.g. OrderFlow, Gamma)
        if elite_strategies and data_hub:
            for strategy in elite_strategies:
                if hasattr(strategy, "set_data_hub"):
                    try:
                        strategy.set_data_hub(data_hub)
                    except Exception as exc:
                        LOGGER.warning(
                            "Failed to inject DataHub into %s: %s",
                            getattr(strategy, "name", "unknown"),
                            exc,
                        )

    except Exception as exc:  # noqa: BLE001
        # ✅ FIX: Proper indentation for the except block
        LOGGER.error(
            "Failure in build_elite_strategies: %s",
            exc,
            exc_info=exc,
            extra={"event": "elite_build_error"},
        )
    else:
        # ✅ FIX: Logic to run only if NO exception occurred
        if elite_strategies:
            LOGGER.info(
                "Condition met: elite_strategies_loaded",
                extra={
                    "event": "elite_strategies_loaded",
                    "count": len(elite_strategies),
                },
            )
        else:
            LOGGER.warning(
                "No elite strategies enabled; trading will be disabled",
                extra={"event": "elite_strategies_missing"},
            )

        # Ensure DataHub is injected into strategies that need complex metrics (IV/Greeks)
        if elite_strategies and data_hub:
            for strategy in elite_strategies:
                if hasattr(strategy, "set_data_hub"):
                    try:
                        strategy.set_data_hub(data_hub)
                        LOGGER.debug(f"Injected DataHub into {strategy.name}")
                    except Exception as exc:
                        LOGGER.warning(
                            f"Failed to inject DataHub into {strategy.name}: {exc}"
                        )

    strategy_instances: list[Any] = list(elite_strategies)
    # Ensure DataHub is injected into all strategies that need enriched data (IV/Greeks).
    if strategy_instances and data_hub:
        for strategy in strategy_instances:
            # Check if the strategy has the required setter method (set_data_hub)
            if hasattr(strategy, "set_data_hub"):
                try:
                    strategy.set_data_hub(data_hub)
                    LOGGER.debug(f"Injected DataHub into {strategy.name}")
                except Exception as exc:
                    LOGGER.warning(
                        f"Failed to inject DataHub into {strategy.name}: {exc}"
                    )
    futures_symbol = ""
    for _method_name in ("get_active_nifty_future_symbol_cached", "resolve_active_nifty_future_symbol"):
        _method = getattr(market_data_manager, _method_name, None)
        if callable(_method):
            try:
                futures_symbol = canonical_nifty_future_symbol(_method()) or ""
            except TypeError:
                futures_symbol = canonical_nifty_future_symbol(_method(now=None)) or ""
            except Exception:
                futures_symbol = ""
            if futures_symbol:
                break
    LOGGER.info("📅 Using active futures symbol: %s", futures_symbol or "unavailable")
    orchestrator = StrategyOrchestrator(
        risk_manager=risk_manager,
        order_manager=safe_order_manager,
        data_hub=data_hub,
        futures_symbol=futures_symbol,
    )
    regime_bias_map: dict[str, dict[str, float]] = {}
    if elite_strategies:
        # ✅ FIX: FORCE-BOOST Position Size to cover Orphans (Capital Block Fix)
        # Prevents 'orchestrator_capital_block' infinite loop when an orphan trade exists.
        raw_pct = settings.elite.position_size_pct
        if raw_pct < 15.0:
            LOGGER.warning(
                f"⚠️ FORCE-BOOSTING Position Size from {raw_pct}% to 15.0% to cover Orphans",
                extra={"event": "elite_fraction_boosted", "original": raw_pct},
            )
            raw_pct = 15.0

        elite_fraction = raw_pct / 100.0

        if elite_fraction <= 0:
            LOGGER.warning(
                "Elite position size pct not positive; defaulting to 1% of capital.",
                extra={"event": "elite_fraction_default"},
            )
            elite_fraction = 0.01
        tag_lookup = elite_strategy_tags(settings.elite)
        # Optimization B: Dynamic Sizing based on Regime
        # Trend = Aggressive (100% size), Chop = Conservative (50% size)
        bias_candidates: dict[str, dict[str, float]] = {
            "trend": {},
            "chop": {},
            "volcrush": {},
            "event": {},
        }
        for strategy in elite_strategies:
            tags = tag_lookup.get(strategy.name, ("elite",))
            try:
                orchestrator.register_strategy(
                    strategy.name,
                    capital_fraction=elite_fraction,
                    correlation_tags=tags,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Failure registering elite strategy %s: %s",
                    strategy.name,
                    exc,
                    exc_info=exc,
                    extra={"event": "elite_register_error", "strategy": strategy.name},
                )
                continue
            tag_set = {tag.lower() for tag in tags}
            if {"momentum", "opening", "orderflow"} & tag_set:
                bias_candidates["trend"][strategy.name] = 1.15
            if {"mean_reversion", "liquidity"} & tag_set:
                bias_candidates["chop"][strategy.name] = 1.2
            if {"volatility", "income"} & tag_set:
                bias_candidates["volcrush"][strategy.name] = 1.15
            if {"structure", "orderflow", "liquidity"} & tag_set:
                bias_candidates["event"][strategy.name] = 1.1
        regime_bias_map = {
            regime: mapping for regime, mapping in bias_candidates.items() if mapping
        }
    else:
        tag_lookup = {}

    def _regime_signal_snapshot() -> dict[str, object] | None:
        """Return lightweight market regime snapshot for scoring.

        Args:
            None.

        Returns:
            dict[str, object] | None: Normalised regime snapshot when
            available.

        Raises:
            None.
        """

        snapshot = market_regime_manager.get_latest_snapshot()
        if snapshot is None:
            return None
        return {
            "regime": snapshot.regime,
            "confidence": snapshot.confidence,
            "updated_at": snapshot.updated_at,
        }

    # ✅ FIX #1: min_confidence was missing → defaulted to 0.60 in parent class.
    # After _apply_weighted_confidence multiplies by score weight (~0.5 cold start),
    # VWAPPro's 0.85 confidence becomes 0.425 < 0.60 → ALL signals silently filtered.
    # Correct threshold is post-weighting aware: 0.35 allows signals to pass.
    _global_min_conf = float(os.getenv("GLOBAL_MIN_SIGNAL_CONFIDENCE", "0.35"))
    strategy_manager = StrategyManager(
        strategies=strategy_instances,
        indicator_engine=indicator_engine,
        position_manager=position_manager,
        min_confidence=_global_min_conf,
        data_hub=data_hub,
        orchestrator=orchestrator,
        futures_symbol=futures_symbol,
        score_weights=None,
        regime_signal_getter=_regime_signal_snapshot,
        regime_bias_map=regime_bias_map,
        market_regime_manager=market_regime_manager,
    )

    unified_manager: UnifiedManager | None = None
    try:
        LOGGER.debug(
            "Entered initialize_components unified manager wiring",
            extra={"event": "init.unified_manager.enter"},
        )
        unified_manager = UnifiedManager(
            broker=broker_client,
            mdm=market_data_manager,
            ws=websocket_manager,
            risk=risk_manager,
            streamer=streamer,
            data_hub=data_hub,
            orders=safe_order_manager or order_manager,
            strategies=strategy_manager,
            logger=LOGGER.getChild("init.unified_manager"),
        )
        with suppress(Exception):
            market_data_manager.set_unified_manager(unified_manager)
        with suppress(Exception):
            risk_manager.set_unified_manager(unified_manager)
        LOGGER.info(
            "Condition met: unified_manager_wired",
            extra={
                "event": "init.unified_manager.ready",
                "has_safe_order_manager": bool(safe_order_manager),
                "has_strategy_manager": bool(strategy_manager),
            },
        )
    except Exception as exc:  # noqa: BLE001 - defensive wiring
        LOGGER.error(
            "Failure in initialize_components unified manager wiring: %s",
            exc,
            extra={"event": "init.unified_manager.error"},
            exc_info=exc,
        )
        unified_manager = None

    strike_selector: StrikeSelector | None = None
    if data_hub is not None:
        strike_selector = StrikeSelector(
            data_hub=data_hub,
            selector_settings=settings.selector,
            liquidity_settings=settings.liquidity,
        )

    state_tracker = StateTracker()
    lifecycle_tracker_adapter = _LifecycleTrackerAdapter(state_tracker)

    lifecycle_manager = LifecycleManager(
        data_hub=data_hub,
        state_tracker=lifecycle_tracker_adapter,
    )
    reconciliation_interval = coalesce_int("RECONCILIATION_INTERVAL_SEC", default=30)
    reconciliation_alert = coalesce_bool(
        "RECONCILIATION_ALERT_ON_MISMATCH", default=True
    )
    post_fill_monitor = PostFillMonitor(
        broker_client=robust_provider,
        state_tracker=state_tracker,
        interval_sec=int(reconciliation_interval),
        alert_on_mismatch=bool(reconciliation_alert),
    )
    

    strategy_runner = StrategyRunner(
        market_data_manager=market_data_manager,
        indicator_engine=indicator_engine,
        strategy_manager=strategy_manager,
        order_manager=order_manager,
        risk_manager=risk_manager,
        position_manager=position_manager,
        config=_get_strategy_config(config),
        data_hub=data_hub,
        strike_selector=strike_selector,
        message_bus=message_bus,
        bracket_manager=bracket_manager,
    )
    strategy_runner.attach_persistent_state(persistent_state)
    market_data_manager.subscribe_bars(strategy_runner.ingest_historical_bar)
    strategy_runner.restore_trades(persistent_state.load_trades())
    settings.enable_live = bool(live_toggle_env)
    mandatory_paper = not live_possible
    paper_state: dict[str, bool] = {"enabled": bool(paper_initial or mandatory_paper)}
    ctx_ref: dict[str, BotContext | None] = {"ctx": None}

    def _apply_paper_mode(enabled: bool) -> bool:
        desired = bool(enabled)
        next_state = bool(mandatory_paper or desired)
        paper_state["enabled"] = next_state
        backend = paper_engine if next_state else broker_client
        order_manager.set_broker_client(backend)
        orders_enabled_now = next_state or live_possible
        safe_order_manager.set_live_enabled(orders_enabled_now)
        risk_manager.force_shadow(next_state)
        ctx_obj = ctx_ref.get("ctx")
        if ctx_obj is not None:
            ctx_obj.shadow_mode_enabled = next_state
        target_mode = "PAPER" if next_state else ("LIVE" if settings.enable_live else "PAPER")
        return paper_state["enabled"]

    def _paper_mode_enabled() -> bool:
        return paper_state["enabled"]

    _apply_paper_mode(paper_state["enabled"])
    shadow_enabled = paper_state["enabled"]
    notifier: TelegramEnhancedNotifier | None = None

    if not ctx_ref.get("telegram_wired", False):
        TelegramEnhancedNotifierCls = _load_telegram_enhanced_notifier()
        if TelegramEnhancedNotifierCls is None:
            notifier = None
            order_manager.set_notifier(None)
            LOGGER.warning("TELEGRAM_NOTIFIER_SKIPPED reason=dependency_unavailable")
        else:
            try:
                notifier = TelegramEnhancedNotifierCls.from_settings(settings.notifications)
                order_manager.set_notifier(notifier)
                ctx_ref["telegram_wired"] = True
                LOGGER.info("✅ Telegram Notifier wired to Order Manager")
            except Exception:
                notifier = None
                order_manager.set_notifier(None)
                LOGGER.exception("Telegram notifier wiring failed")
    else:
        LOGGER.info("ℹ️ Telegram Notifier already wired")

    telegram_logger = get_logger("telegram")
    telegram_mode = (
        "webhook"
        if (
            settings.notifications.enabled
            and settings.notifications.webhook_enabled
            and settings.notifications.public_base_url
        )
        else "polling"
    )
    telegram_logger.info(
        "Telegram controller starting in %s mode",
        telegram_mode,
        extra={
            "event": "telegram_mode",
            "mode": telegram_mode,
            "webhook_env_enabled": telegram_webhook_env_enabled,
            "notifications_enabled": settings.notifications.enabled,
        },
    )

    # Position reconciliation is now handled safely in startup_sequence and _health_loop.

    background_tasks: list[asyncio.Task[Any]] = []
    if not ctx_ref.get("background_tasks_started", False):
        try:
            background_tasks = start_background_tasks(order_manager, LOGGER)
            ctx_ref["background_tasks_started"] = True
            LOGGER.info(
                "Background tasks started",
                extra={
                    "event": "background_tasks.started",
                    "count": len(background_tasks),
                },
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failed to start background tasks",
                extra={
                    "event": "background_tasks.failed",
                    "error": str(exc),
                },
                exc_info=exc,
            )
    else:
        LOGGER.info("ℹ️ Background tasks already started")

    refresh_task = schedule_instrument_refresh(
        settings,
        instrument_manager,
        state=None,
    )
    if refresh_task is not None:
        background_tasks.append(refresh_task)
        # keep a handle for graceful shutdown
        instrument_refresh_task: asyncio.Task[Any] | None = refresh_task
    else:
        instrument_refresh_task = None

    async def cleanup_stale_brackets_task() -> None:
        """Periodically remove stale bracket state entries.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered cleanup_stale_brackets_task",
            extra={"event": "bracket.cleanup.task.enter"},
        )
        while True:
            try:
                await asyncio.sleep(3600)
                removed = bracket_manager.cleanup_stale_brackets(
                    max_age_seconds=settings.execution.bracket_stale_cleanup_seconds,
                )
                if removed > 0:
                    LOGGER.info(
                        "Cleaned up stale brackets",
                        extra={
                            "event": "bracket.cleanup.completed",
                            "count": removed,
                            "max_age": (
                                settings.execution.bracket_stale_cleanup_seconds
                            ),
                        },
                    )
            except (
                asyncio.CancelledError
            ):  # pragma: no cover - cooperative cancellation
                LOGGER.info(
                    "Bracket cleanup task cancelled",
                    extra={"event": "bracket.cleanup.task.cancelled"},
                )
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Bracket cleanup task error: %s",
                    exc,
                    extra={"event": "bracket.cleanup.task.error"},
                    exc_info=exc,
                )

    background_tasks.append(asyncio.create_task(cleanup_stale_brackets_task()))

    def _build_health_snapshot() -> dict[str, object]:
        """Return an aggregate health snapshot for out-of-band alerts.

        Args:
            None.

        Returns:
            Dictionary containing guard, risk, order, and market data signals.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered _build_health_snapshot",
            extra={"event": "health_snapshot_build_enter"},
        )
        snapshot: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "shadow" if paper_state["enabled"] else "live",
        }
        try:
            guard_status = session_guard.evaluate()
        except Exception as exc:  # noqa: BLE001 - defensive
            LOGGER.error(
                "Failure in _build_health_snapshot guard: %s",
                exc,
                extra={"event": "health_snapshot_guard_error"},
            )
            snapshot["session"] = {"error": str(exc)}
        else:
            snapshot["session"] = guard_status.as_dict()

        mdm_payload: dict[str, object] = {}
        if market_data_manager is None:
            mdm_payload = {"error": "unavailable"}
        else:
            try:
                status = market_data_manager.mdm_status()
                mdm_payload = {
                    "ws_connected": bool(status.get("ws_connected")),
                    "fallback_enabled": bool(status.get("fallback_enabled")),
                    "heartbeat_age": status.get("heartbeat_age"),
                    "last_tick_source": status.get("last_tick_source", {}),
                    "last_tick_age": status.get("last_tick_age", {}),
                }
            except Exception as exc:  # noqa: BLE001 - defensive
                LOGGER.error(
                    "Failure in _build_health_snapshot mdm: %s",
                    exc,
                    extra={"event": "health_snapshot_mdm_error"},
                )
                mdm_payload = {"error": str(exc)}
        snapshot["market_data"] = mdm_payload

        try:
            risk_snapshot = risk_manager.snapshot()
            snapshot["risk"] = {
                "breaker_tripped": risk_snapshot.breaker_tripped,
                "cooldown_remaining": risk_snapshot.cooldown_remaining,
                "losses_in_row": risk_snapshot.losses_in_row,
                "last_rejection": risk_snapshot.last_rejection,
                "timestamp": risk_snapshot.timestamp.isoformat(),
            }
        except Exception as exc:  # noqa: BLE001 - defensive
            LOGGER.error(
                "Failure in _build_health_snapshot risk: %s",
                exc,
                extra={"event": "health_snapshot_risk_error"},
            )
            snapshot["risk"] = {"error": str(exc)}

        try:
            open_positions = list(position_manager.get_open_positions())
            snapshot["positions"] = {"open": len(open_positions)}
        except Exception as exc:  # noqa: BLE001 - defensive
            LOGGER.error(
                "Failure in _build_health_snapshot positions: %s",
                exc,
                extra={"event": "health_snapshot_position_error"},
            )
            snapshot["positions"] = {"error": str(exc)}

        try:
            recent = order_manager.recent_orders(limit=5)
            snapshot["orders"] = {"recent": len(recent)}
        except Exception as exc:  # noqa: BLE001 - defensive
            LOGGER.error(
                "Failure in _build_health_snapshot orders: %s",
                exc,
                extra={"event": "health_snapshot_orders_error"},
            )
            snapshot["orders"] = {"error": str(exc)}

        return snapshot

    def _notify(event: str, payload: Mapping[str, object] | None = None) -> None:
        if notifier is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - initialization phase
            LOGGER.debug(
                "No running loop to dispatch telegram notification",
                extra={"event": event},
            )
            return
        loop.create_task(notifier.send_event(event, payload))

    def _emit_health_snapshot(trigger: str, detail: str | None = None) -> None:
        """Dispatch a health snapshot notification for high-impact events.

        Args:
            trigger: Identifier describing the initiating condition.
            detail: Optional human-readable detail string.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.info(
            "Condition met: emitting health snapshot",
            extra={
                "event": "health_snapshot_emit",
                "trigger": trigger,
                "detail": detail,
            },
        )
        payload: dict[str, object] = {
            "trigger": trigger,
            "detail": detail,
            "snapshot": _build_health_snapshot(),
        }
        _notify("HEALTH_SNAPSHOT", payload)

    shadow_trader: ShadowPaperTrader | None = None
    if settings.shadow.drift_threshold_pct > 0:

        def _disable_live(_reason: str) -> None:
            safe_order_manager.set_live_enabled(False)

        shadow_trader = ShadowPaperTrader(
            settings=settings.shadow,
            live_equity_fn=position_manager.get_net_pnl,
            notifier=_notify if notifier is not None else None,
            disable_live_callback=_disable_live,
        )
        shadow_trader.attach_persistent_state(persistent_state)
        safe_order_manager.post_order_hook = (
            lambda symbol, side, qty, price: shadow_trader.record_order(
                symbol, side, qty, price or 0.0
            )
        )

    health_state = HealthState(
        streamer=streamer,
        order_manager=safe_order_manager,
        risk_manager=risk_manager,
        live_enabled=lambda: safe_order_manager.settings.enable_live,
        session_guard=session_guard,
        stream_supervisor=stream_supervisor,
        websocket_enabled=websocket_enabled,
    )
    health_app = create_health_app(health_state)

    option_universe_manager = OptionUniverseManager(settings.option_universe)

    ctx = BotContext(
        settings=settings,
        config=config,
        rate_limiter=rate_limiter,
        broker_client=robust_provider,
        message_bus=message_bus,
        websocket_client=websocket_client,
        websocket_manager=websocket_manager,
        streamer=streamer,
        stream_supervisor=stream_supervisor,
        polling_fallback_streamer=polling_fallback_streamer,
        data_hub=data_hub,
        market_data_manager=market_data_manager,
        market_regime=market_regime_detector,
        market_regime_manager=market_regime_manager,
        indicator_engine=indicator_engine,
        position_manager=position_manager,
        risk_manager=risk_manager,
        persistent_state=persistent_state,
        trade_journal=trade_journal,
        order_manager=order_manager,
        bracket_manager=bracket_manager,
        paper_engine=paper_engine,
        safe_order_manager=safe_order_manager,
        state_tracker=state_tracker,
        lifecycle_manager=lifecycle_manager,
        post_fill_monitor=post_fill_monitor,
        strategy_manager=strategy_manager,
        strategy_runner=strategy_runner,
        unified_manager=unified_manager,
        instrument_manager=instrument_manager,
        instrument_db=None,
        instrument_universe=None,
        instrument_refresh_task=instrument_refresh_task,
        websocket_enabled=websocket_enabled,
        shadow_mode_enabled=shadow_enabled,
        shadow_trader=shadow_trader,
        out_of_hours_override=False,
        telegram_bot=None,
        telegram_application=None,
        telegram_notifier=notifier,
        health_app=health_app,
        session_guard=session_guard,
        option_universe=option_universe_manager,
        stream_supervisor_started=stream_supervisor_started,
        margin_engine_data_hub_attached=margin_engine_data_hub_attached,
        risk_manager_data_hub_attached=risk_manager_data_hub_attached,
        bracket_manager_attached=bracket_manager_attached,
    )

    # Wire InstrumentManager to MDM and UnifiedManager for symbol/token resolution
    if ctx.instrument_manager is not None:
        if ctx.market_data_manager is not None:
            try:
                setattr(ctx.market_data_manager, "_resolver", ctx.instrument_manager)
            except Exception as exc:  # noqa: BLE001 - defensive wiring
                LOGGER.error(
                    "InstrumentManager attach to MDM failed: %s",
                    exc,
                    extra={"event": "resolver_attach_mdm_failed"},
                    exc_info=exc,
                )
        if unified_manager is not None:
            try:
                setattr(unified_manager, "resolver", ctx.instrument_manager)
            except Exception as exc:  # noqa: BLE001 - defensive wiring
                LOGGER.error(
                    "InstrumentManager attach to UM failed: %s",
                    exc,
                    extra={"event": "resolver_attach_um_failed"},
                    exc_info=exc,
                )

    ctx_ref["ctx"] = ctx
    runtime_selfchecker = RuntimeSelfChecker(ctx)
    ctx.selfchecker = runtime_selfchecker
    try:
        health_state.selfchecker = runtime_selfchecker
    except Exception:  # pragma: no cover - defensive assignment
        LOGGER.debug("health_state lacks selfchecker attribute", exc_info=True)
    ctx.shadow_mode_enabled = paper_state["enabled"]

    global _LATEST_CTX
    _LATEST_CTX = ctx
    # NOTE: Full TelegramBot initialization happens later in initialize_components()
    # via the telegram_bot_instance block. The old _setup_telegram() helper was
    # removed because it created a bot that was immediately reset to None, causing
    # a duplicate-init conflict with the proper telegram_bot_instance path below.

    if _HTTP_APP is not None:
        try:
            _HTTP_APP.state.bot_context = ctx
        except AttributeError:  # pragma: no cover - FastAPI state guard
            pass

    order_manager.set_session_guard_getter(session_guard.snapshot)
    order_manager.set_trade_mode_getters(
        enable_live=lambda: bool(
            settings.enable_live and safe_order_manager.settings.enable_live
        ),
        shadow_mode=lambda: bool(ctx.shadow_mode_enabled),
    )

    if websocket_manager is not None:
        _bind_ws_mdm(ctx)

    def _risk_snapshot_to_dict(snapshot: RiskSnapshot) -> dict[str, object]:
        return {
            "daily_realized": snapshot.daily_realized,
            "daily_loss_limit": snapshot.daily_loss_limit,
            "day_loss": snapshot.day_loss,
            "max_day_loss": snapshot.max_day_loss,
            "losses_in_row": snapshot.losses_in_row,
            "cooldown_remaining": snapshot.cooldown_remaining,
            "breaker_tripped": snapshot.breaker_tripped,
            "breaker_reason": snapshot.breaker_reason,
            "shadow_forced": snapshot.shadow_forced,
            "per_trade_risk_pct": snapshot.per_trade_risk_pct,
            "last_rejection": snapshot.last_rejection,
            "timestamp": snapshot.timestamp.isoformat(),
        }

    def _flatten_positions(reason: str) -> list[str]:
        flattened: list[str] = []
        for position in position_manager.get_open_positions():
            qty = getattr(position, "quantity", 0)
            if qty <= 0:
                continue
            exit_side: Literal["BUY", "SELL"] = (
                "SELL" if position.side == "LONG" else "BUY"
            )
            price = market_data_manager.get_latest_price(position.symbol)
            if price is None:
                price = getattr(position, "current_price", None) or getattr(
                    position, "entry_price", None
                )
            try:
                order_manager.place_order(
                    symbol=position.symbol,
                    side=exit_side,
                    quantity=qty,
                    order_type=OrderType.MARKET,
                    price=None,
                )
                flattened.append(position.symbol)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "breaker_flatten_failed",
                    extra={
                        "event": "breaker_flatten_failed",
                        "symbol": position.symbol,
                        "err": str(exc),
                    },
                )
        return flattened

    def _handle_risk_breaker(reason: str, snapshot: RiskSnapshot) -> None:
        ctx.shadow_mode_enabled = True
        safe_order_manager.set_live_enabled(False)
        risk_manager.force_shadow(True)
        cancelled: list[str] = []
        try:
            cancelled = order_manager.cancel_pending_orders()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "breaker_cancel_failed",
                extra={"event": "breaker_cancel_failed", "err": str(exc)},
            )
        flattened = _flatten_positions(reason)
        LOGGER.error(
            "risk_breaker_trip",
            extra={
                "event": "risk_breaker_trip",
                "reason": reason,
                "cancelled_orders": cancelled,
                "flattened": flattened,
            },
        )
        session_guard = ctx.session_guard
        if session_guard is not None:
            session_guard.evaluate()
        else:
            LOGGER.debug(
                "Session guard unavailable during breaker handling",
                extra={"event": "risk_breaker_no_session_guard"},
            )
        _notify(
            "RISK_BREAKER",
            {
                "reason": reason,
                "cancelled_orders": cancelled,
                "flattened": flattened,
                "snapshot": _risk_snapshot_to_dict(snapshot),
            },
        )
        _emit_health_snapshot("session_breaker", reason)

    risk_manager.alert_callback = _handle_risk_breaker

    async def _breaker_alert_sender(reason: str) -> None:
        """Dispatch Telegram alert when the risk breaker trips.

        Args:
            reason: Human-readable reason describing the breaker trigger.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered breaker alert sender",
            extra={"event": "risk_breaker_alert_sender_enter", "reason": reason},
        )
        notifier_ref = ctx.telegram_notifier
        if notifier_ref is None:
            LOGGER.info(
                "Breaker alert skipped: notifier unavailable",
                extra={
                    "event": "risk_breaker_alert_sender_missing",
                    "reason": reason,
                },
            )
            return
        payload: dict[str, object] = {
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            snapshot = _require_component(ctx.risk_manager, "risk_manager").snapshot()
            payload["snapshot"] = _risk_snapshot_to_dict(snapshot)
        except Exception as exc:  # noqa: BLE001 - defensive snapshot capture
            LOGGER.error(
                "Failure in breaker alert snapshot capture: %s",
                exc,
                extra={"event": "risk_breaker_alert_snapshot_error"},
                exc_info=exc,
            )
        try:
            await notifier_ref.send_event("RISK_BREAKER_TRIPPED", payload)
        except Exception as exc:  # noqa: BLE001 - defensive notifier surface
            LOGGER.error(
                "Failure in breaker alert send: %s",
                exc,
                extra={
                    "event": "risk_breaker_alert_send_error",
                    "reason": reason,
                },
                exc_info=exc,
            )
        else:
            LOGGER.info(
                "Condition met: breaker alert dispatched",
                extra={
                    "event": "risk_breaker_alert_sent",
                    "reason": reason,
                },
            )

    _require_component(ctx.risk_manager, "risk_manager").breaker_alert_sender = (
        _breaker_alert_sender
    )

    def _handle_order_rejection(symbol: str, reason: str) -> None:
        """Process order rejections and emit health snapshots when needed.

        Args:
            symbol: Instrument identifier associated with the rejection.
            reason: Textual reason describing the rejection.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered _handle_order_rejection",
            extra={
                "event": "order_rejection_handle",
                "symbol": symbol,
                "reason": reason,
            },
        )
        try:
            risk_manager.record_rejection(reason)
        except Exception as exc:  # noqa: BLE001 - defensive
            LOGGER.error(
                "Failure in _handle_order_rejection record: %s",
                exc,
                extra={"event": "order_rejection_record_error"},
            )
        lower_reason = (reason or "").lower()
        if "storm" in lower_reason:
            _emit_health_snapshot("skip_storm", lower_reason)
        if "brownout" in lower_reason:
            _emit_health_snapshot("api_brownout", lower_reason)

    safe_order_manager.on_order_rejected = _handle_order_rejection

    def set_shadow(on: bool) -> bool:
        desired_shadow = bool(on)
        if desired_shadow:
            ctx.out_of_hours_override = False
            _apply_paper_mode(True)
            LOGGER.info(
                "Shadow mode enabled", extra={"event": "shadow_mode", "enabled": True}
            )
            return True

        if not settings.enable_live:
            LOGGER.warning(
                "Live trading toggle rejected; ENABLE_LIVE is false",
                extra={"event": "live_toggle_blocked"},
            )
            ctx.out_of_hours_override = False
            _apply_paper_mode(True)
            return False

        if not live_possible:
            LOGGER.warning(
                "Live trading unavailable; broker backend disabled",
                extra={"event": "live_toggle_blocked"},
            )
            ctx.out_of_hours_override = False
            _apply_paper_mode(True)
            return False

        allowed, status = session_guard.allow_live()
        override_allowed = False
        session_reason = "OK"
        snapshot: RiskSnapshot | None = None
        if (
            not allowed
            and status.override_out_of_hours
            and status.session_valid
            and status.rate_limits_ok
            and status.risk_green
            and not status.market_open
        ):
            override_allowed = True
            LOGGER.warning(
                "Trading session guard override active outside market hours",
                extra={"event": "session_override", **status.as_dict()},
            )

        soft_override = False
        if not allowed and not override_allowed:
            try:
                snapshot = risk_manager.snapshot()
            except Exception:  # pragma: no cover - defensive
                snapshot = None
            session_reason, soft_override = _resolve_session_reason(status, snapshot)
            if soft_override:
                LOGGER.warning(
                    "Trading session guard soft override engaged",
                    extra={
                        "event": "session_soft_override",
                        **status.as_dict(),
                        "risk_reason": session_reason,
                    },
                )
            if not soft_override:
                ctx.out_of_hours_override = False
                _apply_paper_mode(True)
                LOGGER.warning(
                    "Trading session guard denied live toggle",
                    extra={
                        "event": "session_guard_blocked",
                        **status.as_dict(),
                        "risk_reason": session_reason,
                    },
                )
                block_payload = {**status.as_dict(), "session_reason": session_reason}
                _notify("LIVE_TOGGLE_BLOCKED", block_payload)
                return False
        elif not allowed and override_allowed:
            try:
                snapshot = risk_manager.snapshot()
            except Exception:  # pragma: no cover - defensive
                snapshot = None
            session_reason, _ = _resolve_session_reason(status, snapshot)
        else:
            session_reason = "OK"

        ctx.out_of_hours_override = override_allowed
        override_kind: str | None = None
        if override_allowed:
            override_kind = "out_of_hours"
        elif soft_override:
            override_kind = "soft"
        risk_manager.reset_on_start(override_kind is not None)
        payload = {**status.as_dict(), "session_reason": session_reason}
        if override_kind is not None:
            payload["session_override"] = override_kind
        if override_kind == "out_of_hours":
            payload = {
                **payload,
                "override_used": True,
                "override_kind": "out_of_hours",
            }
            LOGGER.info(
                "Live trading enabled via out-of-hours override",
                extra={
                    "event": "shadow_mode",
                    "enabled": False,
                    "override": True,
                    "override_kind": "out_of_hours",
                    "session_override": "out_of_hours",
                },
            )
        elif override_kind == "soft":
            payload = {
                **payload,
                "override_used": True,
                "override_kind": "soft",
            }
            LOGGER.info(
                "Live trading enabled via soft risk override",
                extra={
                    "event": "shadow_mode",
                    "enabled": False,
                    "override": "soft",
                    "override_kind": "soft",
                    "session_override": "soft",
                },
            )
        else:
            LOGGER.info(
                "Live trading enabled", extra={"event": "shadow_mode", "enabled": False}
            )
        _notify("LIVE_MODE_ENABLED", payload)
        _apply_paper_mode(False)
        return True

    def get_shadow() -> bool:
        return bool(ctx.shadow_mode_enabled)

    if shadow_trader is not None:

        def _force_shadow(reason: str) -> None:
            """Disable live routing and enable paper mode on drift breaches.

            Args:
                reason: Reason text supplied by the drift monitor.

            Returns:
                None.

            Raises:
                None.
            """

            LOGGER.info(
                "Condition met: shadow_forced_by_trader",
                extra={"event": "shadow_forced_by_trader", "reason": reason},
            )
            safe_order_manager.set_live_enabled(False)
            set_shadow(True)

        shadow_trader.disable_live_callback = _force_shadow

    telegram_cfg = getattr(config, "telegram", None)
    ctx.telegram_bot = None
    telegram_transport_mode = _telegram_transport_mode(settings)
    controller = _HTTP_CONTROLLER if _telegram_requires_http_controller(settings) else None
    telegram_bot_instance: TelegramBot | None = None
    telegram_chat_id: int | None = None

    if not settings.notifications.legacy_console_enabled:
        LOGGER.info(
            "legacy_telegram_console_disabled",
            extra={"event": "legacy_console_disabled"},
        )

    try:
        if (
            telegram_cfg
            and getattr(telegram_cfg, "bot_token", None)
            and getattr(telegram_cfg, "chat_id", None) is not None
        ):
            from nifty_scalper_bot.notifications.telegram_controller import (
                TelegramBot,
                TelegramDeps,
            )

            telegram_plain_flag = get_bool("TELEGRAM__PLAIN_TEXT", True)
            # Default polling to True: Without it, _start_polling_if_needed() exits
            # immediately, leaving the Application initialized but never receiving
            # updates. Set TELEGRAM__POLLING_ENABLED=false to explicitly disable.
            polling_enabled_flag = coalesce_bool(
                "TELEGRAM__POLLING_ENABLED",
                default=True,
            )

            paper_mode_getters: dict[str, Callable[[], bool]] = {
                "orders": _paper_mode_enabled,
            }
            paper_mode_setters: dict[str, Callable[[bool], bool]] = {
                "orders": set_shadow,
            }

            runner_ref = ctx.strategy_runner
            if runner_ref is not None:

                def _strategy_paper_getter() -> bool:
                    """Return whether strategy runner is in paper-only mode."""

                    runner = _require_component(runner_ref, "strategy_runner")
                    with suppress(Exception):
                        status = runner.get_status()
                        if isinstance(status, Mapping):
                            if "trading_paused" in status:
                                return bool(status["trading_paused"])
                    paused_flag = getattr(runner, "paused", None)
                    if paused_flag is not None:
                        return bool(paused_flag)
                    return bool(getattr(runner, "_trading_paused", False))

                def _strategy_paper_setter(enabled: bool) -> bool:
                    """Toggle paper mode on the strategy runner."""

                    runner = _require_component(runner_ref, "strategy_runner")
                    try:
                        if enabled:
                            runner.pause_trading()
                        else:
                            runner.resume_trading()
                    except Exception as exc:  # noqa: BLE001 - defensive
                        LOGGER.warning(
                            "telegram_strategy_paper_toggle_failed",
                            extra={
                                "event": "paper_toggle_failed",
                                "section": "strategy",
                                "err": str(exc),
                            },
                        )
                        return False
                    return True

                paper_mode_getters["strategy"] = _strategy_paper_getter
                paper_mode_setters["strategy"] = _strategy_paper_setter

            supervisor_ref = ctx.stream_supervisor
            if supervisor_ref is not None:

                def _stream_paper_getter(
                    supervisor: StreamSupervisor | None = supervisor_ref,
                ) -> bool:
                    if supervisor is None:
                        return True
                    with suppress(Exception):
                        return not bool(supervisor.is_running())
                    return True

                def _stream_paper_setter(
                    enabled: bool,
                    supervisor: StreamSupervisor | None = supervisor_ref,
                ) -> bool:
                    if supervisor is None:
                        return False
                    try:
                        if enabled:
                            supervisor.stop()
                            return True
                        return bool(supervisor.start())
                    except Exception as exc:  # noqa: BLE001 - defensive
                        LOGGER.warning(
                            "telegram_stream_paper_toggle_failed",
                            extra={
                                "event": "paper_toggle_failed",
                                "section": "stream",
                                "err": str(exc),
                            },
                        )
                        return False

                paper_mode_getters["stream"] = _stream_paper_getter
                paper_mode_setters["stream"] = _stream_paper_setter
            cache_settings = getattr(settings, "cache", None)

            instrument_db_path: str | None = None
            instrument_csv_path: str | None = None

            if cache_settings:
                db_path = getattr(cache_settings, "db_path", None)
                csv_path = getattr(cache_settings, "csv_path", None)

                if db_path:
                    instrument_db_path = str(db_path)

                if csv_path:
                    instrument_csv_path = str(csv_path)

            deps = TelegramDeps(
                token=str(telegram_cfg.bot_token),
                chat_id=int(telegram_cfg.chat_id),
                app_version=str(getattr(config, "version", "dev")),
                webhook_url=(
                    str(telegram_cfg.webhook_url)
                    if getattr(telegram_cfg, "webhook_url", None)
                    else None
                ),
                webhook_path=str(telegram_cfg.webhook_path),
                webhook_secret_token=telegram_cfg.webhook_secret_token,
                webhook_max_failures=int(telegram_cfg.webhook_max_failures),
                enable_polling_fallback=polling_enabled_flag,
                polling_interval_seconds=float(telegram_cfg.polling_interval_seconds),
                webhook_listen_host=str(telegram_cfg.webhook_listen_host),
                webhook_listen_port=int(telegram_cfg.webhook_listen_port),
                broker_client=ctx.broker_client,
                websocket_manager=ctx.websocket_manager,
                streamer=ctx.streamer,
                stream_supervisor=ctx.stream_supervisor,
                websocket_enabled=websocket_enabled,
                market_data_manager=ctx.market_data_manager,
                market_regime=ctx.market_regime,
                regime_manager=ctx.market_regime_manager,
                strategy_manager=ctx.strategy_manager,
                strategy_runner=ctx.strategy_runner,
                position_manager=ctx.position_manager,
                order_manager=ctx.order_manager,
                safe_order_manager=ctx.safe_order_manager,
                risk_manager=ctx.risk_manager,
                instrument_resolver=ctx.instrument_manager,
                resolver=ctx.instrument_manager,
                instrument_universe=ctx.instrument_universe,
                metrics=None,
                session_guard=ctx.session_guard,
                rate_limiter=ctx.rate_limiter,
                get_ws_token=_resolve_ws_token,
                get_ws_token_issued_at=_ws_token_issued_at,
                ws_host=ws_host,
                set_shadow_mode=set_shadow,
                get_shadow_mode=get_shadow,
                paper_mode_getters=paper_mode_getters or None,
                paper_mode_setters=paper_mode_setters or None,
                data_hub=ctx.data_hub,
                unified_manager=unified_manager,
                reload_hook=None,
                telegram_plain=telegram_plain_flag,
                selfchecker=ctx.selfchecker,
                bot_context=ctx,
            )
            telegram_bot_instance = TelegramBot(deps)
            telegram_chat_id = int(telegram_cfg.chat_id)
            LOGGER.info("Telegram configured for chat_id=%s", telegram_chat_id)

            if settings.notifications.enabled and telegram_transport_mode == "webhook":
                if controller is None:
                    LOGGER.warning(
                        "telegram_application_controller_missing",
                        extra={
                            "event": "telegram_application_controller_missing",
                            "mode": "webhook",
                        },
                    )
                else:
                    try:
                        application = telegram_bot_instance.build_application(
                            bot=controller.bot
                        )
                    except Exception as exc:  # noqa: BLE001 - defensive wiring
                        LOGGER.exception(
                            "telegram_application_build_failed",
                            extra={
                                "event": "telegram_application_build_failed",
                                "err": str(exc),
                            },
                        )
                    else:
                        ctx.telegram_application = application
                        controller.attach_application(application)
                        LOGGER.info(
                            "telegram_application_attached",
                            extra={"event": "telegram_application_attached"},
                        )
                        version_info = {
                            "build": str(getattr(config, "version", "unknown")),
                            "sha": str(getattr(settings, "git_sha", "unknown")),
                        }
                        services_bundle = TelegramCommandServices(
                            order_manager=ctx.order_manager,
                            risk_manager=ctx.risk_manager,
                            market_data_manager=ctx.market_data_manager,
                            data_hub=ctx.data_hub,
                            bracket_manager=ctx.bracket_manager,
                            strategy_runner=ctx.strategy_runner,
                            config=config,
                            broker=ctx.broker_client,
                            journal=None,
                            metrics=None,
                            market_regime=ctx.market_regime,
                            state_tracker=ctx.state_tracker,
                            version_info=version_info,
                            allowed_chat_id=telegram_chat_id,
                        )
                        try:
                            register_telegram_commands(
                                telegram_bot_instance, application, services_bundle
                            )
                        except Exception as exc:  # noqa: BLE001 - defensive wiring
                            LOGGER.warning(
                                "telegram_command_registration_failed",
                                extra={
                                    "event": "telegram_command_registration_failed",
                                    "err": str(exc),
                                },
                            )
                        hook = getattr(
                            telegram_bot_instance, "after_application_built", None
                        )
                        if callable(hook):
                            result = hook()
                            if inspect.isawaitable(result):
                                awaitable = cast(Coroutine[Any, Any, object], result)
                                try:
                                    loop = asyncio.get_running_loop()
                                except RuntimeError:
                                    asyncio.run(awaitable)
                                else:
                                    loop.create_task(awaitable)
            elif settings.notifications.enabled and telegram_transport_mode == "polling":
                LOGGER.info(
                    "telegram_application_attach_skipped",
                    extra={"event": "telegram_application_attach_skipped", "mode": "polling"},
                )
        else:
            LOGGER.info("Telegram disabled (no token/chat_id provided).")
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Telegram console disabled: %s", exc)

    if telegram_bot_instance is not None and telegram_chat_id is not None:
        ctx.telegram_bot = telegram_bot_instance
        LOGGER.info(
            "✅ TelegramBot ready for chat_id=%s (polling will start on .start())",
            telegram_chat_id,
        )

    return ctx


def _validate_config(config: AppConfig) -> None:
    if not config.broker.api_key or not config.broker.api_secret:
        raise ValueError("Broker credentials are required")
    if not config.broker.access_token:
        raise ValueError("Broker access token is required")
    if config.ratelimit.orders.capacity <= 0:
        raise ValueError("Order rate limit capacity must be positive")
    LOGGER.debug("Configuration validated successfully")


def force_enable_trading_override() -> str:
    """
    Emergency override to force enable trading by resetting all guards.
    Usage: Call from Telegram or REPL.
    """
    ctx = get_latest_bot_context()
    if not ctx:
        return "❌ No Bot Context found."

    logs = []

    # 1. Force Session Valid
    if ctx.session_guard:
        ctx.session_guard.mark_session_valid()
        ctx.session_guard.set_allow_out_of_hours(True)
        ctx.out_of_hours_override = True
        logs.append("✅ Session Guard Force-Validated (Out-of-hours allowed)")

    # 2. Reset Risk Breaker
    if ctx.risk_manager:
        ctx.risk_manager.reset_on_start(override=True)
        # Manually clear flags if needed
        if hasattr(ctx.risk_manager, "_breaker_tripped"):
            ctx.risk_manager._breaker_tripped = False
        logs.append("✅ Risk Manager Reset")

    # 3. Enable Live Orders
    if ctx.safe_order_manager:
        ctx.safe_order_manager.set_live_enabled(True)
        ctx.shadow_mode_enabled = False
        logs.append("✅ Live Trading Enabled (Shadow Mode OFF)")

    LOGGER.critical(f"🚨 MANUAL OVERRIDE ACTIVATED: {', '.join(logs)}")
    return "\n".join(logs)


_SYNTHETIC_FALLBACK_SPOT = 25600.0


def _is_live_execution_mode(configured_mode: str | None = None) -> bool:
    """Return True when the bot is configured to place live broker orders.

    The check honours both the explicit ``EXECUTION_MODE`` value and the
    legacy ``ENABLE_LIVE`` flag.  Synthetic fallbacks (such as the 25600.0
    NIFTY proxy used during off-hours simulation) MUST be gated on the
    inverse of this so live order arming can never select strikes from a
    fake spot price.

    Args:
        configured_mode: Optional pre-resolved ``EXECUTION_MODE`` value.

    Returns:
        True if execution mode is LIVE *or* ENABLE_LIVE is truthy.
    """

    mode = str(
        configured_mode
        if configured_mode is not None
        else os.getenv("EXECUTION_MODE", "")
    ).strip().upper()
    if mode == "LIVE":
        return True
    flag = str(os.getenv("ENABLE_LIVE", "false")).strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _allow_synthetic_market_data() -> bool:
    """Return True when explicit synthetic fallback is permitted.

    Used so unit/integration tests can opt-in to the deterministic 25600.0
    NIFTY fallback even while ``EXECUTION_MODE`` claims LIVE.
    """

    flag = str(os.getenv("ALLOW_SYNTHETIC_MARKET_DATA", "false")).strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _enforce_live_single_replica_safety(*, is_live_execution: bool) -> None:
    """Warn/fail-fast when multiple live replicas/workers are configured."""
    replica_count_raw = str(os.getenv("RAILWAY_REPLICA_COUNT", "")).strip()
    worker_raw = (
        str(os.getenv("WEB_CONCURRENCY", "")).strip()
        or str(os.getenv("GUNICORN_WORKERS", "")).strip()
    )
    replica_count = int(replica_count_raw) if replica_count_raw.isdigit() else 0
    workers = int(worker_raw) if worker_raw.isdigit() else 0
    multi_runner = max(replica_count, workers) > 1
    if not is_live_execution or not multi_runner:
        return
    LOGGER.warning(
        "LIVE_EXECUTION_MULTI_REPLICA_RISK replicas=%s workers=%s reason=single_live_runner_required",
        replica_count_raw or "unknown",
        worker_raw or "unknown",
        extra={
            "event": "LIVE_EXECUTION_MULTI_REPLICA_RISK",
            "replicas": replica_count_raw or "unknown",
            "workers": worker_raw or "unknown",
            "reason": "single_live_runner_required",
        },
    )
    fail_fast = str(os.getenv("BOT_FAIL_FAST_ON_MULTI_REPLICA_LIVE", "false")).strip().lower() in {"1", "true", "yes", "on"}
    if fail_fast:
        raise RuntimeError("Multiple live trading replicas are unsafe")


def _compute_live_execution_enabled() -> bool:
    """Compute whether this instance should be treated as live-execution capable."""
    truthy = {"1", "true", "yes", "on", "live"}
    effective_mode = str(os.getenv("EXECUTION_MODE", "PAPER")).strip().upper()
    return (
        _is_live_execution_mode(effective_mode)
        or str(os.getenv("ENABLE_LIVE_TRADING", "false")).strip().lower() in truthy
        or str(os.getenv("ENABLE_LIVE", "false")).strip().lower() in truthy
    )


def _coerce_spot_price(tick: Mapping[str, Any] | None) -> float | None:
    """Pull a positive spot price out of an MDM tick payload."""

    if not isinstance(tick, Mapping):
        return None
    for key in ("last_price", "ltp", "price", "close"):
        raw = tick.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _resolve_startup_rest_spot_ltp(ctx: BotContext, *, max_age_seconds: float | None = None) -> float | None:
    """Resolve bounded startup spot from cached/REST fallback without treating it as live proof."""

    policy = MarketDataPolicy.from_env()
    mdm = getattr(ctx, "market_data_manager", None)
    if mdm is None:
        return None
    cached_fn = getattr(mdm, "get_cached_ltp", None)
    if callable(cached_fn):
        try:
            cached = float(
                cached_fn(
                    policy.nifty_internal_symbol,
                    max_age_seconds=max_age_seconds,
                    require_ws=False,
                )
                or 0.0
            )
        except (TypeError, ValueError) as exc:
            LOGGER.warning(
                "STARTUP_SPOT_REST_FALLBACK_FAILED stage=get_cached_ltp reason=%s",
                type(exc).__name__,
                extra={"event": "STARTUP_SPOT_REST_FALLBACK_FAILED", "stage": "get_cached_ltp", "reason": type(exc).__name__},
            )
            cached = 0.0
        if cached > 0:
            return cached
    get_ltp_fn = getattr(mdm, "get_ltp", None) or getattr(mdm, "get_latest_price", None)
    if callable(get_ltp_fn):
        rest_ltp = 0.0
        try:
            rest_value = get_ltp_fn(policy.nifty_internal_symbol, allow_rest_fallback=True)
        except TypeError:
            try:
                rest_value = get_ltp_fn(policy.nifty_internal_symbol)
            except (TypeError, RuntimeError, ValueError) as exc:
                LOGGER.warning(
                    "STARTUP_SPOT_REST_FALLBACK_FAILED stage=get_ltp reason=%s",
                    type(exc).__name__,
                    extra={"event": "STARTUP_SPOT_REST_FALLBACK_FAILED", "stage": "get_ltp", "reason": type(exc).__name__},
                )
                rest_value = 0.0
        except (RuntimeError, ValueError) as exc:
            LOGGER.warning(
                "STARTUP_SPOT_REST_FALLBACK_FAILED stage=get_ltp reason=%s",
                type(exc).__name__,
                extra={"event": "STARTUP_SPOT_REST_FALLBACK_FAILED", "stage": "get_ltp", "reason": type(exc).__name__},
            )
            rest_value = 0.0
        try:
            rest_ltp = float(rest_value or 0.0)
        except (TypeError, ValueError) as exc:
            LOGGER.warning(
                "STARTUP_SPOT_REST_FALLBACK_FAILED stage=get_ltp_coerce reason=%s",
                type(exc).__name__,
                extra={"event": "STARTUP_SPOT_REST_FALLBACK_FAILED", "stage": "get_ltp_coerce", "reason": type(exc).__name__},
            )
        if rest_ltp > 0:
            return rest_ltp
    refresh_fn = getattr(mdm, "refresh_quote_now", None)
    if callable(refresh_fn):
        try:
            quote = refresh_fn(policy.nifty_internal_symbol, trace_id="startup_spot_rest_fallback")
        except TypeError:
            try:
                quote = refresh_fn(policy.nifty_internal_symbol)
            except (TypeError, RuntimeError, ValueError) as exc:
                LOGGER.warning(
                    "STARTUP_SPOT_REST_FALLBACK_FAILED stage=refresh_quote_now reason=%s",
                    type(exc).__name__,
                    extra={"event": "STARTUP_SPOT_REST_FALLBACK_FAILED", "stage": "refresh_quote_now", "reason": type(exc).__name__},
                )
                quote = None
        except (RuntimeError, ValueError) as exc:
            LOGGER.warning(
                "STARTUP_SPOT_REST_FALLBACK_FAILED stage=refresh_quote_now reason=%s",
                type(exc).__name__,
                extra={"event": "STARTUP_SPOT_REST_FALLBACK_FAILED", "stage": "refresh_quote_now", "reason": type(exc).__name__},
            )
            quote = None
        price = _coerce_spot_price(quote if isinstance(quote, Mapping) else None)
        if price and price > 0:
            return float(price)
    return None


async def _wait_for_live_spot_or_raise(
    ctx: BotContext,
    *,
    timeout: float = 15.0,
    configured_mode: str | None = None,
) -> float:
    """Resolve the NIFTY spot price to use for live option universe selection.

    In LIVE mode this first waits for fresh WebSocket proof for ``NSE:NIFTY``.
    If the bounded wait expires, a positive cached/REST spot may build and
    hydrate the basket, but live orders remain disarmed until live option
    quote/depth readiness passes.  PAPER/SHADOW may still use the synthetic
    25600.0 reference (with a clear log) so off-hours simulations keep working.

    Args:
        ctx: Bot context with an attached MarketDataManager.
        timeout: Maximum seconds to wait for a fresh WebSocket tick.
        configured_mode: Optional pre-resolved ``EXECUTION_MODE`` value.

    Returns:
        Positive NIFTY spot price.

    Raises:
        RuntimeError: When LIVE mode is active and neither WebSocket nor
            cached/REST spot LTP is available within ``timeout``.
    """

    policy = MarketDataPolicy.from_env()
    mdm = ctx.market_data_manager
    if mdm is None:
        raise RuntimeError("MarketDataManager unavailable")

    live_mode = _is_live_execution_mode(configured_mode)
    wait_fn = getattr(mdm, "wait_for_fresh_spot_tick", None)
    tick: Mapping[str, Any] | None = None
    if callable(wait_fn):
        try:
            tick = await wait_fn(
                policy.nifty_internal_symbol,
                timeout=float(timeout),
                max_age_seconds=policy.startup_spot_max_age_seconds,
                require_ws=policy.require_ws_spot_for_live and live_mode,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "wait_for_fresh_spot_tick failed: %s",
                exc,
                extra={"event": "wait_for_fresh_spot_tick_failed"},
            )
            tick = None

    price = _coerce_spot_price(tick)
    if price is not None:
        LOGGER.info(
            "LIVE_SPOT_READY symbol=%s price=%.2f source=ws",
            policy.nifty_internal_symbol,
            price,
            extra={
                "event": "LIVE_SPOT_READY",
                "symbol": policy.nifty_internal_symbol,
                "price": price,
                "source": "ws",
            },
        )
        return float(price)

    # Bounded fallback: REST/cached spot may select and hydrate the option basket,
    # but it is never live-order proof. Readiness still requires live option quote
    # quality before live_orders_armed can become true.
    fallback = _resolve_startup_rest_spot_ltp(
        ctx,
        max_age_seconds=policy.startup_spot_max_age_seconds,
    )
    if fallback and fallback > 0:
        if live_mode:
            ctx.live_orders_armed = False
            ctx.trading_ready = False
            ctx.live_block_reason = "live_option_quote_required_after_rest_spot_fallback"
            LOGGER.info(
                "STARTUP_SPOT_REST_FALLBACK_USED symbol=%s price=%.2f live_orders_armed=%s",
                policy.nifty_internal_symbol,
                float(fallback),
                bool(getattr(ctx, "live_orders_armed", False)),
                extra={
                    "event": "STARTUP_SPOT_REST_FALLBACK_USED",
                    "symbol": policy.nifty_internal_symbol,
                    "price": float(fallback),
                    "live_orders_armed": bool(getattr(ctx, "live_orders_armed", False)),
                },
            )
        else:
            LOGGER.info(
                "STARTUP_SPOT_FALLBACK_PROBE mode=%s source=rest_or_poll symbol=%s",
                (configured_mode or "PAPER"),
                policy.nifty_internal_symbol,
            )
        LOGGER.info(
            "STARTUP_SPOT_REST_FALLBACK_READY symbol=%s price=%.2f live_proof=%s",
            policy.nifty_internal_symbol,
            float(fallback),
            False,
            extra={
                "event": "STARTUP_SPOT_REST_FALLBACK_READY",
                "symbol": policy.nifty_internal_symbol,
                "price": float(fallback),
                "live_proof": False,
            },
        )
        return float(fallback)

    if live_mode:
        ctx.live_orders_armed = False
        ctx.trading_ready = False
        ctx.live_block_reason = "spot_ltp_unavailable"
        LOGGER.warning(
            "STARTUP_SPOT_PROOF_TIMEOUT symbol=%s timeout=%.2f reason=spot_ltp_unavailable",
            policy.nifty_internal_symbol,
            float(timeout),
            extra={"event": "STARTUP_SPOT_PROOF_TIMEOUT", "symbol": policy.nifty_internal_symbol, "timeout": float(timeout), "reason": "spot_ltp_unavailable"},
        )
        raise RuntimeError("spot_ltp_unavailable")

    LOGGER.warning(
        "SYNTHETIC_SPOT_USED mode=%s price=%.2f reason=no_live_tick",
        configured_mode or ("LIVE" if live_mode else "NON_LIVE"),
        _SYNTHETIC_FALLBACK_SPOT,
        extra={
            "event": "SYNTHETIC_SPOT_USED",
            "mode": configured_mode or ("LIVE" if live_mode else "NON_LIVE"),
            "price": _SYNTHETIC_FALLBACK_SPOT,
        },
    )
    return _SYNTHETIC_FALLBACK_SPOT


async def _wait_for_ws_spot_proof(ctx: BotContext, *, timeout: float) -> float | None:
    """Wait for WS-only NIFTY spot proof. Args: ctx/timeout. Returns: spot or none. Raises: none."""

    policy = MarketDataPolicy.from_env()
    mdm = ctx.market_data_manager
    if mdm is None:
        return None
    wait_fn = getattr(mdm, "wait_for_fresh_spot_tick", None)
    if not callable(wait_fn):
        return None
    tick = await wait_fn(
        policy.nifty_internal_symbol,
        timeout=float(timeout),
        max_age_seconds=policy.startup_spot_max_age_seconds,
        require_ws=True,
    )
    price = _coerce_spot_price(tick)
    return float(price) if price and price > 0 else None


def _safe_startup_log(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    *args: Any,
    **extra_fields: Any,
) -> None:
    """Emit startup log safely. Args: logger/level/event/message/args/extra_fields. Returns: none. Raises: none."""
    try:
        logger.log(level, message, *args, extra={"event": event, **extra_fields})
    except TypeError:
        logger.exception(
            "STARTUP_LOG_FORMAT_ERROR event=%s message=%s args_count=%s",
            event,
            message,
            len(args),
            extra={
                "event": "STARTUP_LOG_FORMAT_ERROR",
                "failed_event": event,
                "message_template": message,
                "args_count": len(args),
            },
        )


def _create_named_task(coro: Any, *, name: str) -> asyncio.Task[Any]:
    """Create startup task with exception logging. Args: coro/name. Returns: task. Raises: None."""
    task = asyncio.create_task(coro, name=name)

    def _done(done_task: asyncio.Task[Any]) -> None:
        try:
            exc = done_task.exception()
        except asyncio.CancelledError:
            return
        except Exception as err:  # noqa: BLE001
            LOGGER.warning(
                "TASK_EXCEPTION_CHECK_FAILED task=%s error=%s",
                name,
                err,
                extra={
                    "event": "TASK_EXCEPTION_CHECK_FAILED",
                    "task": name,
                    "error": str(err),
                },
            )
            return
        if exc is not None:
            LOGGER.exception(
                "BACKGROUND_TASK_FAILED task=%s error=%s",
                name,
                exc,
                exc_info=exc,
                extra={"event": "BACKGROUND_TASK_FAILED", "task": name},
            )

    task.add_done_callback(_done)
    return task


async def _refresh_readiness_after_first_tick(ctx: BotContext, reason: str) -> None:
    """Refresh readiness after startup tick proof. Args: ctx/reason. Returns: none. Raises: none."""
    configured_mode = str(
        getattr(getattr(ctx, "settings", None), "execution_mode", None) or "LIVE"
    ).strip().upper()
    mdm = ctx.market_data_manager
    runner = ctx.strategy_runner
    spot_tick: Mapping[str, Any] | None = None
    if mdm is not None and hasattr(mdm, "get_fresh_spot_tick"):
        spot_tick = mdm.get_fresh_spot_tick("NSE:NIFTY", require_ws=False)
    spot_ready = spot_tick is not None
    bus_running = bool(
        getattr(ctx.message_bus, "running", False)
        or getattr(ctx.message_bus, "_running", False)
    )
    runner_started = _runner_is_running(runner)
    if spot_ready and bus_running:
        ctx.data_observation_ready = True
        if runner is not None and not runner_started:
            active_count = len(getattr(runner, "_active_symbols", set()) or [])
            ensure_runner_started = globals().get("_ensure_strategy_runner_started")
            if active_count <= 0:
                spot_ltp = _coerce_spot_price(spot_tick)
                if (
                    configured_mode in {"LIVE", "PAPER", "SHADOW"}
                    and spot_ltp > 0.0
                    and "_build_and_hydrate_live_basket_from_spot" in globals()
                ):
                    try:
                        basket = await _build_and_hydrate_live_basket_from_spot(
                            ctx,
                            spot_ltp=spot_ltp,
                            configured_mode=configured_mode,
                            hydrate=True,
                        )
                        if basket and not getattr(ctx, "selected_ce", None):
                            _commit_active_dynamic_basket(
                                ctx,
                                basket=basket,
                                option_symbols=basket.get("option_symbols") or [],
                                symbols=basket.get("symbols") or [],
                                atm_strike=basket.get("atm_strike"),
                            )
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning(
                            "LIVE_BASKET_BUILD_FAILED reason=%s",
                            exc,
                            extra={
                                "event": "LIVE_BASKET_BUILD_FAILED",
                                "reason": str(exc),
                            },
                        )
                else:
                    log_throttled(
                        LOGGER,
                        "runner_start_deferred_after_spot_no_symbols",
                        "STRATEGY_RUNNER_START_DEFERRED_AFTER_SPOT reason=no_active_symbols_yet active_symbols=%d",
                        active_count,
                        interval_sec=60.0,
                        level=logging.INFO,
                        extra={
                            "event": "STRATEGY_RUNNER_START_DEFERRED_AFTER_SPOT",
                            "reason": "no_active_symbols_yet",
                            "active_symbols": active_count,
                        },
                    )
            active_count = len(getattr(runner, "_active_symbols", set()) or [])
            if active_count <= 0:
                pass
            elif ensure_runner_started is None:
                LOGGER.warning(
                    "STRATEGY_RUNNER_START_HELPER_MISSING reason=%s",
                    reason,
                    extra={
                        "event": "STRATEGY_RUNNER_START_HELPER_MISSING",
                        "reason": reason,
                    },
                )
            else:
                LOGGER.info(
                    "STRATEGY_RUNNER_START_REQUESTED_AFTER_TICK reason=%s",
                    reason,
                    extra={
                        "event": "STRATEGY_RUNNER_START_REQUESTED_AFTER_TICK",
                        "reason": reason,
                    },
                )
                try:
                    await ensure_runner_started(
                        ctx, reason=f"{reason}:symbols_ready_after_spot"
                    )
                except Exception:
                    LOGGER.exception(
                        "STRATEGY_RUNNER_START_AFTER_TICK_FAILED reason=%s",
                        reason,
                        extra={
                            "event": "STRATEGY_RUNNER_START_AFTER_TICK_FAILED",
                            "reason": reason,
                        },
                    )
                runner_started = _runner_is_running(runner)
        LOGGER.info(
            "DATA_PIPELINE_READY_AFTER_TICK spot_ready=%s bus_running=%s runner_started=%s reason=%s",
            spot_ready,
            bus_running,
            runner_started,
            reason,
            extra={"event": "DATA_PIPELINE_READY_AFTER_TICK", "spot_ready": spot_ready, "bus_running": bus_running, "runner_started": runner_started, "reason": reason},
        )
        return
    LOGGER.info(
        "DATA_PIPELINE_STILL_NOT_READY_AFTER_TICK spot_ready=%s bus_running=%s runner_started=%s reason=%s",
        spot_ready,
        bus_running,
        runner_started,
        reason,
        extra={"event": "DATA_PIPELINE_STILL_NOT_READY_AFTER_TICK", "spot_ready": spot_ready, "bus_running": bus_running, "runner_started": runner_started, "reason": reason},
    )



def _runner_is_running(runner: Any) -> bool:
    """Inspect runner state robustly. Args: runner. Returns: running flag. Raises: none."""

    try:
        is_running_attr = getattr(runner, "is_running", None)
        if callable(is_running_attr):
            return bool(is_running_attr())
        if is_running_attr is not None:
            return bool(is_running_attr)
    except Exception:
        pass

    try:
        status = runner.get_status()
        if isinstance(status, Mapping):
            return bool(status.get("running"))
    except Exception:
        pass

    return bool(getattr(runner, "_running", False))


def _subscribe_ticks_force_live_compat(
    data_hub: Any,
    symbol: str,
    callback: Any,
    *,
    token: int | None = None,
) -> None:
    """Subscribe startup-critical ticks. Args: hub/symbol/callback/token. Returns: none. Raises: none."""
    try:
        data_hub.subscribe_ticks(symbol, callback, token=token, force_live=True)
    except TypeError:
        data_hub.subscribe_ticks(symbol, callback, token=token)


def _coerce_ohlc_row(row: Any) -> dict[str, Any] | None:
    """Normalize OHLC row payload. Args: row. Returns: normalized dict or None. Raises: none."""

    if row is None:
        return None
    if isinstance(row, Mapping):
        ts = row.get("date") or row.get("timestamp") or row.get("time")
        o = row.get("open")
        h = row.get("high")
        l = row.get("low")
        c = row.get("close")
        v = row.get("volume", 0)
    elif isinstance(row, (list, tuple)) and len(row) >= 5:
        ts, o, h, l, c = row[0], row[1], row[2], row[3], row[4]
        v = row[5] if len(row) >= 6 else 0
    else:
        return None
    if ts is None or o is None or h is None or l is None or c is None:
        return None
    try:
        if isinstance(ts, str):
            parsed_ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif isinstance(ts, datetime):
            parsed_ts = ts
        else:
            parsed_ts = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        if parsed_ts.tzinfo is None:
            parsed_ts = parsed_ts.replace(tzinfo=timezone.utc)
        return {
            "timestamp": parsed_ts,
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": int(float(v or 0)),
        }
    except Exception:
        return None


def _wire_and_start_message_bus(ctx: BotContext) -> bool:
    """Wire tick subscriptions and start bus idempotently. Args: ctx. Returns: success. Raises: none."""

    bus = getattr(ctx, "message_bus", None)
    data_hub = getattr(ctx, "data_hub", None)
    runner = getattr(ctx, "strategy_runner", None)
    mdm = getattr(ctx, "market_data_manager", None)
    if bus is None:
        LOGGER.warning("MESSAGE_BUS_START_SKIPPED reason=no_bus")
        return False
    if data_hub is not None:
        data_hub.bus = bus
    if mdm is not None:
        mdm.bus = bus
    if not ctx.message_bus_tick_subscribed:
        if data_hub is not None:
            bus.subscribe_once(
                MessageType.TICK,
                data_hub.ingest_tick_from_bus,
                subscriber_id="data_hub.ingest_tick_from_bus",
            )
            LOGGER.info("MESSAGE_BUS_TICK_OWNER owner=data_hub")
        elif runner is not None:
            bus.subscribe_once(
                MessageType.TICK,
                runner.on_data,
                subscriber_id="strategy_runner.on_data.fallback",
            )
            LOGGER.warning("MESSAGE_BUS_TICK_OWNER owner=strategy_runner_fallback reason=no_data_hub")
        else:
            LOGGER.warning("MESSAGE_BUS_TICK_SUBSCRIPTION_SKIPPED reason=no_data_hub_no_runner")

        ctx.message_bus_tick_subscribed = True
    started = bus.start()
    ctx.message_bus_running = bool(started or getattr(bus, "running", False))
    if ctx.message_bus_running:
        LOGGER.info("MESSAGE_BUS_STARTED")
    return ctx.message_bus_running


def _set_startup_phase(ctx: BotContext, phase: str) -> None:
    """Set startup phase marker. Args: ctx, phase. Returns: none. Raises: none."""
    ctx.startup_phase = phase
    LOGGER.info("STARTUP_PHASE phase=%s", phase)


def _mark_startup_failed(ctx: BotContext, phase: str, exc: BaseException) -> None:
    """Record startup failure diagnostics. Args: ctx, phase, exc. Returns: none. Raises: none."""
    ctx.startup_failed = True
    ctx.startup_phase = phase
    ctx.startup_failure_reason = str(exc)
    ctx.startup_failure_exception = exc.__class__.__name__
    LOGGER.exception(
        "STARTUP_PHASE_FAILED phase=%s exception=%s reason=%s",
        phase,
        exc.__class__.__name__,
        str(exc),
    )


async def _replay_latest_mdm_ticks_to_bus(ctx: BotContext, *, reason: str) -> int:
    """Replay latest cached MDM ticks into bus. Args: ctx/reason. Returns: replay count. Raises: none."""

    mdm = getattr(ctx, "market_data_manager", None)
    bus = getattr(ctx, "message_bus", None)
    if mdm is None or bus is None:
        return 0
    if getattr(ctx, "data_observation_ready", False):
        LOGGER.info(
            "MDM_CACHED_TICKS_REPLAY_SKIPPED reason=live_data_already_observed",
            extra={
                "event": "MDM_CACHED_TICKS_REPLAY_SKIPPED",
                "reason": "live_data_already_observed",
            },
        )
        return 0
    latest_ticks = getattr(mdm, "_latest_ticks", {}) or {}
    replayed = 0
    for symbol, tick in list(latest_ticks.items()):
        if not isinstance(tick, Mapping):
            continue
        try:
            msg = Message(type=MessageType.TICK,timestamp=datetime.now(timezone.utc),data={**dict(tick),"symbol":symbol,"source":"mdm_replay","trace_id":f"replay-{symbol}-{time_module.monotonic_ns()}"},source="market_data_manager_replay")
            result = bus.publish(msg)
            if inspect.isawaitable(result):
                await result
            replayed += 1
        except Exception:
            LOGGER.exception("MDM_CACHED_TICK_REPLAY_FAILED symbol=%s", symbol)
    LOGGER.info("MDM_CACHED_TICKS_REPLAYED count=%d reason=%s", replayed, reason, extra={"event":"MDM_CACHED_TICKS_REPLAYED","count":replayed,"reason":reason})
    return replayed

async def _ensure_strategy_runner_started(ctx: BotContext, *, reason: str) -> None:
    """Start strategy runner idempotently. Args: ctx/reason. Returns: none. Raises: none."""

    runner = getattr(ctx, "strategy_runner", None)
    if runner is None:
        LOGGER.warning("STRATEGY_RUNNER_START_SKIPPED reason=no_runner")
        return
    existing_task = getattr(ctx, "runner_task", None) or getattr(
        ctx, "strategy_runner_task", None
    )
    if existing_task is not None and not existing_task.done():
        return
    if _runner_is_running(runner):
        LOGGER.info("STRATEGY_RUNNER_START_SKIPPED reason=already_running")
        return
    LOGGER.info(
        "STRATEGY_RUNNER_START_DIAG reason=%s active_symbols=%d ready=%s runner_state=%s running=%s",
        reason,
        len(getattr(runner, "_active_symbols", set()) or []),
        getattr(runner, "ready", None),
        getattr(runner, "_runner_state", None),
        _runner_is_running(runner),
        extra={"event":"STRATEGY_RUNNER_START_DIAG","reason":reason,"active_symbols":len(getattr(runner, "_active_symbols", set()) or []),"ready":getattr(runner, "ready", None),"runner_state":str(getattr(runner, "_runner_state", None)),"running":_runner_is_running(runner)},
    )
    if hasattr(runner, "start"):
        result = runner.start()
        if inspect.isawaitable(result):
            await result
        if _runner_is_running(runner):
            LOGGER.info("STRATEGY_RUNNER_STARTED reason=%s", reason, extra={"event": "STRATEGY_RUNNER_STARTED", "reason": reason})
            await _recompute_and_push_runtime_readiness(ctx, reason="runner_started")
        else:
            LOGGER.warning("STRATEGY_RUNNER_START_RETURNED_NOT_RUNNING reason=%s active_symbols=%d runner_state=%s", reason, len(getattr(runner, "_active_symbols", set()) or []), getattr(runner, "_runner_state", None), extra={"event":"STRATEGY_RUNNER_START_RETURNED_NOT_RUNNING","reason":reason,"active_symbols":len(getattr(runner, "_active_symbols", set()) or []),"runner_state":str(getattr(runner, "_runner_state", None))})
        return
    if hasattr(runner, "run"):
        task = asyncio.create_task(runner.run())
        ctx.runner_task = task
        LOGGER.info(
            "STRATEGY_RUNNER_TASK_STARTED reason=%s",
            reason,
            extra={"event": "STRATEGY_RUNNER_TASK_STARTED", "reason": reason},
        )
        await _recompute_and_push_runtime_readiness(ctx, reason="runner_started")
        return
    LOGGER.warning("STRATEGY_RUNNER_START_SKIPPED reason=no_start_or_run_method")




async def _live_readiness_rearm_loop(ctx: BotContext) -> None:
    """Re-arm LIVE trading when market opens. Args: ctx. Returns: none. Raises: none."""

    interval_seconds = 30.0
    if not hasattr(ctx, "live_orders_armed"):
        ctx.live_orders_armed = False
    if not hasattr(ctx, "trading_ready"):
        ctx.trading_ready = False
    if not hasattr(ctx, "readiness_mode"):
        ctx.readiness_mode = "DATA_WARMUP"
    if not hasattr(ctx, "effective_mode"):
        ctx.effective_mode = str(ctx.readiness_mode)
    LOGGER.info("LIVE_READINESS_REARM_LOOP_STARTED")
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            configured_mode = str(getattr(ctx.settings, "execution_mode", None) or os.getenv("EXECUTION_MODE", "PAPER")).upper()
            if configured_mode != "LIVE":
                continue
            LOGGER.info("LIVE_READINESS_REARM_CHECK")
            try:
                market_open_now = get_market_state() == MarketState.OPEN
            except Exception:
                market_open_now = False
            if not market_open_now:
                continue
            runner = getattr(ctx, "strategy_runner", None)
            active_symbols = len(getattr(runner, "_active_symbols", set()) or [])
            if active_symbols == 0:
                try:
                    spot_ltp = await _wait_for_live_spot_or_raise(
                        ctx,
                        timeout=10.0,
                        configured_mode=configured_mode,
                    )
                    basket = await _build_and_hydrate_live_basket_from_spot(
                        ctx,
                        spot_ltp=float(spot_ltp),
                        configured_mode=configured_mode,
                        hydrate=False,
                    )
                    if basket and not getattr(ctx, "selected_ce", None):
                        _commit_active_dynamic_basket(
                            ctx,
                            basket=basket,
                            option_symbols=basket.get("option_symbols") or [],
                            symbols=basket.get("symbols") or [],
                            atm_strike=basket.get("atm_strike"),
                        )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("LIVE_REARM_BASKET_BUILD_FAILED error=%s", exc, exc_info=True)
            try:
                await _ensure_strategy_runner_started(ctx, reason="market_open_rearm_loop")
            except Exception as exc:
                LOGGER.exception("LIVE_REARM_RUNNER_START_FAILED error_type=%s error=%s", type(exc).__name__, str(exc))
            try:
                await _recompute_and_push_runtime_readiness(ctx, reason="market_open_rearm_loop")
            except Exception as exc:
                LOGGER.exception("LIVE_REARM_READINESS_PUSH_FAILED error_type=%s error=%s", type(exc).__name__, str(exc))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception(
                "LIVE_READINESS_REARM_LOOP_CRASHED error_type=%s error=%s",
                type(exc).__name__,
                str(exc),
                extra={
                    "event": "LIVE_READINESS_REARM_LOOP_CRASHED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            await asyncio.sleep(interval_seconds)


def _sync_mdm_bars_to_runner(ctx: BotContext, symbol: str, *, min_bars: int) -> int:
    """Sync OHLC bars from MDM/DataHub into runner. Args: ctx/symbol/min_bars. Returns: ingested count. Raises: none."""
    if ctx.market_data_manager is None or ctx.strategy_runner is None:
        return 0
    try:
        bars = list(ctx.market_data_manager.get_ohlc_bars(symbol, limit=min_bars) or [])
    except Exception:
        bars = []
    if not bars and getattr(ctx, "data_hub", None) is not None:
        fn = getattr(ctx.data_hub, "get_ohlc_bars", None)
        if callable(fn):
            try:
                bars = list(fn(symbol, limit=min_bars) or [])
            except TypeError:
                bars = list(fn(symbol) or [])
            except Exception:
                bars = []
    existing = _runner_bar_count(ctx, symbol)
    if existing >= min_bars:
        return 0
    missing = max(0, min_bars - existing)
    ingested = 0
    for bar in bars[-missing:]:
        try:
            payload = dict(bar)
            payload["symbol"] = symbol
            ctx.strategy_runner.ingest_historical_bar(payload)
            ingested += 1
        except Exception:
            continue
    if ingested:
        log_throttled(
            LOGGER,
            logging.INFO,
            "RUNNER_MDM_BAR_SYNC",
            60.0,
            "RUNNER_MDM_BAR_SYNC symbol=%s ingested=%d min_bars=%d",
            symbol,
            ingested,
            min_bars,
            extra={"event": "RUNNER_MDM_BAR_SYNC", "symbol": symbol, "ingested": ingested, "min_bars": min_bars},
        )
    return ingested


def _best_fresh_option(
    ctx: BotContext, symbols: list[str], *, side: str, max_age_s: float = 60.0
) -> str | None:
    """Pick best fresh quote CE/PE option. Args: ctx/symbols/side/max_age_s. Returns: symbol|None. Raises: none."""
    side = side.upper()
    candidates = [s for s in symbols if str(s).upper().endswith(side)]
    best_symbol: str | None = None
    best_score = -1
    for sym in candidates:
        quote_fresh = False
        try:
            snap = ctx.market_data_manager.get_symbol_snapshot(sym) if ctx.market_data_manager is not None else None
            ltp = float(getattr(snap, "ltp", 0.0) or 0.0)
            age = float(getattr(snap, "tick_age_s", 9999.0) or 9999.0)
            quote_fresh = ltp > 0 and age <= max_age_s
        except Exception:
            quote_fresh = False
        score = 3 if quote_fresh else 0
        if score > best_score:
            best_score = score
            best_symbol = sym
    return best_symbol if best_score >= 3 else None


def _best_hydrated_option(
    ctx: BotContext, symbols: list[str], *, side: str, min_bars: int
) -> str | None:
    """Pick option candidate with hydrated bars. Args: ctx/symbols/side/min_bars. Returns: symbol|None. Raises: none."""
    side = side.upper()
    candidates = [s for s in symbols if str(s).upper().endswith(side)]
    best_symbol: str | None = None
    best_count = -1
    for sym in candidates:
        try:
            bars_count = (
                len(list(ctx.market_data_manager.get_ohlc_bars(sym, limit=min_bars) or []))
                if ctx.market_data_manager is not None
                else 0
            )
        except Exception:
            bars_count = 0
        if bars_count < min_bars:
            _sync_mdm_bars_to_runner(ctx, sym, min_bars=min_bars)
            try:
                bars_count = (
                    len(list(ctx.market_data_manager.get_ohlc_bars(sym, limit=min_bars) or []))
                    if ctx.market_data_manager is not None
                    else bars_count
                )
            except Exception:
                pass
        if bars_count >= min_bars and bars_count > best_count:
            best_count = bars_count
            best_symbol = sym
    return best_symbol


def _fresh_option_quote(
    ctx: BotContext, symbol: str | None, *, max_age_s: float = 60.0
) -> str | None:
    """Return symbol when its quote is fresh. Args: ctx/symbol/max_age_s. Returns: symbol|None. Raises: none."""
    if not symbol or ctx.market_data_manager is None:
        return None
    try:
        snap = ctx.market_data_manager.get_symbol_snapshot(symbol)
        ltp = float(getattr(snap, "ltp", 0.0) or 0.0)
        age = float(getattr(snap, "tick_age_s", 9999.0) or 9999.0)
    except Exception:
        return None
    return symbol if ltp > 0 and age <= max_age_s else None


def _count_symbol_bars(ctx: BotContext, symbol: str | None) -> int:
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


async def _ensure_selected_options_hydrated(
    ctx: BotContext, selected_ce: str | None, selected_pe: str | None, required_bars: int, reason: str
) -> dict[str, dict[str, int | bool]]:
    """Ensure selected options have required bars in MDM and runner. Args: ctx/symbols/required_bars/reason. Returns: none. Raises: none."""
    mdm = getattr(ctx, "market_data_manager", None)
    runner = getattr(ctx, "strategy_runner", None)
    hydration_result: dict[str, dict[str, int | bool]] = {}
    if mdm is None:
        return hydration_result
    for sym in (selected_ce, selected_pe):
        if not sym:
            continue
        before_mdm_bars = len(mdm.get_ohlc_bars(sym) or [])
        before_runner_bars = len(runner._indicator_engine.get_history(sym) or []) if runner is not None and hasattr(runner, "_indicator_engine") else 0
        if before_mdm_bars >= required_bars and before_runner_bars >= required_bars:
            hydration_result[sym] = {
                "before_mdm_bars": before_mdm_bars,
                "after_mdm_bars": before_mdm_bars,
                "before_runner_bars": before_runner_bars,
                "after_runner_bars": before_runner_bars,
                "ready": True,
            }
            continue
        hydrate_fn = getattr(mdm, "hydrate_symbol_history", None)
        if before_mdm_bars < required_bars and callable(hydrate_fn):
            await hydrate_fn(sym, interval="minute", days=_history_lookback_days(required_bars), max_bars=required_bars, reason=f"{reason}_selected_option_force_hydration")
        try:
            bars = mdm.get_ohlc_bars(sym, limit=required_bars) or []
        except TypeError:
            bars = mdm.get_ohlc_bars(sym) or []
        normalized_bars: list[dict[str, Any]] = []
        for row in bars:
            if not isinstance(row, Mapping):
                continue
            bar_data = dict(row)
            bar_data["symbol"] = sym

            if "timestamp" not in bar_data:
                if "start" in bar_data:
                    bar_data["timestamp"] = bar_data["start"]
                elif "date" in bar_data:
                    bar_data["timestamp"] = bar_data["date"]
                elif "time" in bar_data:
                    bar_data["timestamp"] = bar_data["time"]

            if {"open", "high", "low", "close"}.issubset(bar_data):
                bar_data.setdefault("timestamp", str(bar_data.get("date") or bar_data.get("start") or bar_data.get("time") or f"{sym}-bar-{len(normalized_bars)}"))
                normalized_bars.append(bar_data)
        used_reseed = False
        if runner is not None and hasattr(runner, "reseed_history_from_bars"):
            try:
                reseeded_count = runner.reseed_history_from_bars(
                    sym,
                    normalized_bars,
                    source=f"{reason}_selected_option_reseed",
                    min_bars=required_bars,
                )
                used_reseed = True
                LOGGER.info(
                    "SELECTED_OPTION_RUNNER_RESEED_DONE symbol=%s reseeded_count=%s required_bars=%d reason=%s",
                    sym,
                    reseeded_count,
                    required_bars,
                    reason,
                )
                if reseeded_count >= required_bars:
                    LOGGER.info(
                        "SELECTED_OPTION_RUNNER_SYNC_FROM_MDM symbol=%s mdm_bars=%d runner_bars=%d required_bars=%d reason=%s",
                        sym,
                        len(normalized_bars),
                        reseeded_count,
                        required_bars,
                        reason,
                        extra={"event": "SELECTED_OPTION_RUNNER_SYNC_FROM_MDM", "symbol": sym, "mdm_bars": len(normalized_bars), "runner_bars": reseeded_count, "required_bars": required_bars, "reason": reason},
                    )
            except Exception:
                LOGGER.exception(
                    "SELECTED_OPTION_RUNNER_RESEED_FAILED symbol=%s required_bars=%d reason=%s",
                    sym,
                    required_bars,
                    reason,
                )
        elif runner is not None and hasattr(runner, "ingest_historical_bar"):
            for bar_data in normalized_bars:
                runner.ingest_historical_bar(bar_data)
        update_fn = getattr(mdm, "update_hydration_status", None)
        if callable(update_fn):
            update_fn(sym, mdm.get_ohlc_bars(sym))
        after_mdm_bars = len(mdm.get_ohlc_bars(sym) or [])
        after_runner_bars = len(runner._indicator_engine.get_history(sym) or []) if runner is not None and hasattr(runner, "_indicator_engine") else 0
        if used_reseed and after_mdm_bars >= required_bars and after_runner_bars < required_bars:
            LOGGER.warning(
                "SELECTED_OPTION_RESEED_FAILED symbol=%s after_mdm_bars=%d after_runner_bars=%d required_bars=%d reason=%s",
                sym,
                after_mdm_bars,
                after_runner_bars,
                required_bars,
                reason,
            )
        ready = bool(after_mdm_bars >= required_bars and after_runner_bars >= required_bars)
        hydration_result[sym] = {
            "before_mdm_bars": before_mdm_bars,
            "after_mdm_bars": after_mdm_bars,
            "before_runner_bars": before_runner_bars,
            "after_runner_bars": after_runner_bars,
            "ready": ready,
        }
        LOGGER.info(
            "SELECTED_OPTION_FORCE_HYDRATION_RESULT symbol=%s before_mdm_bars=%d after_mdm_bars=%d before_runner_bars=%d after_runner_bars=%d required_bars=%d",
            sym, before_mdm_bars, after_mdm_bars, before_runner_bars, after_runner_bars, required_bars,
        )
        if not ready:
            LOGGER.warning(
                "SELECTED_OPTION_HYDRATION_NOT_READY symbol=%s after_mdm_bars=%d after_runner_bars=%d required_bars=%d reason=%s",
                sym,
                after_mdm_bars,
                after_runner_bars,
                required_bars,
                reason,
            )
    return hydration_result


async def _recompute_and_push_runtime_readiness(ctx: BotContext, *, reason: str) -> None:
    """Recompute app runtime readiness and push to runner. Args: ctx/reason. Returns: none. Raises: none."""
    basket = normalize_active_basket_schema(
        cast(dict[str, object], getattr(ctx, "active_trading_universe", {}) or {})
    )
    mdm = getattr(ctx, "market_data_manager", None)
    old_ce = getattr(ctx, "selected_ce", None)
    old_pe = getattr(ctx, "selected_pe", None)
    option_symbols = [str(s) for s in list(basket.get("option_symbols") or basket.get("symbols") or []) if s]
    picked_ce, picked_pe = pick_atm_option_symbols_from_basket(basket)
    selected_ce = cast(
        str | None,
        basket.get("selected_ce")
        or basket.get("atm_ce")
        or picked_ce,
    )
    selected_pe = cast(
        str | None,
        basket.get("selected_pe")
        or basket.get("atm_pe")
        or picked_pe,
    )
    if not selected_ce and old_ce in option_symbols:
        selected_ce = old_ce
    if not selected_pe and old_pe in option_symbols:
        selected_pe = old_pe
    basket.update(
        {
            "selected_ce": selected_ce,
            "selected_pe": selected_pe,
            "atm_ce": selected_ce,
            "atm_pe": selected_pe,
            "option_symbols": option_symbols,
            "symbols": list(
                dict.fromkeys(
                    [
                        basket.get("spot_symbol") or "NSE:NIFTY",
                        basket.get("futures_symbol") or "",
                        *option_symbols,
                    ]
                )
            ),
        }
    )
    ctx.active_trading_universe = basket
    if selected_ce:
        ctx.selected_ce = str(selected_ce)
    else:
        ctx.selected_ce = None
    if selected_pe:
        ctx.selected_pe = str(selected_pe)
    else:
        ctx.selected_pe = None
    ctx.atm_ce_symbol = selected_ce
    ctx.atm_pe_symbol = selected_pe
    def _snapshot(sym:str|None)->Any:
        if not sym or mdm is None: return None
        try: return mdm.get_symbol_snapshot(sym)
        except Exception: return None
    def _tick_age_threshold()->float:
        env=float(os.getenv("READINESS_TICK_MAX_AGE_SECONDS","0") or 0)
        if env>0: return env
        return 60.0
    def _fresh_ltp(sym:str|None,max_age_s:float|None=None)->bool:
        snap=_snapshot(sym)
        if snap is None: return False
        age_limit=max_age_s or _tick_age_threshold()
        ltp=float(getattr(snap,'ltp',0.0) or 0.0)
        age=getattr(snap,'tick_age_s',None)
        if ltp<=0: return False
        if age is not None:
            return float(age)<=age_limit
        last_tick_ts=getattr(mdm,'_last_tick_ts',{}) if mdm is not None else {}
        ts=last_tick_ts.get(sym) if isinstance(last_tick_ts,dict) else None
        dt=pd.to_datetime(ts,utc=True,errors='coerce')
        if pd.isna(dt): return False
        return (pd.Timestamp.utcnow()-dt).total_seconds()<=age_limit
    def _tradable_quote(sym:str|None)->bool:
        if not sym or mdm is None: return False
        h=getattr(mdm,'has_ws_tradable_quote',None)
        if callable(h):
            try: return bool(h([sym]))
            except TypeError:
                return bool(h(sym))
            except Exception:
                pass
        snap=_snapshot(sym)
        if snap is None: return False
        bid=float(getattr(snap,'bid',0.0) or 0.0); ask=float(getattr(snap,'ask',0.0) or 0.0)
        return bool(getattr(snap,'tradable_quote',False)) and bid>0 and ask>bid
    def _live_tick_seen(sym:str|None)->bool:
        if _fresh_ltp(sym): return True
        if not sym or mdm is None: return False
        limit=_tick_age_threshold()
        for attr in ('_last_tick_ts','_last_tick_snapshot'):
            cache=getattr(mdm,attr,{})
            if isinstance(cache,dict) and sym in cache:
                ts=cache.get(sym)
                dt=pd.to_datetime(ts,utc=True,errors='coerce')
                if not pd.isna(dt) and (pd.Timestamp.utcnow()-dt).total_seconds()<=limit:
                    return True
        return False
    def _selected_option_subscription_state(sym: str | None) -> dict[str, Any]:
        state = {"symbol": sym, "token": None, "desired": False, "subscribed": False, "confirmed": False, "fresh_tick": False, "tick_age_s": None}
        if not sym or mdm is None:
            return state
        active_tokens = getattr(ctx, "active_symbol_tokens", None)
        aliases = [str(sym), _bare_trading_symbol(sym)]
        token = None
        for alias in aliases:
            token = _coerce_positive_token((active_tokens or {}).get(alias) if isinstance(active_tokens, Mapping) else None)
            if token is not None:
                break
        if token is None:
            try:
                token = _coerce_positive_token(getattr(mdm, "_resolve_token_for_symbol", lambda _s: None)(sym))
            except Exception:
                token = None
        if token is None:
            token = _coerce_positive_token(getattr(mdm, "_symbol_to_token", {}).get(sym))
        state["token"] = token
        snap = _snapshot(sym)
        state["tick_age_s"] = getattr(snap, "tick_age_s", None) if snap is not None else None
        state["fresh_tick"] = _live_tick_seen(sym)
        if token is not None:
            desired_snapshot = getattr(mdm, "desired_tokens_snapshot", None)
            desired_tokens = set(desired_snapshot() if callable(desired_snapshot) else getattr(mdm, "_desired_tokens", set()) or set())
            subscribed_tokens = set(getattr(mdm, "_subscribed_tokens", set()) or set())
            ws = getattr(mdm, "_ws", None) or getattr(mdm, "websocket", None)
            subscribed_tokens.update(set(getattr(ws, "_tokens", set()) or set()))
            confirmed_tokens = set(getattr(mdm, "_confirmed_subscriptions", set()) or set())
            desired_ints = {coerced for raw in desired_tokens if (coerced := _coerce_positive_token(raw)) is not None}
            subscribed_ints = {coerced for raw in subscribed_tokens if (coerced := _coerce_positive_token(raw)) is not None}
            confirmed_ints = {coerced for raw in confirmed_tokens if (coerced := _coerce_positive_token(raw)) is not None}
            state["desired"] = int(token) in desired_ints
            state["subscribed"] = int(token) in subscribed_ints
            state["confirmed"] = int(token) in confirmed_ints
        LOGGER.info(
            "SELECTED_OPTION_SUBSCRIPTION_STATE symbol=%s token=%s desired=%s subscribed=%s fresh_tick=%s tick_age_s=%s",
            sym, state["token"], state["desired"], state["subscribed"], state["fresh_tick"], state["tick_age_s"],
            extra={"event": "SELECTED_OPTION_SUBSCRIPTION_STATE", **state},
        )
        return state
    def _subscription_confirmed(sym: str | None) -> bool:
        state = _selected_option_subscription_state(sym)
        return bool(state.get("confirmed") or state.get("desired") or state.get("subscribed") or state.get("fresh_tick"))
    def _subscription_or_live_tick(sym:str|None)->bool:
        state = _selected_option_subscription_state(sym)
        if state.get("confirmed") or state.get("desired") or state.get("subscribed"):
            return True
        if state.get("fresh_tick"):
            LOGGER.info("READINESS_SUBSCRIPTION_PROOF_FROM_LIVE_TICK symbol=%s token=%s", sym, state.get("token"))
            return True
        return False
    def _bars(sym: str | None) -> int:
        if not sym or mdm is None: return 0
        try: return len(mdm.get_ohlc_bars(sym) or [])
        except Exception: return 0
    def _readiness_bars(sym: str | None) -> tuple[int, int, int]:
        mdm_bars=_bars(sym); runner_bars=0
        runner=getattr(ctx,'strategy_runner',None)
        if runner is not None and hasattr(runner,'_indicator_engine'):
            try: runner_bars=len(runner._indicator_engine.get_history(sym) or []) if sym else 0
            except Exception: runner_bars=0
        return min(mdm_bars, runner_bars), mdm_bars, runner_bars
    spot_symbol=str(basket.get('spot_symbol') or 'NSE:NIFTY')
    option_eval_min_live_bars = int(os.getenv("READINESS_OPTION_EVAL_MIN_BARS", os.getenv("OPTION_EVAL_MIN_LIVE_BARS", "20")) or 20)
    option_execution_min_bars = int(os.getenv("READINESS_OPTION_EXEC_MIN_BARS", os.getenv("OPTION_EXECUTION_MIN_BARS", "30")) or 30)
    context_execution_min_bars = int(os.getenv("READINESS_CONTEXT_MIN_BARS", os.getenv("CONTEXT_EXECUTION_MIN_BARS", "20")) or 20)
    hydration_report = _hydrate_committed_active_basket(ctx, reason=reason)
    hydration_hard_ready = bool(hydration_report.get("hard_ready"))
    hydration_missing = list(hydration_report.get("missing") or [])
    _ = await _ensure_selected_options_hydrated(
        ctx, selected_ce, selected_pe, option_execution_min_bars, reason
    )
    if not hydration_hard_ready:
        hydration_report = _hydrate_committed_active_basket(ctx, reason=f"{reason}_post_selected_option_hydration")
        hydration_hard_ready = bool(hydration_report.get("hard_ready"))
        hydration_missing = list(hydration_report.get("missing") or hydration_missing)
    ce_bars, ce_mdm_bars, ce_runner_bars = _readiness_bars(selected_ce)
    pe_bars, pe_mdm_bars, pe_runner_bars = _readiness_bars(selected_pe)
    ce_quote_fresh=_fresh_ltp(selected_ce); pe_quote_fresh=_fresh_ltp(selected_pe)
    ce_exec_ready = bool(selected_ce) and ce_quote_fresh and _tradable_quote(selected_ce) and ce_bars >= option_execution_min_bars
    pe_exec_ready = bool(selected_pe) and pe_quote_fresh and _tradable_quote(selected_pe) and pe_bars >= option_execution_min_bars
    ce_eval_ready = bool(selected_ce) and ce_quote_fresh and ce_bars >= option_eval_min_live_bars
    pe_eval_ready = bool(selected_pe) and pe_quote_fresh and pe_bars >= option_eval_min_live_bars
    spot_ready=_fresh_ltp(spot_symbol) or _bars(spot_symbol)>=1
    futures_symbol = str(basket.get("futures_symbol") or "")
    context_exec_ready = _bars(spot_symbol)>=context_execution_min_bars or (_fresh_ltp(spot_symbol) and _bars(futures_symbol)>=context_execution_min_bars)
    data_hard_ready=bool(spot_ready and ce_eval_ready and pe_eval_ready and hydration_hard_ready)
    runner_running=_runner_is_running(getattr(ctx,'strategy_runner',None))
    evaluation_ready=bool(data_hard_ready and runner_running)
    live_mode = str(getattr(ctx.settings, "execution_mode", "PAPER")).upper() == "LIVE"
    market_open = get_market_state() == MarketState.OPEN
    broker_ready = bool(getattr(ctx, "broker_client", None) and getattr(ctx, "order_manager", None))
    execution_ready_by_symbol: dict[str, bool] = {}
    if selected_ce:
        execution_ready_by_symbol[str(selected_ce)] = bool(ce_exec_ready)
    if selected_pe:
        execution_ready_by_symbol[str(selected_pe)] = bool(pe_exec_ready)
    any_selected_option_exec_ready = bool(ce_exec_ready or pe_exec_ready)
    live_orders_armed=bool(live_mode and market_open and evaluation_ready and context_exec_ready and broker_ready and any_selected_option_exec_ready)
    missing=[]
    if not selected_ce: missing.append('selected_ce_missing')
    if not selected_pe: missing.append('selected_pe_missing')
    if not spot_ready: missing.append('spot_not_ready')
    if not evaluation_ready: missing.append('eval_not_ready')
    if not hydration_hard_ready:
        missing.append('ACTIVE_BASKET_HYDRATION_NOT_READY')
        missing.extend(str(item) for item in hydration_missing)
    if ce_runner_bars < option_eval_min_live_bars: missing.append('ce_eval_bars_missing')
    if pe_runner_bars < option_eval_min_live_bars: missing.append('pe_eval_bars_missing')
    if ce_runner_bars < option_execution_min_bars: missing.append('ce_exec_bars_missing')
    if pe_runner_bars < option_execution_min_bars: missing.append('pe_exec_bars_missing')
    if not ce_quote_fresh: missing.append('ce_quote_not_fresh')
    if not pe_quote_fresh: missing.append('pe_quote_not_fresh')
    if not _tradable_quote(selected_ce): missing.append('ce_depth_not_tradable')
    if not _tradable_quote(selected_pe): missing.append('pe_depth_not_tradable')
    if not runner_running: missing.append('runner_not_running')
    if live_mode:
        if not market_open: missing.append('market_closed')
        if not ce_exec_ready: missing.append('ce_exec_quote_or_history_not_ready')
        if not pe_exec_ready: missing.append('pe_exec_quote_or_history_not_ready')
        if not context_exec_ready: missing.append('context_exec_not_ready')
        if not broker_ready: missing.append('broker_not_ready')
    if not hydration_hard_ready:
        hydration_blockers = list(dict.fromkeys([*(str(m) for m in hydration_missing), *(str(m) for m in missing)]))
        block_reason = f"ACTIVE_BASKET_HYDRATION_NOT_READY:{','.join(hydration_blockers)}"
    else:
        block_reason=None if live_orders_armed else f"execution_not_armed:{','.join(dict.fromkeys(missing))}"
    ctx.data_hard_ready=data_hard_ready; ctx.evaluation_ready=evaluation_ready; ctx.live_orders_armed=live_orders_armed; ctx.trading_ready=evaluation_ready; ctx.live_block_reason=block_reason
    ctx.execution_ready_by_symbol = execution_ready_by_symbol
    ctx.selected_ce_exec_ready = bool(ce_exec_ready)
    ctx.selected_pe_exec_ready = bool(pe_exec_ready)
    ctx.context_exec_ready = bool(context_exec_ready)
    ctx.broker_ready = bool(broker_ready)
    LOGGER.info("LIVE_READINESS_COMPUTED selected_ce=%s selected_pe=%s ce_ltp_fresh=%s pe_ltp_fresh=%s ce_tick_age_s=%s pe_tick_age_s=%s ce_tradable_quote=%s pe_tradable_quote=%s ce_depth_available=%s pe_depth_available=%s ce_subscription_confirmed=%s pe_subscription_confirmed=%s ce_subscription_or_live_tick=%s pe_subscription_or_live_tick=%s ce_bars_effective=%s ce_mdm_bars=%s ce_runner_bars=%s pe_bars_effective=%s pe_mdm_bars=%s pe_runner_bars=%s data_hard_ready=%s evaluation_ready=%s trading_ready=%s live_orders_armed=%s ce_quote_ready=%s pe_quote_ready=%s ce_exec_ready=%s pe_exec_ready=%s direction_context_ready=%s execution_ready_by_symbol=%s live_block_reason=%s", selected_ce, selected_pe, ce_quote_fresh, pe_quote_fresh, getattr(_snapshot(selected_ce),'tick_age_s',None), getattr(_snapshot(selected_pe),'tick_age_s',None), _tradable_quote(selected_ce), _tradable_quote(selected_pe), bool(getattr(_snapshot(selected_ce),'depth_available',False)), bool(getattr(_snapshot(selected_pe),'depth_available',False)), _subscription_confirmed(selected_ce), _subscription_confirmed(selected_pe), _subscription_or_live_tick(selected_ce), _subscription_or_live_tick(selected_pe), ce_bars, ce_mdm_bars, ce_runner_bars, pe_bars, pe_mdm_bars, pe_runner_bars, data_hard_ready, evaluation_ready, bool(ctx.trading_ready), live_orders_armed, ce_quote_fresh, pe_quote_fresh, ce_exec_ready, pe_exec_ready, bool(context_exec_ready), execution_ready_by_symbol, block_reason)
    if ctx.strategy_runner is not None and hasattr(ctx.strategy_runner, 'set_runtime_readiness'):
        ctx.strategy_runner.set_runtime_readiness(data_hard_ready=bool(ctx.data_hard_ready), evaluation_ready=bool(ctx.evaluation_ready), live_orders_armed=bool(ctx.live_orders_armed), reason=str(ctx.live_block_reason or reason), selected_ce=selected_ce, selected_pe=selected_pe, atm_strike=basket.get('atm_strike'), option_symbols=option_symbols, execution_ready_by_symbol=dict(getattr(ctx, "execution_ready_by_symbol", {}) or {}))



def _hydrate_committed_active_basket(ctx: BotContext, *, reason: str) -> dict[str, object]:
    """Hydrate the committed ActiveContractBasket before strategy evaluation."""
    mdm = getattr(ctx, "market_data_manager", None)
    hydrate = getattr(mdm, "hydrate_active_contract_basket", None)
    mode = str(getattr(getattr(ctx, "settings", None), "execution_mode", os.getenv("EXECUTION_MODE", "PAPER")) or "PAPER").upper()
    allow_missing_gate = str(os.getenv("ALLOW_MISSING_HYDRATION_GATE", "false") or "false").strip().lower() == "true"
    basket = getattr(ctx, "active_contract_basket", None) or getattr(ctx, "active_trading_universe", None)
    if not callable(hydrate):
        hard_ready = bool(mode in {"PAPER", "SHADOW"} and allow_missing_gate)
        report: dict[str, object] = {
            "hard_ready": hard_ready,
            "missing": [] if hard_ready else ["hydrate_active_contract_basket_missing"],
            "symbols": {},
        }
        LOGGER.warning(
            "ACTIVE_BASKET_HYDRATION_METHOD_MISSING mode=%s hard_ready=%s reason=%s",
            mode,
            hard_ready,
            reason,
            extra={"event": "ACTIVE_BASKET_HYDRATION_METHOD_MISSING", "mode": mode, "hard_ready": hard_ready, "reason": reason},
        )
        ctx.active_basket_hydration = report
        return report
    try:
        report = dict(hydrate(basket))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "ACTIVE_BASKET_HYDRATION_FAILED reason=%s error=%s",
            reason,
            exc,
            extra={"event": "ACTIVE_BASKET_HYDRATION_FAILED", "reason": reason, "error": str(exc)},
        )
        report = {"hard_ready": False, "missing": [f"hydration_exception:{type(exc).__name__}"], "symbols": {}}

    spot_symbol = "NSE:NIFTY"
    if isinstance(basket, Mapping):
        spot_symbol = str(basket.get("spot_symbol") or spot_symbol)
    else:
        spot_symbol = str(getattr(basket, "spot_symbol", spot_symbol) or spot_symbol)
    spot_quote = None
    if mdm is not None:
        try:
            snap = getattr(mdm, "get_symbol_snapshot", lambda _s: None)(spot_symbol)
            spot_quote = snap if snap is not None else getattr(mdm, "get_latest_tick", lambda _s: None)(spot_symbol)
        except Exception:
            spot_quote = None
    try:
        spot_bars = getattr(mdm, "get_ohlc_bars", lambda _s: [])(spot_symbol) if mdm is not None else []
        spot_bars_count = len(spot_bars) if spot_bars is not None else 0
    except Exception:
        spot_bars_count = 0
    direction_ctx = getattr(ctx, "direction_context", None) or getattr(ctx, "underlying_direction_context", None) or {}
    if not isinstance(direction_ctx, Mapping):
        direction_ctx = {}
    direction_bias = (
        direction_ctx.get("direction_bias")
        or direction_ctx.get("underlying_direction_bias")
        or getattr(ctx, "underlying_direction_bias", None)
        or getattr(ctx, "direction_bias", None)
    )
    direction_age = direction_ctx.get("context_age_seconds") or direction_ctx.get("direction_context_age_seconds")
    report.update(
        {
            "spot_quote_ready": bool(spot_quote),
            "spot_bars_count": int(spot_bars_count),
            "underlying_direction_bias": direction_bias,
            "direction_context_age_seconds": direction_age,
            "direction_context_ready": bool(str(direction_bias or "").upper() in {"CE", "PE"}),
        }
    )
    ctx.active_basket_hydration = report
    if not bool(report.get("hard_ready")):
        missing = list(report.get("missing") or [])
        LOGGER.warning(
            "ACTIVE_BASKET_HYDRATION_NOT_READY reason=%s missing=%s",
            reason,
            missing,
            extra={"event": "ACTIVE_BASKET_HYDRATION_NOT_READY", "reason": reason, "missing": missing},
        )
    return report


def _ensure_active_symbol_tokens(ctx: Any) -> dict[str, int]:
    """Return ctx.active_symbol_tokens, defensively initializing legacy contexts."""
    current = getattr(ctx, "active_symbol_tokens", None)
    if not isinstance(current, dict):
        current = {}
        try:
            setattr(ctx, "active_symbol_tokens", current)
        except AttributeError:
            LOGGER.warning(
                "ACTIVE_SYMBOL_TOKENS_INIT_FAILED context_type=%s",
                type(ctx).__name__,
                extra={"event": "ACTIVE_SYMBOL_TOKENS_INIT_FAILED", "context_type": type(ctx).__name__},
            )
            return {}
    return current


def _bare_trading_symbol(symbol: object) -> str:
    """Return exchange-less trading symbol for token-map alias checks."""
    raw = str(symbol or "").strip()
    return raw.split(":", 1)[1] if ":" in raw else raw


def _coerce_positive_token(value: object) -> int | None:
    """Coerce valid positive instrument token values."""
    try:
        token = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return token if token > 0 else None


def _resolve_committed_symbol_token(
    ctx: BotContext,
    committed: Mapping[str, object],
    symbol: str | None,
    explicit_token_key: str | None = None,
) -> int | None:
    """Resolve a committed basket symbol token from explicit, full, and bare aliases."""
    if not symbol:
        return None
    aliases = list(dict.fromkeys([str(symbol), _bare_trading_symbol(symbol)]))
    if explicit_token_key:
        token = _coerce_positive_token(committed.get(explicit_token_key))
        if token is not None:
            return token
    token_map = committed.get("token_by_symbol")
    if isinstance(token_map, Mapping):
        for alias in aliases:
            token = _coerce_positive_token(token_map.get(alias))
            if token is not None:
                return token
    active_tokens = getattr(ctx, "active_symbol_tokens", None)
    if isinstance(active_tokens, Mapping):
        for alias in aliases:
            token = _coerce_positive_token(active_tokens.get(alias))
            if token is not None:
                return token
    instrument_manager = getattr(ctx, "instrument_manager", None)
    get_token = getattr(instrument_manager, "get_token", None)
    if callable(get_token):
        for alias in aliases:
            try:
                token = _coerce_positive_token(get_token(alias))
            except (KeyError, RuntimeError, TypeError, ValueError):
                token = None
            if token is not None:
                return token
    broker_client = getattr(ctx, "broker_client", None)
    get_instrument_token = getattr(broker_client, "get_instrument_token", None)
    if callable(get_instrument_token):
        for alias in aliases:
            try:
                token = _coerce_positive_token(get_instrument_token(alias))
            except (KeyError, RuntimeError, TypeError, ValueError):
                token = None
            if token is not None:
                return token
    return None

def _commit_active_dynamic_basket(
    ctx: BotContext,
    *,
    basket: Mapping[str, object],
    option_symbols: Sequence[str],
    symbols: Sequence[str],
    atm_strike: int | float | str | None,
) -> tuple[str | None, str | None]:
    """Commit active dynamic basket atomically. Args: ctx/basket/option_symbols/symbols/atm_strike. Returns: selected CE/PE. Raises: none."""
    requested_futures_symbol = basket.get("futures_symbol") or basket.get("future_symbol")
    active_futures_symbol = _resolve_active_futures_for_basket(ctx, requested_futures_symbol) or None
    if requested_futures_symbol and active_futures_symbol != str(requested_futures_symbol):
        LOGGER.info("ACTIVE_BASKET_FUTURES_NORMALIZED requested=%s active=%s", requested_futures_symbol, active_futures_symbol)
    basket_copy = dict(basket or {})
    basket_copy["futures_symbol"] = active_futures_symbol
    basket = normalize_active_basket_schema(basket_copy)
    mdm_for_purge = getattr(ctx, "market_data_manager", None)
    purge_stale_futures = getattr(mdm_for_purge, "purge_stale_nifty_futures", None)
    if callable(purge_stale_futures):
        purge_stale_futures(active_futures_symbol, reason="active_dynamic_basket_commit")
    current_options = [str(sym) for sym in option_symbols if str(sym).endswith(("CE", "PE"))]
    current_symbols = [str(sym) for sym in symbols if sym]
    local_basket = dict(basket or {})
    local_basket["option_symbols"] = list(current_options)
    local_basket["symbols"] = list(current_symbols)
    if atm_strike is not None:
        local_basket["atm_strike"] = atm_strike
    picked_ce, picked_pe = pick_atm_option_symbols_from_basket(local_basket)
    selected_ce = str(basket.get("selected_ce") or basket.get("atm_ce") or picked_ce or "") or None
    selected_pe = str(basket.get("selected_pe") or basket.get("atm_pe") or picked_pe or "") or None
    active_set = set(current_options) | set(current_symbols)
    def _nearest_for_side(side: str) -> str | None:
        candidates: list[tuple[float, str]] = []
        fallback: list[str] = []
        for sym in current_options:
            if not sym.endswith(side):
                continue
            strike_digits = ""
            for char in reversed(sym[:-2]):
                if char.isdigit():
                    strike_digits = char + strike_digits
                elif strike_digits:
                    break
            if not strike_digits:
                fallback.append(sym)
                continue
            if atm_strike is None:
                candidates.append((0.0, sym))
            else:
                candidates.append((abs(float(strike_digits) - float(atm_strike)), sym))
        if not candidates:
            return sorted(fallback)[0] if fallback else None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][1]
    if not (selected_ce and selected_ce.endswith("CE") and selected_ce in active_set):
        selected_ce = _nearest_for_side("CE")
    if not (selected_pe and selected_pe.endswith("PE") and selected_pe in active_set):
        selected_pe = _nearest_for_side("PE")
    old_ce = getattr(ctx, "selected_ce", None)
    old_pe = getattr(ctx, "selected_pe", None)
    if not selected_ce and old_ce in active_set and str(old_ce).endswith("CE"):
        selected_ce = old_ce
    if not selected_pe and old_pe in active_set and str(old_pe).endswith("PE"):
        selected_pe = old_pe
    if current_options and (not selected_ce or not selected_pe):
        LOGGER.warning(
            "ACTIVE_DYNAMIC_BASKET_DEFERRED reason=selected_option_resolution_failed option_count=%d selected_ce=%s selected_pe=%s",
            len(current_options),
            selected_ce,
            selected_pe,
            extra={
                "event": "ACTIVE_DYNAMIC_BASKET_DEFERRED",
                "reason": "selected_option_resolution_failed",
                "option_count": len(current_options),
                "selected_ce": selected_ce,
                "selected_pe": selected_pe,
            },
        )
        return cast(str | None, old_ce), cast(str | None, old_pe)
    ctx.selected_ce = str(selected_ce) if selected_ce else None
    ctx.selected_pe = str(selected_pe) if selected_pe else None
    ctx.atm_ce_symbol = selected_ce
    ctx.atm_pe_symbol = selected_pe
    ce_symbols = [sym for sym in current_options if str(sym).endswith("CE")]
    pe_symbols = [sym for sym in current_options if str(sym).endswith("PE")]
    if selected_ce and selected_ce not in ce_symbols:
        ce_symbols.append(selected_ce)
    if selected_pe and selected_pe not in pe_symbols:
        pe_symbols.append(selected_pe)
    committed = cast(dict[str, object], getattr(ctx, "active_trading_universe", {}) or {})
    committed.update(
        {
            "spot_symbol": basket.get("spot_symbol") or "NSE:NIFTY",
            "spot_token": basket.get("spot_token"),
            "futures_symbol": active_futures_symbol,
            "futures_token": basket.get("futures_token"),
            "selected_ce_token": basket.get("selected_ce_token"),
            "selected_pe_token": basket.get("selected_pe_token"),
            "selected_ce": selected_ce,
            "selected_pe": selected_pe,
            "atm_ce": selected_ce,
            "atm_pe": selected_pe,
            "option_symbols": list(current_options),
            "ce_symbols": list(ce_symbols),
            "pe_symbols": list(pe_symbols),
            "symbols": list(
                dict.fromkeys(
                    [
                        s
                        for s in [
                            basket.get("spot_symbol") or "NSE:NIFTY",
                            active_futures_symbol or None,
                            *current_options,
                        ]
                        if s
                    ]
                )
            ),
            "atm_strike": atm_strike,
            "committed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    committed["option_tokens"] = list(basket.get("option_tokens") or [])
    token_by_symbol = dict(basket.get("token_by_symbol") or {})
    symbol_by_token = dict(basket.get("symbol_by_token") or {})
    all_symbols = list(basket.get("all_symbols") or committed.get("symbols") or [])
    all_tokens = list(basket.get("all_tokens") or [])
    for explicit_symbol, explicit_key in (
        (selected_ce, "selected_ce_token"),
        (selected_pe, "selected_pe_token"),
        (active_futures_symbol, "futures_token"),
    ):
        explicit_token = _coerce_positive_token(committed.get(explicit_key))
        if explicit_symbol and explicit_token is not None:
            token_by_symbol.setdefault(str(explicit_symbol), explicit_token)
            token_by_symbol.setdefault(_bare_trading_symbol(explicit_symbol), explicit_token)
            symbol_by_token.setdefault(explicit_token, str(explicit_symbol))
    for sym in list(all_symbols):
        sym_str = str(sym)
        bare_sym = _bare_trading_symbol(sym_str)
        full_token = _coerce_positive_token(token_by_symbol.get(sym_str))
        bare_token = _coerce_positive_token(token_by_symbol.get(bare_sym))
        if full_token is None and bare_token is not None:
            token_by_symbol[sym_str] = bare_token
        elif full_token is not None and bare_sym:
            token_by_symbol.setdefault(bare_sym, full_token)
    if active_futures_symbol and active_futures_symbol not in all_symbols:
        all_symbols.insert(1 if all_symbols else 0, active_futures_symbol)
    futures_token = basket.get("futures_token")
    if active_futures_symbol and futures_token is not None:
        try:
            fut_token_int = int(futures_token)
            token_by_symbol.setdefault(active_futures_symbol, fut_token_int)
            symbol_by_token.setdefault(fut_token_int, active_futures_symbol)
            if fut_token_int not in [int(t) for t in all_tokens if t is not None]:
                all_tokens.insert(1 if all_tokens else 0, fut_token_int)
        except (TypeError, ValueError):
            pass
    if token_by_symbol and not all_tokens:
        all_tokens = [int(token_by_symbol[s]) for s in all_symbols if s in token_by_symbol]
    if token_by_symbol and all_symbols:
        all_tokens = [int(token_by_symbol[s]) for s in all_symbols if s in token_by_symbol]
    committed["all_symbols"] = list(dict.fromkeys(all_symbols))
    committed["all_tokens"] = list(dict.fromkeys(all_tokens))
    committed["token_by_symbol"] = token_by_symbol
    committed["symbol_by_token"] = symbol_by_token
    ctx.active_trading_universe = committed
    ctx.active_contract_basket = committed
    active_symbol_tokens = _ensure_active_symbol_tokens(ctx)
    active_symbol_tokens.update(dict(committed.get("token_by_symbol") or {}))
    selected_ce_token = _resolve_committed_symbol_token(ctx, committed, selected_ce, "selected_ce_token")
    selected_pe_token = _resolve_committed_symbol_token(ctx, committed, selected_pe, "selected_pe_token")
    for selected_symbol, selected_token, token_key in (
        (selected_ce, selected_ce_token, "selected_ce_token"),
        (selected_pe, selected_pe_token, "selected_pe_token"),
    ):
        token_int = _coerce_positive_token(selected_token)
        if selected_symbol and token_int is not None:
            full_symbol = str(selected_symbol)
            bare_symbol = _bare_trading_symbol(full_symbol)
            token_by_symbol[full_symbol] = token_int
            token_by_symbol[bare_symbol] = token_int
            active_symbol_tokens[full_symbol] = token_int
            active_symbol_tokens[bare_symbol] = token_int
            symbol_by_token.setdefault(token_int, full_symbol)
            committed[token_key] = token_int
            existing_all_tokens = {coerced for raw in committed.get("all_tokens", []) if (coerced := _coerce_positive_token(raw)) is not None}
            existing_option_tokens = {coerced for raw in committed.get("option_tokens", []) if (coerced := _coerce_positive_token(raw)) is not None}
            if token_int not in existing_all_tokens:
                committed.setdefault("all_tokens", []).append(token_int)
            if token_int not in existing_option_tokens:
                committed.setdefault("option_tokens", []).append(token_int)
    committed["token_by_symbol"] = token_by_symbol
    committed["symbol_by_token"] = symbol_by_token
    committed["all_tokens"] = list(dict.fromkeys(coerced for raw in (committed.get("all_tokens") or []) if (coerced := _coerce_positive_token(raw)) is not None))
    committed["option_tokens"] = list(dict.fromkeys(coerced for raw in (committed.get("option_tokens") or []) if (coerced := _coerce_positive_token(raw)) is not None))
    if getattr(ctx, "option_universe", None) is not None and hasattr(ctx.option_universe, "set_active_contract_basket"):
        ctx.option_universe.set_active_contract_basket(committed)
    mdm = getattr(ctx, "market_data_manager", None)
    mdm_set = getattr(mdm, "set_active_contract_basket", None)
    if callable(mdm_set):
        mdm_set(committed)
    request_subscription = getattr(mdm, "request_token_subscription", None)
    request_subscriptions = getattr(mdm, "request_token_subscriptions", None)
    selected_subscription_tokens = [tok for tok in (selected_ce_token, selected_pe_token) if tok is not None]
    if callable(request_subscription):
        for selected_symbol, selected_token in ((selected_ce, selected_ce_token), (selected_pe, selected_pe_token)):
            token_int = _coerce_positive_token(selected_token)
            if selected_symbol and token_int is not None:
                request_subscription(token_int, symbol=str(selected_symbol))
    elif callable(request_subscriptions) and selected_subscription_tokens:
        request_subscriptions(int(tok) for tok in selected_subscription_tokens if tok is not None)
    hub_set = getattr(getattr(ctx, "data_hub", None), "set_active_contract_basket", None)
    if callable(hub_set):
        hub_set(committed)
    desired_count_fn = getattr(mdm, "desired_token_count", None)
    ws_count_fn = getattr(mdm, "ws_token_count", None)
    desired_token_count = int(desired_count_fn()) if callable(desired_count_fn) else len(committed.get("all_tokens") or [])
    ws_count_value = ws_count_fn() if callable(ws_count_fn) else None
    ws_token_count = int(ws_count_value) if ws_count_value is not None else None
    futures_token_for_log = _resolve_committed_symbol_token(ctx, committed, active_futures_symbol, "futures_token")
    if selected_ce and selected_ce_token is None:
        ctx.live_orders_armed = False
        ctx.trading_ready = False
        ctx.live_block_reason = "option_token_missing:selected_ce"
    if selected_pe and selected_pe_token is None:
        ctx.live_orders_armed = False
        ctx.trading_ready = False
        ctx.live_block_reason = "option_token_missing:selected_pe"
    LOGGER.info(
        "ACTIVE_BASKET_SUBSCRIPTION_RECONCILED selected_ce=%s selected_ce_token=%s selected_pe=%s selected_pe_token=%s futures_symbol=%s futures_token=%s desired_token_count=%s ws_token_count=%s",
        selected_ce,
        selected_ce_token,
        selected_pe,
        selected_pe_token,
        active_futures_symbol,
        futures_token_for_log,
        desired_token_count,
        ws_token_count,
        extra={"event": "ACTIVE_BASKET_SUBSCRIPTION_RECONCILED", "selected_ce": selected_ce, "selected_ce_token": selected_ce_token, "selected_pe": selected_pe, "selected_pe_token": selected_pe_token, "futures_symbol": active_futures_symbol, "futures_token": futures_token_for_log, "desired_token_count": desired_token_count, "ws_token_count": ws_token_count},
    )
    LOGGER.info(
        "SELECTED_OPTION_SUBSCRIPTION_CONFIRMED selected_ce=%s selected_ce_token=%s selected_pe=%s selected_pe_token=%s desired_token_count=%s ws_token_count=%s",
        selected_ce,
        selected_ce_token,
        selected_pe,
        selected_pe_token,
        desired_token_count,
        ws_token_count,
        extra={"event": "SELECTED_OPTION_SUBSCRIPTION_CONFIRMED", "selected_ce": selected_ce, "selected_ce_token": selected_ce_token, "selected_pe": selected_pe, "selected_pe_token": selected_pe_token, "desired_token_count": desired_token_count, "ws_token_count": ws_token_count},
    )
    _hydrate_committed_active_basket(ctx, reason="active_dynamic_basket_commit")
    LOGGER.info(
        "ACTIVE_CONTRACT_BASKET_COMMITTED selected_ce=%s selected_pe=%s futures_symbol=%s atm_strike=%s option_count=%d token_count=%d",
        selected_ce, selected_pe, active_futures_symbol, atm_strike, len(current_options), len(committed.get("all_tokens") or []),
        extra={"event": "ACTIVE_CONTRACT_BASKET_COMMITTED", "selected_ce": selected_ce, "selected_pe": selected_pe, "futures_symbol": active_futures_symbol, "atm_strike": atm_strike, "option_count": len(current_options), "token_count": len(committed.get("all_tokens") or [])},
    )
    runner = getattr(ctx, "strategy_runner", None)
    if runner is not None and hasattr(runner, "set_active_trading_universe"):
        runner.set_active_trading_universe(committed)
    strategy_manager = getattr(ctx, "strategy_manager", None)
    set_fut = getattr(strategy_manager, "set_active_futures_symbol", None)
    if callable(set_fut):
        set_fut(active_futures_symbol, source="active_dynamic_basket_commit")
    mdm = getattr(ctx, "market_data_manager", None)
    rotate_result = getattr(mdm, "maybe_rotate_nifty_futures_context_result", None)
    if callable(rotate_result):
        try:
            rotate_result(
                requested_futures_symbol,
                reason="active_dynamic_basket_commit",
                selected_option_symbols=list(current_options),
            )
        except Exception as exc:
            LOGGER.warning(
                "ACTIVE_BASKET_FUTURES_ROTATION_CHECK_FAILED error=%s",
                exc,
                extra={"event": "ACTIVE_BASKET_FUTURES_ROTATION_CHECK_FAILED", "error": str(exc)},
            )
    return selected_ce, selected_pe


def _register_and_subscribe_live_symbol(
    ctx: BotContext, symbol: str | None, token: int | None, reason: str, role: str = "tradable_option"
) -> bool:
    """Register and subscribe symbol in MDM/DataHub/Runner. Args: ctx/symbol/token/reason. Returns: success. Raises: none."""
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return False
    resolved_token = token
    if resolved_token is None:
        resolved_token = (getattr(ctx, "active_symbol_tokens", {}) or {}).get(normalized)
    if resolved_token is None and getattr(ctx, "instrument_manager", None) is not None:
        try:
            resolved_token = ctx.instrument_manager.get_token(normalized)
        except Exception:
            resolved_token = None
    if resolved_token is None and getattr(ctx, "broker_client", None) is not None:
        try:
            resolved_token = ctx.broker_client.get_instrument_token(normalized)
        except Exception:
            resolved_token = None
    LOGGER.info("LIVE_SYMBOL_SUBSCRIBE_REQUESTED symbol=%s token=%s role=%s reason=%s", normalized, resolved_token, role, reason)
    if role == "tradable_option":
        LOGGER.info("LIVE_OPTION_SUBSCRIBE_REQUESTED symbol=%s token=%s reason=%s", normalized, resolved_token, reason)
    mdm = getattr(ctx, "market_data_manager", None)
    if mdm is not None and resolved_token is not None:
        if hasattr(mdm, "register_symbol"):
            mdm.register_symbol(normalized, int(resolved_token))
        if hasattr(mdm, "request_token_subscription"):
            mdm.request_token_subscription(int(resolved_token), symbol=normalized)
    runner = getattr(ctx, "strategy_runner", None)
    if runner is not None and hasattr(runner, "add_symbol"):
        runner.add_symbol(normalized)
    hub = getattr(ctx, "data_hub", None)
    if hub is not None and runner is not None and hasattr(hub, "subscribe_ticks"):
        hub.subscribe_ticks(
            normalized, runner.on_datahub_tick, token=resolved_token, force_live=True
        )
    LOGGER.info("LIVE_SYMBOL_SUBSCRIBE_CONFIRMED symbol=%s token=%s role=%s subscribed=%s", normalized, resolved_token, role, bool(resolved_token))
    if role == "tradable_option":
        LOGGER.info("LIVE_OPTION_SUBSCRIBE_CONFIRMED symbol=%s token=%s subscribed=%s", normalized, resolved_token, bool(resolved_token))
    return bool(resolved_token)
async def _deferred_basket_hydration_retry(
    ctx: BotContext,
    *,
    configured_mode: str,
    max_attempts: int | None = None,
    delay_seconds: float = 15.0,
) -> None:
    """Retry startup basket hydration until fresh WS spot arrives. Args: ctx/mode. Returns: none. Raises: none."""

    if max_attempts is None:
        max_attempts = int(os.getenv("DEFERRED_BASKET_MAX_ATTEMPTS", "160") or "160")
    policy = MarketDataPolicy.from_env()
    for attempt in range(1, max_attempts + 1):
        await asyncio.sleep(delay_seconds)
        if getattr(ctx, "trading_ready", False) and getattr(ctx, "live_orders_armed", False):
            return
        LOGGER.info(
            "DEFERRED_BASKET_RETRY_STARTED attempt=%d/%d",
            attempt,
            max_attempts,
            extra={"event": "DEFERRED_BASKET_RETRY_STARTED", "attempt": attempt, "max_attempts": max_attempts},
        )
        try:
            spot_ltp = await _wait_for_live_spot_or_raise(
                ctx,
                timeout=min(10.0, float(policy.startup_wait_for_ws_spot_seconds or 60.0)),
                configured_mode=configured_mode,
            )
        except RuntimeError as exc:
            LOGGER.info(
                "DEFERRED_BASKET_RETRY_FAILED attempt=%d/%d reason=%s",
                attempt,
                max_attempts,
                str(exc) or "spot_ltp_unavailable",
                extra={
                    "event": "DEFERRED_BASKET_RETRY_FAILED",
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "reason": str(exc) or "spot_ltp_unavailable",
                },
            )
            continue

        try:
            basket = await _build_and_hydrate_live_basket_from_spot(
                ctx,
                spot_ltp=float(spot_ltp),
                configured_mode=configured_mode,
            )
            if isinstance(basket, Mapping) and basket.get("deferred"):
                LOGGER.info(
                    "DEFERRED_BASKET_RETRY_FAILED attempt=%d/%d reason=%s",
                    attempt,
                    max_attempts,
                    basket.get("reason") or "basket_build_deferred",
                    extra={"event": "DEFERRED_BASKET_RETRY_FAILED", "attempt": attempt, "max_attempts": max_attempts, "reason": basket.get("reason") or "basket_build_deferred"},
                )
                continue
            if basket and not getattr(ctx, "selected_ce", None):
                _commit_active_dynamic_basket(
                    ctx,
                    basket=basket,
                    option_symbols=basket.get("option_symbols") or [],
                    symbols=basket.get("symbols") or [],
                    atm_strike=basket.get("atm_strike"),
                )
            await _ensure_strategy_runner_started(
                ctx,
                reason="deferred_basket_hydration_success",
            )
            await _recompute_and_push_runtime_readiness(
                ctx, reason="deferred_basket_hydration_success"
            )
            LOGGER.info(
                "DEFERRED_BASKET_RETRY_SUCCESS attempt=%d spot_ltp=%.2f",
                attempt,
                float(spot_ltp),
                extra={
                    "event": "DEFERRED_BASKET_RETRY_SUCCESS",
                    "attempt": attempt,
                    "spot_ltp": float(spot_ltp),
                },
            )
            return
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "DEFERRED_BASKET_RETRY_FAILED attempt=%d/%d reason=%s",
                attempt,
                max_attempts,
                exc,
                exc_info=True,
                extra={
                    "event": "DEFERRED_BASKET_RETRY_FAILED",
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "reason": str(exc),
                },
            )
    LOGGER.error(
        "DEFERRED_BASKET_RETRY_EXHAUSTED attempts=%d",
        max_attempts,
        extra={"event": "DEFERRED_BASKET_RETRY_EXHAUSTED", "attempts": max_attempts},
    )


async def _build_and_hydrate_live_basket_from_spot(
    ctx: BotContext,
    *,
    spot_ltp: float,
    configured_mode: str,
    hydrate: bool = False,
) -> dict[str, Any]:
    """Build/register/hydrate live basket from a trusted spot. Args: ctx/spot/mode. Returns: basket. Raises: RuntimeError."""

    del configured_mode
    lock = _get_basket_build_lock(ctx)
    if lock.locked():
        LOGGER.info(
            "LIVE_BASKET_BUILD_SKIPPED reason=already_running spot_ltp=%.2f hydrate=%s",
            float(spot_ltp),
            bool(hydrate),
        )
        return {'deferred': True, 'reason': 'already_running'}

    async with lock:
        start = time_module.monotonic()
        ctx.basket_build_in_progress = True
        ctx.basket_build_last_started_mono = start
        ctx.basket_build_last_error = None
        try:
            if float(spot_ltp) <= 0:
                raise RuntimeError('spot_ltp must be positive for live basket hydration')
            LOGGER.info(
                'LIVE_BASKET_BUILD_STARTED spot_ltp=%.2f hydrate=%s trigger=%s',
                float(spot_ltp),
                bool(hydrate),
                'spot_first_tick',
            )
            if ctx.instrument_manager is None or not ctx.instrument_manager.is_loaded():
                duration_ms = int((time_module.monotonic() - start) * 1000)
                LOGGER.warning(
                    'LIVE_BASKET_BUILD_DEFERRED reason=instrument_manager_not_ready duration_ms=%d recoverable=True',
                    duration_ms,
                )
                ctx.basket_build_last_error = 'instrument_manager_not_ready'
                _schedule_deferred_basket_retry(
                    ctx,
                    configured_mode='LIVE',
                    reason='instrument_manager_not_ready',
                    spot_ltp=float(spot_ltp),
                )
                return {'deferred': True, 'reason': 'instrument_manager_not_ready', 'symbol_count': 0}
            if ctx.market_data_manager is None:
                raise RuntimeError('market_data_manager_unavailable_for_live_basket')
            if ctx.broker_client is None:
                raise RuntimeError('broker_client_unavailable_for_live_basket')

            basket = _build_canonical_active_basket(
                instrument_manager=ctx.instrument_manager,
                spot_token_resolver=lambda symbol: int(ctx.broker_client.get_instrument_token(symbol)),
                spot_ltp=float(spot_ltp),
                futures_symbol=None,
                strike_step=int(ctx.settings.option_universe.strike_step or 50),
                strikes_around_atm=2,
            )
            duration_ms = int((time_module.monotonic() - start) * 1000)
            ctx.basket_build_last_completed_mono = time_module.monotonic()
            LOGGER.info(
                'LIVE_BASKET_BUILD_COMPLETE selected_ce=%s selected_pe=%s atm_strike=%s symbol_count=%d hydrated=%s duration_ms=%d',
                basket.get('selected_ce') or basket.get('atm_ce'),
                basket.get('selected_pe') or basket.get('atm_pe'),
                basket.get('atm_strike'),
                len(list(dict.fromkeys(basket.get('symbols', [])))),
                bool(hydrate),
                duration_ms,
            )
            option_symbols = [
                str(sym)
                for sym in dict.fromkeys(basket.get("option_symbols") or [])
                if str(sym).endswith(("CE", "PE"))
            ]
            symbols = [
                str(sym)
                for sym in dict.fromkeys(basket.get("symbols") or [])
                if sym
            ]
            committed_ce, committed_pe = _commit_active_dynamic_basket(
                ctx,
                basket=basket,
                option_symbols=option_symbols,
                symbols=symbols,
                atm_strike=basket.get("atm_strike"),
            )
            LOGGER.info(
                "LIVE_BASKET_COMMITTED_FROM_BUILD selected_ce=%s selected_pe=%s atm_strike=%s option_count=%d",
                committed_ce,
                committed_pe,
                basket.get("atm_strike"),
                len(option_symbols),
                extra={
                    "event": "LIVE_BASKET_COMMITTED_FROM_BUILD",
                    "selected_ce": committed_ce,
                    "selected_pe": committed_pe,
                    "atm_strike": basket.get("atm_strike"),
                    "option_count": len(option_symbols),
                },
            )
            subscription_symbols = list(dict.fromkeys([*symbols, committed_ce, committed_pe]))
            token_map = dict(getattr(ctx, "active_symbol_tokens", {}) or {})
            for live_symbol in subscription_symbols:
                if not live_symbol:
                    continue
                role = "tradable_option"
                if str(live_symbol) == str((basket or {}).get("spot_symbol") or "NSE:NIFTY"):
                    role = "spot_context"
                elif str(live_symbol) == str((basket or {}).get("futures_symbol") or ""):
                    role = "futures_context"
                _register_and_subscribe_live_symbol(
                    ctx,
                    str(live_symbol),
                    token_map.get(str(live_symbol)),
                    "basket_commit_live_startup",
                    role,
                )
            return dict(getattr(ctx, "active_trading_universe", {}) or basket)
        except Exception as exc:
            duration_ms = int((time_module.monotonic() - start) * 1000)
            ctx.basket_build_last_error = str(exc)
            LOGGER.exception(
                'LIVE_BASKET_BUILD_FAILED reason=%s duration_ms=%d recoverable=%s',
                str(exc),
                duration_ms,
                True,
            )
            raise
        finally:
            ctx.basket_build_in_progress = False


def _get_basket_build_lock(ctx: BotContext) -> asyncio.Lock:
    """Get/create basket-build lock. Args: ctx. Returns: lock. Raises: none."""

    lock = getattr(ctx, 'basket_build_lock', None)
    if lock is None:
        lock = asyncio.Lock()
        ctx.basket_build_lock = lock
    return lock


def _schedule_deferred_basket_retry(ctx: BotContext, *, configured_mode: str, reason: str = "market_closed_or_spot_not_ready", spot_ltp: float | None = None) -> None:
    """Schedule one deferred basket retry task. Args: ctx/mode. Returns: none. Raises: none."""

    ctx.live_orders_armed = False
    ctx.trading_ready = False
    ctx.readiness_mode = "DATA_WARMUP"
    ctx.effective_mode = ctx.readiness_mode
    ctx.live_block_reason = "spot_ltp_unavailable" if reason == "spot_ltp_unavailable" else "fresh_ws_spot_unavailable"
    if bool(getattr(ctx, "deferred_basket_retry_started", False)):
        LOGGER.info("DEFERRED_BASKET_RETRY_SCHEDULED reason=%s spot_ltp=%s already_scheduled=%s", reason, spot_ltp, True)
        return

    ctx.deferred_basket_retry_started = True
    ctx.last_deferred_basket_retry_ts = time_module.time()
    ctx.deferred_basket_retry_task = _create_named_task(
        _deferred_basket_hydration_retry(
            ctx,
            configured_mode=configured_mode,
        ),
        name="deferred_basket_hydration_retry",
    )
    LOGGER.info(
        "DEFERRED_BASKET_RETRY_SCHEDULED reason=%s spot_ltp=%s",
        reason,
        spot_ltp,
        extra={
            "event": "DEFERRED_BASKET_RETRY_SCHEDULED",
            "reason": reason,
        },
    )


def _resolve_quote_capability(ctx: BotContext) -> dict[str, Any]:
    """Combine MDM and broker quote capability snapshots. Args: ctx. Returns: combined snapshot. Raises: none."""
    available = True
    error: str | None = None
    sources: list[dict[str, Any]] = []
    for holder in (
        ctx.market_data_manager,
        ctx.broker_client,
        getattr(ctx.broker_client, "_broker", None),
        getattr(ctx.broker_client, "broker", None),
    ):
        if holder is None:
            continue
        snap_fn = getattr(holder, "quote_api_status_snapshot", None)
        if not callable(snap_fn):
            continue
        try:
            snap = snap_fn() or {}
        except Exception:  # noqa: BLE001
            continue
        snap_available = bool(snap.get("available", True))
        snap_error = snap.get("error")
        sources.append({"available": snap_available, "error": snap_error})
        if not snap_available:
            available = False
            if snap_error and not error:
                error = str(snap_error)
    return {"available": available, "error": error, "sources": sources}


async def startup_sequence(ctx: BotContext) -> None:
    """Execute startup sequence with Smart Hydration and Option-Only Trading."""
    policy = MarketDataPolicy.from_env()
    _set_startup_phase(ctx, "create_context")
    configured_mode = str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper()
    if configured_mode not in {"LIVE", "PAPER", "SHADOW"}:
        configured_mode = "SHADOW"
    ctx.live_orders_armed = False
    ctx.trading_ready = False
    ctx.readiness_mode = "DATA_WARMUP" if configured_mode == "LIVE" else configured_mode
    ctx.effective_mode = ctx.readiness_mode
    _ensure_active_symbol_tokens(ctx)
    LOGGER.info(
        "Startup | configured_mode=%s | effective_mode=%s | live_orders_armed=%s | trading_ready=%s | data_dir=%s | port=%s",
        configured_mode,
        ctx.readiness_mode,
        ctx.live_orders_armed,
        ctx.trading_ready,
        ctx.settings.data_dir,
        os.getenv("PORT"),
    )

    loop = asyncio.get_running_loop()
    _set_startup_phase(ctx, "create_data_layer")
    # DataHub forwards to MDM internally; keep MDM call as fallback when DH absent.
    if ctx.data_hub is not None:
        ctx.data_hub.set_event_loop(loop)
    elif ctx.market_data_manager is not None and hasattr(
        ctx.market_data_manager, "set_event_loop"
    ):
        ctx.market_data_manager.set_event_loop(loop)

    _set_startup_phase(ctx, "create_strategy_layer")
    _set_startup_phase(ctx, "wire_message_bus")
    _wire_and_start_message_bus(ctx)
    _set_startup_phase(ctx, "wire_datahub_to_runner")
    _set_startup_phase(ctx, "start_market_data")

    # ── START MDM (CRITICAL) ─────────────────────────────────────────────────
    # MDM.start(defer_ws=True) launches the tick consumer, health monitor,
    # and optional REST poll thread — but defers the WebSocket connect until
    # the full universe of tokens (spot + futures + ATM options) is resolved
    # during the hydration step below.  Bringing up the WebSocket _after_
    # token resolution is what lets the initial ``_on_connect`` subscribe
    # observe every instrument instead of just the spot token — this is the
    # fix for the "WebSocket CONNECTED successfully | tokens=1" boot banner.
    if ctx.market_data_manager is not None and hasattr(ctx.market_data_manager, "start"):
        try:
            ctx.market_data_manager.start(defer_ws=True)
            LOGGER.info(
                "✅ MarketDataManager started — tick consumer active (WS deferred)"
            )
            _safe_startup_log(
                LOGGER,
                logging.INFO,
                "STARTUP_WS_SPOT_FIRST_DECISION",
                "STARTUP_WS_SPOT_FIRST_DECISION websocket_enabled=%s mdm_present=%s has_start_websocket=%s",
                ctx.websocket_enabled,
                ctx.market_data_manager is not None,
                hasattr(ctx.market_data_manager, "start_websocket")
                if ctx.market_data_manager
                else False,
                websocket_enabled=ctx.websocket_enabled,
                mdm_present=ctx.market_data_manager is not None,
                has_start_websocket=hasattr(ctx.market_data_manager, "start_websocket")
                if ctx.market_data_manager
                else False,
            )
        except Exception as _mdm_start_exc:
            LOGGER.error("MarketDataManager.start() failed: %s", _mdm_start_exc)

    # ── Bring the WebSocket online with the NIFTY spot token FIRST ──
    # The bot must observe a real exchange tick before selecting ATM
    # options.  Starting the WS now (with the canonical NIFTY spot token
    # already registered in MarketDataManager.__init__) lets us prove a
    # fresh spot price before any option-universe selection runs and
    # guarantees the readiness gate has a tick to evaluate.  Option
    # tokens are added later via request_token_subscription once the
    # ATM basket is known.
    if (
        ctx.market_data_manager is not None
        and getattr(ctx, "websocket_enabled", True)
        and hasattr(ctx.market_data_manager, "start_websocket")
    ):
        try:
            LOGGER.info(
                "STARTUP_SPOT_FIRST_BLOCK_ENTERED websocket_enabled=%s mdm_id=%s",
                ctx.websocket_enabled,
                id(ctx.market_data_manager),
                extra={
                    "event": "STARTUP_SPOT_FIRST_BLOCK_ENTERED",
                    "websocket_enabled": ctx.websocket_enabled,
                    "mdm_id": id(ctx.market_data_manager),
                },
            )
            ws_status_fn = getattr(ctx.market_data_manager, "ws_status_snapshot", None)
            try:
                ws_status = ws_status_fn() if callable(ws_status_fn) else {}
            except Exception as exc:
                ws_status = {}
                LOGGER.warning(
                    "STARTUP_WS_STATUS_SNAPSHOT_FAILED err=%s",
                    exc,
                    extra={
                        "event": "STARTUP_WS_STATUS_SNAPSHOT_FAILED",
                        "error": str(exc),
                    },
                    exc_info=True,
                )
            if not getattr(ctx, "startup_spot_refresh_done", False):
                def _startup_spot_tick_listener(tick: Mapping[str, Any]) -> None:
                    try:
                        symbol = str(tick.get("symbol") or "").upper()
                        token_raw = tick.get("instrument_token") or tick.get("token")
                        try:
                            token_int = int(token_raw) if token_raw is not None else None
                        except (TypeError, ValueError):
                            token_int = None
                        spot_aliases = {policy.nifty_internal_symbol, "NIFTY", "NIFTY 50", "NSE:NIFTY50", "NIFTY50"}
                        if symbol not in spot_aliases and token_int != int(policy.nifty_spot_token):
                            return
                        if ctx.startup_spot_refresh_done:
                            return

                        ctx.startup_spot_refresh_done = True

                        loop.call_soon_threadsafe(
                            lambda: _create_named_task(
                                _refresh_readiness_after_first_tick(
                                    ctx,
                                    reason="first_spot_tick_listener",
                                ),
                                name="startup_spot_tick_listener",
                            )
                        )
                    except Exception as exc:
                        LOGGER.exception(
                            "STARTUP_SPOT_TICK_LISTENER_FAILED symbol=%s error=%s",
                            tick.get("symbol") if isinstance(tick, Mapping) else None,
                            exc,
                            extra={
                                "event": "STARTUP_SPOT_TICK_LISTENER_FAILED",
                                "error": str(exc),
                            },
                        )

                if ctx.data_hub is not None:
                    _subscribe_ticks_force_live_compat(
                        ctx.data_hub,
                        policy.nifty_internal_symbol,
                        _startup_spot_tick_listener,
                        token=policy.nifty_spot_token,
                    )
            desired_token_count_fn = getattr(
                ctx.market_data_manager, "desired_token_count", None
            )
            desired_token_count = (
                int(desired_token_count_fn()) if callable(desired_token_count_fn) else None
            )
            register_fn = getattr(ctx.market_data_manager, "register_symbol", None)
            _safe_startup_log(
                LOGGER, logging.INFO, "STARTUP_SPOT_REGISTER_BEGIN",
                "STARTUP_SPOT_REGISTER_BEGIN symbol=%s token=%d desired_token_count=%s ws_token_count=%s websocket_enabled=%s phase=spot_first",
                policy.nifty_internal_symbol,
                policy.nifty_spot_token,
                desired_token_count,
                ws_status.get("tokens"),
                ctx.websocket_enabled,
                symbol=policy.nifty_internal_symbol, token=policy.nifty_spot_token, desired_token_count=desired_token_count, ws_token_count=ws_status.get("tokens"), websocket_enabled=ctx.websocket_enabled, phase="spot_first",
            )
            if callable(register_fn):
                try:
                    register_fn(policy.nifty_internal_symbol, policy.nifty_spot_token)
                except Exception:  # noqa: BLE001
                    LOGGER.debug(
                        "Spot symbol registration failed (will retry later)",
                        exc_info=True,
                    )
            _safe_startup_log(
                LOGGER, logging.INFO, "STARTUP_SPOT_REGISTER_DONE",
                "STARTUP_SPOT_REGISTER_DONE symbol=%s token=%d desired_token_count=%s ws_token_count=%s websocket_enabled=%s phase=spot_first",
                policy.nifty_internal_symbol,
                policy.nifty_spot_token,
                desired_token_count,
                ws_status.get("tokens"),
                ctx.websocket_enabled,
                symbol=policy.nifty_internal_symbol, token=policy.nifty_spot_token, desired_token_count=desired_token_count, ws_token_count=ws_status.get("tokens"), websocket_enabled=ctx.websocket_enabled, phase="spot_first",
            )
            subscribe_fn = getattr(
                ctx.market_data_manager, "request_token_subscription", None
            )
            _safe_startup_log(
                LOGGER, logging.INFO, "STARTUP_SPOT_TOKEN_REQUEST_BEGIN",
                "STARTUP_SPOT_TOKEN_REQUEST_BEGIN symbol=%s token=%d desired_token_count=%s ws_token_count=%s websocket_enabled=%s phase=spot_first",
                policy.nifty_internal_symbol,
                policy.nifty_spot_token,
                desired_token_count,
                ws_status.get("tokens"),
                ctx.websocket_enabled,
                symbol=policy.nifty_internal_symbol, token=policy.nifty_spot_token, desired_token_count=desired_token_count, ws_token_count=ws_status.get("tokens"), websocket_enabled=ctx.websocket_enabled, phase="spot_first",
            )
            if callable(subscribe_fn):
                subscribed = bool(
                    subscribe_fn(
                        policy.nifty_spot_token,
                        symbol=policy.nifty_internal_symbol,
                    )
                )
            else:
                subscribed = False
            desired_count_fn = getattr(
                ctx.market_data_manager, "desired_token_count", None
            )
            desired_tokens = (
                int(desired_count_fn()) if callable(desired_count_fn) else None
            )
            ws_token_count_fn = getattr(ctx.market_data_manager, "ws_token_count", None)
            ws_tokens = None
            if callable(ws_token_count_fn):
                ws_count = ws_token_count_fn()
                ws_tokens = int(ws_count) if ws_count is not None else None
            _safe_startup_log(
                LOGGER, logging.INFO, "STARTUP_SPOT_TOKEN_REQUEST_DONE",
                "STARTUP_SPOT_TOKEN_REQUEST_DONE symbol=%s token=%d subscribed=%s desired_token_count=%s ws_token_count=%s websocket_enabled=%s phase=spot_first",
                policy.nifty_internal_symbol,
                policy.nifty_spot_token,
                subscribed,
                desired_tokens,
                ws_tokens,
                ctx.websocket_enabled,
                symbol=policy.nifty_internal_symbol, token=policy.nifty_spot_token, subscribed=subscribed, desired_token_count=desired_tokens, ws_token_count=ws_tokens, websocket_enabled=ctx.websocket_enabled, phase="spot_first",
            )
            _safe_startup_log(
                LOGGER, logging.INFO, "STARTUP_WS_START_BEGIN",
                "STARTUP_WS_START_BEGIN symbol=%s token=%d desired_token_count=%s ws_token_count=%s websocket_enabled=%s phase=spot_first",
                policy.nifty_internal_symbol,
                policy.nifty_spot_token,
                desired_tokens,
                ws_tokens,
                ctx.websocket_enabled,
                symbol=policy.nifty_internal_symbol, token=policy.nifty_spot_token, desired_token_count=desired_tokens, ws_token_count=ws_tokens, websocket_enabled=ctx.websocket_enabled, phase="spot_first",
            )
            ctx.market_data_manager.start_websocket()
            _safe_startup_log(
                LOGGER, logging.INFO, "STARTUP_WS_START_DONE",
                "STARTUP_WS_START_DONE symbol=%s token=%d desired_token_count=%s ws_token_count=%s websocket_enabled=%s phase=spot_first",
                policy.nifty_internal_symbol,
                policy.nifty_spot_token,
                desired_tokens,
                ws_tokens,
                ctx.websocket_enabled,
                symbol=policy.nifty_internal_symbol, token=policy.nifty_spot_token, desired_token_count=desired_tokens, ws_token_count=ws_tokens, websocket_enabled=ctx.websocket_enabled, phase="spot_first",
            )
            try:
                ws_status = ws_status_fn() if callable(ws_status_fn) else {}
            except Exception as exc:
                ws_status = {}
                LOGGER.warning(
                    "STARTUP_WS_STATUS_SNAPSHOT_FAILED err=%s",
                    exc,
                    extra={
                        "event": "STARTUP_WS_STATUS_SNAPSHOT_FAILED",
                        "error": str(exc),
                    },
                    exc_info=True,
                )
            _safe_startup_log(
                LOGGER, logging.INFO, "STARTUP_WS_CONNECT_TASK_CREATED",
                "STARTUP_WS_CONNECT_TASK_CREATED symbol=%s token=%d desired_token_count=%s ws_token_count=%s websocket_enabled=%s phase=spot_first",
                policy.nifty_internal_symbol,
                policy.nifty_spot_token,
                desired_tokens,
                ws_status.get("tokens"),
                ctx.websocket_enabled,
                symbol=policy.nifty_internal_symbol, token=policy.nifty_spot_token, desired_token_count=desired_tokens, ws_token_count=ws_status.get("tokens"), websocket_enabled=ctx.websocket_enabled, phase="spot_first",
            )
            spot_wait_seconds = float(os.getenv("STARTUP_SPOT_WS_PROOF_TIMEOUT_SEC", os.getenv("STARTUP_WAIT_FOR_WS_SPOT_SECONDS", "10")))
            mode = str(getattr(ctx, "effective_mode", None) or os.getenv("EXECUTION_MODE", "PAPER")).upper()
            live_mode = mode == "LIVE" or str(os.getenv("ENABLE_LIVE", "false")).lower() in {"1", "true", "yes", "on"}
            try:
                spot_price = await _wait_for_ws_spot_proof(
                    ctx, timeout=spot_wait_seconds
                )
                if spot_price is not None:
                    LOGGER.info(
                        "STARTUP_WS_SPOT_PROOF_READY symbol=%s price=%.2f mode=%s source=ws",
                        policy.nifty_internal_symbol,
                        spot_price,
                        mode,
                        extra={
                            "event": "STARTUP_WS_SPOT_PROOF_READY",
                            "symbol": policy.nifty_internal_symbol,
                            "price": spot_price,
                            "mode": mode,
                            "source": "ws",
                        },
                    )
                    await _refresh_readiness_after_first_tick(
                        ctx, reason="ws_spot_proof_ready"
                    )
                elif live_mode and get_market_state() == MarketState.OPEN:
                    ctx.live_orders_armed = False
                    ctx.trading_ready = False
                    ctx.live_block_reason = "spot_ltp_unavailable"
                    LOGGER.warning(
                        "STARTUP_SPOT_PROOF_TIMEOUT symbol=%s timeout=%.2f reason=ws_tick_missing",
                        policy.nifty_internal_symbol,
                        spot_wait_seconds,
                        extra={"event": "STARTUP_SPOT_PROOF_TIMEOUT", "symbol": policy.nifty_internal_symbol, "timeout": spot_wait_seconds, "reason": "ws_tick_missing"},
                    )
                    rest_spot = _resolve_startup_rest_spot_ltp(
                        ctx,
                        max_age_seconds=policy.startup_spot_max_age_seconds,
                    )
                    if rest_spot and rest_spot > 0:
                        ctx.live_block_reason = "live_option_quote_required_after_rest_spot_fallback"
                        LOGGER.info(
                            "STARTUP_SPOT_REST_FALLBACK_USED symbol=%s price=%.2f live_orders_armed=%s",
                            policy.nifty_internal_symbol,
                            float(rest_spot),
                            bool(getattr(ctx, "live_orders_armed", False)),
                            extra={"event": "STARTUP_SPOT_REST_FALLBACK_USED", "symbol": policy.nifty_internal_symbol, "price": float(rest_spot), "live_orders_armed": bool(getattr(ctx, "live_orders_armed", False))},
                        )
                        await _build_and_hydrate_live_basket_from_spot(
                            ctx,
                            spot_ltp=float(rest_spot),
                            configured_mode=mode,
                            hydrate=True,
                        )
                    else:
                        _schedule_deferred_basket_retry(
                            ctx,
                            configured_mode=mode,
                            reason="spot_ltp_unavailable",
                            spot_ltp=None,
                        )
                elif live_mode:
                    ctx.live_orders_armed = False
                    ctx.trading_ready = False
                    ctx.readiness_mode = "DATA_WARMUP"
                    ctx.effective_mode = "DATA_WARMUP"
                    ctx.live_block_reason = "outside_market_waiting_for_spot_tick"
                    LOGGER.warning(
                        "STARTUP_WS_SPOT_PROOF_TIMEOUT_WARMUP symbol=%s mode=%s",
                        policy.nifty_internal_symbol,
                        mode,
                        extra={"event": "STARTUP_WS_SPOT_PROOF_TIMEOUT_WARMUP"},
                    )
                else:
                    LOGGER.warning(
                        "STARTUP_WS_SPOT_PROOF_TIMEOUT_NONLIVE symbol=%s mode=%s",
                        policy.nifty_internal_symbol,
                        mode,
                        extra={
                            "event": "STARTUP_WS_SPOT_PROOF_TIMEOUT_NONLIVE",
                            "symbol": policy.nifty_internal_symbol,
                            "mode": mode,
                        },
                    )
            except RuntimeError as exc:
                ctx.live_orders_armed = False
                ctx.trading_ready = False
                ctx.live_block_reason = str(exc) or "spot_ltp_unavailable"
                LOGGER.error("STARTUP_WS_SPOT_PROOF_BLOCKED_LIVE symbol=%s reason=%s", policy.nifty_internal_symbol, exc, extra={"event": "STARTUP_WS_SPOT_PROOF_BLOCKED_LIVE", "symbol": policy.nifty_internal_symbol, "reason": str(exc)})
                _schedule_deferred_basket_retry(
                    ctx,
                    configured_mode=mode,
                    reason=str(exc) or "spot_ltp_unavailable",
                    spot_ltp=None,
                )
        except Exception as _ws_spot_first_exc:  # noqa: BLE001
            is_live_configured = str(os.getenv("EXECUTION_MODE", "PAPER")).strip().upper() == "LIVE" or str(os.getenv("ENABLE_LIVE", "false")).strip().lower() in {"1", "true", "yes", "on"}
            if is_live_configured and get_market_state() == MarketState.OPEN:
                ctx.live_orders_armed = False
                ctx.trading_ready = False
                ctx.live_block_reason = "ws_spot_first_failed"
                LOGGER.error(
                    "STARTUP_WS_SPOT_FIRST_FAILED_LIVE_BLOCKED symbol=%s token=%d websocket_enabled=%s err=%s",
                    policy.nifty_internal_symbol,
                    policy.nifty_spot_token,
                    ctx.websocket_enabled,
                    _ws_spot_first_exc,
                    exc_info=True,
                    extra={"event": "STARTUP_WS_SPOT_FIRST_FAILED_LIVE_BLOCKED"},
                )
                raise
            if is_live_configured:
                LOGGER.warning(
                    "STARTUP_WS_SPOT_FIRST_FAILED_WARMUP symbol=%s token=%d websocket_enabled=%s err=%s",
                    policy.nifty_internal_symbol,
                    policy.nifty_spot_token,
                    ctx.websocket_enabled,
                    _ws_spot_first_exc,
                    extra={"event": "STARTUP_WS_SPOT_FIRST_FAILED_WARMUP"},
                )
                ctx.live_orders_armed = False
                ctx.trading_ready = False
                ctx.readiness_mode = "DATA_WARMUP"
                ctx.effective_mode = "DATA_WARMUP"
                ctx.live_block_reason = "outside_market_waiting_for_spot_tick"
            LOGGER.warning(
                "STARTUP_WS_SPOT_FIRST_FAILED symbol=%s token=%d websocket_enabled=%s err=%s",
                policy.nifty_internal_symbol,
                policy.nifty_spot_token,
                ctx.websocket_enabled,
                _ws_spot_first_exc,
                extra={"event": "STARTUP_WS_SPOT_FIRST_FAILED"},
            )

    # =========================================================
    # Create Data Directory
    # =========================================================
    try:
        get_data_dir()
    except Exception as e:
        critical_errors_total.labels(component="startup", error_type=type(e).__name__).inc()
        LOGGER.critical(f"❌ Startup sequence failed: {e}", exc_info=True)
        raise

    _validate_config(ctx.config)
    instrument_cache_ready.clear()
    broker_ready = True
    guard = ctx.session_guard

    # ---------------------------------------------------------
    # Telegram notifier helper
    # ---------------------------------------------------------
    async def _notify(event: str, payload: Mapping[str, object] | None = None) -> None:
        notifier = ctx.telegram_notifier
        if notifier is None:
            return
        try:
            await notifier.send_event(event, payload)
        except Exception:
            LOGGER.debug("Startup notifier failed", exc_info=True)

    # ---------------------------------------------------------
    # 1. Broker validation
    # ---------------------------------------------------------
    try:
        broker_proxy = getattr(
            ctx.broker_client,
            "_broker",
            getattr(ctx.broker_client, "broker", ctx.broker_client),
        )
        get_profile_fn = getattr(broker_proxy, "get_profile", None)
        if callable(get_profile_fn):
            profile = await asyncio.to_thread(get_profile_fn)
            LOGGER.info(f"Connected to broker: {profile.get('user_name') or 'User'}")
            if guard:
                guard.mark_session_valid()
            try:
                await _reconcile_state(ctx)
                LOGGER.info("startup_position_reconciliation_complete")
            except Exception as reconcile_exc:
                LOGGER.error(
                    "Failure in startup position reconciliation: %s", reconcile_exc
                )
    except Exception as e:
        LOGGER.error(f"Broker connection failed: {e}")
        broker_ready = False

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 2. Pre-load broker instrument caches (NSE + NFO)
    # InstrumentManager.load() in section 2b fetches directly from broker.
    # ---------------------------------------------------------
    startup_trade_ready = False
    polling_fallback = ctx.polling_fallback_streamer

    if broker_ready:
        try:
            inner = getattr(
                ctx.broker_client,
                "broker",
                getattr(ctx.broker_client, "_broker", ctx.broker_client),
            )
            LOGGER.info("📦 Loading NSE instruments...")
            await asyncio.to_thread(inner.load_instruments, "NSE")
            LOGGER.info("✅ NSE instruments loaded")

            LOGGER.info("📦 Loading NFO instruments...")
            await asyncio.to_thread(inner.load_instruments, "NFO")
            LOGGER.info("✅ NFO instruments loaded")
            instrument_cache_ready.set()

        except Exception as e:
            LOGGER.error(f"Instrument load failed: {e}", exc_info=True)

    # ---------------------------------------------------------
    # 2b. InstrumentManager — token-first single source of truth
    # ---------------------------------------------------------
    # Load after broker auth is confirmed (broker_ready) and NFO instruments
    # have been synced into the resolver so the same API round-trip is reused.
    if broker_ready and ctx.instrument_manager is not None:
        try:
            inner_broker = getattr(ctx.broker_client, "_broker",
                                   getattr(ctx.broker_client, "broker",
                                           ctx.broker_client))
            ctx.instrument_manager._kite = inner_broker
            ctx.instrument_manager.load()
            _im_size = ctx.instrument_manager.size()
            # ── STARTUP GUARD ────────────────────────────────────────────────
            assert _im_size > 0, (
                "InstrumentManager.load() completed but zero NIFTY instruments "
                "were found. Check broker authentication and NFO instrument dump."
            )
            LOGGER.info(
                "✅ InstrumentManager loaded %d NIFTY instruments",
                _im_size,
                extra={"event": "instrument_manager_ready", "count": _im_size},
            )
            
            # Initialize OptionsContractStore as single source of truth for options
            ctx.options_store = OptionsContractStore(ctx.instrument_manager)
            ctx.options_store.load()
            LOGGER.info(
                "✅ OptionsContractStore initialized with %d contracts",
                ctx.options_store.contract_count(),
                extra={"event": "options_store_ready", "count": ctx.options_store.contract_count()},
            )
        except AssertionError as _guard_err:
            LOGGER.critical("❌ STARTUP GUARD: %s", _guard_err)
            LOGGER.warning("⚠️ Continuing in degraded mode — no NIFTY instruments")
        except Exception as _im_exc:
            LOGGER.warning(
                "InstrumentManager.load() failed: %s",
                _im_exc,
                exc_info=True,
            )

    # ---------------------------------------------------------
    # 3. Symbol resolution + HYDRATION (FIXED)
    # ---------------------------------------------------------
    if broker_ready:
        try:
            targets = _get_symbols(
                ctx.config,
                ctx.instrument_manager,
                ctx.broker_client,
                option_universe=ctx.option_universe,
                market_data_manager=ctx.market_data_manager,
                active_contract_basket=getattr(ctx, "active_contract_basket", None) or getattr(ctx, "active_trading_universe", None),
            )

            active_futures_symbol = _resolve_active_futures_for_basket(ctx, None)
            if active_futures_symbol:
                try:
                    fut_token = ctx.instrument_manager.get_token(active_futures_symbol) if ctx.instrument_manager and ctx.instrument_manager.is_loaded() else None
                except Exception:
                    fut_token = None
                if fut_token:
                    LOGGER.info(
                        "ACTIVE_FUTURES_TARGET_RESOLVED symbol=%s token=%s",
                        active_futures_symbol,
                        fut_token,
                        extra={"event": "ACTIVE_FUTURES_TARGET_RESOLVED", "symbol": active_futures_symbol, "token": int(fut_token)},
                    )
                    targets.append(active_futures_symbol)
                    purge_stale = getattr(ctx.market_data_manager, "purge_stale_nifty_futures", None)
                    if callable(purge_stale):
                        purge_stale(active_futures_symbol, reason="startup_active_future_resolution")
                    orchestrator = getattr(getattr(ctx, "strategy_manager", None), "orchestrator", None)
                    if orchestrator is not None:
                        orchestrator.futures_symbol = active_futures_symbol
                else:
                    LOGGER.warning(
                        "FUTURES_CONTEXT_UNAVAILABLE symbol=%s reason=startup_active_future_token_missing",
                        active_futures_symbol,
                        extra={"event": "FUTURES_CONTEXT_UNAVAILABLE", "symbol": active_futures_symbol, "reason": "startup_active_future_token_missing"},
                    )
            else:
                LOGGER.warning(
                    "FUTURES_CONTEXT_UNAVAILABLE reason=startup_active_future_unresolved",
                    extra={"event": "FUTURES_CONTEXT_UNAVAILABLE", "reason": "startup_active_future_unresolved"},
                )

            if "NSE:NIFTY" not in targets:
                targets.append("NSE:NIFTY")
            targets = list(dict.fromkeys(targets))

            # Canonical live basket for trading path: spot + futures + ATM±3 CE/PE.
            # Resolve spot via the WS-first helper.  In LIVE mode this raises
            # when no fresh WebSocket tick proof exists so we never select an
            # ATM basket from a synthetic/stale price.
            try:
                spot_ltp = await _wait_for_live_spot_or_raise(
                    ctx,
                    timeout=float(policy.startup_wait_for_ws_spot_seconds or 60.0),
                    configured_mode=configured_mode,
                )
            except RuntimeError as _basket_spot_exc:
                market_state = get_market_state()
                if market_state == MarketState.OPEN:
                    LOGGER.error(
                        "LIVE_STARTUP_SPOT_UNAVAILABLE reason=%s phase=basket",
                        _basket_spot_exc,
                        extra={
                            "event": "LIVE_STARTUP_SPOT_UNAVAILABLE",
                            "reason": str(_basket_spot_exc),
                            "phase": "basket",
                        },
                    )
                    LOGGER.error(
                        "STARTUP_NO_FAKE_SPOT_LIVE_MODE reason=%s",
                        "fresh_spot_tick_unavailable",
                        extra={
                            "event": "STARTUP_NO_FAKE_SPOT_LIVE_MODE",
                            "reason": "fresh_spot_tick_unavailable",
                        },
                    )
                else:
                    defer_reason = str(_basket_spot_exc)
                    LOGGER.info("OUT_OF_HOURS_DATA_MODE option_hydration_nonfatal=true trading_ready=false")
                    LOGGER.info(
                        "DEFERRED_BASKET_RETRY_SCHEDULED reason=%s spot_ltp=%s",
                        defer_reason,
                        None,
                        extra={
                            "event": "BASKET_BUILD_DEFERRED",
                            "reason": defer_reason,
                            "market_state": market_state.value,
                        },
                    )
                _schedule_deferred_basket_retry(
                    ctx,
                    configured_mode=configured_mode,
                )
                raise RuntimeError("startup_basket_deferred")
            basket = await _build_and_hydrate_live_basket_from_spot(
                ctx,
                spot_ltp=float(spot_ltp),
                configured_mode=configured_mode,
                hydrate=False,
            )
            targets = list(dict.fromkeys(basket["symbols"]))

            LOGGER.info(f"⏳ Hydrating {len(targets)} symbols: {targets}")

            # ---------- HISTORICAL HYDRATION (SEQUENTIAL FETCH + SERIAL COMMIT) ----------
            end_dt = datetime.now()
            # ✅ FIX: Use 3-day lookback so same-day Zerodha API gaps (which return
            # 0 bars for the current session) don't cause hydration_zero_bars.
            # Weekly options may have no same-day candles early in the week — a
            # multi-day window guarantees at least 50 bars from recent sessions.
            hydration_lookback_days = max(1, int(os.getenv("HYDRATION_LOOKBACK_DAYS", "2") or 2))
            hydration_min_bars = max(1, int(os.getenv("HYDRATION_MIN_BARS", "100") or 100))
            hydration_max_bars = max(hydration_min_bars, int(os.getenv("HYDRATION_MAX_BARS", "300") or 300))
            hydration_max_contracts = max(1, int(os.getenv("HYDRATION_MAX_CONTRACTS", "12") or 12))
            start_dt = end_dt - timedelta(days=hydration_lookback_days)
            from_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            to_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

            hydrated_counts: dict[str, int] = {}
            runner = ctx.strategy_runner
            
            # ✅ FIX: Combine Spot and final options for multi-symbol hydration
            # Validate tokens before hydration — skip symbols with no resolvable
            # token so we don't waste API quota on unresolvable contracts.
            def _is_nfo_symbol(symbol: str) -> bool:
                """Return True when symbol belongs to NFO. Args: symbol. Returns: bool. Raises: None."""
                return str(symbol or "").strip().upper().startswith("NFO:")

            def _resolve_startup_token(symbol: str) -> int | None:
                """Resolve startup token by symbol class. Args: symbol. Returns: token or None. Raises: None."""
                canonical = str(symbol or "").strip().upper()
                well_known_spot_tokens = {
                    "NSE:NIFTY": 256265,
                    "NIFTY": 256265,
                    "NSE:NIFTY50": 256265,
                    "NIFTY50": 256265,
                }
                if canonical in well_known_spot_tokens:
                    return int(well_known_spot_tokens[canonical])
                if _is_nfo_symbol(symbol):
                    if ctx.instrument_manager and ctx.instrument_manager.is_loaded():
                        try:
                            return int(ctx.instrument_manager.get_token(symbol))
                        except RuntimeError:
                            return None
                        except Exception as exc:
                            LOGGER.warning(
                                "startup_nfo_token_unresolved symbol=%s err=%s",
                                symbol,
                                exc,
                                extra={"event": "startup_nfo_token_unresolved", "symbol": symbol},
                            )
                            return None
                    return None
                if ctx.broker_client and hasattr(ctx.broker_client, "get_instrument_token"):
                    try:
                        return int(ctx.broker_client.get_instrument_token(symbol))
                    except Exception as exc:  # noqa: BLE001 - resolver miss is handled by caller
                        LOGGER.warning(
                            "startup_spot_token_unresolved symbol=%s err=%s",
                            symbol,
                            exc,
                            extra={
                                "event": "startup_spot_token_unresolved",
                                "symbol": symbol,
                            },
                        )
                        return None
                return None

            active_symbol_tokens: dict[str, int] = {}
            symbols_to_hydrate_raw = build_active_trading_basket_symbols(ctx, basket)
            selected_symbols = [
                basket.get("selected_ce") or basket.get("atm_ce"),
                basket.get("selected_pe") or basket.get("atm_pe"),
            ]
            other_option_symbols = [
                sym
                for sym in symbols_to_hydrate_raw
                if sym not in {"NSE:NIFTY", basket.get("futures_symbol"), *selected_symbols}
            ]
            symbols_to_hydrate_raw = list(
                dict.fromkeys(
                    [
                        "NSE:NIFTY",
                        active_futures_symbol,
                        *selected_symbols,
                        *other_option_symbols,
                    ]
                )
            )
            symbols_to_hydrate: list[str] = []
            for _sym in symbols_to_hydrate_raw:
                _tok = _resolve_startup_token(_sym)
                if _tok is None:
                    LOGGER.warning(
                        "Skipping hydration for %s (no token)",
                        _sym,
                        extra={"event": "hydration_skip_no_token", "symbol": _sym},
                    )
                    continue
                active_symbol_tokens[_sym] = int(_tok)
                LOGGER.info("ACTIVE_BASKET_TOKEN_RESOLVED symbol=%s token=%s", _sym, int(_tok))
                symbols_to_hydrate.append(_sym)
            atm_ce = str(basket.get("selected_ce") or basket.get("atm_ce") or "")
            atm_pe = str(basket.get("selected_pe") or basket.get("atm_pe") or "")
            if atm_ce and atm_ce not in active_symbol_tokens:
                LOGGER.error("ACTIVE_BASKET_TOKEN_MISSING symbol=%s fatal_for_live=True", atm_ce)
            if atm_pe and atm_pe not in active_symbol_tokens:
                LOGGER.error("ACTIVE_BASKET_TOKEN_MISSING symbol=%s fatal_for_live=True", atm_pe)
            LOGGER.info(
                "ACTIVE_BASKET_TOKEN_MAP_READY count=%d selected_ce_token=%s selected_pe_token=%s",
                len(active_symbol_tokens),
                active_symbol_tokens.get(atm_ce) if atm_ce else None,
                active_symbol_tokens.get(atm_pe) if atm_pe else None,
            )
            LOGGER.info(
                "HYDRATION_SYMBOLS_FINAL symbols=%s has_spot=%s has_futures=%s",
                symbols_to_hydrate,
                "NSE:NIFTY" in symbols_to_hydrate,
                any(str(s).startswith("NFO:NIFTY") and str(s).endswith("FUT") for s in symbols_to_hydrate),
                extra={
                    "event": "HYDRATION_SYMBOLS_FINAL",
                    "symbols": symbols_to_hydrate,
                    "has_spot": "NSE:NIFTY" in symbols_to_hydrate,
                    "has_futures": any(str(s).startswith("NFO:NIFTY") and str(s).endswith("FUT") for s in symbols_to_hydrate),
                },
            )
            LOGGER.info(
                "HYDRATION_PLAN_STARTUP symbols=%d bars_target=%d lookback_days=%d",
                len(symbols_to_hydrate),
                hydration_max_bars,
                hydration_lookback_days,
                extra={
                    "event": "HYDRATION_PLAN",
                    "symbols": len(symbols_to_hydrate),
                    "bars_target": hydration_max_bars,
                    "lookback_days": hydration_lookback_days,
                },
            )

            LOGGER.info(
                "Starting multi-symbol hydration for %d/%d instruments (after token validation)...",
                len(symbols_to_hydrate),
                len(symbols_to_hydrate_raw),
            )

            async def _fetch_symbol(symbol: str) -> tuple[str, list[Any]]:
                """Fetch historical records for one symbol. Args: symbol. Returns: symbol and raw records. Raises: Exception."""
                records = await _maybe_await(
                    getattr(ctx.market_data_manager, "hydrate_symbol_history", ctx.market_data_manager.fetch_history)(
                        symbol,
                        interval="minute",
                        days=hydration_lookback_days,
                        max_bars=hydration_max_bars,
                        reason="startup",
                    )
                )
                if not hasattr(ctx.market_data_manager, "hydrate_symbol_history"):
                    ctx.market_data_manager.ingest_historical_ohlc(symbol, records or [])
                    if records and len(records) > hydration_max_bars:
                        records = list(records)[-hydration_max_bars:]
                record_count = len(records or [])
                if record_count == 0:
                    LOGGER.error(
                        "hydration_zero_bars: %s returned 0 bars — "
                        "Zerodha historical API returned empty data. "
                        "Extending lookback or check broker auth.",
                        symbol,
                        extra={"event": "hydration_zero_bars", "symbol": symbol},
                    )
                elif record_count < hydration_min_bars:
                    LOGGER.info(
                        "insufficient_bars_for_strategy: %s returned only %d bars (need ≥%d) — "
                        "strategy indicators will be unreliable",
                        symbol,
                        record_count,
                        hydration_min_bars,
                        extra={
                            "event": "insufficient_bars_for_strategy",
                            "symbol": symbol,
                            "bars": record_count,
                        },
                    )
                LOGGER.info(
                    "hydration_fetch_complete",
                    extra={
                        "event": "hydration_fetch_complete",
                        "symbol": symbol,
                        "records": record_count,
                    },
                )
                return symbol, list(records or [])

            fetch_results: list[tuple[str, list[Any]] | Exception] = []
            for target_symbol in symbols_to_hydrate:
                try:
                    fetch_results.append(await _fetch_symbol(target_symbol))
                except Exception as exc:  # noqa: BLE001
                    fetch_results.append(exc)
                await asyncio.sleep(1.15)

            LOGGER.info(
                "hydration_commit_start",
                extra={
                    "event": "hydration_commit_start",
                    "symbols": len(targets),
                },
            )
            for result in fetch_results:
                if isinstance(result, Exception):
                    LOGGER.warning(f"❌ Failed to fetch hydration symbol: {result}")
                    continue
                sym, records = result
                sym_token = None
                try:
                    sym_token = active_symbol_tokens.get(sym)
                except Exception:
                    sym_token = None
                canonical_sym = canonical(sym)
                mdm_bars = list(ctx.market_data_manager.get_ohlc_bars(canonical_sym, limit=hydration_max_bars) or [])
                count = len(mdm_bars)
                runner_ingested = 0
                runner_history_count = 0
                if getattr(ctx, "strategy_runner", None) is not None:
                    required_bars = int(getattr(ctx.strategy_runner, "_context_required_bars", 0) or getattr(ctx.strategy_runner, "_required_candles", 0) or 50)
                    try:
                        ctx.strategy_runner._indicator_engine.replace_history(canonical_sym, mdm_bars, source="startup_hydration", min_bars=required_bars)
                        runner_ingested = len(mdm_bars)
                        runner_history_count = ctx.strategy_runner._indicator_engine.history_count(canonical_sym)
                    except Exception as runner_ingest_exc:
                        LOGGER.warning("startup_runner_hydration_replace_failed symbol=%s err=%s", canonical_sym, runner_ingest_exc)
                LOGGER.info("RUNNER_MDM_HYDRATION_SYNC symbol=%s mdm_bars=%d runner_ingested=%d", canonical_sym, count, runner_ingested,extra={"event":"RUNNER_MDM_HYDRATION_SYNC","symbol":canonical_sym,"mdm_bars":count,"runner_ingested":runner_ingested})
                hydrated_counts[canonical_sym] = count
                LOGGER.info(f"✅ Hydrated {sym}: {count} bars")
                if getattr(ctx, "strategy_runner", None) is not None:
                    mdm_history_count = 0
                    if ctx.market_data_manager is not None:
                        mdm_history_count = len(
                            ctx.market_data_manager.get_ohlc_bars(canonical_sym) or []
                        )
                    LOGGER.info(
                        "RUNNER_HISTORY_CONSISTENCY symbol=%s mdm_bars=%d indicator_bars=%d consistent=%s",
                        canonical_sym,
                        mdm_history_count,
                        runner_history_count,
                        bool(runner_history_count >= required_bars and mdm_history_count >= required_bars),
                        extra={"event": "RUNNER_HISTORY_CONSISTENCY", "symbol": canonical_sym, "mdm_bars": mdm_history_count, "indicator_bars": runner_history_count, "consistent": bool(runner_history_count >= required_bars and mdm_history_count >= required_bars)},
                    )
                    LOGGER.info(
                        "RUNNER_HISTORY_INGESTED symbol=%s token=%s bars_ingested=%d source=%s runner_history_count=%d mdm_history_count=%d",
                        canonical_sym,
                        sym_token,
                        runner_ingested,
                        "startup_hydration",
                        runner_history_count,
                        mdm_history_count,
                        extra={
                            "event": "RUNNER_HISTORY_INGESTED",
                            "symbol": canonical_sym,
                            "token": sym_token,
                            "bars_ingested": runner_ingested,
                            "source": "startup_hydration",
                            "runner_history_count": runner_history_count,
                            "mdm_history_count": mdm_history_count,
                        },
                    )
                    if canonical_sym == "NSE:NIFTY":
                        required = int(
                            getattr(ctx.strategy_runner, "_context_required_bars", 0)
                            or getattr(ctx.strategy_runner, "_required_candles", 0)
                            or 50
                        )
                        if mdm_history_count >= required and runner_history_count < required:
                            runner_before = runner_history_count
                            ctx.strategy_runner._indicator_engine.replace_history(canonical_sym, mdm_bars, source="startup_hydration", min_bars=required)
                            runner_history_count = ctx.strategy_runner._indicator_engine.history_count(canonical_sym)
                            LOGGER.info(
                                "SPOT_CONTEXT_HISTORY_RESEEDED symbol=NSE:NIFTY mdm_bars=%d runner_before=%d runner_after=%d success=%s",
                                mdm_history_count,
                                runner_before,
                                runner_history_count,
                                bool(runner_history_count >= required),
                            )
                        success = runner_history_count >= required and mdm_history_count >= required
                        reason = "hydrated" if success else "insufficient_bars_after_hydration"
                        LOGGER.info(
                            "SPOT_CONTEXT_HYDRATION_RESULT symbol=NSE:NIFTY mdm_bars=%d runner_bars=%d required=%d success=%s reason=%s",
                            mdm_history_count,
                            runner_history_count,
                            required,
                            success,
                            reason,
                            extra={
                                "event": "SPOT_CONTEXT_HYDRATION_RESULT",
                                "symbol": "NSE:NIFTY",
                                "mdm_bars": mdm_history_count,
                                "runner_bars": runner_history_count,
                                "required": required,
                                "success": success,
                                "reason": reason,
                            },
                        )
                if ctx.market_data_manager:
                    bars_snapshot = ctx.market_data_manager.get_ohlc_bars(canonical_sym)
                    ctx.market_data_manager.update_hydration_status(
                        canonical_sym, bars_snapshot or records
                    )

            LOGGER.info(
                "hydration_commit_done",
                extra={
                    "event": "hydration_commit_done",
                    "symbols": len(hydrated_counts),
                    "hydrated_counts": hydrated_counts,
                },
            )
            if ctx.settings.enable_live and ctx.market_data_manager is not None:
                failed = [
                    s
                    for s, status in ctx.market_data_manager._hydration_status.items()
                    if str(status).strip().lower() not in {"hydrated", "ready"}
                ]
                if failed:
                    # WARN only — do NOT raise. RuntimeError here aborts before mark_ready()
                    # is called, leaving runner stuck in HISTORICAL_READY and blocking all trades.
                    LOGGER.warning(
                        "startup_hydration_incomplete_symbols=%s — proceeding to mark_ready",
                        failed,
                        extra={
                            "event": "startup_hydration_incomplete_warn",
                            "failed_symbols": failed,
                        },
                    )

            basket = normalize_active_basket_schema(basket)
            active_futures_symbol = str(basket.get("futures_symbol") or "")
            active_option_symbols = select_active_option_symbols(
                option_symbols=basket.get("option_symbols", []) or [],
                atm=basket.get("atm_strike"),
                max_active=int(os.getenv("MAX_ACTIVE_OPTION_SYMBOLS", "6")),
            )
            readiness_symbols = list(dict.fromkeys([
                basket.get("spot_symbol"),
                active_futures_symbol,
                *active_option_symbols,
            ]))
            readiness_symbols = [str(sym) for sym in readiness_symbols if sym]
            LOGGER.info(
                "READINESS_SYMBOLS_SELECTED count=%d spot=%s futures=%s option_count=%d symbols=%s",
                len(readiness_symbols),
                basket.get("spot_symbol"),
                active_futures_symbol,
                len(basket.get("option_symbols", []) or []),
                readiness_symbols,
                extra={
                    "event": "READINESS_SYMBOLS_SELECTED",
                    "count": len(readiness_symbols),
                    "spot": basket.get("spot_symbol"),
                    "futures": active_futures_symbol,
                    "option_count": len(basket.get("option_symbols", []) or []),
                    "symbols": readiness_symbols,
                },
            )
            ready_symbols: list[str] = []
            default_required_bars = int(
                getattr(runner, "_required_candles", 20) if runner else 20
            )
            option_min_live_bars = int(os.getenv("OPTION_MIN_LIVE_BARS", "3") or 3)
            skipped_symbols: list[str] = []
            skipped_reasons: dict[str, str] = {}
            for sym in readiness_symbols:
                runner_history_count = 0
                mdm_ohlc_count = 0
                if runner is not None:
                    try:
                        runner_history_count = len(
                            runner._indicator_engine.get_history(sym) or []
                        )
                    except Exception:
                        runner_history_count = 0
                if ctx.market_data_manager is not None:
                    mdm_ohlc_count = len(ctx.market_data_manager.get_ohlc_bars(sym) or [])
                if (
                    runner is not None
                    and runner_history_count <= 0
                    and mdm_ohlc_count > 0
                    and hasattr(runner, "ingest_historical_bar")
                    and ctx.market_data_manager is not None
                ):
                    for _bar in ctx.market_data_manager.get_ohlc_bars(sym) or []:
                        if isinstance(_bar, Mapping):
                            runner.ingest_historical_bar(dict(_bar))
                    try:
                        runner_history_count = len(
                            runner._indicator_engine.get_history(sym) or []
                        )
                    except Exception:
                        runner_history_count = 0
                min_required_bars = (
                    option_min_live_bars
                    if str(sym).startswith("NFO:") and (str(sym).endswith("CE") or str(sym).endswith("PE"))
                    else default_required_bars
                )
                effective_bar_count = max(runner_history_count, mdm_ohlc_count)
                if effective_bar_count >= min_required_bars:
                    ready_symbols.append(sym)
                else:
                    skipped_symbols.append(sym)
                    skipped_reasons[sym] = (
                        f"effective_history_below_required:{effective_bar_count}/{min_required_bars}"
                    )
                    LOGGER.info(
                        "Condition met: startup_hydration_incomplete",
                        extra={
                            "event": "startup_hydration_incomplete",
                            "symbol": sym,
                            "bars": runner_history_count,
                            "required": min_required_bars,
                        },
                    )
            pending_runner_symbols: set[str] = set()
            for sym in skipped_symbols:
                pending_runner_symbols.add(sym)
            if runner is not None:
                for sym in ready_symbols:
                    try:
                        runner.add_symbol(sym)
                    except Exception as e:
                        LOGGER.error(
                            "RUNNER_ADD_SYMBOL_FROM_READINESS_FAILED symbol=%s error=%s",
                            sym,
                            e,
                            extra={
                                "event": "RUNNER_ADD_SYMBOL_FROM_READINESS_FAILED",
                                "symbol": sym,
                                "error": str(e),
                            },
                        )
            registered_symbol_count = (
                len(getattr(runner, "_active_symbols", set()) or []) if runner is not None else 0
            )
            LOGGER.info(
                "RUNNER_SYMBOLS_REGISTERED count=%d",
                registered_symbol_count,
                extra={"event": "RUNNER_SYMBOLS_REGISTERED", "count": registered_symbol_count},
            )
            if runner is not None and hasattr(runner, "mark_ready") and ready_symbols:
                runner.mark_ready(ready_symbols)
                LOGGER.info(
                    "RUNNER_READY_MARKED symbol_count=%d initial_ready_symbols=%s skipped_symbols=%s skipped_reasons=%s min_required_bars=%d",
                    len(readiness_symbols),
                    ready_symbols,
                    skipped_symbols,
                    skipped_reasons,
                    default_required_bars,
                    extra={
                        "event": "RUNNER_READY_MARKED",
                        "symbol_count": len(readiness_symbols),
                        "initial_ready_symbols": ready_symbols,
                        "min_required_bars": default_required_bars,
                        "skipped_symbols": skipped_symbols,
                        "skipped_reasons": skipped_reasons,
                    },
                )
            else:
                LOGGER.warning(
                    "RUNNER_READY_MARK_SKIPPED reason=%s skipped_reasons=%s",
                    "no_ready_symbols_or_method_missing",
                    skipped_reasons,
                    extra={
                        "event": "RUNNER_READY_MARK_SKIPPED",
                        "reason": "no_ready_symbols_or_method_missing",
                        "skipped_reasons": skipped_reasons,
                    },
                )
            if (
                ctx.data_hub is not None
                and hasattr(ctx.data_hub, "flush_pending_live_subscriptions")
            ):
                flushed = int(ctx.data_hub.flush_pending_live_subscriptions())
                LOGGER.info(
                    "DATAHUB_PENDING_SUBSCRIPTIONS_FLUSHED count=%d phase=startup_readiness",
                    flushed,
                    extra={
                        "event": "DATAHUB_PENDING_SUBSCRIPTIONS_FLUSHED",
                        "count": flushed,
                        "phase": "startup_readiness",
                    },
                )
            configured_runtime_mode = str(
                getattr(ctx.settings, "execution_mode", None)
                or os.getenv("EXECUTION_MODE", "PAPER")
            ).upper()
            effective_runtime_mode = str(
                getattr(ctx, "effective_mode", None) or configured_runtime_mode
            ).upper()
            evaluation_allowed = configured_runtime_mode in {"PAPER", "SHADOW", "LIVE"}
            if evaluation_allowed and readiness_symbols:
                _wire_and_start_message_bus(ctx)
                if registered_symbol_count > 0:
                    await _ensure_strategy_runner_started(
                        ctx, reason="startup_symbols_registered"
                    )
                else:
                    ctx.live_orders_armed = False
                    ctx.trading_ready = False
                    ctx.readiness_mode = "DATA_WARMUP"
                    ctx.effective_mode = "DATA_WARMUP"
                    ctx.live_block_reason = "no_hydrated_symbols_ready"
                    LOGGER.info(
                        "STRATEGY_RUNNER_START_DEFERRED reason=%s",
                        "no_hydrated_symbols_ready",
                        extra={
                            "event": "STRATEGY_RUNNER_START_DEFERRED",
                            "reason": "no_hydrated_symbols_ready",
                        },
                    )
                await _replay_latest_mdm_ticks_to_bus(ctx, reason="post_runner_start")
                await _refresh_readiness_after_first_tick(ctx, reason="post_runner_start")
                ctx.data_observation_ready = True
                if configured_runtime_mode in {"PAPER", "SHADOW"}:
                    ctx.live_orders_armed = False
                    ctx.trading_ready = False
                    ctx.live_block_reason = "paper_mode_or_startup_warmup"
                    LOGGER.info("PAPER_RUNNER_STARTED symbol_count=%d live_orders_armed=%s", len(readiness_symbols), False, extra={"event":"PAPER_RUNNER_STARTED","symbol_count":len(readiness_symbols),"live_orders_armed":False,"mode":mode})
                    if not ready_symbols:
                        LOGGER.info("PAPER_RUNNER_STARTED_OBSERVATION_ONLY reason=no_hydrated_symbols_yet", extra={"event": "PAPER_RUNNER_STARTED_OBSERVATION_ONLY", "reason": "no_hydrated_symbols_yet"})
                
            if (
                ctx.market_data_manager is not None
                and "basket" in locals()
            ):
                ce_symbols = list(basket.get("ce_symbols") or [])
                pe_symbols = list(basket.get("pe_symbols") or [])
                if not ce_symbols or not pe_symbols:
                    ce_symbols = [s for s in basket.get("option_symbols", []) if str(s).endswith("CE")]
                    pe_symbols = [s for s in basket.get("option_symbols", []) if str(s).endswith("PE")]
                atm_ce = basket.get("selected_ce") or basket.get("atm_ce") or (ce_symbols[len(ce_symbols) // 2] if ce_symbols else None)
                atm_pe = basket.get("selected_pe") or basket.get("atm_pe") or (pe_symbols[len(pe_symbols) // 2] if pe_symbols else None)
                if atm_ce and atm_pe:
                    ctx.market_data_manager.set_readiness_requirements(
                        spot_symbol=str(basket.get("spot_symbol") or "NSE:NIFTY"),
                        futures_symbol=str(basket.get("futures_symbol") or ""),
                        atm_ce_symbol=atm_ce,
                        atm_pe_symbol=atm_pe,
                        option_symbols=list(basket.get("option_symbols") or []),
                    )
                else:
                    LOGGER.info(
                        "READINESS_REQUIREMENTS_DEFERRED reason=missing_ce_pe_symbols"
                    )
            try:
                await ctx.market_regime_manager.refresh_from_indicators()
                await ctx.market_regime_manager.start()
            except Exception as _mrm_exc:
                LOGGER.warning(
                    "market_regime_manager init failed (non-fatal): %s",
                    _mrm_exc,
                    exc_info=True,
                )
            if ready_symbols:
                LOGGER.info(
                    "Indicators hydration pre-check complete; waiting for live readiness gate"
                )
            else:
                LOGGER.info(
                    "Indicators remain in warmup mode because no symbols passed hydration barrier"
                )

            # ---------- Tracking / execution wiring (UNCHANGED) ----------
            mdm = ctx.market_data_manager
            # NOTE: _data_ready() is always False at startup (no ticks yet).
            # Logging INFO here floods Railway with a false alarm on every boot.
            # Demoted to DEBUG so it only appears during deep diagnostics.
            if not _data_ready(mdm):
                LOGGER.debug("startup_tick_gate: waiting_for_live_ticks (expected at boot)")

            streamer = ctx.streamer
            tokens_to_poll = []

            # --- Token Selection via InstrumentManager ---
            im = ctx.instrument_manager
            if im and im.is_loaded():
                # Add Spot and mandatory targets
                for sym in targets:
                    tok = _resolve_startup_token(sym)
                    if tok and tok not in tokens_to_poll:
                        tokens_to_poll.append(tok)

                # Add ATM Options and Futures.  Reuse the spot LTP that the
                # WS-first basket selection already proved fresh; never let a
                # synthetic 25600.0 leak into LIVE option universe selection.
                _active_universe = getattr(ctx, "active_trading_universe", None)
                if isinstance(_active_universe, Mapping):
                    LOGGER.info(
                        "ACTIVE_UNIVERSE_REUSED",
                        extra={
                            "event": "ACTIVE_UNIVERSE_REUSED",
                            "source": _active_universe.get("source"),
                        },
                    )
                spot_price = 0.0
                if isinstance(_active_universe, Mapping):
                    try:
                        spot_price = float(_active_universe.get("spot_ltp") or 0.0)
                    except (TypeError, ValueError):
                        spot_price = 0.0
                if spot_price <= 0:
                    try:
                        spot_price = await _wait_for_live_spot_or_raise(
                            ctx,
                            timeout=5.0,
                            configured_mode=configured_mode,
                        )
                    except RuntimeError as _spot_token_exc:
                        LOGGER.error(
                            "STARTUP_NO_FAKE_SPOT_LIVE_MODE reason=%s phase=token_select",
                            _spot_token_exc,
                            extra={
                                "event": "STARTUP_NO_FAKE_SPOT_LIVE_MODE",
                                "reason": str(_spot_token_exc),
                                "phase": "token_select",
                            },
                        )
                        spot_price = 0.0

                if spot_price > 0:
                    extra_tokens = im.select_tokens_for_universe(
                        base="NIFTY",
                        spot_price=spot_price,
                        strikes_around_atm=3,
                        strike_step=ctx.settings.option_universe.strike_step
                    )
                    for t in extra_tokens:
                        if t and t not in tokens_to_poll:
                            tokens_to_poll.append(t)
                else:
                    LOGGER.warning(
                        "Skipping ATM token expansion — no trustworthy spot price",
                        extra={"event": "atm_token_expansion_skipped"},
                    )

                # Ensure NIFTY spot is also covered by polling fallback.
                try:
                    nifty_spot_token = _resolve_startup_token("NSE:NIFTY")
                    if nifty_spot_token and nifty_spot_token not in tokens_to_poll:
                        tokens_to_poll.append(nifty_spot_token)
                        LOGGER.info(
                            "polling_fallback_added_spot symbol=%s token=%s",
                            "NSE:NIFTY",
                            nifty_spot_token,
                            extra={
                                "event": "polling_fallback_added_spot",
                                "symbol": "NSE:NIFTY",
                                "token": int(nifty_spot_token),
                            },
                        )
                except Exception:
                    LOGGER.exception(
                        "failed_to_add_nifty_spot_to_polling_fallback",
                        exc_info=True,
                    )

            # --- Startup token integrity validation ---
            min_tokens = 10
            if len(tokens_to_poll) < min_tokens:
                msg = f"⚠️ WARNING: Subscribed tokens ({len(tokens_to_poll)}) < MIN_TOKEN_COUNT ({min_tokens})"
                LOGGER.warning(msg)

            LOGGER.info(
                "Market data integrity verified: tokens=%d (min=%d)",
                len(tokens_to_poll),
                min_tokens,
                extra={"event": "market_data_integrity_pass", "tokens": len(tokens_to_poll)}
            )

            LOGGER.info(f"🔧 Processing {len(targets)} symbols for wiring...")
            resolved_count = 0
            unresolved_symbols = []
            live_mode_enabled = bool(ctx.settings.enable_live)
            active_symbols: list[str] = []
            pending_runner_symbols: set[str] = set()
            market_open_now = _resolve_hydration_market_open_state(LOGGER)

            for sym in targets:
                if mdm:
                    mdm.ensure_tracking(
                        sym,
                        seed=not ctx.websocket_enabled,
                        subscribe=False,
                    )

                tok = None
                # BUG-α FIX: Never raise here — a missing token for one symbol
                # must NOT abort streaming/subscription wiring for all others.
                tok = _resolve_startup_token(sym)
                if tok:
                    if mdm:
                        mdm.register_symbol(sym, tok)
                    if tok not in tokens_to_poll:
                        tokens_to_poll.append(tok)
                    active_symbols.append(sym)
                    active_symbol_tokens[sym] = int(tok)
                    resolved_count += 1
                    LOGGER.info(f"✅ Resolved: {sym} -> token {tok}")
                    _gate_runner_symbol_add(
                        ctx,
                        sym,
                        pending_runner_symbols,
                        token=int(tok),
                        source="startup",
                    )
                    mdm_tracked = sym in getattr(mdm, "_tracked_symbols", set()) if mdm else False
                    broker_ws_token_requested = int(tok) in getattr(mdm, "_requested_tokens", set()) if mdm else False
                    runner_callback_registered = bool(getattr(ctx.data_hub, "_tick_subscribers", {}).get(sym)) if ctx.data_hub is not None else False
                    reason_detail = "startup"
                    success = bool(
                        mdm_tracked
                        and (runner_callback_registered or reason_detail in {"deferred_until_ready", "history_pending"})
                        and (broker_ws_token_requested or not market_open_now or reason_detail in {"startup", "history_pending"})
                    )
                    partial = (not success) and mdm_tracked
                    LOGGER.info(
                        "LIVE_SYMBOL_WIRED symbol=%s token=%s mdm_tracked=%s broker_ws_token_requested=%s runner_callback_registered=%s success=%s partial=%s reason=%s reason_detail=%s",
                        sym,
                        int(tok),
                        mdm_tracked,
                        broker_ws_token_requested,
                        runner_callback_registered,
                        success,
                        partial,
                        "startup",
                        reason_detail,
                        extra={
                            "event": "LIVE_SYMBOL_WIRED",
                            "symbol": sym,
                            "token": int(tok),
                            "mdm_tracked": mdm_tracked,
                            "broker_ws_token_requested": broker_ws_token_requested,
                            "runner_callback_registered": runner_callback_registered,
                            "success": success,
                            "partial": partial,
                            "reason": "startup",
                            "reason_detail": reason_detail,
                        },
                    )
                else:
                    unresolved_symbols.append(sym)
                    LOGGER.warning(
                        "⚠️ UNRESOLVED (no token for symbol class, skipping subscription): %s",
                        sym,
                    )
            mandatory_symbols: list[str] = []
            if "basket" in locals():
                mandatory_symbols = [
                    str(basket.get("spot_symbol") or ""),
                    str(basket.get("futures_symbol") or ""),
                ]
                ce_symbols = list(basket.get("ce_symbols") or [])
                pe_symbols = list(basket.get("pe_symbols") or [])
                if ce_symbols:
                    mandatory_symbols.append(str(ce_symbols[len(ce_symbols) // 2]))
                if pe_symbols:
                    mandatory_symbols.append(str(pe_symbols[len(pe_symbols) // 2]))
            mandatory_symbols = [sym for sym in mandatory_symbols if sym]
            missing_mandatory = [
                sym for sym in mandatory_symbols if sym not in active_symbol_tokens
            ]
            if missing_mandatory:
                ctx.live_orders_armed = False
                ctx.trading_ready = False
                ctx.readiness_mode = "DATA_WARMUP"
                ctx.effective_mode = "DATA_WARMUP"
                ctx.live_block_reason = "mandatory_tokens_missing"
                LOGGER.warning(
                    "LIVE_TOKEN_INTEGRITY_BLOCKED missing_symbols=%s",
                    missing_mandatory,
                    extra={
                        "event": "LIVE_TOKEN_INTEGRITY_BLOCKED",
                        "missing_symbols": missing_mandatory,
                    },
                )

            # BUG-β/γ/ζ FIX: mdm.warmup_history() removed.
            # It raised RuntimeError on missing token → aborted streamer.subscribe below.
            # It fed synthetic ticks via _handle_tick → MDM CandleBuilder, never feeding
            # runner.indicator_engine.  Historical hydration is handled by the get_ohlc
            # loop above (Path A) which calls runner.ingest_historical_bar directly.
            # Newly-added option symbols are hydrated on first evaluation via
            # runner._hydrate_missing_bars (per-symbol async fallback).

            LOGGER.info(
                f"📊 Resolution summary: {resolved_count}/{len(targets)} resolved"
            )
            if unresolved_symbols:
                if mdm and not mdm.ws_connected:
                    LOGGER.info("tracking_validation_deferred_ws_not_ready")
                else:
                    LOGGER.warning(
                        f"🔴 UNRESOLVED SYMBOLS (will NOT be polled): {unresolved_symbols}"
                    )
            LOGGER.info(f"✅ tokens_to_poll has {len(tokens_to_poll)} tokens")
            LOGGER.info(
                "WS_SUBSCRIPTION_SUMMARY tokens=%d symbols=%d",
                len(tokens_to_poll),
                len(sorted(set(active_symbols))),
            )
            subscribed_symbols = sorted({sym for sym in targets})
            LOGGER.critical(
                "📊 STRATEGY SUBSCRIBED SYMBOLS: %s",
                subscribed_symbols,
            )
            if not subscribed_symbols:
                LOGGER.critical("⛔ STRATEGY SUBSCRIPTION LIST IS EMPTY")

            deferred_flush_count = 0
            if ctx.data_hub is not None and hasattr(
                ctx.data_hub, "flush_pending_live_subscriptions"
            ):
                deferred_flush_count = int(
                    ctx.data_hub.flush_pending_live_subscriptions()
                )
                if deferred_flush_count:
                    LOGGER.info(
                        "DATAHUB_DEFERRED_SUBSCRIPTIONS_RECHECK count=%d phase=post_runner_start",
                        deferred_flush_count,
                        extra={
                            "event": "DATAHUB_DEFERRED_SUBSCRIPTIONS_RECHECK",
                            "count": deferred_flush_count,
                            "phase": "post_runner_start",
                        },
                    )
            if ctx.market_data_manager is not None:
                last_tick_map = getattr(ctx.market_data_manager, "_last_tick_time", {}) or {}
                live_tick_seen_count = sum(
                    1
                    for _sym in targets
                    if isinstance(last_tick_map.get(_sym), (int, float))
                    and float(last_tick_map.get(_sym, 0.0)) > 0
                )
                LOGGER.info(
                    "LIVE_WIRING_FINAL_STATUS symbols_count=%d tokens_count=%d mdm_tracked_count=%d mdm_subscriber_count=%d datahub_callback_symbols_count=%d mdm_subscribed_token_count=%s live_tick_seen_count=%d",
                    len(targets),
                    len(active_symbol_tokens),
                    len(getattr(ctx.market_data_manager, "_tracked_symbols", set())),
                    len(getattr(ctx.market_data_manager, "_subscribers", {})),
                    sum(
                        1
                        for _sym in targets
                        if bool(getattr(ctx.data_hub, "_tick_subscribers", {}).get(_sym))
                    )
                    if ctx.data_hub is not None
                    else 0,
                    _safe_ws_token_count(ctx),
                    live_tick_seen_count,
                    extra={
                        "event": "LIVE_WIRING_FINAL_STATUS",
                        "symbols_count": len(targets),
                        "tokens_count": len(active_symbol_tokens),
                        "mdm_tracked_count": len(
                            getattr(ctx.market_data_manager, "_tracked_symbols", set())
                        ),
                        "mdm_subscriber_count": len(
                            getattr(ctx.market_data_manager, "_subscribers", {})
                        ),
                        "datahub_callback_symbols_count": (
                            sum(
                                1
                                for _sym in targets
                                if bool(getattr(ctx.data_hub, "_tick_subscribers", {}).get(_sym))
                            )
                            if ctx.data_hub is not None
                            else 0
                        ),
                        "mdm_subscribed_token_count": _safe_ws_token_count(ctx),
                        "live_tick_seen_count": live_tick_seen_count,
                    },
                )

            # Post pre-hydration universe snapshot so logs capture the
            # end-of-startup pipeline state for every tradable symbol.
            try:
                _emit_trading_universe_summary(
                    ctx,
                    startup_symbols=active_symbols,
                    phase="startup_post_flush",
                )
            except Exception:  # pragma: no cover - observability must not raise
                pass

            if tokens_to_poll:
                if mdm is not None:
                    seeded = 0
                    seeded_tokens: set[int] = set()
                    for _sym, _tok in sorted(active_symbol_tokens.items()):
                        if _tok in tokens_to_poll and mdm.request_token_subscription(int(_tok), str(_sym)):
                            seeded += 1
                            seeded_tokens.add(int(_tok))
                    remaining_tokens = [int(_tok) for _tok in tokens_to_poll if int(_tok) not in seeded_tokens]
                    if remaining_tokens:
                        seeded += mdm.request_token_subscriptions(remaining_tokens)
                    LOGGER.info(
                        "✅ Routed %d/%d startup tokens via MarketDataManager",
                        seeded,
                        len(tokens_to_poll),
                    )
                elif streamer and hasattr(streamer, "subscribe"):
                    streamer.subscribe(tokens_to_poll)
                    LOGGER.info(
                        "✅ Wired %d tokens to PollingStreamer",
                        len(tokens_to_poll),
                    )
            if polling_fallback is not None and tokens_to_poll:
                polling_fallback.subscribe(tokens_to_poll)
                if ctx.websocket_manager is not None and ctx.market_data_manager is not None:
                    async def _polling_failover_supervisor() -> None:
                        """Supervise WS health and toggle polling fallback with hysteresis."""
                        degraded_since: float | None = None
                        recovered_since: float | None = None
                        activate_after = 3.0
                        recover_cooldown = 10.0
                        fallback_stale_sec = max(
                            1.0,
                            float(
                                os.getenv(
                                    "POLLING_SUPERVISOR_INDEX_STALE_SECONDS", "120"
                                )
                                or 120.0
                            ),
                        )
                        quote_stale_ms = int(fallback_stale_sec * 1000.0)
                        while True:
                            try:
                                degraded_since, recovered_since = await _polling_failover_supervisor_iteration(
                                    ctx,
                                    polling_fallback,
                                    quote_stale_ms=quote_stale_ms,
                                    degraded_since=degraded_since,
                                    recovered_since=recovered_since,
                                    activate_after=activate_after,
                                    recover_cooldown=recover_cooldown,
                                )
                            except Exception as failover_exc:  # noqa: BLE001
                                LOGGER.error(
                                    "Failure in polling failover supervisor: %s",
                                    failover_exc,
                                    exc_info=True,
                                    extra={
                                        "event": "POLLING_FAILOVER_SUPERVISOR_FAILED",
                                        "error_type": type(failover_exc).__name__,
                                        "error": str(failover_exc),
                                    },
                                )
                            await asyncio.sleep(1.0)

                    asyncio.create_task(_polling_failover_supervisor())

            # ── Bring the WebSocket online now that tokens are registered ──
            # The WS was opened earlier with the spot token only so we could
            # prove a fresh tick before selecting strikes.  Now that the
            # full token universe is known, ensure WS is up (idempotent) and
            # log the option subscription request so we can correlate which
            # tokens just joined the live subscription.
            if (
                ctx.market_data_manager is not None
                and hasattr(ctx.market_data_manager, "start_websocket")
                and ctx.websocket_enabled
            ):
                try:
                    ctx.market_data_manager.start_websocket()
                    LOGGER.info(
                        "OPTION_WS_SUBSCRIBE_REQUESTED count=%d total_tokens=%d",
                        max(0, len(tokens_to_poll) - 1),
                        len(tokens_to_poll),
                        extra={
                            "event": "OPTION_WS_SUBSCRIBE_REQUESTED",
                            "count": max(0, len(tokens_to_poll) - 1),
                            "total_tokens": len(tokens_to_poll),
                        },
                    )
                except Exception as _ws_start_exc:  # noqa: BLE001
                    LOGGER.error(
                        "MarketDataManager.start_websocket() failed: %s",
                        _ws_start_exc,
                        exc_info=True,
                    )

            # ✅ FIX: Disable MDM polling when PollingStreamer is active
            # MDM polling is redundant - PollingStreamer already feeds DataHub
            if mdm and not ctx.streamer and not ctx.websocket_enabled:
                # Only start MDM polling if there's no streamer
                asyncio.create_task(asyncio.to_thread(mdm._rest_poll_loop))
                LOGGER.info(
                    "POLLING_FALLBACK_STANDBY owner=mdm_internal_rest_poller reason=no_streamer",
                    extra={
                        "event": "POLLING_FALLBACK_STANDBY",
                        "owner": "mdm_internal_rest_poller",
                        "reason": "no_streamer",
                    },
                )
            else:
                LOGGER.info(
                    "POLLING_FALLBACK_STANDBY owner=PollingStreamer mdm_internal_rest_poller=false",
                    extra={
                        "event": "POLLING_FALLBACK_STANDBY",
                        "owner": "PollingStreamer",
                        "mdm_internal_rest_poller": False,
                    },
                )

            dynamic_option_symbols = {
                sym
                for sym in targets
                if sym.startswith("NFO:NIFTY")
                and (sym.endswith("CE") or sym.endswith("PE"))
            }
            option_universe_controller = UniverseController()
            option_universe_controller.update(dynamic_option_symbols)
            # Symbols waiting for enough MDM bars before being added to runner.

            async def _option_universe_sync_loop() -> None:
                """Keep option subscriptions aligned with the dynamic option universe."""
                nonlocal dynamic_option_symbols
                while True:
                    try:
                        # Retry symbols deferred from previous iterations.
                        if pending_runner_symbols and ctx.strategy_runner and ctx.market_data_manager:
                            still_pending: set[str] = set()
                            for _psym in list(pending_runner_symbols):
                                _mdm_bars = len(ctx.market_data_manager.get_ohlc_bars(_psym) or [])
                                _runner_bars = 0
                                try:
                                    _runner_engine = getattr(ctx.strategy_runner, "_indicator_engine", None)
                                    if _runner_engine is not None:
                                        _runner_bars = len(_runner_engine.get_history(_psym) or [])
                                except Exception:
                                    _runner_bars = 0
                                if _runner_bars >= _symbol_history_requirement(ctx):
                                    LOGGER.info(
                                        "RUNNER_SYMBOL_STATUS symbol=%s token=%s added_to_runner=%s runner_bars=%d mdm_bars=%d required_bars=%d history_ready=%s source=%s reason=%s",
                                        _psym,
                                        active_symbol_tokens.get(_psym),
                                        True,
                                        _runner_bars,
                                        _mdm_bars,
                                        _symbol_history_requirement(ctx),
                                        True,
                                        "dynamic_universe",
                                        "history_now_ready",
                                        extra={
                                            "event": "RUNNER_SYMBOL_STATUS",
                                            "symbol": _psym,
                                            "runner_bars": _runner_bars,
                                            "mdm_bars": _mdm_bars,
                                            "token": active_symbol_tokens.get(_psym),
                                            "added_to_runner": True,
                                            "required_bars": _symbol_history_requirement(ctx),
                                            "history_ready": True,
                                            "source": "dynamic_universe",
                                            "reason": "history_now_ready",
                                        },
                                    )
                                else:
                                    still_pending.add(_psym)
                            pending_runner_symbols.clear()
                            pending_runner_symbols.update(still_pending)

                        active_basket = getattr(ctx, "active_contract_basket", None) or getattr(ctx, "active_trading_universe", None)
                        latest_symbols = set(_option_symbols_from_active_basket(active_basket))
                        if not latest_symbols:
                            LOGGER.warning(
                                "OPTION_UNIVERSE_SYNC_SKIPPED reason=ACTIVE_BASKET_MISSING",
                                extra={"event": "OPTION_UNIVERSE_SYNC_SKIPPED", "reason": "ACTIVE_BASKET_MISSING"},
                            )

                        added, removed = option_universe_controller.update(
                            latest_symbols
                        )
                        add_symbols = sorted(added)
                        drop_symbols = sorted(removed)
                        market_open = get_market_state() == MarketState.OPEN
                        if not market_open and drop_symbols:
                            LOGGER.info(
                                "DYNAMIC_BASKET_REFRESH_SKIPPED_OFF_MARKET reason=%s",
                                "market_closed_preserve_active_context",
                                extra={
                                    "event": "DYNAMIC_BASKET_REFRESH_SKIPPED_OFF_MARKET",
                                    "reason": "market_closed_preserve_active_context",
                                },
                            )
                            drop_symbols = []

                        if add_symbols or drop_symbols:
                            LOGGER.info(
                                "ACTIVE_BASKET_REFRESH added=%d removed=%d reason=%s",
                                len(add_symbols),
                                len(drop_symbols),
                                "atm_shift_or_expiry",
                                extra={
                                    "event": "active_basket_refresh",
                                    "added": list(add_symbols),
                                    "removed": list(drop_symbols),
                                    "reason": "atm_shift_or_expiry",
                                },
                            )

                        for sym in add_symbols:
                            runner_callback_registered = False
                            mdm_tracked = False
                            broker_ws_token_requested = False
                            if ctx.market_data_manager:
                                ctx.market_data_manager.ensure_tracking(sym)
                                mdm_tracked = sym in getattr(
                                    ctx.market_data_manager, "_tracked_symbols", set()
                                )
                            tok = None
                            if _im and _im.is_loaded():
                                try:
                                    tok = _im.get_token(sym)
                                except RuntimeError:
                                    tok = None
                            if tok is None:
                                LOGGER.warning(
                                    f"⚠️ Universe sync: no token for {sym}, skipping"
                                )
                                continue
                            if tok and ctx.market_data_manager:
                                ctx.market_data_manager.register_symbol(sym, tok)
                            if tok:
                                active_symbol_tokens[sym] = int(tok)
                            if tok and ctx.market_data_manager is not None:
                                ctx.market_data_manager.request_token_subscription(
                                    tok,
                                    symbol=sym,
                                )
                                broker_ws_token_requested = True
                            # Fetch OHLC history into MDM BEFORE adding to
                            # strategy runner so the runner's internal
                            # _prehydrate_symbol_history finds bars already
                            # present and marks the symbol READY instead of
                            # immediately evaluating with no history.
                            if ctx.broker_client and ctx.market_data_manager:
                                try:
                                    required_bars = _symbol_history_requirement(ctx)
                                    lookback_minutes = _history_lookback_minutes(required_bars)
                                    await ctx.market_data_manager.hydrate_symbol_history(
                                        sym,
                                        interval="minute",
                                        days=_history_lookback_days(required_bars),
                                        max_bars=required_bars,
                                        reason="dynamic_option_universe",
                                    )
                                    runner_ingested = 0
                                    bars = ctx.market_data_manager.get_ohlc_bars(sym, limit=required_bars)
                                    for row in list(bars or []):
                                        bar_data = dict(row)
                                        bar_data["symbol"] = sym
                                        if (
                                            getattr(ctx, "strategy_runner", None) is not None
                                            and hasattr(ctx.strategy_runner, "ingest_historical_bar")
                                        ):
                                            ctx.strategy_runner.ingest_historical_bar(bar_data)
                                            runner_ingested += 1
                                    ctx.market_data_manager.update_hydration_status(
                                        sym,
                                        ctx.market_data_manager.get_ohlc_bars(sym),
                                    )
                                    if getattr(ctx, "strategy_runner", None) is not None:
                                        runner_history_count = len(
                                            ctx.strategy_runner._indicator_engine.get_history(sym) or []
                                        )
                                    else:
                                        runner_history_count = 0
                                    mdm_history_count = len(
                                        ctx.market_data_manager.get_ohlc_bars(sym) or []
                                    )
                                    LOGGER.info(
                                        "RUNNER_HISTORY_INGESTED symbol=%s token=%s bars_ingested=%d source=%s runner_history_count=%d mdm_history_count=%d",
                                        sym,
                                        int(tok),
                                        runner_ingested,
                                        "dynamic_hydration",
                                        runner_history_count,
                                        mdm_history_count,
                                        extra={
                                            "event": "RUNNER_HISTORY_INGESTED",
                                            "symbol": sym,
                                            "token": int(tok),
                                            "bars_ingested": runner_ingested,
                                            "source": "dynamic_hydration",
                                            "runner_history_count": runner_history_count,
                                            "mdm_history_count": mdm_history_count,
                                        },
                                    )
                                except Exception as hydration_exc:  # noqa: BLE001
                                    LOGGER.warning(
                                        "option_symbol_hydration_failed symbol=%s err=%s",
                                        sym,
                                        hydration_exc,
                                        extra={
                                            "event": "option_symbol_hydration_failed",
                                            "symbol": sym,
                                        },
                                    )
                            # Add to runner only after MDM holds enough bars.
                            # Defer if still under-hydrated; the loop retries.
                            _gate_runner_symbol_add(
                                ctx,
                                sym,
                                pending_runner_symbols,
                                token=int(tok),
                                source="dynamic_universe",
                            )
                            if ctx.data_hub is not None:
                                tick_subscribers = getattr(ctx.data_hub, "_tick_subscribers", {})
                                runner_callback_registered = bool(tick_subscribers.get(sym))
                            reason_detail = "dynamic_add"
                            success = bool(
                                mdm_tracked
                                and (runner_callback_registered or reason_detail in {"deferred_until_ready", "history_pending"})
                                and (broker_ws_token_requested or not market_open_now or reason_detail in {"startup", "history_pending"})
                            )
                            partial = (not success) and mdm_tracked
                            LOGGER.info(
                                "LIVE_SYMBOL_WIRED symbol=%s token=%s mdm_tracked=%s broker_ws_token_requested=%s runner_callback_registered=%s success=%s partial=%s reason=%s reason_detail=%s",
                                sym,
                                int(tok),
                                mdm_tracked,
                                broker_ws_token_requested,
                                runner_callback_registered,
                                success,
                                partial,
                                "dynamic_add",
                                reason_detail,
                                extra={
                                    "event": "LIVE_SYMBOL_WIRED",
                                    "symbol": sym,
                                    "token": int(tok),
                                    "mdm_tracked": mdm_tracked,
                                    "broker_ws_token_requested": broker_ws_token_requested,
                                    "runner_callback_registered": runner_callback_registered,
                                    "success": success,
                                    "partial": partial,
                                    "reason": "dynamic_add",
                                    "reason_detail": reason_detail,
                                },
                            )

                        for sym in drop_symbols:
                            lock_ts = ctx.execution_lock_timestamps.get(sym)
                            lock_age_s = (datetime.now(timezone.utc) - lock_ts).total_seconds() if lock_ts is not None else None
                            sticky_seconds = int(os.getenv("OPTION_STICKY_SECONDS", "120") or "120")
                            if sym in ctx.execution_locked_symbols and (lock_age_s is None or lock_age_s < sticky_seconds):
                                LOGGER.info("EXECUTION_UNIVERSE_STICKY_KEEP symbol=%s lock_age_s=%s", sym, lock_age_s)
                                continue
                            ctx.execution_locked_symbols.discard(sym)
                            ctx.execution_lock_timestamps.pop(sym, None)
                            removed_from_mdm = False
                            removed_from_datahub = False
                            removed_from_runner = False
                            tok = None
                            if _im and _im.is_loaded():
                                try:
                                    tok = _im.get_token(sym)
                                except RuntimeError:
                                    tok = None
                            if tok and ctx.market_data_manager is not None:
                                ctx.market_data_manager.request_token_unsubscription(
                                    tok,
                                    symbol=sym,
                                )
                            if ctx.market_data_manager is not None:
                                removed_from_mdm = bool(ctx.market_data_manager.untrack(sym))
                            if ctx.data_hub is not None:
                                try:
                                    ctx.data_hub.unsubscribe_ticks(sym)
                                    removed_from_datahub = True
                                except Exception:
                                    removed_from_datahub = False
                            if ctx.strategy_runner:
                                ctx.strategy_runner.remove_symbol(sym)
                                removed_from_runner = True
                            pending_runner_symbols.discard(sym)
                            active_symbol_tokens.pop(sym, None)
                            LOGGER.info(
                                "SYMBOL_REMOVAL_CLEANUP symbol=%s removed_from_mdm=%s removed_from_datahub=%s removed_from_runner=%s removed_from_watchdog=%s pending_backfill_cancelled=%s",
                                sym,
                                removed_from_mdm,
                                removed_from_datahub,
                                removed_from_runner,
                                True,
                                True,
                                extra={
                                    "event": "SYMBOL_REMOVAL_CLEANUP",
                                    "symbol": sym,
                                    "removed_from_mdm": removed_from_mdm,
                                    "removed_from_datahub": removed_from_datahub,
                                    "removed_from_runner": removed_from_runner,
                                    "removed_from_watchdog": True,
                                    "pending_backfill_cancelled": True,
                                },
                            )

                        dynamic_option_symbols = latest_symbols

                        # Universe summary + per-symbol pipeline status after
                        # a dynamic-universe mutation so operators can see
                        # where each option sits in the live lifecycle.
                        if add_symbols or drop_symbols:
                            try:
                                basket_symbols = [
                                    str(sym)
                                    for sym in dict.fromkeys(
                                        ["NSE:NIFTY", *sorted(latest_symbols)]
                                    )
                                ]
                                committed_ce, committed_pe = _commit_active_dynamic_basket(
                                    ctx,
                                    basket=cast(
                                        Mapping[str, object],
                                        getattr(ctx, "active_trading_universe", {}) or {},
                                    ),
                                    option_symbols=sorted(latest_symbols),
                                    symbols=basket_symbols,
                                    atm_strike=cast(
                                        int | float | str | None,
                                        (getattr(ctx, "active_trading_universe", {}) or {}).get("atm_strike"),
                                    ),
                                )
                                for _sym in add_symbols:
                                    _tok = active_symbol_tokens.get(_sym)
                                    _emit_option_symbol_pipeline_status(
                                        ctx,
                                        symbol=_sym,
                                        token=_tok,
                                        selected=_sym in {committed_ce, committed_pe},
                                        hydrated_bars=(
                                            len(
                                                ctx.market_data_manager.get_ohlc_bars(
                                                    _sym
                                                )
                                                or []
                                            )
                                            if ctx.market_data_manager
                                            else None
                                        ),
                                        runner_added=bool(ctx.strategy_runner),
                                        source="dynamic_universe",
                                        reason="post_universe_sync",
                                    )
                                ce_bars = (
                                    len(ctx.market_data_manager.get_ohlc_bars(committed_ce) or [])
                                    if committed_ce and ctx.market_data_manager is not None
                                    else 0
                                )
                                pe_bars = (
                                    len(ctx.market_data_manager.get_ohlc_bars(committed_pe) or [])
                                    if committed_pe and ctx.market_data_manager is not None
                                    else 0
                                )
                                LOGGER.info(
                                    "ACTIVE_DYNAMIC_BASKET_COMMITTED selected_ce=%s selected_pe=%s atm_strike=%s option_count=%d ce_ready=%s pe_ready=%s",
                                    committed_ce,
                                    committed_pe,
                                    (getattr(ctx, "active_trading_universe", {}) or {}).get("atm_strike"),
                                    len(latest_symbols),
                                    ce_bars >= _symbol_history_requirement(ctx),
                                    pe_bars >= _symbol_history_requirement(ctx),
                                )
                                if committed_ce and committed_pe:
                                    await _recompute_and_push_runtime_readiness(
                                        ctx,
                                        reason="dynamic_basket_committed",
                                    )
                                else:
                                    LOGGER.info(
                                        "LIVE_READINESS_DEFERRED reason=dynamic_basket_not_committed_yet selected_ce=%s selected_pe=%s",
                                        committed_ce,
                                        committed_pe,
                                    )
                                _emit_trading_universe_summary(
                                    ctx,
                                    startup_symbols=sorted(latest_symbols),
                                    phase="dynamic_universe_update",
                                )
                                mdm_tracked_count = 0
                                mdm_subscriber_count = 0
                                datahub_callback_symbols_count = 0
                                mdm_subscribed_token_count = 0
                                live_tick_seen_count = 0
                                if ctx.market_data_manager is not None:
                                    mdm_tracked_count = len(
                                        getattr(ctx.market_data_manager, "_tracked_symbols", set())
                                    )
                                    mdm_subscriber_count = len(
                                        getattr(ctx.market_data_manager, "_subscribers", {})
                                    )
                                    mdm_subscribed_token_count = _safe_ws_token_count(ctx)
                                    last_tick_map = getattr(ctx.market_data_manager, "_last_tick_time", {}) or {}
                                    live_tick_seen_count = sum(
                                        1
                                        for _s in latest_symbols
                                        if isinstance(last_tick_map.get(_s), (int, float))
                                        and float(last_tick_map.get(_s, 0.0)) > 0
                                    )
                                if ctx.data_hub is not None:
                                    datahub_callback_symbols_count = sum(
                                        1
                                        for _s in latest_symbols
                                        if bool(getattr(ctx.data_hub, "_tick_subscribers", {}).get(_s))
                                    )
                                LOGGER.info(
                                    "LIVE_WIRING_FINAL_STATUS symbols_count=%d tokens_count=%d mdm_tracked_count=%d mdm_subscriber_count=%d datahub_callback_symbols_count=%d mdm_subscribed_token_count=%s live_tick_seen_count=%d",
                                    len(latest_symbols),
                                    len(active_symbol_tokens),
                                    mdm_tracked_count,
                                    mdm_subscriber_count,
                                    datahub_callback_symbols_count,
                                    mdm_subscribed_token_count,
                                    live_tick_seen_count,
                                    extra={
                                        "event": "LIVE_WIRING_FINAL_STATUS",
                                        "symbols_count": len(latest_symbols),
                                        "tokens_count": len(active_symbol_tokens),
                                        "mdm_tracked_count": mdm_tracked_count,
                                        "mdm_subscriber_count": mdm_subscriber_count,
                                        "datahub_callback_symbols_count": datahub_callback_symbols_count,
                                        "mdm_subscribed_token_count": mdm_subscribed_token_count,
                                        "live_tick_seen_count": live_tick_seen_count,
                                    },
                                )
                            except Exception:  # pragma: no cover - obs must not raise
                                pass
                    except Exception as exc:
                        LOGGER.error(
                            "Failure in option universe sync loop: %s",
                            exc,
                            exc_info=exc,
                        )
                    await asyncio.sleep(60) # Objective 7: 60s interval

            asyncio.create_task(_option_universe_sync_loop())
            startup_trade_ready = True
        except Exception as e:
            warmup_tokens = {"WARMING_UP", "DATA_WARMUP", "HISTORICAL_READY"}
            error_text = str(e).upper()
            is_warmup_like = any(token in error_text for token in warmup_tokens)
            if is_warmup_like:
                LOGGER.warning(
                    "Hydration/Tracking warmup state: %s",
                    e,
                    extra={"event": "HYDRATION_TRACKING_WARMUP", "error": str(e)},
                )
                ctx.live_orders_armed = False
                ctx.trading_ready = False
                ctx.readiness_mode = "DATA_WARMUP"
                ctx.effective_mode = "DATA_WARMUP"
                ctx.live_block_reason = "startup_warmup_waiting_for_live_bars"
                LOGGER.info(
                    "LIVE_STARTUP_CONTINUES_IN_WARMUP reason=%s",
                    e,
                    extra={
                        "event": "LIVE_STARTUP_CONTINUES_IN_WARMUP",
                        "reason": str(e),
                    },
                )
                startup_trade_ready = True
            else:
                norm_basket = normalize_active_basket_schema(dict((getattr(ctx, "active_contract_basket", None) or getattr(ctx, "active_trading_universe", {}) or {})))
                LOGGER.exception(
                    "HYDRATION_TRACKING_FAILED error_type=%s error=%s basket_keys=%s selected_ce=%s selected_pe=%s spot_symbol=%s futures_symbol=%s",
                    type(e).__name__,
                    e,
                    sorted(norm_basket.keys()),
                    norm_basket.get("selected_ce"),
                    norm_basket.get("selected_pe"),
                    norm_basket.get("spot_symbol"),
                    norm_basket.get("futures_symbol"),
                    extra={
                        "event": "HYDRATION_TRACKING_FAILED",
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "basket_keys": sorted(norm_basket.keys()),
                        "selected_ce": norm_basket.get("selected_ce"),
                        "selected_pe": norm_basket.get("selected_pe"),
                        "spot_symbol": norm_basket.get("spot_symbol"),
                        "futures_symbol": norm_basket.get("futures_symbol"),
                    },
                )
                ctx.live_orders_armed = False
                ctx.trading_ready = False
                if norm_basket.get("option_symbols") or norm_basket.get("selected_ce") or norm_basket.get("selected_pe"):
                    ctx.live_block_reason = f"hydration_tracking_degraded:{type(e).__name__}"
                    LOGGER.warning(
                        "HYDRATION_TRACKING_DEGRADED_CONTINUE reason=%s active_basket_available=True",
                        type(e).__name__,
                        extra={"event": "HYDRATION_TRACKING_DEGRADED_CONTINUE", "reason": type(e).__name__, "active_basket_available": True},
                    )
                else:
                    ctx.live_block_reason = "ACTIVE_BASKET_MISSING"
                    LOGGER.warning(
                        "LIVE_READINESS_BLOCKED reason=ACTIVE_BASKET_MISSING stage=hydration_tracking",
                        extra={"event": "LIVE_READINESS_BLOCKED", "reason": "ACTIVE_BASKET_MISSING", "stage": "hydration_tracking"},
                    )

    # ---------------------------------------------------------
    # 4. Start subsystems (guarded singleton startup)
    # ---------------------------------------------------------
    if broker_ready:
        try:
            if not ctx.subsystems_started:
                # ── Subscribe handlers BEFORE starting the bus so dispatchers
                # are created for all registered message types.
                # OrderProcessor registers the SIGNAL handler; without calling
                # start() here, MessageBus.start() sees 0 subscribers and logs
                # "started with 0 active dispatchers" — signals never execute.
                

                _wire_and_start_message_bus(ctx)

                # MDM internals are started once near startup entry with
                # defer_ws=True; websocket bring-up is handled explicitly.



                if ctx.order_manager:
                    ctx.order_manager.start_monitoring()

                instrument_cache_ready.wait()
                # BUG-δ FIX: data_hub.warmup_indicators removed — it always skips because
                # ZerodhaKiteClient has no get_historical_candles method. Historical
                # indicator warmup is handled by runner.ingest_historical_bar in section 3.

                if not ctx.websocket_enabled and ctx.market_data_manager is not None:
                    ctx.market_data_manager._seed_completed = True
                if ctx.websocket_enabled:
                    if ctx.stream_supervisor and not ctx.stream_supervisor_started:
                        ctx.stream_supervisor.start()
                        ctx.stream_supervisor_started = True
                    elif (
                        hasattr(ctx.streamer, "start")
                        and not ctx.stream_supervisor_started
                    ):
                        res = ctx.streamer.start()
                        if inspect.isawaitable(res):
                            await res
                        ctx.stream_supervisor_started = True

                if ctx.strategy_runner:
                    if ctx.market_data_manager is not None:
                        try:
                            def _safe_runtime_mode(local_ctx: BotContext) -> str:
                                mode = str(
                                    getattr(local_ctx, "configured_mode", None)
                                    or getattr(local_ctx, "runtime_mode", None)
                                    or os.getenv("EXECUTION_MODE", "")
                                    or os.getenv("MODE", "")
                                    or "LIVE"
                                ).upper()
                                return mode if mode in {"LIVE", "PAPER", "SHADOW", "DATA_WARMUP"} else "LIVE"

                            configured_runtime_mode = _safe_runtime_mode(ctx)
                            await ctx.market_data_manager.wait_until_ready(timeout=30.0)
                            readiness_ready = bool(ctx.market_data_manager.ready)
                            readiness_state = (
                                ctx.market_data_manager.readiness_state_snapshot()
                            )
                            hard_ready = bool(readiness_state.get("hard_ready"))
                            spot_ready = bool(readiness_state.get("spot_ready"))
                            missing_hard = list(readiness_state.get("missing_hard") or [])
                            indicators_ready_for_trading = hard_ready
                            if ctx.market_regime_manager is not None:
                                ctx.market_regime_manager.indicators_ready = (
                                    indicators_ready_for_trading
                                )
                            if ctx.data_hub is not None:
                                ctx.data_hub.indicators_ready = (
                                    indicators_ready_for_trading
                                )
                            LOGGER.info(
                                "INDICATOR_READINESS_SYNC ready=%s",
                                indicators_ready_for_trading,
                                extra={
                                    "event": "INDICATOR_READINESS_SYNC",
                                    "ready": indicators_ready_for_trading,
                                },
                            )
                            ws_connected = bool(
                                ctx.websocket_manager.is_connected()
                                if ctx.websocket_manager is not None
                                else False
                            )
                            if readiness_ready and spot_ready:
                                LOGGER.info(
                                    "Condition met: startup_pipeline_ready",
                                    extra={
                                        "event": "startup_pipeline_ready",
                                        "readiness_ready": readiness_ready,
                                        "ws_connected": ws_connected,
                                    },
                                )
                            elif hard_ready:
                                LOGGER.warning(
                                    "startup_pipeline_ready_degraded spot_ready=%s",
                                    spot_ready,
                                    extra={
                                        "event": "startup_pipeline_ready_degraded",
                                        "hard_ready": hard_ready,
                                        "spot_ready": spot_ready,
                                        "timed_out": not readiness_ready,
                                        "ws_connected": ws_connected,
                                    },
                                )
                            else:
                                non_live_mode = configured_mode in {"PAPER", "SHADOW"}
                                runner_bars: dict[str, int] = {}
                                mdm_bars: dict[str, int] = {}
                                datahub_bars: dict[str, int] = {}
                                fresh_quote_symbols: list[str] = []
                                for sym in list((readiness_state.get("requirements", {}) or {}).get("options", []) or []):
                                    try:
                                        runner_bars[sym] = int(len(ctx.strategy_runner.get_recent_bars(sym) or [])) if hasattr(ctx.strategy_runner, "get_recent_bars") else 0
                                    except Exception:
                                        runner_bars[sym] = 0
                                    try:
                                        mdm_bars[sym] = int(len(ctx.market_data_manager.get_ohlc_bars(sym, limit=20) or []))
                                    except Exception:
                                        mdm_bars[sym] = 0
                                    if getattr(ctx, "data_hub", None) is not None and hasattr(ctx.data_hub, "get_ohlc_bars"):
                                        try:
                                            datahub_bars[sym] = int(len(ctx.data_hub.get_ohlc_bars(sym, limit=20) or []))
                                        except Exception:
                                            datahub_bars[sym] = 0
                                    try:
                                        snap = ctx.market_data_manager.get_symbol_snapshot(sym)
                                        if float(getattr(snap, "ltp", 0.0) or 0.0) > 0 and float(getattr(snap, "tick_age_s", 9999.0) or 9999.0) <= 60.0:
                                            fresh_quote_symbols.append(sym)
                                    except Exception:
                                        pass
                                LOGGER.info(
                                    "DATA_PIPELINE_NOT_READY hard_ready=%s spot_ready=%s missing=%s",
                                    hard_ready,
                                    spot_ready,
                                    missing_hard,
                                    extra={
                                        "event": "DATA_PIPELINE_NOT_READY",
                                        "hard_ready": hard_ready,
                                        "spot_ready": spot_ready,
                                        "missing": missing_hard,
                                        "runner_bars": runner_bars,
                                        "mdm_bars": mdm_bars,
                                        "datahub_bars": datahub_bars,
                                        "fresh_quote_symbols": fresh_quote_symbols,
                                    },
                                )
                                if non_live_mode and _runner_is_running(ctx.strategy_runner):
                                    ctx.data_observation_ready = True
                                    ctx.live_orders_armed = False
                                    ctx.trading_ready = False
                                    ctx.live_block_reason = "paper_mode_or_startup_warmup"
                                    LOGGER.warning(
                                        "startup_pipeline_warmup_nonlive missing=%s",
                                        ",".join(missing_hard) if missing_hard else "unknown",
                                        extra={"event": "startup_pipeline_warmup_nonlive", "hard_ready": hard_ready, "spot_ready": spot_ready, "timed_out": not readiness_ready, "missing": missing_hard, "ws_connected": ws_connected},
                                    )
                                else:
                                    LOGGER.error(
                                        "startup_pipeline_incomplete missing=%s",
                                        ",".join(missing_hard) if missing_hard else "unknown",
                                        extra={
                                            "event": "startup_pipeline_incomplete",
                                            "hard_ready": hard_ready,
                                            "spot_ready": spot_ready,
                                            "timed_out": not readiness_ready,
                                            "missing": missing_hard,
                                            "ws_connected": ws_connected,
                                        },
                                    )
                            live_mode = configured_mode == "LIVE" or (
                                str(os.getenv("ENABLE_LIVE", "false"))
                                .strip()
                                .lower()
                                in {"1", "true", "yes", "on"}
                            )
                            quote_capability = _resolve_quote_capability(ctx)
                            quote_available = bool(quote_capability["available"])
                            quote_error = quote_capability["error"]
                            ws_quote_proof = False
                            ws_quote_proof_fn = getattr(
                                ctx.market_data_manager, "has_ws_tradable_quote", None
                            )
                            if callable(ws_quote_proof_fn):
                                ws_quote_proof = bool(ws_quote_proof_fn())
                            ws_ltp_proof_fn = getattr(
                                ctx.market_data_manager, "has_fresh_ws_ltp", None
                            )
                            ws_ltp_proof = bool(ws_ltp_proof_fn()) if callable(ws_ltp_proof_fn) else False
                            ws_quote_for_gate = bool(ws_quote_proof or ws_ltp_proof)
                            if (
                                ws_quote_for_gate
                                and not quote_available
                                and not policy.quote_do_not_block_if_ws_healthy
                            ):
                                ws_quote_for_gate = False
                            try:
                                market_state = get_market_state()
                            except Exception:  # noqa: BLE001
                                market_state = None
                            session_state_str = (
                                "open" if market_state == MarketState.OPEN else "closed"
                            )
                            LOGGER.info(
                                "MARKET_SESSION_STATE state=%s",
                                session_state_str,
                                extra={
                                    "event": "MARKET_SESSION_STATE",
                                    "state": session_state_str,
                                },
                            )
                            if not quote_available:
                                LOGGER.info(
                                    "BROKER_QUOTE_CAPABILITY status=unavailable reason=%s",
                                    quote_error or "unknown",
                                    extra={
                                        "event": "BROKER_QUOTE_CAPABILITY",
                                        "status": "unavailable",
                                        "reason": quote_error or "unknown",
                                    },
                                )
                            missing_soft: list[str] = []
                            if "futures" in set(missing_hard):
                                missing_soft.append("futures")
                                missing_hard = [item for item in missing_hard if item != "futures"]
                                LOGGER.info(
                                    "SOFT_READINESS_MISSING missing=futures action=continue_with_spot_option_context",
                                    extra={"event": "SOFT_READINESS_MISSING", "missing": "futures", "action": "continue_with_spot_option_context"},
                                )
                            basket_universe = cast(
                                Mapping[str, Any],
                                getattr(ctx, "active_trading_universe", {}) or {},
                            )
                            option_symbols = list(basket_universe.get("option_symbols", []) or [])
                            selected_ce = cast(
                                str | None,
                                getattr(ctx, "selected_ce", None)
                                or basket_universe.get("selected_ce"),
                            )
                            selected_pe = cast(
                                str | None,
                                getattr(ctx, "selected_pe", None)
                                or basket_universe.get("selected_pe"),
                            )
                            quote_ce = _fresh_option_quote(ctx, selected_ce) if selected_ce else None
                            quote_pe = _fresh_option_quote(ctx, selected_pe) if selected_pe else None
                            hydrated_ce = _count_symbol_bars(ctx, selected_ce) if selected_ce else 0
                            hydrated_pe = _count_symbol_bars(ctx, selected_pe) if selected_pe else 0
                            option_ticks_ready = bool(quote_ce is not None and quote_pe is not None)
                            futures_ready = "futures" not in set(missing_soft)
                            atm_ce_ready = bool(selected_ce and (quote_ce is not None or hydrated_ce >= 3))
                            atm_pe_ready = bool(selected_pe and (quote_pe is not None or hydrated_pe >= 3))
                            req_atm_ce = readiness_state.get("requirements", {}).get("atm_ce")
                            req_atm_pe = readiness_state.get("requirements", {}).get("atm_pe")
                            if quote_ce and req_atm_ce and quote_ce != req_atm_ce:
                                LOGGER.info("ATM_CE_SUBSTITUTED_WITH_READY_OPTION atm_ce=%s ready_ce=%s", req_atm_ce, quote_ce)
                            if quote_pe and req_atm_pe and quote_pe != req_atm_pe:
                                LOGGER.info("ATM_PE_SUBSTITUTED_WITH_READY_OPTION atm_pe=%s ready_pe=%s", req_atm_pe, quote_pe)
                            runner_running = _runner_is_running(ctx.strategy_runner)
                            ctx.spot_ready = bool(spot_ready)
                            ctx.data_observation_ready = bool(spot_ready or ws_quote_proof or ws_ltp_proof)
                            enough_runner_symbols = bool(
                                len(getattr(ctx.strategy_runner, "_active_symbols", set()) or []) > 0
                                if ctx.strategy_runner is not None
                                else False
                            )
                            ctx.evaluation_ready = bool(
                                runner_running
                                and (bool(hydrated_ce) or bool(hydrated_pe) or enough_runner_symbols)
                            )
                            spot_ready_for_live = bool(spot_ready or ws_quote_for_gate or ws_ltp_proof)
                            ctx.data_hard_ready = bool(
                                spot_ready_for_live and atm_ce_ready and atm_pe_ready
                            )
                            LOGGER.info(
                                "OPTION_QUOTE_READY selected_ce=%s selected_pe=%s",
                                selected_ce,
                                selected_pe,
                                extra={"event": "OPTION_QUOTE_READY", "selected_ce": selected_ce, "selected_pe": selected_pe},
                            )
                            LOGGER.info(
                                "OPTION_HISTORY_READY selected_ce=%s selected_pe=%s",
                                selected_ce,
                                selected_pe,
                                extra={"event": "OPTION_HISTORY_READY", "selected_ce": selected_ce, "selected_pe": selected_pe},
                            )
                            ctx.mdm_strict_hard_ready = bool(hard_ready)
                            ctx.data_pipeline_ready = bool(ctx.data_hard_ready)
                            ctx.trading_ready = bool(
                                ctx.data_hard_ready
                                and configured_runtime_mode == "LIVE"
                                and runner_running
                            )
                            if configured_runtime_mode in {"PAPER", "SHADOW"}:
                                ctx.live_orders_armed = False
                                if ctx.strategy_runner is not None:
                                    status = ctx.strategy_runner.get_status()
                                    if (not bool(status.get("running"))) and readiness_symbols:
                                        await _ensure_strategy_runner_started(ctx, reason="paper_readiness_recovery")
                                LOGGER.info("PAPER_EVALUATION_READY hard_ready=%s spot_ready=%s runner_running=%s", hard_ready, spot_ready, bool(ctx.strategy_runner.get_status().get("running")) if ctx.strategy_runner else False, extra={"event":"PAPER_EVALUATION_READY","hard_ready":bool(hard_ready),"spot_ready":bool(spot_ready),"live_orders_armed":False})
                            if ctx.data_hard_ready:
                                LOGGER.info(
                                    "DATA_PIPELINE_READY hard_ready=%s spot_ready=%s symbols_ready=%s",
                                    hard_ready,
                                    spot_ready,
                                    len(readiness_state.get("ready_symbols") or []),
                                    extra={
                                        "event": "DATA_PIPELINE_READY",
                                        "hard_ready": hard_ready,
                                        "spot_ready": spot_ready,
                                        "symbols_ready": len(
                                            readiness_state.get("ready_symbols") or []
                                        ),
                                    },
                                )
                            market_open_now = market_state == MarketState.OPEN
                            option_exec_min_bars = int(
                                os.getenv(
                                    "READINESS_OPTION_EXEC_MIN_BARS",
                                    os.getenv("OPTION_EXECUTION_MIN_BARS", "30"),
                                )
                                or 30
                            )
                            armed, blocking_reasons = compute_live_readiness(
                                live_mode=bool(live_mode),
                                hard_ready=bool(ctx.data_hard_ready),
                                quote_available=bool(quote_available),
                                ws_quote_proof=bool(ws_quote_for_gate),
                                market_open=bool(market_open_now),
                                runner_running=bool(runner_running),
                                selected_ce=selected_ce,
                                selected_pe=selected_pe,
                                ce_bars=int(hydrated_ce),
                                pe_bars=int(hydrated_pe),
                                option_exec_min_bars=option_exec_min_bars,
                                ce_quote_ready=bool(quote_ce is not None),
                                pe_quote_ready=bool(quote_pe is not None),
                            )
                            data_warmup_reasons: list[str] = list(blocking_reasons)
                            if live_mode and not quote_available and ws_quote_proof:
                                LOGGER.info(
                                    "BROKER_QUOTE_DEGRADED_CONTINUING_WITH_WS",
                                    extra={
                                        "event": "BROKER_QUOTE_DEGRADED_CONTINUING_WITH_WS",
                                        "ws_quote_proof": True,
                                        "quote_available": False,
                                    },
                                )
                            if live_mode and armed and bool(getattr(ctx, "live_orders_armed", False)):
                                ctx.trading_ready = bool(ctx.data_hard_ready)
                                previous_mode = str(getattr(ctx, "readiness_mode", "DATA_WARMUP"))
                                ctx.readiness_mode = "LIVE_READY"
                                ctx.effective_mode = ctx.readiness_mode
                                LOGGER.info(
                                    "STARTUP_MODE_TRANSITION from=%s to=%s reason=selected_options_exec_ready",
                                    previous_mode,
                                    ctx.readiness_mode,
                                )
                                LOGGER.info(
                                    "LIVE_TRADING_ARMED hard_ready=%s quote_available=%s ws_quote_proof=%s",
                                    hard_ready,
                                    quote_available,
                                    ws_quote_proof,
                                    extra={
                                        "event": "LIVE_TRADING_ARMED",
                                        "hard_ready": bool(hard_ready),
                                        "quote_available": bool(quote_available),
                                        "ws_quote_proof": bool(ws_quote_proof),
                                    },
                                )
                            elif live_mode:
                                ctx.live_orders_armed = False
                                ctx.trading_ready = False
                                previous_mode = str(getattr(ctx, "readiness_mode", "DATA_WARMUP"))
                                ctx.readiness_mode = (
                                    "EVALUATION_READY"
                                    if bool(ctx.evaluation_ready)
                                    else "DATA_WARMUP"
                                )
                                ctx.effective_mode = ctx.readiness_mode
                                pre_final_reason = (
                                    "selected_option_history_or_quote_pending_pre_final_readiness"
                                    if ctx.readiness_mode == "EVALUATION_READY"
                                    else "awaiting_spot_or_active_basket_pre_final_readiness"
                                )
                                LOGGER.info(
                                    "STARTUP_MODE_PRE_FINAL_READINESS current=%s reason=%s stage=%s",
                                    ctx.readiness_mode,
                                    pre_final_reason,
                                    "pre_final_readiness",
                                    extra={
                                        "event": "STARTUP_MODE_PRE_FINAL_READINESS",
                                        "current": ctx.readiness_mode,
                                        "reason": pre_final_reason,
                                        "stage": "pre_final_readiness",
                                    },
                                )
                                if previous_mode != ctx.readiness_mode:
                                    LOGGER.info(
                                        "STARTUP_MODE_TRANSITION from=%s to=%s reason=%s",
                                        previous_mode,
                                        ctx.readiness_mode,
                                        "active_basket_ready"
                                        if ctx.readiness_mode == "EVALUATION_READY"
                                        else "warmup_guard",
                                    )
                                LOGGER.info(
                                    "LIVE_ORDER_ARM_BLOCKED reason=%s ce_bars=%s pe_bars=%s required=%s indicators_ready=%s quote_ready=%s",
                                    ",".join(data_warmup_reasons) if data_warmup_reasons else "unknown",
                                    hydrated_ce,
                                    hydrated_pe,
                                    option_exec_min_bars,
                                    bool(ctx.evaluation_ready),
                                    bool(option_ticks_ready),
                                )
                                if "startup_pipeline_incomplete" in data_warmup_reasons:
                                    LOGGER.error(
                                        "LIVE_TRADING_BLOCKED reason=startup_pipeline_incomplete missing=%s",
                                        missing_hard,
                                        extra={
                                            "event": "LIVE_TRADING_BLOCKED",
                                            "reason": "startup_pipeline_incomplete",
                                            "missing": missing_hard,
                                            "hard_ready": hard_ready,
                                            "spot_ready": spot_ready,
                                        },
                                    )
                                if "market_data_proof_unavailable" in data_warmup_reasons:
                                    LOGGER.error(
                                        "LIVE_TRADING_BLOCKED reason=market_data_proof_unavailable",
                                        extra={
                                            "event": "LIVE_TRADING_BLOCKED",
                                            "reason": "market_data_proof_unavailable",
                                            "quote_available": bool(quote_available),
                                            "ws_quote_proof": bool(ws_quote_proof),
                                        },
                                    )
                                if "market_closed" in data_warmup_reasons:
                                    LOGGER.info(
                                        "LIVE_TRADING_BLOCKED reason=market_closed",
                                        extra={
                                            "event": "LIVE_TRADING_BLOCKED",
                                            "reason": "market_closed",
                                        },
                                    )
                            data_warmup_reason: str | None = (
                                ",".join(data_warmup_reasons) if data_warmup_reasons else None
                            )
                            ctx.live_block_reason = data_warmup_reason
                            runtime_reason = "LIVE" if (live_mode and armed) else ctx.live_block_reason
                            if ctx.strategy_runner is not None and hasattr(
                                ctx.strategy_runner, "set_runtime_readiness"
                            ):
                                _basket = cast(dict[str, object], getattr(ctx, "active_trading_universe", {}) or {})
                                _selected_ce = getattr(ctx, "selected_ce", None) or _basket.get("selected_ce") or _basket.get("atm_ce")
                                _selected_pe = getattr(ctx, "selected_pe", None) or _basket.get("selected_pe") or _basket.get("atm_pe")
                                _atm_strike = _basket.get("atm_strike")
                                _option_symbols = list(_basket.get("option_symbols") or _basket.get("symbols") or [])
                                if hasattr(ctx.strategy_runner, "set_active_option_context"):
                                    ctx.strategy_runner.set_active_option_context(selected_ce=cast(str | None, _selected_ce), selected_pe=cast(str | None, _selected_pe), atm_strike=_atm_strike, option_symbols=_option_symbols)
                                ctx.strategy_runner.set_runtime_readiness(
                                    data_hard_ready=bool(ctx.data_hard_ready),
                                    evaluation_ready=bool(ctx.evaluation_ready),
                                    live_orders_armed=bool(ctx.live_orders_armed),
                                    reason=runtime_reason,
                                    selected_ce=cast(str | None, _selected_ce),
                                    selected_pe=cast(str | None, _selected_pe),
                                    atm_strike=_atm_strike,
                                    option_symbols=_option_symbols,
                                    execution_ready_by_symbol=dict(getattr(ctx, "execution_ready_by_symbol", {}) or {}),
                                )
                                LOGGER.info(
                                    "RUNTIME_READINESS_PUSHED data_hard_ready=%s evaluation_ready=%s live_orders_armed=%s reason=%s",
                                    bool(ctx.data_hard_ready),
                                    bool(ctx.evaluation_ready),
                                    bool(ctx.live_orders_armed),
                                    runtime_reason,
                                    extra={
                                        "event": "RUNTIME_READINESS_PUSHED",
                                        "data_hard_ready": bool(ctx.data_hard_ready),
                                        "evaluation_ready": bool(ctx.evaluation_ready),
                                        "live_orders_armed": bool(ctx.live_orders_armed),
                                        "reason": runtime_reason,
                                    },
                                )
                            ctx.market_session_state = session_state_str
                            ctx.quote_api_available = quote_available
                            ctx.quote_api_error = quote_error
                            if data_warmup_reason:
                                LOGGER.info(
                                    "DATA_WARMUP reason=%s",
                                    data_warmup_reason,
                                    extra={
                                        "event": "DATA_WARMUP",
                                        "reason": data_warmup_reason,
                                    },
                                )
                            LOGGER.info(
                                "Startup | configured_mode=%s | effective_mode=%s | live_orders_armed=%s | trading_ready=%s",
                                configured_runtime_mode,
                                ctx.readiness_mode,
                                bool(getattr(ctx, "live_orders_armed", False)),
                                bool(getattr(ctx, "trading_ready", False)),
                            )
                        except Exception as ready_exc:
                            LOGGER.critical(
                                "Startup readiness gate failed: %s",
                                ready_exc,
                                exc_info=ready_exc,
                            )
                            ctx.live_orders_armed = False
                            ctx.trading_ready = False
                            ctx.live_block_reason = "startup_pipeline_incomplete:" + ",".join(missing_hard if "missing_hard" in locals() else [])
                            ctx.degraded_mode = True
                            LOGGER.error(
                                "STARTUP_PIPELINE_INCOMPLETE_CONTINUING mode=%s missing_hard=%s missing_soft=%s",
                                configured_runtime_mode,
                                (missing_hard if "missing_hard" in locals() else []),
                                (missing_soft if "missing_soft" in locals() else []),
                            )
                    if not _data_ready(ctx.market_data_manager):
                        LOGGER.debug("startup_tick_gate: waiting_for_live_ticks (expected at boot)")


                if ctx.telegram_bot:
                    LOGGER.info("🚀 Starting Telegram Bot (Polling Mode)...")
                    await ctx.telegram_bot.start()
                    LOGGER.info("✅ Telegram Bot polling active — commands now live.")

                # ✅ FIX: Mark indicators as warmed up now that hydration is complete
                if ctx.indicator_engine and hasattr(ctx.indicator_engine, "atr_provider"):
                    atr_prov = ctx.indicator_engine.atr_provider
                    if hasattr(atr_prov, "mark_warmed_up"):
                        atr_prov.mark_warmed_up()
                indicator_ready = _compute_indicator_readiness(ctx)
                if ctx.data_hub is not None:
                    ctx.data_hub.indicators_ready = indicator_ready
                if ctx.market_regime_manager is not None:
                    ctx.market_regime_manager.indicators_ready = indicator_ready
                LOGGER.info(
                    "INDICATOR_READINESS_SYNC ready=%s",
                    indicator_ready,
                    extra={"event": "INDICATOR_READINESS_SYNC", "ready": indicator_ready},
                )

                ctx.subsystems_started = True
                LOGGER.info("✅ All subsystems started.")
        except Exception as e:
            LOGGER.critical(f"Subsystem start failed: {e}")

    # ---------------------------------------------------------
    # 5. Kill switch + reconciliation (UNCHANGED)
    # ---------------------------------------------------------
    if broker_ready:
        try:
            orders = await asyncio.to_thread(
                _run_sync_locked, ctx.broker_client.get_orders
            )
            for o in orders:
                if o.get("status") == "OPEN":
                    await asyncio.to_thread(
                        ctx.broker_client.cancel_order, o.get("order_id")
                    )
            LOGGER.info("✅ Zombie orders cleared.")

            # ── PHASE 8: Ghost-position reconciliation ──────────────────────
            try:
                await reconcile_with_broker(
                    broker_client=ctx.broker_client,
                    bracket_manager=ctx.bracket_manager,
                    order_manager=ctx.order_manager,
                    logger=LOGGER,
                )
            except Exception as reconcile_exc:
                LOGGER.error("reconcile_with_broker failed (non-fatal): %s", reconcile_exc)
            # ────────────────────────────────────────────────────────────────

            async def _sync_loop():
                from nifty_scalper_bot.utils.market_hours import is_market_open

                while True:
                    try:
                        if is_market_open():
                            await _reconcile_state(ctx)
                        else:
                            _inner = getattr(
                                ctx.broker_client,
                                "_broker",
                                getattr(ctx.broker_client, "client", ctx.broker_client),
                            )
                            reset_fn = getattr(_inner, "_reset_transient_state", None)
                            if callable(reset_fn):
                                reset_fn()
                    except Exception as e:
                        LOGGER.exception(
                            "[CRITICAL] unhandled exception", exc_info=True
                        )
                        raise
                    from nifty_scalper_bot.utils.market_hours import (
                        is_market_open as _imo,
                    )

                    await asyncio.sleep(15 if _imo() else 120)

            asyncio.create_task(_sync_loop())
            asyncio.create_task(_live_readiness_rearm_loop(ctx))

        except Exception as e:
            LOGGER.error(f"Post-start tasks failed: {e}")

    # ---------------------------------------------------------
    # 6. Greeks monitoring (UNCHANGED, SAFE)
    # ---------------------------------------------------------
    if ctx.strategy_runner and hasattr(
        ctx.strategy_runner, "calculate_portfolio_greeks"
    ):
        import threading

        runner = ctx.strategy_runner

        def _log_greeks_periodically():
            time_module.sleep(60)
            stop_event = getattr(ctx, "shutdown_event", None)

            while True:
                if stop_event and stop_event.is_set():
                    break
                if not threading.main_thread().is_alive():
                    break
                try:
                    greeks = runner.calculate_portfolio_greeks()
                    if (
                        abs(greeks.get("net_delta", 0)) > 1.0
                        or greeks.get("net_theta", 0) < -1.0
                    ):
                        LOGGER.info(
                            f"📊 Greeks: Delta={greeks['net_delta']:.1f} "
                            f"Theta={greeks['net_theta']:.1f}/day",
                            extra={"event": "greeks_monitor"},
                        )
                    for _ in range(300):
                        if stop_event and stop_event.is_set():
                            break
                        time_module.sleep(1)
                except Exception as exc:
                    LOGGER.debug(f"Greeks monitor error: {exc}")
                    time_module.sleep(60)

        threading.Thread(target=_log_greeks_periodically, daemon=True).start()
        LOGGER.info("✅ Portfolio Greeks monitoring enabled")

    await _notify(
        "BOT_STARTED", {"mode": "LIVE" if not ctx.shadow_mode_enabled else "SHADOW"}
    )
    # Send rich HTML startup message via the full TelegramBot (if polling started)
    if ctx.telegram_bot is not None and ctx.telegram_bot._app is not None:
        try:
            mode_icon = "🔴" if not ctx.shadow_mode_enabled else "🟡"
            mode_text = (
                "LIVE TRADING" if not ctx.shadow_mode_enabled else "PAPER / SHADOW"
            )
            from datetime import datetime as _dt

            startup_html = (
                f"<b>{mode_icon} Nifty Scalper Bot Online</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🕒 <b>Time:</b> <code>{_dt.now().strftime('%Y-%m-%d %H:%M:%S')}</code>\n"
                f"⚙️ <b>Mode:</b> <code>{mode_text}</code>\n"
                f"📡 <b>Commands:</b> <code>/help • /status • /positions</code>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"<i>Send /help for all 80+ commands.</i>"
            )
            await ctx.telegram_bot._app.bot.send_message(
                chat_id=ctx.telegram_bot.deps.chat_id,
                text=startup_html,
                parse_mode="HTML",
            )
        except Exception as _tg_exc:
            LOGGER.debug("Startup Telegram message failed: %s", _tg_exc)

    # ----------------------------------------------------------------
    # ✅ FIX: Enable Persistence & Background Services
    # ----------------------------------------------------------------
    LOGGER.info("⚙️ Finalizing Startup: Restoring State & Services...")

    # 1. Restore Brackets from Disk (Critical for Restarts)
    if ctx.bracket_manager:
        try:
            # Run in thread to avoid blocking the event loop during file I/O
            await asyncio.to_thread(ctx.bracket_manager.load_state)
            stats = ctx.bracket_manager.get_stats()
            LOGGER.info(f"♻️ Restored virtual brackets: {stats}")
        except Exception as e:
            LOGGER.error(f"Failed to restore brackets: {e}")

    # 2. Start the ATR Feed Task (Fire and Forget)
    loop = asyncio.get_running_loop()
    loop.create_task(_run_atr_feed_task(ctx))

    # 3. Reconcile Open Orders (Sync with Broker)
    if ctx.order_manager:
        LOGGER.info("🔍 Running Startup Reconciliation...")
        try:
            await asyncio.to_thread(ctx.order_manager.reconcile_open_orders)
        except Exception as e:
            LOGGER.error(f"Reconciliation failed: {e}")

    # ----------------------------------------------------------------
    # ✅ FIX: Wire DataHub -> Bracket Manager (Corrected Attribute Name)
    # ----------------------------------------------------------------
    # CHANGE: ctx.market_data -> ctx.market_data_manager
    if ctx.bracket_manager and ctx.market_data_manager:
        try:
            # 1. Define the tick handler using the fully loaded 'ctx'
            def _feed_ticks_to_bracket_safe(sym, tick):
                # Ensure we have LTP
                ltp = tick.get("ltp")
                if ltp and ctx.bracket_manager:
                    ctx.bracket_manager.on_tick(sym, ltp)

            # 2. Subscribe to the DataHub
            # CHANGE: ctx.market_data -> ctx.market_data_manager
            if (
                hasattr(ctx.market_data_manager, "data_hub")
                and ctx.market_data_manager.data_hub
            ):
                ctx.market_data_manager.data_hub.subscribe(
                    "bracket_feed", _feed_ticks_to_bracket_safe
                )
                LOGGER.info("✅ Wired DataHub ticks to BracketManager")

        except Exception as e:
            LOGGER.error(f"Failed to wire bracket ticks: {e}")

    if ctx.bracket_manager:
        try:
            now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
            eod_target_ist = now_ist.replace(hour=15, minute=24, second=0, microsecond=0)
            if now_ist >= eod_target_ist:
                eod_target_ist = eod_target_ist + timedelta(days=1)
            seconds_to_15h24 = max(
                0.0, (eod_target_ist - now_ist).total_seconds()
            )
            loop.call_later(seconds_to_15h24, ctx.bracket_manager.eod_flatten_all)
            LOGGER.info(
                "EOD bracket flatten scheduled for %s IST (in %.1fs)",
                eod_target_ist.isoformat(),
                seconds_to_15h24,
            )
        except Exception as e:
            LOGGER.error("Failed to schedule EOD flatten: %s", e)

    runner_running = False
    try:
        runner_status = ctx.strategy_runner.get_status() if ctx.strategy_runner else {}
        runner_running = bool(runner_status.get("running"))
    except Exception:
        runner_running = bool(getattr(ctx.strategy_runner, "_running", False)) if ctx.strategy_runner else False
    mode = str(getattr(ctx, "effective_mode", None) or getattr(ctx.settings, "execution_mode", None) or os.getenv("EXECUTION_MODE", "PAPER")).upper()
    paper_eval_ready = mode in {"PAPER", "SHADOW"} and bool(getattr(ctx, "data_observation_ready", False)) and runner_running
    live_ready = mode == "LIVE" and bool(getattr(ctx, "trading_ready", False)) and bool(getattr(ctx, "live_orders_armed", False)) and runner_running
    if paper_eval_ready:
        LOGGER.info("STARTUP_COMPLETE_OBSERVATION_READY mode=%s runner_running=%s live_orders_armed=%s", mode, runner_running, bool(getattr(ctx, "live_orders_armed", False)), extra={"event":"STARTUP_COMPLETE_OBSERVATION_READY","mode":mode,"runner_running":runner_running,"live_orders_armed":bool(getattr(ctx, "live_orders_armed", False))})
    elif live_ready:
        LOGGER.info("STARTUP_COMPLETE_LIVE_READY mode=%s runner_running=%s", mode, runner_running, extra={"event":"STARTUP_COMPLETE_LIVE_READY","mode":mode,"runner_running":runner_running})
    else:
        if mode == "LIVE":
            LOGGER.warning("STARTUP_DEGRADED runner_running=%s data_observation_ready=%s trading_ready=%s live_orders_armed=%s", runner_running, bool(getattr(ctx, "data_observation_ready", False)), bool(getattr(ctx, "trading_ready", False)), bool(getattr(ctx, "live_orders_armed", False)), extra={"event":"STARTUP_DEGRADED","runner_running":runner_running,"data_observation_ready":bool(getattr(ctx, "data_observation_ready", False)),"trading_ready":bool(getattr(ctx, "trading_ready", False)),"live_orders_armed":bool(getattr(ctx, "live_orders_armed", False))})
        elif bool(getattr(ctx, "data_observation_ready", False)):
            LOGGER.info("DATA_PIPELINE_OBSERVATION_READY mode=%s runner_running=%s", mode, runner_running, extra={"event":"DATA_PIPELINE_OBSERVATION_READY","mode":mode,"runner_running":runner_running})


async def shutdown_sequence(ctx: BotContext, *, reason: str = "shutdown") -> None:
    """Best-effort, idempotent shutdown. Must never raise during FastAPI lifespan."""
    LOGGER.info("Shutting down bot... reason=%s", reason)
    async def _call_component(name: str, obj: Any, method_names: tuple[str, ...]) -> None:
        if obj is None:
            return
        for method_name in method_names:
            method = getattr(obj, method_name, None)
            if not callable(method):
                continue
            try:
                await _maybe_await(method())
                LOGGER.debug("SHUTDOWN_COMPONENT_OK component=%s method=%s", name, method_name)
                return
            except Exception as exc:
                LOGGER.warning("SHUTDOWN_COMPONENT_FAILED component=%s method=%s error=%r", name, method_name, exc)
                return
    with suppress(Exception):
        ctx.trading_ready = False
    with suppress(Exception):
        ctx.live_orders_armed = False
    with suppress(Exception):
        ctx.effective_mode = "SHUTDOWN"
    for task_name in ("instrument_refresh_task","deferred_basket_retry_task","monitor_task","heartbeat_task","reconcile_task","maintenance_task"):
        task = getattr(ctx, task_name, None)
        if task is not None:
            try:
                task.cancel()
                await _maybe_await(task)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                LOGGER.warning("SHUTDOWN_TASK_FAILED task=%s error=%r", task_name, exc)
            with suppress(Exception):
                setattr(ctx, task_name, None)
    runner = getattr(ctx, "strategy_runner", None)
    order_manager = getattr(ctx, "order_manager", None)
    bracket_manager = getattr(ctx, "bracket_manager", None)
    position_manager = getattr(ctx, "position_manager", None)
    market_data_manager = getattr(ctx, "market_data_manager", None) or getattr(ctx, "mdm", None)
    stream_supervisor = getattr(ctx, "stream_supervisor", None)
    streamer = getattr(ctx, "streamer", None)
    message_bus = getattr(ctx, "message_bus", None)
    data_hub = getattr(ctx, "data_hub", None)
    persistent_state = getattr(ctx, "persistent_state", None)
    instrument_db = getattr(ctx, "instrument_db", None)
    state_tracker = getattr(ctx, "state_tracker", None)
    trade_journal = getattr(ctx, "trade_journal", None)
    telegram = getattr(ctx, "telegram_controller", None) or getattr(ctx, "telegram", None)
    proc = getattr(ctx, "proc", None) or getattr(ctx, "process", None)
    if runner is not None:
        with suppress(Exception):
            runner.pause_trading()
    try:
        if getattr(getattr(ctx, "config", None), "close_positions_on_shutdown", False):
            _close_all_positions(ctx, reason=reason)
    except Exception as exc:
        LOGGER.warning("SHUTDOWN_CLOSE_POSITIONS_FAILED error=%r", exc)
    await _call_component("strategy_runner", runner, ("stop", "shutdown", "close"))
    await _call_component("bracket_manager", bracket_manager, ("stop", "shutdown", "close"))
    await _call_component("order_manager", order_manager, ("stop_monitoring", "stop", "shutdown", "close"))
    await _call_component("stream_supervisor", stream_supervisor, ("stop", "shutdown", "close"))
    await _call_component("streamer", streamer, ("stop", "shutdown", "close"))
    await _call_component("market_data_manager", market_data_manager, ("stop", "shutdown", "close"))
    await _call_component("message_bus", message_bus, ("stop", "shutdown", "close"))
    try:
        if data_hub is not None and hasattr(data_hub, "checkpoint"):
            await _maybe_await(data_hub.checkpoint())
    except Exception as exc:
        LOGGER.warning("SHUTDOWN_DATAHUB_CHECKPOINT_FAILED error=%r", exc)
    await _call_component("data_hub", data_hub, ("shutdown", "close"))
    try:
        if persistent_state is not None and hasattr(persistent_state, "flush"):
            await _maybe_await(persistent_state.flush())
    except Exception as exc:
        LOGGER.warning("SHUTDOWN_PERSISTENT_FLUSH_FAILED error=%r", exc)
    await _call_component("persistent_state", persistent_state, ("close",))
    await _call_component("instrument_db", instrument_db, ("close",))
    await _call_component("state_tracker", state_tracker, ("close",))
    await _call_component("trade_journal", trade_journal, ("stop", "close"))
    await _call_component("telegram", telegram, ("stop", "shutdown", "close"))
    try:
        if proc is not None:
            if hasattr(proc, "terminate"):
                proc.terminate()
            if hasattr(proc, "wait"):
                result = proc.wait()
                if inspect.isawaitable(result):
                    await result
    except Exception as exc:
        LOGGER.warning("SHUTDOWN_PROC_FAILED error=%r", exc)
    LOGGER.info("Bot shutdown complete")


async def _reconcile_state(ctx: BotContext) -> None:
    """
    Syncs local state with Broker (Orders & Positions).
    Features: Non-Blocking Execution, Position Sync, and Auto-Guarding of Orphans.
    """

    def safe_sync_fetch() -> list[Mapping[str, Any]]:
        """Synchronize broker orders/positions under the global sync lock."""

        def _sync_operation() -> list[Mapping[str, Any]]:
            if ctx.order_manager:
                ctx.order_manager.reconcile_open_orders_with_broker()
            # ✅ FIX E: ctx.broker_client is RobustDataProvider whose get_positions()
            # without await returns a coroutine object, not positions.
            # Access the underlying sync ZerodhaKiteClient directly.
            _sync_broker = getattr(
                ctx.broker_client,
                "client",
                getattr(ctx.broker_client, "_broker", ctx.broker_client),
            )
            try:
                raw = _sync_broker.get_positions()
                # Guard: if the resolved broker method is async it returns an awaitable
                # (coroutine, Task, or Future) instead of positions.  Clean up the
                # awaitable correctly and fall back to an empty result.
                if inspect.isawaitable(raw):
                    LOGGER.error(
                        "get_positions() returned an awaitable in sync context – "
                        "broker client wrapping is incorrect. Falling back to empty list."
                    )
                    if asyncio.iscoroutine(raw):
                        raw.close()          # suppress ResourceWarning on coroutines
                    elif hasattr(raw, "cancel"):
                        raw.cancel()         # cancel Tasks / Futures
                    raw = {}
            except Exception as _pos_err:
                LOGGER.warning("Position fetch failed in reconcile: %s", _pos_err)
                raw = {}
            broker_positions: list[Mapping[str, Any]] = []
            if isinstance(raw, list):
                broker_positions = [p for p in raw if isinstance(p, Mapping)]
            elif isinstance(raw, Mapping):
                src = raw.get("net", raw)
                if isinstance(src, list):
                    broker_positions = [p for p in src if isinstance(p, Mapping)]
                elif isinstance(src, Mapping):
                    broker_positions = [src]
            if ctx.position_manager:
                ctx.position_manager.synchronize_with_broker(broker_positions)
            return broker_positions

        return cast(list[Mapping[str, Any]], _run_sync_locked(_sync_operation))

    # 1/2. SYNC ORDERS + POSITIONS & AUTO-GUARD ORPHANS
    if ctx.position_manager:
        try:
            broker_positions = await asyncio.to_thread(safe_sync_fetch)
            # A. Fetch Broker Positions (REQUIRED STEP)
            # Initialise bm here so the ghost-bracket cleanup below never hits NameError
            # even when ctx.order_manager or _bracket_manager is absent.
            bm = None
            # C. Auto-Guard Orphans (CRITICAL SAFETY LOGIC)
            if ctx.order_manager and ctx.order_manager._bracket_manager:
                om = ctx.order_manager
                bm = ctx.order_manager._bracket_manager

                from nifty_scalper_bot.data.data_hub import DataHub

                # Iterate through the FRESHLY synced open positions
                for pos in ctx.position_manager.get_open_positions():
                    if pos.quantity == 0:
                        continue

                    # 1. Normalize Symbol
                    raw_symbol = pos.symbol
                    norm_symbol = DataHub.normalize(raw_symbol) or raw_symbol

                    # 2. Check if this symbol is actively managed
                    is_managed = bm.is_symbol_managed(norm_symbol)

                    # 3. If NOT managed, it is an Orphan -> Guard it!
                    if not is_managed:
                        LOGGER.warning(
                            f"⚠️ ORPHAN DETECTED: {norm_symbol} (Qty: {pos.quantity}). Auto-Guarding...",
                            extra={"event": "orphan_detected", "symbol": norm_symbol},
                        )

                        avg_price = float(
                            getattr(pos, "average_price", 0.0)
                            or getattr(pos, "buy_price", 0.0)
                            or getattr(pos, "last_price", 0.0)
                            or 0.0
                        )

                        # Call the Master Guard Method
                        signed_qty = (
                            pos.quantity if pos.side == "LONG" else -pos.quantity
                        )
                        om.guard_orphan_position(
                            symbol=norm_symbol,
                            quantity=signed_qty,  # ✅ Now negative for SHORT
                            average_price=avg_price,
                            position_side=pos.side,
                        )

            # =================================================================
            # ✅ D. CLEANUP GHOST BRACKETS (Safety Cleanup)
            # =================================================================
            # If a Bracket exists but the Position is gone (Manual Exit), kill the Bracket.
            # This prevents the bot from opening unwanted positions if price hits old levels.
            if bm and ctx.position_manager:
                # 1. Get symbols that currently have active brackets
                # Accessing protected member safely for reconciliation
                if hasattr(bm, "_symbol_map"):
                    managed_symbols = set(bm._symbol_map.keys())

                    # 2. Get symbols that actually have open positions (Real Broker State)
                    real_positions = {
                        p.symbol
                        for p in ctx.position_manager.get_open_positions()
                        if p.quantity != 0
                    }

                    # 3. Identify Ghosts (Managed but no Position)
                    ghosts = managed_symbols - real_positions

                    for ghost_sym in ghosts:
                        # Double check if it actually has active brackets inside
                        if bm.is_symbol_managed(ghost_sym):
                            LOGGER.warning(
                                f"👻 GHOST BRACKET DETECTED: {ghost_sym} has protection but no Open Position. "
                                "Performing Safety Cleanup..."
                            )
                            # Force kill the bracket so it doesn't misfire and open a new trade
                            bm.manual_override_close(
                                ghost_sym, reason="State Reconciliation (Ghost)"
                            )

        except Exception as exc:
            LOGGER.error(f"Position Sync/Adoption Failed: {exc}", exc_info=True)

    _sync_data_hub_positions(getattr(ctx, "data_hub", None), ctx.position_manager)

    # 3. SITUATION REPORT
    if ctx.order_manager:
        ctx.order_manager._log_status_report()


def _close_all_positions(ctx: BotContext, *, reason: str) -> None:
    position_manager = _require_component(ctx.position_manager, "position_manager")
    market_data_manager = _require_component(
        ctx.market_data_manager,
        "market_data_manager",
    )
    for position in position_manager.get_all_positions():
        LOGGER.info("Closing position for %s", position.symbol)
        tick = market_data_manager.get_latest_tick(position.symbol)
        exit_price = position.current_price
        if tick is not None:
            maybe_price = tick.get("ltp") or tick.get("price")
            if isinstance(maybe_price, (int, float)) and maybe_price > 0:
                exit_price = float(maybe_price)
        position_manager.close_position(position.symbol, exit_price, reason)


def _alert_overnight_exposure(ctx: BotContext) -> None:
    """Emit an alert when open positions persist beyond the session.

    Args:
        ctx: Active bot context containing managers and configuration.

    Returns:
        None.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered overnight exposure check",
        extra={"event": "overnight_exposure_check"},
    )
    position_manager = _require_component(ctx.position_manager, "position_manager")
    runner = _require_component(ctx.strategy_runner, "strategy_runner")
    try:
        positions = position_manager.get_all_positions()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _alert_overnight_exposure positions",
            extra={"error": str(exc)},
            exc_info=True,
        )
        return
    if not positions:
        setattr(runner, "_overnight_alerted", False)
        return
    session_close = getattr(ctx.settings.regime, "session_close", None)
    if session_close is None:
        setattr(runner, "_overnight_alerted", False)
        return
    session_guard = ctx.session_guard
    tz = getattr(session_guard, "_tz", ZoneInfo("Asia/Kolkata"))
    if not isinstance(tz, ZoneInfo):
        tz = ZoneInfo("Asia/Kolkata")
    now_local = datetime.now(tz)
    close_today = datetime.combine(now_local.date(), session_close, tzinfo=tz)
    overnight_symbols: set[str] = set()
    exposure_details: list[dict[str, object]] = []
    for position in positions:
        symbol = getattr(position, "symbol", "UNKNOWN") or "UNKNOWN"
        entry_time = getattr(position, "entry_time", None)
        entry_local: datetime | None = None
        if isinstance(entry_time, datetime):
            coerced = entry_time
            if coerced.tzinfo is None:
                coerced = coerced.replace(tzinfo=timezone.utc)
            entry_local = coerced.astimezone(tz)
        is_overnight = False
        if entry_local is not None:
            if entry_local.date() < now_local.date():
                is_overnight = True
            elif now_local > close_today:
                is_overnight = True
        elif now_local > close_today:
            is_overnight = True
        if not is_overnight:
            continue
        overnight_symbols.add(symbol)
        if entry_local is not None:
            age_minutes = max((now_local - entry_local).total_seconds() / 60.0, 0.0)
            exposure_details.append(
                {
                    "symbol": symbol,
                    "entry_time": entry_local.isoformat(),
                    "age_minutes": round(age_minutes, 2),
                }
            )
        else:
            exposure_details.append(
                {
                    "symbol": symbol,
                    "entry_time": None,
                    "age_minutes": None,
                }
            )
    if not overnight_symbols:
        setattr(runner, "_overnight_alerted", False)
        return
    if getattr(runner, "_overnight_alerted", False):
        return
    setattr(runner, "_overnight_alerted", True)
    LOGGER.error(
        "Condition met: overnight_exposure_detected",
        extra={
            "event": "overnight_exposure_detected",
            "symbols": sorted(overnight_symbols),
            "details": exposure_details,
        },
    )


def _health_check(ctx: BotContext) -> None:
    strategy_runner = _require_component(ctx.strategy_runner, "strategy_runner")
    status: Mapping[str, Any] = strategy_runner.get_status()
    if not bool(status.get("running")):
        state = get_market_state()
        now_mono = time_module.monotonic()
        started_mono = float(getattr(ctx, "started_mono", now_mono) or now_mono)
        startup_age_s = max(0.0, now_mono - started_mono)
        configured_mode = str(
            os.getenv("EXECUTION_MODE", getattr(ctx, "effective_mode", "PAPER"))
        ).upper()
        effective_mode = str(
            getattr(ctx, "effective_mode", "")
            or getattr(ctx, "readiness_mode", "")
            or configured_mode
        ).upper()
        trading_ready = bool(getattr(ctx, "trading_ready", False))
        live_orders_armed = bool(getattr(ctx, "live_orders_armed", False))
        evaluation_expected = configured_mode in {"PAPER", "SHADOW", "LIVE"}
        live_orders_expected = configured_mode == "LIVE" and bool(ctx.settings.enable_live)
        expected_active = bool(
            evaluation_expected
            and state == MarketState.OPEN
            and trading_ready
            and (configured_mode in {"PAPER", "SHADOW"} or (live_orders_armed and live_orders_expected))
            and not ctx.shadow_mode_enabled
        )
        inactive_reason = "runner_task_not_started"
        if configured_mode in {"PAPER", "SHADOW"}:
            inactive_reason = "paper_runner_not_started"
            expected_active = bool(evaluation_expected and trading_ready)
        elif not ctx.settings.enable_live:
            inactive_reason = "live_orders_disabled"
            expected_active = False
        elif state != MarketState.OPEN:
            inactive_reason = "market_closed"
        elif effective_mode == "DATA_WARMUP":
            inactive_reason = "data_warmup"
        elif effective_mode != "LIVE":
            inactive_reason = "live_blocked"
        elif startup_age_s < 120.0:
            inactive_reason = "startup_grace"
        elif status.get("readiness_unmet") or not trading_ready or not live_orders_armed:
            inactive_reason = "readiness_unmet"
        elif status.get("startup_degraded"):
            inactive_reason = "degraded_startup"
        elif status.get("basket_build_failed"):
            inactive_reason = "basket_build_failure"
        elif status.get("loop_error"):
            inactive_reason = "runner_loop_exception"
        log_throttled(
            LOGGER,
            key="strategy_runner_inactive_health",
            msg=f"Strategy runner is not active ({inactive_reason})",
            level=logging.WARNING if expected_active else logging.INFO,
            interval_sec=60.0,
            extra={
                "event": "strategy_runner_inactive_health",
                "expected_active": expected_active,
                "market_state": state.value if hasattr(state, "value") else str(state),
                "reason": inactive_reason,
                "startup_age_s": round(startup_age_s, 2),
                "effective_mode": effective_mode,
                "trading_ready": trading_ready,
                "live_orders_armed": live_orders_armed,
            },
        )


def _must_ok(condition: bool, message: str) -> None:
    """Raise :class:`ConfigurationError` when *condition* is falsy."""

    if not condition:
        raise ConfigurationError(message)


class NiftyScalperApp:
    """High level orchestrator exposing lifecycle hooks for the trading stack."""

    def __init__(
        self, config: AppConfig | None = None, settings: Settings | None = None
    ) -> None:
        base_settings = settings or get_settings()
        if config is not None:
            base_settings = replace(base_settings, app=config)
        self._settings = base_settings
        self._config = base_settings.app
        setup_logging(self._config.logging.level)
        setup_structured_logging(self._config.logging.level)
        validation_errors = validate_execution_config()
        if validation_errors:
            joined = "; ".join(validation_errors)
            LOGGER.error(
                "Failure in NiftyScalperApp.__init__: config validation failed",
                extra={
                    "event": "config_validation_failure",
                    "errors": validation_errors,
                },
            )
            raise ConfigurationError(f"Execution configuration invalid: {joined}")
        self._ctx = initialize_components(self._settings)
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._health_task: asyncio.Task[None] | None = None
        self._self_test_task: asyncio.Task[None] | None = None
        self._telegram_task: asyncio.Task[None] | None = None
        self._telegram_application_started = False
        self._self_test_interval = 600.0
        # Edge-trigger runtime self-check state to avoid repeated failure floods.
        self._last_self_check_ok: bool | None = None
        self._self_test_failure_count = 0

    @property
    def config(self) -> AppConfig:
        """Return the loaded configuration."""

        return self._config

    @property
    def settings(self) -> Settings:
        """Return runtime settings including live trading toggles."""

        return self._settings

    @property
    def health_app(self) -> FastAPI:
        """Expose FastAPI app serving /health and /metrics."""

        return _require_component(self._ctx.health_app, "health_app")

    @property
    def ws_manager(self) -> WebSocketManager | None:
        """Return the websocket manager if configured."""

        return self._ctx.websocket_manager

    @property
    def positions(self) -> PositionManager:
        """Return the position manager."""

        return _require_component(self._ctx.position_manager, "position_manager")

    def status_string(self) -> str:
        """Return a human friendly multi-line status string."""

        strategy_runner = _require_component(
            self._ctx.strategy_runner,
            "strategy_runner",
        )
        status = strategy_runner.get_status()
        running = "running" if status.get("running") else "stopped"
        if status.get("trading_paused"):
            running += " (paused)"
        active_symbols = status.get("active_symbols") or []
        symbol_line = ", ".join(active_symbols) if active_symbols else "none"

        position_manager = _require_component(
            self._ctx.position_manager,
            "position_manager",
        )
        positions = position_manager.get_all_positions()
        if not positions:
            position_lines = ["Positions: none"]
        else:
            summary = [
                (
                    f"{pos.symbol} {pos.side} qty={pos.quantity} "
                    f"pnl={pos.unrealized_pnl:.2f}"
                )
                for pos in positions[:5]
            ]
            more = len(positions) - len(summary)
            if more > 0:
                summary.append(f"(+{more} more)")
            position_lines = ["Positions:"] + summary

        lines = [
            "Nifty Scalper Bot",
            f"Core status: {running}",
            f"Active symbols: {symbol_line}",
            *position_lines,
        ]
        return "\n".join(lines)

    def simulate_disconnect(self) -> None:
        """Test helper forcing websocket reconnect."""

        streamer = getattr(self._ctx, "streamer", None)
        simulate = getattr(streamer, "simulate_disconnect", None)
        if callable(simulate):  # pragma: no branch - optional hook
            simulate()

    def is_connected(self) -> bool:
        """Return websocket connectivity state."""

        streamer = getattr(self._ctx, "streamer", None)
        if streamer is None:
            return True
        is_connected = getattr(streamer, "is_connected", None)
        if callable(is_connected):
            try:
                return bool(is_connected())
            except Exception:  # pragma: no cover - defensive
                return False
        return True

    def backlog_size(self) -> int:
        """Return queued tick backlog size."""

        streamer = getattr(self._ctx, "streamer", None)
        if streamer is None:
            return 0
        backlog_fn = getattr(streamer, "backlog_size", None)
        if callable(backlog_fn):
            try:
                return int(backlog_fn())
            except Exception:  # pragma: no cover - defensive
                return 0
        tracked = getattr(streamer, "tracked_tokens", None)
        if callable(tracked):
            try:
                return len(tracked())
            except Exception:  # pragma: no cover - defensive
                return 0
        return 0

    def rejection_count(self) -> int:
        """Return accumulated order rejection count."""

        safe_order_manager = _require_component(
            self._ctx.safe_order_manager,
            "safe_order_manager",
        )
        return safe_order_manager.rejection_count()

    async def start(self) -> None:
        """Start the trading stack and background health monitoring."""

        if self._running:
            LOGGER.info("NiftyScalperApp.start() ignored; already running")
            return
        await startup_sequence(self._ctx)
        self._running = True
        self._shutdown_event.clear()
        self._health_task = asyncio.create_task(
            self._health_loop(), name="core-health-monitor"
        )
        if self._ctx.selfchecker is not None:
            self._self_test_task = asyncio.create_task(
                self._self_test_loop(),
                name="core-runtime-selftest",
            )
        # ------------------------------------------------------------------
        # Telegram Service Initialization (Webhook vs Polling)
        # ------------------------------------------------------------------
        application = self._ctx.telegram_application
        controller = _HTTP_CONTROLLER

        # 1. Telegram App
        if application is not None:
            # Check if Webhook is actually configured
            if (
                self.settings.notifications.webhook_enabled
                and self.settings.notifications.public_base_url
            ):
                try:
                    await application.initialize()
                    await application.start()
                    if controller:
                        controller.notify_application_ready()
                    self._telegram_application_started = True
                    LOGGER.info("telegram_application_started (Webhook)")
                except Exception as exc:
                    LOGGER.exception("telegram_application_start_failed")

            elif self._ctx.telegram_bot:
                LOGGER.info(
                    "telegram_bot_start_skipped_in_app_start; already started in startup_sequence"
                )

    async def stop(self) -> None:
        """Stop the trading stack gracefully."""

        if not self._running:
            return
        self._shutdown_event.set()
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Health task failed during shutdown: %s",
                    exc,
                    extra={"event": "health_task_shutdown_error"},
                    exc_info=True,
                )
            self._health_task = None
        if self._self_test_task:
            self._self_test_task.cancel()
            try:
                await self._self_test_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Self-test task failed during shutdown: %s",
                    exc,
                    extra={"event": "self_test_task_shutdown_error"},
                    exc_info=True,
                )
            self._self_test_task = None
        if (
            self._ctx.telegram_application is not None
            and self._telegram_application_started
        ):
            controller = _HTTP_CONTROLLER
            if controller is not None:
                controller.notify_application_ready(ready=False)
            with suppress(Exception):
                await self._ctx.telegram_application.stop()
            with suppress(Exception):
                await self._ctx.telegram_application.shutdown()
            self._telegram_application_started = False
        
        # ✅ FIX: Properly call telegram_bot.stop() and cancel the task
        if self._ctx.telegram_bot is not None:
            await self._ctx.telegram_bot.stop()
            if self._telegram_task:
                self._telegram_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._telegram_task
                self._telegram_task = None
                
        await shutdown_sequence(self._ctx)
        self._running = False

    def close_all_positions(self, *, reason: str) -> None:
        """Close all known positions immediately."""

        _close_all_positions(self._ctx, reason=reason)

    async def _self_test_loop(self) -> None:
        """Execute periodic runtime self-checks and alert on failures.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        checker = self._ctx.selfchecker
        if checker is None:
            LOGGER.info(
                "Runtime self-test loop skipped: checker unavailable.",
                extra={"event": "runtime_self_test_missing"},
            )
            return
        interval = getattr(checker, "interval_seconds", self._self_test_interval)
        try:
            interval_value = max(float(interval), 60.0)
        except Exception:  # pragma: no cover - defensive parsing
            interval_value = self._self_test_interval
        LOGGER.debug(
            "Entered runtime self-test loop",
            extra={
                "event": "runtime_self_test_loop_enter",
                "interval": interval_value,
            },
        )
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=interval_value
                )
            except asyncio.TimeoutError:
                LOGGER.debug(
                    "Executing runtime self-test iteration",
                    extra={"event": "runtime_self_test_iteration"},
                )
                try:
                    results = checker.run_full_check()
                except Exception as exc:  # noqa: BLE001 - defensive
                    LOGGER.error(
                        "Failure in runtime self-test execution: %s",
                        exc,
                        extra={"event": "runtime_self_test_execute_error"},
                        exc_info=exc,
                    )
                    continue
                current_ok = all(bool(result.get("ok")) for result in results.values())
                previous_ok = self._last_self_check_ok
                self._last_self_check_ok = current_ok
                if previous_ok is not None and current_ok == previous_ok:
                    continue
                if current_ok:
                    self._self_test_failure_count = 0
                    LOGGER.info(
                        "Condition met: runtime self-test recovered",
                        extra={"event": "runtime_self_test_recovered"},
                    )
                    continue
                self._self_test_failure_count += 1
                if self._self_test_failure_count < 3:
                    LOGGER.debug(
                        "Condition met: runtime self-test transient failure",
                        extra={
                            "event": "runtime_self_test_transient_failure",
                            "failure_count": self._self_test_failure_count,
                        },
                    )
                    continue
                for name, result in results.items():
                    if not bool(result.get("ok")):
                        detail = str(result.get("detail", "unknown"))
                        meta_obj = result.get("meta")
                        LOGGER.error(
                            "Silent failure detected: %s check failed: %s",
                            name,
                            detail,
                            extra={
                                "event": "runtime_self_test_failure",
                                "check": name,
                                "detail": detail,
                                "meta": meta_obj,
                            },
                        )
                        meta_payload = (
                            meta_obj if isinstance(meta_obj, Mapping) else None
                        )
                        await self._send_self_test_alert(
                            name,
                            detail,
                            cast(Mapping[str, object] | None, meta_payload),
                        )
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - defensive loop guard
                LOGGER.error(
                    "Failure in runtime self-test loop: %s",
                    exc,
                    extra={"event": "runtime_self_test_loop_error"},
                    exc_info=exc,
                )
                await asyncio.sleep(5.0)
        LOGGER.debug(
            "Runtime self-test loop exiting",
            extra={"event": "runtime_self_test_loop_exit"},
        )

    async def _send_self_test_alert(
        self,
        check_name: str,
        detail: str,
        meta: Mapping[str, object] | None,
    ) -> None:
        """Send Telegram notifier alert for runtime self-test failures.

        Args:
            check_name: Identifier of the failing runtime check.
            detail: Description or reason for the failure.
            meta: Optional metadata describing the failure context.

        Returns:
            None.

        Raises:
            None.
        """

        notifier = self._ctx.telegram_notifier
        payload_meta: dict[str, object]
        if isinstance(meta, Mapping):
            payload_meta = dict(meta)
        else:
            payload_meta = {}
        payload = {
            "check": check_name,
            "detail": detail,
            "meta": payload_meta,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if notifier is None:
            LOGGER.info(
                "Runtime self-test alert skipped: notifier unavailable",
                extra={
                    "event": "runtime_self_test_alert_skipped",
                    "check": check_name,
                    "detail": detail,
                },
            )
            return
        try:
            await notifier.send_event("SILENT_FAILURE", payload)
        except Exception as exc:  # noqa: BLE001 - defensive notifier surface
            LOGGER.error(
                "Failure in runtime self-test alert send: %s",
                exc,
                extra={
                    "event": "runtime_self_test_alert_error",
                    "check": check_name,
                },
                exc_info=exc,
            )
        else:
            LOGGER.info(
                "Condition met: runtime self-test alert dispatched",
                extra={
                    "event": "runtime_self_test_alert_sent",
                    "check": check_name,
                },
            )

    async def _health_loop(self) -> None:
        interval = 30.0
        last_heavy = time_module.monotonic()
        heavy_interval = 30.0
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)
                break
            except asyncio.TimeoutError:
                now = time_module.monotonic()
                if now - last_heavy >= heavy_interval:
                    try:
                        _health_check(self._ctx)
                    except asyncio.CancelledError:
                        LOGGER.debug(
                            "Health loop cancelled",
                            extra={"event": "health_loop_cancelled"},
                        )
                        raise
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.error(
                            "Failure in health loop: %s",
                            exc,
                            extra={"event": "health_loop_error"},
                            exc_info=True,
                        )
                        await asyncio.sleep(5.0)
                    try:
                        await _reconcile_state(self._ctx)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning(
                            "Periodic state reconciliation failed",
                            extra={
                                "event": "state_reconcile_failed_periodic",
                                "error": str(exc),
                            },
                            exc_info=True,
                        )
                    try:
                        _alert_overnight_exposure(self._ctx)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning(
                            "Overnight exposure check failed",
                            extra={
                                "event": "overnight_exposure_check_failed",
                                "error": str(exc),
                            },
                            exc_info=True,
                        )
                    last_heavy = now
                continue
            except asyncio.CancelledError:
                LOGGER.debug(
                    "Health loop cancelled",
                    extra={"event": "health_loop_cancelled"},
                )
                raise


# ----------------------------------------------------------------
# ✅ NEW HELPER: Background ATR Feed
# ----------------------------------------------------------------
async def _run_atr_feed_task(ctx: BotContext) -> None:
    """Periodically pushes fresh ATR data to the Bracket Manager."""
    LOGGER.info("🚀 Starting ATR Feed to BracketManager...")
    while True:
        try:
            # [FIX] Safely route through the runner to access the indicator engine
            runner = getattr(ctx, "runner", getattr(ctx, "strategy_runner", None))
            if ctx.bracket_manager and runner and hasattr(runner, "_indicator_engine"):
                # Access protected member safely for internal core logic
                active_symbols = list(ctx.bracket_manager._symbol_map.keys())
                for symbol in active_symbols:
                    # Safely fetch ATR using the runner's engine
                    atr = runner._indicator_engine.compute_atr(symbol, period=14)
                    if atr and atr > 0:
                        ctx.bracket_manager.update_market_stats(symbol, atr=atr)
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(5)


__all__ = [
    "NiftyScalperApp",
    "initialize_components",
    "startup_sequence",
    "shutdown_sequence",
    "get_http_app",
    "get_telegram_notifier",
    "get_nifty_expiry",
]
