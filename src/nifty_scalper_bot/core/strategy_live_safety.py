"""Live-safety guard for StrategyManager.

Real LIVE mode fails closed on cold history, explicit DataHub indicator
unreadiness, approved-signal filter bypass, and unsafe OrderFlow candidates.
Deterministic signal identity is generated at the live-safety boundary so
approved signals are idempotent without depending on every strategy to stamp it.
"""

from __future__ import annotations

import os
import time
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

from nifty_scalper_bot.config.defaults import QUOTE_STALE_THRESHOLD_MS
from nifty_scalper_bot.execution.readiness import (
    HistoryReadinessPolicy,
    resolve_max_quote_age_seconds,
)
from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.utils.logging import get_logger, log_throttled
from nifty_scalper_bot.utils.symbols import normalize_symbol

LOG = get_logger(__name__)
TRUTHY = {"1", "true", "yes", "y", "on", "live"}
IDENTITY_KEYS = (
    "signal_id",
    "deterministic_signal_id",
    "idempotency_key",
    "bar_timestamp",
    "signal_timestamp",
    "timestamp",
)


def _env_true(name: str) -> bool:
    return str(os.getenv(name, "") or "").strip().lower() in TRUTHY


def _is_live(manager: Any) -> bool:
    checker = getattr(manager, "_is_live_mode", None)
    if callable(checker):
        with suppress(Exception):
            return bool(checker())
    mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
    live = _env_true("ENABLE_LIVE") or _env_true("ENABLE_LIVE_TRADING")
    paper_shadow = (
        _env_true("PAPER_MODE")
        or _env_true("PAPER__ENABLED")
        or _env_true("SHADOW_MODE")
    )
    return mode == "LIVE" and live and not paper_shadow


def _metadata_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in TRUTHY


def _indicator_bars_available(manager: Any, symbol: str) -> int:
    getter = getattr(getattr(manager, "_indicator_engine", None), "get_history", None)
    if not callable(getter):
        return 0
    with suppress(Exception):
        return len(getter(symbol) or [])
    return 0


def _canonical_bars(manager: Any, symbol: str) -> list[Any]:
    authoritative_seen = False
    for source in (_market_data_manager(manager), getattr(manager, "_data_hub", None)):
        if source is None:
            continue
        getter = getattr(source, "get_ohlc_bars", None) or getattr(
            source, "get_ohlc", None
        )
        if callable(getter):
            authoritative_seen = True
            with suppress(Exception):
                bars = getter(symbol)
                return list(bars or [])
    if authoritative_seen:
        return []
    getter = getattr(getattr(manager, "_indicator_engine", None), "get_history", None)
    if callable(getter):
        with suppress(Exception):
            return list(getter(symbol) or [])
    return []


def _canonical_bars_available(manager: Any, symbol: str) -> int:
    return len(_canonical_bars(manager, symbol))


def _required_bars(manager: Any) -> int:
    policy = HistoryReadinessPolicy.from_env()
    raw = getattr(manager, "_required_candles", policy.option_eval_min_bars)
    with suppress(Exception):
        return max(1, int(raw or policy.option_eval_min_bars))
    return max(1, int(policy.option_eval_min_bars))


def _live_tick_max_age_seconds() -> float:
    return resolve_max_quote_age_seconds(
        "HYDRATION_LIVE_TICK_MAX_AGE_SECONDS",
        "HYDRATION_LIVE_TICK_MAX_AGE_MS",
        default_seconds=float(QUOTE_STALE_THRESHOLD_MS) / 1000.0,
    )


def _context_max_age_seconds() -> float:
    return _live_tick_max_age_seconds()


def _bar_interval_seconds(manager: Any) -> float:
    interval = getattr(manager, "_bar_interval_seconds", None)
    if interval in (None, ""):
        interval = getattr(manager, "bar_interval_seconds", None)
    try:
        interval_s = float(interval)
    except (TypeError, ValueError):
        interval_s = 60.0
    return max(1.0, interval_s)


def _clock_skew_tolerance_seconds() -> float:
    return 2.0


def _coerce_epoch_seconds(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        raw = float(value)
        if raw > 1_000_000_000_000:
            return raw / 1000.0
        return raw
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    text = str(value).strip()
    if not text:
        return None
    with suppress(Exception):
        return _coerce_epoch_seconds(float(text))
    with suppress(Exception):
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


def _bar_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _boolish_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    return str(value or "").strip().lower() in {
        "0",
        "false",
        "no",
        "n",
        "open",
        "forming",
    }


def _latest_bar_detail(manager: Any, symbol: str, *, now: float) -> dict[str, Any]:
    mdm = _market_data_manager(manager)
    latest_getter = getattr(mdm, "get_latest_closed_bar", None)
    if callable(latest_getter):
        try:
            latest = latest_getter(symbol)
        except Exception as exc:  # noqa: BLE001 - safety gate must fail closed
            return {
                "reason": "live_latest_closed_bar_unavailable",
                "error_type": type(exc).__name__,
            }
        history = [latest] if latest else []
    else:
        history = _canonical_bars(manager, symbol)
    if not history:
        return {"reason": "live_latest_closed_bar_missing"}
    interval_s = _bar_interval_seconds(manager)
    current_bucket_start = (now // interval_s) * interval_s
    expected_closed_start = current_bucket_start - interval_s
    latest_stale: dict[str, Any] | None = None
    for row in reversed(history):
        for key in ("closed", "is_closed", "complete", "is_complete", "final"):
            value = _bar_get(row, key, None)
            if _boolish_false(value):
                latest_stale = {
                    "reason": "live_latest_closed_bar_open",
                    "closed_field": key,
                }
                break
        else:
            epoch = None
            for key in ("timestamp", "timestamp_ms", "ts", "date", "time"):
                epoch = _coerce_epoch_seconds(_bar_get(row, key, None))
                if epoch is not None:
                    break
            if epoch is None:
                latest_stale = {"reason": "live_latest_closed_bar_missing"}
                continue
            if epoch > now + _clock_skew_tolerance_seconds():
                return {
                    "reason": "live_latest_closed_bar_future_timestamp",
                    "latest_bar_ts": epoch,
                }
            bucket_start = (epoch // interval_s) * interval_s
            if bucket_start >= current_bucket_start:
                latest_stale = {
                    "reason": "live_latest_closed_bar_open",
                    "latest_bar_ts": epoch,
                }
                continue
            if bucket_start == expected_closed_start:
                return {
                    "reason": "ready",
                    "latest_bar_ts": epoch,
                    "latest_bar_bucket_start": bucket_start,
                    "expected_latest_closed_start": expected_closed_start,
                    "interval_s": interval_s,
                }
            if bucket_start < expected_closed_start:
                latest_stale = {
                    "reason": "live_latest_closed_bar_stale",
                    "latest_bar_ts": epoch,
                    "latest_bar_bucket_start": bucket_start,
                    "expected_latest_closed_start": expected_closed_start,
                    "interval_s": interval_s,
                }
                break
    return latest_stale or {"reason": "live_latest_closed_bar_missing"}


def _latest_underlying_bar_fresh_block(
    manager: Any, symbol: str
) -> dict[str, Any] | None:
    detail = _latest_bar_detail(manager, symbol, now=time.time())
    if detail.get("reason") != "ready":
        return detail
    return None


def _market_data_manager(manager: Any) -> Any | None:
    direct = getattr(manager, "_market_data_manager", None)
    if direct is not None:
        return direct
    hub = getattr(manager, "_data_hub", None)
    return getattr(hub, "_mdm", None)


def _live_option_tick_block(manager: Any, symbol: str) -> dict[str, Any] | None:
    max_age = _live_tick_max_age_seconds()
    mdm = _market_data_manager(manager)
    if mdm is None:
        return {
            "reason": "live_market_data_manager_missing",
            "max_live_tick_age_s": max_age,
        }
    age_fn = getattr(mdm, "time_since_last_live_ws_tick", None)
    if not callable(age_fn):
        return {
            "reason": "live_option_tick_freshness_unavailable",
            "max_live_tick_age_s": max_age,
        }
    try:
        age = age_fn(symbol)
    except Exception as exc:  # noqa: BLE001
        return {
            "reason": "live_option_tick_freshness_unavailable",
            "error_type": type(exc).__name__,
            "max_live_tick_age_s": max_age,
        }
    if age is not None and float(age) <= max_age:
        return None
    if callable(getattr(mdm, "request_fallback_refresh", None)):
        with suppress(Exception):
            mdm.request_fallback_refresh(
                symbol, reason="strategy_live_option_tick_stale"
            )
    reason = "live_option_tick_missing" if age is None else "live_option_tick_stale"
    return {"reason": reason, "live_tick_age_s": age, "max_live_tick_age_s": max_age}


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _snapshot_age_seconds(snapshot: dict[str, Any]) -> float | None:
    for key in ("tick_age_s", "age_seconds", "context_age_seconds"):
        age = _num(snapshot.get(key))
        if age is not None:
            return age
    for key in ("tick_age_ms", "quote_age_ms"):
        age_ms = _num(snapshot.get(key))
        if age_ms is not None:
            return age_ms / 1000.0
    for key in ("context_timestamp_epoch", "timestamp"):
        epoch = _coerce_epoch_seconds(snapshot.get(key))
        if epoch is not None:
            return time.time() - epoch
    return None


def _context_fresh_block(manager: Any, symbol: str) -> dict[str, Any] | None:
    required = getattr(manager, "_strategy_required_context", None)
    if required is False:
        return None
    snapshots = getattr(manager, "_latest_context_snapshots", None)
    if not isinstance(snapshots, dict):
        return {"reason": "live_underlying_context_freshness_unknown"}
    max_age = _context_max_age_seconds()
    details: dict[str, Any] = {"max_context_age_s": max_age}
    for role, label in (("spot_context", "spot"), ("futures_context", "futures")):
        snap = snapshots.get(role)
        if not isinstance(snap, dict) or not snap:
            return {"reason": f"live_{label}_context_missing", **details}
        fresh_key = "spot_fresh" if label == "spot" else "fut_fresh"
        explicit = snap.get(fresh_key)
        if explicit is False or snap.get("context_fresh") is False:
            return {
                "reason": f"live_{label}_context_stale",
                fresh_key: False,
                **details,
            }
        age = _snapshot_age_seconds(snap)
        details[f"{label}_context_age_s"] = age
        if age is None:
            return {"reason": f"live_{label}_context_freshness_unknown", **details}
        if age < -_clock_skew_tolerance_seconds():
            return {"reason": f"live_{label}_context_future_timestamp", **details}
        if age > max_age:
            return {"reason": f"live_{label}_context_stale", **details}
    return None


def _basket_get(basket: Any, key: str, default: Any = None) -> Any:
    if isinstance(basket, dict):
        return basket.get(key, default)
    return getattr(basket, key, default)


def _active_basket(manager: Any) -> Any | None:
    hub = getattr(manager, "_data_hub", None)
    getter = getattr(hub, "get_active_contract_basket", None)
    if callable(getter):
        with suppress(Exception):
            basket = getter()
            if basket is not None:
                return basket
    mdm = _market_data_manager(manager)
    getter = getattr(mdm, "get_active_contract_basket", None)
    if callable(getter):
        with suppress(Exception):
            basket = getter()
            if basket is not None:
                return basket
    return getattr(manager, "active_contract_basket", None) or getattr(
        manager, "_active_contract_basket", None
    )


def _selected_contract_block(manager: Any, symbol: str) -> dict[str, Any] | None:
    basket = _active_basket(manager)
    if basket is None:
        return {"reason": "live_active_basket_missing"}
    selected = {
        normalize_symbol(
            _basket_get(basket, "selected_ce") or _basket_get(basket, "atm_ce")
        ),
        normalize_symbol(
            _basket_get(basket, "selected_pe") or _basket_get(basket, "atm_pe")
        ),
    }
    selected.discard("")
    if not selected:
        return {"reason": "live_selected_contract_missing"}
    if normalize_symbol(symbol) not in selected:
        return {
            "reason": "live_selected_contract_mismatch",
            "selected_symbols": sorted(selected),
        }
    basket_version = (
        _basket_get(basket, "basket_version")
        or _basket_get(basket, "active_basket_version")
        or _basket_get(basket, "version")
    )
    owner_version = getattr(manager, "_active_basket_version", None)
    if (
        basket_version not in (None, "")
        and owner_version not in (None, "")
        and str(basket_version) != str(owner_version)
    ):
        return {
            "reason": "live_selected_contract_version_stale",
            "selected_contract_version": basket_version,
            "active_basket_version": owner_version,
        }
    return None


def _hub_ready(manager: Any) -> bool | None:
    hub = getattr(manager, "_data_hub", None)
    if hub is None or not hasattr(hub, "indicators_ready"):
        return None
    with suppress(Exception):
        return bool(getattr(hub, "indicators_ready"))
    return False


def _record(
    manager: Any,
    symbol: str,
    reason: str,
    trace_id: str | None,
    details: dict[str, Any] | None = None,
) -> None:
    payload = dict(details or {})
    payload.setdefault("reason", reason)
    payload.setdefault("trace_id", trace_id)
    decisions = getattr(manager, "_last_no_signal_decision_by_symbol", None)
    if isinstance(decisions, dict):
        try:
            from nifty_scalper_bot.core.strategy_manager import StrategyNoSignalDecision

            decisions[str(symbol).upper()] = StrategyNoSignalDecision(
                symbol=str(symbol),
                eval_id=str(trace_id or payload.get("eval_id") or ""),
                final_block_reason=reason,
                category="live_safety",
                reason=reason,
                blocked_at="strategy_live_safety",
                no_vote_reason_counts={},
                strategy_reasons={},
                direction_bias=None,
                underlying_direction_bias=None,
                context_age_seconds=None,
                trigger_vote_count=0,
                context_vote_count=0,
                selected_ce=None,
                selected_pe=None,
                trace_id=trace_id,
            )
        except Exception:
            decisions[str(symbol).upper()] = payload
    log_throttled(
        LOG,
        f"strategy_live_safety:{symbol}:{reason}",
        "STRATEGY_LIVE_SAFETY_BLOCK symbol=%s reason=%s",
        symbol,
        reason,
        interval_sec=30.0,
        level=30,
        extra={"event": "STRATEGY_LIVE_SAFETY_BLOCK", "symbol": symbol, **payload},
    )


def _evaluation_readiness_block(
    manager: Any, underlying_symbol: str
) -> dict[str, Any] | None:
    if not _is_live(manager):
        return None
    required = _required_bars(manager)
    available = _canonical_bars_available(manager, underlying_symbol)
    indicator_available = _indicator_bars_available(manager, underlying_symbol)
    if available < required:
        return {
            "reason": "live_underlying_history_not_ready",
            "bars_available": available,
            "mdm_bars": available,
            "indicator_bars": indicator_available,
            "required_bars": required,
        }
    if _hub_ready(manager) is False:
        return {
            "reason": "live_hub_indicators_not_ready",
            "bars_available": available,
            "mdm_bars": available,
            "indicator_bars": indicator_available,
            "required_bars": required,
        }
    for checker in (_latest_underlying_bar_fresh_block, _context_fresh_block):
        block = checker(manager, underlying_symbol)
        if block is not None:
            block.setdefault("bars_available", available)
            block.setdefault("mdm_bars", available)
            block.setdefault("indicator_bars", indicator_available)
            block.setdefault("required_bars", required)
            return block
    return None


def _candidate_execution_block(
    manager: Any, candidate_symbol: str
) -> dict[str, Any] | None:
    for checker in (_selected_contract_block, _live_option_tick_block):
        block = checker(manager, candidate_symbol)
        if block is not None:
            return block
    return None


def _readiness_block(manager: Any, symbol: str) -> dict[str, Any] | None:
    return _evaluation_readiness_block(manager, symbol)


def _has_identity(signal: Signal) -> bool:
    metadata = dict(getattr(signal, "metadata", {}) or {})
    return any(metadata.get(key) not in (None, "") for key in IDENTITY_KEYS)


def _add_identity(signal: Signal) -> Signal:
    metadata = dict(getattr(signal, "metadata", {}) or {})
    if metadata.get("timestamp") in (None, ""):
        for key in ("bar_timestamp", "latest_bar_ts", "signal_timestamp"):
            candidate = metadata.get(key)
            if candidate not in (None, ""):
                metadata["timestamp"] = candidate
                break
    deterministic = str(
        metadata.get("deterministic_signal_id") or signal.deterministic_id
    )
    metadata.setdefault("deterministic_signal_id", deterministic)
    metadata.setdefault("signal_id", deterministic)
    metadata.setdefault("idempotency_key", deterministic)
    metadata.setdefault("signal_timestamp", metadata.get("timestamp"))
    metadata.setdefault("identity_source", "strategy_live_safety")
    return signal.with_metadata(**metadata)


def _orderflow_selected_option_block(signal: Signal) -> dict[str, Any] | None:
    metadata = dict(getattr(signal, "metadata", {}) or {})
    strategy_name = str(metadata.get("strategy_name") or metadata.get("strategy") or "")
    canonical = strategy_name.replace("_", "").replace("-", "").strip().lower()
    if canonical not in {"orderflow", "orderflowstrategy"}:
        return None

    role = str(metadata.get("role") or "").strip().lower()
    trigger_like = bool(
        role == "trigger"
        or _metadata_true(metadata.get("can_trigger"))
        or _metadata_true(metadata.get("trigger_conditions_met"))
        or _metadata_true(metadata.get("trigger_eligible"))
        or metadata.get("approval_candidate") == "orderflow_live_depth_trigger"
    )
    if not trigger_like:
        return None
    if _metadata_true(metadata.get("selected_or_near_atm")):
        return None
    return {
        "reason": "live_orderflow_selected_option_required",
        "strategy_name": strategy_name,
        "role": role or None,
        "can_trigger": bool(metadata.get("can_trigger")),
        "trigger_conditions_met": bool(metadata.get("trigger_conditions_met")),
        "selected_or_near_atm": metadata.get("selected_or_near_atm"),
        "candidate_symbol": metadata.get("candidate_symbol"),
    }


def _final_filter(manager: Any, signal: Signal, trace_id: str | None) -> Signal | None:
    filter_fn = getattr(manager, "_filter_signal", None)
    if callable(filter_fn):
        try:
            if not bool(filter_fn(signal)):
                _record(
                    manager,
                    signal.symbol,
                    "live_signal_final_filter_block",
                    trace_id,
                    {"confidence": signal.confidence, "action": signal.action},
                )
                return None
        except Exception as exc:
            _record(
                manager,
                signal.symbol,
                "live_signal_final_filter_error",
                trace_id,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return None
    orchestrator = getattr(manager, "_orchestrator", None)
    filter_signal = getattr(orchestrator, "filter_signal", None)
    if callable(filter_signal):
        try:
            filtered = filter_signal(
                signal,
                dict(getattr(signal, "metadata", {}) or {}),
                getattr(manager, "_position_manager", None),
            )
        except Exception as exc:
            _record(
                manager,
                signal.symbol,
                "live_signal_orchestrator_error",
                trace_id,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return None
        if filtered is None:
            _record(
                manager,
                signal.symbol,
                "live_signal_orchestrator_block",
                trace_id,
                {"confidence": signal.confidence, "action": signal.action},
            )
            return None
        if isinstance(filtered, Signal):
            return filtered
    return signal


def _install_canonical_history_builder(strategy_module: Any) -> None:
    from nifty_scalper_bot.core.strategy_context_builder import (
        build_strategy_history_context as canonical_build_strategy_history_context,
    )
    from nifty_scalper_bot.strategies import signal_generator as signal_generator_module

    strategy_module.build_strategy_history_context = (
        canonical_build_strategy_history_context
    )
    signal_generator_module.build_strategy_history_context = (
        canonical_build_strategy_history_context
    )


def apply_patches() -> None:
    from nifty_scalper_bot.core import strategy_manager as strategy_module

    _install_canonical_history_builder(strategy_module)
    cls = strategy_module.StrategyManager
    if getattr(cls, "_strategy_live_safety_installed", False):
        return
    original = cls.generate_signal

    def generate_signal(
        self: Any,
        symbol: str,
        current_price: float,
        *,
        trace_id: str | None = None,
    ) -> Signal | None:
        symbol_norm = normalize_symbol(symbol)
        block = _evaluation_readiness_block(self, symbol_norm)
        if block is not None:
            _record(
                self,
                symbol_norm,
                str(block.get("reason") or "live_underlying_history_not_ready"),
                trace_id,
                block,
            )
            return None
        signal = original(self, symbol, current_price, trace_id=trace_id)
        if signal is None or not _is_live(self):
            return signal
        signal = _add_identity(signal)
        metadata = dict(getattr(signal, "metadata", {}) or {})
        if bool(metadata.get("is_approved")):
            signal = _final_filter(self, signal, trace_id)
            if signal is None:
                return None
            signal = _add_identity(signal)
            metadata = dict(getattr(signal, "metadata", {}) or {})
        candidate_symbol = normalize_symbol(getattr(signal, "symbol", ""))
        block = _candidate_execution_block(self, candidate_symbol)
        if block is not None:
            _record(
                self,
                candidate_symbol,
                str(block.get("reason") or "live_candidate_not_ready"),
                trace_id,
                block,
            )
            return None
        orderflow_block = _orderflow_selected_option_block(signal)
        if orderflow_block is not None:
            _record(
                self,
                signal.symbol,
                str(
                    orderflow_block.get("reason")
                    or "live_orderflow_selected_option_required"
                ),
                trace_id,
                orderflow_block,
            )
            return None
        if not _has_identity(signal):
            _record(
                self,
                signal.symbol,
                "live_signal_missing_deterministic_identity",
                trace_id,
                {"metadata_keys": sorted(metadata.keys())},
            )
            return None
        return signal

    cls._strategy_live_safety_original_generate_signal = original
    cls.generate_signal = generate_signal
    cls._strategy_live_safety_installed = True


__all__ = ["apply_patches"]
