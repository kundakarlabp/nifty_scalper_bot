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

from nifty_scalper_bot.execution.readiness import HistoryReadinessPolicy
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


def _bars_available(manager: Any, symbol: str) -> int:
    getter = getattr(getattr(manager, "_indicator_engine", None), "get_history", None)
    if not callable(getter):
        return 0
    with suppress(Exception):
        return len(getter(symbol) or [])
    return 0


def _required_bars(manager: Any) -> int:
    policy = HistoryReadinessPolicy.from_env()
    raw = getattr(manager, "_required_candles", policy.option_eval_min_bars)
    with suppress(Exception):
        return max(1, int(raw or policy.option_eval_min_bars))
    return max(1, int(policy.option_eval_min_bars))


def _max_age_seconds(name: str, default: float) -> float:
    with suppress(Exception):
        return max(0.0, float(os.getenv(name, str(default)) or default))
    return default


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


def _latest_bar_epoch(manager: Any, symbol: str) -> float | None:
    getter = getattr(getattr(manager, "_indicator_engine", None), "get_history", None)
    if not callable(getter):
        return None
    with suppress(Exception):
        history = list(getter(symbol) or [])
    if not history:
        return None
    latest = history[-1]
    if isinstance(latest, dict):
        for key in ("timestamp", "timestamp_ms", "ts", "date", "time"):
            epoch = _coerce_epoch_seconds(latest.get(key))
            if epoch is not None:
                return epoch
    for key in ("timestamp", "timestamp_ms", "ts", "date", "time"):
        epoch = _coerce_epoch_seconds(getattr(latest, key, None))
        if epoch is not None:
            return epoch
    return None


def _latest_bar_fresh_block(manager: Any, symbol: str) -> dict[str, Any] | None:
    max_age = _max_age_seconds("STRATEGY_LATEST_BAR_MAX_AGE_SECONDS", 180.0)
    epoch = _latest_bar_epoch(manager, symbol)
    if epoch is None:
        return {"reason": "live_latest_closed_bar_missing", "max_bar_age_s": max_age}
    age = max(0.0, time.time() - epoch)
    if age > max_age:
        return {
            "reason": "live_latest_closed_bar_stale",
            "latest_bar_ts": epoch,
            "bar_age_s": age,
            "max_bar_age_s": max_age,
        }
    return None


def _live_option_tick_block(manager: Any, symbol: str) -> dict[str, Any] | None:
    max_age = _max_age_seconds("STRATEGY_LIVE_TICK_MAX_AGE_SECONDS", 5.0)
    mdm = getattr(manager, "_market_data_manager", None) or getattr(
        getattr(manager, "_data_hub", None), "_mdm", None
    )
    age_fn = getattr(mdm, "time_since_last_live_ws_tick", None)
    if not callable(age_fn):
        return None
    with suppress(Exception):
        age = age_fn(symbol)
        if age is not None and float(age) <= max_age:
            return None
        if callable(getattr(mdm, "request_fallback_refresh", None)):
            with suppress(Exception):
                mdm.request_fallback_refresh(
                    symbol, reason="strategy_live_option_tick_stale"
                )
        return {
            "reason": "live_option_tick_stale",
            "live_tick_age_s": age,
            "max_live_tick_age_s": max_age,
        }
    return {
        "reason": "live_option_tick_freshness_unknown",
        "max_live_tick_age_s": max_age,
    }


def _context_fresh_block(manager: Any, symbol: str) -> dict[str, Any] | None:
    required = getattr(manager, "_strategy_required_context", None)
    if required is False:
        return None
    indicators = getattr(manager, "_strategy_live_safety_last_indicators", {})
    if not isinstance(indicators, dict):
        indicators = {}
    if indicators.get("context_fresh") is False:
        return {"reason": "live_underlying_context_stale", "context_fresh": False}
    max_age = _max_age_seconds("STRATEGY_CONTEXT_MAX_AGE_SECONDS", 120.0)
    ages = {
        k: indicators.get(k)
        for k in (
            "spot_age_seconds",
            "spot_tick_age_s",
            "futures_age_seconds",
            "futures_tick_age_s",
        )
    }
    present = [float(v) for v in ages.values() if isinstance(v, (int, float))]
    if present and min(present) > max_age:
        return {
            "reason": "live_underlying_context_stale",
            "context_age_seconds": min(present),
            "max_context_age_s": max_age,
        }
    if (
        indicators.get("spot_fresh") is False
        and indicators.get("fut_fresh") is False
        and indicators.get("futures_fresh") is False
    ):
        return {
            "reason": "live_underlying_context_stale",
            "spot_fresh": False,
            "fut_fresh": False,
        }
    return None


def _selected_contract_block(manager: Any, symbol: str) -> dict[str, Any] | None:
    indicators = getattr(manager, "_strategy_live_safety_last_indicators", {})
    if not isinstance(indicators, dict):
        return None
    selected = {
        normalize_symbol(indicators.get("selected_ce")),
        normalize_symbol(indicators.get("selected_pe")),
    }
    selected.discard("")
    if selected and normalize_symbol(symbol) not in selected:
        return {
            "reason": "live_selected_contract_mismatch",
            "selected_symbols": sorted(selected),
        }
    local_version = indicators.get("selected_contract_version") or indicators.get(
        "active_basket_version"
    )
    owner_version = getattr(manager, "_active_basket_version", None)
    if (
        local_version not in (None, "")
        and owner_version not in (None, "")
        and str(local_version) != str(owner_version)
    ):
        return {
            "reason": "live_selected_contract_version_stale",
            "selected_contract_version": local_version,
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


def _prime_indicator_snapshot(manager: Any, symbol: str) -> None:
    engine = getattr(manager, "_indicator_engine", None)
    getter = getattr(engine, "get_indicators", None)
    if not callable(getter):
        return
    required = set()
    union = getattr(manager, "_strategy_required_indicator_union", None)
    if callable(union):
        with suppress(Exception):
            required = set(union() or set())
    with suppress(Exception):
        raw = getter(symbol, required)
        if isinstance(raw, dict):
            manager._strategy_live_safety_last_indicators = dict(raw)
        elif hasattr(raw, "items"):
            manager._strategy_live_safety_last_indicators = dict(raw)


def _readiness_block(manager: Any, symbol: str) -> dict[str, Any] | None:
    if not _is_live(manager):
        return None
    required = _required_bars(manager)
    available = _bars_available(manager, symbol)
    if available < required:
        return {
            "reason": "live_indicators_not_ready",
            "bars_available": available,
            "required_bars": required,
        }
    if _hub_ready(manager) is False:
        return {
            "reason": "live_hub_indicators_not_ready",
            "bars_available": available,
            "required_bars": required,
        }
    for checker in (
        _latest_bar_fresh_block,
        _live_option_tick_block,
        _context_fresh_block,
        _selected_contract_block,
    ):
        block = checker(manager, symbol)
        if block is not None:
            block.setdefault("bars_available", available)
            block.setdefault("required_bars", required)
            return block
    return None


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
        _prime_indicator_snapshot(self, symbol_norm)
        block = _readiness_block(self, symbol_norm)
        if block is not None:
            _record(
                self,
                symbol_norm,
                str(block.get("reason") or "live_indicators_not_ready"),
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
