"""Live-safety guard for StrategyManager.

The guard is intentionally narrow: real LIVE mode fails closed on cold history,
missing DataHub readiness, approved-signal final-filter bypass, and missing live
signal identity. PAPER/SHADOW behaviour is unchanged.
"""

from __future__ import annotations

from contextlib import suppress
import os
from typing import Any

from nifty_scalper_bot.execution.readiness import HistoryReadinessPolicy
from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.utils.logging import get_logger, log_throttled
from nifty_scalper_bot.utils.symbols import normalize_symbol

LOG = get_logger(__name__)
TRUTHY = {"1", "true", "yes", "y", "on", "live"}
IDENTITY_KEYS = ("signal_id", "deterministic_signal_id", "bar_timestamp", "signal_timestamp", "timestamp")


def _env_true(name: str) -> bool:
    return str(os.getenv(name, "") or "").strip().lower() in TRUTHY


def _is_live(manager: Any) -> bool:
    checker = getattr(manager, "_is_live_mode", None)
    if callable(checker):
        with suppress(Exception):
            return bool(checker())
    mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
    return mode == "LIVE" and (_env_true("ENABLE_LIVE") or _env_true("ENABLE_LIVE_TRADING")) and not (_env_true("PAPER_MODE") or _env_true("PAPER__ENABLED") or _env_true("SHADOW_MODE"))


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


def _hub_ready(manager: Any) -> bool | None:
    hub = getattr(manager, "_data_hub", None)
    if hub is None or not hasattr(hub, "indicators_ready"):
        return None
    with suppress(Exception):
        return bool(getattr(hub, "indicators_ready"))
    return False


def _record(manager: Any, symbol: str, reason: str, trace_id: str | None, details: dict[str, Any] | None = None) -> None:
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


def _readiness_block(manager: Any, symbol: str) -> dict[str, Any] | None:
    if not _is_live(manager):
        return None
    required = _required_bars(manager)
    available = _bars_available(manager, symbol)
    if available < required:
        return {"reason": "live_indicators_not_ready", "bars_available": available, "required_bars": required}
    if _hub_ready(manager) is False:
        return {"reason": "live_hub_indicators_not_ready", "bars_available": available, "required_bars": required}
    return None


def _has_identity(signal: Signal) -> bool:
    metadata = dict(getattr(signal, "metadata", {}) or {})
    return any(metadata.get(key) not in (None, "") for key in IDENTITY_KEYS)


def _add_identity(signal: Signal) -> Signal:
    metadata = dict(getattr(signal, "metadata", {}) or {})
    metadata.setdefault("deterministic_signal_id", signal.deterministic_id)
    return signal.with_metadata(**metadata)


def _final_filter(manager: Any, signal: Signal, trace_id: str | None) -> Signal | None:
    filter_fn = getattr(manager, "_filter_signal", None)
    if callable(filter_fn):
        try:
            if not bool(filter_fn(signal)):
                _record(manager, signal.symbol, "live_signal_final_filter_block", trace_id, {"confidence": signal.confidence, "action": signal.action})
                return None
        except Exception as exc:
            _record(manager, signal.symbol, "live_signal_final_filter_error", trace_id, {"error": f"{type(exc).__name__}: {exc}"})
            return None
    orchestrator = getattr(manager, "_orchestrator", None)
    filter_signal = getattr(orchestrator, "filter_signal", None)
    if callable(filter_signal):
        try:
            filtered = filter_signal(signal, dict(getattr(signal, "metadata", {}) or {}), getattr(manager, "_position_manager", None))
        except Exception as exc:
            _record(manager, signal.symbol, "live_signal_orchestrator_error", trace_id, {"error": f"{type(exc).__name__}: {exc}"})
            return None
        if filtered is None:
            _record(manager, signal.symbol, "live_signal_orchestrator_block", trace_id, {"confidence": signal.confidence, "action": signal.action})
            return None
        if isinstance(filtered, Signal):
            return filtered
    return signal


def apply_patches() -> None:
    from nifty_scalper_bot.core import strategy_manager as strategy_module

    cls = strategy_module.StrategyManager
    if getattr(cls, "_strategy_live_safety_installed", False):
        return
    original = cls.generate_signal

    def generate_signal(self: Any, symbol: str, current_price: float, *, trace_id: str | None = None) -> Signal | None:
        symbol_norm = normalize_symbol(symbol)
        block = _readiness_block(self, symbol_norm)
        if block is not None:
            _record(self, symbol_norm, str(block.get("reason") or "live_indicators_not_ready"), trace_id, block)
            return None
        signal = original(self, symbol, current_price, trace_id=trace_id)
        if signal is None or not _is_live(self):
            return signal
        metadata = dict(getattr(signal, "metadata", {}) or {})
        if bool(metadata.get("is_approved")):
            signal = _final_filter(self, signal, trace_id)
            if signal is None:
                return None
            metadata = dict(getattr(signal, "metadata", {}) or {})
        if not _has_identity(signal):
            _record(self, signal.symbol, "live_signal_missing_deterministic_identity", trace_id, {"metadata_keys": sorted(metadata.keys())})
            return None
        return _add_identity(signal)

    cls._strategy_live_safety_original_generate_signal = original
    cls.generate_signal = generate_signal
    cls._strategy_live_safety_installed = True


__all__ = ["apply_patches"]
