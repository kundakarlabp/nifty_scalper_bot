"""Skip trade-strategy work for non-tradable spot/futures context symbols."""

from __future__ import annotations

from functools import wraps
import logging
from typing import Any, Mapping

from nifty_scalper_bot.utils.logging import get_logger, log_throttled
from nifty_scalper_bot.utils.symbols import normalize_symbol

_LOG = get_logger(__name__)
_PATCH_ATTR = "_context_only_fast_path_installed"
_CONTEXT_ROLES = {"spot_context", "futures_context"}

# Only fields consumed by StrategyManager._update_context_snapshot() and
# _derive_context_direction(). Strategy-only SMC/ORB/OrderFlow inputs do not
# belong on spot/futures context evaluation.
CONTEXT_REQUIRED_INDICATORS = frozenset(
    {
        "symbol",
        "ltp",
        "price",
        "close",
        "open",
        "day_open",
        "first_ltp",
        "previous_close",
        "prev_close",
        "previous_price",
        "recent_ltp_delta",
        "net_change",
        "price_change_pct",
        "tick_slope",
        "exchange_vwap",
        "session_vwap",
        "vwap",
        "ema_fast",
        "ema_9",
        "ema9",
        "ema_slow",
        "ema_21",
        "ema21",
        "ema_50",
        "ema50",
        "ema_slope",
        "vwap_slope",
        "adx",
        "atr",
        "volume",
        "avg_volume",
        "futures_volume_ratio",
        "direction_bias",
        "underlying_direction_bias",
        "underlying_direction_confidence",
        "direction_confidence",
        "regime",
        "market_regime",
    }
)


def _generate_context_only(
    manager: Any,
    symbol: str,
    current_price: float,
    role: str,
) -> None:
    """Refresh one context snapshot without running trade-strategy machinery."""
    symbol = normalize_symbol(symbol)
    decision_map = getattr(manager, "_last_no_signal_decision_by_symbol", None)
    if isinstance(decision_map, dict):
        decision_map.pop(str(symbol).upper(), None)

    raw = manager._indicator_engine.get_indicators(
        symbol,
        set(CONTEXT_REQUIRED_INDICATORS),
    )
    if isinstance(raw, dict):
        indicators: dict[str, Any] = dict(raw)
    elif hasattr(raw, "items"):
        indicators = dict(raw.items())
    else:
        indicators = {}

    indicators.setdefault("ltp", current_price)
    indicators.setdefault("price", current_price)
    indicators.setdefault("close", current_price)
    indicators["symbol_role"] = role

    augment = getattr(manager, "_augment_futures_metrics", None)
    if callable(augment):
        augment(indicators)
    manager._update_context_snapshot(
        symbol=symbol,
        indicators=indicators,
        role=role,
    )
    log_throttled(
        _LOG,
        key=f"context_symbol_strategy_eval_skipped:{symbol}",
        msg=(
            "CONTEXT_SYMBOL_STRATEGY_EVAL_SKIPPED "
            f"symbol={symbol} role={role} "
            "reason=context_only_fast_path"
        ),
        interval_sec=30.0,
        level=logging.INFO,
        extra={
            "event": "CONTEXT_SYMBOL_STRATEGY_EVAL_SKIPPED",
            "symbol": symbol,
            "symbol_role": role,
            "reason": "context_only_fast_path",
            "required_indicator_count": len(CONTEXT_REQUIRED_INDICATORS),
        },
    )
    return None


def apply_patches() -> bool:
    """Install the context-only short circuit on the production StrategyManager."""
    from nifty_scalper_bot.core.strategy_manager import (
        StrategyManager,
        classify_symbol_role,
    )

    if bool(getattr(StrategyManager, _PATCH_ATTR, False)):
        return True
    original = StrategyManager.generate_signal

    @wraps(original)
    def generate_signal(
        self: Any,
        symbol: str,
        current_price: float,
        *,
        trace_id: str | None = None,
    ) -> Any:
        normalized = normalize_symbol(symbol)
        role = classify_symbol_role(normalized)
        if role in _CONTEXT_ROLES:
            return _generate_context_only(self, normalized, current_price, role)
        return original(self, normalized, current_price, trace_id=trace_id)

    StrategyManager.generate_signal = generate_signal  # type: ignore[method-assign]
    setattr(StrategyManager, _PATCH_ATTR, True)
    return True


__all__ = [
    "CONTEXT_REQUIRED_INDICATORS",
    "_generate_context_only",
    "apply_patches",
]
