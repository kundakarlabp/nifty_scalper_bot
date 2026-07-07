"""Corrected strategy-exit score diagnostics.

StrategyManager's legacy STRATEGY_MANAGER_EXIT log historically used the order
quantity in the signal_score field. This non-invasive adapter emits a corrected
post-return diagnostic without changing trade gating or execution behavior.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.utils.logging import get_logger, log_throttled

LOG = get_logger(__name__)
_PATCH_INSTALLED = False

_SCORE_KEYS = (
    "final_trade_score",
    "consensus_score",
    "setup_score",
    "raw_setup_score",
    "strategy_score",
)


def _extract_signal_score(signal: Signal) -> float | None:
    metadata = dict(getattr(signal, "metadata", {}) or {})
    for key in _SCORE_KEYS:
        raw = metadata.get(key)
        if raw in (None, ""):
            continue
        with suppress(TypeError, ValueError):
            return float(raw)
    return None


def apply_patches() -> bool:
    """Install corrected score diagnostic wrapper. Returns True if installed."""
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return False
    try:
        from nifty_scalper_bot.core import strategy_manager as strategy_module
    except Exception:
        return False

    cls = strategy_module.StrategyManager
    if getattr(cls, "_strategy_exit_score_diagnostics_installed", False):
        _PATCH_INSTALLED = True
        return False

    original = cls.generate_signal

    def generate_signal(
        self: Any,
        symbol: str,
        current_price: float,
        *,
        trace_id: str | None = None,
    ) -> Signal | None:
        signal = original(self, symbol, current_price, trace_id=trace_id)
        if signal is None:
            return None
        score = _extract_signal_score(signal)
        if score is None:
            return signal
        metadata = dict(getattr(signal, "metadata", {}) or {})
        log_throttled(
            LOG,
            f"strategy_exit_score_corrected:{getattr(signal, 'symbol', symbol)}:{metadata.get('approval_path')}",
            "STRATEGY_MANAGER_EXIT_SCORE symbol=%s signal_score=%.3f signal_quantity=%s approval_path=%s",
            getattr(signal, "symbol", symbol),
            score,
            getattr(signal, "quantity", None),
            metadata.get("approval_path"),
            interval_sec=10.0,
            level=20,
            extra={
                "event": "STRATEGY_MANAGER_EXIT_SCORE",
                "symbol": getattr(signal, "symbol", symbol),
                "signal_score": score,
                "signal_quantity": getattr(signal, "quantity", None),
                "approval_path": metadata.get("approval_path"),
                "setup_score": metadata.get("setup_score"),
                "raw_setup_score": metadata.get("raw_setup_score"),
                "final_trade_score": metadata.get("final_trade_score"),
                "consensus_score": metadata.get("consensus_score"),
                "trace_id": trace_id,
            },
        )
        return signal

    cls._strategy_exit_score_diagnostics_original_generate_signal = original
    cls.generate_signal = generate_signal
    cls._strategy_exit_score_diagnostics_installed = True
    _PATCH_INSTALLED = True
    return True


__all__ = ["apply_patches", "_extract_signal_score"]
