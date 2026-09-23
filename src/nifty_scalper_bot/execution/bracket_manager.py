"""File purpose:
    Provide the stable public API for the canonical bracket and exit lifecycle.

Key responsibilities:
    - Re-export bracket state models and helpers from ``bracket_core``.
    - Expose ``BoundBracketManager`` as the single production bracket authority.

Operational constraints:
    - This facade must not own independent bracket state or exit execution logic.
    - Entry release remains blocked until the bound runtime confirms durable closure.
"""

from __future__ import annotations

from collections.abc import Mapping

from nifty_scalper_bot.execution import bracket_core as _core

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

from nifty_scalper_bot.execution.ownership import BoundBracketManager  # noqa: E402
from nifty_scalper_bot.execution.runtime_bracket_manager import (  # noqa: E402,F401
    RuntimeBracketManager,
)

_original_tick_exchange_epoch = _core.tick_exchange_epoch


def _tick_exchange_epoch_with_receipt(tick):
    """Use broker event time first, then explicit receipt time; never invent time."""
    epoch = _original_tick_exchange_epoch(tick)
    if epoch is not None:
        return epoch
    for key in (
        "last_trade_time",
        "last_traded_time",
        "last_trade_timestamp",
        "received_at",
        "received_ts",
        "received_time",
    ):
        value = tick.get(key)
        if hasattr(value, "timestamp"):
            try:
                return float(value.timestamp())
            except (TypeError, ValueError, OSError):
                continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            return value / 1000.0 if value > 1e12 else value
    return None


_core.tick_exchange_epoch = _tick_exchange_epoch_with_receipt
tick_exchange_epoch = _tick_exchange_epoch_with_receipt

from nifty_scalper_bot.execution.market_aware_profit_extension import (  # noqa: E402
    apply_patches as _apply_market_aware_profit_extension,
)

_apply_market_aware_profit_extension(BoundBracketManager)


_original_on_tick = BoundBracketManager.on_tick


def _capture_same_tick_cached_quote(self, symbol, ltp, exchange_ts):
    """Recover executable depth from the cached SSOT without mixing tick identities."""
    source = getattr(self, "_market_data", None)
    getter = getattr(source, "get_latest_tick", None) if source is not None else None
    if not callable(getter):
        return
    try:
        cached = getter(symbol)
    except Exception:
        return
    if not isinstance(cached, Mapping):
        return
    try:
        cached_ltp = float(
            cached.get("ltp") or cached.get("last_price") or cached.get("price") or 0.0
        )
        current_ltp = float(ltp)
    except (TypeError, ValueError):
        return
    if cached_ltp <= 0.0 or current_ltp <= 0.0 or abs(cached_ltp - current_ltp) > 1e-9:
        return
    if exchange_ts is not None:
        cached_ts = tick_exchange_epoch(cached)
        try:
            current_ts = float(exchange_ts)
        except (TypeError, ValueError):
            return
        if cached_ts is None or abs(float(cached_ts) - current_ts) > 0.001:
            return
    self._capture_exit_quote(_core.normalize_symbol(symbol), cached)


def _on_tick_with_cached_executable_quote(
    self, symbol, ltp, exchange_ts=None, *, defer_submission=False
):
    """Preserve executable bid/ask when legacy callers forward only LTP."""
    _capture_same_tick_cached_quote(self, symbol, ltp, exchange_ts)
    return _original_on_tick(
        self,
        symbol,
        ltp,
        exchange_ts,
        defer_submission=defer_submission,
    )


BoundBracketManager.on_tick = _on_tick_with_cached_executable_quote

BracketManager = BoundBracketManager

__all__ = sorted(
    {
        *[name for name in dir(_core) if not name.startswith("_")],
        "BoundBracketManager",
        "BracketManager",
        "RuntimeBracketManager",
    }
)
