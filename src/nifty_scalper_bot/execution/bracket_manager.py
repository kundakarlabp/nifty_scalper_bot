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

from nifty_scalper_bot.execution import bracket_core as _core

for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

from nifty_scalper_bot.execution.runtime_bracket_manager import RuntimeBracketManager  # noqa: E402
from nifty_scalper_bot.execution.ownership import BoundBracketManager  # noqa: E402


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

_original_confirm_entry_fill = BoundBracketManager.confirm_entry_fill


def _confirm_entry_fill_once(self, order_id, fill_price):
    """Ignore an identical repeated COMPLETE callback without resetting protection."""
    bracket = self.get_bracket(order_id)
    try:
        price = float(fill_price)
        prior = float(bracket.entry_fill_price) if bracket is not None else None
    except (TypeError, ValueError):
        price = prior = None
    if (
        bracket is not None
        and bracket.entry_confirmed
        and prior is not None
        and price is not None
        and abs(prior - price) < 1e-9
    ):
        _core.LOGGER.info(
            "BRACKET_ACTIVATION_DUPLICATE_IGNORED order_id=%s symbol=%s fill_price=%.2f",
            order_id,
            bracket.symbol,
            price,
            extra={
                "event": "BRACKET_ACTIVATION_DUPLICATE_IGNORED",
                "order_id": str(order_id),
                "symbol": bracket.symbol,
                "fill_price": price,
            },
        )
        return True
    return _original_confirm_entry_fill(self, order_id, fill_price)


BoundBracketManager.confirm_entry_fill = _confirm_entry_fill_once
BracketManager = BoundBracketManager

__all__ = sorted(
    {
        *[name for name in dir(_core) if not name.startswith("_")],
        "BoundBracketManager",
        "BracketManager",
        "RuntimeBracketManager",
    }
)
