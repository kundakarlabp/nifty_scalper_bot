"""Risk management primitives."""

from .risk_manager import OrderSignal, RiskManager, RiskSnapshot, RiskState
from .entry_guard_patch import apply_patches as _apply_entry_guard_patch
from .time_based_sizer import TimeBasedSizer
from .volatility_sizer import VolatilitySizer


def _resolve_lot_size_from_provider(self: RiskManager, symbol: str | None) -> int:
    """Resolve lot size from the provider already injected by app wiring."""
    lookup = self._lot_size_lookup if callable(self._lot_size_lookup) else None
    target_symbol = symbol or self._lot_size_symbol or "NIFTY"
    if lookup is None:
        raise RuntimeError("lot size provider not configured")
    lot_size = int(lookup(target_symbol) or 0)
    if lot_size <= 0:
        raise ValueError(f"invalid lot size for {target_symbol}: {lot_size}")
    self._logger.info(
        "LOT_SIZE_RESOLVED underlying=%s lot_size=%s source=provider",
        "NIFTY" if "NIFTY" in str(target_symbol).upper() else target_symbol,
        lot_size,
    )
    return lot_size


# risk_manager.py still references a removed resolver helper. Keep the existing
# injected InstrumentManager provider as the single source of truth instead of
# introducing another lot-size lookup path.
RiskManager._resolve_lot_size = _resolve_lot_size_from_provider
_apply_entry_guard_patch()

__all__ = [
    "OrderSignal",
    "RiskManager",
    "RiskSnapshot",
    "RiskState",
    "TimeBasedSizer",
    "VolatilitySizer",
]
