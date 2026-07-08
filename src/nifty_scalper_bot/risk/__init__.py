"""Risk management primitives."""

from .risk_manager import OrderSignal, RiskManager, RiskSnapshot, RiskState
from .entry_guard_patch import apply_patches as _apply_entry_guard_patch
from .time_based_sizer import TimeBasedSizer
from .volatility_sizer import VolatilitySizer

_apply_entry_guard_patch()

__all__ = [
    "OrderSignal",
    "RiskManager",
    "RiskSnapshot",
    "RiskState",
    "TimeBasedSizer",
    "VolatilitySizer",
]
