"""Risk management primitives."""

from .entry_guard_patch import apply_patches as _apply_entry_guard_patch
from .risk_manager import OrderSignal, RiskManager, RiskSnapshot, RiskState
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
