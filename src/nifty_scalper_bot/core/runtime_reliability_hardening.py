"""Compatibility verification for native runtime reliability behavior.

The former runtime monkey patches now live in their canonical owners:
MarketDataManager and DataHub. This module preserves stable bootstrap/test
helpers without reassigning production class methods at import time.
"""

from __future__ import annotations

from typing import Any

from nifty_scalper_bot.data.data_hub import (
    _is_canonical_runtime_tick,
    _runtime_tick_timestamp_ms,
)
from nifty_scalper_bot.utils.logging import get_logger

_LOG = get_logger(__name__)


def _critical_oldest_pending_age_ms_locked(mdm: Any) -> float:
    """Compatibility delegate to MarketDataManager's native critical-age owner."""
    method = getattr(mdm, "_critical_oldest_pending_age_ms_locked", None)
    if not callable(method):
        raise AttributeError("native critical pending-age method unavailable")
    return float(method())


def apply_patches() -> dict[str, bool]:
    """Verify that former runtime reliability adapters are native."""
    from nifty_scalper_bot.data.data_hub import DataHub
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager

    state = {
        "mdm_overload": callable(
            getattr(MarketDataManager, "_critical_oldest_pending_age_ms_locked", None)
        ),
        "datahub_tick_hotpath": callable(
            getattr(DataHub, "_canonicalize_tick_payload", None)
        ),
    }
    if not all(state.values()):
        raise RuntimeError(f"native_runtime_reliability_incomplete state={state}")
    _LOG.info(
        "RUNTIME_RELIABILITY_NATIVE_VERIFIED state=%s",
        state,
        extra={"event": "RUNTIME_RELIABILITY_NATIVE_VERIFIED", **state},
    )
    return state


__all__ = [
    "apply_patches",
    "_critical_oldest_pending_age_ms_locked",
    "_is_canonical_runtime_tick",
    "_runtime_tick_timestamp_ms",
]
