"""Focused runtime hardening for reliability defects observed on 2026-08-07.

No trading threshold, order path, risk control, market-hours guard, or strategy
permission is changed here. The adapters correct overload age attribution,
bounded consensus-quality evidence, tick hot-path duplication, and runtime
diagnostics while preserving fail-closed behavior for entry-critical queues.
"""

# ruff: noqa: E501  # Legacy diagnostic strings; other Ruff rules remain active.

from __future__ import annotations

import time
from typing import Any, Mapping

from nifty_scalper_bot.data.data_hub import (
    _is_canonical_runtime_tick,
    _runtime_tick_timestamp_ms,
)
from nifty_scalper_bot.utils.logging import get_logger

_LOG = get_logger(__name__)
_PATCH_ATTR = "_runtime_reliability_hardening_installed"


def _critical_oldest_pending_age_ms_locked(mdm: Any) -> float:
    """Return oldest entry-critical queue age.

    Non-selected active-basket ``near_atm`` options remain in normal FIFO
    queues so every tick still reaches CandleEngine/OHLC processing. Their age
    is optional strategy context, however, and must not by itself disarm fresh
    selected CE/PE execution. Selected options, spot/future context, open
    positions, and any unclassified normal queue remain age-critical. Global
    pending-count overload remains unchanged and still covers every lane.
    """

    oldest_mono: float | None = None
    for queue in (getattr(mdm, "_pending_tick_queues", {}) or {}).values():
        if not queue:
            continue
        tick = queue[0]
        if not isinstance(tick, Mapping):
            return float(
                getattr(mdm, "_overload_enter_oldest_ms", 2000.0) or 2000.0
            )
        bucket = str(tick.get("_mdm_priority_bucket") or "").strip().lower()
        if bucket == "near_atm":
            continue
        timestamp = tick.get("_mdm_enqueued_mono")
        if not isinstance(timestamp, (int, float)):
            return float(
                getattr(mdm, "_overload_enter_oldest_ms", 2000.0) or 2000.0
            )
        candidate = float(timestamp)
        oldest_mono = (
            candidate if oldest_mono is None else min(oldest_mono, candidate)
        )
    if oldest_mono is None:
        return 0.0
    return max(0.0, (time.monotonic() - oldest_mono) * 1000.0)


def _install_mdm_overload_patch() -> bool:
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager

    if bool(getattr(MarketDataManager, _PATCH_ATTR, False)):
        return True

    def _update_pipeline_overload_locked(self: Any) -> None:
        pending = self._pending_count_locked()
        total_oldest_ms = self._oldest_pending_age_ms_locked()
        critical_oldest_ms = _critical_oldest_pending_age_ms_locked(self)
        if not self._pipeline_overloaded:
            if (
                pending >= self._overload_enter_pending
                or critical_oldest_ms >= self._overload_enter_oldest_ms
            ):
                self._pipeline_overloaded = True
                self._overload_since_mono = time.monotonic()
                self._logger.warning(
                    "DATA_PIPELINE_OVERLOAD_ENTER pending_ticks=%d oldest_pending_age_ms=%.0f "
                    "critical_oldest_pending_age_ms=%.0f enter_pending=%d enter_oldest_ms=%.0f",
                    pending,
                    total_oldest_ms,
                    critical_oldest_ms,
                    self._overload_enter_pending,
                    self._overload_enter_oldest_ms,
                    extra={
                        "event": "DATA_PIPELINE_OVERLOAD_ENTER",
                        "pending_ticks": pending,
                        "oldest_pending_age_ms": total_oldest_ms,
                        "critical_oldest_pending_age_ms": critical_oldest_ms,
                        "enter_pending": self._overload_enter_pending,
                        "enter_oldest_ms": self._overload_enter_oldest_ms,
                    },
                )
        elif (
            pending <= self._overload_exit_pending
            and critical_oldest_ms <= self._overload_exit_oldest_ms
        ):
            duration = 0.0
            if self._overload_since_mono is not None:
                duration = max(0.0, time.monotonic() - self._overload_since_mono)
            self._pipeline_overloaded = False
            self._overload_since_mono = None
            self._logger.warning(
                "DATA_PIPELINE_OVERLOAD_RECOVERED pending_ticks=%d oldest_pending_age_ms=%.0f "
                "critical_oldest_pending_age_ms=%.0f overloaded_for_s=%.1f",
                pending,
                total_oldest_ms,
                critical_oldest_ms,
                duration,
                extra={
                    "event": "DATA_PIPELINE_OVERLOAD_RECOVERED",
                    "pending_ticks": pending,
                    "oldest_pending_age_ms": total_oldest_ms,
                    "critical_oldest_pending_age_ms": critical_oldest_ms,
                    "overloaded_for_s": duration,
                },
            )

    MarketDataManager._update_pipeline_overload_locked = (  # type: ignore[method-assign]
        _update_pipeline_overload_locked
    )
    setattr(MarketDataManager, _PATCH_ATTR, True)
    return True


def apply_patches() -> dict[str, bool]:
    """Install MDM overload protection and verify native DataHub ownership."""

    from nifty_scalper_bot.data.data_hub import DataHub

    state = {
        "mdm_overload": _install_mdm_overload_patch(),
        "datahub_tick_hotpath": callable(
            getattr(DataHub, "_canonicalize_tick_payload", None)
        ),
    }
    if not all(state.values()):
        raise RuntimeError(f"runtime_reliability_hardening_incomplete state={state}")
    _LOG.info(
        "RUNTIME_RELIABILITY_HARDENING_INSTALLED state=%s",
        state,
        extra={"event": "RUNTIME_RELIABILITY_HARDENING_INSTALLED", **state},
    )
    return state


__all__ = [
    "apply_patches",
    "_critical_oldest_pending_age_ms_locked",
    "_is_canonical_runtime_tick",
    "_runtime_tick_timestamp_ms",
]
