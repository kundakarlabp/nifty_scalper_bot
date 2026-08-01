"""Require a genuinely newer setup after a stop-loss before same-side re-entry.

The existing stop-loss thesis guard enforces a minimum cooldown. This patch keeps
that protection but prevents time alone from revalidating the stopped thesis:
a same-underlying/same-option-side entry must carry a setup/bar timestamp newer
than the stop event. Opposite-side entries and non-stop exits are unchanged.
"""

from __future__ import annotations

import json
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from nifty_scalper_bot.execution.position_risk_state_patch import (
    _cooldown_seconds,
    _is_stop_reason,
    _option_thesis,
)

_PATCHED = False
_ORIGINAL_INIT: Any = None
_ORIGINAL_CLOSE: Any = None
_ANCHOR_KEYS = (
    "setup_candle_timestamp",
    "bar_timestamp",
    "latest_bar_ts",
    "signal_timestamp",
    "timestamp",
)
_RISK_KEY = "_risk_runtime"


def _to_epoch(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed / 1000.0 if parsed > 10_000_000_000 else parsed
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    text = str(value).strip()
    if not text:
        return None
    with suppress(ValueError):
        parsed = float(text)
        return parsed / 1000.0 if parsed > 10_000_000_000 else parsed
    with suppress(ValueError):
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    return None


def _signal_setup_epoch(signal: Any) -> float | None:
    metadata = getattr(signal, "metadata", {})
    if not isinstance(metadata, Mapping):
        return None
    for key in _ANCHOR_KEYS:
        anchor = _to_epoch(metadata.get(key))
        if anchor is not None:
            return anchor
    return None


def _load_persisted_stop(owner: Any) -> dict[str, Any] | None:
    path = Path(getattr(owner, "_state_path", ""))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    risk = payload.get(_RISK_KEY, {}) if isinstance(payload, dict) else {}
    stopped = risk.get("recent_stop_thesis") if isinstance(risk, dict) else None
    return dict(stopped) if isinstance(stopped, dict) else None


def _same_trading_day(owner: Any, stopped: Mapping[str, Any]) -> bool:
    stored_date = str(stopped.get("trading_date") or "")
    return not stored_date or stored_date == owner._trading_date_ist()


def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
    _ORIGINAL_INIT(self, *args, **kwargs)
    persisted = _load_persisted_stop(self)
    if persisted and _same_trading_day(self, persisted):
        stopped_at = _to_epoch(persisted.get("stopped_at_epoch"))
        if stopped_at is None:
            expires = _to_epoch(persisted.get("expires_epoch"))
            if expires is not None:
                persisted["stopped_at_epoch"] = expires - _cooldown_seconds()
        self._recent_stop_thesis = persisted


def record_stop_exit(self: Any, symbol: Any, reason: Any) -> bool:
    """Latch the structural stop-loss rearm requirement for a stopped thesis.

    Canonical single implementation. ``PositionManager.close_position`` is only
    reached on session square-off, so the live bracket stop-loss exit callback
    must call this directly or the guard never arms.
    """
    if not _is_stop_reason(reason):
        return False
    thesis = _option_thesis(symbol)
    if thesis is None:
        return False
    underlying, option_side = thesis
    now = time.time()
    with getattr(self, "_lock"):
        self._recent_stop_thesis = {
            "underlying": underlying,
            "option_side": option_side,
            "symbol": str(symbol).strip().upper(),
            "exit_reason": str(reason),
            "expires_epoch": now + _cooldown_seconds(),
            "stopped_at_epoch": now,
            "trading_date": self._trading_date_ist(),
            "rearm_required": True,
        }
    with suppress(Exception):
        self.save_state()
    return True


def _patched_close_position(
    self: Any,
    symbol: str,
    exit_price: float,
    reason: str,
    close_time: Any = None,
) -> Any:
    result = _ORIGINAL_CLOSE(
        self, symbol, exit_price, reason, close_time=close_time
    )
    record_stop_exit(self, symbol, reason)
    return result


def stop_reentry_block_reason(self: Any, signal: Any) -> str | None:
    thesis = _option_thesis(getattr(signal, "symbol", None))
    if thesis is None:
        return None
    with getattr(self, "_lock"):
        stopped = getattr(self, "_recent_stop_thesis", None)
        if not isinstance(stopped, dict):
            return None
        if not _same_trading_day(self, stopped):
            self._recent_stop_thesis = None
            return None
        stopped_thesis = (
            str(stopped.get("underlying", "")),
            str(stopped.get("option_side", "")),
        )
        if thesis != stopped_thesis:
            return None
        now = time.time()
        minimum_until = float(stopped.get("expires_epoch", 0.0) or 0.0)
        if now < minimum_until:
            return f"stop-loss thesis cooldown active: {int(minimum_until - now + 0.999)}s"
        stopped_at = _to_epoch(stopped.get("stopped_at_epoch"))
        if stopped_at is None:
            stopped_at = minimum_until - _cooldown_seconds()
        setup_epoch = _signal_setup_epoch(signal)
        if setup_epoch is None:
            return "stop-loss thesis awaiting newer setup candle"
        if setup_epoch <= stopped_at:
            return "stop-loss thesis setup not rearmed"
        self._recent_stop_thesis = None
    self.save_state()
    return None


def apply_patches() -> None:
    global _PATCHED, _ORIGINAL_INIT, _ORIGINAL_CLOSE
    if _PATCHED:
        return
    from nifty_scalper_bot.execution.position_manager import PositionManager

    if getattr(PositionManager, "_structural_stop_rearm_patch", False):
        _PATCHED = True
        return
    _ORIGINAL_INIT = PositionManager.__init__
    _ORIGINAL_CLOSE = PositionManager.close_position
    PositionManager.__init__ = _patched_init
    PositionManager.close_position = _patched_close_position
    PositionManager.stop_reentry_block_reason = stop_reentry_block_reason
    PositionManager.record_stop_exit = record_stop_exit
    PositionManager._structural_stop_rearm_patch = True
    _PATCHED = True


__all__ = [
    "apply_patches",
    "record_stop_exit",
    "stop_reentry_block_reason",
    "_signal_setup_epoch",
]
