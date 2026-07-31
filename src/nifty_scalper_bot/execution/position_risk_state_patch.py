"""Persist entry-risk state and block immediate same-thesis stop re-entry.

This patch stays deliberately narrow:
* persist the existing PositionManager daily-entry counter in its existing JSON
  state file so an intraday process restart cannot reset max_trades_per_day;
* after a stop-loss exit, temporarily block a new option entry for the same
  underlying and option side, including a strike change.

Protective/reducing orders remain outside this entry-only guard.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

_PATCH_APPLIED = False
_ORIGINAL_INIT: Any = None
_ORIGINAL_SAVE_STATE: Any = None
_ORIGINAL_CLOSE_POSITION: Any = None
_RISK_KEY = "_risk_runtime"


def _option_thesis(symbol: object) -> tuple[str, str] | None:
    text = str(symbol or "").strip().upper()
    if ":" in text:
        text = text.split(":", 1)[1]
    option_side = text[-2:] if text.endswith(("CE", "PE")) else ""
    if not option_side:
        return None
    contract = text[:-2]
    digit_at = next((index for index, char in enumerate(contract) if char.isdigit()), -1)
    if digit_at <= 0:
        return None
    underlying = contract[:digit_at]
    if not underlying.isalpha():
        return None
    return underlying, option_side


def _cooldown_seconds() -> float:
    raw = os.getenv("STOP_LOSS_REENTRY_COOLDOWN_SECONDS", "300")
    with suppress(TypeError, ValueError):
        return max(0.0, float(raw or 0.0))
    return 300.0


def _read_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_risk_state(owner: Any) -> None:
    path = Path(getattr(owner, "_state_path", ""))
    if not str(path) or not path.exists():
        return
    payload = _read_state(path)
    if not payload:
        return
    stopped = getattr(owner, "_recent_stop_thesis", None)
    payload[_RISK_KEY] = {
        "trades_today_date": getattr(owner, "_trades_today_date", None),
        "trades_today_count": int(getattr(owner, "_trades_today_count", 0) or 0),
        "recent_stop_thesis": dict(stopped) if isinstance(stopped, dict) else None,
    }
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            json.dump(payload, handle, separators=(",", ":"), default=str)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_name = handle.name
        os.replace(tmp_name, path)
    except OSError:
        if tmp_name:
            with suppress(OSError):
                os.unlink(tmp_name)


def _restore_risk_state(owner: Any) -> None:
    path = Path(getattr(owner, "_state_path", ""))
    state = _read_state(path).get(_RISK_KEY, {})
    if not isinstance(state, dict):
        return
    today = owner._trading_date_ist()
    if state.get("trades_today_date") == today:
        with suppress(TypeError, ValueError):
            owner._trades_today_date = today
            owner._trades_today_count = max(
                0, int(state.get("trades_today_count", 0) or 0)
            )
    stopped = state.get("recent_stop_thesis")
    if isinstance(stopped, dict):
        with suppress(TypeError, ValueError):
            expires_epoch = float(stopped.get("expires_epoch", 0.0) or 0.0)
            if expires_epoch > time.time():
                owner._recent_stop_thesis = dict(stopped)


def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
    _ORIGINAL_INIT(self, *args, **kwargs)
    self._recent_stop_thesis = None
    with getattr(self, "_lock"):
        _restore_risk_state(self)


def _patched_save_state(self: Any, *args: Any, **kwargs: Any) -> Any:
    result = _ORIGINAL_SAVE_STATE(self, *args, **kwargs)
    with getattr(self, "_lock"):
        _write_risk_state(self)
    return result


def _is_stop_reason(reason: object) -> bool:
    text = str(reason or "").strip().upper().replace("-", "_")
    return "STOP_LOSS" in text or text in {"SL", "STOP", "STOPLOSS"}


def _patched_close_position(
    self: Any,
    symbol: str,
    exit_price: float,
    reason: str,
    close_time: Any = None,
) -> Any:
    result = _ORIGINAL_CLOSE_POSITION(
        self, symbol, exit_price, reason, close_time=close_time
    )
    thesis = _option_thesis(symbol)
    cooldown = _cooldown_seconds()
    if thesis is not None and cooldown > 0.0 and _is_stop_reason(reason):
        underlying, option_side = thesis
        with getattr(self, "_lock"):
            self._recent_stop_thesis = {
                "underlying": underlying,
                "option_side": option_side,
                "symbol": str(symbol).strip().upper(),
                "exit_reason": str(reason),
                "expires_epoch": time.time() + cooldown,
            }
        self.save_state()
    return result


def stop_reentry_block_reason(self: Any, signal: Any) -> str | None:
    """Return an entry-only block reason for an active stop-loss thesis lock."""
    thesis = _option_thesis(getattr(signal, "symbol", None))
    if thesis is None:
        return None
    with getattr(self, "_lock"):
        stopped = getattr(self, "_recent_stop_thesis", None)
        if not isinstance(stopped, dict):
            return None
        expires_epoch = float(stopped.get("expires_epoch", 0.0) or 0.0)
        remaining = expires_epoch - time.time()
        if remaining <= 0.0:
            self._recent_stop_thesis = None
            return None
        if thesis != (
            str(stopped.get("underlying", "")),
            str(stopped.get("option_side", "")),
        ):
            return None
    return f"stop-loss thesis cooldown active: {int(remaining + 0.999)}s"


def apply_patches() -> None:
    global _PATCH_APPLIED, _ORIGINAL_INIT, _ORIGINAL_SAVE_STATE, _ORIGINAL_CLOSE_POSITION
    if _PATCH_APPLIED:
        return
    from nifty_scalper_bot.execution.position_manager import PositionManager

    if getattr(PositionManager, "_position_risk_state_patch", False):
        _PATCH_APPLIED = True
        return
    _ORIGINAL_INIT = PositionManager.__init__
    _ORIGINAL_SAVE_STATE = PositionManager.save_state
    _ORIGINAL_CLOSE_POSITION = PositionManager.close_position
    PositionManager.__init__ = _patched_init
    PositionManager.save_state = _patched_save_state
    PositionManager.close_position = _patched_close_position
    PositionManager.stop_reentry_block_reason = stop_reentry_block_reason
    PositionManager._position_risk_state_patch = True
    _PATCH_APPLIED = True


__all__ = ["apply_patches", "stop_reentry_block_reason", "_option_thesis"]
