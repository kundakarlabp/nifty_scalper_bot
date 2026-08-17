"""Persist entry-risk state and keep PositionManager risk state restart-safe.

This patch stays deliberately narrow:
* persist the existing PositionManager daily-entry counter in its existing JSON
  state file so an intraday process restart cannot reset max_trades_per_day;
* after a stop-loss exit, temporarily block a new option entry for the same
  underlying and option side, including a strike change;
* when a validated broker position snapshot explicitly contains realised P&L,
  make that session-normalised broker value authoritative over a divergent local
  ledger while retaining local P&L as the fallback for positions-only snapshots.

Protective/reducing orders remain outside the entry-only guard.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping, Sequence

from nifty_scalper_bot.utils.symbols import is_strategy_instrument, normalize_symbol

# Standalone "SL" token (SL Hit, HARD_SL_BREACH, FORCED_SL_EXIT, WATCHDOG_HARD_SL)
# or an explicit STOP LOSS / STOP_LOSS phrase. "SLIPPAGE" must not match.
_STOP_REASON_RE = re.compile(r"(?<![A-Z0-9])SL(?![A-Z0-9])|STOP[_ ]?LOSS")
_PATCH_APPLIED = False
_ORIGINAL_INIT: Any = None
_ORIGINAL_SAVE_STATE: Any = None
_ORIGINAL_CLOSE_POSITION: Any = None
_ORIGINAL_REFRESH_REALIZED_PNL: Any = None
_ORIGINAL_SYNCHRONIZE_WITH_BROKER: Any = None
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
    circuit = getattr(owner, "_risk_circuit_state", None)
    payload[_RISK_KEY] = {
        "trades_today_date": getattr(owner, "_trades_today_date", None),
        "trades_today_count": int(getattr(owner, "_trades_today_count", 0) or 0),
        "recent_stop_thesis": dict(stopped) if isinstance(stopped, dict) else None,
        "risk_circuit": dict(circuit) if isinstance(circuit, dict) else None,
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
    circuit = state.get("risk_circuit")
    if isinstance(circuit, dict) and str(circuit.get("trading_date") or "") == today:
        owner._risk_circuit_state = dict(circuit)
    stopped = state.get("recent_stop_thesis")
    if isinstance(stopped, dict):
        with suppress(TypeError, ValueError):
            expires_epoch = float(stopped.get("expires_epoch", 0.0) or 0.0)
            if expires_epoch > time.time():
                owner._recent_stop_thesis = dict(stopped)


def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
    # The original initializer can load persisted P&L and therefore invoke the
    # patched refresh before it returns. Default to local fallback until a fresh
    # validated broker snapshot explicitly proves realised-P&L authority.
    self._broker_realized_authoritative = False
    _ORIGINAL_INIT(self, *args, **kwargs)
    self._recent_stop_thesis = None
    self._risk_circuit_state = {}
    with getattr(self, "_lock"):
        _restore_risk_state(self)


def _patched_save_state(self: Any, *args: Any, **kwargs: Any) -> Any:
    result = _ORIGINAL_SAVE_STATE(self, *args, **kwargs)
    with getattr(self, "_lock"):
        _write_risk_state(self)
    return result


def _snapshot_has_authoritative_realized(
    broker_positions: Sequence[Mapping[str, object]],
) -> bool:
    """Return True only when a managed MIS row explicitly carries realised P&L."""
    for record in broker_positions:
        if not isinstance(record, Mapping):
            continue
        product = str(record.get("product") or "").strip().upper()
        if product != "MIS":
            continue
        raw_symbol = record.get("symbol") or record.get("tradingsymbol")
        symbol = normalize_symbol(str(raw_symbol or ""))
        if not symbol or not is_strategy_instrument(symbol):
            continue
        if "realised" in record or "realized" in record:
            return True
    return False


def _patched_synchronize_with_broker(
    self: Any, broker_positions: Sequence[Mapping[str, object]]
) -> Any:
    """Carry explicit broker P&L authority into the existing reconciliation path."""
    rows = list(broker_positions)
    previous_authority = bool(getattr(self, "_broker_realized_authoritative", False))
    self._broker_realized_authoritative = _snapshot_has_authoritative_realized(rows)
    try:
        return _ORIGINAL_SYNCHRONIZE_WITH_BROKER(self, rows)
    except Exception:
        # A rejected/invalid broker snapshot must never change P&L authority.
        self._broker_realized_authoritative = previous_authority
        raise


def _patched_refresh_realized_pnl_locked(self: Any) -> None:
    """Prefer validated broker session P&L only when the latest snapshot proves it."""
    _ORIGINAL_REFRESH_REALIZED_PNL(self)
    if not bool(getattr(self, "_broker_realized_authoritative", False)):
        return
    broker_realized = getattr(self, "_broker_realized_pnl", None)
    baseline = getattr(self, "_session_opening_realized_baseline", None)
    if broker_realized is None or baseline is None:
        return
    with suppress(TypeError, ValueError):
        broker_confirmed = float(broker_realized) - float(baseline)
        local_confirmed = float(getattr(self, "_local_realized_pnl", 0.0) or 0.0)
        self._authoritative_realized_pnl = broker_confirmed
        self._daily_realized_pnl = broker_confirmed
        self._pnl_authority = "validated_broker_positions"
        self._pnl_reconciliation_status = (
            "broker_authoritative_mismatch"
            if abs(local_confirmed - broker_confirmed) > 1.0
            else "matched"
        )


def _is_stop_reason(reason: object) -> bool:
    """Classify an exit reason as a stop-loss exit.

    The live bracket manager emits free-text reasons such as ``"SL Hit (91.4 <=
    92.0)"``, ``"HARD_SL_BREACH"`` and ``"FORCED_SL_EXIT"``. A plain
    ``"STOP_LOSS" in text`` test misses all of them, so the stop guard never
    latched on real exits. Match a standalone ``SL`` token instead, while
    keeping the legacy exact values.
    """
    text = str(reason or "").strip().upper().replace("-", "_")
    if not text:
        return False
    if text in {"SL", "STOP", "STOPLOSS"}:
        return True
    return bool(_STOP_REASON_RE.search(text))


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


def get_risk_circuit_state(self: Any) -> dict[str, Any]:
    """Return the persisted same-day risk-circuit runtime state."""
    with getattr(self, "_lock"):
        state = getattr(self, "_risk_circuit_state", None)
        return dict(state) if isinstance(state, dict) else {}


def persist_risk_circuit_state(self: Any, **values: Any) -> None:
    """Merge and durably store risk-circuit runtime state for today."""
    with getattr(self, "_lock"):
        state = getattr(self, "_risk_circuit_state", None)
        state = dict(state) if isinstance(state, dict) else {}
        state.update(values)
        state["trading_date"] = self._trading_date_ist()
        self._risk_circuit_state = state
    self.save_state()


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
    global _PATCH_APPLIED
    global _ORIGINAL_INIT, _ORIGINAL_SAVE_STATE, _ORIGINAL_CLOSE_POSITION
    global _ORIGINAL_REFRESH_REALIZED_PNL, _ORIGINAL_SYNCHRONIZE_WITH_BROKER
    if _PATCH_APPLIED:
        return
    from nifty_scalper_bot.execution.position_manager import PositionManager

    if getattr(PositionManager, "_position_risk_state_patch", False):
        _PATCH_APPLIED = True
        return
    _ORIGINAL_INIT = PositionManager.__init__
    _ORIGINAL_SAVE_STATE = PositionManager.save_state
    _ORIGINAL_CLOSE_POSITION = PositionManager.close_position
    _ORIGINAL_REFRESH_REALIZED_PNL = PositionManager._refresh_realized_pnl_locked
    _ORIGINAL_SYNCHRONIZE_WITH_BROKER = PositionManager.synchronize_with_broker
    PositionManager.__init__ = _patched_init
    PositionManager.save_state = _patched_save_state
    PositionManager.close_position = _patched_close_position
    PositionManager._refresh_realized_pnl_locked = _patched_refresh_realized_pnl_locked
    PositionManager.synchronize_with_broker = _patched_synchronize_with_broker
    PositionManager.stop_reentry_block_reason = stop_reentry_block_reason
    PositionManager.get_risk_circuit_state = get_risk_circuit_state
    PositionManager.persist_risk_circuit_state = persist_risk_circuit_state
    PositionManager._position_risk_state_patch = True
    _PATCH_APPLIED = True


__all__ = [
    "apply_patches",
    "stop_reentry_block_reason",
    "get_risk_circuit_state",
    "persist_risk_circuit_state",
    "_option_thesis",
    "_snapshot_has_authoritative_realized",
]
