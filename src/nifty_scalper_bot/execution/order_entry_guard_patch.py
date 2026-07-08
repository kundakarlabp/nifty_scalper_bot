"""Pre-entry geometry guard at the final RuntimeOrderManager choke point."""

from __future__ import annotations

from contextlib import suppress
import inspect
import os
from typing import Any, Mapping

from nifty_scalper_bot.execution import order_manager_core as _core

_PATCH_APPLIED = False
_ORIGINAL_PLACE_ORDER: Any = None
_ENTRY_INTENTS = {"ENTRY", "SCALE_IN", "REVERSAL"}


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _env_float(names: tuple[str, ...], default: float) -> float:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        with suppress(Exception):
            return float(str(raw).strip())
    return default


def _entry_geometry_block_reason(
    manager: Any,
    *,
    symbol: str | None,
    side: str | None,
    price: Any,
    stop_loss: Any,
    take_profit: Any,
    intent: str | None,
) -> dict[str, Any] | None:
    del manager
    normalized_intent = str(intent or "").strip().upper()
    if normalized_intent not in _ENTRY_INTENTS:
        return None
    normalized_side = str(side or "").strip().upper()
    if normalized_side not in {"BUY", "SELL"}:
        return None

    entry = _float_or_none(price)
    sl = _float_or_none(stop_loss)
    tp = _float_or_none(take_profit)
    if entry is None or sl is None or tp is None:
        return None

    if normalized_side == "BUY":
        risk = entry - sl
        reward = tp - entry
    else:
        risk = sl - entry
        reward = entry - tp

    rr_floor = _env_float(("ENTRY_MIN_RR", "MIN_ENTRY_RR", "MIN_BRACKET_RR"), 1.5)
    min_sl_pct = max(
        0.0,
        _env_float(("ENTRY_MIN_SL_DISTANCE_PCT", "MIN_ENTRY_SL_DISTANCE_PCT"), 0.10),
    )
    sl_pct = (abs(risk) / entry * 100.0) if entry > 0 else 0.0

    if risk <= 0 or reward <= 0:
        return {
            "block_reason": "entry_invalid_bracket_geometry",
            "symbol": symbol,
            "entry": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "risk": risk,
            "reward": reward,
            "rr": 0.0,
            "rr_floor": rr_floor,
            "sl_distance_pct": sl_pct,
        }

    rr = reward / risk
    if rr < rr_floor:
        return {
            "block_reason": "entry_rr_below_floor",
            "symbol": symbol,
            "entry": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "risk": risk,
            "reward": reward,
            "rr": rr,
            "rr_floor": rr_floor,
            "sl_distance_pct": sl_pct,
        }

    if min_sl_pct > 0 and sl_pct < min_sl_pct:
        return {
            "block_reason": "entry_sl_distance_too_tight",
            "symbol": symbol,
            "entry": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "risk": risk,
            "reward": reward,
            "rr": rr,
            "rr_floor": rr_floor,
            "sl_distance_pct": sl_pct,
            "min_sl_distance_pct": min_sl_pct,
        }
    return None


def _bind_place_order(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        signature = inspect.signature(_core.OrderManager.place_order)
        bound = signature.bind_partial(None, *args, **dict(kwargs))
        return {key: value for key, value in bound.arguments.items() if key != "self"}
    except Exception:
        return None


def _live_mode(self: Any) -> bool:
    checker = getattr(self, "is_live_mode", None)
    if callable(checker):
        with suppress(Exception):
            return bool(checker())
    return str(os.getenv("EXECUTION_MODE", "")).strip().upper() == "LIVE"


def _patched_place_order(self: Any, *args: Any, **kwargs: Any) -> Any:
    if not _live_mode(self):
        return _ORIGINAL_PLACE_ORDER(self, *args, **kwargs)
    values = _bind_place_order(args, kwargs) or dict(kwargs)
    reason = _entry_geometry_block_reason(
        self,
        symbol=values.get("symbol"),
        side=values.get("side"),
        price=values.get("price"),
        stop_loss=values.get("stop_loss"),
        take_profit=values.get("take_profit"),
        intent=values.get("intent"),
    )
    if reason is not None:
        setter = getattr(self, "set_last_skip_reason", None)
        if callable(setter):
            with suppress(Exception):
                setter(str(reason["block_reason"]))
        self._last_order_decision = {
            "allowed": False,
            "block_reason": reason["block_reason"],
            "details": reason,
            "broker_attempted": False,
            "final_order_gate": True,
        }
        logger = getattr(self, "_logger", None)
        log = getattr(logger, "critical", None)
        if callable(log):
            log(
                "ENTRY_GEOMETRY_BLOCKED symbol=%s reason=%s entry=%s sl=%s tp=%s rr=%s floor=%s",
                reason.get("symbol"),
                reason.get("block_reason"),
                reason.get("entry"),
                reason.get("stop_loss"),
                reason.get("take_profit"),
                round(float(reason.get("rr") or 0.0), 3),
                reason.get("rr_floor"),
                extra={"event": "ENTRY_GEOMETRY_BLOCKED", **reason},
            )
        return None
    return _ORIGINAL_PLACE_ORDER(self, *args, **kwargs)


def apply_patches() -> None:
    global _PATCH_APPLIED, _ORIGINAL_PLACE_ORDER
    if _PATCH_APPLIED:
        return
    from nifty_scalper_bot.execution.runtime_order_manager import RuntimeOrderManager

    if getattr(RuntimeOrderManager, "_order_entry_geometry_patch", False):
        _PATCH_APPLIED = True
        return
    _ORIGINAL_PLACE_ORDER = RuntimeOrderManager.place_order
    RuntimeOrderManager.place_order = _patched_place_order
    RuntimeOrderManager._order_entry_geometry_patch = True
    _PATCH_APPLIED = True


__all__ = ["apply_patches", "_entry_geometry_block_reason"]
