"""Compatibility exports for the native entry-geometry guard."""

from __future__ import annotations

from contextlib import suppress
import os
from typing import Any, Mapping

from nifty_scalper_bot.risk.cost_model import minimum_net_reward_risk
from nifty_scalper_bot.utils.symbols import normalize_symbol

_ENTRY_INTENTS = {"ENTRY", "SCALE_IN", "REVERSAL"}
_EXIT_TAG_TOKENS = ("exit", "stop", "target", "square", "guard")


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


def _entry_identity_block_reason(
    *, intent: Any, tag: Any, symbol: Any
) -> dict[str, Any] | None:
    """Reject explicit entries whose legacy tag would masquerade as an exit."""
    normalized_intent = str(intent or "").strip().upper()
    if normalized_intent not in _ENTRY_INTENTS:
        return None
    normalized_tag = str(tag or "").strip().lower()
    matched = next((token for token in _EXIT_TAG_TOKENS if token in normalized_tag), None)
    if matched is None:
        return None
    return {
        "block_reason": "entry_exit_tag_conflict",
        "symbol": symbol,
        "intent": normalized_intent,
        "tag": tag,
        "matched_exit_token": matched,
    }


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

    sl = _float_or_none(stop_loss)
    if sl is None:
        return {
            "block_reason": "entry_stop_loss_required",
            "symbol": symbol,
            "entry": _float_or_none(price),
            "stop_loss": stop_loss,
            "take_profit": _float_or_none(take_profit),
            "rr": 0.0,
            "rr_floor": minimum_net_reward_risk(),
        }

    entry = _float_or_none(price)
    tp = _float_or_none(take_profit)
    if entry is None or tp is None:
        return None

    if normalized_side == "BUY":
        risk = entry - sl
        reward = tp - entry
    else:
        risk = sl - entry
        reward = entry - tp

    rr_floor = minimum_net_reward_risk()
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


def _release_prebroker_entry_reservation(self: Any, values: Mapping[str, Any]) -> bool:
    """Release only an explicitly rejected entry that never reached the broker."""
    intent = getattr(values.get("intent"), "value", values.get("intent"))
    if str(intent or "").strip().upper() not in _ENTRY_INTENTS:
        return False
    decision = getattr(self, "_last_order_decision", None)
    if not isinstance(decision, Mapping) or decision.get("allowed") is not False:
        return False
    if bool(decision.get("broker_attempted")):
        return False
    symbol = normalize_symbol(str(values.get("symbol") or ""))
    reservations = getattr(self, "_entries_in_flight", None)
    if not symbol or not isinstance(reservations, dict) or symbol not in reservations:
        return False
    lock = getattr(self, "_lock", None)
    if lock is None:
        reservations.pop(symbol, None)
    else:
        with lock:
            reservations.pop(symbol, None)
    return True


def _record_entry_block(self: Any, reason: Mapping[str, Any]) -> None:
    setter = getattr(self, "set_last_skip_reason", None)
    if callable(setter):
        with suppress(Exception):
            setter(str(reason["block_reason"]))
    self._last_order_decision = {
        "allowed": False,
        "block_reason": reason["block_reason"],
        "details": dict(reason),
        "broker_attempted": False,
        "final_order_gate": True,
    }
    logger = getattr(self, "_logger", None)
    log = getattr(logger, "critical", None)
    if not callable(log):
        return
    if reason.get("block_reason") == "entry_exit_tag_conflict":
        log(
            "ENTRY_IDENTITY_BLOCKED symbol=%s reason=%s intent=%s tag=%s",
            reason.get("symbol"),
            reason.get("block_reason"),
            reason.get("intent"),
            reason.get("tag"),
            extra={"event": "ENTRY_IDENTITY_BLOCKED", **dict(reason)},
        )
        return
    log(
        "ENTRY_GEOMETRY_BLOCKED symbol=%s reason=%s entry=%s sl=%s tp=%s rr=%s floor=%s",
        reason.get("symbol"),
        reason.get("block_reason"),
        reason.get("entry"),
        reason.get("stop_loss"),
        reason.get("take_profit"),
        round(float(reason.get("rr") or 0.0), 3),
        reason.get("rr_floor"),
        extra={"event": "ENTRY_GEOMETRY_BLOCKED", **dict(reason)},
    )


def apply_patches() -> None:
    """Compatibility no-op; OrderManager.place_order owns the guard natively."""


__all__ = [
    "apply_patches",
    "_entry_geometry_block_reason",
    "_entry_identity_block_reason",
    "_record_entry_block",
    "_release_prebroker_entry_reservation",
]
