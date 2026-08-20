"""Native final-entry identity and gross-geometry validation."""

from __future__ import annotations

import os
from contextlib import suppress
from typing import Any, Mapping

from nifty_scalper_bot.risk.cost_model import minimum_net_reward_risk
from nifty_scalper_bot.utils.symbols import normalize_symbol

ENTRY_INTENTS = {"ENTRY", "SCALE_IN", "REVERSAL"}
_EXIT_TAG_TOKENS = ("exit", "stop", "target", "square", "guard")


def _positive(value: Any) -> float | None:
    with suppress(TypeError, ValueError):
        parsed = float(value)
        if parsed > 0.0:
            return parsed
    return None


def entry_identity_block_reason(
    *, intent: Any, tag: Any, symbol: Any
) -> dict[str, Any] | None:
    normalized_intent = str(intent or "").strip().upper()
    if normalized_intent not in ENTRY_INTENTS:
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


def entry_geometry_block_reason(
    *,
    symbol: str | None,
    side: str | None,
    price: Any,
    stop_loss: Any,
    take_profit: Any,
    intent: str | None,
) -> dict[str, Any] | None:
    normalized_intent = str(intent or "").strip().upper()
    normalized_side = str(side or "").strip().upper()
    if normalized_intent not in ENTRY_INTENTS or normalized_side not in {"BUY", "SELL"}:
        return None

    rr_floor = minimum_net_reward_risk()
    stop = _positive(stop_loss)
    if stop is None:
        return {
            "block_reason": "entry_stop_loss_required",
            "symbol": symbol,
            "entry": _positive(price),
            "stop_loss": stop_loss,
            "take_profit": _positive(take_profit),
            "rr": 0.0,
            "rr_floor": rr_floor,
        }
    entry = _positive(price)
    target = _positive(take_profit)
    if entry is None or target is None:
        return None
    risk, reward = (
        (entry - stop, target - entry)
        if normalized_side == "BUY"
        else (stop - entry, entry - target)
    )
    sl_pct = abs(risk) / entry * 100.0
    rr = reward / risk if risk > 0.0 and reward > 0.0 else 0.0
    details = {
        "symbol": symbol,
        "entry": entry,
        "stop_loss": stop,
        "take_profit": target,
        "risk": risk,
        "reward": reward,
        "rr": rr,
        "rr_floor": rr_floor,
        "sl_distance_pct": sl_pct,
    }
    if risk <= 0.0 or reward <= 0.0:
        return {"block_reason": "entry_invalid_bracket_geometry", **details}
    if rr < rr_floor:
        return {"block_reason": "entry_rr_below_floor", **details}
    raw_min_sl_pct = os.getenv(
        "ENTRY_MIN_SL_DISTANCE_PCT",
        os.getenv("MIN_ENTRY_SL_DISTANCE_PCT", "0.10"),
    )
    with suppress(TypeError, ValueError):
        min_sl_pct = max(0.0, float(raw_min_sl_pct or 0.10))
        if min_sl_pct > 0.0 and sl_pct < min_sl_pct:
            return {
                "block_reason": "entry_sl_distance_too_tight",
                **details,
                "min_sl_distance_pct": min_sl_pct,
            }
    return None


def release_prebroker_entry_reservation(
    manager: Any, values: Mapping[str, Any]
) -> bool:
    intent = getattr(values.get("intent"), "value", values.get("intent"))
    if str(intent or "").strip().upper() not in ENTRY_INTENTS:
        return False
    decision = getattr(manager, "_last_order_decision", None)
    if not isinstance(decision, Mapping) or decision.get("allowed") is not False:
        return False
    if bool(decision.get("broker_attempted")):
        return False
    symbol = normalize_symbol(str(values.get("symbol") or ""))
    reservations = getattr(manager, "_entries_in_flight", None)
    if not symbol or not isinstance(reservations, dict) or symbol not in reservations:
        return False
    lock = getattr(manager, "_lock", None)
    if lock is None:
        reservations.pop(symbol, None)
    else:
        with lock:
            reservations.pop(symbol, None)
    return True


__all__ = [
    "ENTRY_INTENTS",
    "entry_geometry_block_reason",
    "entry_identity_block_reason",
    "release_prebroker_entry_reservation",
]
