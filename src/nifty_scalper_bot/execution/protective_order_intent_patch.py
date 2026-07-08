"""Protective order intent SSOT patch.

RuntimeOrderManager is the final broker-order choke point.  Protective exits must
arrive there with explicit EXIT/REDUCE intent before the native entry gate runs;
tag text alone is not a live-mode proof of safety.
"""

from __future__ import annotations

from contextlib import suppress
import inspect
from typing import Any, Mapping

from nifty_scalper_bot.execution import order_manager_core as _core

_PATCH_APPLIED = False
_ORIGINAL_PLACE_ORDER: Any = None

_EXIT_TAG_PREFIXES = (
    "EXIT",
    "EXIT_",
    "SL_",
    "TP_",
    "EOD_",
)
_REDUCE_TAG_PREFIXES = (
    "FLATTEN",
    "EXIT_FLATTEN",
    "SQUAREOFF",
    "PANIC",
)


def _truthy_check_risk_disabled(value: Any) -> bool:
    return value is False or str(value).strip().lower() in {"0", "false", "no", "off"}


def _normalise_protective_intent_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(kwargs)
    current_intent = str(cleaned.get("intent") or "").strip().upper()
    if current_intent:
        cleaned["intent"] = current_intent
        return cleaned

    tag = str(cleaned.get("tag") or "").strip().upper()
    check_risk = cleaned.get("check_risk", True)
    if not tag or not _truthy_check_risk_disabled(check_risk):
        return cleaned

    if tag.startswith(_REDUCE_TAG_PREFIXES):
        cleaned["intent"] = "REDUCE"
        cleaned.setdefault("strategy_name", "operator_flatten")
    elif tag.startswith(_EXIT_TAG_PREFIXES) or tag.startswith("EXIT_") or tag.startswith("EXIT"):
        cleaned["intent"] = "EXIT"
        cleaned.setdefault("strategy_name", "protective_exit")
    return cleaned


def _bind_place_order(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        signature = inspect.signature(_core.OrderManager.place_order)
        bound = signature.bind_partial(None, *args, **dict(kwargs))
        return {key: value for key, value in bound.arguments.items() if key != "self"}
    except Exception:
        return None


def _patched_place_order(self: Any, *args: Any, **kwargs: Any) -> Any:
    values = _bind_place_order(args, kwargs)
    if values is None:
        return _ORIGINAL_PLACE_ORDER(self, *args, **_normalise_protective_intent_kwargs(kwargs))
    normalised = _normalise_protective_intent_kwargs(values)
    if normalised != values:
        logger = getattr(self, "_logger", None)
        log = getattr(logger, "info", None)
        if callable(log):
            with suppress(Exception):
                log(
                    "PROTECTIVE_ORDER_INTENT_NORMALISED symbol=%s tag=%s intent=%s",
                    normalised.get("symbol"),
                    normalised.get("tag"),
                    normalised.get("intent"),
                    extra={
                        "event": "PROTECTIVE_ORDER_INTENT_NORMALISED",
                        "symbol": normalised.get("symbol"),
                        "tag": normalised.get("tag"),
                        "intent": normalised.get("intent"),
                    },
                )
    return _ORIGINAL_PLACE_ORDER(self, **normalised)


def apply_patches() -> None:
    global _PATCH_APPLIED, _ORIGINAL_PLACE_ORDER
    if _PATCH_APPLIED:
        return
    from nifty_scalper_bot.execution.runtime_order_manager import RuntimeOrderManager

    if getattr(RuntimeOrderManager, "_protective_order_intent_patch", False):
        _PATCH_APPLIED = True
        return
    _ORIGINAL_PLACE_ORDER = RuntimeOrderManager.place_order
    RuntimeOrderManager.place_order = _patched_place_order
    RuntimeOrderManager._protective_order_intent_patch = True
    _PATCH_APPLIED = True


__all__ = ["apply_patches", "_normalise_protective_intent_kwargs"]
