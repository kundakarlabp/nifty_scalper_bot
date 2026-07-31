"""Preserve strategy-declared option-premium risk geometry.

Elite setup strategies publish ``premium_stop_distance`` in option-premium
points. The runner's legacy premium helper only consumes ``premium_stop_pct``;
therefore an otherwise valid absolute invalidation distance can disappear before
TradePlan construction. Patch the StrategyManager output once, before runner
normalisation, without changing strategies or bracket ownership.
"""

from __future__ import annotations

import dataclasses
from contextlib import suppress
from typing import Any, Mapping

_PATCH_APPLIED = False
_ORIGINAL_GENERATE_SIGNAL: Any = None


def _positive_float(value: Any) -> float | None:
    with suppress(TypeError, ValueError):
        parsed = float(value)
        if parsed > 0.0:
            return parsed
    return None


def apply_premium_risk_contract(signal: Any, premium: float) -> Any:
    """Fill missing option-premium SL/TP from explicit strategy distance."""
    if signal is None or premium <= 0.0:
        return signal
    action = str(getattr(signal, "action", "") or "").upper()
    symbol = str(getattr(signal, "symbol", "") or "").upper()
    if action not in {"BUY", "SELL"} or not symbol.endswith(("CE", "PE")):
        return signal

    metadata = getattr(signal, "metadata", {})
    if not isinstance(metadata, Mapping):
        return signal
    distance = _positive_float(metadata.get("premium_stop_distance"))
    if distance is None:
        return signal
    domain = str(metadata.get("invalidation_level_domain") or "option_premium").lower()
    if domain != "option_premium":
        return signal

    existing_sl = _positive_float(getattr(signal, "stop_loss", None))
    existing_tp = _positive_float(getattr(signal, "take_profit", None))
    rr = _positive_float(metadata.get("premium_target_rr")) or 2.0

    if action == "BUY":
        stop_loss = existing_sl or max(0.05, premium - distance)
        take_profit = existing_tp or premium + distance * rr
        valid = stop_loss < premium < take_profit
    else:
        stop_loss = existing_sl or premium + distance
        take_profit = existing_tp or max(0.05, premium - distance * rr)
        valid = take_profit < premium < stop_loss
    if not valid:
        return signal

    updated_metadata = dict(metadata)
    updated_metadata.setdefault("premium_risk_contract_applied", True)
    updated_metadata.setdefault("premium_risk_source", "premium_stop_distance")
    updated_metadata.setdefault("premium_risk_reference_price", float(premium))
    return dataclasses.replace(
        signal,
        stop_loss=float(stop_loss),
        take_profit=float(take_profit),
        metadata=updated_metadata,
    )


def _patched_generate_signal(self: Any, symbol: str, current_price: float, *args: Any, **kwargs: Any) -> Any:
    signal = _ORIGINAL_GENERATE_SIGNAL(self, symbol, current_price, *args, **kwargs)
    return apply_premium_risk_contract(signal, float(current_price or 0.0))


def apply_patches() -> None:
    global _PATCH_APPLIED, _ORIGINAL_GENERATE_SIGNAL
    if _PATCH_APPLIED:
        return
    from nifty_scalper_bot.core.strategy_manager import StrategyManager

    if getattr(StrategyManager, "_premium_risk_contract_patch", False):
        _PATCH_APPLIED = True
        return
    _ORIGINAL_GENERATE_SIGNAL = StrategyManager.generate_signal
    StrategyManager.generate_signal = _patched_generate_signal
    StrategyManager._premium_risk_contract_patch = True
    _PATCH_APPLIED = True


__all__ = ["apply_patches", "apply_premium_risk_contract"]
