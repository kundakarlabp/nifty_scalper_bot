"""Canonical execution package exports.

Only production runtime owners are part of the package-level public API. Older
stage names are lazy compatibility aliases so existing imports do not create a
second live lifecycle authority.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from nifty_scalper_bot.execution.adaptive_trailing import (
    AdaptiveTrailingController,
    HardenedAdaptiveTrailingController,
)
from nifty_scalper_bot.execution.bracket_manager import BoundBracketManager, BracketManager
from nifty_scalper_bot.execution.order_manager import OrderManager, RuntimeOrderManager
from nifty_scalper_bot.execution.live_safety_identity import apply_patches as _apply_live_safety_identity_patches
from nifty_scalper_bot.execution.position_identity_extension import apply_patches as _apply_position_identity_extension_patches
from nifty_scalper_bot.execution.operator_control_patch import apply_patches as _apply_operator_control_patches
from nifty_scalper_bot.execution.protective_order_intent_patch import apply_patches as _apply_protective_order_intent_patches
from nifty_scalper_bot.execution.order_entry_guard_patch import apply_patches as _apply_order_entry_guard_patches
from nifty_scalper_bot.execution.position_risk_state_patch import apply_patches as _apply_position_risk_state_patches
from nifty_scalper_bot.execution.stop_rearm_contract_patch import apply_patches as _apply_stop_rearm_contract_patches
import nifty_scalper_bot.data.quote_identity_extension as _quote_identity_extension
import nifty_scalper_bot.execution.bracket_ownership_extension as _bracket_ownership_extension
import nifty_scalper_bot.execution.broker_exposure_quarantine_extension as _broker_exposure_quarantine_extension
import nifty_scalper_bot.execution.broker_order_ledger_patch as _broker_order_ledger_patch
import nifty_scalper_bot.execution.position_manager as _position_manager
import nifty_scalper_bot.execution.trade_plan_identity_guard as _trade_plan_identity_guard


def _apply_fresh_entry_cost_basis_repair() -> None:
    """Repair only the broker-sync-before-fill fresh-entry cost-basis race."""
    cls = _position_manager.PositionManager
    attr = "_fresh_entry_cost_basis_repair_installed"
    if bool(getattr(cls, attr, False)):
        return
    original = cls._handle_filled_order

    @wraps(original)
    def _handle_filled_order(self: Any, order: Any) -> Any:
        result = original(self, order)
        if (
            getattr(result, "reason", "")
            != "entry_fill_already_reflected_by_broker_sync"
            or str(getattr(order, "intent", "") or "").upper() != "ENTRY"
            or int(getattr(order, "pre_order_quantity", 0) or 0) != 0
        ):
            return result
        position = getattr(self, "_positions", {}).get(str(getattr(order, "symbol", "")))
        cumulative_qty = int(
            getattr(order, "filled_quantity", 0) or getattr(order, "quantity", 0) or 0
        )
        fill_price = float(getattr(order, "fill_price", 0.0) or 0.0)
        side_matches = position is not None and (
            (getattr(position, "side", None) == "LONG" and getattr(order, "side", None) == "BUY")
            or (getattr(position, "side", None) == "SHORT" and getattr(order, "side", None) == "SELL")
        )
        if not side_matches or cumulative_qty <= 0 or fill_price <= 0:
            return result
        if int(getattr(position, "quantity", 0) or 0) != cumulative_qty:
            return result
        old_entry = float(getattr(position, "entry_price", 0.0) or 0.0)
        position.entry_price = fill_price
        position.order_id = str(getattr(order, "order_id", "") or "") or position.order_id
        if abs(old_entry - fill_price) > 1e-9:
            self._logger.warning(
                "ENTRY_COST_BASIS_REPAIRED_FROM_ORDER_FILL order_id=%s symbol=%s broker_sync_entry=%.2f order_fill=%.2f qty=%s",
                order.order_id,
                order.symbol,
                old_entry,
                fill_price,
                cumulative_qty,
                extra={
                    "event": "ENTRY_COST_BASIS_REPAIRED_FROM_ORDER_FILL",
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "broker_sync_entry_price": old_entry,
                    "order_fill_price": fill_price,
                    "quantity": cumulative_qty,
                },
            )
        return result

    cls._handle_filled_order = _handle_filled_order
    setattr(cls, attr, True)


_apply_live_safety_identity_patches()
_apply_position_identity_extension_patches()
_apply_fresh_entry_cost_basis_repair()
_apply_operator_control_patches()
_apply_protective_order_intent_patches()
_apply_order_entry_guard_patches()
_apply_position_risk_state_patches()
_apply_stop_rearm_contract_patches()
_quote_identity_extension.apply_patches()
_broker_exposure_quarantine_extension.apply_patches()
_broker_order_ledger_patch.apply_patches()
_bracket_ownership_extension.apply_patches()
_trade_plan_identity_guard.apply_patches()

CanonicalBracketManager = BracketManager
_COMPAT_BRACKET_ALIASES = {
    "FillIntegrityBracketManager",
    "HardenedBracketManager",
    "LedgerBracketManager",
    "RuntimeBracketManager",
}


def __getattr__(name: str) -> Any:
    if name in _COMPAT_BRACKET_ALIASES:
        return BracketManager
    raise AttributeError(name)


__all__ = [
    "AdaptiveTrailingController",
    "BoundBracketManager",
    "BracketManager",
    "CanonicalBracketManager",
    "HardenedAdaptiveTrailingController",
    "OrderManager",
    "RuntimeOrderManager",
]
