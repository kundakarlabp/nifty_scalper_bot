"""Regression tests: stale SL/TP bracket re-anchored to live protected price.

Strategy SL/TP are computed off the option premium at signal time. When the
premium moves before submission, the precomputed band can land on the wrong
side of the live protected price, which previously rejected the order with
``protected_price_invalidates_bracket`` (a lost but valid entry). The order
manager now re-anchors an *invalid* band to the live price, preserving the
plan's intended SL/TP distances, while leaving valid brackets untouched.
"""

from __future__ import annotations

import logging
import types
from typing import Any

from nifty_scalper_bot.execution.order_manager import OrderManager, TradePlan


def _plan(side: str, entry: float, sl: float, tp: float) -> TradePlan:
    return TradePlan(
        symbol="NFO:NIFTY2662324050CE",
        side=side,  # type: ignore[arg-type]
        quantity=65,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
    )


class _Stub:
    _logger = logging.getLogger("reanchor-test")


def _reanchor(plan: TradePlan, price: float) -> TradePlan:
    return OrderManager._reanchor_bracket_to_price(_Stub(), plan, price)


async def test_buy_stale_band_reanchored_preserving_distances() -> None:
    # Signal-time band [108.56, 116.93] around entry 112.7; live price jumped
    # to 138.45 -> original TP (116.93) is below live price (invalid).
    plan = _plan("BUY", entry=112.70, sl=108.56, tp=116.93)
    out = _reanchor(plan, 138.45)

    # Distances preserved: sl_dist=4.14, tp_dist=4.23
    assert out.stop_loss == 134.31  # 138.45 - 4.14
    assert out.take_profit == 142.68  # 138.45 + 4.23
    # Now a valid long bracket.
    assert out.stop_loss < 138.45 < out.take_profit


async def test_sell_stale_band_reanchored() -> None:
    # Short: valid means tp < price < sl. Signal entry 112.7, sl above, tp below.
    plan = _plan("SELL", entry=112.70, sl=116.93, tp=108.56)
    out = _reanchor(plan, 90.00)
    # sl_dist=4.23, tp_dist=4.14
    assert out.stop_loss == 94.23  # 90 + 4.23
    assert out.take_profit == 85.86  # 90 - 4.14
    assert out.take_profit < 90.00 < out.stop_loss


async def test_valid_bracket_passes_through_unchanged() -> None:
    plan = _plan("BUY", entry=130.0, sl=125.0, tp=140.0)
    out = _reanchor(plan, 131.0)  # 125 < 131 < 140 already valid
    assert out.stop_loss == 125.0
    assert out.take_profit == 140.0
    assert out is plan  # untouched


async def test_missing_levels_untouched() -> None:
    plan = TradePlan(
        symbol="NFO:NIFTY2662324050CE",
        side="BUY",
        quantity=65,
        entry_price=112.0,
        stop_loss=None,
        take_profit=None,
    )
    out = _reanchor(plan, 138.45)
    assert out is plan


async def test_submit_reanchors_instead_of_rejecting(monkeypatch: Any) -> None:
    """End-to-end: ordinary quote drift is re-anchored, not rejected."""
    from nifty_scalper_bot.execution.order_manager import OrderPreflightResult
    from nifty_scalper_bot.execution.position_manager import PositionManager
    from nifty_scalper_bot.utils.rate_limiter import RateLimiter

    class _Broker:
        def place_order(self, **k: Any) -> dict[str, Any]:
            return {"order_id": "ORD-1", "status": "success"}

    mgr = OrderManager(_Broker(), PositionManager(), RateLimiter())

    captured: dict[str, Any] = {}

    def _fake_managed(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return types.SimpleNamespace(
            accepted=True, order_id="ORD-1", reason="accepted",
            details={}, broker_attempted=True,
        )

    monkeypatch.setattr(mgr, "is_kill_switch_active", lambda: False)
    monkeypatch.setattr(
        mgr, "_validate_trade_plan",
        lambda plan: OrderPreflightResult(True, "ok", {}),
    )
    monkeypatch.setattr(mgr, "_protected_limit_price", lambda plan: 119.00)
    monkeypatch.setattr(mgr, "place_managed_order_result", _fake_managed)
    # Entry margin gate needs a resolvable lot size and a trusted balance;
    # a runner-built plan always carries resolved_lot_size in production.
    monkeypatch.setattr(mgr, "_lot_size_for_symbol", lambda symbol: 65)
    monkeypatch.setattr(
        mgr, "_resolve_available_margin", lambda **kw: (1_000_000.0, "mdm")
    )
    # Stub the sizing engine (an external collaborator of the method under
    # test) so this re-anchoring test stays deterministic and does not depend
    # on wall-clock trading-session windows.
    monkeypatch.setattr(
        mgr,
        "_margin_engine",
        types.SimpleNamespace(
            plan=lambda inputs: types.SimpleNamespace(
                ok=True, quantity=65, reason=None, est_required=7735.0
            )
        ),
    )

    plan = _plan("BUY", entry=112.70, sl=108.56, tp=116.93)
    result = mgr.submit_trade_plan_result(plan)

    assert result.reason != "protected_price_invalidates_bracket"
    assert result.accepted
    # Re-anchored levels were what got submitted.
    assert captured["entry_price"] == 119.00
    assert captured["stop_loss"] < 119.00 < captured["take_profit"]
