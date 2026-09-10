"""Position size has exactly one owner: the risk engine.

The strategy layer used to rescale ``Signal.quantity`` by the regime
multiplier. That could never protect capital -- the requested size is one lot
and ``max(1, round(1 * 0.6))`` is still one lot -- while an expansive
multiplier could still raise size outside the per-trade risk budget. The
regime scale is evidence for the risk layer, not a sizing decision.
"""

from __future__ import annotations

import ast
from pathlib import Path

from nifty_scalper_bot.core.strategy_manager import StrategyManager

_STRATEGY_MANAGER = Path("src/nifty_scalper_bot/core/strategy_manager.py")


def _signal_quantity_arguments(source: str) -> list[ast.expr]:
    tree = ast.parse(source)
    quantities: list[ast.expr] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = getattr(callee, "id", None) or getattr(callee, "attr", None)
        if name != "Signal":
            continue
        for keyword in node.keywords:
            if keyword.arg == "quantity":
                quantities.append(keyword.value)
    return quantities


def test_strategy_manager_never_derives_a_scaled_quantity() -> None:
    source = _STRATEGY_MANAGER.read_text(encoding="utf-8")
    quantities = _signal_quantity_arguments(source)
    assert quantities, "expected Signal(...) constructions to inspect"

    for expression in quantities:
        rendered = ast.unparse(expression)
        # A pass-through reads the quantity straight off the incoming signal.
        # Anything else -- an arithmetic expression, or a local name holding a
        # derived value such as ``scaled_quantity`` -- is the strategy layer
        # deciding size, which belongs to the risk engine.
        assert isinstance(expression, ast.Attribute) and expression.attr == "quantity", (
            f"quantity must be passed through unchanged, got: {rendered}"
        )


def test_regime_scale_is_still_published_as_sizing_evidence() -> None:
    """Removing the multiplication must not remove the observation."""
    manager = StrategyManager([], None, None)

    assert manager._extract_regime_scale({}) == 1.0
    assert manager._extract_regime_scale({"sizing_multiplier": 0.7}) == 0.7
    assert "regime_scale" in _STRATEGY_MANAGER.read_text(encoding="utf-8")


def test_risk_layer_remains_the_regime_sizing_owner() -> None:
    from nifty_scalper_bot.risk.position_sizing import PositionSizer

    sizer = PositionSizer()
    defensive = sizer.size(
        equity=1_000_000.0,
        risk_per_trade_pct=0.02,
        entry=100.0,
        stop_loss=94.0,
        lot_size=75,
        confidence=1.0,
        regime_multiplier=0.6,
        max_lots=100,
        available_margin=10_000_000.0,
        margin_per_lot=10_000.0,
    )
    neutral = sizer.size(
        equity=1_000_000.0,
        risk_per_trade_pct=0.02,
        entry=100.0,
        stop_loss=94.0,
        lot_size=75,
        confidence=1.0,
        regime_multiplier=1.0,
        max_lots=100,
        available_margin=10_000_000.0,
        margin_per_lot=10_000.0,
    )

    # Unlike the strategy-layer seam, the risk engine scales the rupee risk
    # budget, so a defensive regime genuinely reduces exposure.
    assert defensive.allowed and neutral.allowed
    assert defensive.qty < neutral.qty
