from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.execution.affordability import evaluate_minimum_lot_affordability
from nifty_scalper_bot.execution.margin_engine import MarginEngine, MarginInputs
from nifty_scalper_bot.risk.net_rr_gate import evaluate_final_net_rr
from nifty_scalper_bot.strategies.premium_risk_geometry import apply_cost_aware_risk_floor
from nifty_scalper_bot.strategies.signal_generator import Signal


BALANCE = 14_867.10
RISK_PCT = 2.0
LOT_SIZE = 65
RISK_BUDGET = BALANCE * RISK_PCT / 100.0


def _order_manager() -> SimpleNamespace:
    return SimpleNamespace(
        _margin_factor=1.1,
        _margin_buffer=0.9,
        resolve_lot_size=lambda _symbol: LOT_SIZE,
        _risk_manager=SimpleNamespace(
            account_balance=BALANCE,
            settings=SimpleNamespace(per_trade_risk_pct=RISK_PCT),
            _switches=SimpleNamespace(
                max_day_loss=RISK_BUDGET,
                day_loss=lambda: 0.0,
            ),
        ),
    )


def _margin_inputs(*, price: float, stop_loss: float) -> MarginInputs:
    return MarginInputs(
        symbol="NFO:NIFTY2691523400PE",
        side="BUY",
        price=price,
        stop_loss=stop_loss,
        atr=None,
        requested_qty=LOT_SIZE,
        product="NRML",
        lot_size=LOT_SIZE,
        balance=BALANCE,
        per_trade_risk_pct=RISK_PCT,
        per_trade_cap_pct=100.0,
        margin_factor=1.1,
        margin_buffer=0.9,
        contract_multiplier=float(LOT_SIZE),
        ist_now=datetime(2026, 9, 10, 10, 1, tzinfo=ZoneInfo("Asia/Kolkata")),
        min_lots_per_trade=1,
        max_lots_per_trade=1,
        atr_multiple=1.0,
    )


def test_live_cash_affordable_contract_is_not_preblocked_by_fixed_rr_cost_floor() -> None:
    decision = evaluate_minimum_lot_affordability(
        symbol="NFO:NIFTY2691523400PE",
        quote={
            "bid": 81.95,
            "ask": 82.15,
            # Representative cost-derived fixed-2R floor from the blocked live path.
            "candidate_min_risk_distance": 5.65,
        },
        order_manager=_order_manager(),
        data_hub=SimpleNamespace(get_available_balance=lambda force=False: BALANCE),
        fallback_balance=BALANCE,
    )

    assert decision.required == pytest.approx(82.15 * LOT_SIZE * 1.1)
    assert decision.required < decision.executable_capacity
    assert decision.risk_floor_affordable is False
    assert decision.affordable is True
    assert decision.capacity_blocker is None


def test_cost_compensation_never_increases_live_one_lot_stop_risk(monkeypatch) -> None:
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "1.5")
    monkeypatch.setenv("MAX_NET_RR_TARGET_UPLIFT_R", "0.35")
    entry = 82.15
    stop_distance = 4.50
    signal = Signal(
        action="BUY",
        symbol="NFO:NIFTY2691523400PE",
        quantity=LOT_SIZE,
        confidence=0.85,
        reason="VWAPPro",
        stop_loss=entry - stop_distance,
        take_profit=entry + 2.0 * stop_distance,
        metadata={
            "entry_price": entry,
            "premium_stop_distance": stop_distance,
            "premium_target_rr": 2.0,
            "invalidation_level_domain": "option_premium",
            "bracket_anchor_mode": "distance",
            "bid": 81.95,
            "ask": 82.15,
        },
    )

    adjusted = apply_cost_aware_risk_floor(
        signal,
        entry_price=entry,
        quantity=LOT_SIZE,
        half_spread=0.10,
    )

    stop_risk = (entry - adjusted.stop_loss) * LOT_SIZE
    assert stop_risk == pytest.approx(292.50)
    assert stop_risk <= RISK_BUDGET
    assert adjusted.stop_loss == pytest.approx(signal.stop_loss)
    result = evaluate_final_net_rr(adjusted)
    assert result is not None and result.allowed is True


def test_margin_engine_remains_final_2pct_authority() -> None:
    engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )

    safe = engine._max_qty_from_risk(_margin_inputs(price=82.15, stop_loss=77.65))
    unsafe = engine._max_qty_from_risk(_margin_inputs(price=82.15, stop_loss=77.15))

    assert safe == LOT_SIZE  # 4.50 * 65 = 292.50 <= 297.342
    assert unsafe == 0  # 5.00 * 65 = 325.00 > 297.342
