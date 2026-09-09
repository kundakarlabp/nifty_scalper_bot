from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.affordability import (
    evaluate_minimum_lot_affordability,
)
from nifty_scalper_bot.strategies.runner import StrategyRunner
from nifty_scalper_bot.strategies.trade_selector import TradeCandidateSelector


class _Logger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _order_manager(*, balance: float = 15_000.0):
    return SimpleNamespace(
        _margin_factor=1.1,
        _margin_buffer=0.9,
        resolve_lot_size=lambda _symbol: 65,
        _risk_manager=SimpleNamespace(
            account_balance=balance,
            settings=SimpleNamespace(per_trade_risk_pct=2.0),
            _switches=SimpleNamespace(
                max_day_loss=balance * 0.02,
                day_loss=lambda: 0.0,
            ),
        ),
    )


def test_affordability_rejects_cash_affordable_contract_when_one_lot_stop_risk_exceeds_budget() -> None:
    decision = evaluate_minimum_lot_affordability(
        symbol="NFO:NIFTY2691523500PE",
        quote={
            "bid": 105.20,
            "ask": 105.30,
            "candidate_entry_price": 105.55,
            "candidate_stop_loss": 99.40,
        },
        order_manager=_order_manager(),
        data_hub=SimpleNamespace(get_available_balance=lambda force=False: 15_000.0),
        fallback_balance=15_000.0,
    )

    assert decision.cash_affordable is True
    assert decision.risk_affordable is False
    assert decision.affordable is False
    assert decision.reason == "minimum_lot_unaffordable"
    assert decision.capacity_blocker == "stop_risk"
    assert decision.one_lot_stop_risk == pytest.approx((105.55 - 99.40) * 65)
    assert decision.effective_one_lot_risk_budget == pytest.approx(300.0)


def test_candidate_capacity_falls_back_when_preferred_contract_breaks_stop_risk_budget() -> None:
    runner = object.__new__(StrategyRunner)
    runner._logger = _Logger()
    runner._order_manager = _order_manager()
    runner._data_hub = SimpleNamespace(
        get_available_balance=lambda force=False: 15_000.0
    )
    runner._risk_manager = SimpleNamespace(available_balance=15_000.0)
    runner._is_symbol_execution_ready = lambda _symbol: True
    runner._ensure_symbol_execution_ready_for_order = (
        lambda _symbol, trace_id=None: True
    )

    preferred = SimpleNamespace(symbol="NFO:NIFTY2691523500PE")
    fallback = SimpleNamespace(symbol="NFO:NIFTY2691523450PE")
    snapshots = [
        {
            "symbol": preferred.symbol,
            "bid": 105.20,
            "ask": 105.30,
            "ltp": 105.25,
            "candidate_entry_price": 105.55,
            "candidate_stop_loss": 99.40,
        },
        {
            "symbol": fallback.symbol,
            "bid": 79.90,
            "ask": 80.00,
            "ltp": 79.95,
            "candidate_entry_price": 80.00,
            "candidate_stop_loss": 76.50,
        },
    ]

    selected, decisions = runner._select_capital_eligible_candidate(
        ranked_candidates=[fallback, preferred],
        candidate_snapshots=snapshots,
        is_live_mode=True,
        trace_id="risk-fallback",
        preferred_symbol=preferred.symbol,
    )

    assert selected is fallback
    assert decisions[preferred.symbol]["cash_affordable"] is True
    assert decisions[preferred.symbol]["risk_affordable"] is False
    assert decisions[preferred.symbol]["capacity_blocker"] == "stop_risk"
    assert decisions[fallback.symbol]["risk_affordable"] is True


def test_trade_selector_exposes_its_existing_geometry_to_capacity_screen(
    monkeypatch,
) -> None:
    import nifty_scalper_bot.strategies.trade_selector as selector_module

    monkeypatch.setattr(selector_module, "expiry_theta_block", lambda: (False, "ok"))
    monkeypatch.setattr(selector_module, "midday_pause_block", lambda: (False, "ok"))
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.delenv("ENABLE_LIVE", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    monkeypatch.setenv("NIFTY_LOT_SIZE", "65")

    snapshot = {
        "symbol": "NFO:NIFTY2691523500PE",
        "side": "PE",
        "option_type": "PE",
        "strike": 23500,
        "bid": 105.20,
        "ask": 105.30,
        "ltp": 105.25,
        "tick_age_ms": 100.0,
        "real_ticks_last_60s": 5,
        "atr_option": 2.0,
    }
    selector = TradeCandidateSelector(
        min_option_premium=40.0,
        max_option_premium=650.0,
        max_tick_age_s=10.0,
        max_option_spread_pct=5.0,
        require_real_ticks_last_60s=1,
    )

    ranked = selector.select_ranked_candidates(
        direction_bias="PE",
        atm_strike=23500,
        snapshots=[snapshot],
        gross_rr=2.0,
    )

    assert ranked
    assert snapshot["candidate_entry_price"] == pytest.approx(ranked[0].entry_price)
    assert snapshot["candidate_stop_loss"] == pytest.approx(ranked[0].stop_loss)
    assert snapshot["candidate_target"] == pytest.approx(ranked[0].target)
    assert snapshot["candidate_rr"] == pytest.approx(ranked[0].rr)
