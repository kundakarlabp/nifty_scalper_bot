from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.affordability import (
    evaluate_minimum_lot_affordability,
)
from nifty_scalper_bot.risk.net_rr_gate import evaluate_final_net_rr
from nifty_scalper_bot.risk.risk_manager import OrderSignal
from nifty_scalper_bot.strategies.runner import StrategyRunner
from nifty_scalper_bot.strategies.trade_selector import TradeCandidateSelector


class _Logger:
    def __init__(self) -> None:
        self.info_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        self.info_calls.append((args, kwargs))

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


def test_affordability_keeps_theoretical_cost_floor_diagnostic_only() -> None:
    """A fixed-RR cost floor is not the strategy's actual stop-risk exposure."""
    decision = evaluate_minimum_lot_affordability(
        symbol="NFO:NIFTY2691523500PE",
        quote={
            "bid": 105.20,
            "ask": 105.30,
            "candidate_min_risk_distance": 6.15,
        },
        order_manager=_order_manager(),
        data_hub=SimpleNamespace(get_available_balance=lambda force=False: 15_000.0),
        fallback_balance=15_000.0,
    )

    assert decision.cash_affordable is True
    # Preserve observability: the old model still shows that 6.15 * 65 exceeds
    # the 2% budget. It must no longer veto a contract before the actual strategy
    # stop and bounded target repair are materialized.
    assert decision.risk_floor_affordable is False
    assert decision.one_lot_minimum_risk == pytest.approx(6.15 * 65)
    assert decision.effective_one_lot_risk_budget == pytest.approx(300.0)
    assert decision.affordable is True
    assert decision.reason == "affordable"
    assert decision.capacity_blocker is None


def test_candidate_capacity_keeps_preferred_cash_affordable_contract_without_plan() -> (
    None
):
    runner = object.__new__(StrategyRunner)
    runner._logger = _Logger()
    runner._order_manager = _order_manager()
    runner._data_hub = SimpleNamespace(
        get_available_balance=lambda force=False: 15_000.0
    )
    runner._risk_manager = SimpleNamespace(available_balance=15_000.0)
    runner._is_symbol_execution_ready = lambda _symbol: True
    runner._ensure_symbol_execution_ready_for_order = lambda _symbol, trace_id=None: (
        True
    )

    preferred = SimpleNamespace(symbol="NFO:NIFTY2691523500PE")
    fallback = SimpleNamespace(symbol="NFO:NIFTY2691523450PE")
    snapshots = [
        {
            "symbol": preferred.symbol,
            "bid": 105.20,
            "ask": 105.30,
            "ltp": 105.25,
            "candidate_min_risk_distance": 6.15,
        },
        {
            "symbol": fallback.symbol,
            "bid": 79.90,
            "ask": 80.00,
            "ltp": 79.95,
            "candidate_min_risk_distance": 4.00,
        },
    ]

    selected, decisions = runner._select_capital_eligible_candidate(
        ranked_candidates=[fallback, preferred],
        candidate_snapshots=snapshots,
        is_live_mode=True,
        trace_id="risk-fallback",
        preferred_symbol=preferred.symbol,
    )

    assert selected is preferred
    assert decisions[preferred.symbol]["cash_affordable"] is True
    assert decisions[preferred.symbol]["risk_floor_affordable"] is False
    assert decisions[preferred.symbol]["capacity_blocker"] is None
    assert decisions[preferred.symbol]["affordable"] is True


def test_candidate_capacity_uses_materialized_stop_and_selects_risk_safe_fallback() -> (
    None
):
    runner = object.__new__(StrategyRunner)
    runner._logger = _Logger()
    runner._order_manager = _order_manager(balance=15_000.0)
    runner._data_hub = SimpleNamespace(
        get_available_balance=lambda force=False: 15_000.0
    )
    runner._risk_manager = SimpleNamespace(available_balance=15_000.0)
    runner._is_symbol_execution_ready = lambda _symbol: True
    runner._ensure_symbol_execution_ready_for_order = lambda _symbol, trace_id=None: (
        True
    )

    preferred = SimpleNamespace(
        symbol="NFO:NIFTY2691523500PE",
        entry_price=114.00,
        stop_loss=110.00,
    )
    fallback = SimpleNamespace(
        symbol="NFO:NIFTY2691523450PE",
        entry_price=80.00,
        stop_loss=77.00,
    )
    snapshots = [
        {
            "symbol": preferred.symbol,
            "bid": 113.90,
            "ask": 114.00,
        },
        {
            "symbol": fallback.symbol,
            "bid": 79.90,
            "ask": 80.00,
        },
    ]

    selected, decisions = runner._select_capital_eligible_candidate(
        ranked_candidates=[fallback, preferred],
        candidate_snapshots=snapshots,
        is_live_mode=True,
        trace_id="cost-inclusive-fallback",
        preferred_symbol=preferred.symbol,
        preferred_entry_price=114.00,
        preferred_stop_loss=110.00,
    )

    assert selected is fallback
    assert decisions[preferred.symbol]["cash_affordable"] is True
    assert decisions[preferred.symbol]["plan_risk_affordable"] is False
    assert decisions[preferred.symbol]["capacity_blocker"] == "risk"
    assert decisions[preferred.symbol]["reason"] == "minimum_lot_risk_unaffordable"
    assert decisions[fallback.symbol]["plan_risk_affordable"] is True
    assert decisions[fallback.symbol]["affordable"] is True


@pytest.mark.parametrize(
    ("entry", "expected_affordable"),
    [(200.0, False), (100.0, True)],
)
def test_live_balance_uses_actual_strategy_candidate_stop_geometry(
    monkeypatch, entry: float, expected_affordable: bool
) -> None:
    """Today's ₹773.30 budget must apply to the selector-generated stop."""
    import nifty_scalper_bot.strategies.trade_selector as selector_module

    monkeypatch.setattr(selector_module, "expiry_theta_block", lambda: (False, "ok"))
    monkeypatch.setattr(selector_module, "midday_pause_block", lambda: (False, "ok"))
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.delenv("ENABLE_LIVE", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    monkeypatch.setenv("NIFTY_LOT_SIZE", "65")
    selector = TradeCandidateSelector(
        min_option_premium=40.0,
        max_option_premium=650.0,
        max_tick_age_s=2.5,
        max_option_spread_pct=1.0,
        require_real_ticks_last_60s=2,
    )
    symbol = "NFO:NIFTY26N1725250CE"
    snapshot = {
        "symbol": symbol,
        "side": "CE",
        "strike": 25250,
        "bid": entry - 0.10,
        "ask": entry,
        "ltp": entry - 0.05,
        "tick_age_ms": 100.0,
        "real_ticks_last_60s": 5,
        "atr_option": 2.0,
    }
    ranked = selector.select_ranked_candidates(
        direction_bias="CE",
        atm_strike=25200,
        snapshots=[snapshot],
        gross_rr=2.0,
    )
    assert len(ranked) == 1
    candidate = ranked[0]
    assert candidate.entry_price - candidate.stop_loss + 0.001 >= (
        candidate.entry_price * 0.08
    )

    available = 15_466.0
    runner = object.__new__(StrategyRunner)
    runner._logger = _Logger()
    runner._order_manager = _order_manager(balance=available)
    risk = runner._order_manager._risk_manager
    risk.settings.per_trade_risk_pct = 5.0
    risk._switches.max_day_loss = 773.30
    runner._data_hub = SimpleNamespace(
        get_available_balance=lambda force=False: available
    )
    runner._risk_manager = SimpleNamespace(available_balance=available)
    runner._capital_block_counter = 0
    runner._is_symbol_execution_ready = lambda _symbol: True
    runner._ensure_symbol_execution_ready_for_order = (
        lambda _symbol, trace_id=None: True
    )
    selected, decisions = runner._select_capital_eligible_candidate(
        ranked_candidates=ranked,
        candidate_snapshots=[snapshot],
        is_live_mode=True,
        trace_id="2026-11-11-real-stop",
        preferred_symbol=candidate.symbol,
        preferred_entry_price=candidate.entry_price,
        preferred_stop_loss=candidate.stop_loss,
    )
    details = decisions[candidate.symbol]
    assert details["executable_capacity"] == pytest.approx(13_919.40)
    assert details["effective_one_lot_risk_budget"] == pytest.approx(773.30)
    assert details["plan_risk_affordable"] is expected_affordable
    assert selected is (candidate if expected_affordable else None)


def test_replacement_prefilter_skips_affordable_plus_100_before_plus_50(
    monkeypatch,
) -> None:
    """An illegal liquid fallback must not starve a valid, risk-safe fallback."""
    monkeypatch.setenv("ALLOW_CANDIDATE_REPLACEMENT", "true")
    monkeypatch.setenv("STRICT_CANDIDATE_REPLACEMENT", "true")
    monkeypatch.setenv("MAX_REPLACEMENT_STRIKE_DISTANCE", "50")
    monkeypatch.setenv("MAX_REPLACEMENT_PREMIUM_DEVIATION_PCT", "35")
    runner = object.__new__(StrategyRunner)
    runner._logger = _Logger()
    runner._order_manager = _order_manager(balance=20_000.0)
    risk = runner._order_manager._risk_manager
    risk.settings.per_trade_risk_pct = 3.0
    risk._switches.max_day_loss = 600.0
    runner._data_hub = SimpleNamespace(
        get_available_balance=lambda force=False: 20_000.0
    )
    runner._risk_manager = SimpleNamespace(available_balance=20_000.0)
    runner._capital_block_counter = 0
    runner._is_symbol_execution_ready = lambda _symbol: True
    runner._ensure_symbol_execution_ready_for_order = (
        lambda _symbol, trace_id=None: True
    )
    preferred = SimpleNamespace(
        symbol="NFO:NIFTY2691523500CE", entry_price=80.0, stop_loss=65.0
    )
    outside = SimpleNamespace(
        symbol="NFO:NIFTY2691523600CE", entry_price=60.0, stop_loss=55.2
    )
    permitted = SimpleNamespace(
        symbol="NFO:NIFTY2691523550CE", entry_price=65.0, stop_loss=59.8
    )
    runner._active_option_symbols = {preferred.symbol, outside.symbol, permitted.symbol}
    selected, decisions = runner._select_capital_eligible_candidate(
        ranked_candidates=[outside, permitted, preferred],
        candidate_snapshots=[
            {"symbol": preferred.symbol, "bid": 79.9, "ask": 80.0},
            {"symbol": outside.symbol, "bid": 59.9, "ask": 60.0},
            {"symbol": permitted.symbol, "bid": 64.9, "ask": 65.0},
        ],
        is_live_mode=True,
        trace_id="allowed-plus-50-not-plus-100",
        preferred_symbol=preferred.symbol,
        preferred_entry_price=80.0,
        preferred_stop_loss=65.0,
    )
    assert selected is permitted
    assert decisions[outside.symbol]["reason"] == "replacement_not_permitted"
    assert decisions[outside.symbol]["replacement_blocker"] == "strike_jump"
    assert decisions[preferred.symbol]["capacity_blocker"] == "risk"
    assert decisions[permitted.symbol]["plan_risk_affordable"] is True


@pytest.mark.parametrize(
    ("candidate", "price", "expected_reason"),
    [
        ("NFO:NIFTY2691523500PE", 65.0, "side_mismatch"),
        ("NFO:NIFTY2692223550CE", 65.0, "expiry_mismatch"),
        ("NFO:NIFTY2691523600CE", 65.0, "strike_jump"),
        ("NFO:NIFTY2691523550CE", 40.0, "premium_jump"),
        ("NFO:NIFTY2691523450CE", 65.0, "not_in_active_basket"),
    ],
)
def test_replacement_contract_policy_is_fail_closed(
    monkeypatch, candidate: str, price: float, expected_reason: str
) -> None:
    monkeypatch.setenv("STRICT_CANDIDATE_REPLACEMENT", "true")
    monkeypatch.setenv("ALLOW_CANDIDATE_REPLACEMENT", "true")
    monkeypatch.setenv("MAX_REPLACEMENT_STRIKE_DISTANCE", "50")
    monkeypatch.setenv("MAX_REPLACEMENT_PREMIUM_DEVIATION_PCT", "35")
    runner = object.__new__(StrategyRunner)
    runner._active_option_symbols = {
        "NFO:NIFTY2691523500CE",
        "NFO:NIFTY2691523550CE",
        "NFO:NIFTY2691523600CE",
        "NFO:NIFTY2692223550CE",
        "NFO:NIFTY2691523500PE",
    }
    blocked, _policy = runner._candidate_replacement_contract_guard(
        original_symbol="NFO:NIFTY2691523500CE",
        selected_symbol=candidate,
        original_trade_price=80.0,
        candidate_trade_price=price,
    )
    assert blocked == expected_reason


def test_candidate_capacity_log_distinguishes_cash_from_risk_block() -> None:
    runner = object.__new__(StrategyRunner)
    runner._logger = _Logger()
    runner._order_manager = _order_manager(balance=15_000.0)
    runner._data_hub = SimpleNamespace(
        get_available_balance=lambda force=False: 15_000.0
    )
    runner._risk_manager = SimpleNamespace(available_balance=15_000.0)
    runner._capital_block_counter = 0
    runner._is_symbol_execution_ready = lambda _symbol: True
    runner._ensure_symbol_execution_ready_for_order = (
        lambda _symbol, trace_id=None: True
    )
    candidate = SimpleNamespace(
        symbol="NFO:NIFTY2691523500PE",
        entry_price=114.00,
        stop_loss=110.00,
    )

    selected, _ = runner._select_capital_eligible_candidate(
        ranked_candidates=[candidate],
        candidate_snapshots=[
            {"symbol": candidate.symbol, "bid": 113.90, "ask": 114.00}
        ],
        is_live_mode=True,
        trace_id="risk-log",
        preferred_symbol=candidate.symbol,
        preferred_entry_price=114.00,
        preferred_stop_loss=110.00,
    )

    assert selected is None
    capacity_call = next(
        args
        for args, _kwargs in runner._logger.info_calls
        if args and str(args[0]).startswith("CANDIDATE_CAPACITY_DECISION")
    )
    rendered = str(capacity_call[0]) % tuple(capacity_call[1:])
    assert "capacity_blocker=risk" in rendered
    assert "cash_required=" in rendered
    assert "plan_cost_inclusive_risk=" in rendered
    assert "effective_risk_budget=" in rendered
    assert "remaining_daily_risk_budget=" in rendered


def test_plan_affordability_matches_final_cost_inclusive_risk_authority() -> None:
    quote = {"bid": 113.90, "ask": 114.00}
    decision = evaluate_minimum_lot_affordability(
        symbol="NFO:NIFTY2691523500PE",
        quote=quote,
        order_manager=_order_manager(balance=15_000.0),
        data_hub=SimpleNamespace(get_available_balance=lambda force=False: 15_000.0),
        plan_entry_price=114.00,
        plan_stop_loss=110.00,
    )
    final = evaluate_final_net_rr(
        OrderSignal(
            symbol="NFO:NIFTY2691523500PE",
            side="BUY",
            quantity=65,
            price=114.05,
            stop_loss=110.05,
            take_profit=122.05,
            metadata=quote,
        )
    )

    assert final is not None
    assert decision.plan_cost_inclusive_risk == pytest.approx(final.net_risk)
    assert decision.plan_risk_affordable is False


def test_exhausted_risk_capacity_increments_engine_capital_block_counter() -> None:
    runner = object.__new__(StrategyRunner)
    runner._logger = _Logger()
    runner._order_manager = _order_manager(balance=15_000.0)
    runner._data_hub = SimpleNamespace(
        get_available_balance=lambda force=False: 15_000.0
    )
    runner._risk_manager = SimpleNamespace(available_balance=15_000.0)
    runner._capital_block_counter = 0
    runner._is_symbol_execution_ready = lambda _symbol: True
    runner._ensure_symbol_execution_ready_for_order = lambda _symbol, trace_id=None: (
        True
    )
    candidate = SimpleNamespace(
        symbol="NFO:NIFTY2691523500PE",
        entry_price=114.00,
        stop_loss=110.00,
    )

    selected, decisions = runner._select_capital_eligible_candidate(
        ranked_candidates=[candidate],
        candidate_snapshots=[
            {"symbol": candidate.symbol, "bid": 113.90, "ask": 114.00}
        ],
        is_live_mode=True,
        trace_id="risk-exhausted",
        preferred_symbol=candidate.symbol,
        preferred_entry_price=114.00,
        preferred_stop_loss=110.00,
    )

    assert selected is None
    assert decisions[candidate.symbol]["capacity_blocker"] == "risk"
    assert runner._capital_block_counter == 1


def test_trade_selector_exposes_cost_floor_as_diagnostic_metadata(
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
    assert snapshot["candidate_min_risk_distance"] > 0.0
    assert snapshot["candidate_gross_rr"] == pytest.approx(2.0)


def test_runner_reuses_current_net_day_loss_across_ranked_candidates() -> None:
    """Live candidate selection refreshes P&L once, not once per fallback."""
    runner = object.__new__(StrategyRunner)
    runner._logger = _Logger()
    runner._order_manager = _order_manager(balance=11_000.65)
    risk = runner._order_manager._risk_manager
    risk._switches.max_day_loss = 558.645
    calls: list[int] = []

    def refresh_risk():
        calls.append(1)
        return SimpleNamespace(max_day_loss=558.645, day_loss=241.527189)

    risk.snapshot = refresh_risk
    runner._data_hub = SimpleNamespace(
        get_available_balance=lambda force=False: 11_000.65
    )
    runner._risk_manager = SimpleNamespace(available_balance=11_000.65)
    runner._is_symbol_execution_ready = lambda _symbol: True
    runner._ensure_symbol_execution_ready_for_order = lambda _symbol, trace_id=None: (
        True
    )
    primary = SimpleNamespace(
        symbol="NFO:NIFTY2691523500CE", entry_price=134.55, stop_loss=130.10
    )
    fallback = SimpleNamespace(
        symbol="NFO:NIFTY2691523450CE", entry_price=134.25, stop_loss=129.80
    )
    selected, decisions = runner._select_capital_eligible_candidate(
        ranked_candidates=[fallback, primary],
        candidate_snapshots=[
            {"symbol": primary.symbol, "bid": 134.45, "ask": 134.55},
            {"symbol": fallback.symbol, "bid": 134.15, "ask": 134.25},
        ],
        is_live_mode=True,
        trace_id="stale-day-budget-20261009",
        preferred_symbol=primary.symbol,
        preferred_entry_price=134.55,
        preferred_stop_loss=130.10,
    )

    assert calls == [1]
    assert selected is None
    assert all(item["capacity_blocker"] == "risk" for item in decisions.values())
    assert all(
        item["remaining_daily_risk_budget"] == pytest.approx(317.117811)
        for item in decisions.values()
    )
