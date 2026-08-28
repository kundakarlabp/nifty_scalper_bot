from __future__ import annotations

from nifty_scalper_bot.config.settings import RiskSettings
from nifty_scalper_bot.risk.cost_model import estimate_round_trip_cost
from nifty_scalper_bot.risk.net_rr_gate import evaluate_final_net_rr
from nifty_scalper_bot.risk.risk_manager import OrderSignal, RiskManager


class _Positions:
    def get_all_positions(self):
        return []

    def get_realized_pnl(self) -> float:
        return 0.0

    def get_total_exposure(self) -> float:
        return 0.0


def test_final_live_rr_does_not_charge_entry_half_spread_twice(monkeypatch) -> None:
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "0")
    signal = OrderSignal(
        symbol="NFO:NIFTY2690124150PE",
        side="BUY",
        quantity=65,
        price=100.50,
        stop_loss=92.50,
        take_profit=116.50,
        metadata={"bid": 99.50, "ask": 100.50},
    )

    result = evaluate_final_net_rr(signal)

    assert result is not None
    full_crossing_cost = estimate_round_trip_cost(
        entry_price=100.50,
        exit_price=116.50,
        quantity=65,
        half_spread=0.50,
    ).total
    # The order price is already the executable ask, so only the future SELL
    # crossing remains to be modelled. The generic cost helper still models
    # both crossings by default; the final live gate must remove one half-spread.
    assert result.target_cost == full_crossing_cost - (0.50 * 65)
    assert result.half_spread == 0.50


def test_reference_price_rr_keeps_full_round_trip_spread_cost(monkeypatch) -> None:
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "0")
    signal = OrderSignal(
        symbol="NFO:NIFTY2690124150PE",
        side="BUY",
        quantity=65,
        price=100.00,
        stop_loss=92.00,
        take_profit=116.00,
        metadata={"bid": 99.50, "ask": 100.50},
    )

    result = evaluate_final_net_rr(signal)

    assert result is not None
    full_crossing_cost = estimate_round_trip_cost(
        entry_price=100.00,
        exit_price=116.00,
        quantity=65,
        half_spread=0.50,
    ).total
    assert result.target_cost == full_crossing_cost


def test_daily_risk_rejection_reports_full_cap_when_no_loss_today(monkeypatch) -> None:
    monkeypatch.setenv("PAPER_MODE", "true")
    settings = RiskSettings(
        daily_loss_pct=2.0,
        per_trade_risk_pct=10.0,
        cooldown_on_reject_seconds=0.0,
    )
    risk = RiskManager(
        settings=settings,
        position_manager=_Positions(),
        account_balance=15_589.60,
    )
    signal = OrderSignal(
        symbol="NFO:NIFTY2690124150PE",
        side="BUY",
        quantity=65,
        price=64.25,
        stop_loss=58.80,
        take_profit=75.15,
    )

    allowed, reason = risk.check_order(signal, live_enabled=True)

    assert allowed is False
    assert reason == (
        "daily stop-risk cap exceeded: required=354.25 available=311.79 "
        "day_loss=0.00 cap=311.79"
    )
    assert risk._last_rejection == "DAILY_RISK_BUDGET"
    assert risk._switches.day_loss() == 0.0
    assert risk._switches.max_day_loss == 311.792
