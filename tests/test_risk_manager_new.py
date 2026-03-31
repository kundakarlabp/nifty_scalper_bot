from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.config.settings import RiskSettings
from nifty_scalper_bot.risk import OrderSignal, RiskManager, RiskState


@dataclass
class DummyPositionManager:
    realized: float

    def get_all_positions(self):  # noqa: D401 - test stub
        return []

    def get_realized_pnl(self) -> float:
        return self.realized

    def get_total_exposure(self) -> float:
        return 0.0


def _make_risk_manager(
    *, settings: RiskSettings, pm: DummyPositionManager, balance: float
) -> RiskManager:
    risk = RiskManager(settings=settings, position_manager=pm, account_balance=balance)

    def _resolver(_symbol: str) -> int:
        return 75

    risk.set_lot_size_provider(_resolver, symbol="NIFTY")
    return risk


def test_risk_manager_blocks_daily_pnl_cap() -> None:
    settings = RiskSettings(daily_pnl_cap_pct=1.0, per_trade_risk_pct=0.5)
    pm = DummyPositionManager(realized=-200.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=10_000.0)
    signal = OrderSignal(
        symbol="NIFTY",
        side="BUY",
        quantity=10,
        price=100.0,
        stop_loss=99.5,
        take_profit=101.0,
    )
    allowed, reason = risk.check_order(signal, live_enabled=True)
    assert not allowed
    assert "Daily" in reason


def test_risk_manager_enforces_per_trade_risk() -> None:
    settings = RiskSettings(daily_pnl_cap_pct=50.0, per_trade_risk_pct=0.5)
    pm = DummyPositionManager(realized=0.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=10_000.0)
    signal = OrderSignal(
        symbol="NIFTY",
        side="BUY",
        quantity=200,
        price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
    )
    allowed, reason = risk.check_order(signal, live_enabled=True)
    assert not allowed
    assert "Per-trade" in reason


def test_risk_manager_cooldown_after_rejection() -> None:
    settings = RiskSettings(
        daily_pnl_cap_pct=50.0, per_trade_risk_pct=0.5, cooldown_on_reject_seconds=5.0
    )
    pm = DummyPositionManager(realized=0.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=10_000.0)
    risk.record_rejection("broker error")
    assert risk.cooldown_remaining() > 0.0


def test_risk_manager_exposes_available_balance_on_init() -> None:
    settings = RiskSettings(per_trade_risk_pct=1.0)
    pm = DummyPositionManager(realized=0.0)

    risk = _make_risk_manager(settings=settings, pm=pm, balance=12_345.0)

    assert risk.available_balance == 12_345.0


def test_suggest_position_size_returns_lot_multiple() -> None:
    settings = RiskSettings(per_trade_risk_pct=1.0)
    pm = DummyPositionManager(realized=0.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=100_000.0)

    quantity = risk.suggest_position_size(
        side="BUY",
        price=200.0,
        stop_loss=195.0,
        atr=None,
        requested_quantity=150,
        confidence=1.0,
    )

    assert quantity == 150
    assert quantity % 75 == 0


def test_suggest_position_size_scales_with_confidence() -> None:
    settings = RiskSettings(per_trade_risk_pct=2.0)
    pm = DummyPositionManager(realized=0.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=200_000.0)

    quantity = risk.suggest_position_size(
        side="BUY",
        price=200.0,
        stop_loss=195.0,
        atr=None,
        requested_quantity=750,
        confidence=0.4,
    )

    assert quantity == 300


def test_suggest_position_size_skips_when_one_lot_risk_too_high() -> None:
    settings = RiskSettings(per_trade_risk_pct=0.5)
    pm = DummyPositionManager(realized=0.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=1_000.0)

    quantity = risk.suggest_position_size(
        side="BUY",
        price=100.0,
        stop_loss=80.0,
        atr=None,
        requested_quantity=75,
        confidence=1.0,
    )

    assert quantity == 0


def _make_valid_signal() -> OrderSignal:
    return OrderSignal(
        symbol="NIFTY",
        side="BUY",
        quantity=25,
        price=100.0,
        stop_loss=99.0,
        take_profit=102.0,
    )


def test_cooldown_zero_is_not_blocked() -> None:
    settings = RiskSettings(daily_pnl_cap_pct=50.0, per_trade_risk_pct=1.0)
    pm = DummyPositionManager(realized=0.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=50_000.0)
    risk._cooldown_until = datetime.now(timezone.utc) + timedelta(milliseconds=50)

    allowed, reason = risk.check_order(_make_valid_signal(), live_enabled=True)

    assert allowed is True
    assert reason == ""
    assert risk.cooldown_remaining() == 0.0


def test_risk_returns_code_cooldown() -> None:
    settings = RiskSettings(daily_pnl_cap_pct=50.0, per_trade_risk_pct=1.0)
    pm = DummyPositionManager(realized=0.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=50_000.0)
    risk._cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=1.1)

    allowed, reason = risk.check_order(_make_valid_signal(), live_enabled=True)

    assert allowed is False
    assert reason == "COOLDOWN"


def test_stale_is_soft_no_cooldown() -> None:
    settings = RiskSettings(
        daily_pnl_cap_pct=50.0,
        per_trade_risk_pct=1.0,
        cooldown_on_reject_seconds=5.0,
    )
    pm = DummyPositionManager(realized=0.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=50_000.0)

    risk.record_rejection("STALE")

    assert risk.cooldown_remaining() == 0.0
    assert risk._cooldown_until is None
    assert risk._switches.cooldown_remaining() == 0.0
    assert risk._last_rejection == "STALE"


def test_recent_reject_is_soft_no_cooldown() -> None:
    settings = RiskSettings(
        daily_pnl_cap_pct=50.0,
        per_trade_risk_pct=1.0,
        cooldown_on_reject_seconds=5.0,
    )
    pm = DummyPositionManager(realized=0.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=50_000.0)

    risk.record_rejection("RECENT_REJECT")

    assert risk.cooldown_remaining() == 0.0
    assert risk._cooldown_until is None
    assert risk._switches.cooldown_remaining() == 0.0
    assert risk._last_rejection == "RECENT_REJECT"


def test_override_does_not_bypass_breakers() -> None:
    settings = RiskSettings(daily_pnl_cap_pct=1.0, per_trade_risk_pct=0.5)
    pm = DummyPositionManager(realized=0.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=10_000.0)
    risk.reset_on_start(True)
    risk._switches.record_pnl(-500.0)

    allowed, reason = risk.check_order(_make_valid_signal(), live_enabled=True)

    assert allowed is False
    assert "Daily" in reason or "loss" in reason

    risk.reset_on_start(False)
    allowed_after, reason_after = risk.check_order(
        _make_valid_signal(), live_enabled=True
    )

    assert allowed_after is False
    assert "Daily" in reason_after or "loss" in reason_after


def test_override_bypasses_cooldown() -> None:
    settings = RiskSettings(
        per_trade_risk_pct=1.0,
        cooldown_on_reject_seconds=5.0,
    )
    pm = DummyPositionManager(realized=0.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=10_000.0)
    risk.reset_on_start(True)
    risk._cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=5)

    allowed, reason = risk.check_order(_make_valid_signal(), live_enabled=True)

    assert allowed is True
    assert reason == ""


def test_reset_on_start_clears_soft_blocks() -> None:
    settings = RiskSettings(per_trade_risk_pct=1.0)
    pm = DummyPositionManager(realized=0.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=10_000.0)
    risk._cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=10)
    risk._last_rejection = "test"
    risk._switches.engage_cooldown(1.0)

    risk.reset_on_start(True)

    assert risk.cooldown_remaining() == 0.0
    assert risk._last_rejection is None
    assert risk._soft_override is True


def test_record_rejection_canonicalizes_reason() -> None:
    settings = RiskSettings(per_trade_risk_pct=1.0, cooldown_on_reject_seconds=5.0)
    pm = DummyPositionManager(realized=0.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=10_000.0)

    risk.record_rejection("Trading disabled by risk state")

    assert risk._last_rejection == "RISK_STATE"


def test_risk_state_stale_soft_block() -> None:
    settings = RiskSettings(per_trade_risk_pct=1.0)
    pm = DummyPositionManager(realized=0.0)
    risk = _make_risk_manager(settings=settings, pm=pm, balance=50_000.0)
    state = RiskState(
        quote_stale_ms_threshold=200,
        spread_widen_mult=3.0,
        max_intraday_dd=-5_000.0,
        max_consecutive_losses=3,
    )
    risk.bind_risk_state(state)

    cooldown_before = risk.cooldown_remaining()
    assert cooldown_before == 0.0

    allowed, reasons = risk.risk_gate_should_trade()

    assert allowed is False
    assert reasons == ("STALE",)
    assert risk.cooldown_remaining() == 0.0
