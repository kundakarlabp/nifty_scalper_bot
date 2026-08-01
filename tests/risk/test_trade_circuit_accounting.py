"""Daily-loss and consecutive-loss circuit correctness (P1)."""

from __future__ import annotations

from nifty_scalper_bot.risk.limits import RiskSwitches


def _switches(**overrides) -> RiskSwitches:
    payload = dict(
        max_day_loss=1000.0,
        max_consecutive_losses=3,
        cooldown_minutes=0.0,
        reset_hour_utc=0,
    )
    payload.update(overrides)
    return RiskSwitches(**payload)


def test_partial_exit_slices_do_not_inflate_loss_streak() -> None:
    switches = _switches()
    # One losing trade exiting in three partial slices.
    for _ in range(3):
        switches.record_pnl(-100.0)

    assert switches.consecutive_losses() == 0
    assert switches.breach_reason() is None
    assert switches.day_loss() == 300.0


def test_completed_trades_advance_the_loss_streak() -> None:
    switches = _switches()
    switches.record_trade_result(-100.0)
    switches.record_trade_result(-100.0)
    assert switches.consecutive_losses() == 2
    assert switches.breach_reason() is None

    switches.record_trade_result(-100.0)
    assert switches.breach_reason() == "Max consecutive losses reached"


def test_a_winning_trade_resets_the_streak() -> None:
    switches = _switches()
    switches.record_trade_result(-100.0)
    switches.record_trade_result(-100.0)
    switches.record_trade_result(50.0)
    assert switches.consecutive_losses() == 0


def test_loss_cooldown_engages_once_per_trade() -> None:
    switches = _switches(cooldown_minutes=5.0)
    switches.record_pnl(-100.0)
    assert switches.cooldown_remaining() == 0.0

    switches.record_trade_result(-100.0)
    assert switches.cooldown_remaining() > 0.0


def test_charges_are_applied_to_day_loss() -> None:
    from types import MethodType, SimpleNamespace

    from nifty_scalper_bot.risk.risk_manager import RiskManager

    switches = _switches()
    stub = SimpleNamespace(
        _switches=switches,
        settings=SimpleNamespace(max_consecutive_losses=3),
        _format_switch_reason=lambda reason: reason,
        _trip_breaker=lambda reason: None,
    )
    stub.record_completed_trade = MethodType(
        RiskManager.record_completed_trade, stub
    )

    stub.record_completed_trade(-100.0, 60.0)

    assert switches.day_loss() == 60.0
    assert switches.consecutive_losses() == 1
