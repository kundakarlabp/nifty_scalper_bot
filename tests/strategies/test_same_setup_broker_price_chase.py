"""Regression checks for unchanged exact-setup broker-attempt price anchors."""

from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_same_setup_price_chase() -> None:
    runner = object.__new__(StrategyRunner)
    runner._broker_attempt_price_by_setup = {}
    runner._remember_broker_attempt_price(signal_id='setup', symbol='NFO:NIFTYCE', price=100.0, now_epoch=100.0, broker_attempted=True)
    assert runner._same_setup_price_chase(signal_id='setup', symbol='NFO:NIFTYCE', price=106.0, max_drift_pct=2.0, now_epoch=150.0) == 6.0
