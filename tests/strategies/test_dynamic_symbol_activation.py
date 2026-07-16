from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.strategies.runner import StrategyRunner


class MDM:
    def __init__(
        self,
        *,
        confirmed=False,
        current=False,
        reason="current_generation_tick_pending",
    ):
        self.confirmed = confirmed
        self.current = current
        self.reason = reason

    def current_live_token(self, _symbol):
        return 1

    def classify_live_tick_readiness(self, symbol, token, *, max_age_s):
        return {
            "expected": True,
            "confirmed": self.confirmed,
            "current_generation_tick_received": self.current,
            "first_tick_received": self.current,
            "reason": "ready" if self.current and self.confirmed else self.reason,
        }


def runner(mdm):
    r = StrategyRunner.__new__(StrategyRunner)
    r._market_data = mdm
    r._active_symbols = {"NFO:NIFTY26JUN24000CE"}
    r._tracked_symbols = {"NFO:NIFTY26JUN24000CE"}
    r._strategy_manager = object()
    r._symbol_history = {"NFO:NIFTY26JUN24000CE": [object()] * 10}
    r._context_required_bars = 2
    r._option_required_bars = 2
    r._history_count_for_symbol = lambda _s: 10
    return r


def test_requested_but_unconfirmed_remains_pending():
    activation = runner(
        MDM(confirmed=False, current=False, reason="subscription_not_confirmed")
    )._live_symbol_activation("NFO:NIFTY26JUN24000CE")
    assert activation.executable is False
    assert "subscription_not_confirmed" in activation.blockers


def test_confirmed_without_current_generation_tick_remains_pending():
    activation = runner(MDM(confirmed=True, current=False))._live_symbol_activation(
        "NFO:NIFTY26JUN24000CE"
    )
    assert activation.executable is False
    assert "current_generation_tick_pending" in activation.blockers


def test_current_generation_tick_promotes_symbol():
    activation = runner(MDM(confirmed=True, current=True))._live_symbol_activation(
        "NFO:NIFTY26JUN24000CE"
    )
    assert activation.executable is True
    assert activation.blockers == ()
