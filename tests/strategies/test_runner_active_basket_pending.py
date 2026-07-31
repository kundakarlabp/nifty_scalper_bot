from __future__ import annotations

import logging
from nifty_scalper_bot.strategies.runner import StrategyRunner


class _Indicator:
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts

    def get_history(self, symbol: str):
        return list(range(self.counts.get(symbol, 0)))


def _runner(counts: dict[str, int]) -> StrategyRunner:
    runner = object.__new__(StrategyRunner)
    runner._indicator_engine = _Indicator(counts)
    runner._logger = logging.getLogger("test.runner.pending")
    runner._option_required_bars = 30
    runner._active_selected_ce = None
    runner._active_selected_pe = None
    runner._active_atm_strike = None
    runner._active_option_symbols = set()
    runner._pending_selected_ce = None
    runner._pending_selected_pe = None
    runner._pending_atm_strike = None
    runner._prewarm_active_option_history = lambda **_kwargs: None
    return runner


def test_first_selected_cold_basket_is_pending_not_execution_active() -> None:
    runner = _runner({"NFO:NIFTY26MAY23300CE": 1, "NFO:NIFTY26MAY23300PE": 1})

    runner.set_active_option_context(
        selected_ce="NFO:NIFTY26MAY23300CE",
        selected_pe="NFO:NIFTY26MAY23300PE",
        atm_strike=23300,
        option_symbols=["NFO:NIFTY26MAY23300CE", "NFO:NIFTY26MAY23300PE"],
    )

    assert runner._active_selected_ce is None
    assert runner._active_selected_pe is None
    assert runner._pending_selected_ce == "NFO:NIFTY26MAY23300CE"
    assert runner._pending_selected_pe == "NFO:NIFTY26MAY23300PE"
    assert "NFO:NIFTY26MAY23300CE" in runner._active_option_symbols
    assert "NFO:NIFTY26MAY23300PE" in runner._active_option_symbols


def test_pending_basket_promotes_only_after_both_legs_have_required_bars() -> None:
    counts = {"NFO:NIFTY26MAY23300CE": 30, "NFO:NIFTY26MAY23300PE": 29}
    runner = _runner(counts)
    runner._pending_selected_ce = "NFO:NIFTY26MAY23300CE"
    runner._pending_selected_pe = "NFO:NIFTY26MAY23300PE"
    runner._pending_atm_strike = 23300

    assert runner._maybe_promote_pending_active_basket(source="test") is False
    assert runner._active_selected_ce is None

    counts["NFO:NIFTY26MAY23300PE"] = 30

    assert runner._maybe_promote_pending_active_basket(source="test") is True
    assert runner._active_selected_ce == "NFO:NIFTY26MAY23300CE"
    assert runner._active_selected_pe == "NFO:NIFTY26MAY23300PE"
    assert runner._pending_selected_ce is None
    assert runner._pending_selected_pe is None


def test_unchanged_warm_active_context_skips_history_prewarm() -> None:
    ce = "NFO:NIFTY26MAY23300CE"
    pe = "NFO:NIFTY26MAY23300PE"
    runner = _runner({ce: 30, pe: 30})
    runner._active_selected_ce = ce
    runner._active_selected_pe = pe
    runner._active_atm_strike = 23300
    runner._active_option_symbols = {ce, pe}
    runner._symbol_history = {ce: list(range(30)), pe: list(range(30))}
    prewarm_calls = []
    runner._prewarm_active_option_history = lambda **kwargs: prewarm_calls.append(
        kwargs
    )

    runner.set_active_option_context(
        selected_ce=ce,
        selected_pe=pe,
        atm_strike=23300,
        option_symbols=[ce, pe],
    )

    assert prewarm_calls == []


def test_option_prewarm_syncs_only_cold_symbols() -> None:
    ce = "NFO:NIFTY26MAY23300CE"
    pe = "NFO:NIFTY26MAY23300PE"
    cold = "NFO:NIFTY26MAY23350CE"
    runner = _runner({ce: 30, pe: 30, cold: 1})
    runner._active_selected_ce = ce
    runner._active_selected_pe = pe
    runner._active_option_symbols = {ce, pe, cold}
    runner._symbol_history = {
        ce: list(range(30)),
        pe: list(range(30)),
        cold: [1],
    }
    runner._required_bars_for_symbol = lambda _symbol: 30
    synced = []
    requested = []
    runner._sync_history_from_mdm_cache = lambda symbol, **_kwargs: (
        synced.append(symbol) or 1
    )
    runner._request_selected_option_history_prewarm = (
        lambda symbol, **_kwargs: requested.append(symbol)
    )

    StrategyRunner._prewarm_active_option_history(runner, source="test")

    assert synced == [cold]
    assert requested == [cold]
