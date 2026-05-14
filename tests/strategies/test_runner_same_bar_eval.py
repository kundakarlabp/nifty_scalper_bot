from __future__ import annotations

from nifty_scalper_bot.strategies.runner import StrategyRunner


def _build_runner() -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._data_phase = {"NFO:NIFTY26MAY25000CE": "LIVE", "NFO:NIFTY26MAY25200CE": "LIVE"}
    runner._last_same_bar_eval_ts_by_symbol = {}
    runner._last_eval_price_by_symbol = {}
    runner._active_selected_ce = "NFO:NIFTY26MAY25000CE"
    runner._active_selected_pe = "NFO:NIFTY26MAY25000PE"
    runner._required_bars_for_symbol = lambda _s: 50
    return runner


def test_same_bar_periodic_eval_selected(monkeypatch):
    runner = _build_runner()
    symbol = "NFO:NIFTY26MAY25000CE"
    monkeypatch.setenv("RUNNER_ENABLE_INTRABAR_STRATEGY_EVAL", "true")
    monkeypatch.setenv("RUNNER_INTRABAR_EVAL_SELECTED_SECONDS", "10")
    runner._last_same_bar_eval_ts_by_symbol[symbol] = 100.0
    reason = runner._same_bar_eval_reason(
        symbol=symbol,
        price=100.0,
        tick={},
        candle_count=60,
        now_ts=111.0,
    )
    assert reason == "same_bar_periodic_eval"


def test_same_bar_price_move_eval_selected(monkeypatch):
    runner = _build_runner()
    symbol = "NFO:NIFTY26MAY25000CE"
    monkeypatch.setenv("RUNNER_ENABLE_INTRABAR_STRATEGY_EVAL", "true")
    monkeypatch.setenv("RUNNER_INTRABAR_EVAL_SELECTED_SECONDS", "10")
    monkeypatch.setenv("RUNNER_INTRABAR_EVAL_MIN_PRICE_MOVE_PCT", "0.12")
    runner._last_same_bar_eval_ts_by_symbol[symbol] = 100.0
    runner._last_eval_price_by_symbol[symbol] = 100.0
    reason = runner._same_bar_eval_reason(
        symbol=symbol,
        price=100.2,
        tick={},
        candle_count=60,
        now_ts=104.0,
    )
    assert reason == "same_bar_price_move_eval"


def test_same_bar_non_selected_small_move_none(monkeypatch):
    runner = _build_runner()
    symbol = "NFO:NIFTY26MAY25200CE"
    monkeypatch.setenv("RUNNER_ENABLE_INTRABAR_STRATEGY_EVAL", "true")
    monkeypatch.setenv("RUNNER_INTRABAR_EVAL_NON_SELECTED_SECONDS", "60")
    runner._last_same_bar_eval_ts_by_symbol[symbol] = 100.0
    runner._last_eval_price_by_symbol[symbol] = 100.0
    reason = runner._same_bar_eval_reason(
        symbol=symbol,
        price=100.05,
        tick={},
        candle_count=60,
        now_ts=110.0,
    )
    assert reason is None
