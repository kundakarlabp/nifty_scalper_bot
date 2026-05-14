from __future__ import annotations

from nifty_scalper_bot.strategies.runner import StrategyRunner


def _build_runner() -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._data_phase = {
        'NFO:NIFTY26MAY25000CE': 'LIVE',
        'NFO:NIFTY26MAY25100CE': 'LIVE',
        'NFO:NIFTY26MAY25300CE': 'LIVE',
        'NFO:NIFTY26MAY25200CE': 'LIVE',
        'NFO:NIFTY26MAY25000PE': 'HYDRATION',
    }
    runner._last_same_bar_eval_ts_by_symbol = {}
    runner._last_eval_price_by_symbol = {}
    runner._last_same_bar_eval_block_reason_by_symbol = {}
    runner._last_same_bar_eval_block_detail_by_symbol = {}
    runner._active_selected_ce = 'NFO:NIFTY26MAY25000CE'
    runner._active_selected_pe = 'NFO:NIFTY26MAY25000PE'
    runner._required_bars_for_symbol = lambda _s: 50
    runner._extract_strike_from_symbol = StrategyRunner._extract_strike_from_symbol.__get__(runner, StrategyRunner)
    return runner


def test_same_bar_periodic_eval_selected(monkeypatch):
    runner = _build_runner()
    symbol = 'NFO:NIFTY26MAY25000CE'
    monkeypatch.setenv('RUNNER_ENABLE_INTRABAR_STRATEGY_EVAL', 'true')
    monkeypatch.setenv('RUNNER_INTRABAR_EVAL_SELECTED_SECONDS', '10')
    runner._last_same_bar_eval_ts_by_symbol[symbol] = 100.0
    reason = runner._same_bar_eval_reason(symbol=symbol, price=100.0, tick={}, candle_count=60, now_ts=111.0)
    assert reason == 'same_bar_periodic_eval'


def test_same_bar_periodic_eval_near_selected(monkeypatch):
    runner = _build_runner()
    symbol = 'NFO:NIFTY26MAY25100CE'
    monkeypatch.setenv('RUNNER_ENABLE_INTRABAR_STRATEGY_EVAL', 'true')
    monkeypatch.setenv('RUNNER_INTRABAR_EVAL_SELECTED_SECONDS', '10')
    monkeypatch.setenv('RUNNER_INTRABAR_NEAR_SELECTED_POINTS', '100')
    runner._last_same_bar_eval_ts_by_symbol[symbol] = 100.0
    reason = runner._same_bar_eval_reason(symbol=symbol, price=120.0, tick={}, candle_count=60, now_ts=111.0)
    assert reason == 'same_bar_periodic_eval'


def test_same_bar_price_move_eval_near_selected(monkeypatch):
    runner = _build_runner()
    symbol = 'NFO:NIFTY26MAY25100CE'
    monkeypatch.setenv('RUNNER_ENABLE_INTRABAR_STRATEGY_EVAL', 'true')
    monkeypatch.setenv('RUNNER_INTRABAR_EVAL_SELECTED_SECONDS', '10')
    monkeypatch.setenv('RUNNER_INTRABAR_NEAR_SELECTED_POINTS', '100')
    monkeypatch.setenv('RUNNER_INTRABAR_EVAL_MIN_PRICE_MOVE_PCT', '0.12')
    runner._last_same_bar_eval_ts_by_symbol[symbol] = 100.0
    runner._last_eval_price_by_symbol[symbol] = 100.0
    reason = runner._same_bar_eval_reason(symbol=symbol, price=100.2, tick={}, candle_count=60, now_ts=104.0)
    assert reason == 'same_bar_price_move_eval'


def test_same_bar_far_non_selected_block_reason(monkeypatch):
    runner = _build_runner()
    symbol = 'NFO:NIFTY26MAY25300CE'
    monkeypatch.setenv('RUNNER_ENABLE_INTRABAR_STRATEGY_EVAL', 'true')
    monkeypatch.setenv('RUNNER_INTRABAR_EVAL_NON_SELECTED_SECONDS', '60')
    monkeypatch.setenv('RUNNER_INTRABAR_NEAR_SELECTED_POINTS', '100')
    runner._last_same_bar_eval_ts_by_symbol[symbol] = 100.0
    reason = runner._same_bar_eval_reason(symbol=symbol, price=100.05, tick={}, candle_count=60, now_ts=110.0)
    assert reason is None
    assert runner._last_same_bar_eval_block_reason_by_symbol[symbol] in {
        'intrabar_eval_non_selected_waiting',
        'intrabar_eval_interval_not_elapsed',
    }


def test_same_bar_hydration_phase_block_reason(monkeypatch):
    runner = _build_runner()
    symbol = 'NFO:NIFTY26MAY25000PE'
    monkeypatch.setenv('RUNNER_ENABLE_INTRABAR_STRATEGY_EVAL', 'true')
    reason = runner._same_bar_eval_reason(symbol=symbol, price=100.0, tick={}, candle_count=60, now_ts=110.0)
    assert reason is None
    assert runner._last_same_bar_eval_block_reason_by_symbol[symbol] == 'intrabar_eval_non_live_phase'
