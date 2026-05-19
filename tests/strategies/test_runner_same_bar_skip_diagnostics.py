from __future__ import annotations

import logging

from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_same_bar_skip_diagnostics_payload(caplog):
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = logging.getLogger('test_runner_same_bar_skip_diagnostics')
    runner._data_phase = {'NFO:NIFTY26MAY25200CE': 'LIVE'}
    runner._required_bars_for_symbol = lambda _s: 50
    runner._indicator_engine = type('E', (), {'get_history': lambda *_: []})()

    with caplog.at_level(logging.INFO):
        runner._emit_runner_eval_decision(
            symbol='NFO:NIFTY26MAY25200CE',
            stage='phase9',
            reason='strategy_eval_skipped_same_bar',
            allowed=False,
            same_bar_block_reason='intrabar_eval_interval_not_elapsed',
            active_selected_ce='NFO:NIFTY26MAY25000CE',
            active_selected_pe='NFO:NIFTY26MAY25000PE',
            intrabar_selected_seconds='10',
            intrabar_non_selected_seconds='60',
        )

    assert any('strategy_eval_skipped_same_bar' in rec.message for rec in caplog.records)
    payload = next(rec.__dict__ for rec in caplog.records if 'strategy_eval_skipped_same_bar' in rec.message)
    assert payload['same_bar_block_reason'] == 'intrabar_eval_interval_not_elapsed'
    assert payload['active_selected_ce'] == 'NFO:NIFTY26MAY25000CE'
    assert payload['active_selected_pe'] == 'NFO:NIFTY26MAY25000PE'
    assert payload['intrabar_selected_seconds'] == '10'
    assert payload['intrabar_non_selected_seconds'] == '60'


def test_runner_eval_decision_visible_message_uses_clear_version_names(caplog):
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = logging.getLogger('test_runner_versions')
    runner._symbol_state = {}
    runner._symbol_states = {}
    runner._candle_versions = {'NFO:NIFTY26MAY25200CE': 12}
    runner._last_strategy_versions = {'NFO:NIFTY26MAY25200CE': 11}
    runner._last_bar_ts = {}
    runner._market_data = type('M', (), {'_last_tick_time': {}, 'get_ohlc_bars': lambda *_: []})()
    runner._live_bar_seen = set()
    runner._required_bars_for_symbol = lambda _s: 1
    runner._data_phase = {}
    runner._hydration_attempted_symbols = set()
    runner._last_hydration_reason_by_symbol = {}
    runner._active_symbols = set()
    runner._indicator_engine = type('E', (), {'get_history': lambda *_: []})()
    runner._should_log_throttled = lambda *_: True
    with caplog.at_level(logging.INFO):
        runner._emit_runner_eval_decision(symbol='NFO:NIFTY26MAY25200CE',stage='phase9',reason='evaluation_no_signal',allowed=True)
    payload = caplog.records[-1].__dict__
    assert 'indicator_history_count' in payload and 'live_candle_version' in payload and 'last_eval_live_candle_version' in payload
    msg = caplog.records[-1].message
    assert 'indicator_history_count' in msg
    assert 'live_candle_version' in msg
    assert 'last_eval_live_candle_version' in msg


def test_context_symbols_bypass_min_bars_for_snapshot_update():
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._active_symbols = {'NSE:NIFTY'}
    runner._required_bars_for_symbol = lambda _s: 50
    runner._context_required_bars = 50
    runner._option_required_bars = 5
    runner._indicator_engine = type('E', (), {'get_history': lambda *_: [1, 2], 'has_min_bars': lambda *_: False})()
    runner._is_context_symbol = lambda _s: True
    runner._is_tradable_symbol = lambda _s: False
    runner._symbol_role_for_runner = lambda _s: 'spot_context'
    runner._logger = logging.getLogger('test_runner_context_bypass')
    runner._emit_runner_eval_decision = lambda **_: None
    assert runner._strategy_evaluation_allowed('NSE:NIFTY')


def test_quote_update_version_increments_on_full_quote(caplog):
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = logging.getLogger('test_runner_quote_version')
    runner._active_symbols = {'NFO:NIFTY26MAY25200CE'}
    runner._data_phase = {'NFO:NIFTY26MAY25200CE': 'LIVE'}
    runner._required_bars_for_symbol = lambda _s: 1
    runner._symbol_state = {}
    runner._symbol_states = {}
    runner._candle_versions = {'NFO:NIFTY26MAY25200CE': 1}
    runner._last_strategy_versions = {'NFO:NIFTY26MAY25200CE': 0}
    runner._quote_update_versions = {}
    runner._last_bar_ts = {}
    runner._market_data = type('M', (), {'_last_tick_time': {}, 'get_ohlc_bars': lambda *_: []})()
    runner._live_bar_seen = set()
    runner._hydration_attempted_symbols = set()
    runner._last_hydration_reason_by_symbol = {}
    runner._indicator_engine = type('E', (), {'get_history': lambda *_: [1]})()
    runner._should_log_throttled = lambda *_: True
    with caplog.at_level(logging.INFO):
        runner._quote_update_versions['NFO:NIFTY26MAY25200CE'] = runner._quote_update_versions.get('NFO:NIFTY26MAY25200CE', 0) + 1
        runner._emit_runner_eval_decision(symbol='NFO:NIFTY26MAY25200CE',stage='phase9',reason='evaluation_no_signal',allowed=True)
    payload = caplog.records[-1].__dict__
    assert isinstance(payload['quote_update_version'], int)
    assert payload['quote_update_version'] >= 1


def test_quote_update_version_fallback_from_datahub(caplog):
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._logger = logging.getLogger('test_runner_quote_version_fallback')
    runner._active_symbols = {'NFO:NIFTY26MAY25200CE'}
    runner._data_phase = {'NFO:NIFTY26MAY25200CE': 'LIVE'}
    runner._required_bars_for_symbol = lambda _s: 1
    runner._symbol_state = {}
    runner._symbol_states = {}
    runner._candle_versions = {'NFO:NIFTY26MAY25200CE': 1}
    runner._last_strategy_versions = {'NFO:NIFTY26MAY25200CE': 0}
    runner._quote_update_versions = {}
    runner._last_bar_ts = {}
    runner._market_data = type('M', (), {'_last_tick_time': {}, 'get_ohlc_bars': lambda *_: []})()
    runner._data_hub = type('H', (), {'quote_update_version': lambda *_: 3})()
    runner._live_bar_seen = set()
    runner._hydration_attempted_symbols = set()
    runner._last_hydration_reason_by_symbol = {}
    runner._indicator_engine = type('E', (), {'get_history': lambda *_: [1]})()
    runner._should_log_throttled = lambda *_: True
    with caplog.at_level(logging.INFO):
        runner._emit_runner_eval_decision(symbol='NFO:NIFTY26MAY25200CE', stage='phase9', reason='evaluation_no_signal', allowed=True)
    assert caplog.records[-1].__dict__['quote_update_version'] == 3


def test_option_execution_min_bars_applies_only_to_options(monkeypatch):
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._active_symbols = {'NSE:NIFTY', 'NFO:NIFTY26MAYFUT', 'NFO:NIFTY26MAY25200CE'}
    runner._restored_from_cache_symbols = set()
    runner._context_required_bars = 50
    runner._is_option_symbol_tick_fresh = lambda _s: True
    runner._active_selected_ce = 'NFO:NIFTY26MAY25200CE'
    runner._active_selected_pe = 'NFO:NIFTY26MAY25200PE'
    runner._should_log_throttled = lambda *_: False
    runner._logger = logging.getLogger('test_option_min_bars')
    runner._emit_runner_eval_decision = lambda **_: None
    runner._symbol_role_for_runner = lambda s: 'tradable_option' if s.endswith(('CE', 'PE')) else ('spot_context' if s == 'NSE:NIFTY' else 'futures_context')
    runner._is_context_symbol = lambda s: runner._symbol_role_for_runner(s) in {'spot_context', 'futures_context'}
    runner._is_tradable_symbol = lambda s: runner._symbol_role_for_runner(s) == 'tradable_option'
    runner._required_bars_for_symbol = lambda _s: 2
    runner._indicator_engine = type('E', (), {'get_history': lambda *_: [1, 2], 'has_min_bars': lambda *_: True})()
    monkeypatch.setenv("OPTION_EXECUTION_MIN_BARS", "5")
    assert runner._strategy_evaluation_allowed('NSE:NIFTY')
    assert runner._strategy_evaluation_allowed('NFO:NIFTY26MAYFUT')


def test_tick_has_quote_update_depth_without_bid_ask():
    tick = {"depth": {"buy": [{"price": 100.0, "quantity": 10}], "sell": [{"price": 100.5, "quantity": 8}]}}
    assert StrategyRunner._tick_has_quote_update(tick) is True


def test_initial_hydrated_eval_reason_then_same_bar_skip():
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._active_symbols = {"NFO:NIFTY26MAY25200CE"}
    runner._required_candles = 1
    runner._runtime_data_hard_ready = True
    runner._runtime_readiness_reason = ""
    runner._active_selected_ce = "NFO:NIFTY26MAY25200CE"
    runner._active_selected_pe = "NFO:NIFTY26MAY25200PE"
    runner._symbol_has_completed_strategy_eval = set()
    runner._quote_update_versions = {}
    runner._candle_versions = {"NFO:NIFTY26MAY25200CE": 0}
    runner._last_strategy_versions = {"NFO:NIFTY26MAY25200CE": 0}
    runner._last_same_bar_eval_block_reason_by_symbol = {}
    runner._last_same_bar_eval_block_detail_by_symbol = {}
    runner._last_same_bar_eval_ts_by_symbol = {}
    runner._indicator_engine = type("E", (), {"get_history": lambda *_: [1] * 300})()
    decisions: list[dict] = []
    runner._emit_runner_eval_decision = lambda **kw: decisions.append(kw)
    symbol = "NFO:NIFTY26MAY25200CE"
    candle_count = len(runner._indicator_engine.get_history(symbol) or [])
    current_version = 0
    last_version = 0
    first_hydrated_eval = candle_count > 0 and symbol not in runner._symbol_has_completed_strategy_eval
    if current_version <= last_version and not first_hydrated_eval:
        runner._emit_runner_eval_decision(symbol=symbol, stage="phase9", reason="strategy_eval_skipped_same_bar", allowed=False)
    elif first_hydrated_eval:
        runner._emit_runner_eval_decision(symbol=symbol, stage="phase9", reason="initial_hydrated_eval", allowed=True)
    runner._symbol_has_completed_strategy_eval.add(symbol)
    first = decisions[-1]
    assert first["allowed"] is True
    assert first["reason"] == "initial_hydrated_eval"
    decisions.clear()
    first_hydrated_eval = candle_count > 0 and symbol not in runner._symbol_has_completed_strategy_eval
    if current_version <= last_version and not first_hydrated_eval:
        runner._emit_runner_eval_decision(symbol=symbol, stage="phase9", reason="strategy_eval_skipped_same_bar", allowed=False)
    second = decisions[-1]
    assert second["allowed"] is False
    assert second["reason"] == "strategy_eval_skipped_same_bar"
