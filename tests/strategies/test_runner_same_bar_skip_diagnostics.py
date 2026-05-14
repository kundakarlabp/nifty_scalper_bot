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
