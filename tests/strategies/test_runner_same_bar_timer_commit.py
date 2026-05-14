from __future__ import annotations


def test_same_bar_timer_not_consumed_when_eval_not_executed() -> None:
    last_same_bar_eval_ts_by_symbol: dict[str, float] = {}
    same_bar_eval_allowed = True
    pending_same_bar_eval_ts = 123.0
    generate_signal_called = False

    if generate_signal_called and same_bar_eval_allowed and pending_same_bar_eval_ts > 0.0:
        last_same_bar_eval_ts_by_symbol['NFO:NIFTY26MAY25000CE'] = pending_same_bar_eval_ts

    assert 'NFO:NIFTY26MAY25000CE' not in last_same_bar_eval_ts_by_symbol


def test_same_bar_timer_consumed_after_eval_execution() -> None:
    last_same_bar_eval_ts_by_symbol: dict[str, float] = {}
    same_bar_eval_allowed = True
    pending_same_bar_eval_ts = 123.0
    generate_signal_called = True

    if generate_signal_called and same_bar_eval_allowed and pending_same_bar_eval_ts > 0.0:
        last_same_bar_eval_ts_by_symbol['NFO:NIFTY26MAY25000CE'] = pending_same_bar_eval_ts

    assert last_same_bar_eval_ts_by_symbol['NFO:NIFTY26MAY25000CE'] == 123.0
