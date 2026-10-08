from __future__ import annotations


def _volume_from_tick(runner, symbol: str, tick: dict[str, object]) -> int:
    def _extract_int(d, *keys):
        for k in keys:
            if d.get(k) is not None:
                try:
                    return int(float(d[k]))
                except (ValueError, TypeError):
                    continue
        return 0

    raw_volume_delta = _extract_int(tick, 'volume_delta', 'volume')
    raw_volume_cumulative = _extract_int(
        tick, 'volume_cumulative', 'volume_traded', 'volume_traded_today'
    )

    if 'volume_delta' in tick or 'volume' in tick:
        volume = max(int(raw_volume_delta), 0)
    else:
        volume = 0
        first_tick_seen = symbol not in runner._last_cumulative_volume
        if raw_volume_cumulative > 0:
            last_cum = runner._last_cumulative_volume.get(symbol, -1)
            if last_cum < 0:
                volume = 0
            elif raw_volume_cumulative >= last_cum:
                volume = raw_volume_cumulative - last_cum
            else:
                volume = 0
            runner._last_cumulative_volume[symbol] = raw_volume_cumulative
        elif first_tick_seen:
            volume = 0
            runner._last_cumulative_volume[symbol] = raw_volume_cumulative
    return volume


class _RunnerStub:
    def __init__(self) -> None:
        self._last_cumulative_volume: dict[str, int] = {}


def test_runner_prefers_explicit_delta_volume() -> None:
    runner = _RunnerStub()
    symbol = 'NFO:NIFTY26MAY23500CE'
    runner._last_cumulative_volume[symbol] = 900
    tick = {'volume_delta': 25, 'volume_cumulative': 5000}
    assert _volume_from_tick(runner, symbol, tick) == 25
    assert runner._last_cumulative_volume[symbol] == 900


def test_runner_falls_back_to_cumulative_when_needed() -> None:
    runner = _RunnerStub()
    symbol = 'NFO:NIFTY26MAY23500CE'
    tick1 = {'volume_traded_today': 1000}
    tick2 = {'volume_traded_today': 1040}
    assert _volume_from_tick(runner, symbol, tick1) == 0
    assert _volume_from_tick(runner, symbol, tick2) == 40


def test_intrabar_eval_uses_canonical_interval_volume_delta(monkeypatch) -> None:
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    runner = StrategyRunner.__new__(StrategyRunner)
    symbol = "NFO:NIFTY26MAY23500CE"
    runner._data_phase = {symbol: "LIVE"}
    runner._active_selected_ce = symbol
    runner._active_selected_pe = "NFO:NIFTY26MAY23500PE"
    runner._last_same_bar_eval_ts_by_symbol = {symbol: 100.0}
    runner._last_eval_price_by_symbol = {symbol: 100.0}
    runner._last_same_bar_eval_block_reason_by_symbol = {}
    runner._last_same_bar_eval_block_detail_by_symbol = {
        symbol: {
            "spread_now": 1.0,
            "quote_marker": 7,
            "volume_now": 100.0,
        }
    }
    runner._required_bars_for_symbol = lambda _symbol: 30

    monkeypatch.setenv("RUNNER_INTRABAR_EVAL_SELECTED_SECONDS", "10")
    monkeypatch.setenv("RUNNER_INTRABAR_EVAL_MIN_PRICE_MOVE_PCT", "99")
    monkeypatch.setenv("RUNNER_INTRABAR_SPREAD_DELTA_MIN", "99")
    monkeypatch.setenv("RUNNER_INTRABAR_VOLUME_DELTA_MIN", "50")

    reason = runner._same_bar_eval_reason(
        symbol=symbol,
        price=100.0,
        tick={
            "bid": 99.5,
            "ask": 100.5,
            "volume_delta": 80,
            "volume": 80,
            "quote_update_version": 7,
        },
        candle_count=30,
        now_ts=101.0,
    )

    assert reason == "same_bar_market_update_eval"
