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
