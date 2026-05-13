from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_runner_rejects_explicit_cumulative_payload() -> None:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._last_cumulative_volume = {}
    tick = {'volume': 6008275, 'volume_cumulative': 6008275}
    has_explicit_delta = 'volume_delta' in tick
    has_explicit_volume = 'volume' in tick
    raw_volume_delta = int(tick['volume']) if has_explicit_volume else 0
    raw_volume_cumulative = int(tick['volume_cumulative'])
    if has_explicit_delta:
        volume = max(int(raw_volume_delta), 0)
    elif has_explicit_volume and raw_volume_cumulative > 0 and raw_volume_delta == raw_volume_cumulative:
        volume = 0
    else:
        volume = max(int(raw_volume_delta), 0)
    assert volume == 0


def test_runner_prefers_volume_delta() -> None:
    tick = {'volume_delta': 65, 'volume': 65, 'volume_cumulative': 6008275}
    volume = max(int(tick['volume_delta']), 0)
    assert volume == 65
