from src.strategies.warmup import required_bars, check


class C:
    pass


def test_required_bars_consistent() -> None:
    c = C()
    c.warmup_bars_min = 20
    c.atr_period = 14
    c.ema_slow = 21
    c.regime_min_bars = 20
    c.features_min_bars = 20
    need = required_bars(c)
    assert need >= 20 and need >= 16 and need >= 23
    assert check(c, have_bars=need - 1).ok is False
    assert check(c, have_bars=need).ok is True
