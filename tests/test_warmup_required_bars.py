from src.strategies.warmup import compute_required_bars, warmup_status


class C:
    pass


def test_required_bars_consistent() -> None:
    c = C()
    c.min_bars_required = 20
    c.warmup_bars = 25
    need = compute_required_bars(c, default_min=15, atr_period=14)
    assert need >= 20 and need >= 19 and need >= 25
    assert warmup_status(need - 1, need).ok is False
    assert warmup_status(need, need).ok is True
