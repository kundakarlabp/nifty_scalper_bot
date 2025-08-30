from src.backtesting.metrics import reject


def test_metrics_reject():
    bad = {"trades": 10, "PF": 1.0, "Win%": 40, "AvgR": 0.1, "MaxDD_R": 7}
    good = {"trades": 40, "PF": 1.5, "Win%": 60, "AvgR": 0.5, "MaxDD_R": 2}
    assert reject(bad)
    assert not reject(good)
