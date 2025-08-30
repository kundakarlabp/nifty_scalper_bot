from datetime import datetime

from src.backtesting.cli import build_walkforward_windows


def test_walkforward_splits():
    start = datetime(2025, 6, 1)
    end = datetime(2025, 8, 30)
    wins = build_walkforward_windows(start, end, 6, 2)
    assert len(wins) == 2
    assert wins[0] == (
        datetime(2025, 6, 1),
        datetime(2025, 7, 13),
        datetime(2025, 7, 27),
    )
    assert wins[1] == (
        datetime(2025, 7, 13),
        datetime(2025, 8, 24),
        datetime(2025, 9, 7),
    )
