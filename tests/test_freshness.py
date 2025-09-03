from datetime import datetime, timedelta

from src.utils.freshness import compute


def test_freshness_ok_and_fail() -> None:
    now = datetime.utcnow()
    f_ok = compute(
        now=now,
        last_tick_ts=now,
        last_bar_open_ts=now - timedelta(minutes=1),
        tf_seconds=60,
        max_tick_lag_s=8,
        max_bar_lag_s=75,
    )
    assert f_ok.ok
    f_bad = compute(
        now=now,
        last_tick_ts=now - timedelta(seconds=20),
        last_bar_open_ts=now - timedelta(minutes=2),
        tf_seconds=60,
        max_tick_lag_s=8,
        max_bar_lag_s=75,
    )
    assert not f_bad.ok
