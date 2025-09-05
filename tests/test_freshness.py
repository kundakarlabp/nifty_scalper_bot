from datetime import datetime, timedelta, timezone

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


def test_freshness_handles_naive_and_aware() -> None:
    now = datetime.now(timezone.utc)
    f = compute(
        now=now,
        last_tick_ts=datetime.utcnow(),
        last_bar_open_ts=(now - timedelta(minutes=1)).replace(tzinfo=None),
        tf_seconds=60,
        max_tick_lag_s=8,
        max_bar_lag_s=75,
    )
    assert f.tick_lag_s is not None and f.bar_lag_s is not None


def test_freshness_accepts_iso_strings() -> None:
    now = datetime.utcnow()
    tick_ts = now.isoformat()
    bar_ts = (now - timedelta(minutes=1)).isoformat()
    f = compute(
        now=now,
        last_tick_ts=tick_ts,
        last_bar_open_ts=bar_ts,
        tf_seconds=60,
        max_tick_lag_s=8,
        max_bar_lag_s=75,
    )
    assert f.tick_lag_s is not None and f.bar_lag_s is not None
