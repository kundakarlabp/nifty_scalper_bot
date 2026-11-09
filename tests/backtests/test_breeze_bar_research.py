"""One-minute Breeze aggregation is causal, session-aware and never fills gaps."""

from datetime import date, datetime, time, timedelta

import pytest

from nifty_scalper_bot.backtesting.breeze_bar_research import aggregate_breeze_session

DAY = date(2025, 2, 3)


def rows(count=15, *, start="09:15", skip=()):
    base = datetime.fromisoformat(f"2025-02-03T{start}:00+05:30")
    result = []
    for i in range(count):
        if i in skip:
            continue
        stamp = (base + timedelta(minutes=i)).isoformat()
        result.append([stamp, 100+i, 104+i, 98+i, 101+i, 50, i*100])
    return result


def test_complete_five_and_fifteen_minute_bar_time_availability():
    for interval, total in ((5, 3), (15, 1)):
        result = aggregate_breeze_session(
            rows(), day=DAY, interval_minutes=interval,
            session_start=time(9,15), session_end=time(9,30)
        )
        assert len(result.bars) == total
        assert not result.missing_windows
        first = result.bars[0]
        assert first.open == 100 and first.close == 100 + interval
        assert first.volume == interval * 50
        assert first.source_minutes == interval
        assert first.available_at == first.start + timedelta(minutes=interval)
        assert not first.execution_eligible
        assert not result.source_time_convention_verified


def test_missing_minute_invalidates_entire_5m_window_without_interpolation():
    result=aggregate_breeze_session(
        rows(skip=(2,)), day=DAY, interval_minutes=5,
        session_start=time(9,15), session_end=time(9,30)
    )
    assert len(result.bars)==2
    assert result.missing_windows == ("2025-02-03T09:15:00+05:30",)
    assert result.bars[0].start.time()==time(9,20)


def test_duplicate_or_broken_timestamps_fail_closed():
    r=rows()
    with pytest.raises(ValueError,match="invalid_or_conflicting_candle"):
        aggregate_breeze_session(r+[r[0]],day=DAY,interval_minutes=5,
          session_start=time(9,15),session_end=time(9,30))
    r[0][0]="2025-02-03T09:15:00"
    with pytest.raises(ValueError,match="invalid_or_conflicting_candle"):
        aggregate_breeze_session(r,day=DAY,interval_minutes=5,
          session_start=time(9,15),session_end=time(9,30))


def test_explicit_special_session_grid_never_assumes_regular_hours():
    data=rows(60,start="13:45")
    data=[[v[0].replace("2025-02-03","2025-10-21"),*v[1:]] for v in data]
    result=aggregate_breeze_session(
        data,day=date(2025,10,21),interval_minutes=15,
        session_start=time(13,45),session_end=time(14,45)
    )
    assert len(result.bars)==4
    assert result.bars[-1].available_at.time()==time(14,45)
    with pytest.raises(ValueError,match="invalid_or_conflicting_candle"):
        aggregate_breeze_session(data,day=date(2025,10,21),
          interval_minutes=15,session_start=time(9,15),session_end=time(15,30))
