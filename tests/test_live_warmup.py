from datetime import datetime, timedelta, timezone

from src.data.source import LiveKiteSource, MinuteBarBuilder


def test_live_warmup_from_ticks() -> None:
    src = LiveKiteSource(kite=None)
    src.hist_mode = "live_warmup"
    src.bar_builder = MinuteBarBuilder(max_bars=5)
    t = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    src.on_tick({"last_price": 100, "exchange_timestamp": t})
    src.on_tick({"last_price": 101, "exchange_timestamp": t + timedelta(minutes=1)})
    assert src.have_min_bars(2).status.name == "OK"
    df = src.fetch_ohlc_df(token=1, start=t, end=t + timedelta(minutes=2), timeframe="minute")
    assert len(df) == 2
    assert df.iloc[0]["open"] == 100
