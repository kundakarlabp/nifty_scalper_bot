import pytest

from src.data.source import LiveKiteSource


def test_backfill_skipped_logs(caplog):
    src = LiveKiteSource(kite=None)
    src.hist_mode = "backfill"
    with caplog.at_level("INFO"):
        src.ensure_backfill(required_bars=1, token=1, timeframe="minute")
    assert "Backfill skipped (no OHLC). Using live_warmup bars." in caplog.text


def test_backfill_raises_in_broker_mode():
    src = LiveKiteSource(kite=None)
    src.hist_mode = "broker"
    with pytest.raises(RuntimeError, match="historical_data empty"):
        src.ensure_backfill(required_bars=1, token=1, timeframe="minute")
