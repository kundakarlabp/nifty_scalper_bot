from unittest.mock import Mock

from freezegun import freeze_time

from src.config import settings
from src.data.source import LiveKiteSource


@freeze_time("2024-01-03 03:40:00")
def test_ensure_backfill_preopen_no_data(monkeypatch, caplog) -> None:
    kite = Mock()
    kite.historical_data.return_value = []
    ds = LiveKiteSource(kite)
    monkeypatch.setattr(settings.instruments, "instrument_token", 256265, raising=False)
    with caplog.at_level("INFO"):
        ds.ensure_backfill(required_bars=15, token=256265, timeframe="minute")
    assert "Backfill skipped (no OHLC). Using live_warmup bars." in caplog.text
    assert ds.have_min_bars(15) is False
