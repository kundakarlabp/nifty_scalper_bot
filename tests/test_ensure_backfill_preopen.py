from unittest.mock import Mock

from freezegun import freeze_time

from src.config import settings
from src.data.source import LiveKiteSource


@freeze_time("2024-01-03 03:40:00")
def test_ensure_backfill_synthetic_preopen(monkeypatch) -> None:
    kite = Mock()
    kite.historical_data.return_value = []
    ds = LiveKiteSource(kite)
    ds.get_last_price = lambda _t: 100.0
    seeded: list = []
    ds.seed_ohlc = lambda df: seeded.append(df)
    monkeypatch.setattr(settings.instruments, "instrument_token", 256265, raising=False)
    ds.ensure_backfill(required_bars=15, token=256265, timeframe="minute")
    assert int(getattr(ds, "_synth_bars_n", 0)) >= 15
    assert ds.have_min_bars(15) is True
    assert seeded and len(seeded[-1]) >= 15
