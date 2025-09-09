from datetime import datetime, timedelta
from types import SimpleNamespace

from src.data.source import LiveKiteSource
from src.config import settings
from src.utils import time_windows


def test_live_warmup_from_ticks(monkeypatch):
    monkeypatch.setattr(
        settings, "instruments", SimpleNamespace(instrument_token=1), raising=False
    )
    base = datetime(2024, 1, 1, 9, 15)
    src = LiveKiteSource(kite=None)
    src.hist_mode = "live_warmup"
    for i in range(35):
        tick = {"last_price": 100 + i, "timestamp": base + timedelta(minutes=i)}
        src.on_tick(tick)
    monkeypatch.setattr(time_windows, "now_ist", lambda: base + timedelta(minutes=35))
    assert src.have_min_bars(30)
