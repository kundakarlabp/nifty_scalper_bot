from datetime import datetime

import src.data.source as source
from src.data.source import LiveKiteSource


def test_live_warmup_ticks(monkeypatch):
    monkeypatch.setattr(source, "data_warmup_disable", lambda: True)
    src = LiveKiteSource(kite=None)
    assert src.hist_mode == "live_warmup"
    assert src.bar_builder is not None

    t1 = datetime(2024, 1, 1, 9, 15, 5)
    t2 = datetime(2024, 1, 1, 9, 15, 40)
    t3 = datetime(2024, 1, 1, 9, 16, 5)
    src.on_tick({"last_price": 100.0, "timestamp": t1, "volume": 10})
    src.on_tick({"last_price": 101.0, "timestamp": t2, "volume": 20})
    src.on_tick({"last_price": 102.0, "timestamp": t3, "volume": 30})

    df = src.get_recent_bars(2)
    assert len(df) >= 2
    res = src.have_min_bars(2)
    assert isinstance(res, bool)
    assert res
    assert not src.have_min_bars(100)

    src.bar_builder = None
    src._synth_bars_n = 50
    assert src.have_min_bars(30)
