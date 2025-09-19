from types import SimpleNamespace

from src.data.source import LiveKiteSource, WARMUP_BARS


class _Gate:
    def should_emit(self, key: str, force: bool = False) -> bool:
        return True


def test_connect_subscribes_and_backfills(monkeypatch):
    calls = {}

    class DummyKite:
        MODE_FULL = "full"

        def subscribe(self, tokens):
            calls["subscribe"] = tokens

        def set_mode(self, mode, tokens):
            calls["mode"] = (mode, tokens)

    src = LiveKiteSource(DummyKite())

    def fake_backfill(self, *, required_bars, token, timeframe):
        calls["backfill"] = (required_bars, token, timeframe)

    monkeypatch.setattr(LiveKiteSource, "ensure_backfill", fake_backfill, raising=False)
    settings_stub = SimpleNamespace(
        instruments=SimpleNamespace(spot_token=1, instrument_token=1),
        build_log_gate=lambda interval_s=None: _Gate(),
    )
    monkeypatch.setattr("src.data.source.settings", settings_stub, raising=False)

    src.connect()

    assert calls["subscribe"] == [1]
    assert calls["mode"] == (DummyKite.MODE_FULL, [1])
    assert calls["backfill"] == (WARMUP_BARS, 1, "minute")


def test_connect_without_kite(monkeypatch):
    calls = {}

    def fake_backfill(self, *, required_bars, token, timeframe):
        calls["backfill"] = (required_bars, token, timeframe)

    src = LiveKiteSource(kite=None)
    monkeypatch.setattr(LiveKiteSource, "ensure_backfill", fake_backfill, raising=False)
    settings_stub = SimpleNamespace(
        instruments=SimpleNamespace(spot_token=2, instrument_token=2),
        build_log_gate=lambda interval_s=None: _Gate(),
    )
    monkeypatch.setattr("src.data.source.settings", settings_stub, raising=False)

    src.connect()

    assert calls["backfill"] == (WARMUP_BARS, 2, "minute")
