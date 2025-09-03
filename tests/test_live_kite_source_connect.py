from src.data.source import LiveKiteSource, WARMUP_BARS
from src.config import settings


def test_connect_subscribes_and_backfills(monkeypatch):
    calls = {}

    class DummyKite:
        MODE_FULL = "full"

        def subscribe(self, tokens):  # type: ignore[no-untyped-def]
            pass

        def set_mode(self, mode, tokens):  # type: ignore[no-untyped-def]
            pass

    called = {"sub": False, "mode": False}

    def sub(self, tokens):  # type: ignore[no-untyped-def]
        called["sub"] = True

    def mode(self, mode, tokens):  # type: ignore[no-untyped-def]
        called["mode"] = True
        raise RuntimeError("boom")

    monkeypatch.setattr(DummyKite, "subscribe", sub)
    monkeypatch.setattr(DummyKite, "set_mode", mode)

    import src.data.source as srcmod
    monkeypatch.setattr(srcmod, "settings", settings, raising=False)
    monkeypatch.setattr(srcmod.settings.instruments, "spot_token", 123, raising=False)
    monkeypatch.setattr(srcmod.settings.instruments, "instrument_token", 123, raising=False)

    dummy = DummyKite()

    def fake_backfill(source, *, required_bars, token, timeframe):  # type: ignore[no-untyped-def]
        calls["args"] = (required_bars, token, timeframe)

    monkeypatch.setattr("src.data.source.ensure_backfill", fake_backfill)

    src = LiveKiteSource(dummy)
    src.connect()

    token = getattr(settings.instruments, "spot_token", settings.instruments.instrument_token)
    assert called["sub"]
    assert called["mode"]
    assert calls["args"] == (WARMUP_BARS, token, settings.data.timeframe)


def test_connect_handles_errors(monkeypatch):
    class BadKite:
        MODE_FULL = "full"

        def subscribe(self, tokens):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        def set_mode(self, mode, tokens):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    def bad_backfill(source, *, required_bars, token, timeframe):  # type: ignore[no-untyped-def]
        raise RuntimeError("bf")

    monkeypatch.setattr("src.data.source.ensure_backfill", bad_backfill)

    src = LiveKiteSource(BadKite())
    src.connect()
