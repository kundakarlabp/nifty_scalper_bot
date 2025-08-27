import pandas as pd

from src import main as main_mod
from src.config import settings
from src.strategies.runner import StrategyRunner


def _prep_live_env(monkeypatch):
    monkeypatch.setattr(settings, "enable_live_trading", True)
    monkeypatch.setattr(settings.zerodha, "api_key", "k")
    monkeypatch.setattr(settings.zerodha, "access_token", "t")
    monkeypatch.setattr(settings.telegram, "bot_token", "x")
    monkeypatch.setattr(settings.telegram, "chat_id", 1)

    class DummySource:
        def __init__(self, kite):
            pass

        def connect(self) -> None:
            pass

        def fetch_ohlc(self, *_, **__):
            ts = pd.Timestamp("2024-01-01")
            return pd.DataFrame(
                {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [0]}, index=[ts]
            )

    monkeypatch.setattr("src.config.LiveKiteSource", DummySource)


def test_build_kite_session_success(monkeypatch):
    _prep_live_env(monkeypatch)
    kite = main_mod._build_kite_session()
    assert kite is not None


def test_main_calls_set_live_mode(monkeypatch):
    _prep_live_env(monkeypatch)
    monkeypatch.setattr(main_mod, "_stop_flag", True, raising=False)
    monkeypatch.setattr(main_mod, "_wire_real_telegram", lambda runner: None)
    monkeypatch.setattr(main_mod, "_install_signal_handlers", lambda runner: None)
    monkeypatch.setattr(StrategyRunner, "start", lambda self: None)
    monkeypatch.setattr(StrategyRunner, "shutdown", lambda self: None)
    called = {}

    def fake_set_live_mode(self, val):
        called["val"] = val

    monkeypatch.setattr(StrategyRunner, "set_live_mode", fake_set_live_mode)

    assert main_mod.main() == 0
    assert called.get("val") is True
