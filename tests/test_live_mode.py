import pandas as pd
import pytest
from datetime import datetime, timedelta

from src import main as main_mod
from src.config import settings
from src.strategies.runner import StrategyRunner
from src.boot import validate_env


def _prep_live_env(monkeypatch):
    if not (
        settings.zerodha.api_key
        and settings.zerodha.api_secret
        and settings.zerodha.access_token
    ):
        pytest.skip("Zerodha credentials not provided")

    monkeypatch.setattr(settings, "enable_live_trading", True)
    monkeypatch.setattr(settings.telegram, "bot_token", "x")
    monkeypatch.setattr(settings.telegram, "chat_id", 1)
    monkeypatch.setattr(validate_env, "SKIP_BROKER_VALIDATION", True)


def test_build_kite_session_success(monkeypatch):
    _prep_live_env(monkeypatch)
    try:
        kite = main_mod._build_kite_session()
        end = datetime.utcnow()
        start = end - timedelta(days=1)
        kite.historical_data(256265, start, end, "minute")
    except Exception as e:  # pragma: no cover - network issues
        pytest.skip(f"Kite session unavailable: {e}")
    assert kite is not None


class DummyTelegram:
    @classmethod
    def create(cls, *_, **__):
        return cls()

    def send_message(self, msg: str) -> None:  # pragma: no cover - logging only
        pass

    def stop_polling(self) -> None:  # pragma: no cover - logging only
        pass


def test_main_calls_set_live_mode(monkeypatch):
    _prep_live_env(monkeypatch)
    monkeypatch.setattr(main_mod, "_stop_flag", True, raising=False)
    monkeypatch.setattr(main_mod, "_import_telegram_class", lambda: DummyTelegram)
    monkeypatch.setattr(main_mod, "_install_signal_handlers", lambda runner: None)
    monkeypatch.setattr(StrategyRunner, "start", lambda self: None)
    monkeypatch.setattr(StrategyRunner, "shutdown", lambda self: None)
    called = {}

    def fake_set_live_mode(self, val):
        called["val"] = val

    monkeypatch.setattr(StrategyRunner, "set_live_mode", fake_set_live_mode)

    assert main_mod.main() == 0
    assert called.get("val") is True
