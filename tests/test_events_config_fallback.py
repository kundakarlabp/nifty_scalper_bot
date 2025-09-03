from src.strategies.runner import StrategyRunner
from src.config import settings


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - simple stub
        pass


def test_events_config_fallback(monkeypatch) -> None:
    monkeypatch.setenv("EVENTS_CONFIG_FILE", "config/missing.yaml")
    monkeypatch.setenv("ACTIVE_DATA_PROVIDER", "yfinance")
    monkeypatch.setenv("ACTIVE_CONNECTOR", "shadow")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    object.__setattr__(settings, "ACTIVE_DATA_PROVIDER", "yfinance")
    object.__setattr__(settings, "ACTIVE_CONNECTOR", "shadow")
    orig_live = settings.enable_live_trading
    object.__setattr__(settings, "enable_live_trading", False)
    try:
        runner = StrategyRunner(telegram_controller=DummyTelegram())
        assert runner.events_path == "config/events.yaml"
        assert runner.event_cal is not None
        assert runner.event_cal.source_path == "config/events.yaml"
    finally:
        delattr(settings, "ACTIVE_DATA_PROVIDER")
        delattr(settings, "ACTIVE_CONNECTOR")
        object.__setattr__(settings, "enable_live_trading", orig_live)
