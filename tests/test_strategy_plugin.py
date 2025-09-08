from src.strategies.runner import StrategyRunner
from src.config import settings


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - no behavior
        pass


def test_component_selection_via_env(monkeypatch) -> None:
    """Runtime components follow environment variables."""
    monkeypatch.setenv("ACTIVE_DATA_PROVIDER", "kite")
    monkeypatch.setenv("ACTIVE_CONNECTOR", "shadow")
    object.__setattr__(settings, "ACTIVE_DATA_PROVIDER", "kite")
    object.__setattr__(settings, "ACTIVE_CONNECTOR", "shadow")
    try:
        runner = StrategyRunner(telegram_controller=DummyTelegram())
        snap = runner.get_status_snapshot()
        comps = snap["components"]
        assert comps["data_provider"] == "kite"
        assert comps["order_connector"] == "shadow"
    finally:
        delattr(settings, "ACTIVE_DATA_PROVIDER")
        delattr(settings, "ACTIVE_CONNECTOR")
