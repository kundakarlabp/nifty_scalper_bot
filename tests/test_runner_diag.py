from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - simple stub
        pass


def test_risk_gates_skipped_when_paused(monkeypatch):
    runner = StrategyRunner(kite=None, telegram_controller=DummyTelegram())
    monkeypatch.setattr(runner, "_within_trading_window", lambda: True)
    runner.pause()
    runner.process_tick(tick=None)
    summary = runner.get_compact_diag_summary()
    assert summary["status_messages"]["risk_gates"] == "skipped"
