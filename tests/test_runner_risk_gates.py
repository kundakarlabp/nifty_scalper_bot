from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:
        pass


def test_risk_gates_blocked_when_signal_missing_prices():
    runner = StrategyRunner(kite=None, telegram_controller=DummyTelegram())
    gates = runner._risk_gates_for({})
    assert gates["sl_valid"] is False
    runner._last_flow_debug = {"risk_gates": gates}
    summary = runner.get_compact_diag_summary()
    assert summary["status_messages"]["risk_gates"] == "blocked"
