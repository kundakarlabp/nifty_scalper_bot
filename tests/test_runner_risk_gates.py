from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:
        pass


def test_risk_gates_blocked_when_signal_missing_prices():
    runner = StrategyRunner(kite=None, telegram_controller=DummyTelegram())
    gates = runner._risk_gates_for({})
    assert gates["sl_valid"] is False
    assert isinstance(gates["sl_valid"], bool)
    assert runner._last_error and "invalid entry_price" in runner._last_error
    runner._last_flow_debug = {"risk_gates": gates}
    summary = runner.get_compact_diag_summary()
    assert summary["status_messages"]["risk_gates"] == "blocked"


def test_risk_gates_invalid_stop_sets_message():
    runner = StrategyRunner(kite=None, telegram_controller=DummyTelegram())
    gates = runner._risk_gates_for({"entry_price": 100, "stop_loss": "oops"})
    assert gates["sl_valid"] is False
    assert all(isinstance(v, bool) for v in gates.values())
    assert runner._last_error and "invalid stop_loss" in runner._last_error


def test_health_check_clears_last_error():
    runner = StrategyRunner(kite=None, telegram_controller=DummyTelegram())
    runner._last_error = "some error"
    runner.health_check()
    assert runner._last_error is None
