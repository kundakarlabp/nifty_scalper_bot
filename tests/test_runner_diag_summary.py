from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:
        pass


def test_diag_summary_handles_non_dict_risk_gates():
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    # Inject a non-dict risk_gates to ensure graceful handling
    runner._last_flow_debug = {"risk_gates": "not_a_dict"}
    summary = runner.get_compact_diag_summary()
    assert summary["status_messages"]["risk_gates"] == "no-eval"
