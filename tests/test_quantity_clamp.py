from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - stub
        pass


def test_calculate_quantity_clamps_zero_sl():
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    qty, diag = runner._calculate_quantity_diag(
        entry=100.0, stop=100.0, lot_size=75, equity=100_000.0
    )
    assert diag["sl_points"] == 0.5
    assert diag["rupee_risk_per_lot"] == 37.5
    assert qty > 0
