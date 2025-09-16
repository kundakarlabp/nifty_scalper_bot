from src.config import GuardsSettings, settings
from src.strategies.runner import StrategyRunner
from src.risk.guards import RiskConfig, RiskGuards


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - no behavior
        pass


def test_risk_guard_blocks_after_loss(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "guards",
        GuardsSettings(
            max_orders_per_min=30,
            daily_loss_cap=100,
            trading_start_hhmm="00:00",
            trading_end_hhmm="23:59",
            kill_env=True,
            kill_file="",
        ),
    )
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    guards = RiskGuards(RiskConfig())
    assert guards.ok_to_trade()
    runner._on_trade_closed(-150)
    guards.set_pnl_today(-runner._risk_state.realised_loss)
    assert not guards.ok_to_trade()
