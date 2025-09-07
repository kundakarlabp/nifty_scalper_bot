from src.strategies.runner import StrategyRunner
from src.risk.guards import RiskConfig, RiskGuards


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - no behavior
        pass


def test_risk_guard_blocks_after_loss() -> None:
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    cfg = RiskConfig(daily_loss_cap=100, trading_start_hm="00:00", trading_end_hm="23:59")
    guards = RiskGuards(cfg)
    assert guards.ok_to_trade()
    runner._on_trade_closed(-150)
    guards.set_pnl_today(-runner._risk_state.realised_loss)
    assert not guards.ok_to_trade()
