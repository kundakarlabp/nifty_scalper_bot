from src.risk.guards import RiskGuards, RiskConfig


def test_rate_limit_blocks_excess_orders() -> None:
    cfg = RiskConfig(max_orders_per_min=2, trading_start_hm="00:00", trading_end_hm="23:59")
    guards = RiskGuards(cfg)
    assert guards.ok_to_trade()
    assert guards.ok_to_trade()
    assert not guards.ok_to_trade()


def test_daily_loss_cap_blocks() -> None:
    cfg = RiskConfig(daily_loss_cap=10, trading_start_hm="00:00", trading_end_hm="23:59")
    guards = RiskGuards(cfg)
    guards.set_pnl_today(-20)
    assert not guards.ok_to_trade()


def test_kill_switch_env(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_TRADING", "false")
    cfg = RiskConfig(trading_start_hm="00:00", trading_end_hm="23:59")
    guards = RiskGuards(cfg)
    assert not guards.ok_to_trade()
