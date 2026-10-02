from nifty_scalper_bot.core.strategy_manager import StrategyManager


def test_live_simulation_uses_live_quality_policy_without_real_orders(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE_SIMULATION")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    manager = object.__new__(StrategyManager)
    profile = manager.get_strategy_mode_profile()
    assert profile["mode"] == "LIVE"
    assert profile["min_trade_quality"] == 7.0
