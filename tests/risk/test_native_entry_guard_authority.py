from nifty_scalper_bot.risk.risk_manager import RiskManager


def test_final_entry_guard_is_owned_by_risk_manager_module() -> None:
    assert RiskManager.check_order.__module__ == "nifty_scalper_bot.risk.risk_manager"
    assert not hasattr(RiskManager, "_entry_guard_patch")
