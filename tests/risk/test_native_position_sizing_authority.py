from nifty_scalper_bot.risk.risk_manager import RiskManager


def test_position_sizing_is_owned_by_risk_manager_module() -> None:
    assert RiskManager.suggest_position_size.__module__ == (
        "nifty_scalper_bot.risk.risk_manager"
    )
