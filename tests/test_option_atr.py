from src.strategies.scalping_strategy import EnhancedScalpingStrategy


def test_option_atr_pct_from_ticks() -> None:
    strat = EnhancedScalpingStrategy()
    for price in range(100, 116):
        strat._update_option_atr(float(price))
    assert round(strat.option_atr_pct, 2) == round(1.0 / 115 * 100, 2)
