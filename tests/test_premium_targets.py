import pytest
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from tests.test_strategy import create_test_dataframe


def test_generate_signal_adds_premium_targets(monkeypatch):
    strat = EnhancedScalpingStrategy()
    df = create_test_dataframe(trending_up=True)
    monkeypatch.setattr(
        "src.strategies.scalping_strategy.fetch_quote_with_depth",
        lambda *args, **kwargs: {"bid": 100.0, "ask": 102.0, "ltp": 101.0},
    )
    monkeypatch.setattr(
        "src.strategies.scalping_strategy.resolve_weekly_atm",
        lambda price: {"ce": ("TESTCE", 50), "pe": ("TESTPE", 50)},
    )
    monkeypatch.setattr(
        "src.strategies.scalping_strategy.select_strike",
        lambda price, score: type("SI", (), {"strike": int(round(price / 50.0) * 50)})(),
    )
    plan = strat.generate_signal(df, current_price=float(df["close"].iloc[-1]))
    assert plan["opt_entry"] == pytest.approx(101.0)
    delta = 0.5
    spot_entry = plan["entry"]
    for key in ("tp1", "tp2", "sl"):
        expected = plan["opt_entry"] + delta * (plan[key] - spot_entry)
        assert plan[f"opt_{key}"] == pytest.approx(expected)
