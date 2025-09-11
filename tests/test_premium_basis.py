import pytest
from tests.test_strategy import create_test_dataframe, strategy_config  # noqa: F401
from src.strategies.scalping_strategy import EnhancedScalpingStrategy


def test_premium_targets_are_computed(strategy_config, monkeypatch):
    strategy = EnhancedScalpingStrategy(
        min_signal_score=strategy_config.min_signal_score,
        confidence_threshold=strategy_config.confidence_threshold,
        atr_period=strategy_config.atr_period,
        atr_sl_multiplier=strategy_config.atr_sl_multiplier,
        atr_tp_multiplier=strategy_config.atr_tp_multiplier,
    )
    df = create_test_dataframe(trending_up=True)

    monkeypatch.setattr(
        "src.strategies.scalping_strategy.fetch_quote_with_depth",
        lambda *a, **k: {"mid": 5.0},
    )
    monkeypatch.setattr(
        "src.strategies.scalping_strategy.resolve_weekly_atm",
        lambda price: {"ce": ("TESTCE", 50), "pe": ("TESTPE", 50)},
    )
    monkeypatch.setattr(
        "src.strategies.scalping_strategy.select_strike",
        lambda price, score: type("SI", (), {"strike": int(round(price / 50.0) * 50)})(),
    )
    monkeypatch.setattr(
        "src.strategies.scalping_strategy.evaluate_micro",
        lambda *a, **k: {"spread_pct": 0.1, "depth_ok": True, "mode": "HARD"},
    )
    monkeypatch.setattr(
        "src.strategies.scalping_strategy.cap_for_mid", lambda mid, cfg: 1.0
    )

    plan = strategy.generate_signal(df, current_price=float(df["close"].iloc[-1]))

    assert plan["opt_entry"] == 5.0
    entry = float(plan["spot_entry"])

    def opt_target(spot_target: float) -> float:
        spot_move = abs(spot_target - entry)
        spot_dir = 1.0 if spot_target >= entry else -1.0
        parity = 1.0 if plan["option_type"] == "CE" else -1.0
        opt_dir = spot_dir * parity
        prem_move = 0.5 * spot_move
        if plan["action"] == "BUY":
            px = 5.0 + opt_dir * prem_move
        else:
            px = 5.0 - opt_dir * prem_move
        return round(px / 0.05) * 0.05

    assert plan["opt_sl"] == pytest.approx(opt_target(plan["spot_sl"]))
    assert plan["opt_tp1"] == pytest.approx(opt_target(plan["spot_tp1"]))
    assert plan["opt_tp2"] == pytest.approx(opt_target(plan["spot_tp2"]))
