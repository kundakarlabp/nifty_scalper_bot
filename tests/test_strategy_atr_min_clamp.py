import smoke  # noqa: F401
from types import SimpleNamespace

import pandas as pd

import src.strategies.scalping_strategy as ss
from src.strategies.strategy_config import resolve_config_path
from tests.test_strategy import create_test_dataframe

def test_strategy_allows_clamped_min_atr(monkeypatch):
    """Strategy should not block when ATR matches resolved minimum."""
    monkeypatch.setattr(ss, "_resolve_min_atr_pct", lambda: 0.05)
    cfg = ss.StrategyConfig.load(resolve_config_path())
    cfg.atr_min = 0.30
    cfg.raw.setdefault("strategy", {})["min_score"] = 0.0
    monkeypatch.setattr(ss.StrategyConfig, "load", classmethod(lambda cls, path: cfg))

    df = create_test_dataframe(trending_up=True)
    monkeypatch.setattr(ss, "latest_atr_value", lambda *a, **k: 0.06)
    monkeypatch.setattr(ss, "compute_atr", lambda *a, **k: pd.Series([0.06] * len(df), index=df.index))

    monkeypatch.setattr(ss, "fetch_quote_with_depth", lambda *args, **kwargs: {"bid": 100.0, "ask": 100.0})
    monkeypatch.setattr(ss, "evaluate_micro", lambda *a, **k: {"spread_pct": 0.1, "depth_ok": True})
    monkeypatch.setattr(ss, "resolve_weekly_atm", lambda price: {"ce": ("SYMCE", 1), "pe": ("SYMPE", 1)})
    monkeypatch.setattr(ss, "cap_for_mid", lambda mid, cfg: 0.0)
    monkeypatch.setattr(ss, "select_strike", lambda price, score: ("SYM", 100))
    monkeypatch.setattr(ss, "detect_market_regime", lambda **k: SimpleNamespace(regime="TREND"))
    monkeypatch.setattr(ss, "compute_score", lambda df, regime, cfg: (1.0, None))

    strat = ss.EnhancedScalpingStrategy(
        min_signal_score=0.0,
        confidence_threshold=0.0,
        atr_period=14,
        atr_sl_multiplier=1.5,
        atr_tp_multiplier=3.0,
    )
    plan = strat.generate_signal(df, current_price=100.0)
    assert plan.get("reason_block") != "atr_out_of_band"
