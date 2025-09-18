import smoke  # noqa: F401
from types import SimpleNamespace

import pandas as pd

import src.strategies.scalping_strategy as ss
from src.signals.patches import resolve_atr_band
from src.strategies.strategy_config import resolve_config_path
from tests.test_strategy import create_test_dataframe

def test_strategy_allows_clamped_min_atr(monkeypatch):
    """Strategy should not block when ATR matches resolved minimum."""
    cfg = ss.StrategyConfig.load(resolve_config_path())
    cfg.raw.setdefault("thresholds", {})["min_atr_pct_nifty"] = 0.05
    cfg.raw.setdefault("gates", {})["atr_pct_min"] = 0.03
    cfg.atr_min = 0.03
    cfg.min_atr_pct_nifty = 0.05
    cfg.raw.setdefault("strategy", {})["min_score"] = 0.0
    monkeypatch.setattr(ss.StrategyConfig, "load", classmethod(lambda cls, path: cfg))

    df = create_test_dataframe(trending_up=True)
    atr_value = float(df["close"].iloc[-1]) * 0.05 / 100.0
    monkeypatch.setattr(ss, "latest_atr_value", lambda *a, **k: atr_value)
    monkeypatch.setattr(
        ss,
        "compute_atr",
        lambda *a, **k: pd.Series([atr_value] * len(df), index=df.index),
    )

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
    assert plan.get("atr_min") == 0.05
    assert plan.get("atr_band") == (0.05, plan.get("atr_max"))
    resolved_min, resolved_max = resolve_atr_band(cfg, symbol="NIFTY")
    assert (resolved_min, resolved_max) == plan["atr_band"]


def test_strategy_allows_unbounded_atr_max(monkeypatch):
    """Zero ``atr_max`` should disable the upper guard instead of blocking."""

    cfg = ss.StrategyConfig.load(resolve_config_path())
    cfg.atr_min = 0.20
    cfg.atr_max = 0.0
    cfg.min_atr_pct_nifty = 0.02
    cfg.raw.setdefault("thresholds", {})["min_atr_pct_nifty"] = 0.02
    cfg.raw.setdefault("gates", {})["atr_pct_min"] = 0.02
    cfg.raw.setdefault("gates", {})["atr_pct_max"] = 0.0
    cfg.raw.setdefault("strategy", {})["min_score"] = 0.0
    monkeypatch.setattr(ss.StrategyConfig, "load", classmethod(lambda cls, path: cfg))

    df = create_test_dataframe(trending_up=True)
    monkeypatch.setattr(ss, "latest_atr_value", lambda *a, **k: 1.0)
    monkeypatch.setattr(
        ss,
        "compute_atr",
        lambda *a, **k: pd.Series([1.0] * len(df), index=df.index),
    )

    monkeypatch.setattr(ss, "fetch_quote_with_depth", lambda *args, **kwargs: {"bid": 100.0, "ask": 100.0})
    monkeypatch.setattr(
        ss,
        "evaluate_micro",
        lambda *a, **k: {"spread_pct": 0.1, "depth_ok": True},
    )
    monkeypatch.setattr(
        ss,
        "resolve_weekly_atm",
        lambda price: {"ce": ("SYMCE", 1), "pe": ("SYMPE", 1)},
    )
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
    assert plan.get("atr_max") == 0.0
    assert resolve_atr_band(cfg, symbol="NIFTY")[1] == 0.0


def test_resolve_atr_band_prefers_threshold_over_gate():
    """Threshold minima should win over lower gates when both exist."""

    cfg = SimpleNamespace(
        raw={
            "thresholds": {"min_atr_pct_nifty": 0.08},
            "gates": {"atr_pct_min": 0.03, "atr_pct_max": 0.7},
        },
        atr_min=0.02,
        atr_max=0.6,
        min_atr_pct_nifty=0.08,
        min_atr_pct_banknifty=0.1,
    )

    assert resolve_atr_band(cfg, symbol="NIFTY") == (0.08, 0.7)


def test_resolve_atr_band_falls_back_to_gate_when_threshold_missing():
    """Resolver should use gate minima when no threshold is configured."""

    cfg = SimpleNamespace(
        raw={"gates": {"atr_pct_min": 0.07, "atr_pct_max": 0.5}},
        atr_min=0.05,
        atr_max=0.4,
    )

    assert resolve_atr_band(cfg, symbol="NIFTY") == (0.07, 0.5)
