import smoke  # noqa: F401
from types import SimpleNamespace

import pandas as pd

import src.strategies.scalping_strategy as ss
from src.strategies.strategy_config import resolve_config_path
from tests.test_strategy import create_test_dataframe


def test_strategy_uses_prepared_atm_tokens(monkeypatch) -> None:
    """Strategy should not block when ATM tokens are available."""
    cfg = ss.StrategyConfig.load(resolve_config_path())
    cfg.raw.setdefault("strategy", {})["min_score"] = 0.0
    monkeypatch.setattr(ss.StrategyConfig, "load", classmethod(lambda cls, path: cfg))

    df = create_test_dataframe(trending_up=True)
    monkeypatch.setattr(ss, "latest_atr_value", lambda *a, **k: 0.06)
    monkeypatch.setattr(
        ss,
        "compute_atr",
        lambda *a, **k: pd.Series([0.06] * len(df), index=df.index),
    )

    seen: dict[str, object] = {}

    def fake_fetch(kite, ident):
        seen["ident"] = ident
        return {"bid": 100.0, "ask": 100.0}

    monkeypatch.setattr(ss, "fetch_quote_with_depth", fake_fetch)
    monkeypatch.setattr(ss, "_token_to_symbol_and_lot", lambda k, t: ("FOO", 50))
    monkeypatch.setattr(
        ss,
        "evaluate_micro",
        lambda *a, **k: {"spread_pct": 0.1, "depth_ok": True, "mode": "HARD"},
    )
    monkeypatch.setattr(ss, "cap_for_mid", lambda mid, cfg: 1.0)
    monkeypatch.setattr(
        ss,
        "resolve_weekly_atm",
        lambda price: {"ce": ("CE", 1), "pe": ("PE", 1)},
    )
    monkeypatch.setattr(
        ss,
        "get_instrument_tokens",
        lambda *a, **k: {"atm_tokens": {"ce": 1, "pe": 2}, "atm_strike": 17050},
    )
    monkeypatch.setattr(
        ss,
        "detect_market_regime",
        lambda **k: SimpleNamespace(regime="TREND"),
    )
    monkeypatch.setattr(
        ss,
        "compute_score",
        lambda df, regime, cfg: (10.0, None),
    )
    monkeypatch.setattr(
        ss,
        "select_strike",
        lambda price, score: ss.StrikeInfo(strike=17050),
    )

    strat = ss.EnhancedScalpingStrategy(
        min_signal_score=0.0,
        confidence_threshold=0.0,
        atr_period=14,
        atr_sl_multiplier=1.5,
        atr_tp_multiplier=3.0,
    )
    strat.data_source = SimpleNamespace(atm_tokens=(1, 2), current_atm_strike=17050)
    plan = strat.generate_signal(df, current_price=100.0)
    assert plan.get("reason_block") != "no_option_token"
    assert plan["micro"]["spread_pct"] == 0.1
    assert plan["micro"]["depth_ok"] is True
    assert seen["ident"] == plan["option_token"]


def test_strategy_rolls_on_strike_drift(monkeypatch) -> None:
    cfg = ss.StrategyConfig.load(resolve_config_path())
    cfg.raw.setdefault("strategy", {})["min_score"] = 0.0
    monkeypatch.setattr(ss.StrategyConfig, "load", classmethod(lambda cls, path: cfg))

    df = create_test_dataframe(trending_up=True)
    monkeypatch.setattr(ss, "latest_atr_value", lambda *a, **k: 0.06)
    monkeypatch.setattr(
        ss, "compute_atr", lambda *a, **k: pd.Series([0.06] * len(df), index=df.index)
    )
    monkeypatch.setattr(
        ss, "evaluate_micro", lambda *a, **k: {"spread_pct": 0.1, "depth_ok": True, "mode": "HARD"}
    )
    monkeypatch.setattr(ss, "cap_for_mid", lambda mid, cfg: 1.0)
    monkeypatch.setattr(
        ss,
        "resolve_weekly_atm",
        lambda price: {"ce": ("CE", 1), "pe": ("PE", 1)},
    )
    monkeypatch.setattr(
        ss,
        "get_instrument_tokens",
        lambda *a, **k: {"atm_tokens": {"ce": 1, "pe": 2}, "atm_strike": 17050},
    )
    monkeypatch.setattr(
        ss, "detect_market_regime", lambda **k: SimpleNamespace(regime="TREND")
    )
    monkeypatch.setattr(ss, "compute_score", lambda df, regime, cfg: (10.0, None))
    monkeypatch.setattr(
        ss, "select_strike", lambda price, score: ss.StrikeInfo(strike=17050)
    )
    monkeypatch.setattr(
        ss,
        "fetch_quote_with_depth",
        lambda kite, ident: {"bid": 100.0, "ask": 100.0},
    )
    monkeypatch.setattr(ss, "_token_to_symbol_and_lot", lambda k, t: ("FOO", 50))

    calls = {"count": 0}

    ds = SimpleNamespace(atm_tokens=(1, 2), current_atm_strike=0)

    def fake_ensure() -> None:
        calls["count"] += 1
        ds.atm_tokens = (10, 20)
        ds.current_atm_strike = 120

    ds.ensure_atm_tokens = fake_ensure  # type: ignore[attr-defined]

    strat = ss.EnhancedScalpingStrategy(
        min_signal_score=0.0,
        confidence_threshold=0.0,
        atr_period=14,
        atr_sl_multiplier=1.5,
        atr_tp_multiplier=3.0,
    )
    strat.data_source = ds

    plan1 = strat.generate_signal(df, current_price=100.0)
    assert calls["count"] == 1
    assert plan1["atm_strike"] == 120
    assert plan1["option_token"] == 10

    ds.current_atm_strike = 0
    ds.atm_tokens = (1, 2)
    _ = strat.generate_signal(df, current_price=100.0)
    assert calls["count"] == 1
