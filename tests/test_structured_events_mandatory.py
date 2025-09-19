"""Ensure mandated structured events are emitted with diagnostic detail."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from src.data import source as source_mod
from src.data.source import LiveKiteSource
from src.logs import structured_log
from src.risk.limits import Exposure, LimitConfig, RiskEngine
from src.risk.position_sizing import PositionSizer
from src.signals.regime_detector import detect_market_regime
from src.strategies.scalping_strategy import EnhancedScalpingStrategy


class EventRecorder:
    """Helper capturing structured log events for assertions."""

    def __init__(self) -> None:
        self.events: List[Tuple[str, Dict[str, Any]]] = []

    def __call__(self, event: str, /, **fields: Any) -> None:
        self.events.append((event, fields))

    def last(self, name: str) -> Dict[str, Any]:
        for evt, payload in reversed(self.events):
            if evt == name:
                return payload
        raise AssertionError(f"event {name!r} not recorded: {self.events!r}")


def _make_trend_df(length: int = 160) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=length, freq="1min")
    prices = np.linspace(100.0, 110.0, length)
    data = {
        "open": prices,
        "high": prices + 0.5,
        "low": prices - 0.5,
        "close": prices,
        "volume": np.full(length, 1_000, dtype=float),
    }
    return pd.DataFrame(data, index=idx)


def test_live_kite_source_structured_events(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = EventRecorder()
    monkeypatch.setattr(structured_log, "event", recorder)
    monkeypatch.setattr(source_mod, "data_warmup_disable", lambda: False)
    monkeypatch.setattr(
        source_mod,
        "settings",
        SimpleNamespace(
            instruments=SimpleNamespace(nifty_lot_size=50),
            strategy=SimpleNamespace(max_tick_lag_s=0.1),
        ),
        raising=False,
    )

    ds = LiveKiteSource(kite=None)
    token = 12345
    now_ms = int(datetime.utcnow().timestamp() * 1000)
    ds._option_quote_cache[token] = {"ltp": 101.25, "ts_ms": now_ms, "mode": "full"}

    price, source, ts_ms = ds.prime_option_quote(token)
    assert price == pytest.approx(101.25)
    assert source == "ws_ltp"
    assert ts_ms == now_ms

    snapshot = recorder.last("market_data_snapshot")
    assert snapshot["token"] == token
    assert snapshot["price"] == pytest.approx(101.25)
    assert snapshot["source"] == "ws_ltp"

    ds._last_tick_ts = datetime.utcnow() - timedelta(seconds=2)
    ds._stale_tick_checks = 0
    ds._stale_tick_thresh = 1
    assert ds.tick_watchdog(max_age_s=0.1) is True

    stale_evt = recorder.last("stale_block")
    assert stale_evt["checks"] == 1
    assert stale_evt["max_age_s"] == pytest.approx(0.1)


def test_strategy_emits_score_and_block_events(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = EventRecorder()
    monkeypatch.setattr(structured_log, "event", recorder)
    df = _make_trend_df()

    strategy = EnhancedScalpingStrategy(
        min_signal_score=0.0,
        confidence_threshold=0.0,
        atr_period=14,
        atr_sl_multiplier=1.5,
        atr_tp_multiplier=3.0,
    )

    monkeypatch.setattr(
        "src.strategies.scalping_strategy.fetch_quote_with_depth",
        lambda *args, **kwargs: {
            "bid": 100.0,
            "ask": 100.5,
            "depth": {"buy": [{"quantity": 1000}], "sell": [{"quantity": 1000}]},
        },
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
        lambda *args, **kwargs: {"spread_pct": 0.05, "depth_ok": True, "mode": "SOFT"},
    )
    monkeypatch.setattr("src.strategies.scalping_strategy.cap_for_mid", lambda *args, **kwargs: 0.10)

    plan = strategy.generate_signal(df, current_price=float(df["close"].iloc[-1]))
    assert not plan.get("reason_block")

    score_evt = recorder.last("strategy_score")
    gates = score_evt["gates"]
    assert isinstance(gates, dict)
    assert set(gates) >= {"warmup", "regime", "atr_band", "microstructure", "score_threshold", "iv_adx"}

    def _reject(self, plan: Dict[str, Any], price: float) -> Tuple[str, Dict[str, Any]]:
        return "iv_extreme", {"iv": 95.0}

    monkeypatch.setattr(
        EnhancedScalpingStrategy,
        "_iv_adx_reject_reason",
        _reject,
        raising=False,
    )

    recorder.events.clear()
    blocked = strategy.generate_signal(df, current_price=float(df["close"].iloc[-1]))
    assert blocked.get("reason_block") == "iv_extreme"

    block_evt = recorder.last("score_block")
    assert block_evt["reason"] == "iv_extreme"
    assert block_evt["gates"]["iv_adx"]["ok"] is False


def test_regime_detector_emits_eval_event(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = EventRecorder()
    monkeypatch.setattr(structured_log, "event", recorder)
    df = _make_trend_df()

    res = detect_market_regime(df=df)
    assert res.regime in {"TREND", "RANGE", "NO_TRADE"}

    evt = recorder.last("regime_eval")
    assert evt["regime"] == res.regime
    assert set(evt["thresholds"]) == {
        "trend_adx",
        "trend_di_delta",
        "trend_bb_width",
        "range_adx",
        "range_di_delta",
        "range_bb_width",
    }


def test_risk_engine_emits_pretrade_and_block(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = EventRecorder()
    monkeypatch.setattr(structured_log, "event", recorder)

    engine = RiskEngine(LimitConfig())
    plan = {"regime": "TREND", "atr_pct": 0.5}
    exposure = Exposure(lots_by_symbol={})

    ok = engine.pre_trade_check(
        equity_rupees=1_000_000.0,
        plan=plan,
        exposure=exposure,
        intended_symbol="NIFTY",
        intended_lots=1,
        lot_size=50,
        entry_price=100.0,
        stop_loss_price=90.0,
        spot_price=18000.0,
        portfolio_delta_units=0.0,
    )
    assert ok[0] is True if isinstance(ok, tuple) else bool(ok)

    pretrade_evt = recorder.last("pretrade_limits")
    assert pretrade_evt["status"] == "ok"
    assert pretrade_evt["summary"]

    engine.state.cooloff_until = engine._now() + timedelta(minutes=5)
    blocked = engine.pre_trade_check(
        equity_rupees=1_000_000.0,
        plan=plan,
        exposure=exposure,
        intended_symbol="NIFTY",
        intended_lots=1,
        lot_size=50,
        entry_price=100.0,
        stop_loss_price=90.0,
        spot_price=18000.0,
        portfolio_delta_units=0.0,
    )
    assert blocked[0] is False if isinstance(blocked, tuple) else bool(blocked) is False

    block_evt = recorder.last("risk_block")
    assert block_evt["reason"] == "loss_cooloff"
    assert block_evt["intended_lots"] == 1
    assert block_evt["summary"]


def test_position_sizer_emits_sizing_events(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = EventRecorder()
    monkeypatch.setattr(structured_log, "event", recorder)

    sizer = PositionSizer(
        risk_per_trade=0.01,
        min_lots=1,
        max_lots=5,
        max_position_size_pct=0.25,
    )

    qty, lots, diag = sizer.size_from_signal(
        entry_price=100.0,
        stop_loss=95.0,
        lot_size=50,
        equity=1_000_000.0,
    )
    assert lots >= 1
    sizing_evt = recorder.last("sizing_calc")
    assert sizing_evt["lots"] == lots
    assert sizing_evt["diag"]["risk_rupees"] > 0

    recorder.events.clear()
    qty, lots, diag = sizer.size_from_signal(
        entry_price=0.0,
        stop_loss=95.0,
        lot_size=50,
        equity=1_000_000.0,
    )
    assert lots == 0
    fail_evt = recorder.last("sizing_fail")
    assert fail_evt["reason"] == "invalid_inputs"
    assert fail_evt["diag"]["block_reason"] == "invalid"

