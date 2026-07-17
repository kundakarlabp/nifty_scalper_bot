from __future__ import annotations

import importlib

import pandas as pd


def _row(timestamp: pd.Timestamp, close: float = 100.5) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "open": 100.0,
        "high": max(101.0, close),
        "low": 99.0,
        "close": close,
        "volume": 10.0,
    }


def _tick(timestamp: pd.Timestamp, price: float = 101.0) -> dict[str, object]:
    return {
        "symbol": "NFO:NIFTY26JULFUT",
        "timestamp": timestamp,
        "ltp": price,
        "volume": 1.0,
    }


def test_production_bootstrap_replay_cannot_reopen_finalized_minute() -> None:
    # This import is the production installation path. The test deliberately does
    # not call any hardening installer directly.
    importlib.import_module("nifty_scalper_bot.core.app")

    from nifty_scalper_bot.data.candle_engine import CandleEngine, IST

    minute = pd.Timestamp.now(tz=IST).floor("min") - pd.Timedelta(minutes=5)
    engine = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    engine.replace_history(pd.DataFrame([_row(minute)]))

    for second in (5, 15, 30, 45):
        assert engine.on_tick(_tick(minute + pd.Timedelta(seconds=second))) is None

    assert engine.current_candle is None
    assert len(engine.get_df()) == 1
    diagnostics = engine.diagnostics()
    assert diagnostics["finalized_minute_tick_reject_total"] == 4
    assert diagnostics["state_consistent"] is True


def test_production_bootstrap_reconciles_history_over_live_partial() -> None:
    importlib.import_module("nifty_scalper_bot.core.app")

    from nifty_scalper_bot.data.candle_engine import CandleEngine, IST

    minute = pd.Timestamp.now(tz=IST).floor("min") - pd.Timedelta(minutes=5)
    engine = CandleEngine(symbol="NFO:NIFTY26JULFUT")
    engine.current_candle = {
        "timestamp": minute,
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.5,
        "volume": 4.0,
    }

    engine.replace_history(pd.DataFrame([_row(minute, close=100.5)]))

    assert engine.current_candle is None
    assert float(engine.get_df().iloc[-1]["close"]) == 100.5
    diagnostics = engine.diagnostics()
    assert diagnostics["history_current_reconcile_total"] == 1
    assert diagnostics["state_consistent"] is True
