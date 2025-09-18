import pandas as pd
from types import SimpleNamespace

from src.signals.patches import resolve_atr_band
from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - stub
        pass


def test_atr_pct_rounding_allows_min_threshold(monkeypatch):
    runner = StrategyRunner(telegram_controller=DummyTelegram())

    monkeypatch.setattr(runner, "_within_trading_window", lambda *a, **k: True)
    now = pd.Timestamp("2024-01-01 01:00").to_pydatetime()
    monkeypatch.setattr(runner, "_now_ist", lambda: now)
    runner.event_guard_enabled = False
    runner.event_cal = None
    monkeypatch.setattr(
        runner,
        "_fetch_spot_ohlc",
        lambda: pd.DataFrame(
            {
                "open": [1.0] * 60,
                "high": [1.0] * 60,
                "low": [1.0] * 60,
                "close": [1.0] * 60,
                "volume": [0] * 60,
            },
            index=pd.date_range(pd.Timestamp(now) - pd.Timedelta(minutes=59), periods=60, freq="1min"),
        ),
    )
    monkeypatch.setattr(runner, "_ensure_day_state", lambda: None)
    monkeypatch.setattr(runner, "_refresh_equity_if_due", lambda: None)
    monkeypatch.setattr(runner, "_maybe_emit_minute_diag", lambda plan: None)
    monkeypatch.setattr(runner.option_resolver, "resolve_atm", lambda *a, **k: {"token": 1, "expiry": "2024-01-01"})
    monkeypatch.setattr(runner, "_active_equity", lambda: 100000)
    exec_stub = SimpleNamespace(
        micro_ok=lambda **k: (True, {"spread_pct": 0.1, "depth_ok": True}),
        step_queue=lambda now: None,
        on_order_timeout_check=lambda: None,
        cb_orders=None,
        cb_modify=None,
    )
    runner.order_executor = exec_stub
    runner.executor = exec_stub
    runner.data_source = SimpleNamespace(
        cb_hist=None, cb_quote=None, current_atm_strike=17000
    )

    import src.strategies.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "evaluate_micro",
        lambda *a, **k: {"spread_pct": None, "depth_ok": True, "mode": "HARD", "would_block": False},
    )
    monkeypatch.setattr(runner_mod, "compute_score", lambda df, regime, cfg: (1.0, None))
    monkeypatch.setattr(runner_mod, "atr_pct", lambda df, period=14: 0.0196)
    monkeypatch.setattr(runner, "_record_plan", lambda plan: setattr(runner, "last_plan", plan))
    monkeypatch.setattr(runner, "_emit_diag", lambda plan, micro: None)
    monkeypatch.setattr(
        runner,
        "_risk_gates_for",
        lambda plan: {
            "equity_floor": True,
            "daily_drawdown": True,
            "loss_streak": True,
            "trades_per_day": True,
            "sl_valid": True,
        },
    )
    monkeypatch.setattr(runner.risk_engine, "pre_trade_check", lambda **k: (True, "", {}))
    monkeypatch.setattr(runner, "_lots_by_symbol", lambda: {})
    monkeypatch.setattr(runner, "_notional_rupees", lambda: 0)
    monkeypatch.setattr(runner, "_portfolio_delta_units", lambda: 0)
    monkeypatch.setattr(runner, "_calculate_quantity_diag", lambda **k: (75, {"rupee_risk_per_lot": 1, "lots_final": 1}))
    monkeypatch.setattr(runner.risk, "day_realized_loss", 0, raising=False)
    monkeypatch.setattr(runner.risk, "consecutive_losses", 0, raising=False)
    monkeypatch.setattr(runner.risk, "trades_today", 0, raising=False)
    runner.strategy_cfg = SimpleNamespace(
        raw={
            "thresholds": {"min_atr_pct_nifty": 0.02},
            "gates": {"atr_pct_min": 0.05, "atr_pct_max": 0.9},
        },
        delta_enable_score=999,
        min_atr_pct_nifty=0.02,
        min_atr_pct_banknifty=0.04,
        atr_min=0.05,
        atr_max=0.9,
    )

    def fake_signal(df, current_tick=None):
        return {
            "regime": "TREND",
            "rr": 1.5,
            "entry": 100.0,
            "sl": 99.0,
            "tp1": 101.0,
            "tp2": 102.0,
            "score": 1.0,
            "option_type": "CE",
            "strike": "OPT",
            "qty_lots": 1,
            "reasons": [],
        }

    monkeypatch.setattr(runner.strategy, "generate_signal", fake_signal)

    runner.process_tick({})
    flow = runner.get_last_flow_debug()
    assert flow["reason_block"] != "atr_out_of_band"
    reasons = runner.last_plan.get("reasons", [])
    assert not any(
        isinstance(reason, str) and reason.startswith("atr_out_of_band")
        for reason in reasons
    )
    assert runner.last_plan["atr_min"] == 0.02
    assert runner.last_plan["atr_band"] == (0.02, 0.9)
    assert resolve_atr_band(runner.strategy_cfg, symbol=runner.under_symbol) == (
        0.02,
        0.9,
    )
    assert runner.last_plan["option_token"] == 1
    assert runner.last_plan["atm_strike"] == 17000


def test_atr_pct_float_noise_does_not_block(monkeypatch):
    runner = StrategyRunner(telegram_controller=DummyTelegram())

    monkeypatch.setattr(runner, "_within_trading_window", lambda *a, **k: True)
    now = pd.Timestamp("2024-01-01 01:00").to_pydatetime()
    monkeypatch.setattr(runner, "_now_ist", lambda: now)
    runner.event_guard_enabled = False
    runner.event_cal = None
    monkeypatch.setattr(
        runner,
        "_fetch_spot_ohlc",
        lambda: pd.DataFrame(
            {
                "open": [1.0] * 60,
                "high": [1.0] * 60,
                "low": [1.0] * 60,
                "close": [1.0] * 60,
                "volume": [0] * 60,
            },
            index=pd.date_range(pd.Timestamp(now) - pd.Timedelta(minutes=59), periods=60, freq="1min"),
        ),
    )
    monkeypatch.setattr(runner, "_ensure_day_state", lambda: None)
    monkeypatch.setattr(runner, "_refresh_equity_if_due", lambda: None)
    monkeypatch.setattr(runner, "_maybe_emit_minute_diag", lambda plan: None)
    monkeypatch.setattr(
        runner.option_resolver,
        "resolve_atm",
        lambda *a, **k: {"token": 1, "expiry": "2024-01-01"},
    )
    monkeypatch.setattr(runner, "_active_equity", lambda: 100000)
    exec_stub = SimpleNamespace(
        micro_ok=lambda **k: (True, {"spread_pct": 0.1, "depth_ok": True}),
        step_queue=lambda now: None,
        on_order_timeout_check=lambda: None,
        cb_orders=None,
        cb_modify=None,
    )
    runner.order_executor = exec_stub
    runner.executor = exec_stub
    runner.data_source = SimpleNamespace(
        cb_hist=None, cb_quote=None, current_atm_strike=17000
    )

    import src.strategies.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "evaluate_micro",
        lambda *a, **k: {"spread_pct": None, "depth_ok": True, "mode": "HARD", "would_block": False},
    )
    monkeypatch.setattr(runner_mod, "compute_score", lambda df, regime, cfg: (1.0, None))
    monkeypatch.setattr(runner_mod, "atr_pct", lambda df, period=14: 0.0399999)
    monkeypatch.setattr(runner, "_record_plan", lambda plan: setattr(runner, "last_plan", plan))
    monkeypatch.setattr(runner, "_emit_diag", lambda plan, micro: None)
    monkeypatch.setattr(
        runner,
        "_risk_gates_for",
        lambda plan: {
            "equity_floor": True,
            "daily_drawdown": True,
            "loss_streak": True,
            "trades_per_day": True,
            "sl_valid": True,
        },
    )
    monkeypatch.setattr(runner.risk_engine, "pre_trade_check", lambda **k: (True, "", {}))
    monkeypatch.setattr(runner, "_lots_by_symbol", lambda: {})
    monkeypatch.setattr(runner, "_notional_rupees", lambda: 0)
    monkeypatch.setattr(runner, "_portfolio_delta_units", lambda: 0)
    monkeypatch.setattr(
        runner,
        "_calculate_quantity_diag",
        lambda **k: (75, {"rupee_risk_per_lot": 1, "lots_final": 1}),
    )
    monkeypatch.setattr(runner.risk, "day_realized_loss", 0, raising=False)
    monkeypatch.setattr(runner.risk, "consecutive_losses", 0, raising=False)
    monkeypatch.setattr(runner.risk, "trades_today", 0, raising=False)
    runner.strategy_cfg = SimpleNamespace(
        raw={
            "thresholds": {"min_atr_pct_nifty": 0.04},
            "gates": {"atr_pct_min": 0.02, "atr_pct_max": 0.9},
        },
        delta_enable_score=999,
        min_atr_pct_nifty=0.04,
        min_atr_pct_banknifty=0.05,
        atr_min=0.02,
        atr_max=0.9,
    )

    def fake_signal(df, current_tick=None):
        return {
            "regime": "TREND",
            "rr": 1.5,
            "entry": 100.0,
            "sl": 99.0,
            "tp1": 101.0,
            "tp2": 102.0,
            "score": 1.0,
            "option_type": "CE",
            "strike": "OPT",
            "qty_lots": 1,
            "reasons": [],
        }

    monkeypatch.setattr(runner.strategy, "generate_signal", fake_signal)

    runner.process_tick({})
    flow = runner.get_last_flow_debug()
    assert flow["reason_block"] != "atr_out_of_band"
    reasons = runner.last_plan.get("reasons", [])
    assert not any(
        isinstance(reason, str) and reason.startswith("atr_out_of_band")
        for reason in reasons
    )
    assert runner.last_plan["atr_pct"] == 0.04

def test_nearest_strike_consistency():
    """Both resolvers should pick the same ATM strike for odd spot prices."""
    from datetime import datetime
    from src.options.instruments_cache import InstrumentsCache
    from src.options.resolver import OptionResolver
    from src.utils.strike_selector import (
        _nearest_strike,
        resolve_weekly_atm,
    )

    now = datetime(2024, 1, 1)
    spot = 24977
    expected = _nearest_strike(spot, 50)

    cache = InstrumentsCache(instruments=[])
    resolver = OptionResolver(cache)
    opt = resolver.resolve_atm("NIFTY", spot, "CE", now)
    assert opt["strike"] == expected

    expiry = opt["expiry"]
    inst_dump = [
        {
            "segment": "NFO-OPT",
            "name": "NIFTY",
            "expiry": expiry,
            "strike": expected,
            "instrument_type": "CE",
            "tradingsymbol": f"NIFTYXX{expected}CE",
            "lot_size": 50,
        },
        {
            "segment": "NFO-OPT",
            "name": "NIFTY",
            "expiry": expiry,
            "strike": expected,
            "instrument_type": "PE",
            "tradingsymbol": f"NIFTYXX{expected}PE",
            "lot_size": 50,
        },
    ]
    weekly = resolve_weekly_atm(spot, inst_dump)
    assert weekly["ce"][0].endswith(f"{expected}CE")
    assert weekly["pe"][0].endswith(f"{expected}PE")
