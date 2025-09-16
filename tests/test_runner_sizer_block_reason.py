import pandas as pd
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - stub
        pass


def _setup_runner(
    monkeypatch,
    qty_diag: Tuple[int, Dict[str, Any]],
    plan_extra: Optional[Dict[str, Any]] = None,
) -> StrategyRunner:
    """Return a runner patched to reach sizing stage."""

    runner = StrategyRunner(telegram_controller=DummyTelegram())

    # Basic patches to reach sizing stage
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
            index=pd.date_range(
                pd.Timestamp(now) - pd.Timedelta(minutes=59), periods=60, freq="1min"
            ),
        ),
    )
    monkeypatch.setattr(runner, "_ensure_day_state", lambda: None)
    monkeypatch.setattr(runner, "_refresh_equity_if_due", lambda: None)
    monkeypatch.setattr(runner, "_maybe_emit_minute_diag", lambda plan: None)
    monkeypatch.setattr(
        runner.option_resolver,
        "resolve_atm",
        lambda *a, **k: {"token": 1, "expiry": "2024-01-01", "tradingsymbol": "OPT"},
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
    runner.kite = SimpleNamespace(
        quote=lambda symbols: {
            symbols[0]: {
                "depth": {
                    "buy": [{"price": 1.0, "quantity": 1}],
                    "sell": [{"price": 1.1, "quantity": 1}],
                },
                "last_price": 1.05,
            }
        }
    )

    import src.strategies.runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "evaluate_micro",
        lambda *a, **k: {
            "spread_pct": 0.1,
            "depth_ok": True,
            "mode": "HARD",
            "would_block": False,
        },
    )
    monkeypatch.setattr(runner_mod, "compute_score", lambda df, regime, cfg: (1.0, None))
    monkeypatch.setattr(runner_mod, "atr_pct", lambda df, period=14: 0.03)
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
    monkeypatch.setattr(runner, "_calculate_quantity_diag", lambda **k: qty_diag)
    monkeypatch.setattr(runner.risk, "day_realized_loss", 0, raising=False)
    monkeypatch.setattr(runner.risk, "consecutive_losses", 0, raising=False)
    monkeypatch.setattr(runner.risk, "trades_today", 0, raising=False)
    runner.strategy_cfg = SimpleNamespace(
        raw={}, delta_enable_score=999, min_atr_pct_nifty=0.02, min_atr_pct_banknifty=0.04
    )

    def fake_signal(df, current_tick=None):
        plan = {
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
        if plan_extra:
            plan.update(plan_extra)
        return plan

    monkeypatch.setattr(runner.strategy, "generate_signal", fake_signal)

    return runner


def test_sizer_block_reason_propagates(monkeypatch):
    runner = _setup_runner(
        monkeypatch,
        (0, {"rupee_risk_per_lot": 1, "lots_final": 0, "block_reason": "equity_low"}),
    )
    runner.process_tick({})
    flow = runner.get_last_flow_debug()
    assert flow["reason_block"] == "equity_low"
    assert runner.last_plan["reason_block"] == "equity_low"
    assert "equity_low" in runner.last_plan["reasons"]


def test_qty_zero_uses_existing_reason(monkeypatch):
    runner = _setup_runner(
        monkeypatch,
        (0, {"rupee_risk_per_lot": 1, "lots_final": 0, "block_reason": None}),
        {"reason_block": "preexisting"},
    )
    runner.process_tick({})
    flow = runner.get_last_flow_debug()
    assert flow["reason_block"] == "preexisting"
    assert runner.last_plan["reason_block"] == "preexisting"


def test_cap_lt_one_lot_reason_added(monkeypatch):
    runner = _setup_runner(
        monkeypatch,
        (0, {"rupee_risk_per_lot": 1, "lots_final": 0, "block_reason": "cap_lt_one_lot"}),
    )
    runner.process_tick({})
    plan = runner.last_plan
    assert plan["reason_block"] == "cap_lt_one_lot"
    assert "cap_lt_one_lot" in plan["reasons"]

