"""Tests covering option token selection and quote priming paths."""

from types import SimpleNamespace
from typing import Any, Dict, Tuple

import pandas as pd
import pytest

from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - stub
        raise AssertionError(f"send_message unexpected: {msg}")


class TokenOverridePlan(dict):
    """Plan dict that can override the token returned via :meth:`get`."""

    token_override: int | None = None

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if key == "token" and self.token_override is not None:
            return self.token_override
        return super().get(key, default)


@pytest.fixture
def runner_base(monkeypatch: pytest.MonkeyPatch) -> Tuple[StrategyRunner, Dict[str, Any]]:
    """Return a runner wired to reach the option selection stage."""

    runner = StrategyRunner(telegram_controller=DummyTelegram())
    captured: Dict[str, Any] = {}
    now = pd.Timestamp("2024-01-01 09:45", tz="Asia/Kolkata").to_pydatetime()
    monkeypatch.setattr(runner, "_within_trading_window", lambda *a, **k: True)
    monkeypatch.setattr(runner, "_now_ist", lambda: now)
    monkeypatch.setattr(runner, "_ensure_day_state", lambda: None)
    monkeypatch.setattr(runner, "_refresh_equity_if_due", lambda: None)
    monkeypatch.setattr(runner, "_maybe_emit_minute_diag", lambda plan: None)
    monkeypatch.setattr(runner, "_prime_atm_quotes", lambda: (True, None, [111, 222]))
    monkeypatch.setattr(runner, "_active_equity", lambda: 100_000)
    monkeypatch.setattr(runner, "_record_plan", lambda plan: captured.update(plan))
    monkeypatch.setattr(runner, "_emit_diag", lambda plan, micro: None)
    monkeypatch.setattr(runner, "_lots_by_symbol", lambda: {})
    monkeypatch.setattr(runner, "_notional_rupees", lambda: 0)
    monkeypatch.setattr(runner, "_portfolio_delta_units", lambda: 0)
    monkeypatch.setattr(runner.strategy, "get_debug", lambda: {})
    monkeypatch.setattr(runner.strategy, "generate_signal", lambda df, current_tick=None: {})

    df = pd.DataFrame(
        {
            "open": [1.0] * 60,
            "high": [1.0] * 60,
            "low": [1.0] * 60,
            "close": [1.0] * 60,
            "volume": [0] * 60,
        },
        index=pd.date_range(now - pd.Timedelta(minutes=59), periods=60, freq="1min"),
    )
    monkeypatch.setattr(runner, "_fetch_spot_ohlc", lambda: df)

    monkeypatch.setattr(runner, "event_guard_enabled", False, raising=False)
    monkeypatch.setattr(runner, "event_cal", None, raising=False)

    runner.order_executor = SimpleNamespace(
        micro_ok=lambda **k: (True, {"spread_pct": 0.1, "depth_ok": True}),
        step_queue=lambda now: None,
        on_order_timeout_check=lambda: None,
        cb_orders=None,
        cb_modify=None,
    )
    runner.executor = runner.order_executor

    runner.data_source = SimpleNamespace(
        cb_hist=None,
        cb_quote=None,
        current_atm_strike=17000,
        atm_tokens=(111, 222),
    )

    runner.kite = SimpleNamespace(
        subscribe=lambda tokens: None,
        set_mode=lambda mode, tokens: None,
        quote=lambda tokens: {
            tokens[0]: {
                "depth": {
                    "buy": [{"price": 100.0, "quantity": 150}],
                    "sell": [{"price": 100.2, "quantity": 150}],
                },
                "last_price": 100.1,
            }
        },
    )

    runner.strategy_cfg = SimpleNamespace(
        raw={},
        delta_enable_score=999,
        min_atr_pct_nifty=0.02,
        min_atr_pct_banknifty=0.04,
    )
    monkeypatch.setattr(
        runner, "risk_engine", SimpleNamespace(pre_trade_check=lambda **k: (True, "", {}))
    )
    monkeypatch.setattr(runner, "risk", SimpleNamespace(day_realized_loss=0, consecutive_losses=0))

    monkeypatch.setattr(
        "src.strategies.runner.compute_score", lambda df, regime, cfg: (1.0, SimpleNamespace(parts=None, total=None))
    )
    monkeypatch.setattr("src.strategies.runner.atr_pct", lambda df, period=14: 0.03)
    monkeypatch.setattr(
        "src.strategies.runner.evaluate_micro",
        lambda q, lot_size, atr_pct, cfg: {
            "spread_pct": 0.1,
            "depth_ok": True,
            "mode": "HARD",
            "would_block": False,
        },
    )

    return runner, captured


def test_token_mismatch_blocks(monkeypatch: pytest.MonkeyPatch, runner_base) -> None:
    runner, captured = runner_base

    plan = TokenOverridePlan(
        {
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
    )
    plan.token_override = 999

    monkeypatch.setattr(runner.strategy, "generate_signal", lambda df, current_tick=None: plan)
    monkeypatch.setattr(
        runner.option_resolver,
        "resolve_atm",
        lambda *a, **k: {"token": 555, "expiry": "2024-01-01", "tradingsymbol": "OPTCE"},
    )
    monkeypatch.setattr(
        runner, "_prime_option_quote", lambda **k: (_ for _ in ()).throw(AssertionError("prime should not run"))
    )

    runner.process_tick({"token": 111})
    assert captured["reason_block"] == "token_mismatch"
    assert "token_mismatch" in captured["reasons"]


def test_no_quote_when_bid_ask_missing(monkeypatch: pytest.MonkeyPatch, runner_base) -> None:
    runner, captured = runner_base

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

    monkeypatch.setattr(runner.strategy, "generate_signal", lambda df, current_tick=None: plan)
    monkeypatch.setattr(
        runner.option_resolver,
        "resolve_atm",
        lambda *a, **k: {"token": 555, "expiry": "2024-01-01", "tradingsymbol": "OPTCE"},
    )
    monkeypatch.setattr(
        runner,
        "_prime_option_quote",
        lambda **k: (
            {
                "mid": 100.0,
                "ltp": 100.0,
                "bid": 0.0,
                "ask": 0.0,
                "source": "cache",
            },
            "ltp",
        ),
    )

    runner.process_tick({"token": 111})
    assert captured["reason_block"] == "no_quote"
    assert "no_quote" in captured["reasons"]
    assert captured["micro"]["reason"] == "no_quote"


def test_prime_option_uses_cached_tick(monkeypatch: pytest.MonkeyPatch, runner_base) -> None:
    runner, _ = runner_base

    token = 777
    cached = {
        "bid": 10.0,
        "ask": 10.2,
        "mid": 10.1,
        "ltp": 10.1,
        "source": "cache",
    }
    runner._option_l1_cache[token] = cached

    quote, mode = runner._prime_option_quote(token=token, option={"token": token})
    assert quote is not None
    assert quote["source"] == "cache"
    assert quote["bid"] == pytest.approx(10.0)
    assert quote["ask"] == pytest.approx(10.2)
    assert mode == "depth"

