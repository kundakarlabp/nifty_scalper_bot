from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from types import SimpleNamespace

from src.config import settings
from src.notifications.telegram_controller import TelegramController
from src.strategies.runner import StrategyRunner
from src.utils import strike_selector


def _prep_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        settings,
        "telegram",
        SimpleNamespace(bot_token="t", chat_id=1, enabled=True, extra_admin_ids=[]),
    )


def _make_runner(monkeypatch) -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner.log = logging.getLogger("StrategyRunnerTest")
    runner.strategy_cfg = SimpleNamespace(
        raw={"strategy": {}},
        min_signal_score=4,
        min_signal_score_relaxed=3,
    )
    runner.strategy = SimpleNamespace(
        min_score_strict=4,
        min_score_relaxed=3,
        min_conf_strict=5.5,
        min_conf_relaxed=3.5,
        atr_period=14,
        base_sl_mult=1.3,
        base_tp_mult=2.2,
        trend_tp_boost=0.6,
        trend_sl_relax=0.2,
        range_tp_tighten=-0.4,
        range_sl_tighten=-0.2,
    )
    monkeypatch.setattr(settings.strategy, "min_signal_score", 4, raising=False)
    monkeypatch.setattr(settings.strategy, "confidence_threshold", 55.0, raising=False)
    monkeypatch.setattr(settings.strategy, "atr_period", 14, raising=False)
    monkeypatch.setattr(settings.strategy, "atr_sl_multiplier", 1.3, raising=False)
    monkeypatch.setattr(settings.strategy, "atr_tp_multiplier", 2.2, raising=False)

    def _set_extra(name: str, value: object) -> None:
        monkeypatch.setitem(settings.strategy.__dict__, name, value)

    _set_extra("min_signal_score_relaxed", 3)
    _set_extra("confidence_threshold_relaxed", 35.0)
    _set_extra("trend_tp_boost", 0.6)
    _set_extra("trend_sl_relax", 0.2)
    _set_extra("range_tp_tighten", -0.4)
    _set_extra("range_sl_tighten", -0.2)
    return runner


def test_why_reports_gates_and_micro(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    now = datetime.now()
    plan = {
        "last_bar_ts": (now - timedelta(seconds=60)).isoformat(),
        "bar_count": 25,
        "regime": "TREND",
        "atr_pct": 0.6,
        "atr_min": 0.5,
        "score": 10,
        "spread_pct": 0.1,
        "depth_ok": True,
        "quote_src": "test",
        "feature_ok": True,
        "entry": 100.0,
        "sl": 95.0,
        "tp1": 110.0,
        "tp2": 120.0,
        "opt_entry": 10.0,
        "opt_sl": 9.5,
        "opt_tp1": 11.0,
        "opt_tp2": 12.0,
        "opt_atr": 0.4,
        "opt_atr_pct": 4.0,
        "opt_lot_cost": 750.0,
        "reason_block": "cap_lt_one_lot",
    }
    status = {"within_window": True, "cooloff_until": "-", "daily_dd_hit": False}
    import src.diagnostics.registry as diag_registry

    monkeypatch.setattr(
        diag_registry,
        "run",
        lambda name: SimpleNamespace(ok=True, name=name, msg="ok", fix=None),
    )
    tc = TelegramController(
        status_provider=lambda: status,
        last_signal_provider=lambda: plan,
    )
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/why"}})
    msg = sent[0]
    assert "/why gates" in msg
    assert "window: PASS" in msg
    assert "micro:" in msg
    assert "Option → entry" in msg and "atr" in msg and "atr_pct" in msg
    assert "premium cap too small for 1 lot" in msg


def test_emergency_stop_runs_shutdown(monkeypatch, tmp_path) -> None:
    _prep_settings(monkeypatch)
    called: list[str] = []

    def cancel_all() -> None:
        called.append("cancel")

    runner = SimpleNamespace(shutdown=lambda: called.append("shutdown"))
    monkeypatch.setattr(
        StrategyRunner,
        "get_singleton",
        classmethod(lambda cls: runner),
    )
    kill_file = tmp_path / "kill"
    monkeypatch.setenv("KILL_SWITCH_FILE", str(kill_file))
    tc = TelegramController(status_provider=lambda: {}, cancel_all=cancel_all)
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/emergency_stop"}})
    assert called == ["cancel", "shutdown"]
    assert sent[0] == "Emergency stop executed."
    assert kill_file.exists()


def test_probe_returns_snapshot(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    runner = SimpleNamespace(
        debug_snapshot=lambda: {
            "bars": 10,
            "last_bar_ts": "ts",
            "lag_s": 5,
            "rr_threshold": 2,
            "risk_pct": 1.0,
        }
    )
    monkeypatch.setattr(
        StrategyRunner,
        "get_singleton",
        classmethod(lambda cls: runner),
    )
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/probe"}})
    msg = sent[0]
    assert "bars=10" in msg and "rr=2" in msg


def test_bars_returns_snapshot(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    runner = SimpleNamespace(
        debug_snapshot=lambda: {
            "bars": 7,
            "last_bar_ts": "ts",
            "lag_s": 4,
            "gates": {"a": 1},
        }
    )
    monkeypatch.setattr(
        StrategyRunner,
        "get_singleton",
        classmethod(lambda cls: runner),
    )
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/bars"}})
    msg = sent[0]
    assert "bars=7" in msg and "gates={'a': 1}" in msg


def test_quotes_formats_micro(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    plan = {
        "strike": "SYM",
        "micro": {
            "source": "src",
            "ltp": 100,
            "bid": 99,
            "ask": 101,
            "spread_pct": 0.5,
            "bid5": 5,
            "ask5": 6,
            "depth_ok": True,
        },
        "last_bar_ts": "2024-01-01T00:00:00",
        "last_bar_lag_s": 3,
    }
    tc = TelegramController(last_signal_provider=lambda: plan, status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/quotes"}})
    msg = sent[0]
    assert "📈 *Quotes*" in msg and "quote: SYM" in msg


def test_orders_invokes_provider(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    calls: list[str] = []

    def provider() -> list[dict[str, object]]:
        calls.append("orders")
        return [
            {
                "trade": "T1",
                "leg": "T1:ENTRY",
                "sym": "NIFTY",
                "state": "PENDING",
                "qty": 50,
                "filled": 0,
                "avg": 0.0,
                "status": "OPEN",
                "age_s": 12,
            }
        ]

    tc = TelegramController(status_provider=lambda: {}, open_trades_provider=provider)
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/orders"}})
    assert calls == ["orders"]
    msg = sent[0]
    assert "Trade T1 [OPEN]" in msg
    assert "age=12s" in msg


def test_cancel_trade_wires_helper(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    cancelled: list[str] = []

    def cancel_trade(trade_id: str) -> None:
        cancelled.append(trade_id)

    tc = TelegramController(status_provider=lambda: {}, cancel_trade=cancel_trade)
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/cancel T42"}})
    assert cancelled == ["T42"]
    assert sent[0] == "Cancel requested for T42."


def test_reconcile_runs_once(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    calls: list[int] = []

    def reconcile_once() -> int:
        calls.append(1)
        return 3

    tc = TelegramController(status_provider=lambda: {}, reconcile_once=reconcile_once)
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/reconcile"}})
    assert calls == [1]
    assert sent[0] == "Reconciled 3 legs."


def test_expiry_shows_dates(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    monkeypatch.setattr(
        "src.notifications.telegram_controller.next_tuesday_expiry",
        lambda: "2024-07-02",
    )
    monkeypatch.setattr(
        "src.notifications.telegram_controller.last_tuesday_of_month",
        lambda: "2024-07-30",
    )
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/expiry"}})
    assert sent[0] == "weekly=2024-07-02 | monthly=2024-07-30"


def test_plan_displays_option_numbers(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    plan = {
        "entry": 100.0,
        "sl": 95.0,
        "tp1": 110.0,
        "tp2": 120.0,
        "opt_entry": 10.0,
        "opt_sl": 9.5,
        "opt_tp1": 11.0,
        "opt_tp2": 12.0,
        "opt_lot_cost": 750.0,
    }
    runner = SimpleNamespace(last_plan=plan)
    monkeypatch.setattr(
        StrategyRunner,
        "get_singleton",
        classmethod(lambda cls: runner),
    )
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/plan"}})
    msg = sent[0]
    assert "Option → entry" in msg and "lot" in msg


def test_config_outputs_strategy_config(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    cfg = SimpleNamespace(
        name="strat",
        version=1,
        tz="UTC",
        atr_min=0.1,
        atr_max=1.0,
        depth_min_lots=1,
        min_oi=1000,
        delta_min=0.2,
        delta_max=0.8,
        tp1_R_min=1,
        tp1_R_max=2,
        tp2_R_trend=3,
        tp2_R_range=2,
        trail_atr_mult=1.5,
        time_stop_min=30,
        gamma_enabled=True,
        gamma_after=10,
        min_bars_required=30,
        raw={"strategy": {"min_score": 0.35}, "micro": {"mode": "SOFT", "max_spread_pct": 1.0}},
    )

    class Runner:
        def __init__(self, cfg: SimpleNamespace) -> None:
            self.strategy_cfg = cfg

        def tick(self) -> None:  # pragma: no cover - not used
            pass

    runner = Runner(cfg)
    tc = TelegramController(status_provider=lambda: {})
    tc._runner_tick = runner.tick
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/config"}})
    msg = sent[0]
    assert "Strategy Config" in msg and "name: `strat` v1" in msg


def test_state_outputs_metrics(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    runner = SimpleNamespace(
        _equity_cached_value=1000.0,
        eval_count=7,
        get_status_snapshot=lambda: {
            "trades_today": 2,
            "cooloff_until": "-",
            "consecutive_losses": 1,
        },
        get_last_signal_debug=lambda: {
            "entry": 100.0,
            "sl": 95.0,
            "tp1": 110.0,
            "tp2": 120.0,
            "opt_entry": 10.0,
            "opt_sl": 9.5,
            "opt_tp1": 11.0,
            "opt_tp2": 12.0,
            "opt_atr": 0.5,
            "opt_lot_cost": 750.0,
        },
        order_executor=SimpleNamespace(enable_trailing=True),
    )
    monkeypatch.setattr(
        StrategyRunner,
        "get_singleton",
        classmethod(lambda cls: runner),
    )
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/state"}})
    msg = sent[0]
    assert "eq=1000.0" in msg and "trades=2" in msg and "losses=1" in msg and "evals=7" in msg
    assert "Option → entry" in msg and "tp_basis=" in msg
    assert "trail=on" in msg


def test_atrmin_updates_config(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    cfg = SimpleNamespace(raw={}, atr_min=0.1)
    runner = SimpleNamespace(strategy_cfg=cfg)
    monkeypatch.setattr(
        StrategyRunner,
        "get_singleton",
        classmethod(lambda cls: runner),
    )
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/atrmin 0.2"}})
    assert cfg.raw["atr_min"] == 0.2 and cfg.atr_min == 0.2
    assert sent[0] == "atr_min set to 0.2"


def test_depthmin_updates_config(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    runner = SimpleNamespace(strategy_cfg=SimpleNamespace(raw={}))
    monkeypatch.setattr(
        StrategyRunner,
        "get_singleton",
        classmethod(lambda cls: runner),
    )
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/depthmin 5"}})
    assert runner.strategy_cfg.raw["micro"]["depth_min_lots"] == 5
    assert sent[0] == "micro depth_min_lots set to 5"


def test_micromode_updates_config(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    runner = SimpleNamespace(strategy_cfg=SimpleNamespace(raw={"micro": {}}))
    monkeypatch.setattr(
        StrategyRunner,
        "get_singleton",
        classmethod(lambda cls: runner),
    )
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/micromode HARD"}})
    assert runner.strategy_cfg.raw["micro"]["mode"] == "HARD"
    assert sent[0] == "micro mode = HARD"


def test_warmup_command(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    import pandas as pd

    cfg = SimpleNamespace(
        warmup_pad=2,
        warmup_bars_min=15,
        atr_period=14,
        ema_slow=21,
        regime_min_bars=20,
        features_min_bars=20,
        min_bars_required=20,
    )

    ds = SimpleNamespace(get_last_bars=lambda n: pd.DataFrame({"open": [1] * 22}))
    runner = SimpleNamespace(strategy_cfg=cfg, data_source=ds)
    monkeypatch.setattr(
        StrategyRunner,
        "get_singleton",
        classmethod(lambda cls: runner),
    )
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/warmup"}})
    msg = sent[0]
    assert "Warmup" in msg and "have=22" in msg


def test_fresh_command(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    now = datetime.utcnow()
    ds = SimpleNamespace(
        last_tick_ts=lambda: now - timedelta(seconds=10),
        last_bar_open_ts=lambda: now - timedelta(seconds=60),
        timeframe_seconds=60,
    )
    cfg = SimpleNamespace(max_tick_lag_s=8, max_bar_lag_s=75)
    runner = SimpleNamespace(strategy_cfg=cfg, data_source=ds)
    monkeypatch.setattr(
        StrategyRunner,
        "get_singleton",
        classmethod(lambda cls: runner),
    )
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/fresh"}})
    msg = sent[0]
    assert "Freshness" in msg and "tick_lag=" in msg


def test_micro_command(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    cfg = SimpleNamespace(depth_min_lots=1)
    runner = SimpleNamespace(kite=None, strategy_cfg=cfg)
    monkeypatch.setattr(
        StrategyRunner,
        "get_singleton",
        classmethod(lambda cls: runner),
    )
    monkeypatch.setattr(strike_selector, "_fetch_instruments_nfo", lambda kite: [])
    monkeypatch.setattr(strike_selector, "_get_spot_ltp", lambda kite, sym: 100.0)
    monkeypatch.setattr(
        strike_selector,
        "resolve_weekly_atm",
        lambda spot, inst: {"ce": ("TSYM", 50)},
    )
    import src.notifications.telegram_controller as tc_mod

    monkeypatch.setattr(
        tc_mod,
        "fetch_quote_with_depth",
        lambda kite, tsym: {"bid": 99, "ask": 101, "bid5_qty": 50, "ask5_qty": 60, "source": "t"},
    )
    monkeypatch.setattr(
        tc_mod,
        "micro_from_quote",
        lambda q, lot_size, depth_min_lots: (0.5, True),
    )

    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/micro"}})
    msg = sent[0]
    assert "spread%=0.5" in msg and "depth_ok=True" in msg


def test_logtail_returns_tail(monkeypatch, tmp_path) -> None:
    _prep_settings(monkeypatch)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "bot.log").write_text("a\nb\nc\n")
    monkeypatch.chdir(tmp_path)
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/logtail 2"}})
    msg = sent[0]
    assert msg.startswith("```") and "b\nc" in msg and msg.endswith("```")


def test_healthjson_outputs_json(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    import src.notifications.telegram_controller as tc_mod

    monkeypatch.setattr(
        tc_mod,
        "run_all",
        lambda: [SimpleNamespace(name="check", ok=True)],
    )
    tc = TelegramController(status_provider=lambda: {})
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)
    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/healthjson"}})
    data = json.loads(sent[0])
    assert data[0]["name"] == "check"


def test_strategy_mutator_commands_adjust_runtime(monkeypatch) -> None:
    _prep_settings(monkeypatch)
    runner = _make_runner(monkeypatch)
    tc = TelegramController(
        status_provider=lambda: {},
        set_min_score=runner.set_min_score,
        set_conf_threshold=runner.set_conf_threshold,
        set_atr_period=runner.set_atr_period,
        set_sl_mult=runner.set_sl_mult,
        set_tp_mult=runner.set_tp_mult,
        set_trend_boosts=runner.set_trend_boosts,
        set_range_tighten=runner.set_range_tighten,
    )
    sent: list[str] = []
    tc._send = lambda text, parse_mode=None: sent.append(text)

    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/minscore 6"}})
    assert sent[-1] == "Min signal score set to 6."
    assert runner.strategy.min_score_strict == 6
    assert runner.strategy.min_score_relaxed == 5
    assert settings.strategy.min_signal_score == 6
    assert getattr(settings.strategy, "min_signal_score_relaxed") == 5
    assert runner.strategy_cfg.min_signal_score == 6
    assert runner.strategy_cfg.min_signal_score_relaxed == 5
    assert runner.strategy_cfg.raw["strategy"]["min_signal_score"] == 6
    assert runner.strategy_cfg.raw["strategy"]["min_signal_score_relaxed"] == 5

    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/conf 60"}})
    assert sent[-1] == "Confidence threshold set to 60.00."
    assert settings.strategy.confidence_threshold == 60.0
    assert getattr(settings.strategy, "confidence_threshold_relaxed") == 40.0
    assert runner.strategy.min_conf_strict == 6.0
    assert runner.strategy.min_conf_relaxed == 4.0
    assert runner.strategy_cfg.raw["strategy"]["confidence_threshold"] == 60.0
    assert (
        runner.strategy_cfg.raw["strategy"]["confidence_threshold_relaxed"] == 40.0
    )

    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/atrp 21"}})
    assert sent[-1] == "ATR period set to 21."
    assert settings.strategy.atr_period == 21
    assert runner.strategy.atr_period == 21
    assert runner.strategy_cfg.raw["strategy"]["atr_period"] == 21

    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/slmult 1.4"}})
    assert sent[-1] == "SL ATR multiplier set to 1.40."
    assert settings.strategy.atr_sl_multiplier == 1.4
    assert runner.strategy.base_sl_mult == 1.4
    assert runner.strategy_cfg.raw["strategy"]["atr_sl_multiplier"] == 1.4

    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/tpmult 2.6"}})
    assert sent[-1] == "TP ATR multiplier set to 2.60."
    assert settings.strategy.atr_tp_multiplier == 2.6
    assert runner.strategy.base_tp_mult == 2.6
    assert runner.strategy_cfg.raw["strategy"]["atr_tp_multiplier"] == 2.6

    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/trend 0.7 0.3"}})
    assert sent[-1] == "Trend boosts: tp+0.7, sl+0.3"
    assert getattr(settings.strategy, "trend_tp_boost") == 0.7
    assert getattr(settings.strategy, "trend_sl_relax") == 0.3
    assert runner.strategy.trend_tp_boost == 0.7
    assert runner.strategy.trend_sl_relax == 0.3
    assert runner.strategy_cfg.raw["strategy"]["trend_tp_boost"] == 0.7
    assert runner.strategy_cfg.raw["strategy"]["trend_sl_relax"] == 0.3

    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/range -0.5 -0.3"}})
    assert sent[-1] == "Range tighten: tp-0.5, sl-0.3"
    assert getattr(settings.strategy, "range_tp_tighten") == -0.5
    assert getattr(settings.strategy, "range_sl_tighten") == -0.3
    assert runner.strategy.range_tp_tighten == -0.5
    assert runner.strategy.range_sl_tighten == -0.3
    assert runner.strategy_cfg.raw["strategy"]["range_tp_tighten"] == -0.5
    assert runner.strategy_cfg.raw["strategy"]["range_sl_tighten"] == -0.3
