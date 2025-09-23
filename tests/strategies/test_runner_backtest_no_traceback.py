import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from src.strategies.runner import StrategyRunner


def test_run_backtest_no_traceback(tmp_path, caplog):
    """Run backtest with an invalid CSV path and ensure no traceback is logged."""
    invalid_dir = tmp_path / "dir"
    invalid_dir.mkdir()
    runner = StrategyRunner(telegram_controller=SimpleNamespace())
    with caplog.at_level(logging.ERROR, logger="StrategyRunner"):
        result = runner.run_backtest(str(invalid_dir))
    assert result.startswith("Backtest error")
    assert "Traceback" not in caplog.text


def test_process_tick_blocks_when_option_resolution_fails(monkeypatch):
    """Option resolution errors should produce a blocked plan instead of raising."""

    class FakeDataSource:
        def __init__(self) -> None:
            tick = lambda *_args, **_kwargs: None
            self.cb_hist = SimpleNamespace(tick=tick)
            self.cb_quote = SimpleNamespace(tick=tick)

        def connect(self) -> None:  # pragma: no cover - interface requirement
            return None

        def have_min_bars(self, _min_bars: int) -> bool:
            return True

        def ensure_warmup(self, _min_bars: int) -> bool:
            return True

        def fetch_ohlc_df(self, *args, **kwargs):  # pragma: no cover - init hook
            return pd.DataFrame()

        def ensure_history(self, *args, **kwargs) -> None:  # pragma: no cover
            return None

    class FakeOrderExecutor:
        def __init__(self) -> None:
            tick = lambda *_args, **_kwargs: None
            self.cb_orders = SimpleNamespace(tick=tick)
            self.cb_modify = SimpleNamespace(tick=tick)
            self.state_store = None
            self.kite = None
            self.tick_size = 0.05

        def get_active_orders(self):
            return []

        def place_order(self, *args, **kwargs):  # pragma: no cover
            return None

        def step_queue(self, *args, **kwargs) -> None:  # pragma: no cover
            return None

        def on_order_timeout_check(self) -> None:  # pragma: no cover
            return None

        def get_positions_kite(self) -> dict:
            return {}

        def update_trailing_stop(self, *args, **kwargs) -> None:  # pragma: no cover
            return None

        def cancel_all_orders(self) -> None:  # pragma: no cover
            return None

        def close_all_positions_eod(self) -> None:  # pragma: no cover
            return None

        def get_or_create_fsm(self, *args, **kwargs):  # pragma: no cover
            return SimpleNamespace()

        def attach_leg_from_journal(self, *args, **kwargs) -> None:  # pragma: no cover
            return None

    class FakeJournal:
        def rehydrate_open_legs(self):
            return []

        def save_checkpoint(self, *_args, **_kwargs) -> None:
            return None

    class FakeStrategy:
        def generate_signal(self, *_args, **_kwargs):
            return {
                "has_signal": True,
                "option_type": "CE",
                "action": "BUY",
                "qty_lots": 1,
                "rr": 2.0,
                "token": 12345,
                "option_token": 12345,
            }

    fake_components = SimpleNamespace(
        strategy=FakeStrategy(),
        data_provider=FakeDataSource(),
        order_connector=FakeOrderExecutor(),
        names={"strategy": "fake", "data_provider": "fake"},
    )

    monkeypatch.setattr(
        "src.strategies.runner.init_default_registries",
        lambda *args, **kwargs: fake_components,
    )
    monkeypatch.setattr(
        "src.strategies.runner.Journal",
        SimpleNamespace(open=lambda *_args, **_kwargs: FakeJournal()),
    )
    monkeypatch.setattr(
        "src.strategies.runner.OrderManager",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "src.strategies.runner.OrderReconciler",
        lambda *args, **kwargs: SimpleNamespace(step=lambda *_a, **_k: None),
    )
    monkeypatch.setattr(
        "src.strategies.runner.required_bars",
        lambda *_args, **_kwargs: 1,
    )
    monkeypatch.setattr(
        "src.strategies.runner.warmup_check",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True, reasons=[]),
    )
    monkeypatch.setattr(
        "src.strategies.runner.compute_freshness",
        lambda **_kwargs: SimpleNamespace(ok=True, tick_lag_s=0.0, bar_lag_s=0.0),
    )
    monkeypatch.setattr(
        "src.strategies.runner.atr_pct",
        lambda *_args, **_kwargs: 0.5,
    )
    monkeypatch.setattr(
        "src.strategies.runner.check_atr",
        lambda *_args, **_kwargs: (True, None, 0.1, 2.0),
    )
    monkeypatch.setattr(
        "src.strategies.runner.compute_score",
        lambda *_args, **_kwargs: (1.0, None),
    )

    runner = StrategyRunner(telegram_controller=SimpleNamespace())
    df_index = pd.date_range("2023-01-01", periods=30, freq="T", tz="UTC")
    df = pd.DataFrame({"close": [100.0] * len(df_index)}, index=df_index)
    runner._fetch_spot_ohlc = Mock(return_value=df)
    runner._prime_atm_quotes = Mock(return_value=(True, None, []))
    runner._maybe_emit_minute_diag = Mock()
    runner._maybe_hot_reload_cfg = Mock()
    runner._maybe_reload_events = Mock()
    runner._ensure_day_state = Mock()
    runner._refresh_equity_if_due = Mock()
    runner._min_score_threshold = Mock(return_value=0.0)

    runner.option_resolver = SimpleNamespace(
        resolve_atm=Mock(side_effect=RuntimeError("boom"))
    )

    runner.process_tick({})

    assert runner.last_plan is not None
    assert runner.last_plan.get("reason_block") == "no_option_token"
    assert "no_option_token" in runner.last_plan.get("reasons", [])
    assert runner.last_plan.get("option_token") is None
