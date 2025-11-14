"""Parity regression tests ensuring live and backtest replay equivalence."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from src.nifty_scalper_bot.backtesting.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
)

from nifty_scalper_bot.backtest.parity import SinglePipelineParity
from nifty_scalper_bot.execution.paper_fill_engine import PaperFillEngine


class _ParityStrategy:
    """Deterministic strategy emitting signals for parity validation."""

    name = "parity"

    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        momentum = market_data["close"].pct_change().fillna(0.0)
        signals = pd.Series(0, index=market_data.index)
        signals.loc[momentum > 0] = 1
        signals.loc[momentum < 0] = -1
        return signals


class _StubDataHub:
    """In-memory data hub returning static quotes for fills."""

    def __init__(self) -> None:
        self.quotes: dict[str, dict[str, float]] = {}

    def get_quote(self, symbol: str, allow_pull: bool = False) -> dict[str, float]:
        return self.quotes.setdefault(symbol, {"ltp": 100.0, "bid": 99.8, "ask": 100.2})


class _StubResolver:
    """Resolver returning a constant lot size."""

    def lot_size_for_symbol(self, symbol: str) -> int:
        return 50


class _ParityRunner:
    """Replay-aware runner tracking ticks and producing trade history."""

    def __init__(self, threshold: float = 101.0) -> None:
        self.ticks: list[tuple[str, float]] = []
        self._trades: list[dict[str, Any]] = []
        self._threshold = threshold

    def on_replay_tick(self, symbol: str, tick: dict[str, Any]) -> None:
        close_price = float(tick.get("close", 0.0))
        self.ticks.append((symbol, close_price))
        if symbol == "OPT" and close_price >= self._threshold:
            self._trades.append({"symbol": symbol, "close": close_price})

    def get_status(self) -> dict[str, Any]:
        return {"symbols": {"OPT": {"trade_history": list(self._trades)}}}


def _runner_factory() -> _ParityRunner:
    return _ParityRunner()


def _load_daily_report_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "reporting"
        / "generate_daily_report.py"
    )
    spec = importlib.util.spec_from_file_location("generate_daily_report", module_path)
    assert spec and spec.loader, "Failed to load daily report module"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_replay_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "replay_daily_report.py"
    )
    spec = importlib.util.spec_from_file_location("replay_daily_report", module_path)
    assert spec and spec.loader, "Failed to load replay daily report module"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paper_factory() -> PaperFillEngine:
    return PaperFillEngine(_StubDataHub(), _StubResolver())


def _backtest_frame() -> pd.DataFrame:
    index = pd.date_range(datetime(2024, 1, 1, 9, 15), periods=6, freq="5T")
    open_price = [100.0, 100.4, 101.2, 100.6, 101.4, 102.0]
    close_price = [100.2, 101.0, 101.4, 100.8, 101.8, 102.4]
    high_price = [p + 0.3 for p in close_price]
    low_price = [p - 0.4 for p in close_price]
    volume = [1200, 1250, 1300, 1280, 1350, 1400]
    return pd.DataFrame(
        {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
        },
        index=index,
    )


# Tests follow...


def test_backtest_pipeline_parity_generates_identical_trade_logs(
    tmp_path: Path,
) -> None:
    data = _backtest_frame()
    strategy = _ParityStrategy()
    config = BacktestConfig(output_directory=tmp_path, generate_visualizations=False)
    engine = BacktestEngine(data, strategy, config)
    parity = SinglePipelineParity(
        runner_factory=_runner_factory,
        paper_factory=_paper_factory,
        option_symbol="OPT",
    )

    result = engine.run(parity=parity)

    summary = result.analytics.get("pipeline_parity") if result.analytics else None
    assert summary is not None, "Expected pipeline parity analytics to be present"
    assert summary["diff"] == ""
    assert summary["live_json"] == summary["back_json"]
    assert summary["live_orders"] == summary["backtest_orders"]

    snapshot = result.to_snapshot()
    assert snapshot["signals"], "Signals snapshot should not be empty"
    diff = result.diff_against(snapshot)
    assert diff["has_drift"] is False
    metrics = result.metrics_summary()
    assert metrics["daily_pnl"], "Daily PnL should be populated"
    assert metrics["order_latency"] is not None
    assert metrics["error_events"] == 0

    assert result.json_path is not None and result.json_path.exists()
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert "pnl" in payload and payload["pnl"]["daily"], payload
    assert "latency" in payload and "tick_latency" in payload, payload
    assert "risk" in payload, payload
    assert "drawdown" in payload and "max" in payload["drawdown"], payload


def test_backtest_pipeline_parity_detects_mismatched_trade_logs(
    tmp_path: Path,
) -> None:
    data = _backtest_frame()
    strategy = _ParityStrategy()
    config = BacktestConfig(output_directory=tmp_path, generate_visualizations=False)
    engine = BacktestEngine(data, strategy, config)
    state = {"count": 0}

    def mismatch_factory() -> _ParityRunner:
        state["count"] += 1
        threshold = 101.0 if state["count"] == 1 else 105.0
        return _ParityRunner(threshold=threshold)

    parity = SinglePipelineParity(
        runner_factory=mismatch_factory,
        paper_factory=_paper_factory,
        option_symbol="OPT",
    )

    result = engine.run(parity=parity)

    summary = result.analytics.get("pipeline_parity") if result.analytics else None
    assert summary is not None, "Expected pipeline parity analytics to be present"
    assert summary["diff"], "Parity diff should not be empty for mismatched logs"
    metrics = result.metrics_summary()
    assert metrics["error_events"] >= 1


def test_backtest_result_snapshot_and_diff(tmp_path: Path) -> None:
    data = _backtest_frame()
    strategy = _ParityStrategy()
    config = BacktestConfig(output_directory=tmp_path, generate_visualizations=False)
    engine = BacktestEngine(data, strategy, config)
    result = engine.run()

    snapshot = result.to_snapshot()
    assert snapshot["trades"] or snapshot["orders"], snapshot

    drift = result.diff_against(snapshot)
    assert drift["has_drift"] is False

    mutated = data.copy()
    mutated.loc[mutated.index[-1], "close"] += 5.0
    mutated_engine = BacktestEngine(mutated, strategy, config)
    mutated_result = mutated_engine.run()
    mutated_drift = mutated_result.diff_against(snapshot)
    assert mutated_drift["has_drift"] is True
    assert (
        mutated_drift["signals"]
        or mutated_drift["trades"]
        or mutated_drift["orders"]
        or mutated_drift["performance_delta"]
    ), mutated_drift

    message_module = _load_replay_module()
    summary_message = message_module.format_summary_message(
        mutated_result.metrics_summary(), mutated_drift
    )
    assert "Nightly Replay Summary" in summary_message


def test_replay_daily_report_script_detects_drift(tmp_path: Path) -> None:
    module = _load_replay_module()
    data = _backtest_frame().reset_index()
    data = data.rename(columns={"index": "timestamp"})
    csv_path = tmp_path / "sample.csv"
    data.to_csv(csv_path, index=False)

    baseline_path = tmp_path / "baseline.json"
    snapshot_path = tmp_path / "snapshot.json"
    output_dir = tmp_path / "reports"

    exit_code = module.main(
        [
            "--historical-path",
            str(csv_path),
            "--baseline-path",
            str(baseline_path),
            "--snapshot-path",
            str(snapshot_path),
            "--output-dir",
            str(output_dir),
            "--update-baseline",
            "--log-level",
            "INFO",
        ]
    )
    assert exit_code == 0
    assert baseline_path.exists()

    second_run = module.main(
        [
            "--historical-path",
            str(csv_path),
            "--baseline-path",
            str(baseline_path),
            "--snapshot-path",
            str(snapshot_path),
            "--output-dir",
            str(output_dir),
            "--log-level",
            "INFO",
        ]
    )
    assert second_run == 0

    mutated = data.copy()
    mutated.loc[mutated.index[-1], "close"] += 3.0
    mutated.to_csv(csv_path, index=False)

    drift_run = module.main(
        [
            "--historical-path",
            str(csv_path),
            "--baseline-path",
            str(baseline_path),
            "--snapshot-path",
            str(snapshot_path),
            "--output-dir",
            str(output_dir),
            "--log-level",
            "INFO",
        ]
    )
    assert drift_run == 1


def test_generate_daily_report_builds_markdown(tmp_path: Path) -> None:
    summary_path = tmp_path / "PnL_20240101.json"
    summary_payload = {
        "pnl": {
            "daily": {"2024-01-01": 1250.75, "2024-01-02": -320.25},
            "total": 930.5,
        },
        "drawdown": {"max": -0.045},
        "latency": {"p50": 0.12, "p95": 0.31, "max": 0.52},
        "tick_latency": {"p50": 0.05, "p95": 0.09, "max": 0.15},
        "risk": {"max_position": 50},
        "generated_at": "2024-01-03T09:30:00Z",
    }
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

    module = _load_daily_report_module()
    payload = module.load_summary(summary_path)
    report = module.build_daily_report(payload)
    output_path = module.write_report(tmp_path / "report.md", report)

    content = output_path.read_text(encoding="utf-8")
    assert "Daily Strategy Performance Report" in content
    assert "2024-01-01" in content and "2024-01-02" in content
    assert "Maximum Drawdown" in content
