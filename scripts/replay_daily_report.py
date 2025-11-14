"""Nightly replay runner detecting strategy drift and publishing summaries."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd
from telegram import Bot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from nifty_scalper_bot.backtesting.backtest_engine import (  # noqa: E402
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    StrategyProtocol,
)
from nifty_scalper_bot.notifications.safe_messenger import SafeMessenger  # noqa: E402
from nifty_scalper_bot.utils.logging import get_logger, setup_logging  # noqa: E402

LOGGER = get_logger(__name__)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backtest_reports" / "ci"
DEFAULT_DATA_PATH = PROJECT_ROOT / "backtests" / "data" / "sample_replay.csv"
DEFAULT_BASELINE_PATH = (
    PROJECT_ROOT / "backtests" / "baselines" / "sample_replay_snapshot.json"
)
DEFAULT_SNAPSHOT_PATH = DEFAULT_OUTPUT_DIR / "latest_snapshot.json"
DEFAULT_INDEX_COLUMN = "timestamp"


class ReplayMomentumStrategy:
    """Simple momentum-driven strategy for CI drift detection."""

    name = "replay_momentum"

    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        """Return +1/-1 momentum signals for ``market_data`` rows.

        Args:
            market_data: Historical dataframe indexed by timestamp.

        Returns:
            pd.Series: Integer signals aligned with ``market_data`` index.

        Raises:
            RuntimeError: If signal generation encounters invalid input.
        """

        LOGGER.debug(
            "Entered ReplayMomentumStrategy.generate_signals",
            extra={"event": "replay_strategy_generate_enter"},
        )
        try:
            momentum = market_data["close"].pct_change().fillna(0.0)
            signals = pd.Series(0, index=market_data.index)
            signals.loc[momentum > 0] = 1
            signals.loc[momentum < 0] = -1
            return signals.astype(int)
        except Exception as exc:  # noqa: BLE001 - defensive wrapping
            LOGGER.error(
                "Failure in ReplayMomentumStrategy.generate_signals: %s",
                exc,
                extra={"event": "replay_strategy_generate_error"},
            )
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the replay runner.

    Args:
        argv: Optional sequence of CLI tokens overriding ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed command line options.

    Raises:
        RuntimeError: If parsing fails unexpectedly.
    """

    LOGGER.debug(
        "Entered parse_args",
        extra={"event": "replay_parse_args_enter"},
    )
    parser = argparse.ArgumentParser(
        description="Run nightly replay backtests and compare against baselines.",
    )
    parser.add_argument(
        "--historical-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV file or directory containing historical tick data.",
    )
    parser.add_argument(
        "--index-column",
        type=str,
        default=DEFAULT_INDEX_COLUMN,
        help="Column name containing the timestamp index.",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Snapshot JSON baseline used for drift detection.",
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=DEFAULT_SNAPSHOT_PATH,
        help="Location where the latest snapshot will be written.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for backtest artefacts (reports, logs).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Optional dotted path 'module:ClassName' for custom strategy.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("REPLAY_LOG_LEVEL", "INFO"),
        help="Log level for the replay runner (default: INFO).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Overwrite the baseline snapshot with the newly generated one.",
    )
    parser.add_argument(
        "--telegram-token",
        type=str,
        default=os.getenv("TELEGRAM_REPLAY_BOT_TOKEN"),
        help="Override Telegram bot token (defaults to environment variable).",
    )
    parser.add_argument(
        "--telegram-chat-id",
        type=int,
        default=None,
        help="Override Telegram chat id (defaults to environment variable).",
    )
    try:
        return parser.parse_args(list(argv) if argv is not None else None)
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in parse_args: %s",
            exc,
            extra={"event": "replay_parse_args_error"},
        )
        raise


def collect_sources(path: Path) -> List[Path]:
    """Return ordered CSV sources discovered at ``path``.

    Args:
        path: File or directory pointing to historical data.

    Returns:
        list[Path]: Sorted list of CSV files ready for ingestion.

    Raises:
        RuntimeError: If the path cannot be resolved or no files are found.
    """

    LOGGER.debug(
        "Entered collect_sources",
        extra={"event": "replay_collect_sources_enter", "path": str(path)},
    )
    try:
        resolved = path.expanduser().resolve()
        if resolved.is_dir():
            files = sorted(p for p in resolved.glob("*.csv"))
        else:
            files = [resolved]
        if not files:
            raise FileNotFoundError(f"No CSV files discovered at {resolved}")
        return files
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in collect_sources: %s",
            exc,
            extra={"event": "replay_collect_sources_error"},
        )
        raise


def load_market_data(paths: Sequence[Path], index_column: str) -> pd.DataFrame:
    """Load and concatenate market data from ``paths``.

    Args:
        paths: Collection of CSV files ordered by timestamp.
        index_column: Column name to convert into the datetime index.

    Returns:
        pd.DataFrame: Combined dataframe ready for backtesting.

    Raises:
        RuntimeError: If parsing fails or required columns are missing.
    """

    LOGGER.debug(
        "Entered load_market_data",
        extra={"event": "replay_load_market_data_enter", "count": len(paths)},
    )
    try:
        frames: List[pd.DataFrame] = []
        for source in paths:
            frame = pd.read_csv(source)
            if index_column not in frame.columns:
                raise ValueError(f"Missing index column '{index_column}' in {source}")
            frame[index_column] = pd.to_datetime(
                frame[index_column], utc=True, errors="coerce"
            )
            if frame[index_column].isna().all():
                raise ValueError(f"Failed to parse timestamps for {source}")
            frame = frame.set_index(index_column).sort_index()
            frames.append(frame)
        combined = pd.concat(frames).sort_index()
        if combined.empty:
            raise ValueError("Combined market data is empty")
        return combined
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in load_market_data: %s",
            exc,
            extra={"event": "replay_load_market_data_error"},
        )
        raise


def load_strategy(strategy_path: str | None) -> StrategyProtocol:
    """Instantiate a strategy specified via ``strategy_path``.

    Args:
        strategy_path: Dotted path ``module:ClassName`` pointing to a strategy.

    Returns:
        StrategyProtocol: Strategy instance ready for backtesting.

    Raises:
        RuntimeError: If loading fails or the class does not implement the protocol.
    """

    LOGGER.debug(
        "Entered load_strategy",
        extra={"event": "replay_load_strategy_enter", "strategy": strategy_path},
    )
    try:
        if not strategy_path:
            return ReplayMomentumStrategy()
        module_name, _, class_name = strategy_path.partition(":")
        if not module_name or not class_name:
            raise ValueError("Strategy must be formatted as 'module:ClassName'")
        module = import_module(module_name)
        candidate = getattr(module, class_name)
        strategy = candidate()  # type: ignore[call-arg]
        if not hasattr(strategy, "generate_signals"):
            raise TypeError("Strategy is missing generate_signals implementation")
        return strategy
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in load_strategy: %s",
            exc,
            extra={"event": "replay_load_strategy_error"},
        )
        raise


def run_backtest(
    data: pd.DataFrame,
    strategy: StrategyProtocol,
    output_dir: Path,
) -> BacktestResult:
    """Execute the backtest for the supplied ``data`` and ``strategy``.

    Args:
        data: Historical market data.
        strategy: Strategy instance implementing ``generate_signals``.
        output_dir: Directory where reports should be persisted.

    Returns:
        BacktestResult: Resulting simulation artefacts and analytics.

    Raises:
        RuntimeError: If execution fails due to configuration or runtime issues.
    """

    LOGGER.debug(
        "Entered run_backtest",
        extra={"event": "replay_run_backtest_enter", "rows": len(data)},
    )
    try:
        config = BacktestConfig(
            output_directory=output_dir,
            generate_visualizations=False,
        )
        engine = BacktestEngine(data, strategy, config)
        result = engine.run()
        LOGGER.info(
            "Condition met: replay_backtest_complete",
            extra={"event": "replay_backtest_complete"},
        )
        return result
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in run_backtest: %s",
            exc,
            extra={"event": "replay_run_backtest_error"},
        )
        raise


def summarise_metrics(result: BacktestResult) -> Dict[str, Any]:
    """Return high level metrics extracted from ``result``.

    Args:
        result: Completed backtest result payload.

    Returns:
        dict[str, Any]: Metrics dictionary including PnL and latency data.

    Raises:
        RuntimeError: If metrics extraction fails.
    """

    LOGGER.debug(
        "Entered summarise_metrics",
        extra={"event": "replay_summarise_metrics_enter"},
    )
    try:
        metrics = result.metrics_summary()
        return metrics
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in summarise_metrics: %s",
            exc,
            extra={"event": "replay_summarise_metrics_error"},
        )
        raise


def load_snapshot(path: Path) -> Dict[str, Any]:
    """Load a baseline snapshot from ``path`` if it exists.

    Args:
        path: Location of the snapshot JSON file.

    Returns:
        dict[str, Any]: Parsed JSON payload, empty when file is missing.

    Raises:
        RuntimeError: If the snapshot exists but cannot be read or parsed.
    """

    LOGGER.debug(
        "Entered load_snapshot",
        extra={"event": "replay_load_snapshot_enter", "path": str(path)},
    )
    try:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            LOGGER.info(
                "Condition met: replay_baseline_missing",
                extra={"event": "replay_baseline_missing", "path": str(resolved)},
            )
            return {}
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Snapshot payload must be a JSON object")
        return payload
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in load_snapshot: %s",
            exc,
            extra={"event": "replay_load_snapshot_error"},
        )
        raise


def save_snapshot(path: Path, snapshot: Mapping[str, Any]) -> Path:
    """Persist ``snapshot`` to ``path`` and return the written location.

    Args:
        path: Target path where the snapshot should be stored.
        snapshot: Mapping produced by :meth:`BacktestResult.to_snapshot`.

    Returns:
        Path: Absolute path of the persisted snapshot.

    Raises:
        RuntimeError: If writing to disk fails.
    """

    LOGGER.debug(
        "Entered save_snapshot",
        extra={"event": "replay_save_snapshot_enter", "path": str(path)},
    )
    try:
        resolved = path.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        LOGGER.info(
            "Condition met: replay_snapshot_written",
            extra={"event": "replay_snapshot_written", "path": str(resolved)},
        )
        return resolved
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in save_snapshot: %s",
            exc,
            extra={"event": "replay_save_snapshot_error"},
        )
        raise


def format_summary_message(metrics: Mapping[str, Any], drift: Mapping[str, Any]) -> str:
    """Create a human-readable summary message for Telegram and logs.

    Args:
        metrics: Aggregated metrics dictionary from ``summarise_metrics``.
        drift: Drift summary returned by :meth:`BacktestResult.diff_against`.

    Returns:
        str: Summary message ready to be delivered to Telegram.

    Raises:
        RuntimeError: If formatting fails unexpectedly.
    """

    LOGGER.debug(
        "Entered format_summary_message",
        extra={"event": "replay_format_summary_enter"},
    )
    try:
        daily_pnl = metrics.get("daily_pnl", {}) if isinstance(metrics, Mapping) else {}
        lines = [
            "Nightly Replay Summary",
            f"Total PnL: {metrics.get('total_pnl', 0.0):+.2f}",
            f"Max Drawdown: {metrics.get('max_drawdown', 0.0):.4f}",
        ]
        order_latency = metrics.get("order_latency", {})
        if isinstance(order_latency, Mapping) and order_latency:
            lines.append(
                "Order Latency (p95): " f"{float(order_latency.get('p95', 0.0)):.4f} s",
            )
        tick_latency = metrics.get("tick_latency", {})
        if isinstance(tick_latency, Mapping) and tick_latency:
            lines.append(
                "Tick Latency (p95): " f"{float(tick_latency.get('p95', 0.0)):.4f} s",
            )
        error_events = metrics.get("error_events")
        if isinstance(error_events, (int, float)):
            lines.append(f"Errors: {int(error_events)}")
        if daily_pnl:
            lines.append("Daily PnL:")
            for day, pnl in sorted(daily_pnl.items()):
                lines.append(f"- {day}: {float(pnl):+.2f}")
        if drift.get("has_drift"):
            mismatches = sum(
                len(drift.get(key, []))
                for key in ("signals", "trades", "orders")
                if isinstance(drift.get(key), list)
            )
            lines.append(f"⚠️ Drift detected in {mismatches} elements.")
        else:
            lines.append("✅ Baseline parity maintained.")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in format_summary_message: %s",
            exc,
            extra={"event": "replay_format_summary_error"},
        )
        raise


def send_telegram_summary(token: str | None, chat_id: int | None, message: str) -> None:
    """Send ``message`` to Telegram when credentials are available.

    Args:
        token: Telegram bot token with send message privileges.
        chat_id: Target chat identifier for notifications.
        message: Message body to deliver.

    Returns:
        None.

    Raises:
        RuntimeError: If Telegram delivery fails unexpectedly.
    """

    LOGGER.debug(
        "Entered send_telegram_summary",
        extra={"event": "replay_send_telegram_enter"},
    )
    if not token or not chat_id:
        LOGGER.info(
            "Condition met: replay_telegram_skipped",
            extra={"event": "replay_telegram_skipped"},
        )
        return
    try:

        async def _send() -> None:
            bot = Bot(token=token)
            messenger = SafeMessenger(bot)
            await messenger.send_text(chat_id=chat_id, text=message)

        asyncio.run(_send())
        LOGGER.info(
            "Condition met: replay_telegram_sent",
            extra={"event": "replay_telegram_sent"},
        )
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in send_telegram_summary: %s",
            exc,
            extra={"event": "replay_send_telegram_error"},
        )
        raise


def _resolve_chat_id(value: int | None) -> int | None:
    """Resolve the Telegram chat id from CLI or environment variables.

    Args:
        value: Chat id provided through CLI flags.

    Returns:
        Optional[int]: Parsed chat id when available.

    Raises:
        RuntimeError: If the environment override is invalid.
    """

    LOGGER.debug(
        "Entered _resolve_chat_id",
        extra={"event": "replay_resolve_chat_enter"},
    )
    if value is not None:
        return value
    try:
        env_value = os.getenv("TELEGRAM_REPLAY_CHAT_ID")
        if not env_value:
            return None
        return int(env_value)
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in _resolve_chat_id: %s",
            exc,
            extra={"event": "replay_resolve_chat_error"},
        )
        raise


def run_replay_job(
    args: argparse.Namespace,
) -> tuple[BacktestResult, Dict[str, Any], Dict[str, Any], str]:
    """Execute the nightly replay workflow and return artefacts.

    Args:
        args: Parsed command line namespace containing replay options.

    Returns:
        tuple[BacktestResult, dict[str, Any], dict[str, Any], str]:
            Simulation result, aggregated metrics, drift diff, and summary text.

    Raises:
        RuntimeError: If any stage of the replay workflow fails.
    """

    LOGGER.debug(
        "Entered run_replay_job",
        extra={"event": "replay_run_job_enter"},
    )
    try:
        sources = collect_sources(args.historical_path)
        data = load_market_data(sources, args.index_column)
        strategy = load_strategy(args.strategy)
        result = run_backtest(data, strategy, args.output_dir)
        metrics = summarise_metrics(result)
        baseline = load_snapshot(args.baseline_path)
        drift = result.diff_against(baseline if baseline else None)
        snapshot = result.to_snapshot()
        snapshot_path = (
            args.baseline_path if args.update_baseline else args.snapshot_path
        )
        save_snapshot(snapshot_path, snapshot)
        error_events = int(metrics.get("error_events", 0) or 0)
        if isinstance(drift, Mapping):
            drift_counts = sum(
                len(drift.get(key, []))
                for key in ("signals", "trades", "orders")
                if isinstance(drift.get(key), list)
            )
            if drift.get("has_drift"):
                error_events = max(error_events, 0)
                error_events += drift_counts or 1
        metrics["error_events"] = int(error_events)
        message = format_summary_message(metrics, drift)
        return result, metrics, drift, message
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in run_replay_job: %s",
            exc,
            extra={"event": "replay_run_job_error"},
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point executing the nightly replay workflow.

    Args:
        argv: Optional sequence of CLI arguments overriding ``sys.argv``.

    Returns:
        int: ``0`` when baseline parity holds, ``1`` if drift is detected.

    Raises:
        RuntimeError: If execution fails before producing a result.
    """

    args = parse_args(argv)
    setup_logging(args.log_level)
    LOGGER.info(
        "Condition met: replay_main_start",
        extra={"event": "replay_main_start"},
    )
    try:
        _result, metrics, drift, message = run_replay_job(args)
        LOGGER.info(
            "Condition met: replay_metrics_logged",
            extra={
                "event": "replay_metrics_logged",
                "total_pnl": metrics.get("total_pnl", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
            },
        )
        token = args.telegram_token or os.getenv("TELEGRAM_REPLAY_BOT_TOKEN")
        chat_id = _resolve_chat_id(args.telegram_chat_id)
        send_telegram_summary(token, chat_id, message)
        LOGGER.info(
            "Condition met: replay_main_complete",
            extra={"event": "replay_main_complete"},
        )
        return 1 if drift.get("has_drift") else 0
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in main: %s",
            exc,
            extra={"event": "replay_main_error"},
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
