"""Nightly replay orchestrator adding retries and anomaly alerts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
SCRIPTS_PATH = PROJECT_ROOT / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

import replay_daily_report as replay  # noqa: E402

from nifty_scalper_bot.utils.logging import get_logger, setup_logging  # noqa: E402

LOGGER = get_logger(__name__)

DEFAULT_MAX_RETRIES = int(os.getenv("REPLAY_MAX_RETRIES", "2"))
DEFAULT_RETRY_WAIT = float(os.getenv("REPLAY_RETRY_WAIT_SECONDS", "60"))
DEFAULT_DRAWDOWN_THRESHOLD = float(os.getenv("REPLAY_ALERT_DRAWDOWN", "0.05"))
DEFAULT_PNL_THRESHOLD = float(os.getenv("REPLAY_ALERT_PNL", "-500.0"))
DEFAULT_LATENCY_THRESHOLD = float(os.getenv("REPLAY_ALERT_LATENCY", "0.75"))


def parse_job_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, Sequence[str]]:
    """Parse job-level arguments and return remaining replay arguments.

    Args:
        argv: Optional CLI tokens overriding ``sys.argv``.

    Returns:
        tuple[argparse.Namespace, Sequence[str]]: Parsed job arguments and
            replay tokens.

    Raises:
        RuntimeError: If parsing fails unexpectedly.
    """

    LOGGER.debug(
        "Entered parse_job_args",
        extra={"event": "replay_report_parse_enter"},
    )
    parser = argparse.ArgumentParser(
        description="Run nightly replay with retries and anomaly detection.",
        add_help=True,
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Number of retry attempts when the replay fails (default: 2).",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=DEFAULT_RETRY_WAIT,
        help="Delay between retry attempts in seconds (default: 60).",
    )
    parser.add_argument(
        "--drawdown-threshold",
        type=float,
        default=DEFAULT_DRAWDOWN_THRESHOLD,
        help=(
            "Absolute max drawdown threshold triggering anomaly alerts "
            "(default: 0.05)."
        ),
    )
    parser.add_argument(
        "--pnl-threshold",
        type=float,
        default=DEFAULT_PNL_THRESHOLD,
        help="Total PnL floor triggering anomaly alerts (default: -500.0).",
    )
    parser.add_argument(
        "--latency-threshold",
        type=float,
        default=DEFAULT_LATENCY_THRESHOLD,
        help=(
            "Order latency p95 threshold triggering anomaly alerts " "(default: 0.75)."
        ),
    )
    parser.add_argument(
        "--alert-telegram-token",
        type=str,
        default=os.getenv("TELEGRAM_REPLAY_BOT_TOKEN"),
        help="Override Telegram token for anomaly and failure alerts.",
    )
    parser.add_argument(
        "--alert-telegram-chat-id",
        type=int,
        default=None,
        help="Override Telegram chat id for anomaly and failure alerts.",
    )
    try:
        args, remainder = parser.parse_known_args(
            list(argv) if argv is not None else None
        )
        remainder = list(remainder)
        if remainder and remainder[0] == "--":
            remainder = remainder[1:]
        return args, remainder
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in parse_job_args: %s",
            exc,
            extra={"event": "replay_report_parse_error"},
        )
        raise


def detect_anomalies(
    metrics: Mapping[str, Any],
    drift: Mapping[str, Any] | None,
    *,
    drawdown_threshold: float,
    pnl_threshold: float,
    latency_threshold: float,
) -> list[str]:
    """Return anomaly descriptions derived from replay metrics and drift.

    Args:
        metrics: Aggregated metrics dictionary from the replay run.
        drift: Diff payload returned by the replay baseline comparison.
        drawdown_threshold: Absolute drawdown tolerance before alerting.
        pnl_threshold: Minimum acceptable total PnL before alerting.
        latency_threshold: Maximum acceptable order latency p95 before alerting.

    Returns:
        list[str]: Human readable descriptions for each detected anomaly.

    Raises:
        RuntimeError: If anomaly detection fails unexpectedly.
    """

    LOGGER.debug(
        "Entered detect_anomalies",
        extra={"event": "replay_report_anomaly_enter"},
    )
    try:
        findings: list[str] = []
        max_drawdown = float(metrics.get("max_drawdown", 0.0))
        if abs(max_drawdown) > abs(drawdown_threshold):
            findings.append(
                (
                    "Drawdown "
                    f"{max_drawdown:.4f} breached threshold "
                    f"{drawdown_threshold:.4f}"
                )
            )
        total_pnl = float(metrics.get("total_pnl", 0.0))
        if total_pnl < pnl_threshold:
            findings.append(
                f"Total PnL {total_pnl:.2f} fell below threshold {pnl_threshold:.2f}"
            )
        order_latency = metrics.get("order_latency", {})
        if isinstance(order_latency, Mapping):
            p95_latency = order_latency.get("p95")
            if isinstance(p95_latency, (int, float)) and (
                p95_latency > latency_threshold
            ):
                findings.append(
                    (
                        "Order latency p95 "
                        f"{float(p95_latency):.4f}s exceeded "
                        f"{latency_threshold:.4f}s"
                    )
                )
        if isinstance(drift, Mapping) and drift.get("has_drift"):
            mismatch_total = sum(
                len(drift.get(key, []))
                for key in ("signals", "trades", "orders")
                if isinstance(drift.get(key), list)
            )
            findings.append(
                f"Baseline drift detected across {mismatch_total or 1} artefacts"
            )
        return findings
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in detect_anomalies: %s",
            exc,
            extra={"event": "replay_report_anomaly_error"},
        )
        raise


def run_with_retries(
    job_args: argparse.Namespace,
    replay_args: Sequence[str],
) -> int:
    """Execute the replay workflow with retries and anomaly alerts.

    Args:
        job_args: Parsed job-level configuration for retries and alerts.
        replay_args: Token sequence forwarded to the replay runner parser.

    Returns:
        int: ``0`` when parity holds and no anomalies detected, ``1`` otherwise.

    Raises:
        RuntimeError: If all retry attempts fail.
    """

    LOGGER.debug(
        "Entered run_with_retries",
        extra={"event": "replay_report_retry_enter"},
    )
    try:
        replay_namespace = replay.parse_args(replay_args)
        setup_logging(replay_namespace.log_level)
        LOGGER.info(
            "Condition met: replay_report_start",
            extra={"event": "replay_report_start"},
        )
        primary_token = replay_namespace.telegram_token or os.getenv(
            "TELEGRAM_REPLAY_BOT_TOKEN"
        )
        primary_chat = replay._resolve_chat_id(replay_namespace.telegram_chat_id)
        alert_token = job_args.alert_telegram_token or primary_token
        alert_chat = (
            replay._resolve_chat_id(job_args.alert_telegram_chat_id) or primary_chat
        )
        attempt = 0
        last_error: Exception | None = None
        while attempt <= job_args.max_retries:
            attempt += 1
            try:
                result, metrics, drift, message = replay.run_replay_job(
                    replay_namespace
                )
                replay.send_telegram_summary(primary_token, primary_chat, message)
                anomalies = detect_anomalies(
                    metrics,
                    drift if isinstance(drift, Mapping) else None,
                    drawdown_threshold=job_args.drawdown_threshold,
                    pnl_threshold=job_args.pnl_threshold,
                    latency_threshold=job_args.latency_threshold,
                )
                if anomalies:
                    alert_body = "\n".join(
                        [
                            "⚠️ Replay anomalies detected",
                            *anomalies,
                        ]
                    )
                    if alert_token and alert_chat:
                        replay.send_telegram_summary(
                            alert_token,
                            alert_chat,
                            alert_body,
                        )
                    LOGGER.error(
                        "Condition met: replay_report_anomalies",
                        extra={
                            "event": "replay_report_anomalies",
                            "anomalies": anomalies,
                        },
                    )
                    return 1
                LOGGER.info(
                    "Condition met: replay_report_complete",
                    extra={"event": "replay_report_complete"},
                )
                return 1 if drift.get("has_drift") else 0
            except Exception as exc:  # noqa: BLE001 - defensive wrapping
                last_error = exc
                LOGGER.error(
                    "Failure in run_with_retries attempt %s: %s",
                    attempt,
                    exc,
                    extra={
                        "event": "replay_report_retry_error",
                        "attempt": attempt,
                    },
                )
                if attempt > job_args.max_retries:
                    failure_message = (
                        "❌ Replay run failed after " f"{attempt} attempts: {exc}"
                    )
                    if alert_token and alert_chat:
                        try:
                            replay.send_telegram_summary(
                                alert_token,
                                alert_chat,
                                failure_message,
                            )
                        except (
                            Exception
                        ) as alert_exc:  # noqa: BLE001 - defensive wrapping
                            LOGGER.error(
                                "Failure sending replay failure alert: %s",
                                alert_exc,
                                extra={"event": "replay_report_alert_error"},
                            )
                    raise
                time.sleep(job_args.retry_wait_seconds)
        if last_error is not None:
            raise last_error
        return 0
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in run_with_retries: %s",
            exc,
            extra={"event": "replay_report_retry_failure"},
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point applying retries around the replay runner.

    Args:
        argv: Optional CLI tokens overriding ``sys.argv``.

    Returns:
        int: Exit code mirroring anomaly detection and parity drift.

    Raises:
        RuntimeError: If execution fails unexpectedly.
    """

    LOGGER.debug(
        "Entered replay_report.main",
        extra={"event": "replay_report_main_enter"},
    )
    try:
        job_args, replay_tokens = parse_job_args(argv)
        return run_with_retries(job_args, replay_tokens)
    except Exception as exc:  # noqa: BLE001 - defensive wrapping
        LOGGER.error(
            "Failure in replay_report.main: %s",
            exc,
            extra={"event": "replay_report_main_error"},
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
