"""Utility to generate markdown daily performance reports from backtest output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from nifty_scalper_bot.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger(__name__)


def load_summary(path: Path) -> Dict[str, Any]:
    """Load and validate a JSON summary emitted by the backtest engine.

    Args:
        path: Location of the JSON summary file.

    Returns:
        dict[str, Any]: Parsed JSON payload.

    Raises:
        RuntimeError: If the file is missing or malformed.
    """

    LOGGER.debug(
        "Entered load_summary",
        extra={"event": "daily_report_load_enter", "path": str(path)},
    )
    try:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Summary file not found: {resolved}")
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Summary payload must be a JSON object")
        return payload
    except Exception as exc:
        LOGGER.error(
            "Failure in load_summary: %s",
            exc,
            extra={"event": "daily_report_load_error"},
        )
        raise


def build_daily_report(summary: Dict[str, Any]) -> str:
    """Construct a markdown report from the aggregated summary payload.

    Args:
        summary: Parsed JSON summary emitted by the backtest engine.

    Returns:
        str: Markdown document capturing key performance metrics.

    Raises:
        RuntimeError: If required keys are missing in the payload.
    """

    LOGGER.debug(
        "Entered build_daily_report",
        extra={"event": "daily_report_build_enter"},
    )
    try:
        pnl = summary.get("pnl", {}) if isinstance(summary, dict) else {}
        daily_pnl = pnl.get("daily", {}) if isinstance(pnl, dict) else {}
        total_pnl = pnl.get("total") if isinstance(pnl, dict) else None
        drawdown = summary.get("drawdown", {}) if isinstance(summary, dict) else {}
        max_drawdown = drawdown.get("max") if isinstance(drawdown, dict) else None
        max_drawdown_value = float(max_drawdown) if max_drawdown is not None else 0.0
        order_latency = summary.get("latency", {}) if isinstance(summary, dict) else {}
        tick_latency = (
            summary.get("tick_latency", {}) if isinstance(summary, dict) else {}
        )
        generated_at = summary.get("generated_at", "N/A")
        if not daily_pnl:
            raise RuntimeError("Daily PnL data missing from summary")
        report_lines = [
            "# Daily Strategy Performance Report",
            "",
            f"Generated: {generated_at}",
            "",
            "## Profit and Loss",
        ]
        for day, pnl_value in sorted(daily_pnl.items()):
            report_lines.append(f"- **{day}**: {pnl_value:+.2f}")
        if total_pnl is not None:
            report_lines.append("")
            report_lines.append(f"**Total PnL**: {float(total_pnl):+.2f}")
        report_lines.extend(
            [
                "",
                "## Drawdown",
                f"**Maximum Drawdown**: {max_drawdown_value:.4f}",
                "",
                "## Order Latency (seconds)",
                _format_latency(order_latency),
                "",
                "## Tick Latency (seconds)",
                _format_latency(tick_latency),
            ]
        )
        return "\n".join(report_lines)
    except Exception as exc:
        LOGGER.error(
            "Failure in build_daily_report: %s",
            exc,
            extra={"event": "daily_report_build_error"},
        )
        raise


def _format_latency(latency: Dict[str, Any]) -> str:
    """Format latency statistics as a bullet list.

    Args:
        latency: Mapping containing latency percentiles.

    Returns:
        str: Bullet list friendly string summarising latency data.

    Raises:
        RuntimeError: If formatting fails unexpectedly.
    """

    LOGGER.debug(
        "Entered _format_latency",
        extra={"event": "daily_report_format_latency_enter"},
    )
    try:
        if not isinstance(latency, dict) or not latency:
            return "No latency data available."
        parts = []
        for key in ("p50", "p95", "max"):
            value = latency.get(key)
            if value is None:
                continue
            parts.append(f"- {key.upper()}: {float(value):.4f}")
        return "\n".join(parts) if parts else "No latency data available."
    except Exception as exc:
        LOGGER.error(
            "Failure in _format_latency: %s",
            exc,
            extra={"event": "daily_report_format_latency_error"},
        )
        raise


def write_report(path: Path, content: str) -> Path:
    """Persist the generated markdown to ``path`` and return the location.

    Args:
        path: Target file location for the markdown report.
        content: Markdown payload to be written.

    Returns:
        Path: Location of the persisted report.

    Raises:
        RuntimeError: If the report cannot be written.
    """

    LOGGER.debug(
        "Entered write_report",
        extra={"event": "daily_report_write_enter", "path": str(path)},
    )
    try:
        resolved = path.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        LOGGER.info(
            "Condition met: daily_report_written",
            extra={"event": "daily_report_written", "path": str(resolved)},
        )
        return resolved
    except Exception as exc:
        LOGGER.error(
            "Failure in write_report: %s",
            exc,
            extra={"event": "daily_report_write_error"},
        )
        raise


def _build_default_output(summary_path: Path) -> Path:
    """Derive a default output file path relative to the summary location.

    Args:
        summary_path: Path to the input JSON summary.

    Returns:
        Path: Suggested markdown output path.

    Raises:
        RuntimeError: If the summary path cannot be resolved.
    """

    LOGGER.debug(
        "Entered _build_default_output",
        extra={"event": "daily_report_default_output_enter"},
    )
    try:
        resolved = summary_path.expanduser().resolve()
        stem = resolved.stem.replace("PnL_", "daily_report_")
        return resolved.with_name(f"{stem}.md")
    except Exception as exc:
        LOGGER.error(
            "Failure in _build_default_output: %s",
            exc,
            extra={"event": "daily_report_default_output_error"},
        )
        raise


def main(argv: list[str] | None = None) -> int:
    """Entry point for the daily report generation CLI.

    Args:
        argv: Optional list of command line arguments.

    Returns:
        int: Zero on success, non-zero otherwise.

    Raises:
        RuntimeError: Propagates unexpected failures for visibility.
    """

    LOGGER.debug(
        "Entered main",
        extra={"event": "daily_report_main_enter"},
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        required=True,
        type=Path,
        help="Path to a PnL JSON summary emitted by the backtest engine.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional markdown output path. Defaults next to the summary file.",
    )
    args = parser.parse_args(argv)
    try:
        summary = load_summary(args.summary)
        report = build_daily_report(summary)
        output_path = args.output or _build_default_output(args.summary)
        write_report(output_path, report)
        return 0
    except Exception as exc:
        LOGGER.error(
            "Failure in main: %s",
            exc,
            extra={"event": "daily_report_main_error"},
        )
        return 1


__all__ = [
    "build_daily_report",
    "load_summary",
    "main",
    "write_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
