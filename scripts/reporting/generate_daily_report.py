"""Utility to generate markdown daily performance reports from backtest output."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from nifty_scalper_bot.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger(__name__)


def load_completed_trades(
    db_path: Path,
    *,
    report_date: str,
    timezone_name: str,
) -> list[dict[str, Any]]:
    """Load one local trading day's authoritative completed outcomes."""

    resolved = db_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Trade journal not found: {resolved}")
    local_zone = ZoneInfo(timezone_name)
    local_day = date.fromisoformat(report_date)
    start = datetime.combine(local_day, time.min, tzinfo=local_zone)
    end = start + timedelta(days=1)
    outcomes: dict[str, dict[str, Any]] = {}
    uri = f"{resolved.as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            """
            SELECT id, timestamp, meta_json
            FROM trade_events
            WHERE event_type = ?
              AND timestamp >= ?
              AND timestamp < ?
            ORDER BY timestamp, id
            """,
            ("BRACKET_CLOSED", start.timestamp(), end.timestamp()),
        )
        for event_id, timestamp, meta_json in rows:
            try:
                meta = json.loads(meta_json or "{}")
                if not isinstance(meta, Mapping):
                    continue
                outcome = meta.get("completed_trade")
                if not isinstance(outcome, Mapping):
                    continue
                completed = dict(outcome)
                completed["closed_timestamp"] = float(timestamp)
                bracket_id = str(completed.get("bracket_id") or "").strip()
                outcomes[bracket_id or f"event:{event_id}"] = completed
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
    return list(outcomes.values())


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _measured_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        trade
        for trade in trades
        if trade.get("ledger_complete") is True
        and _number(trade.get("net_pnl")) is not None
    ]


def _profit_factor(values: list[float]) -> float | None:
    positive = sum(value for value in values if value > 0)
    negative = abs(sum(value for value in values if value < 0))
    return round(positive / negative, 4) if negative > 0 else None


def _max_drawdown(trades: list[dict[str, Any]]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for trade in sorted(
        trades, key=lambda item: _number(item.get("closed_timestamp")) or 0.0
    ):
        equity += _number(trade.get("net_pnl")) or 0.0
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return round(drawdown, 2)


def _score_summary(
    measured: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for trade in measured:
        score = _number(trade.get("final_score"))
        if score is not None:
            grouped[math.floor(score)].append(trade)

    buckets = []
    for lower, trades in sorted(grouped.items()):
        net_values = [_number(trade.get("net_pnl")) or 0.0 for trade in trades]
        buckets.append(
            {
                "score_bucket": f"{lower:.1f}–<{lower + 1:.1f}",
                "measured_trades": len(trades),
                "wins": sum(value > 0 for value in net_values),
                "win_rate_pct": round(
                    sum(value > 0 for value in net_values) / len(trades) * 100.0,
                    1,
                ),
                "average_net_pnl": round(sum(net_values) / len(trades), 2),
                "profit_factor": _profit_factor(net_values),
                "max_drawdown": _max_drawdown(trades),
            }
        )
    trades_with_score = sum(len(trades) for trades in grouped.values())
    return buckets, {
        "measured_trades": len(measured),
        "trades_with_score": trades_with_score,
        "trades_without_score": len(measured) - trades_with_score,
    }


def _metric_summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    percentile_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "coverage": len(ordered),
        "average": round(sum(ordered) / len(ordered), 3),
        "p95": round(ordered[percentile_index], 3),
    }


def _execution_summary(measured: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "entry_slippage_bps",
        "entry_spread_points",
        "decision_to_entry_fill_seconds",
        "entry_submit_to_fill_seconds",
        "exit_slippage_bps",
        "exit_spread_points",
        "exit_trigger_to_fill_seconds",
        "exit_submit_to_fill_seconds",
    )
    values: dict[str, list[float]] = {field: [] for field in fields}
    total_slippage_costs: list[float] = []
    market_fallback_trades = 0
    rejected_exit_attempts = 0
    trades_with_quality = 0
    for trade in measured:
        quality = trade.get("execution_quality")
        if not isinstance(quality, Mapping):
            continue
        trades_with_quality += 1
        for field in fields:
            value = _number(quality.get(field))
            if value is not None:
                values[field].append(value)
        entry_cost = _number(quality.get("entry_slippage_cost"))
        exit_cost = _number(quality.get("exit_slippage_cost"))
        if entry_cost is not None and exit_cost is not None:
            total_slippage_costs.append(entry_cost + exit_cost)
        if quality.get("exit_market_fallback") is True:
            market_fallback_trades += 1
        rejected_exit_attempts += int(
            _number(quality.get("exit_rejected_attempts")) or 0
        )
    return {
        "measured_trades": len(measured),
        "trades_with_quality": trades_with_quality,
        **{
            field: _metric_summary(metric_values) if metric_values else None
            for field, metric_values in values.items()
        },
        "total_slippage_cost": (
            _metric_summary(total_slippage_costs) if total_slippage_costs else None
        ),
        "market_fallback_trades": market_fallback_trades,
        "rejected_exit_attempts": rejected_exit_attempts,
    }


def _exit_summary(measured: list[dict[str, Any]]) -> dict[str, Any]:
    excursions: list[tuple[float, float]] = []
    by_reason: dict[str, list[float]] = defaultdict(list)
    for trade in measured:
        gross = _number(trade.get("gross_pnl"))
        mfe = _number(trade.get("mfe_pnl"))
        if gross is not None and mfe is not None:
            excursions.append((gross, mfe))
        reason = str(trade.get("exit_reason") or "UNKNOWN").strip() or "UNKNOWN"
        by_reason[reason].append(_number(trade.get("net_pnl")) or 0.0)
    total_mfe = sum(mfe for _gross, mfe in excursions)
    return {
        "measured_trades": len(measured),
        "trades_with_excursions": len(excursions),
        "average_profit_surrendered": (
            round(
                sum(mfe - gross for gross, mfe in excursions) / len(excursions),
                2,
            )
            if excursions
            else None
        ),
        "mfe_capture_pct": (
            round(sum(gross for gross, _mfe in excursions) / total_mfe * 100.0, 2)
            if total_mfe > 0
            else None
        ),
        "exit_reasons": [
            {
                "exit_reason": reason,
                "trades": len(values),
                "net_pnl": round(sum(values), 2),
            }
            for reason, values in sorted(by_reason.items())
        ],
    }


def summarise_completed_trades(
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate completed outcomes without converting incomplete trades to losses."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        strategy = str(trade.get("strategy_name") or "UNKNOWN").strip() or "UNKNOWN"
        regime = str(trade.get("regime") or "UNKNOWN").strip() or "UNKNOWN"
        grouped[(strategy, regime)].append(trade)

    groups: list[dict[str, Any]] = []
    total_measured = 0
    total_gross = 0.0
    total_costs = 0.0
    total_net = 0.0
    for (strategy, regime), outcomes in sorted(grouped.items()):
        measured = [
            outcome
            for outcome in outcomes
            if outcome.get("ledger_complete") is True
            and _number(outcome.get("net_pnl")) is not None
        ]
        net_values = [_number(outcome.get("net_pnl")) or 0.0 for outcome in measured]
        gross_values = [
            _number(outcome.get("gross_pnl")) or 0.0 for outcome in measured
        ]
        cost_values = []
        for outcome in measured:
            costs = outcome.get("estimated_costs")
            cost_values.append(
                _number(costs.get("total")) or 0.0
                if isinstance(costs, Mapping)
                else 0.0
            )
        wins = [value for value in net_values if value > 0]
        losses = [value for value in net_values if value < 0]
        mfe_values = [
            value
            for outcome in measured
            if (value := _number(outcome.get("mfe_pnl"))) is not None
        ]
        mae_values = [
            value
            for outcome in measured
            if (value := _number(outcome.get("mae_pnl"))) is not None
        ]
        holding_values = [
            value
            for outcome in measured
            if (value := _number(outcome.get("holding_seconds"))) is not None
        ]
        measured_count = len(measured)
        positive_pnl = sum(wins)
        negative_pnl = abs(sum(losses))
        gross_pnl = round(sum(gross_values), 2)
        estimated_costs = round(sum(cost_values), 2)
        net_pnl = round(sum(net_values), 2)
        groups.append(
            {
                "strategy": strategy,
                "regime": regime,
                "setups": sorted(
                    {
                        str(
                            outcome.get("setup_name") or outcome.get("setup_type")
                        ).strip()
                        for outcome in outcomes
                        if str(
                            outcome.get("setup_name") or outcome.get("setup_type") or ""
                        ).strip()
                    }
                ),
                "closed_trades": len(outcomes),
                "measured_trades": measured_count,
                "wins": len(wins),
                "losses": len(losses),
                "win_rate_pct": (
                    round(len(wins) / measured_count * 100.0, 1)
                    if measured_count
                    else None
                ),
                "gross_pnl": gross_pnl,
                "estimated_costs": estimated_costs,
                "net_pnl": net_pnl,
                "average_net_pnl": (
                    round(net_pnl / measured_count, 2) if measured_count else None
                ),
                "profit_factor": (
                    round(positive_pnl / negative_pnl, 4) if negative_pnl > 0 else None
                ),
                "average_mfe_pnl": (
                    round(sum(mfe_values) / len(mfe_values), 2) if mfe_values else None
                ),
                "average_mae_pnl": (
                    round(sum(mae_values) / len(mae_values), 2) if mae_values else None
                ),
                "average_holding_seconds": (
                    round(sum(holding_values) / len(holding_values), 2)
                    if holding_values
                    else None
                ),
                "incomplete_trades": len(outcomes) - measured_count,
            }
        )
        total_measured += measured_count
        total_gross += gross_pnl
        total_costs += estimated_costs
        total_net += net_pnl

    measured = _measured_trades(trades)
    score_buckets, score_coverage = _score_summary(measured)
    return {
        "groups": groups,
        "totals": {
            "closed_trades": len(trades),
            "measured_trades": total_measured,
            "incomplete_trades": len(trades) - total_measured,
            "gross_pnl": round(total_gross, 2),
            "estimated_costs": round(total_costs, 2),
            "net_pnl": round(total_net, 2),
        },
        "score_buckets": score_buckets,
        "score_coverage": score_coverage,
        "execution_quality": _execution_summary(measured),
        "exit_quality": _exit_summary(measured),
    }


def _display(value: Any, *, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value}{suffix}"


def _append_observational_sections(
    lines: list[str],
    summary: Mapping[str, Any],
) -> None:
    score_coverage = summary.get("score_coverage", {})
    lines.extend(
        [
            "",
            "## Score Calibration (observational)",
            "",
            f"Score coverage: **{int(score_coverage.get('trades_with_score', 0))}/"
            f"{int(score_coverage.get('measured_trades', 0))}** measured trades",
            "",
            "| Score bucket | Trades | Win rate | Avg net P&L | Profit factor | "
            "Max drawdown |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    score_buckets = summary.get("score_buckets", [])
    if not score_buckets:
        lines.append("| N/A | 0 | N/A | N/A | N/A | N/A |")
    for bucket in score_buckets:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(bucket.get("score_bucket", "N/A")),
                    str(bucket.get("measured_trades", 0)),
                    _display(bucket.get("win_rate_pct"), suffix="%"),
                    _display(bucket.get("average_net_pnl")),
                    _display(bucket.get("profit_factor")),
                    _display(bucket.get("max_drawdown")),
                ]
            )
            + " |"
        )

    execution = summary.get("execution_quality", {})
    lines.extend(
        [
            "",
            "## Execution Quality (observational)",
            "",
            f"Execution-quality coverage: **"
            f"{int(execution.get('trades_with_quality', 0))}/"
            f"{int(execution.get('measured_trades', 0))}** measured trades",
            "",
            "| Metric | Coverage | Average | P95 |",
            "|---|---:|---:|---:|",
        ]
    )
    execution_metrics = (
        ("Entry slippage (bps)", "entry_slippage_bps"),
        ("Entry spread (points)", "entry_spread_points"),
        ("Decision → entry fill (s)", "decision_to_entry_fill_seconds"),
        ("Entry submit → fill (s)", "entry_submit_to_fill_seconds"),
        ("Exit slippage (bps)", "exit_slippage_bps"),
        ("Exit spread (points)", "exit_spread_points"),
        ("Exit trigger → fill (s)", "exit_trigger_to_fill_seconds"),
        ("Exit submit → fill (s)", "exit_submit_to_fill_seconds"),
        ("Total slippage cost", "total_slippage_cost"),
    )
    for label, key in execution_metrics:
        metric = execution.get(key)
        coverage = int(metric.get("coverage", 0)) if isinstance(metric, Mapping) else 0
        average = metric.get("average") if isinstance(metric, Mapping) else None
        p95 = metric.get("p95") if isinstance(metric, Mapping) else None
        lines.append(
            f"| {label} | {coverage} | {_display(average)} | {_display(p95)} |"
        )
    lines.extend(
        [
            "",
            f"Market-fallback exits: **"
            f"{int(execution.get('market_fallback_trades', 0))}**  ",
            f"Rejected exit attempts: **"
            f"{int(execution.get('rejected_exit_attempts', 0))}**",
        ]
    )

    exit_quality = summary.get("exit_quality", {})
    lines.extend(
        [
            "",
            "## Exit Quality (observational)",
            "",
            f"Excursion coverage: **"
            f"{int(exit_quality.get('trades_with_excursions', 0))}/"
            f"{int(exit_quality.get('measured_trades', 0))}** measured trades  ",
            f"Average profit surrendered: **"
            f"{_display(exit_quality.get('average_profit_surrendered'))}**  ",
            f"Aggregate MFE capture: **"
            f"{_display(exit_quality.get('mfe_capture_pct'), suffix='%')}**",
            "",
            "| Exit reason | Trades | Net P&L |",
            "|---|---:|---:|",
        ]
    )
    exit_reasons = exit_quality.get("exit_reasons", [])
    if not exit_reasons:
        lines.append("| N/A | 0 | 0.0 |")
    for reason in exit_reasons:
        safe_reason = str(reason.get("exit_reason", "UNKNOWN")).replace("|", "/")
        lines.append(
            f"| {safe_reason} | {int(reason.get('trades', 0))} | "
            f"{float(reason.get('net_pnl', 0.0)):+.2f} |"
        )


def build_trade_outcome_report(
    summary: Mapping[str, Any],
    *,
    report_date: str,
    timezone_name: str,
) -> str:
    """Build an observational strategy-by-regime Markdown report."""

    totals = summary.get("totals", {})
    groups = summary.get("groups", [])
    lines = [
        "# Daily Strategy-by-Regime Outcome Report",
        "",
        f"Trading date: **{report_date}** ({timezone_name})",
        "",
        f"Closed trades: **{int(totals.get('closed_trades', 0))}**  ",
        f"Measured outcomes: **{int(totals.get('measured_trades', 0))}**  ",
        f"Incomplete outcomes: **{int(totals.get('incomplete_trades', 0))}**  ",
        f"Gross P&L: **{float(totals.get('gross_pnl', 0.0)):+.2f}**  ",
        f"Estimated costs: **{float(totals.get('estimated_costs', 0.0)):.2f}**  ",
        f"Net P&L: **{float(totals.get('net_pnl', 0.0)):+.2f}**",
        "",
        "| Strategy | Regime | Setup | Closed | Measured | Win rate | Net P&L | "
        "Avg net | Profit factor | Avg MFE | Avg MAE | Avg hold (s) | Incomplete |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        setups = ", ".join(group.get("setups", [])) or "UNKNOWN"
        safe_text = [
            str(group.get("strategy", "UNKNOWN")).replace("|", "/"),
            str(group.get("regime", "UNKNOWN")).replace("|", "/"),
            setups.replace("|", "/"),
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    *safe_text,
                    str(group.get("closed_trades", 0)),
                    str(group.get("measured_trades", 0)),
                    _display(group.get("win_rate_pct"), suffix="%"),
                    f"{float(group.get('net_pnl', 0.0)):+.2f}",
                    _display(group.get("average_net_pnl")),
                    _display(group.get("profit_factor")),
                    _display(group.get("average_mfe_pnl")),
                    _display(group.get("average_mae_pnl")),
                    _display(group.get("average_holding_seconds")),
                    str(group.get("incomplete_trades", 0)),
                ]
            )
            + " |"
        )
    _append_observational_sections(lines, summary)
    lines.extend(
        [
            "",
            "This report is observational and does not change live strategy "
            "parameters.",
            "Incomplete outcomes are excluded from win rate and expectancy metrics.",
        ]
    )
    return "\n".join(lines)


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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--summary",
        type=Path,
        help="Path to a PnL JSON summary emitted by the backtest engine.",
    )
    source.add_argument(
        "--trades-db",
        type=Path,
        help="Path to the live trades.db journal containing BRACKET_CLOSED outcomes.",
    )
    parser.add_argument(
        "--date",
        help="Local trading date for --trades-db in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Kolkata",
        help="IANA timezone used to bound --date (default: Asia/Kolkata).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional markdown output path. Defaults next to the summary file.",
    )
    args = parser.parse_args(argv)
    try:
        if args.trades_db is not None:
            local_day = (
                args.date or datetime.now(ZoneInfo(args.timezone)).date().isoformat()
            )
            trades = load_completed_trades(
                args.trades_db,
                report_date=local_day,
                timezone_name=args.timezone,
            )
            summary = summarise_completed_trades(trades)
            report = build_trade_outcome_report(
                summary,
                report_date=local_day,
                timezone_name=args.timezone,
            )
            output_path = args.output or args.trades_db.with_name(
                f"daily_strategy_outcomes_{local_day}.md"
            )
        else:
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
    "build_trade_outcome_report",
    "load_completed_trades",
    "load_summary",
    "main",
    "summarise_completed_trades",
    "write_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
