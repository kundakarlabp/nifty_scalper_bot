"""Synthetic nightly replay harness for scheduled health checks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Iterable


def _build_parser() -> argparse.ArgumentParser:
    """Return argument parser for the nightly replay utility."""

    parser = argparse.ArgumentParser(description="Run synthetic replay ticks")
    parser.add_argument(
        "--ticks",
        type=int,
        default=10_000,
        help="Number of synthetic ticks to simulate (default: 10000)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional path to write the JSON summary.",
    )
    parser.add_argument(
        "--dashboard",
        type=Path,
        default=Path("docs/deployment/nightly_replay_summary.json"),
        help="Dashboard output path for CI publishing.",
    )
    return parser


def _simulate_ticks(count: int) -> tuple[list[float], list[float]]:
    """Return synthetic PnL samples and latency readings for ``count`` ticks."""

    pnl: list[float] = []
    latency: list[float] = []
    equity = 0.0
    draw = 0.0
    for index in range(max(count, 0)):
        delta = ((index % 13) - 6) * 0.05
        equity += delta
        draw = min(draw, equity)
        pnl.append(delta)
        latency.append(0.5 + (index % 7) * 0.01)
    return pnl, latency


def _compute_win_rate(pnl: Iterable[float]) -> float:
    """Return win rate from iterable of trade deltas."""

    trades = list(pnl)
    if not trades:
        return 0.0
    wins = sum(1 for value in trades if value > 0.0)
    return round(100.0 * wins / len(trades), 2)


def _compute_max_drawdown(pnl: Iterable[float]) -> float:
    """Return cumulative max drawdown from the provided pnl sequence."""

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        drawdown = peak - equity
        max_dd = max(max_dd, drawdown)
    return round(max_dd, 2)


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    """Write JSON ``payload`` to ``path`` creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    """Execute the nightly replay routine returning shell exit status."""

    parser = _build_parser()
    args = parser.parse_args()
    tick_count = max(int(args.ticks), 0)
    start = time.perf_counter()
    pnl, latency = _simulate_ticks(tick_count)
    duration = time.perf_counter() - start
    win_rate = _compute_win_rate(pnl)
    max_drawdown = _compute_max_drawdown(pnl)
    avg_latency = round(statistics.fmean(latency), 4) if latency else 0.0
    summary = {
        "ticks": tick_count,
        "win_rate_pct": win_rate,
        "max_drawdown": max_drawdown,
        "avg_latency_ms": avg_latency,
        "duration_seconds": round(duration, 3),
    }
    print(json.dumps(summary, indent=2))
    if args.summary is not None:
        _write_summary(Path(args.summary), summary)
    if args.dashboard is not None:
        _write_summary(Path(args.dashboard), summary)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
