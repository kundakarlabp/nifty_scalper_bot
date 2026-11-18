"""Tests of private Breeze CSV causal signal-only replay."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.reporting.range_vwap_reclaim_backtest import (
    run_futures_signal_audit,
)


def test_futures_only_is_never_executable_profitability(tmp_path: Path) -> None:
    from datetime import datetime, timedelta
    from math import sin, pi
    from zoneinfo import ZoneInfo

    start_at = datetime(2026, 11, 18, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    history = []
    for i in range(40):
        close = 25250 + 7 * sin(i * pi * 2 / 13)
        history.append({
            "timestamp": (start_at + timedelta(minutes=i)).isoformat(),
            "open": close - .5, "high": close + 2, "low": close - 2,
            "close": close, "volume": 1000.0,
        })
    file = tmp_path / "minute.csv"
    with file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "start_ist", "available_at_ist", "open",
                "high", "low", "close", "volume",
            ],
        )
        writer.writeheader()
        for row in history:
            start = datetime.fromisoformat(row["timestamp"])
            writer.writerow(
                {
                    "start_ist": start.isoformat(),
                    "available_at_ist": (
                        start + timedelta(minutes=1, seconds=5)
                    ).isoformat(),
                    **{key: row[key] for key in ("open", "high", "low", "close", "volume")},
                }
            )
    report = run_futures_signal_audit(file)
    assert report["futures_bars"] == 40
    assert report["option_fills"] == 0
    assert report["net_pnl"] is None
    assert report["live_promotion_eligible"] is False


def test_cannot_use_unavailable_bar_before_its_end(tmp_path: Path) -> None:
    file = tmp_path / "bad.csv"
    file.write_text(
        "start_ist,available_at_ist,open,high,low,close,volume\n"
        "2026-11-18T10:00:00+05:30,2026-11-18T10:00:25+05:30,"
        "1,2,0.5,1.5,100\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bar_unavailable"):
        run_futures_signal_audit(file)
