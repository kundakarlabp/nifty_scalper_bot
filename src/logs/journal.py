"""Simple trade journal utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List
import csv

JOURNAL_PATH = Path("trade_journal.csv")
FIELDNAMES = ["ts_close", "pnl_R", "spread_pct_median", "ack_p95_ms"]


def append(record: Dict[str, object]) -> None:
    """Append a trade record to the CSV journal."""
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = JOURNAL_PATH.exists()
    with JOURNAL_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({k: record.get(k) for k in FIELDNAMES})


def read_trades_between(start_dt: datetime, end_dt: datetime) -> List[Dict[str, object]]:
    """Return trades closed between ``start_dt`` and ``end_dt`` (inclusive)."""
    if not JOURNAL_PATH.exists():
        return []
    rows: List[Dict[str, object]] = []
    with JOURNAL_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("ts_close")
            pnl = row.get("pnl_R")
            if not ts_str or pnl is None:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except Exception:
                continue
            if start_dt <= ts <= end_dt:
                rec: Dict[str, object] = {"ts_close": ts, "pnl_R": float(pnl)}
                sp = row.get("spread_pct_median")
                if sp not in (None, "", "nan"):
                    try:
                        rec["spread_pct_median"] = float(sp)  # type: ignore[arg-type]
                    except Exception:
                        pass
                rows.append(rec)
    return rows
