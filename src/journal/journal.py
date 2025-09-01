"""Simple CSV trade journal."""

from __future__ import annotations

import csv
import os
from typing import Dict


def log(tr: Dict[str, object]) -> None:
    """Append a trade record to ``trade_log.csv``."""
    fn = "trade_log.csv"
    hdr = [
        "ts_entry",
        "ts_exit",
        "symbol",
        "side",
        "qty",
        "avg_entry",
        "avg_exit",
        "fees",
        "pnl_rupees",
        "pnl_R",
    ]
    new = not os.path.exists(fn)
    with open(fn, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if new:
            w.writeheader()
        w.writerow(tr)


__all__ = ["log"]
