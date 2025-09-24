from __future__ import annotations

"""Logging filters for suppressing noisy warm-up messages."""

import logging
import os
import re
import time
from collections.abc import Iterable


class DedupFilter(logging.Filter):
    """Suppress duplicate log records matching regex patterns within a window."""

    def __init__(self, patterns: Iterable[str], window_s: int = 600) -> None:
        super().__init__()
        self._res = [re.compile(p) for p in patterns]
        self._last: dict[str, float] = {}
        self._win = int(window_s)

    def filter(
        self, record: logging.LogRecord
    ) -> bool:  # pragma: no cover - integration
        msg = f"{record.name} - {record.getMessage()}"
        now = time.time()
        for rx in self._res:
            if rx.search(msg):
                key = rx.pattern
                ts = self._last.get(key)
                if ts is not None and now - ts < self._win:
                    return False
                self._last[key] = now
                break
        return True


def install_warmup_filters() -> None:
    """Install a deduplicating filter for noisy warm-up log messages.

    Controlled by environment variables:
    - ``LOG_SUPPRESS_WARMUP``: enable/disable (defaults to true on Railway)
    - ``LOG_DEDUP_WINDOW_S``: time window for de-duplication (seconds)
    """

    suppress = str(
        os.getenv(
            "LOG_SUPPRESS_WARMUP",
            (
                "true"
                if os.getenv("RAILWAY_PROJECT_ID") or os.getenv("RAILWAY_STATIC_URL")
                else "false"
            ),
        )
    ).lower() in {"1", "true", "yes"}
    if not suppress:
        return

    window_s = int(os.getenv("LOG_DEDUP_WINDOW_S", "600"))
    patterns = [
        r"historical_data short",
        r"Insufficient historical_data",
        r"OHLC fetch returned no data",
        r"LiveKiteSource\.fetch_ohlc: broker unavailable",
        r"yfinance .* throttled",
        r"yfinance .* returned no data",
        r"possibly delisted",
    ]
    flt = DedupFilter(patterns, window_s=window_s)

    root = logging.getLogger()
    root.addFilter(flt)
    for name in ("StrategyRunner", "src.data.source", "yfinance"):
        logging.getLogger(name).addFilter(flt)
