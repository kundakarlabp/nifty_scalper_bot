"""Unified logging setup for the whole app."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Optional


# --- JSON & logfmt formatters -------------------------------------------------

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "lvl": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Allow structured payloads via extra={"extra": {...}}
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            base.update(extra)
        return json.dumps(base, ensure_ascii=False)


class _LogfmtFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # minimal logfmt: key=value space-separated
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
        msg_value = json.dumps(record.getMessage(), ensure_ascii=False)
        kv = [
            f"ts={ts}",
            f"lvl={record.levelname}",
            f"logger={record.name}",
            f"msg={msg_value}",
        ]
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            for k, v in extra.items():
                kv.append(f"{k}={json.dumps(v, ensure_ascii=False)}")
        return " ".join(kv)


# --- Core setup ----------------------------------------------------------------

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json: bool = False,
) -> None:
    """
    Configure the ROOT logger with a single StreamHandler and optional rotating file.
    Removes any pre-existing handlers to prevent mixed formats and duplicates.
    """
    # Normalize level
    lvl = getattr(logging, str(level).upper(), logging.INFO)

    # Remove any existing handlers
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    # Choose formatter
    formatter = _JsonFormatter() if json else _LogfmtFormatter()

    # Stream handler to stdout
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(lvl)
    sh.setFormatter(formatter)
    root.addHandler(sh)
    root.setLevel(lvl)

    # Optional rotating file
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
            fh.setLevel(lvl)
            fh.setFormatter(formatter)
            root.addHandler(fh)
        except Exception:
            # Do not crash on file IO issues; stdout is enough
            pass

    # Force third-party loggers to propagate to root and not install their own handlers
    for noisy in ("uvicorn", "gunicorn", "werkzeug", "waitress", "urllib3", "kiteconnect", "flask"):
        lg = logging.getLogger(noisy)
        lg.handlers.clear()
        lg.propagate = True
        # keep them at WARNING except kiteconnect which is informative
        lg.setLevel(logging.WARNING if noisy != "kiteconnect" else logging.INFO)
