from __future__ import annotations

import json
import logging
import os
import sys
import time

from .log_filters_compact import DedupFilter, RateLimitFilter


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "lvl": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            base.update(extra)
        return json.dumps(base, ensure_ascii=False)


class _LineFormatter(logging.Formatter):
    def format(self, record):
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict) and extra:
            return f"{record.levelname} {record.name} {extra}"
        return super().format(record)


def setup_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    json_mode = os.getenv("LOG_JSON", "true").lower() == "true"
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    formatter: logging.Formatter
    if json_mode:
        formatter = _JsonFormatter()
    else:
        formatter = _LineFormatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(formatter)
    # Global dedup + rate-limit (tunable via env)
    dedup_ttl = float(os.getenv("LOG_DEDUP_TTL_S", "5"))
    rate_n = int(os.getenv("LOG_RATE_N", "8"))
    rate_win = float(os.getenv("LOG_RATE_WIN_S", "60"))
    h.addFilter(DedupFilter(ttl_s=dedup_ttl))
    h.addFilter(RateLimitFilter(per_key=rate_n, window_s=rate_win))
    root.addHandler(h)
    log_file = os.getenv("LOG_FILE") or os.getenv("LOG_PATH")
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            fh.addFilter(DedupFilter(ttl_s=dedup_ttl))
            fh.addFilter(RateLimitFilter(per_key=rate_n, window_s=rate_win))
            root.addHandler(fh)
        except Exception:
            logging.getLogger(__name__).warning("log_file_setup_failed", exc_info=True)
    root.setLevel(getattr(logging, level, logging.INFO))
    for noisy in ("uvicorn", "gunicorn", "werkzeug", "waitress", "urllib3", "kiteconnect", "flask"):
        lg = logging.getLogger(noisy)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(logging.WARNING if noisy != "kiteconnect" else logging.INFO)
    logging.getLogger("structured").setLevel(getattr(logging, level, logging.INFO))
