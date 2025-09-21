from __future__ import annotations

import hashlib
import logging
import time


def _round(x, n=3):
    try:
        return round(float(x), n)
    except Exception:
        return x


class TruncateFormatter(logging.Formatter):
    def __init__(self, max_len: int = 160):
        super().__init__()
        self.max_len = max_len

    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        if len(s) <= self.max_len:
            return s
        return s[: self.max_len - 3] + "..."


class DedupFilter(logging.Filter):
    """Suppress identical messages within a TTL window."""

    def __init__(self, ttl_s: float = 10.0):
        super().__init__()
        self.ttl_s = ttl_s
        self.cache: dict[str, float] = {}  # hash -> expires_at

    def filter(self, record: logging.LogRecord) -> bool:
        key = f"{record.name}|{record.levelno}|{getattr(record, 'msg', '')}"
        h = hashlib.md5(key.encode()).hexdigest()
        now = time.time()
        exp = self.cache.get(h, 0)
        if exp > now:
            return False
        self.cache[h] = now + self.ttl_s
        return True


class RateLimitFilter(logging.Filter):
    """Allow at most N messages per key per window."""

    def __init__(self, per_key: int = 6, window_s: float = 60.0):
        super().__init__()
        self.per_key = per_key
        self.window_s = window_s
        self.buckets: dict[str, tuple[int, float]] = {}  # key -> [count, window_start]

    def filter(self, record: logging.LogRecord) -> bool:
        key = getattr(record, "rate_key", record.name)
        now = time.time()
        cnt, start = self.buckets.get(key, (0, now))
        if now - start > self.window_s:
            cnt, start = (0, now)
        if cnt >= self.per_key:
            return False
        self.buckets[key] = (cnt + 1, start)
        return True


def summarize_decision(payload: dict) -> dict:
    """Pick only high-signal fields and round numbers."""

    g = payload or {}
    return {
        "event": g.get("event", "decision"),
        "label": g.get("label"),
        "mode": g.get("mode"),
        "regime": g.get("regime"),
        "score": _round(g.get("score")),
        "rr": _round(g.get("rr")),
        "atr_pct": _round((g.get("atr_pct") or g.get("atr%") or g.get("atrpct"))),
        "action": g.get("action"),
        "side": g.get("side"),
        "strike": g.get("strike"),
        "lots": g.get("lots") or g.get("lot_size"),
        "cap_pct": _round(g.get("cap_pct")),
        "reason_block": g.get("reason_block") or g.get("reason"),
        "entry": g.get("entry"),
        "sl": g.get("sl"),
        "tp": g.get("tp"),
    }


def summarize_micro(payload: dict) -> dict:
    m = payload or {}
    return {
        "event": "micro",
        "depth_ok": m.get("depth_ok"),
        "spread_pct": _round(m.get("spread_pct")),
        "quote": "ok" if m.get("tick_ltp_only") or m.get("tick_full") else "missing",
    }


def compact_extra(extra: dict, max_keys: int = 18) -> dict:
    if not isinstance(extra, dict):
        return {}
    out = {}
    for k in list(extra.keys())[:max_keys]:
        v = extra[k]
        if isinstance(v, float):
            v = _round(v)
        out[k] = v
    return out
