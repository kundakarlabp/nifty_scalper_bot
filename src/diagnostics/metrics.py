"""Lightweight in-process metrics collection."""

from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Deque, Dict
from zoneinfo import ZoneInfo

from src.config import settings


@dataclass
class Metrics:
    """Thread-safe metrics recorder."""

    ticks: int = 0
    signals: int = 0
    orders_placed: int = 0
    orders_rejected: int = 0
    queue_depth: int = 0
    last_tick_ts: float = field(default_factory=time.time)
    _start_ts: float = field(default_factory=time.time)
    _latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def inc_ticks(self) -> None:
        with self._lock:
            self.ticks += 1
            self.last_tick_ts = time.time()

    def inc_signal(self) -> None:
        with self._lock:
            self.signals += 1

    def inc_orders(self, *, placed: int = 0, rejected: int = 0) -> None:
        with self._lock:
            self.orders_placed += int(placed)
            self.orders_rejected += int(rejected)

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self.queue_depth = int(depth)

    def observe_latency(self, ms: float) -> None:
        with self._lock:
            self._latencies_ms.append(float(ms))

    def snapshot(self) -> Dict[str, float | str]:
        with self._lock:
            now = time.time()
            age = now - self.last_tick_ts
            tps = self.ticks / max(now - self._start_ts, 1e-6)
            avg_latency = (
                sum(self._latencies_ms) / len(self._latencies_ms)
                if self._latencies_ms
                else 0.0
            )
            return {
                "ticks_per_sec": tps,
                "queue_depth": self.queue_depth,
                "last_tick_age": age,
                "signals": self.signals,
                "orders_placed": self.orders_placed,
                "orders_rejected": self.orders_rejected,
                "avg_latency_ms": avg_latency,
            }


metrics = Metrics()


@dataclass
class RuntimeMetrics:
    """Thread-safe recorder for trade execution metrics."""

    fills: int = 0
    cancels: int = 0
    slippage_bps: float = 0.0
    avg_entry_spread: float = 0.0
    micro_wait_ratio: float = 0.0
    auto_relax: float = 0.0
    minutes_since_last_trade: float = 0.0
    delta: float = 0.0
    elasticity: float = 0.0
    exposure_basis: str = ""
    unit_notional: float = 0.0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def inc_fills(self, n: int = 1) -> None:
        with self._lock:
            self.fills += int(n)

    def inc_cancels(self, n: int = 1) -> None:
        with self._lock:
            self.cancels += int(n)

    def set_slippage_bps(self, value: float) -> None:
        with self._lock:
            self.slippage_bps = float(value)

    def set_avg_entry_spread(self, value: float) -> None:
        with self._lock:
            self.avg_entry_spread = float(value)

    # Backwards compat alias
    def set_spread_at_entry(self, value: float) -> None:  # pragma: no cover - alias
        self.set_avg_entry_spread(value)

    def set_micro_wait_ratio(self, value: float) -> None:
        with self._lock:
            self.micro_wait_ratio = float(value)

    def set_auto_relax(self, value: float) -> None:
        with self._lock:
            self.auto_relax = float(value)

    def set_minutes_since_last_trade(self, value: float) -> None:
        with self._lock:
            self.minutes_since_last_trade = float(value)

    def set_delta(self, value: float) -> None:
        with self._lock:
            self.delta = float(value)

    def set_elasticity(self, value: float) -> None:
        with self._lock:
            self.elasticity = float(value)

    def set_exposure_basis(self, value: str) -> None:
        with self._lock:
            self.exposure_basis = str(value)

    def set_unit_notional(self, value: float) -> None:
        with self._lock:
            self.unit_notional = float(value)

    def reset(self) -> None:
        """Reset all metrics to their default values."""
        with self._lock:
            self.fills = 0
            self.cancels = 0
            self.slippage_bps = 0.0
            self.avg_entry_spread = 0.0
            self.micro_wait_ratio = 0.0
            self.auto_relax = 0.0
            self.minutes_since_last_trade = 0.0
            self.delta = 0.0
            self.elasticity = 0.0
            self.exposure_basis = ""
            self.unit_notional = 0.0

    def snapshot(self) -> Dict[str, float | str]:
        with self._lock:
            return {
                "fills": self.fills,
                "cancels": self.cancels,
                "slippage_bps": self.slippage_bps,
                "avg_entry_spread": self.avg_entry_spread,
                "micro_wait_ratio": self.micro_wait_ratio,
                "auto_relax": self.auto_relax,
                "minutes_since_last_trade": self.minutes_since_last_trade,
                "delta": self.delta,
                "elasticity": self.elasticity,
                "exposure_basis": self.exposure_basis,
                "unit_notional": self.unit_notional,
            }


runtime_metrics = RuntimeMetrics()


TZ = ZoneInfo(getattr(settings, "TZ", "Asia/Kolkata"))
_JOURNAL_HDR = ["ts", "pnl_R", "slippage_bps"]


def _journal_path(dt: datetime | None = None) -> Path:
    dt = dt or datetime.now(TZ)
    return Path("data/journal") / f"{dt.date()}.csv"


def record_trade(pnl_r: float, slippage_bps: float) -> None:
    """Append a trade row to today's CSV journal."""
    path = _journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_JOURNAL_HDR)
        if new:
            w.writeheader()
        w.writerow(
            {
                "ts": datetime.now(TZ).isoformat(),
                "pnl_R": float(pnl_r),
                "slippage_bps": float(slippage_bps),
            }
        )


def daily_summary(dt: datetime | None = None) -> Dict[str, float]:
    """Return aggregated trade metrics for the day."""
    path = _journal_path(dt)
    if not path.exists():
        return {"R": 0.0, "hit_rate": 0.0, "avg_R": 0.0, "slippage_bps": 0.0, "trades": 0}
    with path.open() as f:
        rows = list(csv.DictReader(f))
    rs = [float(r.get("pnl_R", 0.0)) for r in rows]
    sls = [float(r.get("slippage_bps", 0.0)) for r in rows]
    trades = len(rs)
    if trades == 0:
        return {"R": 0.0, "hit_rate": 0.0, "avg_R": 0.0, "slippage_bps": 0.0, "trades": 0}
    total_r = sum(rs)
    hit_rate = (sum(1 for r in rs if r > 0) / trades) * 100.0
    avg_r = total_r / trades
    avg_slip = sum(sls) / trades if sls else 0.0
    return {
        "R": round(total_r, 2),
        "hit_rate": round(hit_rate, 1),
        "avg_R": round(avg_r, 2),
        "slippage_bps": round(avg_slip, 2),
        "trades": trades,
    }

