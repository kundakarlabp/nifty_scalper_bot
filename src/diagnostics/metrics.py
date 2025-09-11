"""Lightweight in-process metrics collection."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Dict


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

    def snapshot(self) -> Dict[str, float]:
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
    spread_at_entry: float = 0.0
    micro_wait_ratio: float = 0.0
    auto_relax: float = 0.0
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

    def set_spread_at_entry(self, value: float) -> None:
        with self._lock:
            self.spread_at_entry = float(value)

    def set_micro_wait_ratio(self, value: float) -> None:
        with self._lock:
            self.micro_wait_ratio = float(value)

    def set_auto_relax(self, value: float) -> None:
        with self._lock:
            self.auto_relax = float(value)

    def reset(self) -> None:
        """Reset all metrics to their default values."""
        with self._lock:
            self.fills = 0
            self.cancels = 0
            self.slippage_bps = 0.0
            self.spread_at_entry = 0.0
            self.micro_wait_ratio = 0.0
            self.auto_relax = 0.0

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {
                "fills": self.fills,
                "cancels": self.cancels,
                "slippage_bps": self.slippage_bps,
                "spread_at_entry": self.spread_at_entry,
                "micro_wait_ratio": self.micro_wait_ratio,
                "auto_relax": self.auto_relax,
            }


runtime_metrics = RuntimeMetrics()
