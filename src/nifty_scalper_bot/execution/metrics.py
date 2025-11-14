"""Prometheus metric handles for execution subsystems."""

from __future__ import annotations

from nifty_scalper_bot.utils.metrics import Counter, Gauge, Histogram

QUEUE_SUBMISSIONS = Counter(
    "order_queue_submissions_total",
    "Total number of orders submitted to the queue",
    ["intent", "symbol"],
)
QUEUE_DEPTH = Gauge(
    "order_queue_depth",
    "Current number of pending orders in queue",
)

EXECUTION_LATENCY = Histogram(
    "order_execution_latency_seconds",
    "Latency of order submission attempts",
    ["executor", "status"],
)
SHADOW_DRIFT = Histogram(
    "shadow_mode_drift_basis_points",
    "Slippage drift in shadow mode (bps)",
    ["symbol"],
)

LIFECYCLE_EXITS = Counter(
    "lifecycle_exits_total",
    "Total number of lifecycle exit orders submitted",
    ["symbol", "reason"],
)

VALIDATION_FAILURES = Counter(
    "preflight_validation_failures_total",
    "Total number of preflight validation failures",
    ["symbol", "gate", "level"],
)

__all__ = [
    "QUEUE_SUBMISSIONS",
    "QUEUE_DEPTH",
    "EXECUTION_LATENCY",
    "SHADOW_DRIFT",
    "LIFECYCLE_EXITS",
    "VALIDATION_FAILURES",
]
