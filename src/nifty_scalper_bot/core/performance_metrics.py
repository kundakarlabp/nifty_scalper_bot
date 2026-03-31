"""Aggregate runtime performance metrics for diagnostics and telemetry."""

from __future__ import annotations

import math
import statistics
import typing as t
from dataclasses import dataclass

from nifty_scalper_bot.infra.metrics import METRICS, MetricsCollector
from nifty_scalper_bot.utils.logging import get_logger

log = get_logger(__name__)


_FILL_RATIO_BUCKETS: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 1.01)


@dataclass(slots=True)
class FillRateSnapshot:
    """Summary of order fill outcomes over a sample window."""

    total_orders: int
    filled_orders: int
    partially_filled_orders: int
    cancelled_orders: int
    rejected_orders: int
    average_fill_ratio: float
    fill_ratio_histogram: dict[str, float]

    def as_dict(self) -> dict[str, t.Any]:
        """Return the snapshot as a serialisable dictionary."""

        return {
            "total_orders": self.total_orders,
            "filled_orders": self.filled_orders,
            "partially_filled_orders": self.partially_filled_orders,
            "cancelled_orders": self.cancelled_orders,
            "rejected_orders": self.rejected_orders,
            "fill_rate": self.fill_rate,
            "partial_fill_rate": self.partial_fill_rate,
            "average_fill_ratio": self.average_fill_ratio,
            "fill_quality_score": self.fill_quality_score,
            "fill_ratio_histogram": dict(self.fill_ratio_histogram),
        }

    @property
    def fill_rate(self) -> float:
        """Return the proportion of fully filled orders."""

        if self.total_orders <= 0:
            return 0.0
        return self.filled_orders / self.total_orders

    @property
    def partial_fill_rate(self) -> float:
        """Return the proportion of partially filled orders."""

        if self.total_orders <= 0:
            return 0.0
        return self.partially_filled_orders / self.total_orders

    @property
    def fill_quality_score(self) -> float:
        """Return composite score balancing fill rates and ratios.

        Args:
            None.

        Returns:
            float: Weighted fill quality score clamped to ``[0, 1]``.

        Raises:
            None.
        """

        if self.total_orders <= 0:
            return 0.0
        return max(0.0, min(self.fill_rate * self.average_fill_ratio, 1.0))


@dataclass(slots=True)
class PerformanceSnapshot:
    """Composite performance metrics surfaced to operators."""

    metrics: dict[str, t.Any]
    fill_stats: FillRateSnapshot

    def as_dict(self) -> dict[str, t.Any]:
        """Return payload suitable for JSON serialisation."""

        payload = dict(self.metrics)
        payload["fill_stats"] = self.fill_stats.as_dict()
        return payload


def _build_ratio_histogram(values: t.Sequence[float]) -> dict[str, float]:
    """Return normalised histogram buckets for fill ratios.

    Args:
        values: Sequence of fill ratio observations within ``[0, 1]``.

    Returns:
        dict[str, float]: Mapping of bucket labels to proportional counts.

    Raises:
        None.
    """

    buckets: list[tuple[float, float, str]] = []
    lower_bound = 0.0
    for edge in _FILL_RATIO_BUCKETS:
        upper_bound = max(edge, lower_bound)
        label = f"{lower_bound:.2f}-{min(upper_bound, 1.0):.2f}"
        buckets.append((lower_bound, upper_bound, label))
        lower_bound = upper_bound
    counts: dict[str, float] = {label: 0.0 for _, _, label in buckets}
    for raw in values:
        try:
            ratio = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(ratio) or ratio < 0:
            continue
        for lower, upper, label in buckets:
            if ratio <= upper:
                counts[label] += 1.0
                break
    total = sum(counts.values())
    if total <= 0:
        return counts
    return {label: count / total for label, count in counts.items()}


def _extract_orders(order_manager: t.Any, sample_size: int) -> list[t.Any]:
    """Extract up to ``sample_size`` orders from *order_manager*.

    Args:
        order_manager: Order manager instance exposing history helpers.
        sample_size: Number of records to fetch for statistics.

    Returns:
        list[t.Any]: Normalised list of order entries.

    Raises:
        None.
    """

    log.debug(
        "Entered _extract_orders",
        extra={"event": "perf_extract_orders_enter", "sample": sample_size},
    )
    if order_manager is None:
        return []
    orders: list[t.Any] = []
    try:
        history = getattr(order_manager, "get_order_history", None)
        if callable(history):
            orders = list(history(limit=sample_size))
        else:
            recent = getattr(order_manager, "recent_orders", None)
            if callable(recent):
                orders = list(recent(limit=sample_size))
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Failure in _extract_orders: %s",
            exc,
            extra={"event": "perf_extract_orders_error"},
        )
        return []
    log.info(
        "Condition met: extracted orders",
        extra={"event": "perf_extract_orders_ready", "count": len(orders)},
    )
    return orders[-sample_size:]


def _normalise_status(status: t.Any) -> str:
    """Normalise various status representations to lowercase text."""

    if status is None:
        return "unknown"
    if isinstance(status, str):
        return status.strip().lower() or "unknown"
    name = getattr(status, "value", None)
    if isinstance(name, str):
        return name.strip().lower() or "unknown"
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name.strip().lower() or "unknown"
    return str(status).strip().lower() or "unknown"


def _coerce_number(candidate: t.Any) -> float:
    """Safely convert numeric candidates to ``float``."""

    try:
        if candidate is None:
            return math.nan
        return float(candidate)
    except (TypeError, ValueError):
        return math.nan


def calculate_fill_rates(
    order_manager: t.Any | None,
    *,
    sample_size: int = 200,
) -> FillRateSnapshot:
    """Compute fill outcomes and ratios for recent orders.

    Args:
        order_manager: Order manager providing access to order history.
        sample_size: Number of orders to include in the aggregation window.

    Returns:
        FillRateSnapshot: Aggregated fill outcome snapshot.

    Raises:
        None.
    """

    log.debug(
        "Entered calculate_fill_rates",
        extra={"event": "perf_fill_rates_enter", "sample": sample_size},
    )
    orders = _extract_orders(order_manager, sample_size)
    total = len(orders)
    filled = 0
    partial = 0
    cancelled = 0
    rejected = 0
    fill_ratios: list[float] = []
    for entry in orders:
        try:
            raw_status = (
                entry.get("status")
                if isinstance(entry, dict)
                else getattr(entry, "status", None)
            )
            status = _normalise_status(raw_status)
            raw_quantity = (
                entry.get("quantity")
                if isinstance(entry, dict)
                else getattr(entry, "quantity", None)
            )
            quantity = _coerce_number(raw_quantity)
            raw_filled = (
                entry.get("filled_quantity")
                if isinstance(entry, dict)
                else getattr(entry, "filled_quantity", None)
            )
            filled_qty = _coerce_number(raw_filled)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in calculate_fill_rates entry parse: %s",
                exc,
                extra={"event": "perf_fill_rates_entry_error"},
            )
            continue
        if status in {"filled", "complete", "executed"}:
            filled += 1
        elif status in {"partially filled", "partially_filled", "partial_filled"}:
            partial += 1
        elif status in {"cancelled", "cancelled_pending", "cancelled by user"}:
            cancelled += 1
        elif status in {"rejected", "rejected_pending"}:
            rejected += 1
        if not math.isnan(quantity) and quantity > 0:
            ratio = 0.0
            if math.isnan(filled_qty) or filled_qty < 0:
                ratio = 0.0
            else:
                ratio = min(max(filled_qty / quantity, 0.0), 1.0)
            fill_ratios.append(ratio)
    average_ratio = statistics.mean(fill_ratios) if fill_ratios else 0.0
    histogram = _build_ratio_histogram(fill_ratios)
    snapshot = FillRateSnapshot(
        total_orders=total,
        filled_orders=filled,
        partially_filled_orders=partial,
        cancelled_orders=cancelled,
        rejected_orders=rejected,
        average_fill_ratio=average_ratio,
        fill_ratio_histogram=histogram,
    )
    log.info(
        "Condition met: fill rates calculated",
        extra={
            "event": "perf_fill_rates_ready",
            "total": total,
            "fill_rate": snapshot.fill_rate,
            "partial_rate": snapshot.partial_fill_rate,
        },
    )
    return snapshot


def build_performance_snapshot(
    *,
    order_manager: t.Any | None = None,
    metrics: MetricsCollector | None = None,
    sample_size: int = 200,
) -> PerformanceSnapshot:
    """Return consolidated metrics for diagnostics dashboards.

    Args:
        order_manager: Optional order manager for fill-rate statistics.
        metrics: Optional metrics collector override.
        sample_size: Number of orders to evaluate for fill statistics.

    Returns:
        PerformanceSnapshot: Aggregated metrics bundle.

    Raises:
        None.
    """

    log.debug(
        "Entered build_performance_snapshot",
        extra={"event": "perf_snapshot_enter", "sample": sample_size},
    )
    collector: MetricsCollector = metrics or METRICS
    summary: dict[str, t.Any]
    try:
        raw_summary = collector.metric_summary()
        summary = dict(raw_summary)
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Failure in build_performance_snapshot metric summary: %s",
            exc,
            extra={"event": "perf_snapshot_metric_error"},
        )
        summary = {}
    fill_snapshot = calculate_fill_rates(order_manager, sample_size=sample_size)
    snapshot = PerformanceSnapshot(metrics=summary, fill_stats=fill_snapshot)
    log.info(
        "Condition met: performance snapshot ready",
        extra={
            "event": "perf_snapshot_ready",
            "fill_rate": fill_snapshot.fill_rate,
            "orders": fill_snapshot.total_orders,
        },
    )
    return snapshot


__all__ = [
    "FillRateSnapshot",
    "PerformanceSnapshot",
    "build_performance_snapshot",
    "calculate_fill_rates",
]
