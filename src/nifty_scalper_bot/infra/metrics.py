"""Centralised runtime metrics and alert evaluation helpers."""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Callable, Deque, Dict, Iterable, Mapping, MutableMapping, Set, Tuple

from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.pricing import canonical_price_source
from nifty_scalper_bot.utils.metrics import Counter, Gauge, Histogram
from nifty_scalper_bot.utils.reasons import canonical

LOGGER = get_logger(__name__)

POLL_RECONNECTS = Counter(
    "poll_reconnects_total",
    "Total polling restarts/backoffs",
    ["reason"],
)
POLL_ERRORS = Counter(
    "poll_errors_total",
    "Total polling errors encountered",
    ["reason"],
)
POLL_HEARTBEAT_SKIPS = Counter(
    "poll_heartbeat_skips_total",
    "Polling heartbeat misses grouped by market phase",
    ["phase"],
)
POLL_TICK_LAG_MS = Gauge(
    "poll_tick_lag_ms",
    "Lag in milliseconds between exchange tick timestamp and processing",
)
POLL_LAST_TICK_TS = Gauge(
    "poll_last_tick_epoch_ms",
    "Epoch milliseconds of the most recent processed poll tick",
)

RESOLVER_TOKENS = Counter(
    "resolver_tokens_total",
    "Resolver tokens loaded grouped by source",
    ["source"],
)
RESOLVER_MISSES = Counter(
    "resolver_misses_total",
    "Resolver lookups that returned no instrument token",
    ["symbol"],
)
RESOLVER_LEARN_EVENTS = Counter(
    "resolver_learn_events_total",
    "Resolver learn events grouped by label",
    ["label"],
)

_TICK_LATENCY_BUCKETS = (
    0.01,
    0.02,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
)
_ORDER_LATENCY_BUCKETS = (
    0.05,
    0.1,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
    2.0,
)


@dataclass(slots=True)
class AlertState:
    """Snapshot of an alert, exposed via health endpoints and tests."""

    name: str
    active: bool
    severity: str
    observed: float | int | None
    threshold: float
    triggered_at: float | None = None
    details: dict[str, float | str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return serialisable representation used by the health endpoint."""

        return {
            "name": self.name,
            "active": self.active,
            "severity": self.severity,
            "observed": self.observed,
            "threshold": self.threshold,
            "triggered_at": self.triggered_at,
            "details": dict(self.details),
        }


class MetricsCollector:
    """Maintain metric handles and evaluate SLO-style alert rules."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._tick_latency = Histogram(
            "tick_latency_seconds",
            "Latency between tick timestamp and processing in seconds",
            labels=["symbol"],
            buckets=_TICK_LATENCY_BUCKETS,
        )
        self._tick_staleness = Gauge(
            "tick_staleness_seconds",
            "Seconds since latest tick was observed",
            ["symbol"],
        )
        self._order_latency = Histogram(
            "order_latency_seconds",
            "Latency from order submission to fill acknowledgement",
            labels=["order_type"],
            buckets=_ORDER_LATENCY_BUCKETS,
        )
        self._error_counter = Counter(
            "error_count_by_type",
            "Count of classified runtime errors",
            ["type"],
        )
        self._broker_sync_counter = Counter(
            "broker_sync_total",
            "Broker reconciliation outcomes by status and reason",
            ["status", "reason"],
        )
        self._broker_sync_latency = Histogram(
            "broker_sync_latency_seconds",
            "Latency in seconds for broker reconciliation cycles",
            labels=["status"],
            buckets=(
                0.1,
                0.25,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                30.0,
            ),
        )
        self._regime_changes = Counter(
            "market_regime_change_total",
            "Market regime transitions observed by detector",
            ["regime"],
        )
        self._regime_confidence = Gauge(
            "market_regime_confidence",
            "Latest confidence score reported per regime",
            ["regime"],
        )
        self._strategy_allocation = Gauge(
            "strategy_allocation_fraction",
            "Current capital allocation fraction per strategy",
            ["strategy"],
        )
        self._strategy_allocation_change = Counter(
            "strategy_allocation_change_total",
            "Count of strategy allocation change events by direction",
            ["strategy", "direction"],
        )
        self._heartbeat_flush_counter = Counter(
            "persistent_state_flush_total",
            "Persistent state heartbeat flush attempts by outcome",
            ["outcome"],
        )
        self._heartbeat_flush_latency = Histogram(
            "persistent_state_flush_latency_seconds",
            "Latency in seconds for heartbeat-triggered persistent flushes",
            buckets=(
                0.01,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.0,
                5.0,
            ),
        )
        self._heartbeat_flush_age = Gauge(
            "persistent_state_flush_age_seconds",
            "Seconds since the last successful persistent state flush",
        )
        self._retry_counter = Counter(
            "retry_events_total",
            "Retry attempts by label, stage, and outcome",
            ["label", "stage", "outcome"],
        )
        self._plan_no_price = Counter(
            "plan_no_price_total",
            "Unified manager plans blocked due to missing price",
            ["symbol"],
        )
        self._plan_estimate_zero = Counter(
            "plan_estimate_zero_total",
            "Unified manager plans resulting in zero estimated notional",
            ["symbol"],
        )
        self._telegram_margin_um_fallback = Counter(
            "telegram_margin_um_fallback_total",
            "/margin unified manager fallback usage grouped by price source",
            ["source"],
        )
        self._resolver_tokens_metric = RESOLVER_TOKENS
        self._resolver_misses = RESOLVER_MISSES
        self._resolver_learn_metric = RESOLVER_LEARN_EVENTS
        self._resolver_token_total = 0
        self._resolver_miss_total = 0
        self._resolver_miss_symbols: Set[str] = set()
        self._mdm_rest_seed = Counter(
            "mdm_rest_seed_total",
            "Market data manager REST seed attempts by outcome",
            ["symbol", "outcome"],
        )
        self._live_pnl = Gauge(
            "live_pnl_gauge",
            "Live running PnL per book",
            ["book"],
        )
        self._pnl_breakdown = Gauge(
            "pnl_breakdown",
            "PnL breakdown by type (realized/unrealized)",
            ["book", "kind"],
        )
        self._order_rejections = Counter(
            "order_rejections_total",
            "Orders rejected by reason",
            ["reason"],
        )
        self.bracket_registrations = Counter(
            "bracket_registrations_total",
            "Total bracket orders registered",
        )
        self.bracket_oco_cancellations = Counter(
            "bracket_oco_cancellations_total",
            "OCO cancellations triggered",
            ["leg_type"],
        )
        self.bracket_oco_failures = Counter(
            "bracket_oco_failures_total",
            "Failed OCO cancellations",
        )
        self.bracket_active = Gauge(
            "bracket_active_total",
            "Currently active bracket orders",
        )
        self.bracket_sl_adjust_failures = Counter(
            "bracket_sl_adjust_failures_total",
            "Failed SL quantity adjustments",
        )
        self.bracket_stale_cleanups = Counter(
            "bracket_stale_cleanups_total",
            "Stale brackets cleaned up",
        )
        self._queue_depth = Gauge(
            "order_queue_depth",
            "Estimated queue depth observed when routing orders",
            ["symbol"],
        )
        self._tick_samples: Deque[tuple[float, float]] = deque()
        self._order_samples: Deque[tuple[float, float]] = deque()
        self._stale_samples: Deque[tuple[float, float]] = deque()
        self._error_events: Deque[tuple[float, str]] = deque()
        self._queue_samples: Deque[tuple[float, float]] = deque()
        self._rejection_events: Deque[tuple[float, str]] = deque()
        self._heartbeat_flush_events: Deque[Tuple[float, str, float]] = deque()
        self._diagnostic_events = Counter(
            "diagnostic_events_total",
            "Diagnostic events by name and reason",
            ["event", "reason"],
        )
        self._diagnostic_recent: Deque[
            tuple[float, str, str, str, dict[str, object]]
        ] = deque()
        self._pnl_state: Dict[str, Dict[str, float]] = {}
        self._alert_state: Dict[str, AlertState] = {}
        self._notifiers: list[Callable[[AlertState], None]] = []
        self._broker_sync_events: Set[str] = set()
        self._regime_events: Set[str] = set()
        self._allocation_state: Dict[str, float] = {}
        self._allocation_change_last: Dict[str, str] = {}
        self._latest_regime: Tuple[str, float, float] | None = None
        self._last_successful_flush: float | None = None

    # ------------------------------------------------------------------
    # Registration -----------------------------------------------------
    def register_notifier(self, callback: Callable[[AlertState], None]) -> None:
        """Register *callback* invoked whenever an alert toggles state."""

        LOGGER.debug(
            "Entered MetricsCollector.register_notifier",
            extra={"event": "metrics_register_notifier"},
        )
        if callback is None:
            return
        with self._lock:
            self._notifiers.append(callback)

    # ------------------------------------------------------------------
    # Observation helpers ----------------------------------------------
    def observe_tick(
        self,
        *,
        symbol: str,
        latency_seconds: float | None,
        staleness_seconds: float | None,
        now: float | None = None,
    ) -> None:
        """Record latency/staleness observations for *symbol* ticks.

        Args:
            symbol: Canonical instrument identifier.
            latency_seconds: Latency between tick timestamp and ingestion.
            staleness_seconds: Derived age of the feed relative to wall-clock.
            now: Optional monotonic timestamp for deterministic testing.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.observe_tick",
            extra={"event": "metrics_observe_tick", "symbol": symbol},
        )
        reference_time = float(now if now is not None else time.time())
        symbol_label = symbol or "UNKNOWN"
        if latency_seconds is not None and latency_seconds >= 0:
            if latency_seconds > 0.5:
                LOGGER.info(
                    "Condition met: tick_latency_high",
                    extra={
                        "event": "tick_latency_high",
                        "symbol": symbol_label,
                        "latency_seconds": latency_seconds,
                    },
                )
            try:
                self._tick_latency.labels(symbol_label).observe(latency_seconds)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Failure in MetricsCollector.observe_tick: %s",
                    exc,
                    extra={
                        "event": "tick_latency_metric_error",
                        "symbol": symbol_label,
                    },
                )
            with self._lock:
                self._tick_samples.append((reference_time, float(latency_seconds)))
        if staleness_seconds is not None and staleness_seconds >= 0:
            if staleness_seconds > 2.0:
                LOGGER.info(
                    "Condition met: tick_staleness_high",
                    extra={
                        "event": "tick_staleness_high",
                        "symbol": symbol_label,
                        "staleness_seconds": staleness_seconds,
                    },
                )
            try:
                self._tick_staleness.labels(symbol_label).set(staleness_seconds)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Failure in MetricsCollector.observe_tick: %s",
                    exc,
                    extra={
                        "event": "tick_staleness_metric_error",
                        "symbol": symbol_label,
                    },
                )
            with self._lock:
                self._stale_samples.append((reference_time, float(staleness_seconds)))
        self._purge_expired(reference_time)
        self._evaluate_alerts_locked(reference_time)

    def observe_order_latency(
        self,
        *,
        order_type: str,
        latency_seconds: float,
        now: float | None = None,
    ) -> None:
        """Record latency for order acknowledgements.

        Args:
            order_type: Human readable identifier of the order type.
            latency_seconds: Observed latency between submission and fill.
            now: Optional monotonic timestamp for deterministic testing.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.observe_order_latency",
            extra={"event": "metrics_observe_order_latency", "type": order_type},
        )
        reference_time = float(now if now is not None else time.time())
        safe_type = order_type or "UNKNOWN"
        try:
            self._order_latency.labels(safe_type).observe(max(latency_seconds, 0.0))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in MetricsCollector.observe_order_latency: %s",
                exc,
                extra={"event": "order_latency_metric_error", "type": safe_type},
            )
        with self._lock:
            self._order_samples.append((reference_time, max(latency_seconds, 0.0)))
            self._purge_expired(reference_time)
            self._evaluate_alerts_locked(reference_time)
        if latency_seconds > 0.5:
            LOGGER.info(
                "Condition met: order_latency_threshold_crossed",
                extra={
                    "event": "order_latency_threshold_crossed",
                    "type": safe_type,
                    "latency_seconds": latency_seconds,
                },
            )

    def increment_error(self, *, error_type: str, now: float | None = None) -> None:
        """Increment error counter for *error_type* and track bursts.

        Args:
            error_type: Canonical classification for the failure.
            now: Optional timestamp override for tests.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.increment_error",
            extra={"event": "metrics_increment_error", "type": error_type},
        )
        safe_type = error_type or "unknown"
        try:
            self._error_counter.labels(safe_type).inc()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in MetricsCollector.increment_error: %s",
                exc,
                extra={"event": "error_count_metric_error", "type": safe_type},
            )
        reference_time = float(now if now is not None else time.time())
        with self._lock:
            self._error_events.append((reference_time, safe_type))
            self._purge_expired(reference_time)
            self._evaluate_alerts_locked(reference_time)
        LOGGER.info(
            "Condition met: error_counter_incremented",
            extra={"event": "error_counter_incremented", "type": safe_type},
        )

    def record_broker_sync(
        self,
        *,
        success: bool,
        reason: str | None = None,
        latency_seconds: float | None = None,
        event_id: str | None = None,
        now: float | None = None,
    ) -> None:
        """Record broker reconciliation outcome with optional deduplication.

        Args:
            success: ``True`` when reconciliation completed successfully.
            reason: Optional descriptive reason for failures.
            latency_seconds: Optional duration of the reconciliation attempt.
            event_id: Optional identifier used to deduplicate duplicate events.
            now: Optional timestamp override for deterministic testing.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.record_broker_sync",
            extra={"event": "metrics_broker_sync", "success": success},
        )
        reference_time = float(now if now is not None else time.time())
        status = "success" if success else "failure"
        token = canonical(reason) if reason else ("ok" if success else "unknown")
        dedup_key = event_id or f"{status}:{token}:{int(reference_time * 1000)}"
        with self._lock:
            if dedup_key in self._broker_sync_events:
                return
            if len(self._broker_sync_events) >= 512:
                self._broker_sync_events.clear()
            self._broker_sync_events.add(dedup_key)
        try:
            self._broker_sync_counter.labels(status, token).inc()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in record_broker_sync counter: %s",
                exc,
                extra={"event": "metrics_broker_sync_counter_error"},
            )
        if latency_seconds is not None and latency_seconds >= 0.0:
            try:
                self._broker_sync_latency.labels(status).observe(latency_seconds)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Failure in record_broker_sync latency: %s",
                    exc,
                    extra={"event": "metrics_broker_sync_latency_error"},
                )

    def update_regime_confidence(
        self,
        *,
        regime: str | None,
        confidence: float,
        now: float | None = None,
    ) -> None:
        """Update the cached regime confidence gauge.

        Args:
            regime: Regime identifier whose confidence is recorded.
            confidence: Confidence score bounded to ``0.0``–``1.0``.
            now: Optional timestamp used when recording update time.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.update_regime_confidence",
            extra={"event": "metrics_update_regime_confidence", "regime": regime},
        )
        normalized = (regime or "unknown").strip().lower() or "unknown"
        bounded_confidence = max(0.0, min(float(confidence), 1.0))
        try:
            self._regime_confidence.labels(normalized).set(bounded_confidence)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in update_regime_confidence gauge: %s",
                exc,
                extra={
                    "event": "metrics_regime_confidence_error",
                    "regime": normalized,
                },
            )
        reference_time = float(now if now is not None else time.time())
        with self._lock:
            self._latest_regime = (normalized, bounded_confidence, reference_time)

    def record_regime_change(
        self,
        *,
        regime: str,
        confidence: float,
        event_id: str | None = None,
        now: float | None = None,
    ) -> None:
        """Record a market regime transition with deduplication safeguards.

        Args:
            regime: Canonical name of the detected market regime.
            confidence: Detector confidence score expressed as 0.0–1.0.
            event_id: Optional identifier preventing duplicate increments.
            now: Optional timestamp override for deterministic testing.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.record_regime_change",
            extra={"event": "metrics_regime_change", "regime": regime},
        )
        reference_time = float(now if now is not None else time.time())
        normalized = (regime or "unknown").strip().lower() or "unknown"
        self.update_regime_confidence(
            regime=normalized, confidence=confidence, now=reference_time
        )
        dedup_key = event_id or f"{normalized}:{int(reference_time * 1000)}"
        with self._lock:
            if dedup_key in self._regime_events:
                return
            if len(self._regime_events) >= 512:
                self._regime_events.clear()
            self._regime_events.add(dedup_key)
        try:
            self._regime_changes.labels(normalized).inc()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in record_regime_change counter: %s",
                exc,
                extra={"event": "metrics_regime_counter_error"},
            )

    def record_heartbeat_flush(
        self,
        *,
        success: bool,
        latency_seconds: float | None = None,
        now: float | None = None,
    ) -> None:
        """Record a heartbeat-triggered persistent state flush event.

        Args:
            success: ``True`` when the flush completed successfully.
            latency_seconds: Optional duration of the flush invocation.
            now: Optional timestamp override for deterministic testing.

        Returns:
            None.

        Raises:
            None.
        """

        outcome = "success" if success else "failure"
        LOGGER.debug(
            "Entered MetricsCollector.record_heartbeat_flush",
            extra={"event": "metrics_record_heartbeat_flush", "outcome": outcome},
        )
        reference_time = float(now if now is not None else time.time())
        try:
            self._heartbeat_flush_counter.labels(outcome).inc()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in record_heartbeat_flush counter: %s",
                exc,
                extra={
                    "event": "metrics_heartbeat_flush_counter_error",
                    "outcome": outcome,
                },
            )
        # Counter-only; no latency/heartbeat handling here.

    def record_plan_no_price(self, *, symbol: str) -> None:
        """Increment the missing-price counter for unified manager plans.

        Args:
            symbol: Trading symbol associated with the failed plan.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.record_plan_no_price",
            extra={"event": "metrics_plan_no_price", "symbol": symbol},
        )
        safe_symbol = (symbol or "UNKNOWN").strip().upper() or "UNKNOWN"
        try:
            self._plan_no_price.labels(safe_symbol).inc()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in record_plan_no_price counter: %s",
                exc,
                extra={"event": "metrics_plan_no_price_error", "symbol": safe_symbol},
            )

    def record_plan_estimate_zero(self, *, symbol: str) -> None:
        """Increment the zero-estimate counter for unified manager plans.

        Args:
            symbol: Trading symbol associated with the zero estimate.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.record_plan_estimate_zero",
            extra={"event": "metrics_plan_estimate_zero", "symbol": symbol},
        )
        safe_symbol = (symbol or "UNKNOWN").strip().upper() or "UNKNOWN"
        try:
            self._plan_estimate_zero.labels(safe_symbol).inc()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in record_plan_estimate_zero counter: %s",
                exc,
                extra={
                    "event": "metrics_plan_estimate_zero_error",
                    "symbol": safe_symbol,
                },
            )

    def record_telegram_margin_um_fallback(self, *, source: str) -> None:
        """Track unified manager fallback usage for the ``/margin`` command.

        Args:
            source: Canonical price source label associated with the fallback.

        Returns:
            None.

        Raises:
            None.
        """

        label = canonical_price_source(source)
        LOGGER.debug(
            "Entered MetricsCollector.record_telegram_margin_um_fallback",
            extra={
                "event": "metrics_margin_um_fallback",
                "source": label,
            },
        )
        try:
            self._telegram_margin_um_fallback.labels(label).inc()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in record_telegram_margin_um_fallback counter: %s",
                exc,
                extra={
                    "event": "metrics_margin_um_fallback_error",
                    "source": label,
                },
            )

    def record_resolver_tokens(self, *, source: str, count: int) -> None:
        """Track resolver token loads grouped by *source*.

        Args:
            source: Human readable identifier describing the load origin.
            count: Number of tokens warmed during the load operation.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.record_resolver_tokens",
            extra={
                "event": "metrics_resolver_tokens",
                "source": source,
                "count": count,
            },
        )
        safe_source = (source or "unknown").strip().lower() or "unknown"
        try:
            if count > 0:
                self._resolver_tokens_metric.labels(safe_source).inc(max(int(count), 0))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in record_resolver_tokens counter: %s",
                exc,
                extra={
                    "event": "metrics_resolver_tokens_error",
                    "source": safe_source,
                },
            )
        with self._lock:
            self._resolver_token_total = max(0, int(count))

    def record_resolver_learn(self, *, count: int, label: str) -> None:
        """Record resolver learn events triggered during cache refresh.

        Args:
            count: Number of option rows learned during the refresh.
            label: Qualitative label describing the refresh mode.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.record_resolver_learn",
            extra={
                "event": "metrics_resolver_learn",
                "count": count,
                "label": label,
            },
        )
        safe_label = (label or "refresh").strip().lower() or "refresh"
        try:
            if count > 0:
                self._resolver_learn_metric.labels(safe_label).inc(max(int(count), 0))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in record_resolver_learn counter: %s",
                exc,
                extra={
                    "event": "metrics_resolver_learn_error",
                    "label": safe_label,
                },
            )

    def record_resolver_miss(self, *, symbol: str) -> None:
        """Increment the resolver miss counter for unresolved symbols.

        Args:
            symbol: Trading symbol that failed to resolve to a token.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.record_resolver_miss",
            extra={"event": "metrics_resolver_miss", "symbol": symbol},
        )
        safe_symbol = (symbol or "UNKNOWN").strip().upper() or "UNKNOWN"
        try:
            self._resolver_misses.labels(safe_symbol).inc()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in record_resolver_miss counter: %s",
                exc,
                extra={"event": "metrics_resolver_miss_error", "symbol": safe_symbol},
            )
        with self._lock:
            self._resolver_miss_total += 1
            if safe_symbol:
                self._resolver_miss_symbols.add(safe_symbol)

    def record_mdm_rest_seed(self, *, symbol: str, outcome: str) -> None:
        """Increment REST seed metrics for market data manager attempts.

        Args:
            symbol: Trading symbol targeted by the REST seed.
            outcome: Qualitative outcome label (e.g. ``"success"``).

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.record_mdm_rest_seed",
            extra={
                "event": "metrics_mdm_rest_seed",
                "symbol": symbol,
                "outcome": outcome,
            },
        )
        safe_symbol = (symbol or "UNKNOWN").strip().upper() or "UNKNOWN"
        safe_outcome = (outcome or "unknown").strip().lower() or "unknown"
        try:
            self._mdm_rest_seed.labels(safe_symbol, safe_outcome).inc()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in record_mdm_rest_seed counter: %s",
                exc,
                extra={
                    "event": "metrics_mdm_rest_seed_error",
                    "symbol": safe_symbol,
                    "outcome": safe_outcome,
                },
            )

    def resolver_summary(self) -> dict[str, int]:
        """Return aggregated resolver statistics for observability."""

        LOGGER.debug(
            "Entered MetricsCollector.resolver_summary",
            extra={"event": "metrics_resolver_summary"},
        )
        with self._lock:
            return {
                "tokens": int(self._resolver_token_total),
                "miss_total": int(self._resolver_miss_total),
                "miss_symbols": len(self._resolver_miss_symbols),
            }

    def record_strategy_allocation(
        self,
        *,
        strategy: str,
        weight: float,
        score: float,
        regime: str | None = None,
    ) -> None:
        """Publish the latest allocation weight for *strategy*.

        Args:
            strategy: Strategy identifier whose allocation changed.
            weight: Capital allocation fraction (0.0–1.0) being applied.
            score: Composite score backing the allocation decision.
            regime: Optional regime context associated with the allocation.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.record_strategy_allocation",
            extra={
                "event": "metrics_strategy_allocation",
                "strategy": strategy,
                "weight": weight,
            },
        )
        normalized_strategy = (strategy or "unknown").strip() or "unknown"
        bounded_weight = max(0.0, float(weight))
        previous = self._allocation_state.get(normalized_strategy)
        if previous is not None and abs(previous - bounded_weight) < 1e-4:
            return
        self._allocation_state[normalized_strategy] = bounded_weight
        try:
            self._strategy_allocation.labels(normalized_strategy).set(bounded_weight)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in record_strategy_allocation gauge: %s",
                exc,
                extra={"event": "metrics_strategy_allocation_error"},
            )
        regime_label = (regime or "unknown").strip().lower() or "unknown"
        try:
            self._regime_confidence.labels(regime_label)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in record_strategy_allocation regime gauge: %s",
                exc,
                extra={
                    "event": "metrics_regime_confidence_error",
                    "regime": regime_label,
                },
            )

    def increment_strategy_allocation_change(
        self,
        *,
        strategy: str,
        direction: str,
    ) -> None:
        """Increment allocation change counter for *strategy*.

        Args:
            strategy: Strategy identifier associated with the adjustment.
            direction: Direction descriptor such as ``increase`` or ``decrease``.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.increment_strategy_allocation_change",
            extra={
                "event": "metrics_strategy_allocation_change",
                "strategy": strategy,
                "direction": direction,
            },
        )
        normalized_strategy = (strategy or "unknown").strip() or "unknown"
        normalized_direction = (direction or "unknown").strip().lower() or "unknown"
        previous_direction = self._allocation_change_last.get(normalized_strategy)
        if previous_direction == normalized_direction:
            return
        self._allocation_change_last[normalized_strategy] = normalized_direction
        try:
            self._strategy_allocation_change.labels(
                normalized_strategy,
                normalized_direction,
            ).inc()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in increment_strategy_allocation_change: %s",
                exc,
                extra={
                    "event": "metrics_strategy_allocation_change_error",
                    "strategy": normalized_strategy,
                    "direction": normalized_direction,
                },
            )

    def increment_retry_event(
        self,
        *,
        label: str,
        stage: str,
        outcome: str,
    ) -> None:
        """Increment retry counter with structured labels.

        Args:
            label: High-level identifier for the retrying operation.
            stage: Pipeline stage (for example ``connect`` or ``submit``).
            outcome: Result descriptor such as ``failure`` or ``success``.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.increment_retry_event",
            extra={
                "event": "metrics_retry_event",
                "label": label,
                "stage": stage,
                "outcome": outcome,
            },
        )
        try:
            self._retry_counter.labels(label, stage, outcome).inc()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in increment_retry_event: %s",
                exc,
                extra={"event": "metrics_retry_event_error"},
            )

    def record_order_rejection(self, *, reason: str, now: float | None = None) -> None:
        """Track an order rejection reason for burst analysis.

        Args:
            reason: Broker-provided or derived rejection explanation.
            now: Optional timestamp override for deterministic testing.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.record_order_rejection",
            extra={"event": "metrics_record_order_rejection", "reason": reason},
        )
        token = canonical(reason)
        safe_reason = token if token != "OK" else "unknown"
        try:
            self._order_rejections.labels(safe_reason).inc()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in MetricsCollector.record_order_rejection: %s",
                exc,
                extra={
                    "event": "order_rejection_metric_error",
                    "reason": safe_reason,
                },
            )
        reference_time = float(now if now is not None else time.time())
        with self._lock:
            self._rejection_events.append((reference_time, safe_reason))
            self._purge_expired(reference_time)
        LOGGER.info(
            "Condition met: order_rejection_recorded",
            extra={
                "event": "order_rejection_recorded",
                "reason": safe_reason,
            },
        )

    def record_diagnostic_event(
        self,
        *,
        event: str,
        reason: str | None = None,
        severity: str = "info",
        metadata: Mapping[str, object] | None = None,
        now: float | None = None,
    ) -> None:
        """Capture diagnostic events for downstream summaries and metrics.

        Args:
            event: Canonical event name describing the diagnostic scenario.
            reason: Optional free-form reason to canonicalise for reporting.
            severity: Human readable severity indicator (info|warning|critical).
            metadata: Optional structured payload describing the event context.
            now: Optional timestamp override used for deterministic testing.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.record_diagnostic_event",
            extra={
                "event": "metrics_record_diagnostic_event",
                "diag_event": event,
                "reason": reason,
                "severity": severity,
            },
        )
        safe_event = (event or "unknown").strip() or "unknown"
        token = canonical(reason)
        normalized_severity = (severity or "info").strip().lower() or "info"
        reference_time = float(now if now is not None else time.time())
        try:
            self._diagnostic_events.labels(safe_event, token).inc()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in record_diagnostic_event counter: %s",
                exc,
                extra={
                    "event": "metrics_diagnostic_counter_error",
                    "diag_event": safe_event,
                    "reason": token,
                },
            )
        cleaned_meta: dict[str, object] = {}
        if metadata:
            for key, value in metadata.items():
                try:
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        cleaned_meta[str(key)] = value
                    else:
                        cleaned_meta[str(key)] = str(value)
                except Exception as exc:  # noqa: BLE001 - defensive
                    LOGGER.debug(
                        "diagnostic_event_metadata_coercion_failed",
                        exc_info=exc,
                        extra={
                            "event": "diagnostic_event_metadata_coercion_failed",
                            "diag_event": safe_event,
                            "key": key,
                        },
                    )
        with self._lock:
            self._diagnostic_recent.append(
                (reference_time, safe_event, token, normalized_severity, cleaned_meta)
            )
            self._purge_expired(reference_time)
        LOGGER.info(
            "Condition met: diagnostic_event_recorded",
            extra={
                "event": "diagnostic_event_recorded",
                "diag_event": safe_event,
                "reason": token,
                "severity": normalized_severity,
                "metadata": cleaned_meta,
            },
        )

    def record_queue_depth(
        self, *, symbol: str, depth: float, now: float | None = None
    ) -> None:
        """Store queue depth estimates for aggregate reporting.

        Args:
            symbol: Symbol associated with the queue estimate.
            depth: Estimated contracts queued ahead of the order.
            now: Optional timestamp override for deterministic testing.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.record_queue_depth",
            extra={"event": "metrics_record_queue_depth", "symbol": symbol},
        )
        safe_symbol = symbol or "UNKNOWN"
        clamped_depth = max(float(depth), 0.0)
        try:
            self._queue_depth.labels(safe_symbol).set(clamped_depth)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in MetricsCollector.record_queue_depth: %s",
                exc,
                extra={
                    "event": "queue_depth_metric_error",
                    "symbol": safe_symbol,
                },
            )
        reference_time = float(now if now is not None else time.time())
        with self._lock:
            self._queue_samples.append((reference_time, clamped_depth))
            self._purge_expired(reference_time)
        if clamped_depth > 0.0:
            LOGGER.info(
                "Condition met: queue_depth_observed",
                extra={
                    "event": "queue_depth_observed",
                    "symbol": safe_symbol,
                    "depth": clamped_depth,
                },
            )

    def set_live_pnl(self, *, book: str, value: float) -> None:
        """Update the live PnL gauge for *book* with *value*.

        Args:
            book: Identifier representing the PnL book being updated.
            value: Latest realised PnL value.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.set_live_pnl",
            extra={"event": "metrics_set_live_pnl", "book": book},
        )
        safe_book = book or "default"
        try:
            self._live_pnl.labels(safe_book).set(float(value))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in MetricsCollector.set_live_pnl: %s",
                exc,
                extra={"event": "live_pnl_metric_error", "book": safe_book},
            )
        LOGGER.info(
            "Condition met: live_pnl_updated",
            extra={
                "event": "live_pnl_updated",
                "book": safe_book,
                "value": float(value),
            },
        )

    def set_pnl_breakdown(
        self,
        *,
        book: str,
        realized: float | None = None,
        unrealized: float | None = None,
    ) -> None:
        """Update realized/unrealized PnL gauges for *book*.

        Args:
            book: Identifier representing the PnL book being updated.
            realized: Realized PnL component to record.
            unrealized: Unrealized PnL component to record.

        Returns:
            None.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered MetricsCollector.set_pnl_breakdown",
            extra={"event": "metrics_set_pnl_breakdown", "book": book},
        )
        safe_book = book or "default"
        with self._lock:
            entry = self._pnl_state.setdefault(
                safe_book, {"realized": 0.0, "unrealized": 0.0}
            )
            if realized is not None:
                entry["realized"] = float(realized)
            if unrealized is not None:
                entry["unrealized"] = float(unrealized)
        try:
            if realized is not None:
                self._pnl_breakdown.labels(safe_book, "realized").set(float(realized))
            if unrealized is not None:
                self._pnl_breakdown.labels(safe_book, "unrealized").set(
                    float(unrealized)
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in MetricsCollector.set_pnl_breakdown: %s",
                exc,
                extra={
                    "event": "pnl_breakdown_metric_error",
                    "book": safe_book,
                },
            )
        LOGGER.info(
            "Condition met: pnl_breakdown_updated",
            extra={
                "event": "pnl_breakdown_updated",
                "book": safe_book,
                "realized": None if realized is None else float(realized),
                "unrealized": None if unrealized is None else float(unrealized),
            },
        )

    # ------------------------------------------------------------------
    # Public snapshot helpers -----------------------------------------
    def snapshot_alerts(self) -> dict[str, dict[str, object]]:
        """Return a dictionary keyed by alert name describing current state."""

        with self._lock:
            return {name: state.as_dict() for name, state in self._alert_state.items()}

    def metric_summary(self) -> dict[str, object]:
        """Return summarised metrics used by the health endpoint."""

        with self._lock:
            tick_values = [value for _, value in self._tick_samples]
            order_values = [value for _, value in self._order_samples]
            stale_values = [value for _, value in self._stale_samples]
            errors_by_type: MutableMapping[str, int] = {}
            for _, error_type in self._error_events:
                errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
            queue_values = [value for _, value in self._queue_samples]
            rejections_by_reason: MutableMapping[str, int] = {}
            for _, reason in self._rejection_events:
                rejections_by_reason[reason] = rejections_by_reason.get(reason, 0) + 1
            recent_diag = list(self._diagnostic_recent)
            diagnostic_counts: MutableMapping[str, int] = {}
            for _, _event_name, token, _severity, _meta in recent_diag:
                diagnostic_counts[token] = diagnostic_counts.get(token, 0) + 1
            pnl_snapshot = dict(self._pnl_state)
            errors_by_minute: MutableMapping[str, int] = {}
            for timestamp, _reason in self._error_events:
                minute_key = datetime.fromtimestamp(
                    timestamp, tz=timezone.utc
                ).strftime("%H:%M")
                errors_by_minute[minute_key] = errors_by_minute.get(minute_key, 0) + 1
            flush_events = list(self._heartbeat_flush_events)
            last_success = self._last_successful_flush
            latest_regime = self._latest_regime

        return {
            "tick_latency_p95": self._percentile(tick_values, 0.95),
            "order_latency_p95": self._percentile(order_values, 0.95),
            "max_tick_staleness": max(stale_values) if stale_values else 0.0,
            "error_events_5m": dict(errors_by_type),
            "queue_depth_max": max(queue_values) if queue_values else 0.0,
            "queue_depth_avg": (
                sum(queue_values) / len(queue_values) if queue_values else 0.0
            ),
            "order_rejections_5m": dict(rejections_by_reason),
            "pnl_breakdown": pnl_snapshot,
            "error_events_per_minute": dict(errors_by_minute),
            "heartbeat_flushes_5m": self._summarise_flush_counts(flush_events),
            "heartbeat_last_latency": flush_events[-1][2] if flush_events else None,
            "heartbeat_flush_age": (
                max(0.0, time.time() - last_success)
                if last_success is not None
                else None
            ),
            "diagnostic_events_5m": dict(diagnostic_counts),
            "diagnostic_recent": [
                {
                    "timestamp": entry[0],
                    "event": entry[1],
                    "reason": entry[2],
                    "severity": entry[3],
                    "metadata": dict(entry[4]),
                }
                for entry in recent_diag[-20:]
            ],
            "current_regime": (
                {
                    "regime": latest_regime[0],
                    "confidence": latest_regime[1],
                    "updated_at": latest_regime[2],
                }
                if latest_regime is not None
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers -------------------------------------------------
    def _purge_expired(self, reference_time: float) -> None:
        cutoff_five_min = reference_time - 300.0
        cutoff_two_min = reference_time - 120.0
        with self._lock:
            while self._tick_samples and self._tick_samples[0][0] < cutoff_five_min:
                self._tick_samples.popleft()
            while self._order_samples and self._order_samples[0][0] < cutoff_five_min:
                self._order_samples.popleft()
            while self._stale_samples and self._stale_samples[0][0] < cutoff_two_min:
                self._stale_samples.popleft()
            while self._error_events and self._error_events[0][0] < cutoff_five_min:
                self._error_events.popleft()
            while self._queue_samples and self._queue_samples[0][0] < cutoff_five_min:
                self._queue_samples.popleft()
            while (
                self._rejection_events
                and self._rejection_events[0][0] < cutoff_five_min
            ):
                self._rejection_events.popleft()
            while (
                self._heartbeat_flush_events
                and self._heartbeat_flush_events[0][0] < cutoff_five_min
            ):
                self._heartbeat_flush_events.popleft()
            while (
                self._diagnostic_recent
                and self._diagnostic_recent[0][0] < cutoff_five_min
            ):
                self._diagnostic_recent.popleft()

    def _evaluate_alerts_locked(self, reference_time: float) -> None:
        with self._lock:
            order_values = [value for _, value in self._order_samples]
            tick_values = list(self._stale_samples)
            error_events = list(self._error_events)

        order_p95 = self._percentile(order_values, 0.95)
        order_active = order_p95 is not None and order_p95 > 0.5
        self._update_alert(
            name="OrderLatencyHigh",
            active=bool(order_active),
            severity="critical" if order_active else "info",
            observed=order_p95,
            threshold=0.5,
            triggered_at=reference_time if order_active else None,
            details={"window_seconds": 300, "p95": order_p95 or 0.0},
        )

        stale_active = False
        max_stale = None
        if tick_values:
            max_stale = max(value for _, value in tick_values)
            stale_active = all(value > 2.0 for _, value in tick_values)
        self._update_alert(
            name="TickStalenessCritical",
            active=stale_active,
            severity="critical" if stale_active else "info",
            observed=max_stale,
            threshold=2.0,
            triggered_at=reference_time if stale_active else None,
            details={"window_seconds": 120, "max": max_stale or 0.0},
        )

        error_count = len(error_events)
        burst_active = error_count > 10
        self._update_alert(
            name="ErrorBurst",
            active=burst_active,
            severity="warning" if burst_active else "info",
            observed=error_count,
            threshold=10,
            triggered_at=reference_time if burst_active else None,
            details={"window_seconds": 300, "count": error_count},
        )

    @staticmethod
    def _summarise_flush_counts(
        events: Iterable[Tuple[float, str, float]],
    ) -> dict[str, int]:
        """Return flush outcome counts keyed by outcome label.

        Args:
            events: Sequence of ``(timestamp, outcome, latency)`` tuples.

        Returns:
            dict[str, int]: Mapping of outcome labels to observation counts.

        Raises:
            None.
        """

        counts: Dict[str, int] = {}
        for _timestamp, outcome, _latency in events:
            counts[outcome] = counts.get(outcome, 0) + 1
        return counts

    def _update_alert(
        self,
        *,
        name: str,
        active: bool,
        severity: str,
        observed: float | int | None,
        threshold: float,
        triggered_at: float | None,
        details: Mapping[str, float | str],
    ) -> None:
        previous = self._alert_state.get(name)
        changed = previous is None or previous.active != active
        state = AlertState(
            name=name,
            active=active,
            severity=severity,
            observed=observed,
            threshold=threshold,
            triggered_at=triggered_at if active else None,
            details=dict(details),
        )
        self._alert_state[name] = state
        if changed:
            LOGGER.info(
                "Condition met: alert_state_changed",
                extra={
                    "event": "alert_state_changed",
                    "alert_name": name,
                    "active": active,
                    "severity": severity,
                },
            )
            for callback in list(self._notifiers):
                try:
                    callback(state)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error(
                        "Failure in MetricsCollector notifier for %s: %s",
                        name,
                        exc,
                        extra={
                            "event": "metrics_notifier_failure",
                            "alert_name": name,
                        },
                    )

    @staticmethod
    def _percentile(values: Iterable[float], percentile: float) -> float | None:
        dataset: list[float] = []
        for raw in values:
            try:
                numeric = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isnan(numeric):
                continue
            dataset.append(numeric)
        if not dataset:
            return None
        if len(dataset) == 1:
            return dataset[0]
        try:
            quantiles = statistics.quantiles(dataset, n=100, method="inclusive")
        except (statistics.StatisticsError, ValueError):
            return None
        index = max(min(int(percentile * 100) - 1, len(quantiles) - 1), 0)
        return quantiles[index]


METRICS = MetricsCollector()

__all__ = ["AlertState", "METRICS", "MetricsCollector"]
