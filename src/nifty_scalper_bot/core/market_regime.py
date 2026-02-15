"""Market regime detection and fan-out utilities for trading components."""

from __future__ import annotations

import queue
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, Mapping, Sequence, cast

import nifty_scalper_bot.config.settings as app_settings
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.risk.regime_sizing import RegimeSizer, RegimeType
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class RegimeSnapshot:
    """Point-in-time regime snapshot produced by the detector."""

    symbol: str
    regime: str
    confidence: float
    reason: str
    updated_at: float
    adjustments: Dict[str, object] = field(default_factory=dict)


RegimeListener = Callable[[RegimeSnapshot], None]
RegimeQueue = queue.SimpleQueue[RegimeSnapshot]


@dataclass(slots=True)
class RegimeParameters:
    """Thresholds controlling how market regimes are classified."""

    trend_score_trend: float = 0.6
    trend_score_range: float = 0.2
    ema_ratio_trend: float = 0.0015
    ema_ratio_flat: float = 0.0004
    atr_range_threshold: float = 12.0
    atr_volatile_threshold: float = 22.0
    atr_event_threshold: float = 35.0
    volume_trend_multiple: float = 1.15
    volume_event_multiple: float = 1.8
    calm_volume_multiple: float = 0.75
    iv_event_rank: float = 75.0
    price_momentum_trend: float = 0.004
    price_momentum_range: float = 0.0015
    realized_vol_volatile_threshold: float = 30.0
    realized_vol_calm_threshold: float = 15.0
    iv_realized_gap_event: float = 25.0
    adx_trend_threshold: float = 25.0
    adx_chop_threshold: float = 15.0
    price_momentum_volatile: float = 0.0045
    price_momentum_event: float = 0.007
    event_signal_threshold: float = 0.6
    realized_vol_event_threshold: float = 45.0


class MarketRegimeDetector:
    """Classify and broadcast market regimes for downstream consumers."""

    def __init__(
        self,
        parameters: RegimeParameters | None = None,
        *,
        history_limit: int = 200,
    ) -> None:
        """Initialise the detector with optional parameter overrides.

        Args:
            parameters: Optional custom threshold configuration.
            history_limit: Maximum number of snapshots retained in history.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger = LOGGER
        self._params = parameters or RegimeParameters()
        self._history: Deque[RegimeSnapshot] = deque(maxlen=max(10, history_limit))
        self._listeners: set[RegimeListener] = set()
        self._queues: weakref.WeakSet[RegimeQueue] = weakref.WeakSet()
        self._lock = threading.RLock()
        self._state: dict[str, RegimeSnapshot] = {}

    def subscribe(self, listener: RegimeListener) -> None:
        """Register *listener* to receive regime change notifications.

        Args:
            listener: Callback invoked when the regime changes.

        Returns:
            None.

        Raises:
            Exception: Propagated if registration fails.
        """

        self._logger.debug(
            "Entered MarketRegimeDetector.subscribe",
            extra={"event": "regime_subscribe"},
        )
        try:
            with self._lock:
                self._listeners.add(listener)
            self._logger.info(
                "Condition met: regime_listener_registered",
                extra={"event": "regime_listener_registered"},
            )
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in MarketRegimeDetector.subscribe: %s",
                exc,
                extra={"event": "regime_subscribe_error"},
            )
            raise

    def unsubscribe(self, listener: RegimeListener) -> None:
        """Remove *listener* from the registry.

        Args:
            listener: Previously registered callback to detach.

        Returns:
            None.

        Raises:
            Exception: Propagated if unregistration fails.
        """

        self._logger.debug(
            "Entered MarketRegimeDetector.unsubscribe",
            extra={"event": "regime_unsubscribe"},
        )
        try:
            with self._lock:
                self._listeners.discard(listener)
            self._logger.info(
                "Condition met: regime_listener_removed",
                extra={"event": "regime_listener_removed"},
            )
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in MarketRegimeDetector.unsubscribe: %s",
                exc,
                extra={"event": "regime_unsubscribe_error"},
            )
            raise

    def register_queue(self) -> RegimeQueue:
        """Return a thread-safe queue receiving regime change events.

        Args:
            None.

        Returns:
            Queue delivering :class:`RegimeSnapshot` entries.

        Raises:
            Exception: Propagated if queue registration fails.
        """

        self._logger.debug(
            "Entered MarketRegimeDetector.register_queue",
            extra={"event": "regime_register_queue"},
        )
        try:
            queue_ref: RegimeQueue = queue.SimpleQueue()
            with self._lock:
                self._queues.add(queue_ref)
            self._logger.info(
                "Condition met: regime_queue_registered",
                extra={"event": "regime_queue_registered"},
            )
            return queue_ref
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in MarketRegimeDetector.register_queue: %s",
                exc,
                extra={"event": "regime_register_queue_error"},
            )
            raise

    def unregister_queue(self, queue_ref: RegimeQueue) -> None:
        """Stop delivering regime events to *queue_ref*.

        Args:
            queue_ref: Queue previously returned by :meth:`register_queue`.

        Returns:
            None.

        Raises:
            Exception: Propagated if deregistration fails.
        """

        self._logger.debug(
            "Entered MarketRegimeDetector.unregister_queue",
            extra={"event": "regime_unregister_queue"},
        )
        try:
            with self._lock:
                self._queues.discard(queue_ref)
            self._logger.info(
                "Condition met: regime_queue_removed",
                extra={"event": "regime_queue_removed"},
            )
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in MarketRegimeDetector.unregister_queue: %s",
                exc,
                extra={"event": "regime_unregister_queue_error"},
            )
            raise

    def latest(self, limit: int = 5) -> list[RegimeSnapshot]:
        """Return the most recent regime snapshots up to *limit*.

        Args:
            limit: Maximum number of entries to fetch.

        Returns:
            List containing up to *limit* snapshots ordered oldest to newest.

        Raises:
            Exception: Propagated if retrieval fails.
        """

        self._logger.debug(
            "Entered MarketRegimeDetector.latest",
            extra={"event": "regime_latest"},
        )
        try:
            if limit <= 0:
                self._logger.info(
                    "Condition met: regime_latest_empty",
                    extra={"event": "regime_latest_empty"},
                )
                return []
            with self._lock:
                snapshots = list(self._history)[-limit:]
            self._logger.info(
                "Condition met: regime_latest_returned",
                extra={"event": "regime_latest_returned", "count": len(snapshots)},
            )
            return snapshots
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in MarketRegimeDetector.latest: %s",
                exc,
                extra={"event": "regime_latest_error"},
            )
            raise

    def evaluate(
        self,
        symbol: str,
        indicators: Mapping[str, float | Sequence[float] | None],
        snapshot: Mapping[str, float | Sequence[float]] | None = None,
    ) -> RegimeSnapshot:
        """Evaluate the current regime for *symbol* using *indicators*.

        Args:
            symbol: Instrument identifier under evaluation.
            indicators: Indicator payload including price, volatility, ADX, and volume.
            snapshot: Optional fallback with cached price/volatility indicators.

        Returns:
            Fresh :class:`RegimeSnapshot` describing detected state.

        Raises:
            Exception: Propagated if evaluation fails.
        """

        self._logger.debug(
            "Entered MarketRegimeDetector.evaluate",
            extra={"event": "regime_evaluate_enter", "symbol": symbol},
        )
        try:
            params = self._params
            ema_fast = self._extract_float(indicators, ("ema_fast", "ema_12"))
            ema_slow = self._extract_float(indicators, ("ema_slow", "ema_26"))
            trend_score = self._extract_float(indicators, ("trend_score", "adx"))
            atr = self._extract_float(indicators, ("atr", "atr_14"))
            volume_ratio = self._extract_float(indicators, ("volume_ratio",))
            iv_rank = self._extract_float(indicators, ("iv_rank", "iv_percentile"))
            realized_vol = self._extract_float(
                indicators,
                (
                    "realized_volatility",
                    "historical_volatility",
                    "realized_vol",
                    "rv",
                ),
            )
            price = self._extract_float(indicators, ("price", "close", "ltp"))
            price_series = self._extract_series(
                indicators,
                (
                    "price_series",
                    "close_series",
                    "price_history",
                    "close_history",
                ),
            )
            event_signal = self._extract_float(
                indicators, ("event_score", "event_flag")
            )
            if snapshot is not None:
                if ema_fast is None:
                    ema_fast = self._snapshot_float(
                        snapshot,
                        ("ema_fast", "ema_12", "fast_ema"),
                    )
                if ema_slow is None:
                    ema_slow = self._snapshot_float(
                        snapshot,
                        ("ema_slow", "ema_26", "slow_ema"),
                    )
                if trend_score is None:
                    trend_score = self._snapshot_float(
                        snapshot,
                        ("trend_score", "adx", "trend_strength"),
                    )
                if volume_ratio is None:
                    volume_ratio = self._snapshot_float(
                        snapshot,
                        ("volume_ratio", "volume_zscore", "vol_ratio"),
                    )
                if atr is None:
                    atr = self._snapshot_float(snapshot, ("atr", "atr_14", "atr_21"))
                if iv_rank is None:
                    iv_rank = self._snapshot_float(
                        snapshot,
                        ("iv_rank", "iv_percentile", "ivr"),
                    )
                if realized_vol is None:
                    realized_vol = self._snapshot_float(
                        snapshot,
                        (
                            "realized_volatility",
                            "historical_volatility",
                            "realized_vol",
                            "rv",
                        ),
                    )
                if price is None:
                    price = self._snapshot_float(
                        snapshot,
                        ("price", "close", "ltp"),
                    )
                if not price_series:
                    price_series = self._snapshot_series(
                        snapshot,
                        (
                            "price_series",
                            "close_series",
                            "price_history",
                            "close_history",
                        ),
                    )
                if event_signal is None:
                    event_signal = self._snapshot_float(
                        snapshot,
                        ("event_score", "event_flag", "event_signal"),
                    )
            price_momentum = self._price_momentum(price, price_series)
            regime, confidence, reason = self._classify(
                params,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                trend_score=trend_score,
                atr=atr,
                volume_ratio=volume_ratio,
                iv_rank=iv_rank,
                event_signal=event_signal,
                price_momentum=price_momentum,
                realized_vol=realized_vol,
            )
            adjustments = self._build_adjustments(regime)
            display_regime = adjustments.get("display_regime")
            snapshot_obj = RegimeSnapshot(
                symbol=symbol,
                regime=regime,
                confidence=confidence,
                reason=reason,
                updated_at=time.time(),
                adjustments=adjustments,
            )
            self._logger.debug(
                "Computed regime features",
                extra={
                    "event": "regime_features",
                    "symbol": symbol,
                    "regime": regime,
                    "display_regime": display_regime,
                    "confidence": confidence,
                    "trend_score": trend_score,
                    "atr": atr,
                    "volume_ratio": volume_ratio,
                    "iv_rank": iv_rank,
                    "realized_vol": realized_vol,
                    "price_momentum": price_momentum,
                },
            )
            previous = self._record_snapshot(snapshot_obj)
            if previous is None or previous.regime != regime:
                self._logger.info(
                    "Condition met: regime_changed",
                    extra={
                        "event": "regime_changed",
                        "symbol": symbol,
                        "from": getattr(previous, "regime", "none"),
                        "to": regime,
                        "display_regime": display_regime,
                        "confidence": confidence,
                    },
                )
            else:
                self._logger.debug(
                    "Condition met: regime_unchanged",
                    extra={
                        "event": "regime_unchanged",
                        "symbol": symbol,
                        "regime": regime,
                        "display_regime": display_regime,
                    },
                )
            return snapshot_obj
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in MarketRegimeDetector.evaluate: %s",
                exc,
                extra={"event": "regime_evaluate_error", "symbol": symbol},
            )
            raise

    def get_snapshot(self, symbol: str) -> RegimeSnapshot | None:
        """Return cached regime snapshot for *symbol*. Args: symbol. Returns: snapshot. Raises: None."""
        self._logger.debug(
            "Entered MarketRegimeDetector.get_snapshot",
            extra={"event": "regime_get_snapshot", "symbol": symbol},
        )
        try:
            with self._lock:
                return self._state.get(symbol)
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in MarketRegimeDetector.get_snapshot: %s",
                exc,
                extra={"event": "regime_get_snapshot_error", "symbol": symbol},
                exc_info=exc,
            )
            return None

    def get_latest_snapshot(self) -> RegimeSnapshot | None:
        """Args: none; Returns: latest regime snapshot; Raises: none."""
        try:
            with self._lock:
                if not self._history:
                    return None
                return self._history[-1]
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in MarketRegimeDetector.get_latest_snapshot: %s", exc
            )
            return None

    def ingest_tick(self, tick: Mapping[str, object]) -> RegimeSnapshot | None:
        """Process *tick* payloads and emit regime snapshots when available.

        Args:
            tick: Market data update containing either direct regime values or
                indicator fields required for evaluation.

        Returns:
            RegimeSnapshot | None: Snapshot emitted after processing or ``None``
            when insufficient data is provided.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered MarketRegimeDetector.ingest_tick",
            extra={"event": "regime_ingest_tick"},
        )
        try:
            symbol = self._tick_symbol(tick)
            if not symbol:
                self._logger.debug(
                    "Condition met: regime_tick_missing_symbol",
                    extra={"event": "regime_tick_missing_symbol"},
                )
                return None

            snapshot_payload = tick.get("regime_snapshot")
            if isinstance(snapshot_payload, Mapping):
                snapshot = self._snapshot_from_payload(symbol, snapshot_payload)
                if snapshot is None:
                    return None
                self._record_snapshot(snapshot)
                return snapshot

            direct_snapshot = self._snapshot_from_payload(symbol, tick)
            if direct_snapshot is not None:
                self._record_snapshot(direct_snapshot)
                return direct_snapshot

            indicators = self._extract_indicators_from_tick(tick)
            if not indicators:
                self._logger.debug(
                    "Condition met: regime_tick_insufficient_features",
                    extra={
                        "event": "regime_tick_insufficient_features",
                        "symbol": symbol,
                    },
                )
                return None

            previous = self.get_snapshot(symbol)
            reference: Dict[str, float | Sequence[float]] | None = None
            if previous is not None and isinstance(previous.adjustments, Mapping):
                reference_candidates: Dict[str, float | Sequence[float]] = {}
                for key in ("price", "close", "ltp"):
                    raw_value = previous.adjustments.get(key)
                    if raw_value is None:
                        continue
                    coerced = self._coerce_indicator_value(raw_value)
                    if coerced is not None:
                        reference_candidates[key] = coerced
                if reference_candidates:
                    reference = reference_candidates
            snapshot = self.evaluate(symbol, indicators, snapshot=reference)
            return snapshot
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in MarketRegimeDetector.ingest_tick: %s",
                exc,
                extra={"event": "regime_ingest_tick_error"},
                exc_info=exc,
            )
            return None

    def _snapshot_from_payload(
        self, symbol: str, payload: Mapping[str, object]
    ) -> RegimeSnapshot | None:
        """Return snapshot parsed from *payload* when regime data exists.

        Args:
            symbol: Symbol identifier extracted from the tick.
            payload: Mapping containing potential regime fields.

        Returns:
            RegimeSnapshot | None: Parsed snapshot or ``None`` when missing.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered MarketRegimeDetector._snapshot_from_payload",
            extra={"event": "regime_snapshot_from_payload", "symbol": symbol},
        )
        try:
            regime = payload.get("regime")
            confidence_value = payload.get("confidence")
            if not isinstance(regime, str) or not regime:
                return None
            confidence = MarketRegimeDetector._coerce_float(confidence_value)
            if confidence is None:
                return None
            reason = str(payload.get("reason") or "tick_update")
            updated_raw = payload.get("updated_at")
            updated_at = time.time()
            coerced_updated = MarketRegimeDetector._coerce_float(updated_raw)
            if coerced_updated is not None:
                updated_at = coerced_updated
            adjustments: Dict[str, object] = {}
            raw_adjustments = payload.get("adjustments")
            if isinstance(raw_adjustments, Mapping):
                adjustments = dict(raw_adjustments)
            snapshot = RegimeSnapshot(
                symbol=symbol,
                regime=regime,
                confidence=confidence,
                reason=reason,
                updated_at=updated_at,
                adjustments=adjustments,
            )
            self._logger.debug(
                "Condition met: regime_snapshot_from_tick",
                extra={
                    "event": "regime_snapshot_from_tick",
                    "symbol": symbol,
                    "regime": regime,
                    "confidence": confidence,
                },
            )
            return snapshot
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in MarketRegimeDetector._snapshot_from_payload: %s",
                exc,
                extra={"event": "regime_snapshot_from_payload_error", "symbol": symbol},
                exc_info=exc,
            )
            return None

    def _extract_indicators_from_tick(
        self, tick: Mapping[str, object]
    ) -> Dict[str, float | Sequence[float] | None]:
        """Extract indicator fields from raw *tick* payloads.

        Args:
            tick: Raw market data update.

        Returns:
            Dict[str, float | Sequence[float] | None]: Indicator payload passed to
            :meth:`evaluate` or an empty mapping when unavailable.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered MarketRegimeDetector._extract_indicators_from_tick",
            extra={"event": "regime_extract_indicators"},
        )
        indicators: Dict[str, float | Sequence[float] | None] = {}
        try:
            indicator_keys = (
                "trend_score",
                "adx",
                "atr",
                "volume_ratio",
                "iv_rank",
                "event_score",
                "event_flag",
                "realized_volatility",
                "historical_volatility",
                "price",
                "close",
                "ltp",
                "price_series",
                "close_series",
            )
            for key in indicator_keys:
                if key not in tick:
                    continue
                value = tick.get(key)
                if value is None:
                    indicators[key] = None
                    continue
                coerced = self._coerce_indicator_value(value)
                if coerced is not None:
                    indicators[key] = coerced
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in MarketRegimeDetector._extract_indicators_from_tick: %s",
                exc,
                extra={"event": "regime_extract_indicators_error"},
                exc_info=exc,
            )
            return {}
        return indicators

    @staticmethod
    def _coerce_indicator_value(
        value: object,
    ) -> float | tuple[float, ...] | None:
        """Normalise *value* into scalar or sequence indicator payloads.

        Args:
            value: Raw indicator value extracted from ticks or snapshots.

        Returns:
            Either a ``float`` or tuple of ``float`` values when conversion is
            successful. ``None`` is returned when coercion is not possible.

        Raises:
            None.
        """

        if value is None:
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            normalised: list[float] = []
            for item in value:
                if isinstance(item, (int, float, bool)):
                    normalised.append(float(item))
                    continue
                try:
                    normalised.append(float(item))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return None
            return tuple(normalised)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _tick_symbol(tick: Mapping[str, object]) -> str:
        """Return symbol identifier from *tick* payload.

        Args:
            tick: Raw market data update.

        Returns:
            str: Normalised symbol string or an empty string when absent.

        Raises:
            None.
        """

        try:
            symbol_candidate = tick.get("symbol") or tick.get("tradingsymbol")
            if symbol_candidate is None:
                return ""
            return str(symbol_candidate).strip()
        except Exception:
            return ""

    def _record_snapshot(self, snapshot: RegimeSnapshot) -> RegimeSnapshot | None:
        """Persist *snapshot* and broadcast when regime changes.

        Args:
            snapshot: Snapshot to persist and fan-out.

        Returns:
            Previously stored snapshot for the same symbol when present.

        Raises:
            Exception: Propagated if persistence fails.
        """

        self._logger.debug(
            "Entered MarketRegimeDetector._record_snapshot",
            extra={"event": "regime_record_enter", "symbol": snapshot.symbol},
        )
        previous: RegimeSnapshot | None = None
        try:
            with self._lock:
                previous = self._state.get(snapshot.symbol)
                self._state[snapshot.symbol] = snapshot
                self._history.append(snapshot)
                listeners = list(self._listeners)
                queues = list(self._queues)
            regime_changed = previous is None or previous.regime != snapshot.regime
            try:
                METRICS.update_regime_confidence(
                    regime=snapshot.regime,
                    confidence=snapshot.confidence,
                    now=snapshot.updated_at,
                )
            except Exception as exc:  # noqa: BLE001 - metrics safety
                self._logger.error(
                    "Failure in regime confidence metric publish: %s",
                    exc,
                    extra={"event": "regime_confidence_metric_error"},
                )
            if regime_changed:
                try:
                    METRICS.record_regime_change(
                        regime=snapshot.regime,
                        confidence=snapshot.confidence,
                        event_id=f"{snapshot.symbol}:{snapshot.updated_at:.6f}",
                        now=snapshot.updated_at,
                    )
                except Exception as exc:  # noqa: BLE001 - metrics safety
                    self._logger.error(
                        "Failure in regime metrics publish: %s",
                        exc,
                        extra={"event": "regime_metric_failure"},
                    )
            for listener in listeners:
                try:
                    listener(snapshot)
                except Exception as exc:  # noqa: BLE001 - defensive
                    self._logger.error(
                        "Failure in regime listener: %s",
                        exc,
                        extra={
                            "event": "regime_listener_failure",
                            "symbol": snapshot.symbol,
                            "listener": getattr(listener, "__name__", str(listener)),
                        },
                    )
            for queue_ref in queues:
                try:
                    queue_ref.put(snapshot)
                except Exception as exc:  # noqa: BLE001 - defensive
                    self._logger.error(
                        "Failure in regime queue delivery: %s",
                        exc,
                        extra={
                            "event": "regime_queue_failure",
                            "symbol": snapshot.symbol,
                        },
                    )
            return previous
        except Exception as exc:  # noqa: BLE001 - defensive
            self._logger.error(
                "Failure in MarketRegimeDetector._record_snapshot: %s",
                exc,
                extra={"event": "regime_record_error", "symbol": snapshot.symbol},
            )
            raise

    @staticmethod
    def _aggregate_components(
        components: Sequence[tuple[float, str]],
        *,
        default_reason: str,
    ) -> tuple[float, str]:
        """Return aggregate score and dominant reason for *components*.

        Args:
            components: Sequence of ``(score, reason)`` tuples to aggregate.
            default_reason: Fallback reason when no component has signal.

        Returns:
            Tuple of ``(score, reason)`` for downstream classification logic.

        Raises:
            None.
        """

        if not components:
            return 0.0, default_reason
        positive_scores = [max(0.0, value) for value, _ in components]
        score = sum(positive_scores) / len(components)
        top_value, top_reason = max(components, key=lambda item: item[0])
        if top_value <= 0.0:
            top_reason = default_reason
        return max(0.0, score), top_reason

    def _compute_feature_scores(
        self,
        params: RegimeParameters,
        *,
        adx: float | None,
        realized_vol: float | None,
        volume_ratio: float | None,
        price_momentum: float | None,
        event_signal: float | None,
    ) -> tuple[Dict[str, float], Dict[str, str]]:
        """Return weighted scores describing the likelihood of each regime.

        Args:
            params: Tunable thresholds guiding the classification logic.
            adx: Average directional index or normalised trend score.
            realized_vol: Realised volatility expressed as percent or ratio.
            volume_ratio: Relative volume metric compared to average activity.
            price_momentum: Normalised price momentum derived from prices.
            event_signal: Optional external event score between 0 and 1.

        Returns:
            Tuple containing score and reason mappings keyed by regime label.

        Raises:
            None.
        """

        adx_norm = self._normalise_adx(adx)
        realized_pct = self._normalise_vol(realized_vol)
        volume = float(volume_ratio) if volume_ratio is not None else None
        momentum = abs(price_momentum) if price_momentum is not None else None

        event_components: list[tuple[float, str]] = []
        if event_signal is not None:
            strength = max(0.0, float(event_signal) - params.event_signal_threshold)
            if strength > 0.0:
                event_components.append((0.8 + strength, "event_signal"))
        if momentum is not None and volume is not None and realized_pct is not None:
            momentum_ratio = momentum / max(params.price_momentum_event, 1e-6)
            volume_ratio_norm = volume / max(params.volume_event_multiple, 1e-6)
            realized_ratio = realized_pct / max(
                params.realized_vol_event_threshold, 1e-6
            )
            combo = (
                0.45 * momentum_ratio + 0.3 * volume_ratio_norm + 0.25 * realized_ratio
            )
            if combo > 0.0:
                event_components.append((combo, "momentum_volume_realized"))
        event_score, event_reason = self._aggregate_components(
            event_components, default_reason="insufficient_event_features"
        )

        volatile_components: list[tuple[float, str]] = []
        if realized_pct is not None:
            volatile_components.append(
                (
                    realized_pct / max(params.realized_vol_volatile_threshold, 1e-6),
                    "realized_vol_high",
                )
            )
        if momentum is not None:
            volatile_components.append(
                (
                    momentum
                    / max(
                        params.price_momentum_volatile,
                        params.price_momentum_trend,
                        1e-6,
                    ),
                    "momentum_spike",
                )
            )
        if volume is not None and volume > 1.0:
            volatile_components.append(
                (
                    (volume - 1.0) / max(params.volume_trend_multiple, 1.0),
                    "volume_spike",
                )
            )
        volatile_score, volatile_reason = self._aggregate_components(
            volatile_components, default_reason="insufficient_volatility_features"
        )

        trend_components: list[tuple[float, str]] = []
        if adx_norm is not None:
            spread = max(params.adx_trend_threshold - params.adx_chop_threshold, 1.0)
            trend_components.append(
                (
                    max(0.0, (adx_norm - params.adx_chop_threshold) / spread),
                    "adx_trending",
                )
            )
        if momentum is not None:
            trend_components.append(
                (
                    momentum / max(params.price_momentum_trend, 1e-6),
                    "momentum_trending",
                )
            )
        if volume is not None and volume >= 1.0:
            trend_components.append(
                (
                    volume / max(params.volume_trend_multiple, 1e-6),
                    "volume_confirmation",
                )
            )
        trend_score, trend_reason = self._aggregate_components(
            trend_components, default_reason="insufficient_trend_features"
        )

        range_components: list[tuple[float, str]] = []
        if adx_norm is not None and params.adx_trend_threshold > 0:
            range_components.append(
                (
                    max(
                        0.0,
                        (params.adx_trend_threshold - adx_norm)
                        / max(params.adx_trend_threshold, 1e-6),
                    ),
                    "adx_low",
                )
            )
        if realized_pct is not None and params.realized_vol_calm_threshold > 0:
            range_components.append(
                (
                    max(
                        0.0,
                        (params.realized_vol_calm_threshold - realized_pct)
                        / max(params.realized_vol_calm_threshold, 1e-6),
                    ),
                    "realized_vol_calm",
                )
            )
        if momentum is not None and params.price_momentum_range > 0:
            range_components.append(
                (
                    max(
                        0.0,
                        (params.price_momentum_range - momentum)
                        / max(params.price_momentum_range, 1e-6),
                    ),
                    "momentum_flat",
                )
            )
        if volume is not None:
            calm_multiple = max(params.calm_volume_multiple, 0.1)
            range_components.append(
                (
                    max(0.0, (calm_multiple - volume) / calm_multiple),
                    "volume_calm",
                )
            )
        range_score, range_reason = self._aggregate_components(
            range_components, default_reason="insufficient_range_features"
        )

        scores = {
            "event": max(0.0, event_score),
            "volatile": max(0.0, volatile_score),
            "trend": max(0.0, trend_score),
            "range": max(0.0, range_score),
        }
        reasons = {
            "event": event_reason,
            "volatile": volatile_reason,
            "trend": trend_reason,
            "range": range_reason,
        }
        return scores, reasons

    def _classify(
        self,
        params: RegimeParameters,
        *,
        ema_fast: float | None,
        ema_slow: float | None,
        trend_score: float | None,
        atr: float | None,
        volume_ratio: float | None,
        iv_rank: float | None,
        event_signal: float | None,
        price_momentum: float | None,
        realized_vol: float | None,
    ) -> tuple[str, float, str]:
        """Return ``(regime, confidence, reason)`` for the current context.

        Args:
            params: Tunable thresholds guiding the classification logic.
            ema_fast: Fast EMA value when available.
            ema_slow: Slow EMA value when available.
            trend_score: Trend or momentum indicator score (e.g. ADX).
            atr: Average true range value for volatility measurement.
            volume_ratio: Relative volume metric compared to average.
            iv_rank: Implied volatility rank or percentile.
            event_signal: Optional event score normalised between 0 and 1.
            price_momentum: Normalised price momentum derived from recent data.
            realized_vol: Realised volatility expressed as a percentage.

        Returns:
            Tuple describing the detected regime, its confidence, and reason.

        Raises:
            None.
        """

        scores, reasons = self._compute_feature_scores(
            params,
            adx=trend_score,
            realized_vol=realized_vol,
            volume_ratio=volume_ratio,
            price_momentum=price_momentum,
            event_signal=event_signal,
        )
        realized_pct = self._normalise_vol(realized_vol)
        adx_norm = self._normalise_adx(trend_score)
        volume = float(volume_ratio) if volume_ratio is not None else None
        momentum = abs(price_momentum) if price_momentum is not None else None

        if (
            iv_rank is not None
            and realized_pct is not None
            and (iv_rank - realized_pct) >= params.iv_realized_gap_event
        ):
            confidence = min(1.0, 0.75 + (iv_rank - realized_pct) / 120.0)
            return "event", round(confidence, 4), "iv_realized_gap"

        if event_signal is not None and event_signal >= params.event_signal_threshold:
            confidence = min(
                1.0,
                0.75 + (event_signal - params.event_signal_threshold) * 0.25,
            )
            return "event", round(confidence, 4), "event_signal_triggered"

        if atr is not None and atr >= params.atr_event_threshold:
            confidence = min(1.0, 0.75 + (atr - params.atr_event_threshold) / 60.0)
            return "event", round(confidence, 4), "atr_event_threshold"

        if (
            (volume is not None and volume >= params.volume_event_multiple)
            and momentum is not None
            and realized_pct is not None
        ):
            confidence = min(
                1.0,
                0.72
                + min(
                    scores.get("event", 0.0)
                    + momentum / max(params.price_momentum_event, 1e-6),
                    2.5,
                )
                * 0.12,
            )
            return "event", round(confidence, 4), "volume_price_spike"

        if scores.get("event", 0.0) >= 1.0:
            confidence = min(1.0, 0.72 + min(scores["event"], 2.5) * 0.12)
            return "event", round(confidence, 4), reasons["event"]

        volatile_score = scores.get("volatile", 0.0)
        if atr is not None and atr >= params.atr_volatile_threshold:
            boost = 1.0 + max(0.0, atr - params.atr_volatile_threshold) / max(
                params.atr_volatile_threshold, 1.0
            )
            volatile_score = max(volatile_score, boost)
            reasons["volatile"] = "atr_above_volatile"
        elif (
            realized_pct is not None
            and realized_pct >= params.realized_vol_volatile_threshold
        ):
            volatile_score = max(
                volatile_score,
                realized_pct / max(params.realized_vol_volatile_threshold, 1e-6),
            )
            reasons["volatile"] = "realized_vol_high"

        trend_score_value = scores.get("trend", 0.0)
        if (
            adx_norm is not None
            and adx_norm >= params.adx_trend_threshold
            and momentum is not None
            and momentum >= params.price_momentum_range
        ):
            trend_score_value = max(
                trend_score_value,
                (adx_norm / max(params.adx_trend_threshold, 1e-6))
                + momentum / max(params.price_momentum_trend, 1e-6),
            )
            reasons["trend"] = "adx_momentum_alignment"
        elif (
            ema_fast is not None
            and ema_slow is not None
            and ema_slow != 0.0
            and abs((ema_fast - ema_slow) / abs(ema_slow)) >= params.ema_ratio_trend
        ):
            trend_score_value = max(
                trend_score_value,
                abs((ema_fast - ema_slow) / abs(ema_slow))
                / max(params.ema_ratio_trend, 1e-6),
            )
            reasons["trend"] = "ema_divergence"

        range_score_value = scores.get("range", 0.0)
        if (
            atr is not None
            and atr <= params.atr_range_threshold
            and (adx_norm is None or adx_norm <= params.adx_chop_threshold)
        ):
            range_score_value = max(range_score_value, 1.0)
            reasons["range"] = "atr_within_range"

        scored = {
            "volatile": volatile_score,
            "trend": trend_score_value,
            "range": max(range_score_value, 0.3),
        }
        regime = max(scored, key=lambda name: scored.get(name, 0.0))
        reason = reasons.get(regime, "insufficient_features")

        base_confidence = {"volatile": 0.62, "trend": 0.6, "range": 0.55}
        multiplier = {"volatile": 0.12, "trend": 0.12, "range": 0.08}
        intensity = min(max(scored.get(regime, 0.0), 0.0), 3.0)
        confidence = base_confidence[regime] + intensity * multiplier[regime]
        confidence = min(1.0, confidence)
        return regime, round(confidence, 4), reason

    @staticmethod
    def _extract_float(
        indicators: Mapping[str, float | Sequence[float] | None],
        keys: Iterable[str],
    ) -> float | None:
        """Return the first numeric indicator among *keys* if present.

        Args:
            indicators: Mapping of indicator names to numeric payloads.
            keys: Ordered sequence of keys inspected for numeric values.

        Returns:
            Optional float extracted from the indicators.

        Raises:
            None.
        """

        for key in keys:
            value = indicators.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, Sequence):
                if value:
                    head = value[0]
                    if isinstance(head, (int, float)):
                        return float(head)
        return None

    @staticmethod
    def _extract_series(
        indicators: Mapping[str, float | Sequence[float] | None],
        keys: Iterable[str],
    ) -> list[float]:
        """Return numeric series from *indicators* for the first matching key.

        Args:
            indicators: Mapping of indicator names to numeric payloads.
            keys: Ordered sequence of keys inspected for series data.

        Returns:
            List of floats extracted from the indicators.

        Raises:
            None.
        """

        for key in keys:
            value = indicators.get(key)
            if isinstance(value, Sequence):
                numeric = [
                    float(item) for item in value if isinstance(item, (int, float))
                ]
                if numeric:
                    return numeric
        return []

    @staticmethod
    def _coerce_float(value: object | None) -> float | None:
        """Return ``float(value)`` when *value* is numeric.

        Args:
            value: Potential numeric value.

        Returns:
            Optional float produced from *value*.

        Raises:
            None.
        """

        if value is None:
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _snapshot_float(
        snapshot: Mapping[str, object],
        keys: Iterable[str],
    ) -> float | None:
        """Extract optional float from *snapshot* using ordered *keys*.

        Args:
            snapshot: Mapping containing cached numeric values.
            keys: Sequence of candidate keys inspected for numeric payloads.

        Returns:
            Optional float parsed from the snapshot mapping.

        Raises:
            None.
        """

        for key in keys:
            value = snapshot.get(key)
            if value is None:
                continue
            coerced = MarketRegimeDetector._coerce_float(cast(object, value))
            if coerced is not None:
                return coerced
        return None

    @staticmethod
    def _snapshot_series(
        snapshot: Mapping[str, object],
        keys: Iterable[str],
    ) -> list[float]:
        """Extract numeric series from *snapshot* when available.

        Args:
            snapshot: Mapping containing cached numeric values.
            keys: Candidate keys referencing list-like payloads.

        Returns:
            List of floats extracted from the snapshot or empty list when missing.

        Raises:
            None.
        """

        for key in keys:
            value = snapshot.get(key)
            if isinstance(value, Sequence):
                numeric = [
                    float(item) for item in value if isinstance(item, (int, float))
                ]
                if numeric:
                    return numeric
        return []

    @staticmethod
    def _price_momentum(price: float | None, history: Sequence[float]) -> float | None:
        """Return normalised price momentum using *history* and optional *price*.

        Args:
            price: Latest traded price when available.
            history: Sequence of recent prices ordered oldest to newest.

        Returns:
            Optional float representing percentage change (0.0 == flat).

        Raises:
            None.
        """

        data = [float(value) for value in history if isinstance(value, (int, float))]
        if price is not None:
            data.append(float(price))
        if len(data) < 2:
            return None
        start = data[0]
        end = data[-1]
        if start == 0:
            return None
        return (end - start) / abs(start)

    @staticmethod
    def _normalise_adx(value: float | None) -> float | None:
        """Return ADX on a 0-100 scale regardless of the input range.

        Args:
            value: Raw ADX measurement, e.g. 0-1 ratio or 0-100 reading.

        Returns:
            Normalised ADX value on a 0-100 scale or ``None`` when invalid.

        Raises:
            None.
        """

        if value is None:
            return None
        adx_value = float(value)
        if adx_value < 0.0:
            return None
        if adx_value <= 1.0:
            return max(0.0, adx_value * 100.0)
        if adx_value > 100.0:
            return 100.0
        return adx_value

    @staticmethod
    def _normalise_vol(value: float | None) -> float | None:
        """Return volatility in percentage terms regardless of the input scale.

        Args:
            value: Raw volatility metric (e.g. 0.25 or 25).

        Returns:
            Float in percentage terms suitable for comparisons or ``None``.

        Raises:
            None.
        """

        if value is None:
            return None
        vol = float(value)
        if vol < 0:
            return None
        if vol <= 1.0:
            return vol * 100.0
        return vol

    def _build_adjustments(self, regime: str) -> Dict[str, object]:
        """Return sizing and bias adjustments for *regime*.

        Args:
            regime: Regime label returned by :meth:`evaluate`.

        Returns:
            Mapping with sizing multiplier, bias, and blocklist guidance.

        Raises:
            None.
        """

        sizer = RegimeSizer(
            trend_mult=app_settings.REGIME_TREND_SIZING_MULT,
            range_mult=app_settings.REGIME_RANGE_SIZING_MULT,
            volatile_mult=app_settings.REGIME_VOLATILE_SIZING_MULT,
            event_mult=app_settings.REGIME_EVENT_SIZING_MULT,
        )
        multipliers = {
            RegimeType.TREND: sizer.apply_regime_adjustment(1.0, RegimeType.TREND),
            RegimeType.RANGE: sizer.apply_regime_adjustment(1.0, RegimeType.RANGE),
            RegimeType.VOLATILE: sizer.apply_regime_adjustment(
                1.0, RegimeType.VOLATILE
            ),
            RegimeType.EVENT: sizer.apply_regime_adjustment(1.0, RegimeType.EVENT),
            RegimeType.UNKNOWN: sizer.apply_regime_adjustment(1.0, RegimeType.UNKNOWN),
        }
        mapping: Dict[str, Dict[str, object]] = {
            "trend": {
                "bias": "trend",
                "sizing_multiplier": multipliers[RegimeType.TREND],
                "blocklist": ["mean_reversion"],
                "display_regime": "trending",
            },
            "range": {
                "bias": "neutral",
                "sizing_multiplier": multipliers[RegimeType.RANGE],
                "blocklist": ["breakout"],
                "display_regime": "ranging/choppy",
            },
            "volatile": {
                "bias": "volatility",
                "sizing_multiplier": multipliers[RegimeType.VOLATILE],
                "blocklist": ["mean_reversion", "scalper"],
                "display_regime": "volatile",
            },
            "event": {
                "bias": "protect",
                "sizing_multiplier": multipliers[RegimeType.EVENT],
                "blocklist": ["trend_follow", "mean_reversion"],
                "display_regime": "event-driven",
            },
        }
        base = mapping.get(
            regime,
            {
                "bias": "neutral",
                "sizing_multiplier": multipliers[RegimeType.UNKNOWN],
                "blocklist": [],
                "display_regime": "unknown",
            },
        )
        return dict(base)


__all__ = ["MarketRegimeDetector", "RegimeParameters", "RegimeSnapshot"]
