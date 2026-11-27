"""Central market regime coordinator with gating and diagnostics.

Responsibilities: refresh timestamped underlying indicators and completed
in-memory price history, stabilise directional transitions on distinct adjacent
completed observations, and explain the accepted regime evidence. The live
path uses the indicator owner's completed spot snapshot and aligned futures volume.

Safety: Missing, stale, future-dated, UNKNOWN or LOW_ACTIVITY raw evidence
cannot inherit trade permission from an older stable regime. Raw EVENT and
VOLATILE thresholds are checked at the latest observation; execution routing
uses current safety regimes and confirmed TREND/RANGE transitions.
A low ATR alone cannot license mean-reversion entries. Risk and broker gates
remain owned by their execution components.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Mapping, MutableMapping
from zoneinfo import ZoneInfo

from nifty_scalper_bot.config.regime_ontology import MarketRegime, normalize_regime
from nifty_scalper_bot.core.market_regime import (
    MarketRegimeDetector,
    RegimeSnapshot,
)
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.instruments.active_contracts import (
    canonical_nifty_future_symbol,
)
from nifty_scalper_bot.utils.env import coalesce_bool, coalesce_float, coalesce_int
from nifty_scalper_bot.utils.logging import get_logger, log_throttled

logger = get_logger(__name__)


@dataclass(slots=True)
class RegimeDecision:
    """Structured filter decision emitted by :class:`MarketRegimeManager`."""

    allowed: bool
    reasons: tuple[str, ...]
    bypassed: bool
    decided_at: float
    regime: str | None = None
    confidence: float | None = None


@dataclass(slots=True)
class MarketRegimeManager:
    """Coordinate regime state, gating decisions, and diagnostics."""

    detector: MarketRegimeDetector
    history_limit: int = 64
    min_confidence: float = 0.40
    stale_after_seconds: float = 300.0
    transition_confirmations: int = 2
    block_thresholds: MutableMapping[str, float] = field(
        default_factory=lambda: {"event": 0.8, "volatile": 0.95}
    )
    fail_closed: bool = True
    datahub: Any | None = None
    indicators: Any | None = None
    regime_settings: Mapping[str, Any] | None = None
    _lock: threading.RLock = field(init=False, repr=False)
    _history: Deque[RegimeSnapshot] = field(init=False, repr=False)
    _decisions: Deque[RegimeDecision] = field(init=False, repr=False)
    _current: RegimeSnapshot | None = field(init=False, default=None, repr=False)
    _raw_current: RegimeSnapshot | None = field(init=False, default=None, repr=False)
    _pending_regime: str | None = field(init=False, default=None, repr=False)
    _pending_count: int = field(init=False, default=0, repr=False)
    _pending_evidence_at: float | None = field(init=False, default=None, repr=False)
    _last_verified_evidence_at: float | None = field(init=False, default=None, repr=False)
    _last_filter_reasons: tuple[str, ...] = field(
        init=False, default_factory=tuple, repr=False
    )
    _bypass: bool = field(init=False, default=False, repr=False)
    _listener_registered: bool = field(init=False, default=False, repr=False)
    _stats: MutableMapping[str, int] = field(
        init=False, default_factory=dict, repr=False
    )
    _last_logged_decision: tuple[bool, tuple[str, ...], bool] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _last_logged_at: float = field(init=False, default=0.0, repr=False)
    _log_cooldown: float = field(init=False, default=12.0, repr=False)
    _indicator_settings: dict[str, Any] = field(
        init=False, repr=False, default_factory=dict
    )
    _indicator_lock: asyncio.Lock = field(init=False, repr=False)
    _indicator_update_interval: float = field(init=False, repr=False, default=0.0)
    _indicator_symbol: str = field(init=False, repr=False, default="NIFTY")
    _last_indicator_refresh: float = field(init=False, repr=False, default=0.0)

    # ✅ FIXED: Missing field that caused the crash
    _refresh_task_started: bool = field(init=False, default=False, repr=False)

    # ✅ FIX: Add indicators_ready field (required by startup_sequence)
    indicators_ready: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        """Initialise shared state and subscribe to detector updates."""
        settings: dict[str, Any] = {}
        try:
            if self.regime_settings is not None:
                settings = dict(self.regime_settings)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.__post_init__ settings: %s",
                exc,
                extra={"event": "regime_manager_settings_error"},
            )
            settings = {}

        self._indicator_settings = settings
        history_override = settings.get("history_length")
        if history_override is not None:
            try:
                self.history_limit = max(5, int(history_override))
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failure in MarketRegimeManager.__post_init__ history override: %s",
                    exc,
                    extra={"event": "regime_manager_history_override_error"},
                )

        transition_override = settings.get("transition_confirmations")
        if transition_override is not None:
            try:
                self.transition_confirmations = max(1, int(transition_override))
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failure in MarketRegimeManager transition override: %s",
                    exc,
                    extra={"event": "regime_manager_transition_override_error"},
                )

        self._lock = threading.RLock()

        # ✅ FIXED: Ensure this is initialized
        self._refresh_task_started = False

        self._history: Deque[RegimeSnapshot] = deque(maxlen=max(5, self.history_limit))
        self._decisions: Deque[RegimeDecision] = deque(maxlen=200)
        self._current: RegimeSnapshot | None = None
        self._raw_current: RegimeSnapshot | None = None
        self._pending_regime = None
        self._pending_count = 0
        self._last_filter_reasons: tuple[str, ...] = tuple()
        self._bypass = False
        self._listener_registered = False
        self._stats: MutableMapping[str, int] = {"pass": 0, "block": 0, "bypass": 0}
        self._last_logged_decision = None
        self._last_logged_at = 0.0

        self._apply_env_overrides()

        self._indicator_lock = asyncio.Lock()
        self._last_indicator_refresh = 0.0
        try:
            interval_setting = settings.get("update_interval_sec")
            self._indicator_update_interval = (
                float(interval_setting) if interval_setting is not None else 0.0
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.__post_init__ interval: %s",
                exc,
                extra={"event": "regime_manager_interval_error"},
            )
            self._indicator_update_interval = 0.0

        symbol_setting = settings.get("symbol") or "NIFTY"
        try:
            self._indicator_symbol = str(symbol_setting)
        except Exception:  # pragma: no cover - defensive cast
            self._indicator_symbol = "NIFTY"

        self._register_listener()
        self._bootstrap_state()

    # ✅ FIXED: New explicit start method
    async def start(self) -> None:
        """Explicitly start the background refresh loop."""
        if self.indicators is not None and self._indicator_update_interval > 0.0:
            with self._lock:
                if self._refresh_task_started:
                    return
                self._refresh_task_started = True

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._run_indicator_refresh_loop())
                logger.info("✅ MarketRegimeManager: Refresh loop started via start().")
            except Exception as exc:
                with self._lock:
                    self._refresh_task_started = False
                logger.error(
                    f"❌ Failed to start regime refresh loop: {exc}", exc_info=True
                )
                raise

    # ------------------------------------------------------------------
    def build_diagnostics(self) -> dict[str, object]:
        """Return structured diagnostics for operators."""
        logger.debug(
            "Entered MarketRegimeManager.build_diagnostics",
            extra={"event": "regime_manager_build_diag"},
        )
        try:
            snapshot = self.get_latest_snapshot()
            raw_snapshot = self.get_raw_snapshot()
            with self._lock:
                pending_regime = self._pending_regime
                pending_count = self._pending_count
            history = self.get_history(limit=10)
            decisions = self.get_decision_history(limit=20)
            history_payload = [
                {
                    "regime": snap.regime,
                    "confidence": snap.confidence,
                    "reason": snap.reason,
                    "updated_at": snap.updated_at,
                }
                for snap in history
            ]
            decisions_payload = [
                {
                    "allowed": entry.allowed,
                    "reasons": entry.reasons,
                    "bypassed": entry.bypassed,
                    "regime": entry.regime,
                    "confidence": entry.confidence,
                    "decided_at": entry.decided_at,
                }
                for entry in decisions
            ]
            return {
                "current": {
                    "regime": getattr(snapshot, "regime", None),
                    "confidence": getattr(snapshot, "confidence", None),
                    "updated_at": getattr(snapshot, "updated_at", None),
                    "reason": getattr(snapshot, "reason", None),
                },
                "raw": {
                    "regime": getattr(raw_snapshot, "regime", None),
                    "confidence": getattr(raw_snapshot, "confidence", None),
                    "updated_at": getattr(raw_snapshot, "updated_at", None),
                    "reason": getattr(raw_snapshot, "reason", None),
                    "completed_bar_timestamp": (
                        raw_snapshot.adjustments.get("completed_bar_timestamp")
                        if raw_snapshot is not None
                        else None
                    ),
                    "atr_bps": (
                        raw_snapshot.adjustments.get("atr_bps")
                        if raw_snapshot is not None
                        else None
                    ),
                },
                "transition": {
                    "pending_regime": pending_regime,
                    "pending_count": pending_count,
                    "required_confirmations": self.transition_confirmations,
                },
                "bypass": self.get_regime_filter_bypass(),
                "stats": self.get_regime_filter_stats(),
                "history": history_payload,
                "decisions": decisions_payload,
            }
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.build_diagnostics: %s",
                exc,
                extra={"event": "regime_manager_build_diag_error"},
            )
            return {}

    # ------------------------------------------------------------------
    def _register_listener(self) -> None:
        """Attach the manager as a listener to the detector."""
        logger.debug(
            "Entered MarketRegimeManager._register_listener",
            extra={"event": "regime_manager_register_listener"},
        )
        try:
            self.detector.subscribe(self.ingest_snapshot)
            self._listener_registered = True
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager._register_listener: %s",
                exc,
                extra={"event": "regime_manager_register_error"},
            )
            self._listener_registered = False

    def _bootstrap_state(self) -> None:
        """Prime the cache with detector history when available."""
        logger.debug(
            "Entered MarketRegimeManager._bootstrap_state",
            extra={"event": "regime_manager_bootstrap"},
        )
        try:
            snapshots = self.detector.latest(limit=self.history_limit)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager._bootstrap_state: %s",
                exc,
                extra={"event": "regime_manager_bootstrap_error"},
            )
            return
        for snapshot in snapshots:
            self._store_snapshot(snapshot)

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides for gating thresholds."""
        logger.debug(
            "Entered MarketRegimeManager._apply_env_overrides",
            extra={"event": "regime_manager_env_overrides"},
        )
        try:
            min_conf = float(
                coalesce_float(
                    "REGIME_MIN_CONFIDENCE",
                    default=self.min_confidence,
                )
            )
            self.min_confidence = max(0.0, min(min_conf, 1.0))
            stale_after = float(
                coalesce_float(
                    "REGIME_STALE_AFTER_SEC",
                    default=self.stale_after_seconds,
                )
            )
            self.stale_after_seconds = max(30.0, stale_after)
            event_threshold = float(
                coalesce_float(
                    "REGIME_BLOCK_EVENT",
                    default=self.block_thresholds.get("event", 0.8),
                )
            )
            volatile_threshold = float(
                coalesce_float(
                    "REGIME_BLOCK_VOLATILE",
                    default=self.block_thresholds.get("volatile", 0.95),
                )
            )
            self.block_thresholds["event"] = max(0.0, min(event_threshold, 1.0))
            self.block_thresholds["volatile"] = max(0.0, min(volatile_threshold, 1.0))
            self.fail_closed = bool(
                coalesce_bool("REGIME_FAIL_CLOSED", default=self.fail_closed)
            )
            self.transition_confirmations = max(
                1,
                int(
                    coalesce_int(
                        "REGIME_TRANSITION_CONFIRMATIONS",
                        default=self.transition_confirmations,
                    )
                ),
            )
        except Exception as exc:  # noqa: BLE001 - defensive
            logger.error(
                "Failure in MarketRegimeManager._apply_env_overrides: %s",
                exc,
                extra={"event": "regime_manager_env_overrides_error"},
                exc_info=exc,
            )

    def ingest_snapshot(self, snapshot: RegimeSnapshot) -> None:
        """Record a raw detector snapshot and update the accepted stable state."""
        logger.debug(
            "Entered MarketRegimeManager.ingest_snapshot",
            extra={
                "event": "regime_manager_ingest",
                "symbol": snapshot.symbol,
                "regime": snapshot.regime,
            },
        )
        try:
            previous = self._store_snapshot(snapshot)
            current = self.get_latest_snapshot()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.ingest_snapshot: %s",
                exc,
                extra={"event": "regime_manager_ingest_error"},
            )
            return

        previous_key = normalize_regime(getattr(previous, "regime", None))
        current_key = normalize_regime(getattr(current, "regime", None))
        incoming_key = normalize_regime(snapshot.regime)
        if current is not None and previous_key is not current_key:
            logger.info(
                "Condition met: regime_manager_transition",
                extra={
                    "event": "regime_manager_transition",
                    "symbol": current.symbol,
                    "from": getattr(previous, "regime", None),
                    "to": current.regime,
                    "confidence": current.confidence,
                },
            )
        elif current is not None and incoming_key is not current_key:
            logger.debug(
                "REGIME_TRANSITION_PENDING symbol=%s stable=%s raw=%s "
                "count=%s required=%s",
                snapshot.symbol,
                current.regime,
                snapshot.regime,
                self._pending_count,
                self.transition_confirmations,
                extra={
                    "event": "REGIME_TRANSITION_PENDING",
                    "symbol": snapshot.symbol,
                    "stable_regime": current.regime,
                    "raw_regime": snapshot.regime,
                    "pending_count": self._pending_count,
                    "required_confirmations": self.transition_confirmations,
                },
            )
        try:
            metric_snapshot = current or snapshot
            METRICS.update_regime_confidence(
                regime=metric_snapshot.regime,
                confidence=metric_snapshot.confidence,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.ingest_snapshot metric update: %s",
                exc,
                extra={"event": "regime_manager_metric_error"},
            )

    def _block_threshold_for(self, regime: object) -> float:
        """Return a block threshold for any canonical or legacy regime label."""
        canonical = normalize_regime(regime)
        candidates = (
            str(regime or ""),
            str(regime or "").lower(),
            str(regime or "").upper(),
            canonical.value,
            canonical.value.lower(),
        )
        for key in candidates:
            if key in self.block_thresholds:
                return float(self.block_thresholds[key])
        return 0.0

    def _store_snapshot(self, snapshot: RegimeSnapshot) -> RegimeSnapshot | None:
        """Persist raw state and advance stable state only on confirmed transitions."""
        with self._lock:
            previous = self._current
            prior_raw = self._raw_current
            strict = self._indicator_settings.get("require_completed_evidence") is True
            if strict and snapshot.symbol != self._indicator_symbol:
                return previous
            incoming = normalize_regime(snapshot.regime)
            directional = incoming in {MarketRegime.TREND, MarketRegime.RANGE}
            evidence_at = snapshot.adjustments.get("completed_bar_timestamp")
            valid_evidence = bool(
                isinstance(evidence_at, (int, float))
                and not isinstance(evidence_at, bool)
                and math.isfinite(evidence_at)
                and evidence_at % 60 == 0
                and 60 <= snapshot.updated_at - evidence_at <= 180
            )
            if strict and valid_evidence and isinstance(evidence_at, (int, float)):
                if (
                    self._last_verified_evidence_at is not None
                    and evidence_at < self._last_verified_evidence_at
                ):
                    return previous
                self._last_verified_evidence_at = float(evidence_at)
            if (
                prior_raw is not None
                and prior_raw.symbol == snapshot.symbol
                and prior_raw.regime == snapshot.regime
                and prior_raw.confidence == snapshot.confidence
                and prior_raw.reason == snapshot.reason
                and prior_raw.updated_at == snapshot.updated_at
            ):
                return previous

            self._raw_current = snapshot
            self._history.append(snapshot)

            current = normalize_regime(getattr(previous, "regime", None))

            if strict and (
                not math.isfinite(snapshot.confidence)
                or not 0 <= snapshot.confidence <= 1
                or not math.isfinite(snapshot.updated_at)
                or (directional and not valid_evidence)
            ):
                self._pending_regime = incoming.value
                self._pending_count = 0
                self._pending_evidence_at = None
                return previous

            if previous is None and not (strict and directional):
                self._current = snapshot
                self._pending_regime = None
                self._pending_count = 0
                return previous

            if incoming is MarketRegime.UNKNOWN:
                self._pending_regime = incoming.value
                self._pending_count = 0
                return previous

            stable_at = (
                previous.adjustments.get("completed_bar_timestamp")
                if previous is not None
                else None
            )
            same_directional_sequence = bool(
                valid_evidence
                and isinstance(stable_at, (int, float))
                and isinstance(evidence_at, (int, float))
                and evidence_at - stable_at in (0, 60)
                and prior_raw is not None
                and normalize_regime(prior_raw.regime) is incoming
                and prior_raw.adjustments.get("completed_bar_timestamp") == stable_at
            )
            if incoming is current and (
                not strict or not directional or same_directional_sequence
            ):
                self._current = snapshot
                self._pending_regime = None
                self._pending_count = 0
                self._pending_evidence_at = None
                return previous

            safety_threshold = self._block_threshold_for(snapshot.regime)
            if safety_threshold and snapshot.confidence >= safety_threshold:
                self._current = snapshot
                self._pending_regime = None
                self._pending_count = 0
                return previous

            if snapshot.confidence < self.min_confidence:
                self._pending_regime = incoming.value
                self._pending_count = 0
                return previous

            if (
                incoming in {MarketRegime.TREND, MarketRegime.RANGE}
                and "completed_bar_timestamp" in snapshot.adjustments
            ):
                evidence_at = snapshot.adjustments.get("completed_bar_timestamp")
                if (
                    not isinstance(evidence_at, (int, float))
                    or isinstance(evidence_at, bool)
                    or not math.isfinite(evidence_at)
                    or evidence_at <= 0
                ):
                    self._pending_regime = incoming.value
                    self._pending_count = 0
                    self._pending_evidence_at = None
                    return previous
                prior_evidence = self._pending_evidence_at
                same_pending = (
                    self._pending_regime == incoming.value and self._pending_count > 0
                )
                if same_pending and prior_evidence is not None:
                    if evidence_at <= prior_evidence:
                        return previous
                    self._pending_count = (
                        self._pending_count + 1
                        if evidence_at - prior_evidence == 60.0
                        else 1
                    )
                else:
                    self._pending_count = 1
                self._pending_regime = incoming.value
                self._pending_evidence_at = float(evidence_at)
            else:
                # Explicit external observations retain their existing contract.
                # They cannot complete a partially observed candle transition.
                if (
                    self._pending_regime == incoming.value
                    and self._pending_evidence_at is None
                ):
                    self._pending_count += 1
                else:
                    self._pending_regime = incoming.value
                    self._pending_count = 1
                self._pending_evidence_at = None

            if self._pending_count >= self.transition_confirmations:
                self._current = snapshot
                self._pending_regime = None
                self._pending_count = 0
        return previous

    # ------------------------------------------------------------------
    def get_current_regime(self) -> str | None:
        """Return the accepted stable regime label when available."""
        logger.debug(
            "Entered MarketRegimeManager.get_current_regime",
            extra={"event": "regime_manager_current_regime"},
        )
        try:
            with self._lock:
                current = self._current
            if current is None:
                return None
            return str(current.regime) if current.regime is not None else None
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.get_current_regime: %s",
                exc,
                extra={"event": "regime_manager_current_regime_error"},
            )
            return None

    def get_regime_confidence(self) -> float:
        """Return the latest regime confidence score."""
        logger.debug(
            "Entered MarketRegimeManager.get_regime_confidence",
            extra={"event": "regime_manager_regime_confidence"},
        )
        try:
            with self._lock:
                current = self._current
            if current is None:
                return 0.0
            return max(0.0, min(float(current.confidence), 1.0))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.get_regime_confidence: %s",
                exc,
                extra={"event": "regime_manager_confidence_error"},
            )
            return 0.0

    def get_latest_snapshot(self) -> RegimeSnapshot | None:
        """Return the accepted stable snapshot without modifying state."""
        logger.debug(
            "Entered MarketRegimeManager.get_latest_snapshot",
            extra={"event": "regime_manager_latest_snapshot"},
        )
        try:
            with self._lock:
                return self._current
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.get_latest_snapshot: %s",
                exc,
                extra={"event": "regime_manager_latest_snapshot_error"},
            )
            return None

    def get_execution_snapshot(self) -> RegimeSnapshot | None:
        """Expose current safety regimes, retaining confirmed directional state."""
        with self._lock:
            raw = self._raw_current
            stable = self._current
            if raw is None:
                return None
            if normalize_regime(raw.regime) in {
                MarketRegime.UNKNOWN,
                MarketRegime.LOW_ACTIVITY,
                MarketRegime.EVENT,
                MarketRegime.VOLATILE,
            }:
                return raw
            return stable

    def get_raw_snapshot(self) -> RegimeSnapshot | None:
        """Return the freshest detector observation, including pending transitions."""
        try:
            with self._lock:
                return self._raw_current
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.get_raw_snapshot: %s",
                exc,
                extra={"event": "regime_manager_raw_snapshot_error"},
            )
            return None

    def get_history(self, limit: int = 10) -> list[RegimeSnapshot]:
        """Return a list of recent regime snapshots up to *limit*."""
        logger.debug(
            "Entered MarketRegimeManager.get_history",
            extra={"event": "regime_manager_history", "limit": limit},
        )
        if limit <= 0:
            return []
        try:
            with self._lock:
                return list(self._history)[-limit:]
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.get_history: %s",
                exc,
                extra={"event": "regime_manager_history_error"},
            )
            return []

    async def refresh_from_indicators(self) -> None:
        """Update regime snapshot by pulling fresh indicators."""
        if self.indicators is None:
            return

        # 1. Pull required metrics from IndicatorEngine
        strict = self._indicator_settings.get("require_completed_evidence") is True
        getter = getattr(
            self.indicators,
            "get_completed_regime_inputs" if strict else "get_indicators",
            None,
        )
        if not callable(getter):
            logger.warning(
                "REGIME_REFRESH_SKIPPED reason=indicator_engine_unavailable",
                extra={
                    "event": "REGIME_REFRESH_SKIPPED",
                    "reason": "indicator_engine_unavailable",
                    "symbol": self._indicator_symbol,
                },
            )
            return

        required_keys = [
            "ema_fast",
            "ema_slow",
            "adx",
            "atr",
            "volume_spike_ratio",
            "iv_rank",
            "price",
            "close",
        ]

        try:
            # 2. Fetch indicators
            if strict:
                now = datetime.now(ZoneInfo("Asia/Kolkata"))
                data = getter(self._indicator_symbol, now=now) or {}
                active_future = getattr(self.datahub, "get_active_futures_symbol", None)
                if callable(active_future) and data:
                    future = canonical_nifty_future_symbol(active_future())
                    if future is not None:
                        future_data = getter(future, now=now) or {}
                        ratio = future_data.get("volume_spike_ratio")
                        if (
                            canonical_nifty_future_symbol(active_future()) == future
                            and future_data.get("completed_bar_timestamp")
                            == data.get("completed_bar_timestamp")
                            and isinstance(ratio, (int, float))
                            and not isinstance(ratio, bool)
                            and math.isfinite(ratio)
                            and ratio >= 0
                        ):
                            data["volume_spike_ratio"] = ratio
                # Unavailable history is an explicit raw UNKNOWN, not a refresh
                # of the previous observation or a fabricated feature.
                data.setdefault("completed_bar_timestamp", 0.0)
            else:
                data = getter(self._indicator_symbol, required_keys)
            if not data:
                logger.debug(
                    "No indicator data returned",
                    extra={"event": "regime_indicators_empty"},
                )
                return

            # Fresh price alone cannot establish a new RANGE regime.
            # Preserve the existing stable state until an actual classifier
            # feature is available.
            has_regime_features = any(
                data.get(key) is not None
                for key in ("adx", "atr", "volume_spike_ratio")
            ) or (data.get("ema_fast") is not None and data.get("ema_slow") is not None)
            if not has_regime_features and not strict:
                logger.debug(
                    "REGIME_REFRESH_SKIPPED reason=insufficient_regime_features"
                )
                return

            # 3. Use only timestamped, completed in-memory underlying bars.
            # Price-only snapshots cannot prove flat momentum or low activity.
            price_series: list[float] = []
            completed_bar_timestamp = 0.0
            history = getattr(self.indicators, "get_history", None)
            if strict:
                price_series = list(data.get("price_series") or ())
                completed_bar_timestamp = float(data.get("completed_bar_timestamp") or 0)
            elif callable(history):
                try:
                    bars = history(self._indicator_symbol, count=16, field="bars")
                    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
                    completed: list[tuple[datetime, float]] = []
                    for bar in bars or []:
                        if not isinstance(bar, Mapping):
                            continue
                        if (
                            bar.get("is_complete") is not True
                            or bar.get("is_provisional") is True
                        ):
                            continue
                        stamp = bar.get("timestamp")
                        ts = (
                            stamp
                            if isinstance(stamp, datetime)
                            else datetime.fromisoformat(
                                str(stamp).replace("Z", "+00:00")
                            )
                        )
                        if ts.tzinfo is None:
                            continue
                        ts = ts.astimezone(now_ist.tzinfo)
                        if ts.second != 0 or ts.microsecond != 0:
                            continue
                        age = (now_ist - ts).total_seconds()
                        close_raw = bar.get("close")
                        if close_raw is None:
                            continue
                        value = float(close_raw)
                        if (
                            ts.date() == now_ist.date()
                            and 9 * 60 + 15 <= ts.hour * 60 + ts.minute < 15 * 60 + 30
                            and 60 <= age <= 1200
                            and math.isfinite(value)
                            and value > 0
                        ):
                            completed.append((ts, value))
                    unique = dict(completed)
                    if any(unique[ts] != close for ts, close in completed):
                        raise ValueError("conflicting_regime_completed_bar")
                    completed = sorted(unique.items())
                    # Use the contiguous tail, not a history with silent gaps.
                    start = len(completed) - 1
                    while (
                        start > 0
                        and (
                            completed[start][0] - completed[start - 1][0]
                        ).total_seconds()
                        == 60.0
                    ):
                        start -= 1
                    completed = completed[start:]
                    if (
                        len(completed) >= 6
                        and (now_ist - completed[-1][0]).total_seconds() <= 180
                    ):
                        price_series = [close for _, close in completed]
                        completed_bar_timestamp = completed[-1][0].timestamp()
                except (TypeError, ValueError, OverflowError, AttributeError):
                    price_series = []

            # 4. Map to detector schema without replacing explicit indicators.
            enrichment = {
                "trend_score": data.get("adx"),
                "volume_ratio": data.get("volume_spike_ratio"),
                "price": data.get("close"),
                **{k: v for k, v in data.items() if v is not None},
            }
            for key in (
                "price_series",
                "close_series",
                "price_history",
                "close_history",
            ):
                enrichment.pop(key, None)
            if price_series:
                enrichment["price_series"] = price_series
            enrichment["completed_bar_timestamp"] = completed_bar_timestamp

            # 5. Evaluate regime
            snapshot = self.detector.evaluate(self._indicator_symbol, enrichment)

            # 🔴 CRITICAL VALIDATION
            if snapshot is None:
                logger.warning(
                    "Regime detector returned no snapshot",
                    extra={
                        "event": "regime_snapshot_none",
                        "symbol": self._indicator_symbol,
                    },
                )
                return

            stable_snapshot = self.get_latest_snapshot()
            history_momentum = (
                (price_series[-1] - price_series[0]) / abs(price_series[0])
                if len(price_series) >= 2 and price_series[0] != 0
                else None
            )
            log_throttled(
                logger,
                f"regime_input_verification:{self._indicator_symbol}",
                (
                    "REGIME_INPUT_VERIFICATION symbol=%s regime=%s reason=%s "
                    "adx_available=%s ema_pair_available=%s atr_available=%s "
                    "completed_history_bars=%s volume_available=%s confidence=%.3f "
                    "stable_regime=%s adx_value=%s history_momentum=%s"
                ),
                self._indicator_symbol,
                snapshot.regime,
                snapshot.reason,
                data.get("adx") is not None,
                data.get("ema_fast") is not None and data.get("ema_slow") is not None,
                data.get("atr") is not None,
                data.get("verified_completed_bars", len(price_series)),
                enrichment.get("volume_ratio") is not None,
                snapshot.confidence,
                getattr(stable_snapshot, "regime", None),
                data.get("adx"),
                (
                    f"{history_momentum:.6f}"
                    if history_momentum is not None
                    else "unavailable"
                ),
                interval_sec=60.0,
                extra={
                    "event": "REGIME_INPUT_VERIFICATION",
                    "symbol": self._indicator_symbol,
                    "regime": snapshot.regime,
                    "regime_reason": snapshot.reason,
                    "verified_completed_history_bars": data.get(
                        "verified_completed_bars", len(price_series)
                    ),
                    "stable_regime": getattr(stable_snapshot, "regime", None),
                    "adx_value": data.get("adx"),
                    "history_momentum": history_momentum,
                },
            )
            logger.debug(
                "Regime snapshot refreshed from indicators",
                extra={
                    "event": "regime_snapshot_refreshed",
                    "symbol": self._indicator_symbol,
                    "regime": str(snapshot.regime),
                    "confidence": snapshot.confidence,
                    "updated_at": str(snapshot.updated_at),
                },
            )

            # Detector.evaluate() records and broadcasts the snapshot to this
            # manager's registered listener. Calling ingest_snapshot() again here
            # double-counted every indicator refresh and defeated transition
            # confirmation semantics.
            self._last_indicator_refresh = time.time()

        except Exception as exc:
            logger.error(
                "Regime refresh error",
                extra={
                    "event": "regime_refresh_error",
                    "symbol": self._indicator_symbol,
                },
                exc_info=True,
            )

    # ✅ FIXED: New Background Loop Method
    async def _run_indicator_refresh_loop(self) -> None:
        """Background task to actively pull regime snapshots from indicators."""
        try:
            await self.refresh_from_indicators()
        except Exception as exc:
            logger.error("❌ Initial regime refresh failed", exc_info=True)

        while True:
            try:
                await asyncio.sleep(self._indicator_update_interval)
                await self.refresh_from_indicators()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"❌ Regime refresh failed: {exc}", exc_info=True)
                await asyncio.sleep(5.0)

    def can_trade(
        self,
        *,
        context: Mapping[str, object] | None = None,
        record_decision: bool = True,
        expected_regime: object | None = None,
    ) -> bool:
        """Gate current evidence; optionally reject a changed entry regime."""
        logger.debug(
            "Entered MarketRegimeManager.can_trade",
            extra={"event": "regime_manager_can_trade"},
        )

        try:
            snapshot: RegimeSnapshot | None
            raw_snapshot: RegimeSnapshot | None
            bypass: bool
            with self._lock:
                snapshot = self.get_execution_snapshot()
                raw_snapshot = self._raw_current
                stable_snapshot = self._current
                strict = (
                    self._indicator_settings.get("require_completed_evidence") is True
                )
                bypass = self._bypass and not strict
            if bypass:
                bypass_reasons = ("bypass_enabled",)
                decision = RegimeDecision(
                    allowed=True,
                    reasons=bypass_reasons,
                    bypassed=True,
                    decided_at=time.time(),
                    regime=getattr(snapshot, "regime", None),
                    confidence=getattr(snapshot, "confidence", None),
                )
                self._record_decision(decision, record_decision)
                self._emit_decision_log(
                    allowed=True,
                    bypassed=True,
                    reasons=bypass_reasons,
                    snapshot=snapshot,
                )
                return True
            reasons: list[str] = []
            allowed = True
            now = time.time()
            if raw_snapshot is None:
                # Check for auto-bypass when regime data is unavailable
                # This provides graceful degradation during API outages
                auto_bypass = coalesce_bool(
                    "REGIME_BYPASS_ON_UNAVAILABLE", default=False
                )

                if auto_bypass and not strict:
                    # Log warning but allow trading with caution
                    logger.warning(
                        "⚠️ Regime unavailable - BYPASS ACTIVE (trading with reduced confidence)",
                        extra={
                            "event": "regime_unavailable_bypass",
                            "action": "allow_with_caution",
                        },
                    )
                    # Allow trading but record the bypass reason
                    allowed = True
                    reasons.append("regime_unavailable_bypassed")

                    # ✅ Create a synthetic "cautious" regime decision
                    decision = RegimeDecision(
                        allowed=True,
                        reasons=("regime_unavailable_bypassed",),
                        bypassed=True,  # Mark as bypassed
                        decided_at=time.time(),
                        regime="unknown",
                        confidence=0.5,  # Conservative confidence
                    )
                    self._stats["bypass"] = self._stats.get("bypass", 0) + 1
                    self._record_decision(decision, record_decision)
                    self._emit_decision_log(
                        allowed=True,
                        bypassed=True,
                        reasons=("regime_unavailable_bypassed",),
                        snapshot=None,
                    )
                    return True
                else:
                    # Original behavior: block trading
                    allowed = False
                    reasons.append("regime_unavailable")
            else:
                raw_age = now - float(raw_snapshot.updated_at)
                age = max(0.0, raw_age) if math.isfinite(raw_age) else float("inf")
                if not math.isfinite(raw_age) or raw_age < -1.0:
                    allowed = False
                    reasons.append("regime_timestamp_invalid")
                elif age > self.stale_after_seconds:
                    # ✅ FIX: Check for stale bypass
                    stale_bypass = coalesce_bool(
                        "REGIME_BYPASS_ON_STALE", default=False
                    )

                    if (
                        stale_bypass
                        and not strict
                        and age < (self.stale_after_seconds * 3)
                    ):
                        # Allow if data is stale but not ancient (3x threshold)
                        logger.warning(
                            f"⚠️ Regime data stale ({age:.1f}s) - BYPASS ACTIVE",
                            extra={
                                "event": "regime_stale_bypass",
                                "age_seconds": age,
                                "threshold": self.stale_after_seconds,
                            },
                        )
                        reasons.append("regime_stale_bypassed")
                        # Don't set allowed=False, continue to other checks
                    else:
                        allowed = False
                        reasons.append("regime_stale")
                confidence = float(raw_snapshot.confidence)
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    allowed = False
                    reasons.append("confidence_invalid")
                elif confidence < self.min_confidence:
                    allowed = False
                    reasons.append("confidence_below_floor")
                raw_regime = normalize_regime(raw_snapshot.regime)
                if strict:
                    evidence_at = raw_snapshot.adjustments.get("completed_bar_timestamp")
                    if (
                        not isinstance(evidence_at, (int, float))
                        or isinstance(evidence_at, bool)
                        or not math.isfinite(evidence_at)
                        or not 60 <= now - evidence_at <= 180
                        or evidence_at % 60 != 0
                    ):
                        allowed = False
                        reasons.append("regime_completed_evidence_unavailable_or_stale")
                    if raw_regime in {MarketRegime.TREND, MarketRegime.RANGE} and (
                        stable_snapshot is None
                        or normalize_regime(stable_snapshot.regime) is not raw_regime
                        or stable_snapshot.adjustments.get("completed_bar_timestamp")
                        != evidence_at
                    ):
                        allowed = False
                        reasons.append("regime_transition_unconfirmed")
                if expected_regime is not None and (
                    normalize_regime(expected_regime) is MarketRegime.UNKNOWN
                    or normalize_regime(expected_regime) is not raw_regime
                    or normalize_regime(getattr(snapshot, "regime", None))
                    is not raw_regime
                ):
                    allowed = False
                    reasons.append("regime_changed_before_entry")
                if raw_regime is MarketRegime.UNKNOWN:
                    allowed = False
                    reasons.append("regime_unknown")
                elif raw_regime is MarketRegime.LOW_ACTIVITY:
                    allowed = False
                    reasons.append("regime_low_activity")
                # Stable hysteresis is not authority to accept an unsafe
                # current raw EVENT/VOLATILE observation.
                threshold = self._block_threshold_for(raw_snapshot.regime)
                if threshold and raw_snapshot.confidence >= threshold:
                    allowed = False
                    reasons.append(f"regime_block_{raw_snapshot.regime}")
            if context:
                reasons.extend(
                    [f"ctx_{key}" for key in context.keys() if isinstance(key, str)]
                )
            if allowed:
                self._stats["pass"] = self._stats.get("pass", 0) + 1
            else:
                self._stats["block"] = self._stats.get("block", 0) + 1
            decision = RegimeDecision(
                allowed=allowed,
                reasons=tuple(reasons),
                bypassed=False,
                decided_at=now,
                regime=getattr(snapshot, "regime", None),
                confidence=getattr(snapshot, "confidence", None),
            )
            self._record_decision(decision, record_decision)
            self._emit_decision_log(
                allowed=allowed,
                bypassed=False,
                reasons=decision.reasons,
                snapshot=snapshot,
            )
            return allowed
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.can_trade: %s",
                exc,
                extra={"event": "regime_manager_can_trade_error"},
            )
            if (
                self.fail_closed
                or self._indicator_settings.get("require_completed_evidence") is True
            ):
                failure_reasons = ("regime_fail_closed_error",)
                decision = RegimeDecision(
                    allowed=False,
                    reasons=failure_reasons,
                    bypassed=False,
                    decided_at=time.time(),
                    regime=None,
                    confidence=None,
                )
                self._stats["block"] = self._stats.get("block", 0) + 1
                self._record_decision(decision, record_decision)
                self._emit_decision_log(
                    allowed=False,
                    bypassed=False,
                    reasons=failure_reasons,
                    snapshot=None,
                )
                return False
            return True

    def _record_decision(self, decision: RegimeDecision, record: bool) -> None:
        """Persist decision and update last reasons when *record* is true."""
        with self._lock:
            self._last_filter_reasons = decision.reasons
            if record:
                self._decisions.append(decision)
            if decision.bypassed:
                self._stats["bypass"] = self._stats.get("bypass", 0) + 1

    def _emit_decision_log(
        self,
        *,
        allowed: bool,
        bypassed: bool,
        reasons: tuple[str, ...],
        snapshot: RegimeSnapshot | None,
    ) -> None:
        """Emit deduplicated decision logs to avoid per-tick noise."""
        logger.debug(
            "Entered MarketRegimeManager._emit_decision_log",
            extra={"event": "regime_manager_emit_log"},
        )
        now = time.time()
        decision_key = (allowed, reasons, bypassed)
        should_log = True
        with self._lock:
            last = self._last_logged_decision
            last_at = self._last_logged_at
            if last == decision_key and (now - last_at) < self._log_cooldown:
                should_log = False
            else:
                self._last_logged_decision = decision_key
                self._last_logged_at = now
        if not should_log:
            return
        extras = {
            "event": "regime_manager_can_trade",
            "regime": getattr(snapshot, "regime", None),
            "confidence": getattr(snapshot, "confidence", None),
            "reasons": list(reasons),
        }
        if bypassed:
            extras["event"] = "regime_manager_can_trade_bypass"
            logger.info("Condition met: regime_manager_can_trade_bypass", extra=extras)
            return
        if allowed:
            logger.info("Condition met: regime_manager_can_trade", extra=extras)
        else:
            extras["event"] = "regime_manager_trade_blocked"
            logger.warning("Condition met: regime_manager_trade_blocked", extra=extras)

    # ------------------------------------------------------------------
    def get_filter_reasons(self) -> tuple[str, ...]:
        """Return reasons for the last gating decision."""
        logger.debug(
            "Entered MarketRegimeManager.get_filter_reasons",
            extra={"event": "regime_manager_filter_reasons"},
        )
        try:
            with self._lock:
                return self._last_filter_reasons
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.get_filter_reasons: %s",
                exc,
                extra={"event": "regime_manager_filter_reasons_error"},
            )
            return tuple()

    def set_regime_filter_bypass(self, enabled: bool) -> bool:
        """Enable or disable the regime bypass flag."""
        logger.debug(
            "Entered MarketRegimeManager.set_regime_filter_bypass",
            extra={"event": "regime_manager_set_bypass", "enabled": bool(enabled)},
        )
        try:
            with self._lock:
                self._bypass = bool(enabled)
            logger.info(
                "Condition met: regime_manager_bypass_updated",
                extra={
                    "event": "regime_manager_bypass_updated",
                    "enabled": bool(enabled),
                },
            )
            return self._bypass
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.set_regime_filter_bypass: %s",
                exc,
                extra={"event": "regime_manager_set_bypass_error"},
            )
            return self._bypass

    def toggle_regime_filter_bypass(self) -> bool:
        """Invert the bypass flag and return the new state."""
        logger.debug(
            "Entered MarketRegimeManager.toggle_regime_filter_bypass",
            extra={"event": "regime_manager_toggle_bypass"},
        )
        try:
            with self._lock:
                next_state = not self._bypass
            return self.set_regime_filter_bypass(next_state)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.toggle_regime_filter_bypass: %s",
                exc,
                extra={"event": "regime_manager_toggle_bypass_error"},
            )
            return self._bypass

    def toggle_bypass(self) -> bool:
        """Alias for :meth:`toggle_regime_filter_bypass` for compatibility."""
        logger.debug(
            "Entered MarketRegimeManager.toggle_bypass",
            extra={"event": "regime_manager_toggle_bypass_alias"},
        )
        try:
            return self.toggle_regime_filter_bypass()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.toggle_bypass: %s",
                exc,
                extra={"event": "regime_manager_toggle_bypass_alias_error"},
            )
            return self.get_regime_filter_bypass()

    def get_regime_filter_bypass(self) -> bool:
        """Return ``True`` when bypass is currently enabled."""
        logger.debug(
            "Entered MarketRegimeManager.get_regime_filter_bypass",
            extra={"event": "regime_manager_get_bypass"},
        )
        try:
            with self._lock:
                return self._bypass
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.get_regime_filter_bypass: %s",
                exc,
                extra={"event": "regime_manager_get_bypass_error"},
            )
            return False

    def get_regime_filter_stats(self) -> dict[str, int]:
        """Return pass, block, and bypass counters."""
        logger.debug(
            "Entered MarketRegimeManager.get_regime_filter_stats",
            extra={"event": "regime_manager_filter_stats"},
        )
        try:
            with self._lock:
                return {key: int(value) for key, value in self._stats.items()}
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.get_regime_filter_stats: %s",
                exc,
                extra={"event": "regime_manager_filter_stats_error"},
            )
            return {}

    def get_decision_history(self, limit: int = 25) -> list[RegimeDecision]:
        """Return the most recent gating decisions up to *limit*."""
        logger.debug(
            "Entered MarketRegimeManager.get_decision_history",
            extra={"event": "regime_manager_decision_history", "limit": limit},
        )
        if limit <= 0:
            return []
        try:
            with self._lock:
                return list(self._decisions)[-limit:]
        except Exception as exc:
            logger.error(
                "Failure in MarketRegimeManager.get_decision_history: %s",
                exc,
                extra={"event": "regime_manager_decision_history_error"},
            )
            return []


__all__ = ["MarketRegimeManager", "RegimeDecision"]
