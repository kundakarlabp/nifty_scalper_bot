"""Central market regime coordinator with gating and diagnostics."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Deque, Mapping, MutableMapping

from nifty_scalper_bot.core.market_regime import (
    MarketRegimeDetector,
    RegimeSnapshot,
)
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.utils.env import coalesce_bool, coalesce_float
from nifty_scalper_bot.utils.logging import get_logger

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
    min_confidence: float = 0.45
    stale_after_seconds: float = 180.0
    block_thresholds: MutableMapping[str, float] = field(
        default_factory=lambda: {"event": 0.8, "volatile": 0.95}
    )
    fail_closed: bool = False
    datahub: Any | None = None
    indicators: Any | None = None
    regime_settings: Mapping[str, Any] | None = None
    _lock: threading.RLock = field(init=False, repr=False)
    _history: Deque[RegimeSnapshot] = field(init=False, repr=False)
    _decisions: Deque[RegimeDecision] = field(init=False, repr=False)
    _current: RegimeSnapshot | None = field(init=False, default=None, repr=False)
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

    def __post_init__(self) -> None:
        """Initialise shared state and subscribe to detector updates.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

        settings: dict[str, Any] = {}
        try:
            if self.regime_settings is not None:
                settings = dict(self.regime_settings)
        except Exception as exc:  # noqa: BLE001 - defensive
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
        self._lock = threading.RLock()
        self._history: Deque[RegimeSnapshot] = deque(maxlen=max(5, self.history_limit))
        self._decisions: Deque[RegimeDecision] = deque(maxlen=200)
        self._current: RegimeSnapshot | None = None
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

    # ------------------------------------------------------------------
    def _register_listener(self) -> None:
        """Attach the manager as a listener to the detector.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

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
        """Prime the cache with detector history when available.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """

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
        except Exception as exc:  # noqa: BLE001 - defensive
            logger.error(
                "Failure in MarketRegimeManager._apply_env_overrides: %s",
                exc,
                extra={"event": "regime_manager_env_overrides_error"},
                exc_info=exc,
            )

    def ingest_snapshot(self, snapshot: RegimeSnapshot) -> None:
        """Record incoming *snapshot* from the detector.

        Args:
            snapshot: Latest regime snapshot broadcast by the detector.

        Returns:
            None.

        Raises:
            None.
        """

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
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.ingest_snapshot: %s",
                exc,
                extra={"event": "regime_manager_ingest_error"},
            )
            return
        if previous is None or previous.regime != snapshot.regime:
            logger.info(
                "Condition met: regime_manager_transition",
                extra={
                    "event": "regime_manager_transition",
                    "symbol": snapshot.symbol,
                    "from": getattr(previous, "regime", None),
                    "to": snapshot.regime,
                    "confidence": snapshot.confidence,
                },
            )
        try:
            METRICS.update_regime_confidence(
                regime=snapshot.regime,
                confidence=snapshot.confidence,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.ingest_snapshot metric update: %s",
                exc,
                extra={"event": "regime_manager_metric_error"},
            )

    def _store_snapshot(self, snapshot: RegimeSnapshot) -> RegimeSnapshot | None:
        """Persist *snapshot* into caches and return previous entry.

        Args:
            snapshot: Snapshot to persist.

        Returns:
            Optional previously stored snapshot for the symbol.

        Raises:
            None.
        """

        with self._lock:
            previous = self._current
            self._current = snapshot
            self._history.append(snapshot)
        return previous

    # ------------------------------------------------------------------
    def get_current_regime(self) -> str | None:
        """Return the latest regime label when available.

        Args:
            None.

        Returns:
            str | None: Regime label or ``None`` when unavailable.

        Raises:
            None.
        """

        logger.debug(
            "Entered MarketRegimeManager.get_current_regime",
            extra={"event": "regime_manager_current_regime"},
        )
        try:
            with self._lock:
                current = self._current
            return None if current is None else current.regime
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.get_current_regime: %s",
                exc,
                extra={"event": "regime_manager_current_regime_error"},
            )
            return None

    def get_regime_confidence(self) -> float:
        """Return the latest regime confidence score.

        Args:
            None.

        Returns:
            float: Confidence bounded between ``0`` and ``1``.

        Raises:
            None.
        """

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
        """Return the most recent snapshot without modifying state.

        Args:
            None.

        Returns:
            RegimeSnapshot | None: Cached snapshot when present.

        Raises:
            None.
        """

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

    def get_history(self, limit: int = 10) -> list[RegimeSnapshot]:
        """Return a list of recent regime snapshots up to *limit*.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            list[RegimeSnapshot]: Ordered history oldest to newest.

        Raises:
            None.
        """

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
        """Update regime snapshot based on configured indicator adapter."""

        logger.debug(
            "Entered MarketRegimeManager.refresh_from_indicators",
            extra={"event": "regime_manager_refresh_indicators"},
        )
        if self.indicators is None:
            logger.info(
                "Condition met: regime_manager_refresh_skipped",
                extra={"event": "regime_manager_refresh_skipped"},
            )
            return
        getter = getattr(self.indicators, "get", None)
        if not callable(getter):
            logger.info(
                "Condition met: regime_manager_refresh_skipped",
                extra={"event": "regime_manager_refresh_skipped"},
            )
            return
        now = time.time()
        if (
            self._indicator_update_interval > 0
            and now - self._last_indicator_refresh < self._indicator_update_interval
        ):
            logger.debug(
                "Condition met: regime_manager_refresh_throttled",
                extra={"event": "regime_manager_refresh_throttled"},
            )
            return
        async with self._indicator_lock:
            try:
                atr_trend = await asyncio.to_thread(getter, "atr_trend")
                vol_score = await asyncio.to_thread(getter, "volatility_index")
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failure in MarketRegimeManager.refresh_from_indicators fetch: %s",
                    exc,
                    extra={"event": "regime_manager_refresh_error"},
                )
                return
            try:
                atr_value = float(atr_trend) if atr_trend is not None else 0.0
            except Exception:  # pragma: no cover - defensive cast
                atr_value = 0.0
            try:
                vol_value = float(vol_score) if vol_score is not None else 0.0
            except Exception:  # pragma: no cover - defensive cast
                vol_value = 0.0
            atr_threshold = float(
                self._indicator_settings.get("atr_trend_threshold", 1.5)
            )
            vol_threshold = float(self._indicator_settings.get("vol_threshold", 25.0))
            if atr_value > atr_threshold:
                regime = "TREND"
            elif vol_value > vol_threshold:
                regime = "VOLATILE"
            else:
                regime = "RANGE"
            confidence = ((atr_value / 3.0) + (vol_value / 50.0)) / 2.0
            confidence = max(0.0, min(confidence, 1.0))
            snapshot = RegimeSnapshot(
                symbol=self._indicator_symbol,
                regime=regime,
                confidence=confidence,
                reason="indicator_refresh",
                updated_at=now,
                adjustments={
                    "atr_trend": atr_value,
                    "volatility_index": vol_value,
                },
            )
            self.ingest_snapshot(snapshot)
            self._last_indicator_refresh = now
            logger.info(
                "Condition met: regime_manager_indicator_refresh",
                extra={
                    "event": "regime_manager_indicator_refresh",
                    "regime": regime,
                    "confidence": confidence,
                },
            )

    # ------------------------------------------------------------------
    def can_trade(
        self,
        *,
        context: Mapping[str, object] | None = None,
        record_decision: bool = True,
    ) -> bool:
        """Return ``True`` when trades are allowed under the current regime.

        Args:
            context: Optional decision context metadata.
            record_decision: Whether to store the decision in history.

        Returns:
            bool: ``True`` when trading should proceed.

        Raises:
            None.
        """

        logger.debug(
            "Entered MarketRegimeManager.can_trade",
            extra={"event": "regime_manager_can_trade"},
        )
        try:
            snapshot: RegimeSnapshot | None
            bypass: bool
            with self._lock:
                snapshot = self._current
                bypass = self._bypass
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
            if snapshot is None:
                allowed = False
                reasons.append("regime_unavailable")
            else:
                age = max(0.0, now - float(snapshot.updated_at))
                if age > self.stale_after_seconds:
                    allowed = False
                    reasons.append("regime_stale")
                if snapshot.confidence < self.min_confidence:
                    allowed = False
                    reasons.append("confidence_below_floor")
                threshold = self.block_thresholds.get(snapshot.regime, 0.0)
                if threshold and snapshot.confidence >= threshold:
                    allowed = False
                    reasons.append(f"regime_block_{snapshot.regime}")
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
            if self.fail_closed:
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
        """Persist decision and update last reasons when *record* is true.

        Args:
            decision: Decision entry recorded for diagnostics.
            record: Whether to append the decision to history.

        Returns:
            None.

        Raises:
            None.
        """

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
        """Emit deduplicated decision logs to avoid per-tick noise.

        Args:
            allowed: ``True`` when the trade was permitted.
            bypassed: ``True`` when bypass forced the allow decision.
            reasons: Tuple of textual reasons associated with the decision.
            snapshot: Latest snapshot associated with the decision.

        Returns:
            None.

        Raises:
            None.
        """

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
        """Return reasons for the last gating decision.

        Args:
            None.

        Returns:
            tuple[str, ...]: Ordered reasons captured during gating.

        Raises:
            None.
        """

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
        """Enable or disable the regime bypass flag.

        Args:
            enabled: Desired bypass state.

        Returns:
            bool: Updated bypass state.

        Raises:
            None.
        """

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
        """Invert the bypass flag and return the new state.

        Args:
            None.

        Returns:
            bool: New bypass state after toggling.

        Raises:
            None.
        """

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
        """Alias for :meth:`toggle_regime_filter_bypass` for compatibility.

        Args:
            None.

        Returns:
            bool: Updated bypass state after toggling.

        Raises:
            None.
        """

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
        """Return ``True`` when bypass is currently enabled.

        Args:
            None.

        Returns:
            bool: Current bypass state.

        Raises:
            None.
        """

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
        """Return pass, block, and bypass counters.

        Args:
            None.

        Returns:
            dict[str, int]: Copy of decision counters.

        Raises:
            None.
        """

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
        """Return the most recent gating decisions up to *limit*.

        Args:
            limit: Maximum number of decision records to return.

        Returns:
            list[RegimeDecision]: Ordered decision snapshots.

        Raises:
            None.
        """

        logger.debug(
            "Entered MarketRegimeManager.get_decision_history",
            extra={"event": "regime_manager_decision_history", "limit": limit},
        )
        if limit <= 0:
            return []
        try:
            with self._lock:
                return list(self._decisions)[-limit:]
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in MarketRegimeManager.get_decision_history: %s",
                exc,
                extra={"event": "regime_manager_decision_history_error"},
            )
            return []

    # ------------------------------------------------------------------
    def build_diagnostics(self) -> dict[str, object]:
        """Return structured diagnostics for operators.

        Args:
            None.

        Returns:
            dict[str, object]: Diagnostic payload for telemetry.

        Raises:
            None.
        """

        logger.debug(
            "Entered MarketRegimeManager.build_diagnostics",
            extra={"event": "regime_manager_build_diag"},
        )
        try:
            snapshot = self.get_latest_snapshot()
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


__all__ = ["MarketRegimeManager", "RegimeDecision"]
