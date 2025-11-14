"""Shadow (paper) trading utilities."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, Mapping, Tuple

from nifty_scalper_bot.config.settings import ShadowSettings
from nifty_scalper_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from nifty_scalper_bot.data.persistent_state import PersistentStateManager


@dataclass(slots=True)
class ShadowPaperTrader:
    """Track paper positions alongside live trading to detect drift."""

    settings: ShadowSettings
    live_equity_fn: Callable[[], float]
    notifier: Callable[[str, Mapping[str, object]], None] | None = None
    disable_live_callback: Callable[[str], None] | None = None
    _logger: Any = field(init=False, repr=False)
    _paper_positions: Dict[str, float] = field(
        init=False, repr=False, default_factory=dict
    )
    _live_disabled: bool = field(init=False, repr=False, default=False)
    _persistent_state: "PersistentStateManager | None" = field(
        init=False, repr=False, default=None
    )
    _breach_started_at: float | None = field(init=False, repr=False, default=None)
    _drift_samples: Deque[Tuple[float, float]] = field(
        init=False, repr=False, default_factory=deque
    )
    _last_alert_at: float | None = field(init=False, repr=False, default=None)
    _clock: Callable[[], float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = get_logger(__name__)
        self._clock = time.monotonic

    def record_order(self, symbol: str, side: str, quantity: int, price: float) -> None:
        """Record a paper trade fill and persist the resulting exposure.

        Args:
            symbol: Canonical instrument identifier for the trade.
            side: Order side recorded (``BUY`` or ``SELL``).
            quantity: Matched quantity for the execution.
            price: Execution price used for notional adjustments.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered record_order",
            extra={
                "event": "shadow_record_order",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
            },
        )
        direction = 1 if side.upper() == "BUY" else -1
        delta = direction * quantity * price
        self._paper_positions[symbol] = self._paper_positions.get(symbol, 0.0) + delta
        manager = self._persistent_state
        if manager is not None:
            snapshot = dict(self._paper_positions)
            try:
                manager.save_shadow_equity(sum(snapshot.values()), snapshot)
            except Exception as exc:  # noqa: BLE001
                self._logger.error("Failure in record_order persistence: %s", exc)
        self._check_drift()

    def current_drift_pct(self) -> float:
        live = self.live_equity_fn()
        paper = sum(self._paper_positions.values())
        if live == 0:
            return 0.0
        return abs(paper - live) / abs(live) * 100.0

    def _check_drift(self) -> None:
        drift = self.current_drift_pct()
        now = self._clock()
        self._record_sample(now, drift)
        self._evaluate_alerts(now, drift)
        self._evaluate_live_toggle(now, drift)

    def _record_sample(self, now: float, drift: float) -> None:
        """Maintain drift history for alert evaluation windows.

        Args:
            now: Monotonic timestamp when the measurement was taken.
            drift: Drift percentage snapshot for the sample.

        Returns:
            None.

        Raises:
            None.
        """

        self._drift_samples.append((now, drift))
        window = max(self.settings.alert_trend_window_seconds, 0.0)
        if window <= 0:
            while len(self._drift_samples) > 1:
                self._drift_samples.popleft()
            return
        cutoff = now - window
        while self._drift_samples and self._drift_samples[0][0] < cutoff:
            self._drift_samples.popleft()

    def _evaluate_alerts(self, now: float, drift: float) -> None:
        """Dispatch divergence alerts when thresholds or trends breach.

        Args:
            now: Current monotonic timestamp used for cooldown tracking.
            drift: Latest drift percentage value.

        Returns:
            None.

        Raises:
            None.
        """

        if self.notifier is None:
            return
        threshold_met = drift >= self.settings.alert_threshold_pct
        trending = self._is_trending()
        if not threshold_met and not trending:
            return
        cooldown = max(self.settings.alert_cooldown_seconds, 0.0)
        if self._last_alert_at is not None and now - self._last_alert_at < cooldown:
            return
        payload = self._snapshot_payload(drift)
        payload["trending"] = trending
        self._logger.info(
            "Condition met: shadow_drift_alert",
            extra={"event": "shadow_drift_alert", **payload},
        )
        try:
            self.notifier("shadow_drift_alert", payload)
        except Exception as exc:  # noqa: BLE001 - notifier is external
            self._logger.error(
                "Failure in shadow drift notifier: %s",
                exc,
                extra={"event": "shadow_drift_alert_failed"},
            )
        self._last_alert_at = now

    def _evaluate_live_toggle(self, now: float, drift: float) -> None:
        """Trigger live disable once drift breaches for sustained duration.

        Args:
            now: Current monotonic timestamp.
            drift: Latest drift percentage measurement.

        Returns:
            None.

        Raises:
            None.
        """

        threshold = self.settings.drift_threshold_pct
        sustain = max(self.settings.drift_sustain_seconds, 0.0)
        if drift >= threshold:
            if self._breach_started_at is None:
                self._breach_started_at = now
                payload = self._snapshot_payload(drift)
                self._logger.warning(
                    "Paper/live drift threshold breached",
                    extra={"event": "shadow_drift", **payload},
                )
                self._emit_notifier("shadow_drift", payload)
            elapsed = (
                0.0
                if self._breach_started_at is None
                else now - self._breach_started_at
            )
            if (
                self.settings.auto_disable_live_on_drift
                and not self._live_disabled
                and (sustain == 0.0 or elapsed >= sustain)
            ):
                self._live_disabled = True
                payload = self._snapshot_payload(drift)
                payload["elapsed_seconds"] = round(elapsed, 2)
                self._logger.error(
                    "Condition met: shadow_auto_disable",
                    extra={"event": "shadow_auto_disable", **payload},
                )
                self._emit_notifier("shadow_drift_disable", payload)
                if self.disable_live_callback is not None:
                    try:
                        self.disable_live_callback("shadow-drift")
                    except Exception as exc:  # noqa: BLE001 - defensive
                        self._logger.error(
                            "Failure in disable_live_callback: %s",
                            exc,
                            extra={"event": "shadow_disable_live_failed"},
                        )
        else:
            self._breach_started_at = None
            if not self.settings.auto_disable_live_on_drift:
                self._live_disabled = False

    def _is_trending(self) -> bool:
        """Return ``True`` when drift trend exceeds configured delta."""

        if len(self._drift_samples) < 2:
            return False
        oldest = self._drift_samples[0][1]
        newest = self._drift_samples[-1][1]
        return (
            newest - oldest >= self.settings.alert_trend_delta_pct and newest > oldest
        )

    def _snapshot_payload(self, drift: float) -> dict[str, float]:
        """Build a telemetry payload including live and paper equity snapshots."""

        try:
            live_equity = float(self.live_equity_fn())
        except Exception as exc:  # noqa: BLE001 - defensive guard
            self._logger.error(
                "Failure in live equity function: %s",
                exc,
                extra={"event": "shadow_live_equity_failed"},
            )
            live_equity = 0.0
        paper_equity = float(sum(self._paper_positions.values()))
        return {
            "drift_pct": round(drift, 3),
            "live_equity": round(live_equity, 2),
            "paper_equity": round(paper_equity, 2),
        }

    def _emit_notifier(self, event: str, payload: Mapping[str, object]) -> None:
        """Send *event* to the configured notifier with exception safety."""

        if self.notifier is None:
            return
        try:
            self.notifier(event, payload)
        except Exception as exc:  # noqa: BLE001 - notifier errors are external
            self._logger.error(
                "Failure in shadow notifier: %s",
                exc,
                extra={"event": "shadow_notifier_failed", "event_name": event},
            )

    def attach_persistent_state(self, manager: "PersistentStateManager") -> None:
        """Attach persistent manager and restore saved shadow state.

        Args:
            manager: Persistent state manager coordinating snapshots.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered attach_persistent_state",
            extra={"event": "shadow_attach_persistent"},
        )
        self._persistent_state = manager
        equity, positions = manager.load_shadow_state()
        if positions:
            self._paper_positions.update(positions)
        elif equity is not None:
            self._paper_positions["__BASE__"] = float(equity)
        if self._paper_positions:
            self._logger.info(
                "Condition met: shadow_state_restored",
                extra={
                    "event": "shadow_state_restored",
                    "count": len(self._paper_positions),
                },
            )
            self._check_drift()


__all__ = ["ShadowPaperTrader"]
