"""Strategy orchestration utilities for capital and correlation controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Iterable, Mapping

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class StrategyAllocation:
    """Registered allocation metadata for a strategy."""

    capital_fraction: float
    tags: tuple[str, ...]


@dataclass(slots=True)
class ActiveAllocation:
    """Track strategies actively controlling an underlying."""

    strategy: str
    tags: tuple[str, ...]
    timestamp: datetime


class StrategyOrchestrator:
    """Coordinate strategy level capital and correlation constraints."""

    def __init__(
        self,
        *,
        risk_manager: Any,
        order_manager: Any | None = None,
        data_hub: Any | None = None,
        futures_symbol: str | None = None,
    ) -> None:
        """Initialise orchestrator with runtime dependencies.

        Args:
            risk_manager: Risk manager exposing balance information.
            order_manager: Order manager for communicating skip reasons.
            data_hub: Optional data hub for futures data lookups.
            futures_symbol: Symbol used when requesting futures context.

        Returns:
            None.

        Raises:
            None.
        """

        self._risk_manager = risk_manager
        self._order_manager = order_manager
        self._data_hub = data_hub
        self._futures_symbol = (futures_symbol or "NIFTY").strip().upper()
        self._allocations: dict[str, StrategyAllocation] = {}
        self._active: dict[str, ActiveAllocation] = {}
        self._lock = RLock()
        self._logger = LOGGER

    def register_strategy(
        self, name: str, *, capital_fraction: float, correlation_tags: Iterable[str]
    ) -> None:
        """Register strategy metadata used for orchestration decisions.

        Args:
            name: Strategy identifier.
            capital_fraction: Fraction of capital the strategy may utilise.
            correlation_tags: Iterable of tags representing correlation clusters.

        Returns:
            None.

        Raises:
            ValueError: If ``capital_fraction`` is not positive.
        """

        self._logger.debug(
            "Entered StrategyOrchestrator.register_strategy",
            extra={"event": "orchestrator_register", "strategy": name},
        )
        fraction = float(capital_fraction)
        if fraction <= 0:
            raise ValueError("capital_fraction must be positive")
        tags = tuple(sorted({tag.strip().lower() for tag in correlation_tags if tag}))
        with self._lock:
            self._allocations[name] = StrategyAllocation(
                capital_fraction=fraction, tags=tags
            )

    def filter_signal(
        self,
        signal: Any,
        indicators: Mapping[str, Any],
        position_manager: Any,
    ) -> Any | None:
        """Return signal when allowed else ``None`` if blocked.

        Args:
            signal: Candidate signal provided by strategy manager.
            indicators: Indicator snapshot for contextual checks.
            position_manager: Position manager for exposure inspection.

        Returns:
            Signal when permitted else ``None``.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered StrategyOrchestrator.filter_signal",
            extra={
                "event": "orchestrator_filter",
                "symbol": getattr(signal, "symbol", ""),
            },
        )
        strategy_name = self._resolve_strategy_name(signal)
        if not strategy_name:
            return signal
        allocation = self._allocations.get(strategy_name)
        if allocation is None:
            return signal
        action = getattr(signal, "action", "")
        if action not in {"BUY", "SELL"}:
            return signal
        underlying = self._normalize_underlying(getattr(signal, "symbol", ""))
        if not underlying:
            return signal

        if not self._has_capital_headroom(allocation, position_manager):
            self._logger.info(
                "Condition met: orchestrator_capital_block",
                extra={
                    "event": "orchestrator_capital_block",
                    "strategy": strategy_name,
                },
            )
            self._set_skip_reason("orchestrator_capital")
            return None
        if self._is_correlated(underlying, position_manager):
            self._logger.info(
                "Condition met: orchestrator_correlation_block",
                extra={
                    "event": "orchestrator_correlation_block",
                    "strategy": strategy_name,
                },
            )
            self._set_skip_reason("orchestrator_correlation")
            return None
        if not self._futures_context_ready(indicators):
            self._logger.info(
                "Condition met: orchestrator_futures_block",
                extra={
                    "event": "orchestrator_futures_block",
                    "strategy": strategy_name,
                },
            )
            self._set_skip_reason("orchestrator_futures")
            return None
        return signal

    def notify_submission(self, signal: Any, underlying: str) -> None:
        """Register that *signal* secured control over *underlying*.

        Args:
            signal: Signal associated with the submission.
            underlying: Underlying symbol controlled by the strategy.

        Returns:
            None.

        Raises:
            None.
        """

        strategy_name = self._resolve_strategy_name(signal)
        if not strategy_name:
            return
        allocation = self._allocations.get(strategy_name)
        if allocation is None:
            return
        normalized = self._normalize_underlying(underlying)
        if not normalized:
            return
        with self._lock:
            self._active[normalized] = ActiveAllocation(
                strategy=strategy_name,
                tags=allocation.tags,
                timestamp=datetime.now(timezone.utc),
            )
        self._logger.info(
            "Condition met: orchestrator_registered_active",
            extra={
                "event": "orchestrator_active",
                "strategy": strategy_name,
                "underlying": normalized,
            },
        )

    def notify_exit(self, underlying: str) -> None:
        """Release orchestration control for *underlying*.

        Args:
            underlying: Underlying symbol whose control is released.

        Returns:
            None.

        Raises:
            None.
        """

        normalized = self._normalize_underlying(underlying)
        if not normalized:
            return
        with self._lock:
            self._active.pop(normalized, None)
        self._logger.info(
            "Condition met: orchestrator_release",
            extra={"event": "orchestrator_release", "underlying": normalized},
        )

    # ------------------------------------------------------------------
    def _resolve_strategy_name(self, signal: Any) -> str:
        """Return strategy name from *signal* metadata if available.

        Args:
            signal: Signal to introspect.

        Returns:
            Extracted strategy name or empty string.

        Raises:
            None.
        """

        metadata = getattr(signal, "metadata", None)
        if isinstance(metadata, Mapping):
            value = metadata.get("strategy")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _has_capital_headroom(
        self, allocation: StrategyAllocation, position_manager: Any
    ) -> bool:
        """Return True if strategy has remaining capital headroom.

        Args:
            allocation: Strategy allocation configuration.
            position_manager: Position manager providing exposure data.

        Returns:
            bool: ``True`` when headroom exists else ``False``.

        Raises:
            None.
        """

        balance = float(getattr(self._risk_manager, "current_balance", 0.0) or 0.0)
        if balance <= 0:
            return True
        max_allocation = balance * allocation.capital_fraction
        exposure = 0.0
        getter = getattr(position_manager, "get_total_exposure", None)
        if callable(getter):
            try:
                exposure = float(getter())
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "orchestrator_exposure_lookup_failed",
                    extra={
                        "event": "orchestrator_exposure_lookup_failed",
                        "error": str(exc),
                    },
                )
        return exposure < max_allocation

    def _is_correlated(self, underlying: str, position_manager: Any) -> bool:
        """Return True when *underlying* conflicts with active allocations.

        Args:
            underlying: Normalised underlying symbol.
            position_manager: Position manager exposing current positions.

        Returns:
            bool: ``True`` when correlation prohibits new trades.

        Raises:
            None.
        """

        with self._lock:
            active = self._active.get(underlying)
            if active is not None:
                return True
        get_all = getattr(position_manager, "get_all_positions", None)
        if not callable(get_all):
            return False
        try:
            positions = get_all()
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "orchestrator_positions_failed",
                extra={"event": "orchestrator_positions_failed", "error": str(exc)},
            )
            return False
        for position in positions or []:
            symbol = getattr(position, "symbol", "")
            if self._normalize_underlying(symbol) == underlying:
                return True
        return False

    def _futures_context_ready(self, indicators: Mapping[str, Any]) -> bool:
        """Return whether futures volume context is available.

        Args:
            indicators: Indicator snapshot containing futures ratio.

        Returns:
            bool: ``True`` when context exists or not required.

        Raises:
            None.
        """

        if self._data_hub is None:
            return True
        ratio = indicators.get("futures_volume_ratio")
        return isinstance(ratio, (int, float)) and float(ratio) > 0

    def _normalize_underlying(self, symbol: str) -> str:
        """Normalise option/futures symbol into underlying token.

        Args:
            symbol: Instrument identifier from the signal.

        Returns:
            str: Upper-case underlying token.

        Raises:
            None.
        """

        token = (symbol or "").strip().upper()
        if token.endswith("CE") or token.endswith("PE"):
            token = token[:-2]
        if token.endswith("FUT"):
            token = token[:-3]
        return token

    def _set_skip_reason(self, reason: str) -> None:
        """Set skip reason on the order manager when available.

        Args:
            reason: Canonical skip reason string.

        Returns:
            None.

        Raises:
            None.
        """

        if self._order_manager is None:
            return
        setter = getattr(self._order_manager, "set_last_skip_reason", None)
        if not callable(setter):
            return
        try:
            setter(reason)
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "orchestrator_skip_reason_failed",
                extra={"event": "orchestrator_skip_reason_failed", "error": str(exc)},
            )
