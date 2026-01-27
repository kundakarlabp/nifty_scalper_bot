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
        """Initialise orchestrator with runtime dependencies."""

        self._risk_manager = risk_manager
        self._order_manager = order_manager
        self._data_hub = data_hub
        self._futures_symbol = (futures_symbol or "NIFTY").strip().upper()
        self._allocations: dict[str, StrategyAllocation] = {}
        self._active: dict[str, ActiveAllocation] = {}
        self._lock = RLock()
        self._logger = LOGGER
        # Rate limiting state
        self._last_signal_time: float = 0.0
        self._pending_underlyings: dict[str, float] = {}

    def register_strategy(
        self, name: str, *, capital_fraction: float, correlation_tags: Iterable[str]
    ) -> None:
        """Register strategy metadata used for orchestration decisions."""

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
        """
        Return signal when allowed else ``None`` if blocked.
        
        ✅ FIXES:
        - Added time guard (earliest possible check)
        - Added signal flood prevention (1 signal per 5 seconds)
        - Added SELL signal blocking for options buying mode
        - Added single-underlying constraint
        """
        import os
        import time
        
        symbol = getattr(signal, "symbol", "")
        action = getattr(signal, "action", "")
        
        self._logger.debug(
            "Entered StrategyOrchestrator.filter_signal",
            extra={
                "event": "orchestrator_filter",
                "symbol": symbol,
            },
        )
        
        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX 1: EARLY TIME GUARD
        # ═══════════════════════════════════════════════════════════
        from nifty_scalper_bot.utils.market_hours import is_market_hours_cached, get_time_status
        
        if not is_market_hours_cached():
            _, reason = get_time_status()
            # Log throttled (handled by market_hours module)
            return None
        
        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX 2: BLOCK FUTURES TRADING
        # ═══════════════════════════════════════════════════════════
        if self._is_futures(symbol):
            self._logger.info(
                "Orchestrator blocked Futures trade (Options Only Mode)",
                extra={"event": "orchestrator_futures_blocked", "symbol": symbol}
            )
            return None

        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX 3: BLOCK SELL SIGNALS FOR OPTIONS BUYING STRATEGY
        # ═══════════════════════════════════════════════════════════
        # For options BUYING strategy, SELL should mean "close position"
        # not "open short". Block naked SELL entries.
        options_long_only = os.getenv("OPTIONS_LONG_ONLY", "true").lower() == "true"
        
        if options_long_only and action == "SELL":
            # Check if this is a position close or a new entry
            is_position_close = False
            
            if position_manager:
                # Check if we have an open position on this symbol
                try:
                    pos = position_manager.get_position(symbol)
                    if pos and getattr(pos, "quantity", 0) > 0:
                        is_position_close = True
                except Exception:
                    pass
            
            if not is_position_close:
                self._logger.debug(
                    f"🛡️ SELL blocked (Options Long Only): {symbol}",
                    extra={"event": "orchestrator_sell_blocked", "symbol": symbol}
                )
                return None
        
        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX 4: SIGNAL FLOOD PREVENTION (Rate Limiting)
        # ═══════════════════════════════════════════════════════════
        # Only allow 1 entry signal every 5 seconds globally
        if not hasattr(self, "_last_signal_time"):
            self._last_signal_time = 0.0
        
        signal_cooldown = float(os.getenv("ORCHESTRATOR_SIGNAL_COOLDOWN", "5.0"))
        now = time.time()
        
        if action in {"BUY", "SELL"} and (now - self._last_signal_time) < signal_cooldown:
            self._logger.debug(
                f"⏳ Signal rate limited: {symbol} (cooldown: {signal_cooldown - (now - self._last_signal_time):.1f}s)",
                extra={"event": "orchestrator_rate_limit", "symbol": symbol}
            )
            return None
        
        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX 5: SINGLE UNDERLYING CONSTRAINT
        # ═══════════════════════════════════════════════════════════
        # Prevent multiple signals for same underlying (e.g., NIFTY)
        # This stops the flood of 150CE, 150PE, 200CE, 200PE, etc.
        underlying = self._normalize_underlying(symbol)
        
        if not hasattr(self, "_pending_underlyings"):
            self._pending_underlyings = {}
        
        pending_cooldown = float(os.getenv("UNDERLYING_SIGNAL_COOLDOWN", "30.0"))
        last_underlying_signal = self._pending_underlyings.get(underlying, 0.0)
        
        if action in {"BUY"} and (now - last_underlying_signal) < pending_cooldown:
            self._logger.debug(
                f"🛡️ Underlying rate limited: {underlying} already has pending signal",
                extra={"event": "orchestrator_underlying_limit", "symbol": symbol}
            )
            return None
        
        # ═══════════════════════════════════════════════════════════
        # EXISTING LOGIC (Strategy allocation, correlation checks)
        # ═══════════════════════════════════════════════════════════
        strategy_name = self._resolve_strategy_name(signal)
        if not strategy_name:
            # Update rate limit timestamp for untracked signals
            if action in {"BUY"}:
                self._last_signal_time = now
                self._pending_underlyings[underlying] = now
            return signal
            
        allocation = self._allocations.get(strategy_name)
        if allocation is None:
            if action in {"BUY"}:
                self._last_signal_time = now
                self._pending_underlyings[underlying] = now
            return signal
            
        if action not in {"BUY", "SELL"}:
            return signal
            
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
            self._logger.debug(
                "Futures context missing, but allowing Option trade.",
                extra={"event": "orchestrator_futures_context_missing", "symbol": symbol}
            )

        # ✅ Update rate limit timestamps on successful pass
        if action in {"BUY"}:
            self._last_signal_time = now
            self._pending_underlyings[underlying] = now
        
        return signal

    def notify_submission(self, signal: Any, underlying: str) -> None:
        """Register that *signal* secured control over *underlying*."""

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
        """Release orchestration control for *underlying*."""

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
    def _is_futures(self, symbol: str) -> bool:
        """Check if symbol is a futures contract (Safe Version)."""
        normalized = symbol.strip().upper()
        
        # Options (CE/PE) are NOT futures
        if normalized.endswith("CE") or normalized.endswith("PE"):
            return False
            
        # Only true Futures contain "FUT"
        return "FUT" in normalized

    def _resolve_strategy_name(self, signal: Any) -> str:
        """Return strategy name from *signal* metadata if available."""
        metadata = getattr(signal, "metadata", None)
        if isinstance(metadata, Mapping):
            value = metadata.get("strategy")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _has_capital_headroom(
        self, allocation: StrategyAllocation, position_manager: Any
    ) -> bool:
        """Return True if strategy has remaining capital headroom."""
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
        """Return True when *underlying* conflicts with active allocations."""
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
        """Return whether futures volume context is available."""
        if self._data_hub is None:
            return True
        ratio = indicators.get("futures_volume_ratio")
        return isinstance(ratio, (int, float)) and float(ratio) > 0

    def _normalize_underlying(self, symbol: str) -> str:
        """Normalise option/futures symbol into underlying token."""
        token = (symbol or "").strip().upper()
        if token.endswith("CE") or token.endswith("PE"):
            token = token[:-2]
        if token.endswith("FUT"):
            token = token[:-3]
        return token

    def _set_skip_reason(self, reason: str) -> None:
        """Set skip reason on the order manager when available."""
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
