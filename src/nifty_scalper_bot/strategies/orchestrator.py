"""
Strategy orchestration utilities for capital and correlation controls.

✅ PRODUCTION FIX: _has_capital_headroom now excludes orphan positions
"""

from __future__ import annotations

import os
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
        try:
            from nifty_scalper_bot.utils.market_hours import is_market_hours_cached, get_time_status
            
            if not is_market_hours_cached():
                _, reason = get_time_status()
                return None
        except ImportError:
            pass
        
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
        options_long_only = os.getenv("OPTIONS_LONG_ONLY", "true").lower() == "true"
        
        if options_long_only and action == "SELL":
            is_position_close = False
            
            if position_manager:
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
        signal_cooldown = float(os.getenv("ORCHESTRATOR_SIGNAL_COOLDOWN", "5.0"))
        now = time.time()
        
        if action in {"BUY", "SELL"} and (now - self._last_signal_time) < signal_cooldown:
            self._logger.debug(
                f"⏳ Signal rate limited: {symbol}",
                extra={"event": "orchestrator_rate_limit", "symbol": symbol}
            )
            return None
        
        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX 5: SINGLE UNDERLYING CONSTRAINT
        # ═══════════════════════════════════════════════════════════
        underlying = self._normalize_underlying(symbol)
        
        pending_cooldown = float(os.getenv("UNDERLYING_SIGNAL_COOLDOWN", "30.0"))
        last_underlying_signal = self._pending_underlyings.get(underlying, 0.0)
        
        if action in {"BUY"} and (now - last_underlying_signal) < pending_cooldown:
            self._logger.debug(
                f"🛡️ Underlying rate limited: {underlying}",
                extra={"event": "orchestrator_underlying_limit", "symbol": symbol}
            )
            return None
        
        # ═══════════════════════════════════════════════════════════
        # STRATEGY ALLOCATION CHECKS
        # ═══════════════════════════════════════════════════════════
        strategy_name = self._resolve_strategy_name(signal)
        if not strategy_name:
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

        # Update rate limit timestamps on successful pass
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
        """Check if symbol is a futures contract."""
        normalized = symbol.strip().upper()
        
        if normalized.endswith("CE") or normalized.endswith("PE"):
            return False
            
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
        """
        Return True if strategy has remaining capital headroom.
        
        ✅ PRODUCTION FIX: Excludes orphan/manual positions from exposure calculation.
        This prevents orphan trades from blocking ALL new signals.
        """
        balance = float(getattr(self._risk_manager, "current_balance", 0.0) or 0.0)
        if balance <= 0:
            self._logger.debug("💰 Capital check: No balance info, allowing trade")
            return True
            
        max_allocation = balance * allocation.capital_fraction
        
        # ═══════════════════════════════════════════════════════════
        # ✅ FIX: Calculate exposure excluding orphan positions
        # ═══════════════════════════════════════════════════════════
        exposure = 0.0
        orphan_exposure = 0.0
        tracked_count = 0
        orphan_count = 0
        
        # First try the simple get_total_exposure method
        getter = getattr(position_manager, "get_total_exposure", None)
        get_all = getattr(position_manager, "get_all_positions", None)
        
        # Use detailed calculation to exclude orphans
        if callable(get_all):
            try:
                positions = get_all()
                for pos in positions or []:
                    qty = abs(float(getattr(pos, "quantity", 0) or 0))
                    if qty <= 0:
                        continue
                        
                    price = float(
                        getattr(pos, "entry_price", 0) or 
                        getattr(pos, "avg_price", 0) or 
                        getattr(pos, "last_price", 0) or 0
                    )
                    pos_exposure = qty * price
                    
                    # Check if this is an orphan position
                    strategy = (
                        getattr(pos, "strategy", "") or 
                        getattr(pos, "strategy_name", "") or 
                        getattr(pos, "tag", "") or
                        ""
                    )
                    strategy_lower = strategy.lower().strip()
                    
                    # Identify orphan positions
                    is_orphan = (
                        not strategy_lower or
                        strategy_lower in ("manual", "unknown", "manual/unknown", "none", "")
                    )
                    
                    if is_orphan:
                        orphan_exposure += pos_exposure
                        orphan_count += 1
                    else:
                        exposure += pos_exposure
                        tracked_count += 1
                        
            except Exception as exc:
                self._logger.debug(f"orchestrator_exposure_calc_failed: {exc}")
                # Fall back to simple method
                if callable(getter):
                    try:
                        exposure = float(getter())
                    except Exception:
                        pass
        elif callable(getter):
            try:
                exposure = float(getter())
            except Exception as exc:
                self._logger.debug(f"orchestrator_exposure_lookup_failed: {exc}")
        
        # Log detailed capital check info
        total_exposure = exposure + orphan_exposure
        self._logger.info(
            f"💰 Capital Check: tracked_exposure={exposure:.2f} | orphan_exposure={orphan_exposure:.2f} | "
            f"max_allocation={max_allocation:.2f} | balance={balance:.2f} | "
            f"fraction={allocation.capital_fraction:.3f} | "
            f"tracked_positions={tracked_count} | orphan_positions={orphan_count}",
            extra={
                "event": "orchestrator_capital_check",
                "tracked_exposure": exposure,
                "orphan_exposure": orphan_exposure,
                "max_allocation": max_allocation,
                "result": "ALLOW" if exposure < max_allocation else "BLOCK"
            }
        )
        
        # ✅ Only check tracked exposure, not orphan exposure
        return exposure < max_allocation

    def _has_capital_headroom_quick(self) -> bool:
        """
        Quick capital check without position manager.
        Used by base_elite.py for early filtering.
        
        ✅ FIX: This method was missing, causing AttributeError.
        Returns True to allow signal generation (detailed check happens later).
        """
        balance = float(getattr(self._risk_manager, "current_balance", 0.0) or 0.0)
        return balance > 0  # Allow if we have any balance

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
        except Exception as exc:
            self._logger.debug(f"orchestrator_positions_failed: {exc}")
            return False
            
        for position in positions or []:
            symbol = getattr(position, "symbol", "")
            # ✅ FIX: Also check if position is orphan before blocking
            strategy = (
                getattr(position, "strategy", "") or 
                getattr(position, "strategy_name", "") or ""
            )
            is_orphan = strategy.lower().strip() in ("manual", "unknown", "manual/unknown", "none", "")
            
            # Only block correlation for tracked positions, not orphans
            if not is_orphan and self._normalize_underlying(symbol) == underlying:
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
        except Exception as exc:
            self._logger.debug(f"orchestrator_skip_reason_failed: {exc}")


__all__ = ["StrategyOrchestrator", "StrategyAllocation", "ActiveAllocation"]
