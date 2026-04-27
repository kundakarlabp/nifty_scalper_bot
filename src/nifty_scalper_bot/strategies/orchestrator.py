"""
Strategy orchestration utilities for capital and correlation controls.

✅ PRODUCTION FIX v2 (Feb 2, 2026):
- _has_capital_headroom now excludes orphan positions
- Rate limit logs changed from DEBUG → INFO for visibility
- Reduced default rate limits (5s → 2s, 30s → 10s)
- Added comprehensive logging for all filter decisions
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from threading import RLock
from typing import Any, Iterable, Mapping

from nifty_scalper_bot.config.settings import get_settings
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
        self._skip_reason: str = ""
        # ✅ FIX (6 Feb 2026): Direction conflict guard — only one direction at a time
        self._active_direction: str | None = None  # "CE" or "PE" or None
        self._active_direction_symbol: str = ""
        self._direction_lock_time: float = 0.0

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

    def _set_skip_reason(self, reason: str) -> None:
        """Set skip reason for diagnostics."""
        self._skip_reason = reason

    def get_skip_reason(self) -> str:
        """Get last skip reason."""
        return self._skip_reason

    def filter_signal(
        self,
        signal: Any,
        indicators: Mapping[str, Any],
        position_manager: Any,
    ) -> Any | None:
        """
        Return signal when allowed else ``None`` if blocked.

        ✅ PRODUCTION FIX v2:
        - All filter rejections now log at INFO level for visibility
        - Reduced default rate limits for better signal throughput
        - Added time guard (earliest possible check)
        - Added signal flood prevention
        - Added SELL signal blocking for options buying mode
        - Added single-underlying constraint
        """
        import time

        symbol = getattr(signal, "symbol", "")
        action = getattr(signal, "action", "")
        confidence = getattr(signal, "confidence", 0.0)

        self._skip_reason = ""  # Reset skip reason

        self._logger.debug(
            f"🔍 Orchestrator evaluating: {symbol} | {action} | conf={confidence:.2f}",
            extra={
                "event": "orchestrator_filter_start",
                "symbol": symbol,
                "action": action,
            },
        )

        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX 1: EARLY TIME GUARD
        # ═══════════════════════════════════════════════════════════
        try:
            from nifty_scalper_bot.utils.market_hours import is_market_hours_cached

            settings = get_settings()
            if not settings.allow_offmarket_trading:
                if not is_market_hours_cached():
                    self._logger.warning(
                        "⛔ BLOCKED: Market hours filter active",
                        extra={
                            "event": "orchestrator_market_hours_blocked",
                            "symbol": symbol,
                        },
                    )
                    self._set_skip_reason("market_closed")
                    return None
            else:
                self._logger.warning(
                    "⚠️ OFF-MARKET TRADING ENABLED (ENV OVERRIDE)",
                    extra={
                        "event": "orchestrator_offmarket_override",
                        "symbol": symbol,
                    },
                )
        except Exception as e:
            self._logger.error("Failure in filter_signal: %s", e)

        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX 2: BLOCK FUTURES TRADING
        # ═══════════════════════════════════════════════════════════
        if self._is_futures(symbol):
            self._logger.info(
                f"🚫 FUTURES BLOCKED: {symbol} (Options Only Mode)",
                extra={"event": "orchestrator_futures_blocked", "symbol": symbol},
            )
            self._set_skip_reason("futures_blocked")
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
                except Exception as e:
                    __import__("logging").getLogger(__name__).exception("[CRITICAL] unhandled exception", exc_info=True)
                    raise

            if not is_position_close:
                self._logger.info(
                    f"🛡️ SELL BLOCKED: {symbol} (Options Long Only mode, no position to close)",
                    extra={"event": "orchestrator_sell_blocked", "symbol": symbol},
                )
                self._set_skip_reason("sell_blocked_long_only")
                return None

        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX 6: DIRECTION CONFLICT GUARD (No simultaneous CE+PE)
        # ✅ Added 6 Feb 2026: Prevents bot from taking both CE and PE trades
        # ═══════════════════════════════════════════════════════════
        if action == "BUY":
            import time as _t

            _sym_upper = symbol.upper()
            _direction = (
                "CE"
                if _sym_upper.endswith("CE")
                else ("PE" if _sym_upper.endswith("PE") else None)
            )
            _dir_cooldown = float(os.getenv("DIRECTION_LOCK_SECONDS", "20"))

            if _direction and self._active_direction:
                _time_since_lock = _t.time() - self._direction_lock_time
                if (
                    self._active_direction != _direction
                    and _time_since_lock < _dir_cooldown
                ):
                    self._logger.info(
                        f"🛡️ DIRECTION CONFLICT: {symbol} is {_direction} but "
                        f"active direction is {self._active_direction} "
                        f"({self._active_direction_symbol}) | "
                        f"Wait {_dir_cooldown - _time_since_lock:.0f}s",
                        extra={
                            "event": "orchestrator_direction_conflict",
                            "symbol": symbol,
                        },
                    )
                    self._set_skip_reason("direction_conflict")
                    self._logger.warning(
                        "EVENT|signal_blocked|symbol=%s|reason=%s",
                        symbol,
                        "direction_lock",
                    )
                    return None

            # Direction lock deferred to after ALL checks pass (was here before,
            # but capital/correlation rejections would leave direction locked)

        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX 4: SIGNAL FLOOD PREVENTION (Rate Limiting)
        # ✅ CHANGED: Default reduced from 5.0 to 2.0 seconds
        # ✅ CHANGED: Log level from DEBUG to INFO
        # ═══════════════════════════════════════════════════════════
        signal_cooldown = float(os.getenv("SIGNAL_COOLDOWN_S", "0.5"))
        now = time.time()

        if (
            action in {"BUY", "SELL"}
            and (now - self._last_signal_time) < signal_cooldown
        ):
            elapsed = now - self._last_signal_time
            self._logger.info(  # ✅ CHANGED: DEBUG → INFO
                f"⏳ RATE LIMIT: {symbol} | Wait {signal_cooldown - elapsed:.1f}s (global {signal_cooldown}s cooldown)",
                extra={
                    "event": "orchestrator_rate_limit",
                    "symbol": symbol,
                    "cooldown": signal_cooldown,
                },
            )
            self._set_skip_reason("global_rate_limit")
            self._logger.warning(
                "EVENT|signal_blocked|symbol=%s|reason=%s",
                symbol,
                "global_cooldown",
            )
            return None

        # ═══════════════════════════════════════════════════════════
        # 🛡️ FIX 5: SINGLE UNDERLYING CONSTRAINT
        # ✅ CHANGED: Default reduced from 30.0 to 10.0 seconds
        # ✅ CHANGED: Log level from DEBUG to INFO
        # ═══════════════════════════════════════════════════════════
        underlying = self._normalize_underlying(symbol)

        pending_cooldown = float(
            os.getenv("UNDERLYING_SIGNAL_COOLDOWN", "10.0")
        )  # ✅ Reduced
        last_underlying_signal = self._pending_underlyings.get(underlying, 0.0)

        if action in {"BUY"} and (now - last_underlying_signal) < pending_cooldown:
            elapsed = now - last_underlying_signal
            self._logger.info(  # ✅ CHANGED: DEBUG → INFO
                f"🛡️ UNDERLYING LIMIT: {symbol} | {underlying} | Wait {pending_cooldown - elapsed:.1f}s",
                extra={
                    "event": "orchestrator_underlying_limit",
                    "symbol": symbol,
                    "underlying": underlying,
                },
            )
            self._set_skip_reason("underlying_rate_limit")
            self._logger.warning(
                "EVENT|signal_blocked|symbol=%s|reason=%s",
                symbol,
                "underlying_cooldown",
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
            self._logger.info(
                f"✅ SIGNAL PASSED (no strategy allocation): {symbol} | {action}",
                extra={"event": "orchestrator_pass_no_allocation", "symbol": symbol},
            )
            return signal

        allocation = self._allocations.get(strategy_name)
        if allocation is None:
            if action in {"BUY"}:
                self._last_signal_time = now
                self._pending_underlyings[underlying] = now
            self._logger.info(
                f"✅ SIGNAL PASSED (strategy not registered): {symbol} | {action} | {strategy_name}",
                extra={"event": "orchestrator_pass_unregistered", "symbol": symbol},
            )
            return signal

        if action not in {"BUY", "SELL"}:
            return signal

        if not underlying:
            return signal

        if not self._has_capital_headroom(allocation, position_manager):
            self._logger.info(
                f"💰 CAPITAL BLOCK: {symbol} | Strategy: {strategy_name}",
                extra={
                    "event": "orchestrator_capital_block",
                    "strategy": strategy_name,
                    "symbol": symbol,
                },
            )
            self._set_skip_reason("orchestrator_capital")
            self._logger.warning("⛔ FILTER REJECT: reason=capital_headroom")
            return None

        if self._is_correlated(underlying, position_manager):
            self._logger.info(
                f"🔗 CORRELATION BLOCK: {symbol} | {underlying} already active",
                extra={
                    "event": "orchestrator_correlation_block",
                    "strategy": strategy_name,
                    "symbol": symbol,
                },
            )
            self._set_skip_reason("orchestrator_correlation")
            self._logger.warning("⛔ FILTER REJECT: reason=correlation_block")
            return None

        if not self._futures_context_ready(indicators):
            self._logger.debug(
                "Futures context missing, but allowing Option trade.",
                extra={
                    "event": "orchestrator_futures_context_missing",
                    "symbol": symbol,
                },
            )

        # Update rate limit timestamps on successful pass
        if action in {"BUY"}:
            self._last_signal_time = now
            self._pending_underlyings[underlying] = now

            # Lock direction ONLY after ALL checks pass
            _sym_upper = symbol.upper()
            _final_direction = (
                "CE"
                if _sym_upper.endswith("CE")
                else ("PE" if _sym_upper.endswith("PE") else None)
            )
            if _final_direction:
                import time as _t

                self._active_direction = _final_direction
                self._active_direction_symbol = symbol
                self._direction_lock_time = _t.time()

        self._logger.info(
            "ORCHESTRATOR_PREFILTER_PASSED symbol=%s action=%s conf=%.2f strategy=%s",
            symbol,
            action,
            confidence,
            strategy_name,
            extra={
                "event": "orchestrator_prefilter_passed",
                "symbol": symbol,
                "action": action,
            },
        )

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
            self._pending_underlyings.pop(normalized, None)
            # ✅ FIX (6 Feb 2026): Clear direction lock on exit
            self._active_direction = None
            self._active_direction_symbol = ""
            self._direction_lock_time = 0.0
            self._logger.info(f"🔓 Direction lock cleared on exit: {normalized}")
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
        Check if there is remaining capital for this strategy allocation.

        ✅ PRODUCTION FIX: Excludes orphan/manual positions from exposure calculation.
        This prevents orphan trades from blocking ALL new signals.
        """
        balance = float(getattr(self._risk_manager, "current_balance", 0.0) or 0.0)
        max_allocation = balance * allocation.capital_fraction

        # ✅ FIX: Calculate exposure excluding orphan positions
        exposure = 0.0
        tracked_exposure = 0.0
        orphan_exposure = 0.0
        tracked_count = 0
        orphan_count = 0

        get_all = getattr(position_manager, "get_all_positions", None)
        if callable(get_all):
            positions = get_all()

            for pos in positions or []:
                try:
                    pos_symbol = str(getattr(pos, "symbol", "") or "")
                    if not pos_symbol.startswith("NFO:NIFTY"):
                        continue
                    qty = abs(float(getattr(pos, "quantity", 0) or 0))
                    price = float(
                        getattr(pos, "entry_price", 0)
                        or getattr(pos, "avg_price", 0)
                        or 0
                    )
                    pos_exposure = qty * price

                    if pos_exposure <= 0:
                        continue

                    # Check if this is an orphan position
                    strategy = (
                        getattr(pos, "strategy", "")
                        or getattr(pos, "strategy_name", "")
                        or ""
                    )

                    # Identify orphan positions
                    is_orphan = not strategy or strategy.lower().strip() in (
                        "manual",
                        "unknown",
                        "manual/unknown",
                        "none",
                        "",
                    )

                    if is_orphan:
                        orphan_exposure += pos_exposure
                        orphan_count += 1
                    else:
                        tracked_exposure += pos_exposure
                        tracked_count += 1

                except (TypeError, ValueError, AttributeError):
                    continue

        total_exposure = tracked_exposure + orphan_exposure
        result = tracked_exposure < max_allocation

        self._logger.info(
            f"💰 Capital Check: tracked={tracked_exposure:.2f} | orphan={orphan_exposure:.2f} | "
            f"max={max_allocation:.2f} | balance={balance:.2f} | "
            f"tracked_pos={tracked_count} | orphan_pos={orphan_count} | "
            f"result={'ALLOW' if result else 'BLOCK'}",
            extra={
                "event": "capital_headroom_check",
                "tracked_exposure": tracked_exposure,
                "orphan_exposure": orphan_exposure,
                "max_allocation": max_allocation,
                "result": result,
            },
        )

        # ✅ Only check tracked exposure, not orphan exposure
        return result

    def _has_capital_headroom_quick(self) -> bool:
        """
        Quick check used by base_elite.py.
        ✅ FIX: Was missing, causing AttributeError.
        """
        balance = float(getattr(self._risk_manager, "current_balance", 0.0) or 0.0)
        return balance > 0

    def _is_correlated(self, underlying: str, position_manager: Any) -> bool:
        """
        Check if taking position in *underlying* would violate correlation rules.

        ✅ PRODUCTION FIX: Excludes orphan positions from correlation blocking.
        """
        with self._lock:
            active = self._active.get(underlying)

        if active is None:
            return False

        # Check if the active position is an orphan
        get_all = getattr(position_manager, "get_all_positions", None)
        if callable(get_all):
            positions = get_all()
            for pos in positions or []:
                try:
                    pos_symbol = str(getattr(pos, "symbol", "") or "")
                    if not pos_symbol.startswith("NFO:NIFTY"):
                        continue
                    pos_underlying = self._normalize_underlying(pos_symbol)

                    if pos_underlying != underlying:
                        continue

                    strategy = (
                        getattr(pos, "strategy", "")
                        or getattr(pos, "strategy_name", "")
                        or ""
                    )

                    is_orphan = not strategy or strategy.lower().strip() in (
                        "manual",
                        "unknown",
                        "manual/unknown",
                        "none",
                        "",
                    )

                    if is_orphan:
                        # Don't block correlation for orphan positions
                        self._logger.debug(
                            f"Ignoring orphan position in correlation check: {pos_symbol}",
                            extra={
                                "event": "correlation_orphan_skip",
                                "symbol": pos_symbol,
                            },
                        )
                        return False

                except (TypeError, ValueError, AttributeError):
                    continue

        return True

    def _futures_context_ready(self, indicators: Mapping[str, Any]) -> bool:
        """Return True when futures context indicators are available."""
        fut_vol = indicators.get("futures_volume")
        if fut_vol is None:
            return False
        try:
            return float(fut_vol) > 0
        except (TypeError, ValueError):
            return False

    def _normalize_underlying(self, symbol: str) -> str:
        """Extract underlying name from symbol."""
        if not symbol:
            return ""
        normalized = symbol.strip().upper()

        # Remove exchange prefix
        for prefix in ("NFO:", "NSE:", "BSE:"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break

        # Extract underlying (e.g., "NIFTY" from "NIFTY2620324650CE")
        for underlying in ("NIFTY", "BANKNIFTY", "FINNIFTY"):
            if normalized.startswith(underlying):
                return underlying

        return normalized.split()[0] if normalized else ""


__all__ = ["StrategyOrchestrator", "StrategyAllocation", "ActiveAllocation"]
