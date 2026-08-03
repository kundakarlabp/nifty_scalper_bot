"""
Base abstractions and helpers for elite strategies.
Production-Grade: Optimized Dispatch & Attribute Injection.
Fixed: 'Signal' object has no attribute 'strategy_name' crash.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategyConfig,
)
from nifty_scalper_bot.strategies.signal_generator import Signal, Strategy
from nifty_scalper_bot.strategies.signal_quality import build_trade_quality_evidence
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class EliteSignal:
    """
    Container for elite strategy signal output.
    Optimized with __slots__ for reduced memory footprint.
    """

    symbol: str
    signal: str  # Standardized name (was 'side')
    confidence: float
    entry_price: float
    stop_loss: float | None
    target: float | None
    quantity: int = 1
    strategy_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Backwards compatibility fields
    take_profit_1: float | None = None
    take_profit_2: float | None = None
    side: str = field(init=False)
    action: str = field(init=False)

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", self.signal)
        object.__setattr__(self, "action", self.signal)
        if self.target and not self.take_profit_1:
            object.__setattr__(self, "take_profit_1", self.target)

    def to_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.signal,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "quantity": self.quantity,
            "strategy": self.strategy_name,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


def _as_confidence_fraction(value: Any) -> float:
    """Normalise a configured confidence threshold to a 0..1 fraction.

    EliteStrategyConfig.min_confidence is expressed in percent while
    EliteSignal.confidence is a 0..1 fraction. The previous unconditional
    ``/ 100.0`` silently disabled the gate for any config already written as a
    fraction (0.6 became 0.006). Values at or below 1.0 are treated as
    fractions; anything larger is treated as a percentage.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number <= 0.0:
        return 0.0
    if number > 1.0:
        number /= 100.0
    return min(number, 1.0)


class EliteStrategy(Strategy):
    """
    World-Class Base Class for Elite Strategies.
    Implements the Bridge Pattern and Signal Patching.
    """

    def __init__(self, config: EliteStrategyConfig, indicator_engine: Any) -> None:
        # Auto-detect name from config class
        name = config.__class__.__name__.replace("Config", "").replace("Strategy", "")

        # ✅ FIX 1: Satisfy parent requirements
        params = asdict(config) if hasattr(config, "__dataclass_fields__") else {}
        super().__init__(name=name, parameters=params)

        self._config = config
        self._indicator_engine = indicator_engine
        self._last_signal_at: datetime | None = None
        self._signals_generated = 0
        self._last_signal: EliteSignal | None = None
        self._consecutive_evaluation_failures = 0
        self._last_evaluation_error: str | None = None

        # Inspect signature once at startup
        sig = inspect.signature(self._evaluate_signal)
        self._is_legacy_signature = len(sig.parameters) == 0

        if self._is_legacy_signature:
            LOGGER.debug(f"⚠️ {self.name}: Running in Legacy Mode (Pull-Based)")
        else:
            LOGGER.debug(f"🚀 {self.name}: Running in Modern Mode (Push-Based)")

    def get_required_indicators(self) -> list[str]:
        """Args: none. Returns: list[str]. Raises: Exception."""
        LOGGER.debug("Entered EliteStrategy.get_required_indicators")
        try:
            LOGGER.info(
                "Condition met: using base indicator set",
                extra={
                    "event": "elite_strategy_required_indicators",
                    "strategy": self.name,
                },
            )
            return []
        except Exception as exc:
            LOGGER.error("Failure in get_required_indicators: %s", exc, exc_info=exc)
            return []

    def generate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> Signal | None:
        """Args: symbol, indicators, price, pos. Returns: Signal|None. Raises: Err."""
        LOGGER.debug("Entered EliteStrategy.generate_signal")
        try:
            if not self._config.enabled:
                LOGGER.info(
                    "Condition met: strategy disabled",
                    extra={"event": "elite_strategy_disabled", "strategy": self.name},
                )
                return None

            if not symbol:
                LOGGER.info(
                    "Condition met: missing symbol",
                    extra={
                        "event": "elite_strategy_missing_symbol",
                        "strategy": self.name,
                    },
                )
                return None

            if current_price <= 0:
                LOGGER.info(
                    "Condition met: invalid current price",
                    extra={
                        "event": "elite_strategy_invalid_price",
                        "strategy": self.name,
                        "price": current_price,
                    },
                )
                return None

            if indicators is None:
                LOGGER.info(
                    "Condition met: missing indicators payload",
                    extra={
                        "event": "elite_strategy_missing_indicators",
                        "strategy": self.name,
                    },
                )
                return None

            # ✅ Early exit if capital is exhausted (prevents wasted computation)
            if hasattr(self, "_orchestrator") and self._orchestrator:
                if not self._orchestrator._has_capital_headroom_quick():
                    LOGGER.info(
                        "Condition met: capital headroom exhausted",
                        extra={
                            "event": "elite_strategy_no_capital",
                            "strategy": self.name,
                        },
                    )
                    return None

            indicators_payload: dict[str, Any] = dict(indicators)
            min_bars_required = int(getattr(self, "MIN_BARS_REQUIRED", 0) or 0)
            if min_bars_required > 0:
                available_history = self._available_history_count(indicators_payload)
                if (
                    available_history is not None
                    and available_history < min_bars_required
                ):
                    LOGGER.info(
                        "STRATEGY_SKIPPED_HISTORY_COLD strategy=%s symbol=%s history_count=%s min_bars=%s",
                        self.name,
                        symbol,
                        available_history,
                        min_bars_required,
                        extra={
                            "event": "STRATEGY_SKIPPED_HISTORY_COLD",
                            "strategy": self.name,
                            "symbol": symbol,
                            "history_count": available_history,
                            "min_bars": min_bars_required,
                        },
                    )
                    self._no_vote("history_cold")
                    return None
            elite_signal = self._evaluate_signal(
                symbol=symbol,
                indicators=indicators_payload,
                current_price=current_price,
                position=position,
            )

            # The evaluation completed without raising, whatever it decided.
            self._consecutive_evaluation_failures = 0
            if elite_signal:
                self._stamp_setup_anchor(elite_signal, indicators_payload)
                self._stamp_structural_setup_id(elite_signal, indicators_payload)
                quality = self._stamp_quality_evidence(elite_signal, indicators_payload)
                if (
                    str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper()
                    == "LIVE"
                    and bool(quality.get("quality_spread_observed"))
                    and not bool(quality.get("quality_spread_pass"))
                ):
                    self._no_vote("wide_spread")
                    return None
                min_conf = _as_confidence_fraction(self._config.min_confidence)
                if float(elite_signal.confidence) < min_conf:
                    self._no_vote("below_strategy_min_confidence")
                    LOGGER.info(
                        "Condition met: below strategy min confidence",
                        extra={
                            "event": "elite_strategy_below_min_conf",
                            "strategy": self.name,
                        },
                    )
                    return None
                LOGGER.info(
                    "Condition met: elite signal generated",
                    extra={"event": "elite_strategy_signal", "strategy": self.name},
                )
                return self._process_signal(elite_signal)

            LOGGER.debug(
                "No signal generated",
                extra={"event": "elite_strategy_no_signal", "strategy": self.name},
            )
        except Exception as exc:
            self._record_evaluation_failure(exc)

        return None

    def _record_evaluation_failure(self, exc: BaseException) -> None:
        """Surface a crashed evaluation instead of it looking like no signal.

        A raised exception and a strategy that legitimately declined were both
        reported as "no signal". VWAPPro once produced no vote on every
        evaluation because a None reached a %.4f specifier inside its own log
        call and this handler swallowed the TypeError.
        """
        self._consecutive_evaluation_failures = (
            int(getattr(self, "_consecutive_evaluation_failures", 0)) + 1
        )
        self._last_evaluation_error = f"{type(exc).__name__}: {exc}"
        self._no_vote("evaluation_failed")
        LOGGER.error(
            "STRATEGY_EVALUATION_FAILED strategy=%s error_type=%s consecutive=%s error=%s",
            self.name,
            type(exc).__name__,
            self._consecutive_evaluation_failures,
            exc,
            exc_info=exc,
            extra={
                "event": "STRATEGY_EVALUATION_FAILED",
                "strategy": self.name,
                "error_type": type(exc).__name__,
                "consecutive_failures": self._consecutive_evaluation_failures,
            },
        )

    @property
    def evaluation_health(self) -> dict[str, Any]:
        """Report evaluation failure state for health/status surfaces."""
        failures = int(getattr(self, "_consecutive_evaluation_failures", 0))
        return {
            "strategy": self.name,
            "consecutive_evaluation_failures": failures,
            "last_evaluation_error": getattr(self, "_last_evaluation_error", None),
            "healthy": failures == 0,
        }

    def evaluate(self) -> Signal | None:
        """Args: none. Returns: Signal|None. Raises: Exception."""
        LOGGER.debug("Entered EliteStrategy.evaluate")
        try:
            if not self._config.enabled:
                LOGGER.info(
                    "Condition met: strategy disabled",
                    extra={
                        "event": "elite_strategy_disabled_poll",
                        "strategy": self.name,
                    },
                )
                return None

            if self._last_signal_at:
                elapsed = (
                    datetime.now(timezone.utc) - self._last_signal_at
                ).total_seconds()
                if elapsed < self._config.cooldown_seconds:
                    LOGGER.info(
                        "Condition met: cooldown active",
                        extra={
                            "event": "elite_strategy_cooldown",
                            "strategy": self.name,
                            "elapsed": elapsed,
                        },
                    )
                    return None

            if self._is_legacy_signature:
                elite_signal = self._evaluate_signal()  # type: ignore
                if elite_signal:
                    LOGGER.info(
                        "Condition met: legacy signal generated",
                        extra={
                            "event": "elite_strategy_legacy_signal",
                            "strategy": self.name,
                        },
                    )
                    return self._process_signal(elite_signal)
                LOGGER.debug(
                    "No legacy signal generated",
                    extra={
                        "event": "elite_strategy_legacy_no_signal",
                        "strategy": self.name,
                    },
                )
                return None

            symbol = getattr(self._config, "symbol", None)
            if not symbol:
                LOGGER.info(
                    "Condition met: missing symbol",
                    extra={
                        "event": "elite_strategy_missing_symbol",
                        "strategy": self.name,
                    },
                )
                return None

            req_inds = self.get_required_indicators()
            indicators = self._indicator_engine.get_indicators(symbol, list(req_inds))
            ltp = float(indicators.get("ltp") or 0.0)

            return self.generate_signal(symbol, indicators, ltp)
        except Exception as exc:
            self._record_evaluation_failure(exc)

        return None

    @staticmethod
    def _available_history_count(indicators: Mapping[str, Any]) -> int | None:
        """Return best available history count from strategy context."""
        for key in (
            "history_resolved_count",
            "history_count",
            "option_history_count",
            "indicator_history_count",
            "underlying_history_count",
            "spot_history_count",
        ):
            value = indicators.get(key)
            if value is None:
                continue
            try:
                return max(0, int(float(value)))
            except (TypeError, ValueError):
                continue
        return None

    _SETUP_ANCHOR_SOURCES = (
        "latest_bar_ts",
        "bar_timestamp",
        "setup_candle_timestamp",
        "signal_timestamp",
    )
    _SETUP_VOTE_ID_SOURCES = (
        "setup_id",
        "setup_structure_id",
        "structure_id",
        *_SETUP_ANCHOR_SOURCES,
    )

    @classmethod
    def _stamp_setup_anchor(
        cls, elite_signal: EliteSignal, indicators: Mapping[str, Any]
    ) -> None:
        """Stamp the evaluation bar identity onto the signal metadata."""
        metadata = elite_signal.metadata
        anchor = None
        for key in cls._SETUP_ANCHOR_SOURCES:
            candidate = metadata.get(key)
            if candidate in (None, ""):
                candidate = indicators.get(key)
            if candidate not in (None, ""):
                anchor = candidate
                break
        if anchor in (None, ""):
            LOGGER.warning(
                "SIGNAL_SETUP_ANCHOR_MISSING strategy=%s symbol=%s",
                getattr(elite_signal, "strategy_name", "") or "",
                elite_signal.symbol,
                extra={
                    "event": "SIGNAL_SETUP_ANCHOR_MISSING",
                    "strategy": getattr(elite_signal, "strategy_name", "") or "",
                    "symbol": elite_signal.symbol,
                },
            )
            return
        metadata.setdefault("latest_bar_ts", anchor)
        metadata.setdefault("setup_candle_timestamp", anchor)
        metadata.setdefault("bar_timestamp", anchor)

    @staticmethod
    def _stamp_structural_setup_id(
        elite_signal: EliteSignal, indicators: Mapping[str, Any]
    ) -> None:
        """Use market structure, not process memory, as the reusable setup key."""
        metadata = elite_signal.metadata
        if metadata.get("setup_id") not in (None, ""):
            return
        if str(metadata.get("role") or "trigger").lower() == "context":
            return
        strategy = str(
            elite_signal.strategy_name or metadata.get("strategy") or ""
        ).lower()
        side = str(
            metadata.get("contract_side")
            or metadata.get("trade_side")
            or metadata.get("side")
            or ""
        ).upper()
        if "orb" in strategy:
            session = str(
                indicators.get("session_date")
                or str(metadata.get("latest_bar_ts") or "")[:10]
                or datetime.now(timezone.utc).date().isoformat()
            )
            high = metadata.get("opening_range_high")
            low = metadata.get("opening_range_low")
            if high is not None and low is not None:
                metadata["setup_id"] = f"orb:{session}:{side}:{high}:{low}"
        elif "smc" in strategy:
            reference = (
                indicators.get("prior_swing_low")
                if side == "CE"
                else indicators.get("prior_swing_high")
                if side == "PE"
                else None
            )
            if reference is None:
                reference = metadata.get("sweep_level")
            if reference is not None:
                metadata["setup_id"] = f"smc:{side}:{reference}"

    @staticmethod
    def _stamp_quality_evidence(
        elite_signal: EliteSignal, indicators: Mapping[str, Any]
    ) -> dict[str, object]:
        """Attach the canonical quality contract to every elite trigger vote."""
        metadata = elite_signal.metadata
        side = str(
            metadata.get("contract_side")
            or metadata.get("trade_side")
            or metadata.get("side")
            or ""
        ).upper()
        evidence = build_trade_quality_evidence(indicators, side=side)
        for key, value in evidence.items():
            metadata.setdefault(key, value)
        for key in (
            "spread_pct",
            "quote_depth_valid",
            "tradable_quote",
            "stale_data_used",
        ):
            if metadata.get(key) is None and indicators.get(key) is not None:
                metadata[key] = indicators.get(key)
        return evidence

    def _setup_vote_key(
        self, elite_signal: EliteSignal
    ) -> tuple[str, tuple[str, str, str]] | None:
        metadata = elite_signal.metadata
        if str(metadata.get("role") or "trigger").strip().lower() == "context":
            return None
        anchor = next(
            (
                metadata.get(key)
                for key in self._SETUP_VOTE_ID_SOURCES
                if metadata.get(key) not in (None, "")
            ),
            None,
        )
        if anchor is None:
            return None
        symbol_key = str(elite_signal.symbol or "").strip().upper()
        side = str(
            metadata.get("contract_side")
            or metadata.get("trade_side")
            or metadata.get("side")
            or ""
        ).strip().upper()
        return symbol_key, (str(anchor), str(elite_signal.signal), side)

    def _is_duplicate_setup_vote(self, elite_signal: EliteSignal) -> bool:
        """Compatibility shim; runner deterministic identity owns deduplication."""
        del elite_signal
        return False

    def notify_entry_accepted(self, side: str) -> None:
        """Compatibility hook; order lifecycle is owned by the runner."""
        del side

    def _no_vote(self, reason: str) -> None:
        """Record a single no-vote reason. Args: reason. Returns: None. Raises: none."""
        self.last_no_vote_reason = reason

    def _evaluate_signal(
        self,
        symbol: str = "",
        indicators: Dict[str, Any] | None = None,
        current_price: float = 0.0,
        position: Any | None = None,
    ) -> EliteSignal | None:
        raise NotImplementedError("Strategy must implement _evaluate_signal")

    def _process_signal(self, elite_signal: EliteSignal) -> Signal:
        """
        Converts internal EliteSignal to core Signal format.
        Standard implementation (Runner now handles metadata extraction).
        """
        self._last_signal_at = elite_signal.timestamp
        self._last_signal = elite_signal
        self._signals_generated += 1

        # 1. Consolidate all extra data into metadata
        canonical_reason = elite_signal.strategy_name or self.name
        metadata = elite_signal.metadata.copy()
        metadata.update(
            {
                "strategy": self.name,
                "mode": "Legacy" if self._is_legacy_signature else "Push",
                "quantity": elite_signal.quantity,
                "price": elite_signal.entry_price,  # Moved from argument
                "stop_loss": elite_signal.stop_loss,  # Moved from argument
                "take_profit": elite_signal.target,  # Moved from argument
                "tag": f"{self.name}",  # Moved from argument
            }
        )

        # 2. Create Signal using ONLY valid arguments
        # Valid args are: action, symbol, confidence, reason, metadata
        return Signal(
            action=elite_signal.signal,
            symbol=elite_signal.symbol,
            confidence=elite_signal.confidence,
            reason=canonical_reason,
            quantity=elite_signal.quantity,  # <--- Added mandatory arg
            stop_loss=elite_signal.stop_loss,  # <--- Added mandatory arg
            take_profit=elite_signal.target,
            metadata=metadata,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return diagnostic statistics."""
        last_payload = self._last_signal.to_payload() if self._last_signal else None
        return {
            "strategy": self.name,
            "enabled": self._config.enabled,
            "signals_generated": self._signals_generated,
            "last_signal": last_payload,
            "mode": "Legacy" if self._is_legacy_signature else "Push",
            **self.evaluation_health,
        }

    @property
    def config(self) -> EliteStrategyConfig:
        return self._config


__all__ = ["EliteSignal", "EliteStrategy"]