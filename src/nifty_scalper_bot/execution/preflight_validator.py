"""Unified gating logic for execution readiness checks."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Mapping

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from nifty_scalper_bot.utils.logging import get_logger


@dataclass(slots=True)
class ValidationResult:
    """Structured outcome emitted by :class:`PreFlightValidator`."""

    allowed: bool
    reasons: list[dict[str, Any]]
    blocking_level: str


class PreFlightValidatorSettings(BaseSettings):
    """Configuration payload for :class:`PreFlightValidator`."""

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", populate_by_name=True
    )

    quote_max_age_ms: int = Field(default=2000, ge=0)
    spread_max_pct: float = Field(default=1.5, ge=0.0)
    risk_max_drawdown_pct_day: float = Field(
        default=7.0,
        ge=0.0,
        validation_alias=AliasChoices(
            "RISK__MAX_DRAWDOWN_PCT_DAY", "risk_max_drawdown_pct_day"
        ),
    )
    risk_max_consecutive_losses: int = Field(
        default=3,
        ge=0,
        validation_alias=AliasChoices(
            "RISK__MAX_CONSECUTIVE_LOSSES", "risk_max_consecutive_losses"
        ),
    )
    atr_floor_percentile: float = Field(default=25.0)
    atr_ceiling_percentile: float = Field(default=75.0)
    regime_block_volatile: float = Field(default=0.85, ge=0.0, le=1.0)
    order_rate_limit_per_sec: int = Field(default=5, ge=1)


class PreFlightValidator:
    """Run execution gate checks before submitting broker orders."""

    def __init__(
        self,
        *,
        risk_manager: Any,
        regime_manager: Any,
        datahub: Any,
        session_guard: Any,
        settings: PreFlightValidatorSettings | None = None,
    ) -> None:
        """Store dependencies and initialise internal counters.

        Args:
            risk_manager: Provider for risk and PnL metrics.
            regime_manager: Surface current market regime labels.
            datahub: Market data accessor.
            session_guard: Trading session gating helper.
            settings: Optional override for validator configuration.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger = get_logger(__name__)
        self._logger.debug(
            "Entered PreFlightValidator.__init__",
            extra={"event": "preflight_init"},
        )
        try:
            self._risk_manager = risk_manager
            self._regime_manager = regime_manager
            self._datahub = datahub
            self._session_guard = session_guard
            self._settings = settings or PreFlightValidatorSettings()
            self._recent_validations: Deque[float] = deque()
            self._validation_cache: dict[str, tuple[float, ValidationResult]] = {}
            self._cache_ttl_sec = 1.0
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator.__init__: %s",
                exc,
                extra={"event": "preflight_init_error"},
                exc_info=exc,
            )
            raise

    def validate(
        self,
        symbol: str,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> ValidationResult:
        """Evaluate gate checks for *symbol* and return the combined verdict.

        Args:
            symbol: Trading symbol under evaluation.
            context: Optional auxiliary payload used by certain gates.

        Returns:
            ValidationResult: Structured validation outcome.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PreFlightValidator.validate",
            extra={"event": "preflight_validate", "symbol": symbol},
        )
        try:
            risk_manager = self._risk_manager
            breaker_reason = "Circuit breaker active"
            if risk_manager is not None:
                active_check = getattr(risk_manager, "is_circuit_breaker_active", None)
                breaker_active = (
                    bool(active_check()) if callable(active_check) else False
                )
                if breaker_active:
                    reason_lookup = getattr(
                        risk_manager, "is_circuit_breaker_tripped", None
                    )
                    if callable(reason_lookup):
                        tripped, detail = reason_lookup()
                        if tripped and detail:
                            breaker_reason = detail
                    self._logger.info(
                        "Condition met: preflight_circuit_breaker_block",
                        extra={
                            "event": "preflight_circuit_breaker",
                            "symbol": symbol,
                            "reason": breaker_reason,
                        },
                    )
                    return ValidationResult(
                        allowed=False,
                        reasons=[
                            {
                                "gate": "circuit_breaker",
                                "detail": breaker_reason,
                                "current_value": None,
                                "limit": None,
                            }
                        ],
                        blocking_level="HARD",
                    )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator.validate breaker check: %s",
                exc,
                extra={"event": "preflight_circuit_breaker_error", "symbol": symbol},
                exc_info=exc,
            )
        context_payload = dict(context or {})
        now = time.time()
        cache_key = self._cache_key(symbol, context_payload)
        cached = self._validation_cache.get(cache_key)
        if cached is not None and now - cached[0] <= self._cache_ttl_sec:
            self._logger.debug(
                "Condition met: preflight_cache_hit",
                extra={"event": "preflight_cache_hit", "symbol": symbol},
            )
            return self._clone_result(cached[1])
        try:
            result = self._do_validation(symbol, now, context_payload)
            self._validation_cache[cache_key] = (now, self._clone_result(result))
            self._prune_cache(now)
            return result
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator.validate: %s",
                exc,
                extra={"event": "preflight_validate_error", "symbol": symbol},
                exc_info=exc,
            )
            return ValidationResult(
                allowed=False,
                reasons=[
                    {
                        "gate": "preflight_internal",
                        "detail": "Validator failure",
                        "current_value": str(exc),
                        "limit": None,
                    }
                ],
                blocking_level="HARD",
            )

    def _do_validation(
        self,
        symbol: str,
        now: float,
        context_payload: Mapping[str, Any],
    ) -> ValidationResult:
        """Aggregate gate evaluations for *symbol* at the supplied timestamp.

        Args:
            symbol: Trading symbol under evaluation.
            now: Epoch timestamp associated with this validation.
            context_payload: Mutable gate context for downstream checks.

        Returns:
            ValidationResult: Structured validation outcome.

        Raises:
            Exception: Propagates gate execution failures for caller handling.
        """

        self._logger.debug(
            "Entered PreFlightValidator._do_validation",
            extra={"event": "preflight_do_validation", "symbol": symbol},
        )
        reasons: list[dict[str, Any]] = []
        blocking_level = "NONE"
        gates = [
            ("market_hours", self._check_market_hours, "HARD"),
            ("quote_staleness", self._check_quote_staleness, "HARD"),
            ("spread_check", self._check_spread, "HARD"),
            ("daily_loss_limit", self._check_daily_loss_limit, "HARD"),
            ("consecutive_losses", self._check_consecutive_losses, "HARD"),
            ("position_limit", self._check_position_limit, "SOFT"),
            ("atr_range", self._check_atr_range, "SOFT"),
            ("regime_block", self._check_regime, "SOFT"),
            ("margin_available", self._check_margin_available, "SOFT"),
            ("rate_limit", self._check_rate_limit, "HARD"),
        ]
        severity_order = {"NONE": 0, "SOFT": 1, "HARD": 2}
        try:
            for gate_name, gate_fn, severity in gates:
                reason = gate_fn(symbol, now, context_payload)
                if reason is None:
                    continue
                reason_payload = {"gate": gate_name, **reason}
                reasons.append(reason_payload)
                self._logger.info(
                    "preflight_gate_block",
                    extra={"gate": gate_name, "reason": reason_payload},
                )
                if severity_order[severity] > severity_order[blocking_level]:
                    blocking_level = severity
            allowed = not reasons
            return ValidationResult(
                allowed=allowed,
                reasons=reasons,
                blocking_level=blocking_level if reasons else "NONE",
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator._do_validation: %s",
                exc,
                extra={"event": "preflight_do_validation_error", "symbol": symbol},
                exc_info=exc,
            )
            raise

    def _cache_key(self, symbol: str, context: Mapping[str, Any]) -> str:
        """Return deterministic cache key for *symbol* and *context*.

        Args:
            symbol: Instrument identifier.
            context: Gate evaluation payload.

        Returns:
            str: Cache key derived from the payload content.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PreFlightValidator._cache_key",
            extra={"event": "preflight_cache_key", "symbol": symbol},
        )
        try:
            serialized = [symbol]
            for key in sorted(context):
                value = context[key]
                serialized.append(f"{key}={value}")
            return "|".join(serialized)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator._cache_key: %s",
                exc,
                extra={"event": "preflight_cache_key_error", "symbol": symbol},
                exc_info=exc,
            )
            return symbol

    @staticmethod
    def _clone_result(result: ValidationResult) -> ValidationResult:
        """Return a shallow copy of *result* for cache safety.

        Args:
            result: Validation outcome to duplicate.

        Returns:
            ValidationResult: Independent copy of the provided result.

        Raises:
            None.
        """

        return ValidationResult(
            allowed=result.allowed,
            reasons=list(result.reasons),
            blocking_level=result.blocking_level,
        )

    def _prune_cache(self, now: float) -> None:
        """Remove expired cache entries based on ``now`` timestamp.

        Args:
            now: Current epoch timestamp.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PreFlightValidator._prune_cache",
            extra={"event": "preflight_cache_prune"},
        )
        try:
            expiry = now - self._cache_ttl_sec
            stale_keys = [
                key
                for key, (timestamp, _) in self._validation_cache.items()
                if timestamp <= expiry
            ]
            for key in stale_keys:
                self._validation_cache.pop(key, None)
            if len(self._validation_cache) > 256:
                sorted_items = sorted(
                    self._validation_cache.items(), key=lambda item: item[1][0]
                )
                for key, _ in sorted_items[: len(self._validation_cache) - 256]:
                    self._validation_cache.pop(key, None)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator._prune_cache: %s",
                exc,
                extra={"event": "preflight_cache_prune_error"},
                exc_info=exc,
            )

    def _check_market_hours(
        self,
        symbol: str,
        now: float,
        context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Evaluate market hours via the attached session guard.

        Args:
            symbol: Symbol currently evaluated.
            now: Epoch timestamp used for diagnostics.
            context: Supplemental gate context.

        Returns:
            dict[str, Any] | None: Diagnostic payload on failure.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PreFlightValidator._check_market_hours",
            extra={"event": "preflight_gate_market_hours", "symbol": symbol},
        )
        try:
            guard = self._session_guard
            allowed = True
            detail = ""
            current_value: Any = None
            if hasattr(guard, "cantradenow"):
                try:
                    outcome = guard.cantradenow()
                except TypeError:
                    outcome = guard.cantradenow  # pragma: no cover - legacy attr
                if isinstance(outcome, (tuple, list)):
                    allowed = bool(outcome[0])
                    if len(outcome) > 1:
                        detail = str(outcome[1])
                    current_value = {
                        "allowed": allowed,
                        "reason": detail or "",
                    }
                else:
                    allowed = bool(outcome)
                    detail = "Session disallowed"
            elif hasattr(guard, "is_trading_allowed"):
                allowed = bool(guard.is_trading_allowed())
                detail = "Session disallowed"
            elif hasattr(guard, "evaluate"):
                status = guard.evaluate()
                market_open = getattr(status, "market_open", True)
                override = getattr(status, "override_out_of_hours", False)
                allowed = bool(market_open or override)
                reasons = getattr(status, "reasons", [])
                detail = ", ".join(reasons) if reasons else "Market closed"
                current_value = {
                    "market_open": market_open,
                    "override": override,
                }
            if allowed:
                return None
            return {
                "detail": detail,
                "current_value": current_value,
                "limit": "market_open",
            }
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator._check_market_hours: %s",
                exc,
                extra={"event": "preflight_gate_market_hours_error", "symbol": symbol},
                exc_info=exc,
            )
            return {
                "detail": "Session guard unavailable",
                "current_value": str(exc),
                "limit": "market_open",
            }

    def _check_quote_staleness(
        self,
        symbol: str,
        now: float,
        context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Ensure latest quote age does not breach configured stale limit.

        Args:
            symbol: Symbol currently evaluated.
            now: Epoch timestamp used for age computation.
            context: Supplemental gate context.

        Returns:
            dict[str, Any] | None: Diagnostic payload on failure.

        Raises:
            None.
        """
        self._logger.debug(
            "Entered PreFlightValidator._check_quote_staleness",
            extra={"event": "preflight_gate_quote_stale", "symbol": symbol},
        )
        try:
            fetch = getattr(self._datahub, "getlatesttick", None)
            if fetch is None:
                fetch = getattr(self._datahub, "get_latest_tick", None)
            tick = fetch(symbol) if callable(fetch) else None
            if not tick:
                self._logger.warning("Quote staleness: No tick for %s", symbol)
                return {
                    "detail": "No tick available",
                    "current_value": None,
                    "limit": self._settings.quote_max_age_ms,
                }
            timestamp = tick.get("timestamp")
            if timestamp is None:
                self._logger.warning(
                    "Quote staleness: Tick missing timestamp for %s, tick: %s",
                    symbol,
                    tick,
                )
                return {
                    "detail": "Tick missing timestamp",
                    "current_value": tick,
                    "limit": self._settings.quote_max_age_ms,
                }
            age_ms = abs(now - float(timestamp)) * 1000.0
            if age_ms <= self._settings.quote_max_age_ms:
                return None
            return {
                "detail": "Quote too stale",
                "current_value": age_ms,
                "limit": self._settings.quote_max_age_ms,
            }
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator._check_quote_staleness: %s",
                exc,
                extra={"event": "preflight_gate_quote_stale_error", "symbol": symbol},
                exc_info=exc,
            )
            return {
                "detail": "Quote staleness check failed",
                "current_value": str(exc),
                "limit": self._settings.quote_max_age_ms,
            }

    def _check_spread(
        self,
        symbol: str,
        now: float,
        context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Validate the bid-ask spread against configured thresholds.

        Args:
            symbol: Symbol currently evaluated.
            now: Epoch timestamp used for diagnostics.
            context: Supplemental gate context.

        Returns:
            dict[str, Any] | None: Diagnostic payload on failure.

        Raises:
            None.
        """
        self._logger.debug(
            "Entered PreFlightValidator._check_spread",
            extra={"event": "preflight_gate_spread", "symbol": symbol},
        )
        try:
            fetch = getattr(self._datahub, "getlatesttick", None)
            if fetch is None:
                fetch = getattr(self._datahub, "get_latest_tick", None)
            tick = (fetch(symbol) if callable(fetch) else None) or {}
            bid = tick.get("best_bid") or tick.get("bid")
            ask = tick.get("best_ask") or tick.get("ask")
            if bid is None or ask is None:
                self._logger.warning(
                    "Spread check: Missing bid/ask for %s, tick: %s", symbol, tick
                )
                return {
                    "detail": "Incomplete order book",
                    "current_value": tick,
                    "limit": self._settings.spread_max_pct,
                }
            try:
                bid_f = float(bid)
                ask_f = float(ask)
            except (TypeError, ValueError):
                self._logger.warning(
                    "Spread check: Spread inputs invalid for %s, tick: %s", symbol, tick
                )
                return {
                    "detail": "Spread inputs invalid",
                    "current_value": tick,
                    "limit": self._settings.spread_max_pct,
                }
            if bid_f <= 0 or ask_f <= 0:
                self._logger.warning(
                    "Spread check: Non-positive bid/ask for %s, tick: %s", symbol, tick
                )
                return {
                    "detail": "Spread inputs non-positive",
                    "current_value": tick,
                    "limit": self._settings.spread_max_pct,
                }
            if bid_f >= ask_f:
                self._logger.warning(
                    "Spread check: Inverted orderbook for %s, bid: %s, ask: %s",
                    symbol,
                    bid_f,
                    ask_f,
                )
                return {
                    "detail": "Inverted order book",
                    "current_value": {"bid": bid_f, "ask": ask_f},
                    "limit": self._settings.spread_max_pct,
                }
            mid = (bid_f + ask_f) / 2.0
            spread_pct = abs(ask_f - bid_f) / mid * 100.0 if mid else math.inf
            if spread_pct <= self._settings.spread_max_pct:
                return None
            return {
                "detail": "Spread too wide",
                "current_value": spread_pct,
                "limit": self._settings.spread_max_pct,
            }
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator._check_spread: %s",
                exc,
                extra={"event": "preflight_gate_spread_error", "symbol": symbol},
                exc_info=exc,
            )
            return {
                "detail": "Spread check failed",
                "current_value": str(exc),
                "limit": self._settings.spread_max_pct,
            }

    def _check_daily_loss_limit(
        self,
        symbol: str,
        now: float,
        context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Block when daily drawdown breaches configured percentage.

        Args:
            symbol: Symbol currently evaluated.
            now: Epoch timestamp used for diagnostics.
            context: Supplemental gate context containing balances.

        Returns:
            dict[str, Any] | None: Diagnostic payload on failure.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PreFlightValidator._check_daily_loss_limit",
            extra={"event": "preflight_gate_drawdown", "symbol": symbol},
        )
        try:
            snapshot = None
            if hasattr(self._risk_manager, "snapshot"):
                snapshot = self._risk_manager.snapshot()
            daily_realized = 0.0
            loss_limit_abs = 0.0
            if snapshot is not None:
                realized_attr = getattr(snapshot, "daily_realized", None)
                if realized_attr is None:
                    realized_attr = getattr(snapshot, "dailyrealized", 0.0)
                if realized_attr is None:
                    realized_attr = 0.0
                daily_realized = float(realized_attr)
                limit_attr = getattr(snapshot, "daily_loss_limit", None)
                if limit_attr is None:
                    limit_attr = getattr(snapshot, "dailylosslimit", 0.0)
                if limit_attr is None:
                    limit_attr = 0.0
                loss_limit_abs = float(limit_attr)
            else:
                realized_value = getattr(
                    self._risk_manager, "get_daily_pnl", lambda: 0.0
                )()
                daily_realized = float(realized_value or 0.0)
                limit_value = context.get("daily_loss_limit")
                if limit_value is None:
                    limit_value = getattr(self._risk_manager, "daily_loss_limit", 0.0)
                loss_limit_abs = float(limit_value or 0.0)
            if loss_limit_abs > 0 and daily_realized <= -loss_limit_abs:
                return {
                    "detail": "Daily loss limit breached",
                    "current_value": abs(daily_realized),
                    "limit": loss_limit_abs,
                }
            balance = context.get("account_balance")
            if balance is None:
                balance = getattr(self._risk_manager, "current_balance", None)
            balance_value = float(balance) if balance is not None else 0.0
            if balance_value <= 0:
                return None
            if daily_realized < 0:
                drawdown_pct = abs(daily_realized) / balance_value * 100.0
                if drawdown_pct > self._settings.risk_max_drawdown_pct_day:
                    return {
                        "detail": "Daily loss limit breached",
                        "current_value": drawdown_pct,
                        "limit": self._settings.risk_max_drawdown_pct_day,
                    }
            return None
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator._check_daily_loss_limit: %s",
                exc,
                extra={"event": "preflight_gate_drawdown_error", "symbol": symbol},
                exc_info=exc,
            )
            return {
                "detail": "Drawdown check failed",
                "current_value": str(exc),
                "limit": self._settings.risk_max_drawdown_pct_day,
            }

    def _check_consecutive_losses(
        self,
        symbol: str,
        now: float,
        context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Ensure loss streaks remain under configured ceiling.

        Args:
            symbol: Symbol currently evaluated.
            now: Epoch timestamp used for diagnostics.
            context: Supplemental gate context.

        Returns:
            dict[str, Any] | None: Diagnostic payload on failure.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PreFlightValidator._check_consecutive_losses",
            extra={"event": "preflight_gate_losses", "symbol": symbol},
        )
        try:
            snapshot = None
            if hasattr(self._risk_manager, "snapshot"):
                snapshot = self._risk_manager.snapshot()
            losses: int | None = None
            if snapshot is not None:
                losses = int(
                    getattr(
                        snapshot,
                        "losses_in_row",
                        getattr(snapshot, "lossesinrow", 0),
                    )
                )
            if losses is None and hasattr(self._risk_manager, "get_consecutive_losses"):
                losses = int(self._risk_manager.get_consecutive_losses())
            if losses is None:
                losses = int(context.get("consecutive_losses", 0))
            if losses <= self._settings.risk_max_consecutive_losses:
                return None
            return {
                "detail": "Too many consecutive losses",
                "current_value": losses,
                "limit": self._settings.risk_max_consecutive_losses,
            }
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator._check_consecutive_losses: %s",
                exc,
                extra={"event": "preflight_gate_losses_error", "symbol": symbol},
                exc_info=exc,
            )
            return {
                "detail": "Consecutive loss check failed",
                "current_value": str(exc),
                "limit": self._settings.risk_max_consecutive_losses,
            }

    def _check_position_limit(
        self,
        symbol: str,
        now: float,
        context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Validate aggregate open positions against configured limit.

        Args:
            symbol: Symbol currently evaluated.
            now: Epoch timestamp used for diagnostics.
            context: Supplemental gate context providing limits.

        Returns:
            dict[str, Any] | None: Diagnostic payload on failure.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PreFlightValidator._check_position_limit",
            extra={"event": "preflight_gate_position", "symbol": symbol},
        )
        try:
            limit = context.get("position_limit")
            if limit is None:
                settings = getattr(self._risk_manager, "settings", None)
                limit = getattr(settings, "max_concurrent_positions", None)
            if limit is None:
                return None
            manager = getattr(
                self._risk_manager,
                "position_manager",
                getattr(self._risk_manager, "positionmanager", None),
            )
            open_positions: list[Any] = []
            if manager is not None:
                if hasattr(manager, "get_all_positions"):
                    open_positions = list(manager.get_all_positions())
                elif hasattr(manager, "getallpositions"):
                    open_positions = list(manager.getallpositions())
                elif hasattr(manager, "get_open_positions"):
                    open_positions = list(manager.get_open_positions())
            open_count = len(open_positions)
            limit_value = int(limit)
            if open_count < limit_value:
                return None
            return {
                "detail": "Position limit reached",
                "current_value": open_count,
                "limit": limit_value,
            }
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator._check_position_limit: %s",
                exc,
                extra={"event": "preflight_gate_position_error", "symbol": symbol},
                exc_info=exc,
            )
            return {
                "detail": "Position limit check failed",
                "current_value": str(exc),
                "limit": context.get("position_limit"),
            }

    def _check_atr_range(
        self,
        symbol: str,
        now: float,
        context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Block when ATR percentile falls outside configured range.

        Args:
            symbol: Symbol currently evaluated.
            now: Epoch timestamp used for diagnostics.
            context: Supplemental gate context with ATR percentile.

        Returns:
            dict[str, Any] | None: Diagnostic payload on failure.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PreFlightValidator._check_atr_range",
            extra={"event": "preflight_gate_atr", "symbol": symbol},
        )
        try:
            percentile = context.get("atr_percentile")
            if percentile is None:
                return None
            percentile_value = float(percentile)
            if (
                self._settings.atr_floor_percentile
                <= percentile_value
                <= self._settings.atr_ceiling_percentile
            ):
                return None
            return {
                "detail": "ATR percentile out of band",
                "current_value": percentile_value,
                "limit": (
                    self._settings.atr_floor_percentile,
                    self._settings.atr_ceiling_percentile,
                ),
            }
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator._check_atr_range: %s",
                exc,
                extra={"event": "preflight_gate_atr_error", "symbol": symbol},
                exc_info=exc,
            )
            return {
                "detail": "ATR check failed",
                "current_value": str(exc),
                "limit": (
                    self._settings.atr_floor_percentile,
                    self._settings.atr_ceiling_percentile,
                ),
            }

    def _check_regime(
        self,
        symbol: str,
        now: float,
        context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Fail closed when the regime detector marks market as volatile.

        Args:
            symbol: Symbol currently evaluated.
            now: Epoch timestamp used for diagnostics.
            context: Supplemental gate context.

        Returns:
            dict[str, Any] | None: Diagnostic payload on failure.

        Raises:
            None.
        """
        self._logger.debug(
            "Entered PreFlightValidator._check_regime",
            extra={"event": "preflight_gate_regime", "symbol": symbol},
        )
        try:
            regime = None
            confidence = None

            # 1. Fetch Regime Data
            if hasattr(self._regime_manager, "get_current_regime"):
                regime = self._regime_manager.get_current_regime()

            if hasattr(self._regime_manager, "get_regime_confidence"):
                try:
                    confidence = float(self._regime_manager.get_regime_confidence())
                except (TypeError, ValueError):
                    confidence = 0.0
            
            # [FIX] CRITICAL: Fail-Closed if Regime is Missing
            # If the bot is "blind" (no regime data), we BLOCK trading.
            if regime is None:
                 return {
                    "detail": "Regime data missing",
                    "current_value": None,
                    "limit": "valid_snapshot"
                 }

            # 2. Volatility Logic
            # If the market is NOT Volatile (e.g., TRENDING, RANGING), allow trade.
            if str(regime).upper() != "VOLATILE":
                return None

            # If Market IS Volatile, check if confidence is high enough to block.
            threshold = self._settings.regime_block_volatile
            
            # Block if confidence is high (or unknown/None default safe)
            if confidence is None or confidence >= threshold:
                return {
                    "detail": "Volatile regime",
                    "current_value": confidence,
                    "limit": threshold,
                }
            
            return None

        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator._check_regime: %s",
                exc,
                extra={"event": "preflight_gate_regime_error", "symbol": symbol},
                exc_info=exc,
            )
            return {
                "detail": "Regime check failed",
                "current_value": str(exc),
                "limit": self._settings.regime_block_volatile,
            }

    def _check_margin_available(
        self,
        symbol: str,
        now: float,
        context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Ensure sufficient margin is available for the intended trade.

        Args:
            symbol: Symbol currently evaluated.
            now: Epoch timestamp used for diagnostics.
            context: Supplemental gate context containing margin data.

        Returns:
            dict[str, Any] | None: Diagnostic payload on failure.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PreFlightValidator._check_margin_available",
            extra={"event": "preflight_gate_margin", "symbol": symbol},
        )
        try:
            order_cost = context.get("order_cost")
            if order_cost is None:
                return None
            available = context.get("margin_available")
            if available is None and hasattr(
                self._risk_manager, "get_margin_available"
            ):
                available = self._risk_manager.get_margin_available()
            if available is None:
                return None
            available_f = float(available)
            order_cost_f = float(order_cost)
            if available_f >= order_cost_f:
                return None
            return {
                "detail": "Insufficient margin",
                "current_value": available_f,
                "limit": order_cost_f,
            }
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator._check_margin_available: %s",
                exc,
                extra={"event": "preflight_gate_margin_error", "symbol": symbol},
                exc_info=exc,
            )
            return {
                "detail": "Margin check failed",
                "current_value": str(exc),
                "limit": context.get("order_cost"),
            }

    def _check_rate_limit(
        self,
        symbol: str,
        now: float,
        context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Throttle validations when they exceed configured rate limit.

        Args:
            symbol: Symbol currently evaluated.
            now: Epoch timestamp used for diagnostics.
            context: Supplemental gate context.

        Returns:
            dict[str, Any] | None: Diagnostic payload on failure.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PreFlightValidator._check_rate_limit",
            extra={"event": "preflight_gate_rate_limit", "symbol": symbol},
        )
        try:
            window = 1.0
            limit = self._settings.order_rate_limit_per_sec
            while (
                self._recent_validations and now - self._recent_validations[0] > window
            ):
                self._recent_validations.popleft()
            if len(self._recent_validations) >= limit:
                return {
                    "detail": "Rate limit exceeded",
                    "current_value": len(self._recent_validations),
                    "limit": limit,
                }
            self._recent_validations.append(now)
            return None
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PreFlightValidator._check_rate_limit: %s",
                exc,
                extra={"event": "preflight_gate_rate_limit_error", "symbol": symbol},
                exc_info=exc,
            )
            return {
                "detail": "Rate limit check failed",
                "current_value": str(exc),
                "limit": self._settings.order_rate_limit_per_sec,
            }


__all__ = [
    "PreFlightValidator",
    "PreFlightValidatorSettings",
    "ValidationResult",
]
