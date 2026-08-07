"""Focused runtime hardening for reliability defects observed on 2026-08-07.

No trading threshold, order path, risk control, market-hours guard, or strategy
permission is changed here. The adapters correct overload age attribution,
bounded consensus-quality evidence, tick hot-path duplication, and runtime
diagnostics while preserving fail-closed behavior for entry-critical queues.
"""

from __future__ import annotations

from dataclasses import replace
import logging
import time
from typing import Any, Mapping

from nifty_scalper_bot.utils.logging import get_logger, log_throttled
from nifty_scalper_bot.utils.symbols import normalize_symbol

_LOG = get_logger(__name__)
_PATCH_ATTR = "_runtime_reliability_hardening_installed"


def _critical_oldest_pending_age_ms_locked(mdm: Any) -> float:
    """Return oldest entry-critical queue age.

    Non-selected active-basket ``near_atm`` options remain in normal FIFO
    queues so every tick still reaches CandleEngine/OHLC processing. Their age
    is optional strategy context, however, and must not by itself disarm fresh
    selected CE/PE execution. Selected options, spot/future context, open
    positions, and any unclassified normal queue remain age-critical. Global
    pending-count overload remains unchanged and still covers every lane.
    """

    oldest_mono: float | None = None
    for queue in (getattr(mdm, "_pending_tick_queues", {}) or {}).values():
        if not queue:
            continue
        tick = queue[0]
        if not isinstance(tick, Mapping):
            return float(
                getattr(mdm, "_overload_enter_oldest_ms", 2000.0) or 2000.0
            )
        bucket = str(tick.get("_mdm_priority_bucket") or "").strip().lower()
        if bucket == "near_atm":
            continue
        timestamp = tick.get("_mdm_enqueued_mono")
        if not isinstance(timestamp, (int, float)):
            # Unknown age/role on normal queued work is uncertainty: preserve
            # fail-closed behavior rather than silently treating it as context.
            return float(
                getattr(mdm, "_overload_enter_oldest_ms", 2000.0) or 2000.0
            )
        candidate = float(timestamp)
        oldest_mono = (
            candidate if oldest_mono is None else min(oldest_mono, candidate)
        )
    if oldest_mono is None:
        return 0.0
    return max(0.0, (time.monotonic() - oldest_mono) * 1000.0)


def _install_mdm_overload_patch() -> bool:
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager

    if bool(getattr(MarketDataManager, _PATCH_ATTR, False)):
        return True

    def _update_pipeline_overload_locked(self: Any) -> None:
        pending = self._pending_count_locked()
        total_oldest_ms = self._oldest_pending_age_ms_locked()
        critical_oldest_ms = _critical_oldest_pending_age_ms_locked(self)
        if not self._pipeline_overloaded:
            if (
                pending >= self._overload_enter_pending
                or critical_oldest_ms >= self._overload_enter_oldest_ms
            ):
                self._pipeline_overloaded = True
                self._overload_since_mono = time.monotonic()
                self._logger.warning(
                    "DATA_PIPELINE_OVERLOAD_ENTER pending_ticks=%d oldest_pending_age_ms=%.0f "
                    "critical_oldest_pending_age_ms=%.0f enter_pending=%d enter_oldest_ms=%.0f",
                    pending,
                    total_oldest_ms,
                    critical_oldest_ms,
                    self._overload_enter_pending,
                    self._overload_enter_oldest_ms,
                    extra={
                        "event": "DATA_PIPELINE_OVERLOAD_ENTER",
                        "pending_ticks": pending,
                        "oldest_pending_age_ms": total_oldest_ms,
                        "critical_oldest_pending_age_ms": critical_oldest_ms,
                        "enter_pending": self._overload_enter_pending,
                        "enter_oldest_ms": self._overload_enter_oldest_ms,
                    },
                )
        elif (
            pending <= self._overload_exit_pending
            and critical_oldest_ms <= self._overload_exit_oldest_ms
        ):
            duration = 0.0
            if self._overload_since_mono is not None:
                duration = max(0.0, time.monotonic() - self._overload_since_mono)
            self._pipeline_overloaded = False
            self._overload_since_mono = None
            self._logger.warning(
                "DATA_PIPELINE_OVERLOAD_RECOVERED pending_ticks=%d oldest_pending_age_ms=%.0f "
                "critical_oldest_pending_age_ms=%.0f overloaded_for_s=%.1f",
                pending,
                total_oldest_ms,
                critical_oldest_ms,
                duration,
                extra={
                    "event": "DATA_PIPELINE_OVERLOAD_RECOVERED",
                    "pending_ticks": pending,
                    "oldest_pending_age_ms": total_oldest_ms,
                    "critical_oldest_pending_age_ms": critical_oldest_ms,
                    "overloaded_for_s": duration,
                },
            )

    MarketDataManager._update_pipeline_overload_locked = (  # type: ignore[method-assign]
        _update_pipeline_overload_locked
    )
    setattr(MarketDataManager, _PATCH_ATTR, True)
    return True


def _is_canonical_runtime_tick(payload: Mapping[str, Any]) -> bool:
    """Return whether MDM has already produced the full DataHub runtime contract."""

    symbol = str(payload.get("symbol") or "").strip()
    if not symbol or normalize_symbol(symbol) != symbol:
        return False
    token = payload.get("instrument_token") or payload.get("token")
    price = payload.get("ltp") or payload.get("last_price")
    timestamp_ms = payload.get("timestamp_ms")
    try:
        if int(token) <= 0 or float(price) <= 0 or float(timestamp_ms) <= 0:
            return False
    except (TypeError, ValueError):
        return False
    source = str(payload.get("source") or "").strip().lower()
    if source not in {"ws", "websocket", "stream", "poll", "rest"}:
        return False
    # These are the quote/readiness fields MDM's normalized live tick owns.
    # Requiring them prevents a merely timestamped raw caller from entering
    # this fast path and preserves the generic canonicalizer for partial input.
    return all(
        key in payload
        for key in (
            "timestamp",
            "received_at",
            "depth_available",
            "tradable_quote",
            "hard_readiness_eligible",
        )
    )


def _install_datahub_tick_hotpath_patch() -> bool:
    """Avoid rebuilding MDM's already-canonical tick before Runner dispatch."""

    from nifty_scalper_bot.data.data_hub import DataHub

    attr = "_mdm_tick_hotpath_hardening_installed"
    if bool(getattr(DataHub, attr, False)):
        return True
    original = DataHub._canonicalize_tick_payload

    def _canonicalize_tick_payload(
        self: Any, payload: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if _is_canonical_runtime_tick(payload):
            tick = dict(payload)
            quality = str(tick.get("timestamp_quality") or "").strip().lower()
            if not quality:
                if tick.get("exchange_timestamp") not in (None, ""):
                    quality = "exchange"
                elif tick.get("timestamp") not in (None, ""):
                    quality = "broker"
                else:
                    quality = "received_at"
                tick["timestamp_quality"] = quality
            tick.setdefault("quote_source", tick.get("source"))
            return tick
        return original(self, payload)

    DataHub._canonicalize_tick_payload = _canonicalize_tick_payload  # type: ignore[method-assign]
    setattr(DataHub, attr, True)
    return True


def _trigger_confirmation_details(
    signals: list[tuple[Any, Any]],
) -> tuple[bool, list[str]]:
    """Return bounded independent same-side trigger confirmation evidence."""

    trigger_votes = [
        vote
        for signal, vote in signals
        if str((getattr(vote, "metadata", {}) or {}).get("role") or "trigger").lower()
        != "context"
        and str(getattr(signal, "action", "")) not in {"CLOSE_LONG", "CLOSE_SHORT"}
    ]
    if len(trigger_votes) < 2:
        return False, []
    try:
        best = max(trigger_votes, key=lambda vote: float(getattr(vote, "score", 0.0) or 0.0))
    except Exception:  # noqa: BLE001 - uncertainty never creates confirmation
        return False, []
    best_side = str(getattr(best, "side", "") or "").upper()
    best_strategy = str(getattr(best, "strategy", "") or "").strip().lower()
    if best_side not in {"CE", "PE"} or not best_strategy:
        return False, []
    confirming = sorted(
        {
            str(getattr(vote, "strategy", "") or "").strip()
            for vote in trigger_votes
            if str(getattr(vote, "side", "") or "").upper() == best_side
            and str(getattr(vote, "strategy", "") or "").strip().lower()
            not in {"", best_strategy}
        }
    )
    return bool(confirming), confirming


def _install_trade_quality_patch() -> bool:
    """Credit one bounded independent trigger before the unchanged quality gate."""

    from nifty_scalper_bot.core.strategy_manager import StrategyManager

    attr = "_independent_trigger_quality_hardening_installed"
    if bool(getattr(StrategyManager, attr, False)):
        return True
    original = StrategyManager._compute_trade_quality_score

    def _compute_trade_quality_score(
        self: Any,
        vote: Any,
        indicators: Mapping[str, Any],
        *,
        symbol: str,
        selected_ok: bool,
        near_atm_ok: bool,
        context_votes: list[Any],
    ) -> tuple[float, dict[str, Any]]:
        score, metadata = original(
            self,
            vote,
            indicators,
            symbol=symbol,
            selected_ok=selected_ok,
            near_atm_ok=near_atm_ok,
            context_votes=context_votes,
        )
        details = dict(metadata or {})
        components = dict(details.get("trade_quality_components") or {})
        confirmation = bool(indicators.get("independent_trigger_confirmation"))
        already_blocked = bool(details.get("already_blocked_by_strategy"))
        # Exactly one capped 0.5-point contribution. Multiple trigger votes do
        # not stack, and an already-invalid best trigger receives no rescue.
        bonus = 0.5 if confirmation and not already_blocked else 0.0
        components["independent_trigger_confirmation"] = bonus
        adjusted = max(0.0, min(10.0, float(score) + bonus))
        details["trade_quality_components"] = components
        details["trade_quality_score"] = round(adjusted, 3)
        details["independent_trigger_confirmation"] = bool(bonus)
        details["independent_trigger_confirmation_strategies"] = list(
            indicators.get("independent_trigger_confirmation_strategies") or []
        )
        return adjusted, details

    StrategyManager._compute_trade_quality_score = (  # type: ignore[method-assign]
        _compute_trade_quality_score
    )
    setattr(StrategyManager, attr, True)
    return True


def _install_strategy_reason_patch() -> bool:
    from nifty_scalper_bot.core.strategy_manager import StrategyManager

    attr = "_weighted_rejection_reason_hardening_installed"
    if bool(getattr(StrategyManager, attr, False)):
        return True
    original = StrategyManager._combine_strategy_votes

    def _combine_strategy_votes(
        self: Any,
        *,
        symbol: str,
        signals: list[tuple[Any, Any]],
        indicators: Mapping[str, Any],
        no_vote_reason_counts: Mapping[str, int] | None = None,
    ) -> Any:
        confirmation, confirmation_strategies = _trigger_confirmation_details(signals)
        effective_indicators = dict(indicators or {})
        if confirmation:
            effective_indicators["independent_trigger_confirmation"] = True
            effective_indicators["independent_trigger_confirmation_strategies"] = (
                confirmation_strategies
            )
        result = original(
            self,
            symbol=symbol,
            signals=signals,
            indicators=effective_indicators,
            no_vote_reason_counts=no_vote_reason_counts,
        )
        if result is not None:
            return result

        symbol_norm = str(symbol or "").strip().upper()
        decision_map = getattr(self, "_last_no_signal_decision_by_symbol", None)
        decision = decision_map.get(symbol_norm) if isinstance(decision_map, dict) else None
        if decision is None or str(getattr(decision, "reason", "")) != "raw_score_below_min":
            return result

        trigger_votes = [
            vote
            for signal, vote in signals
            if str((getattr(vote, "metadata", {}) or {}).get("role") or "trigger").lower()
            != "context"
            and str(getattr(signal, "action", "")) not in {"CLOSE_LONG", "CLOSE_SHORT"}
        ]
        if len(trigger_votes) != 1:
            return result

        vote = trigger_votes[0]
        try:
            raw_score = float(self._extract_raw_score(vote))
            weighted_score = float(getattr(vote, "score", 0.0) or 0.0)
            score_min = float(self._single_vote_thresholds(vote.strategy)[0])
        except Exception:  # noqa: BLE001 - never rewrite a reason on uncertainty
            return result

        # Rewrite only the mathematically-proven logging defect. The rejected
        # result and every trading threshold remain unchanged.
        if raw_score < score_min or weighted_score >= score_min:
            return result
        corrected_reason = "regime_weighted_score_below_min"
        try:
            decision_map[symbol_norm] = replace(
                decision,
                reason=corrected_reason,
                final_block_reason=corrected_reason,
            )
        except Exception:  # noqa: BLE001 - diagnostic correction must be non-fatal
            return result
        _LOG.info(
            "STRATEGY_REJECTION_REASON_CORRECTED symbol=%s strategy=%s raw_score=%.2f "
            "weighted_score=%.2f score_min=%.2f reason=%s",
            symbol_norm,
            getattr(vote, "strategy", None),
            raw_score,
            weighted_score,
            score_min,
            corrected_reason,
            extra={
                "event": "STRATEGY_REJECTION_REASON_CORRECTED",
                "symbol": symbol_norm,
                "strategy": getattr(vote, "strategy", None),
                "raw_score": raw_score,
                "weighted_score": weighted_score,
                "score_min": score_min,
                "reason": corrected_reason,
            },
        )
        return result

    StrategyManager._combine_strategy_votes = _combine_strategy_votes  # type: ignore[method-assign]
    setattr(StrategyManager, attr, True)
    return True


def _is_dynamic_option_symbol(symbol: object) -> bool:
    upper = normalize_symbol(str(symbol or "")).upper()
    return upper.startswith("NFO:NIFTY") and upper.endswith(("CE", "PE"))


def _install_runner_cpu_telemetry_patch() -> bool:
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    attr = "_dynamic_cpu_telemetry_hardening_installed"
    if bool(getattr(StrategyRunner, attr, False)):
        return True

    def _bump_cpu_metric(self: Any, key: str) -> None:
        metrics = getattr(self, "_cpu_opt_metrics", None)
        if metrics is None:
            metrics = {}
            self._cpu_opt_metrics = metrics
        metrics[key] = int(metrics.get(key, 0)) + 1
        if not self._should_log_throttled("cpu_optimization_summary", 60.0):
            return

        whitelist = getattr(self, "_eval_option_whitelist", None) or set()
        if whitelist:
            option_count = len(whitelist)
            count_source = "eval_option_whitelist"
        else:
            active = getattr(self, "_active_symbols", None) or set()
            option_count = sum(
                1 for symbol in active if _is_dynamic_option_symbol(symbol)
            )
            count_source = "dynamic_active_symbols"

        self._logger.info(
            "CPU_OPTIMIZATION_SUMMARY evaluated_symbols_count=%s skipped_by_midday_pause=%s "
            "skipped_by_eval_throttle=%s skipped_by_option_cap=%s active_option_symbols_count=%s",
            metrics.get("evaluated_symbols", 0),
            metrics.get("skipped_by_midday_pause", 0),
            metrics.get("skipped_by_eval_throttle", 0),
            metrics.get("skipped_by_option_cap", 0),
            option_count,
            extra={
                "event": "CPU_OPTIMIZATION_SUMMARY",
                **{name: int(value) for name, value in metrics.items()},
                "active_option_symbols_count": option_count,
                "active_option_count_source": count_source,
            },
        )
        metrics.clear()

    StrategyRunner._bump_cpu_metric = _bump_cpu_metric  # type: ignore[method-assign]
    setattr(StrategyRunner, attr, True)
    return True


def _install_runner_tick_latency_telemetry_patch() -> bool:
    """Expose whether remaining DataHub callback latency is inside Runner."""

    from nifty_scalper_bot.strategies.runner import StrategyRunner

    attr = "_datahub_tick_latency_telemetry_installed"
    if bool(getattr(StrategyRunner, attr, False)):
        return True
    original = StrategyRunner.on_datahub_tick

    def on_datahub_tick(self: Any, tick: dict[str, Any]) -> None:
        started = time.perf_counter()
        try:
            return original(self, tick)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            if duration_ms < 50.0:
                return
            symbol = normalize_symbol(str(tick.get("symbol") or ""))
            try:
                route = self._entry_evaluation_route(symbol)
                route_value = str(getattr(route, "value", route))
            except Exception:  # noqa: BLE001 - telemetry only
                route_value = "unknown"
            log_throttled(
                self._logger,
                f"runner_datahub_tick_slow:{symbol}:{route_value}",
                "RUNNER_DATAHUB_TICK_SLOW symbol=%s route=%s duration_ms=%.1f",
                symbol,
                route_value,
                duration_ms,
                interval_sec=30.0,
                level=logging.WARNING,
                extra={
                    "event": "RUNNER_DATAHUB_TICK_SLOW",
                    "symbol": symbol,
                    "route": route_value,
                    "duration_ms": duration_ms,
                },
            )

    StrategyRunner.on_datahub_tick = on_datahub_tick  # type: ignore[method-assign]
    setattr(StrategyRunner, attr, True)
    return True


def apply_patches() -> dict[str, bool]:
    """Install focused reliability adapters idempotently."""

    state = {
        "mdm_overload": _install_mdm_overload_patch(),
        "datahub_tick_hotpath": _install_datahub_tick_hotpath_patch(),
        "trade_quality": _install_trade_quality_patch(),
        "strategy_reason": _install_strategy_reason_patch(),
        "runner_cpu_telemetry": _install_runner_cpu_telemetry_patch(),
        "runner_tick_latency_telemetry": _install_runner_tick_latency_telemetry_patch(),
    }
    if not all(state.values()):
        raise RuntimeError(f"runtime_reliability_hardening_incomplete state={state}")
    _LOG.info(
        "RUNTIME_RELIABILITY_HARDENING_INSTALLED state=%s",
        state,
        extra={"event": "RUNTIME_RELIABILITY_HARDENING_INSTALLED", **state},
    )
    return state


__all__ = [
    "apply_patches",
    "_critical_oldest_pending_age_ms_locked",
    "_is_canonical_runtime_tick",
    "_trigger_confirmation_details",
]
