"""Focused runtime hardening for live-path reliability defects seen on 2026-08-07.

The adapters in this module intentionally do not change trading thresholds,
order placement, risk controls, market-hours checks, or strategy permissions.
They only:
- prevent stale non-entry/far-context work from tripping the *age* overload gate;
- keep the global pending-count overload fail-safe intact;
- correct single-vote rejection observability when the compared score is the
  regime-weighted score rather than the raw setup score; and
- make CPU telemetry reflect the authoritative dynamic option universe.
"""

from __future__ import annotations

from dataclasses import replace
import logging
import time
from typing import Any, Mapping

from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.symbols import normalize_symbol

_LOG = get_logger(__name__)
_PATCH_ATTR = "_runtime_reliability_hardening_installed"


def _critical_oldest_pending_age_ms_locked(mdm: Any) -> float:
    """Return oldest pending age for entry-critical queues only.

    MarketDataManager already routes priorities 0..2 (open position, selected
    option, spot/futures context and near-ATM context) into
    ``_pending_tick_queues`` while priority-3/far context is latest-only in
    ``_pending_far_ticks``.  A far-context tick must not globally disarm fresh,
    executable selected options merely because that optional work is old.
    """

    oldest_mono: float | None = None
    queues = getattr(mdm, "_pending_tick_queues", {}) or {}
    for symbol, queue in list(queues.items()):
        if not queue:
            continue
        try:
            priority, _bucket = mdm._tick_priority(symbol)
        except Exception:  # noqa: BLE001 - uncertainty stays fail-closed
            priority = 0
        if int(priority) > 2:
            continue
        tick = queue[0]
        timestamp = tick.get("_mdm_enqueued_mono") if isinstance(tick, Mapping) else None
        if not isinstance(timestamp, (int, float)):
            # A critical pending tick without age provenance is uncertainty.  By
            # treating it as old enough to trip the age gate we preserve the
            # existing fail-closed contract.
            return float(getattr(mdm, "_overload_enter_oldest_ms", 2000.0) or 2000.0)
        candidate = float(timestamp)
        oldest_mono = candidate if oldest_mono is None else min(oldest_mono, candidate)
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

    MarketDataManager._update_pipeline_overload_locked = _update_pipeline_overload_locked  # type: ignore[method-assign]
    setattr(MarketDataManager, _PATCH_ATTR, True)
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
        result = original(
            self,
            symbol=symbol,
            signals=signals,
            indicators=indicators,
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
            score_min, _confidence_min = self._single_vote_thresholds(vote.strategy)
            score_min = float(score_min)
        except Exception:  # noqa: BLE001 - never rewrite a reason on uncertainty
            return result

        # Relabel only the exact mathematically-proven case from the live logs:
        # raw setup clears the floor, but regime weighting puts the score below
        # the floor that the code actually compares. Trading outcome is unchanged.
        if raw_score < score_min or weighted_score >= score_min:
            return result

        corrected_reason = "regime_weighted_score_below_min"
        try:
            corrected = replace(
                decision,
                reason=corrected_reason,
                final_block_reason=corrected_reason,
            )
        except Exception:  # noqa: BLE001 - frozen dataclass contract may evolve
            return result
        decision_map[symbol_norm] = corrected
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
    normalized = normalize_symbol(str(symbol or ""))
    upper = normalized.upper()
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
            active_option_count = len(whitelist)
            active_option_count_source = "eval_option_whitelist"
        else:
            active_symbols = getattr(self, "_active_symbols", None) or set()
            active_option_count = sum(
                1 for symbol in active_symbols if _is_dynamic_option_symbol(symbol)
            )
            active_option_count_source = "dynamic_active_symbols"

        self._logger.info(
            "CPU_OPTIMIZATION_SUMMARY evaluated_symbols_count=%s skipped_by_midday_pause=%s "
            "skipped_by_eval_throttle=%s skipped_by_option_cap=%s active_option_symbols_count=%s",
            metrics.get("evaluated_symbols", 0),
            metrics.get("skipped_by_midday_pause", 0),
            metrics.get("skipped_by_eval_throttle", 0),
            metrics.get("skipped_by_option_cap", 0),
            active_option_count,
            extra={
                "event": "CPU_OPTIMIZATION_SUMMARY",
                **{name: int(value) for name, value in metrics.items()},
                "active_option_symbols_count": active_option_count,
                "active_option_count_source": active_option_count_source,
            },
        )
        metrics.clear()

    StrategyRunner._bump_cpu_metric = _bump_cpu_metric  # type: ignore[method-assign]
    setattr(StrategyRunner, attr, True)
    return True


def apply_patches() -> dict[str, bool]:
    """Install the focused reliability adapters idempotently."""

    state = {
        "mdm_overload": _install_mdm_overload_patch(),
        "strategy_reason": _install_strategy_reason_patch(),
        "runner_cpu_telemetry": _install_runner_cpu_telemetry_patch(),
    }
    if not all(state.values()):
        raise RuntimeError(f"runtime_reliability_hardening_incomplete state={state}")
    _LOG.info(
        "RUNTIME_RELIABILITY_HARDENING_INSTALLED state=%s",
        state,
        extra={"event": "RUNTIME_RELIABILITY_HARDENING_INSTALLED", **state},
    )
    return state


__all__ = ["apply_patches", "_critical_oldest_pending_age_ms_locked"]
