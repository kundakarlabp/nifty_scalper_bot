"""Purpose: Guarded LIVE pilot and research NIFTY futures session-VWAP reclaim.

Responsibilities: Verify or strictly recover authenticated completed futures
minutes, pre-confirmation volatility pressure and live-quote target proximity;
has_ist_session_anchor verifies UTC/IST session-open
identity for bounded local recovery. Frozen ATR/sigma and session-anchored
bands are compared without look-ahead, before later quote admission.
assess_option_cost_risk accepts spot-referenced delta and observed spot/futures
basis and emits conservative option IV/theta/time-stop and early-exit scenarios.
RangeVWAPReclaimStrategy records first-seen futures excursions, shadow local
range evidence within TREND independently of session-VWAP admission, and applies a
bounded failed-breakout exception to center disagreement with stronger
same-quote option flow and first-barrier after-cost payoff requirements.
It owns regime confirmation and live setup de-duplication; the manager,
runner and broker still own execution and protective orders.
Entry lifecycle: notify_entry_accepted and notify_entry_rejected use existing
StrategyManager callbacks; exact setup/consumption state is restored and
write-verified through PositionManager's same-day risk circuit before LIVE
emission; only a proven pre-broker rejection releases a reservation.
assess_option_cost_risk models the bid-price stop trigger separately from
an adverse stop liquidation fill.
Safety: LIVE requires an explicit pilot opt-in, attested reversal and quote
provenance; broker-stop, order affordability and capital gates remain authoritative.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from threading import RLock
from typing import Any
from zoneinfo import ZoneInfo

from nifty_scalper_bot.config.strategy_taxonomy import range_vwap_live_enabled
from nifty_scalper_bot.instruments.active_contracts import (
    canonical_nifty_future_symbol,
    canonical_nifty_option_symbol,
)
from nifty_scalper_bot.risk.cost_model import estimate_round_trip_cost
from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    RangeVWAPReclaimStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)
_IST = ZoneInfo("Asia/Kolkata")


def _value(raw: Any) -> float | None:
    try:
        result = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _time(raw: Any) -> datetime | None:
    try:
        if isinstance(raw, datetime):
            value = raw
        elif isinstance(raw, (float, int)):
            epoch = float(raw)
            value = datetime.fromtimestamp(
                epoch / 1000 if epoch > 1e11 else epoch, tz=_IST
            )
        else:
            value = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return value.astimezone(_IST) if value.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def has_ist_session_anchor(observed: Any, *, decision_at: Any) -> bool:
    """Recognize the current 09:15 anchor independent of UTC/IST encoding."""
    if not isinstance(observed, Sequence) or not observed:
        return False
    first = observed[0]
    if not isinstance(first, Mapping):
        return False
    stamp = _time(first.get("timestamp") or first.get("datetime"))
    decision = _time(decision_at)
    return bool(
        stamp is not None
        and decision is not None
        and stamp.date() == decision.date()
        and (stamp.hour, stamp.minute, stamp.second, stamp.microsecond) == (9, 15, 0, 0)
    )


def _weighted_center(rows: Sequence[Mapping[str, Any]]) -> float | None:
    total = sum(float(item["volume"]) for item in rows)
    if total <= 0:
        return None
    return (
        sum(
            ((float(item["high"]) + float(item["low"]) + float(item["close"])) / 3)
            * float(item["volume"])
            for item in rows
        )
        / total
    )


def _verified_completed_bars(
    observed: Any, *, decision_at: datetime, min_history: int
) -> tuple[list[dict[str, Any]], str]:
    """Verify *every* minute from 09:15 to the last completed bar.

    A partial 40-bar tail must never masquerade as an anchored session VWAP.
    Timestamps are canonical start-of-bar, with availability at the bar end.
    """
    if not isinstance(observed, Sequence) or isinstance(observed, (str, bytes)):
        return [], "missing_completed_futures_bars"
    decoded: dict[datetime, dict[str, Any]] = {}
    for raw in observed:
        if not isinstance(raw, Mapping):
            return [], "invalid_futures_bar"
        if raw.get("is_provisional") is True or raw.get("is_complete") is not True:
            continue
        stamp = _time(raw.get("timestamp") or raw.get("datetime") or raw.get("time"))
        if stamp is None:
            return [], "invalid_futures_timestamp"
        if stamp.date() != decision_at.date():
            continue
        minute = stamp.hour * 60 + stamp.minute
        if not 9 * 60 + 15 <= minute < 15 * 60 + 30:
            continue
        if stamp.second != 0 or stamp.microsecond != 0:
            return [], "noncanonical_minute_timestamp"
        if stamp + timedelta(minutes=1) > decision_at:
            continue
        values = {
            key: _value(raw.get(key))
            for key in ("open", "high", "low", "close", "volume")
        }
        if any(number is None for number in values.values()):
            return [], "invalid_futures_bar"
        bar = {key: float(value) for key, value in values.items() if value is not None}
        if (
            min(bar["open"], bar["high"], bar["low"], bar["close"]) <= 0
            or bar["volume"] <= 0
            or not (
                bar["low"] <= bar["open"] <= bar["high"]
                and bar["low"] <= bar["close"] <= bar["high"]
            )
        ):
            return [], "invalid_futures_bar"
        if stamp in decoded and decoded[stamp] != bar:
            return [], "conflicting_futures_bar"
        decoded[stamp] = bar
    ordered = sorted(decoded.items())
    if len(ordered) < min_history:
        return [], "insufficient_verified_futures_history"
    anchor = decision_at.replace(hour=9, minute=15, second=0, microsecond=0)
    if ordered[0][0] != anchor:
        return [], "session_open_anchor_missing"
    if (
        not 0
        <= (decision_at - ordered[-1][0] - timedelta(minutes=1)).total_seconds()
        <= 90.0
    ):
        return [], "futures_completed_bar_stale"
    for left, right in zip(ordered, ordered[1:]):
        if right[0] - left[0] != timedelta(minutes=1):
            return [], "futures_history_gap"
    return [
        {"timestamp": stamp.isoformat(), **bar} for stamp, bar in ordered
    ], "verified"


def _restore_authenticated_session_bars(
    live_bars: Any,
    recovered: Any,
    *,
    future: str,
    decision_at: datetime,
) -> tuple[Any, str]:
    """Merge only source-attested, completed 1m local history.

    Recovered bars must carry their original contract, canonical bar start
    and publication time; a provider that omits provenance cannot restore
    the 09:15 anchor. The normal structural checker enforces continuity.
    """
    if not isinstance(recovered, Sequence) or isinstance(recovered, (str, bytes)):
        return live_bars, "local_history_recovery_unavailable"
    live = list(live_bars) if isinstance(live_bars, Sequence) else []
    verified: list[Mapping[str, Any]] = []
    for bar in recovered:
        if not isinstance(bar, Mapping):
            return live_bars, "unverified_recovered_bar"
        start = _time(bar.get("timestamp"))
        available = _time(bar.get("available_at"))
        if (
            bar.get("source") != "local_persisted_1m_futures"
            or canonical_nifty_future_symbol(bar.get("symbol")) != future
            or bar.get("is_complete") is not True
            or bar.get("is_provisional") is True
            or start is None
            or available is None
            or start.date() != decision_at.date()
            or start.second != 0
            or start.microsecond != 0
            or not timedelta(minutes=1) <= available - start <= timedelta(minutes=3)
            or available > decision_at
        ):
            return live_bars, "unverified_recovered_bar"
        verified.append(bar)
    # Merge by bar identity, preserving independent collision checks: if
    # cached and in-memory candles disagree, downstream rejects conflict.
    return [*verified, *live], "authenticated_local_session_recovery"


def assess_range_vwap_reclaim(
    bars: Any,
    *,
    decision_at: datetime,
    config: RangeVWAPReclaimStrategyConfig,
) -> dict[str, Any]:
    """Qualify a *completed* band re-entry using only prior-bar band inputs.

    Session VWAP is the primary target, never the rolling 20-bar center.
    The rolling center independently corroborates a plausible local equilibrium.
    """
    rows, reason = _verified_completed_bars(
        bars, decision_at=decision_at, min_history=config.min_history_bars
    )
    if reason != "verified":
        return {"qualified": False, "reason": reason}
    history, previous, latest = rows[:-1], rows[-2], rows[-1]
    if len(history) < 30:
        return {"qualified": False, "reason": "insufficient_prior_band_history"}

    # The last completed candle confirms the reclaim; its OHLCV must never
    # influence the already-existing session VWAP or entry band threshold.
    session_vwap = _weighted_center(history)
    session_vwap_5m_ago = _weighted_center(history[:-5])
    rolling_center = _weighted_center(history[-20:])
    if None in {session_vwap, session_vwap_5m_ago, rolling_center}:
        return {"qualified": False, "reason": "session_vwap_unavailable"}
    assert session_vwap is not None
    assert session_vwap_5m_ago is not None
    assert rolling_center is not None

    true_ranges = []
    for i in range(1, len(history)):
        candle = history[i]
        previous_close = float(history[i - 1]["close"])
        true_ranges.append(
            max(
                float(candle["high"]) - float(candle["low"]),
                abs(float(candle["high"]) - previous_close),
                abs(float(candle["low"]) - previous_close),
            )
        )
    atr = statistics.mean(true_ranges[-14:])
    if atr <= 0:
        return {"qualified": False, "reason": "futures_atr_unavailable"}
    sigma = statistics.pstdev(float(r["close"]) for r in history[-20:])
    band = max(config.band_atr_multiple * atr, config.band_sigma_multiple * sigma)
    slope_bps = abs(session_vwap - session_vwap_5m_ago) / session_vwap * 10000 / 5
    closes = [float(row["close"]) for row in history[-25:]]
    travel = sum(abs(b - a) for a, b in zip(closes, closes[1:]))
    efficiency = abs(closes[-1] - closes[0]) / travel if travel > 0 else 1.0
    width = (
        max(float(r["high"]) for r in history[-30:])
        - min(float(r["low"]) for r in history[-30:])
    ) / atr
    baseline = statistics.mean(float(r["volume"]) for r in history[-20:])
    rvol = float(latest["volume"]) / baseline if baseline > 0 else 0.0
    midpoint_gap_atr = abs(session_vwap - rolling_center) / atr
    lower, upper = session_vwap - band, session_vwap + band
    bar_range = float(latest["high"]) - float(latest["low"])
    candle_body = abs(float(latest["close"]) - float(latest["open"]))
    body_fraction = candle_body / bar_range if bar_range > 0 else 0.0
    # Descriptive transition diagnostics use ONLY completed futures bars
    # before the reclaim candle. No untested volatility threshold gates LIVE.
    recent_closes = [float(row["close"]) for row in history[-6:]]
    recent_travel = sum(abs(b - a) for a, b in zip(recent_closes, recent_closes[1:]))
    recent_efficiency = (
        abs(recent_closes[-1] - recent_closes[0]) / recent_travel
        if recent_travel > 0
        else 1.0
    )
    recent_tr_ratio = statistics.mean(true_ranges[-5:]) / atr
    # Shadow comparator only: full-session dispersion, not a replacement
    # for the frozen 20-bar ATR/sigma entry band.
    session_sigma = statistics.pstdev(float(r["close"]) for r in history)
    anchored_band = max(
        config.band_atr_multiple * atr,
        config.band_sigma_multiple * session_sigma,
    )
    # Research-only local structure: use the same completed-bar efficiency
    # and width limits, without relabeling the canonical session regime.
    prior_rolling_center = _weighted_center(history[-25:-5])
    local_slope_bps = (
        abs(rolling_center - prior_rolling_center) / rolling_center * 10000 / 5
        if prior_rolling_center is not None and rolling_center > 0
        else None
    )
    local_range_candidate = bool(
        travel > 0
        and efficiency <= config.max_directional_efficiency
        and 1.5 <= width <= config.max_range_width_atr
        and local_slope_bps is not None
        and local_slope_bps <= config.max_vwap_slope_bps_per_min
    )
    # Confirm local structure across two adjacent completed windows. The
    # overlapping windows establish persistence, not independent samples.
    previous_closes = [float(row["close"]) for row in history[-26:-1]]
    previous_travel = sum(
        abs(b - a) for a, b in zip(previous_closes, previous_closes[1:])
    )
    previous_atr = statistics.mean(true_ranges[-15:-1])
    previous_center = _weighted_center(history[-21:-1])
    previous_center_5m_ago = _weighted_center(history[-26:-6])
    previous_range = history[-31:-1]
    previous_width = (
        max(float(row["high"]) for row in previous_range)
        - min(float(row["low"]) for row in previous_range)
    )
    local_persistent = bool(
        local_range_candidate
        and len(previous_range) == 30
        and previous_atr > 0
        and previous_travel > 0
        and abs(previous_closes[-1] - previous_closes[0]) / previous_travel
        <= config.max_directional_efficiency
        and 1.5 <= previous_width / previous_atr <= config.max_range_width_atr
        and previous_center is not None
        and previous_center_5m_ago is not None
        and previous_center > 0
        and abs(previous_center - previous_center_5m_ago) / previous_center * 2000
        <= config.max_vwap_slope_bps_per_min
    )
    local_lower, local_upper = rolling_center - band, rolling_center + band
    # Prior completed candles only: descriptive two-sided acceptance at the
    # frozen local band; these are not retrospectively executable entries.
    lower_rejections = sum(
        float(row["low"]) < local_lower <= float(row["close"]) < local_upper
        for row in history[-30:-1]
    )
    upper_rejections = sum(
        local_lower < float(row["close"]) <= local_upper < float(row["high"])
        for row in history[-30:-1]
    )
    current_close = float(latest["close"])
    prior_close = float(previous["close"])
    local_breakout = bool(
        current_close > max(float(row["high"]) for row in history[-30:])
        or current_close < min(float(row["low"]) for row in history[-30:])
        or (current_close > local_upper and current_close > prior_close > local_upper)
        or (current_close < local_lower and current_close < prior_close < local_lower)
    )
    two_sided = lower_rejections > 0 and upper_rejections > 0
    local_confirmed = bool(
        local_persistent
        and two_sided
        and recent_tr_ratio <= config.max_failed_breakout_recent_tr_ratio
        and not local_breakout
    )
    local_state = (
        "BREAKOUT"
        if local_breakout
        else "RANGE"
        if local_confirmed
        else "EMERGING_RANGE"
        if local_range_candidate
        else "UNPROVEN"
    )
    # A transparent evidence fraction is NOT a calibrated regime probability.
    local_score = sum(
        (local_range_candidate, local_persistent, two_sided, recent_tr_ratio <= 1.0)
    ) / 4.0
    if local_breakout:
        local_score = 0.0
    broader_closes = [float(row["close"]) for row in history[-60:]]
    broader_travel = sum(
        abs(b - a) for a, b in zip(broader_closes, broader_closes[1:])
    )
    broader_move = (
        broader_closes[-1] - broader_closes[0] if len(broader_closes) == 60 else None
    )
    broader_efficiency = (
        abs(broader_move) / broader_travel
        if broader_move is not None and broader_travel > 0
        else None
    )
    local_reentry = max(config.min_band_reentry_atr * atr, 0.05 * band)
    local_reclaim_side = None
    if (
        local_confirmed
        and rvol >= config.min_reclaim_volume_ratio
        and 0 < bar_range <= 2.0 * atr
        and body_fraction >= config.min_reclaim_body_fraction
    ):
        if (
            prior_close < local_lower
            and local_lower + local_reentry <= current_close < rolling_center
            and current_close > float(latest["open"])
            and float(latest["low"]) >= float(previous["low"])
        ):
            local_reclaim_side = "CE"
        elif (
            prior_close > local_upper
            and rolling_center < current_close <= local_upper - local_reentry
            and current_close < float(latest["open"])
            and float(latest["high"]) <= float(previous["high"])
        ):
            local_reclaim_side = "PE"
    evidence: dict[str, Any] = {
        "qualified": False,
        "local_range_candidate": local_range_candidate,
        "local_range_state": local_state,
        "local_range_confirmed": local_confirmed,
        "local_range_persistent": local_persistent,
        "local_range_evidence_fraction": local_score,
        "local_lower_boundary_rejections": lower_rejections,
        "local_upper_boundary_rejections": upper_rejections,
        "local_reclaim_side": local_reclaim_side,
        "local_range_breakout": local_breakout,
        "broader_60bar_directional_efficiency": broader_efficiency,
        "broader_60bar_displacement_atr": (
            broader_move / atr if broader_move is not None else None
        ),
        "futures_atr_bps": atr / float(previous["close"]) * 10000,
        "local_rolling_center_slope_bps_per_min": local_slope_bps,
        "candidate_session_anchored_band_points": round(anchored_band, 4),
        "candidate_session_anchored_vs_primary_ratio": (
            round(anchored_band / band, 4) if band > 0 else None
        ),
        "source": "contiguous_session_futures_1m",
        "bar_timestamp": latest["timestamp"],
        "primary_vwap_source": "session_0915_completed_volume_weighted_typical",
        "session_vwap_before_confirmation": round(session_vwap, 4),
        "rolling_20m_secondary_center": round(rolling_center, 4),
        "secondary_divergence_atr": round(midpoint_gap_atr, 4),
        "futures_vwap": round(session_vwap, 4),
        "futures_atr": round(atr, 4),
        "futures_sigma": round(sigma, 4),
        "band_points": round(band, 4),
        "lower_band_before_confirmation": round(lower, 4),
        "upper_band_before_confirmation": round(upper, 4),
        "session_vwap_slope_bps_per_min": round(slope_bps, 4),
        "directional_efficiency": round(efficiency, 4),
        "prior_5m_directional_efficiency": round(recent_efficiency, 4),
        "prior_5m_to_14m_true_range_ratio": round(recent_tr_ratio, 4),
        "reclaim_bar_range_atr": round(bar_range / atr, 4),
        "range_width_atr": round(width, 4),
        "volume_ratio": round(rvol, 4),
        "volume_reference": "trailing_20_completed_futures_minutes_not_clock_matched",
        "volume_reference_mean": round(baseline, 4),
        "underlying_latest_close": float(latest["close"]),
        # Prospective diagnostics only: do not curve-fit new LIVE filters
        # without an independent after-cost option-execution sample.
        # Excursion identity is observed from the previous completed bar,
        # BEFORE any filter, without treating it as a tradeable reversal.
        "excursion_side": (
            "CE"
            if float(previous["close"]) < lower and float(previous["low"]) < lower
            else (
                "PE"
                if float(previous["close"]) > upper and float(previous["high"]) > upper
                else None
            )
        ),
        "reclaim_body_fraction": round(body_fraction, 4),
        "reclaim_lower_wick_fraction": (
            round(
                (
                    min(float(latest["open"]), float(latest["close"]))
                    - float(latest["low"])
                )
                / bar_range,
                4,
            )
            if bar_range > 0
            else 0.0
        ),
        "reclaim_upper_wick_fraction": (
            round(
                (
                    float(latest["high"])
                    - max(float(latest["open"]), float(latest["close"]))
                )
                / bar_range,
                4,
            )
            if bar_range > 0
            else 0.0
        ),
    }
    excursion_side = evidence["excursion_side"]
    if excursion_side is not None:
        first_excursion_bar = None
        for bar in reversed(history):
            close = float(bar["close"])
            if (excursion_side == "CE" and close >= lower) or (
                excursion_side == "PE" and close <= upper
            ):
                break
            first_excursion_bar = bar["timestamp"]
        evidence["excursion_first_bar_timestamp"] = first_excursion_bar

    # Reuse the *two completed futures bars*; do not add a correlated RSI/ADX
    # confirmation or use the forming bar. The volatility comparison excludes
    # the reclaim candle and is not by itself a prediction of profitability.
    previous_close = float(previous["close"])
    latest_close = float(latest["close"])
    if excursion_side == "CE":
        extension_atr = (lower - previous_close) / atr
        reentry_atr = (latest_close - lower) / atr
        reversal_progress_atr = (latest_close - previous_close) / atr
        direction_confirmed = (
            latest_close > float(latest["open"])
            and latest_close < min(session_vwap, rolling_center)
            and float(latest["low"]) >= float(previous["low"])
        )
    elif excursion_side == "PE":
        extension_atr = (previous_close - upper) / atr
        reentry_atr = (upper - latest_close) / atr
        reversal_progress_atr = (previous_close - latest_close) / atr
        direction_confirmed = (
            latest_close < float(latest["open"])
            and latest_close > max(session_vwap, rolling_center)
            and float(latest["high"]) <= float(previous["high"])
        )
    else:
        extension_atr = 0.0
        reentry_atr = 0.0
        reversal_progress_atr = 0.0
        direction_confirmed = False
    failed_breakout_quality = bool(
        direction_confirmed
        and extension_atr >= 0.05
        and reentry_atr >= config.min_failed_breakout_reentry_atr
        and reversal_progress_atr >= config.min_failed_breakout_reentry_atr
        and recent_tr_ratio <= config.max_failed_breakout_recent_tr_ratio
        and rvol >= config.min_reclaim_volume_ratio
        and 0 < bar_range <= 2.0 * atr
        and body_fraction >= config.min_reclaim_body_fraction
    )
    evidence["failed_breakout_quality"] = {
        "exterior_close_distance_atr": round(extension_atr, 4),
        "completed_reentry_depth_atr": round(reentry_atr, 4),
        "completed_reversal_progress_atr": round(reversal_progress_atr, 4),
        "prior_5m_tr_ratio": round(recent_tr_ratio, 4),
        "prior_5m_directional_efficiency": round(recent_efficiency, 4),
        "qualified_for_center_exception": failed_breakout_quality,
    }
    evidence["conditional_center_exception_used"] = False
    if slope_bps > config.max_vwap_slope_bps_per_min:
        return {**evidence, "reason": "session_vwap_sloping"}
    if efficiency > config.max_directional_efficiency:
        return {**evidence, "reason": "directional_trend_not_range"}
    if not 1.5 <= width <= config.max_range_width_atr:
        return {**evidence, "reason": "range_width_unproven"}
    if midpoint_gap_atr > config.max_secondary_center_deviation_atr:
        # A wider center is *not* a generic signal. Only the original 1.0
        # to 1.25 ATR interval is eligible, with completed-bar exhaustion
        # evidence and every unchanged range, data and broker risk gate.
        conditional_upper_gap = min(config.max_conditional_center_deviation_atr, 1.25)
        conditional = bool(
            config.enable_conditional_center_tolerance
            and config.max_secondary_center_deviation_atr >= 1.0
            and 1.0 < midpoint_gap_atr <= conditional_upper_gap
            and failed_breakout_quality
        )
        if not conditional:
            return {**evidence, "reason": "session_rolling_center_disagree"}
        evidence["conditional_center_exception_used"] = True
    if rvol < config.min_reclaim_volume_ratio:
        return {**evidence, "reason": "reclaim_volume_weak"}
    if bar_range <= 0 or bar_range > 2.0 * atr:
        return {**evidence, "reason": "impulsive_or_invalid_reclaim_bar"}
    if body_fraction < config.min_reclaim_body_fraction:
        return {**evidence, "reason": "weak_reclaim_body"}

    reentry = max(config.min_band_reentry_atr * atr, 0.05 * band)
    # Reclaim geometry and remaining payoff are independent. Net option RR
    # below determines whether a genuine structural reclaim is tradable.
    last_close = float(latest["close"])
    bullish = bool(
        float(previous["low"]) < lower
        and float(previous["close"]) < lower
        and last_close >= lower + reentry
        and float(latest["low"]) >= float(previous["low"])
        and last_close > float(latest["open"])
        and last_close > float(previous["close"])
        and last_close < min(session_vwap, rolling_center)
    )
    bearish = bool(
        float(previous["high"]) > upper
        and float(previous["close"]) > upper
        and last_close <= upper - reentry
        and float(latest["high"]) <= float(previous["high"])
        and last_close < float(latest["open"])
        and last_close < float(previous["close"])
        and last_close > max(session_vwap, rolling_center)
    )
    if bullish == bearish:
        return {**evidence, "reason": "no_completed_band_reentry"}
    side = "CE" if bullish else "PE"
    invalidation = float(previous["low"]) if bullish else float(previous["high"])
    return {
        **evidence,
        "qualified": True,
        "side": side,
        "reason": "precomputed_band_closed_reentry",
        "reversal_direction_source": "two_completed_futures_candles",
        "underlying_entry_close": last_close,
        "reachable_futures_distance": round(abs(session_vwap - last_close), 4),
        "invalidation_futures": invalidation,
        "previous_excursion_timestamp": previous["timestamp"],
        "remaining_vwap_distance_atr": round(abs(session_vwap - last_close) / atr, 4),
        "reclaim_confirmation_without_minimum_distance_gate": True,
    }


def assess_option_cost_risk(
    option_quote: Mapping[str, Any],
    *,
    config: RangeVWAPReclaimStrategyConfig,
    side: str | None = None,
    futures_target_distance: float | None = None,
    futures_stop_distance: float | None = None,
) -> dict[str, Any]:
    """Research-only ask-entry/bid-exit repricing with explicit spot basis.

    Greeks must be attributed to the NIFTY spot underlying. A contemporaneous
    spot/futures pair is mandatory; no futures-points=spot-points identity is
    silently assumed. Fixed-basis and adverse-basis/IV scenarios are disclosed.
    """
    bid = _value(option_quote.get("bid"))
    ask = _value(option_quote.get("ask"))
    atr = _value(option_quote.get("atr"))
    lot = _value(option_quote.get("lot_size"))
    if lot is None or not 1 <= lot <= 1000 or not lot.is_integer():
        return {"qualified": False, "reason": "historical_lot_size_unverified"}
    quantity = int(lot)
    if bid is None or ask is None or atr is None or bid <= 0 or ask <= bid or atr <= 0:
        return {"qualified": False, "reason": "option_quote_or_atr_missing"}
    if side not in {"CE", "PE"}:
        return {"qualified": False, "reason": "option_contract_side_missing"}
    delta = _value(option_quote.get("delta"))
    if delta is None or not (
        config.min_abs_option_delta <= abs(delta) <= config.max_abs_option_delta
    ):
        return {"qualified": False, "reason": "option_delta_unverified"}
    if (side == "CE" and delta <= 0) or (side == "PE" and delta >= 0):
        return {"qualified": False, "reason": "option_delta_side_conflict"}
    distance = _value(futures_target_distance)
    invalidation_gap = _value(futures_stop_distance)
    if (
        distance is None
        or invalidation_gap is None
        or distance <= 0
        or invalidation_gap <= 0
    ):
        return {"qualified": False, "reason": "structural_futures_distance_missing"}
    if (ask - bid) / ask * 100 > config.max_option_spread_pct:
        return {"qualified": False, "reason": "option_spread_too_wide"}

    spot = _value(option_quote.get("spot_ltp"))
    future = _value(option_quote.get("futures_ltp"))
    if spot is None or future is None or spot <= 0 or future <= 0:
        return {"qualified": False, "reason": "spot_futures_basis_unverified"}
    if option_quote.get("delta_underlying") != "NIFTY_SPOT":
        return {"qualified": False, "reason": "option_delta_underlying_unverified"}
    fut_atr = _value(option_quote.get("futures_atr"))
    if fut_atr is None or fut_atr <= 0:
        return {"qualified": False, "reason": "futures_volatility_unverified"}
    basis = future - spot
    direction = 1.0 if side == "CE" else -1.0
    # Constant-basis spot target is explicitly an assumption, stress it.
    fixed_basis_target = future + direction * distance - basis
    adverse_spot_distance = max(
        0.0,
        direction * (fixed_basis_target - spot)
        - config.adverse_basis_change_atr * fut_atr,
    )
    theta = _value(option_quote.get("theta"))
    theta_period = str(option_quote.get("theta_period") or "")
    if theta is not None and theta_period in {"calendar_day", "trading_day"}:
        minutes_per_day = 1440.0 if theta_period == "calendar_day" else 375.0
        theta_loss = max(0.0, -theta) * config.assumed_hold_minutes / minutes_per_day
        theta_source = "observed_theta_" + theta_period + "_linear_stress"
    else:
        theta_loss = ask * config.unknown_theta_haircut_pct / 100.0
        theta_source = "unverified_theta_convention_premium_haircut"

    gamma = _value(option_quote.get("gamma"))
    gamma_valid = bool(
        gamma is not None
        and gamma >= 0.0
        and option_quote.get("gamma_underlying") == "NIFTY_SPOT"
    )
    vega = _value(option_quote.get("vega_per_iv_point"))
    vega_valid = bool(
        vega is not None
        and vega >= 0.0
        and option_quote.get("vega_convention") == "per_iv_percentage_point"
    )
    iv_penalty = (
        vega * config.adverse_iv_change_points
        if vega_valid and vega is not None
        else ask * config.unknown_iv_haircut_pct / 100.0
    )
    # Do not use positive gamma to subsidize conservative entry admission.
    nominal_gain = abs(delta) * distance - theta_loss
    adverse_gain = abs(delta) * adverse_spot_distance - theta_loss - iv_penalty
    reachable_gain = max(0.0, adverse_gain)
    premium_stop_points = max(
        0.75 * atr,
        0.035 * ask,
        abs(delta) * invalidation_gap * 1.10,
    )
    if premium_stop_points >= ask:
        return {"qualified": False, "reason": "option_stop_geometry_invalid"}
    if reachable_gain <= 0.0:
        return {"qualified": False, "reason": "vwap_target_not_reachable_after_decay"}

    slippage = config.slippage_stress_bps / 10000.0
    paid = ask * (1.0 + slippage)
    spread = ask - bid
    # Keep the executable-bid stop trigger distinct from an adverse fill.
    # The additional spread allows for bid deterioration after the trigger;
    # neither estimate guarantees a stop fill through a gap.
    stop_trigger = ask - premium_stop_points - spread
    stop_fill = stop_trigger * (1.0 - slippage) - spread
    if stop_trigger <= 0.0 or stop_fill <= 0.0:
        return {"qualified": False, "reason": "liquidation_price_invalid"}
    stop_cost = estimate_round_trip_cost(
        entry_price=paid, exit_price=stop_fill, quantity=quantity
    ).total
    loss = (paid - stop_fill) * quantity + stop_cost

    def modeled_exit(gain: float) -> dict[str, float]:
        capped = min(max(0.0, gain), config.gross_target_r * premium_stop_points)
        exit_bid = (ask + capped) * (1.0 - slippage) - spread
        fees = estimate_round_trip_cost(
            entry_price=paid,
            exit_price=max(exit_bid, 0.01),
            quantity=quantity,
        ).total
        net_reward = (exit_bid - paid) * quantity - fees
        return {
            "exit_bid": round(exit_bid, 4),
            "net_reward": round(net_reward, 2),
            "net_rr": round(net_reward / loss, 4) if loss > 0 else 0.0,
        }

    scenarios = {
        "pre_vwap_80pct": modeled_exit(
            abs(delta) * min(distance * 0.8, adverse_spot_distance)
            - theta_loss
            - iv_penalty
        ),
        "at_vwap_fixed_basis": modeled_exit(nominal_gain),
        "at_vwap_adverse_basis_iv": modeled_exit(reachable_gain),
        "stalled_time_stop": modeled_exit(-theta_loss - iv_penalty),
    }
    # Use the achievable 80%-of-VWAP exit *under stress*, not an optimistic
    # full-VWAP target, to admit research opportunities.
    selected = scenarios["pre_vwap_80pct"]
    net_rr = selected["net_rr"]
    report = {
        "qualified": False,
        "model_only": True,
        "observed_contract_lot_size": quantity,
        "observed_delta": round(delta, 5),
        "delta_underlying_source": "NIFTY_SPOT",
        "observed_spot": round(spot, 4),
        "observed_future": round(future, 4),
        "observed_basis_points": round(basis, 4),
        "assumed_spot_target_fixed_basis": round(fixed_basis_target, 4),
        "adverse_spot_move_points": round(adverse_spot_distance, 4),
        "futures_session_target_points": round(distance, 4),
        "futures_structural_stop_points": round(invalidation_gap, 4),
        "assumed_hold_minutes": config.assumed_hold_minutes,
        "theta_cost_source": theta_source,
        "estimated_theta_cost_points": round(theta_loss, 4),
        "gamma_usable": gamma_valid,
        "gamma_upside_at_vwap_points": (
            round(0.5 * float(gamma or 0.0) * distance * distance, 4)
            if gamma_valid
            else None
        ),
        "vega_usable": vega_valid,
        "adverse_iv_cost_points": round(iv_penalty, 4),
        "projected_premium_gain_points": round(reachable_gain, 4),
        "slippage_each_side_bps": config.slippage_stress_bps,
        "modeled_extra_exit_spread_points": round(spread, 4),
        "premium_stop_points": round(premium_stop_points, 4),
        "stop_price_model": round(stop_trigger, 4),
        "modeled_stressed_stop_fill_price": round(stop_fill, 4),
        "vwap_limited_target_price_model": selected["exit_bid"],
        "worst_stop_loss_including_costs": round(loss, 2),
        "target_net_reward_after_costs": selected["net_reward"],
        "net_reward_risk": net_rr,
        "pricing_scenarios": scenarios,
        "unmodeled_risks": [
            "basis_jump",
            "iv_skew",
            "quote_depth",
            "stop_gap",
            "partial_fills",
            "latency",
        ],
    }
    if loss > min(config.max_trade_cash_risk, 773.0):
        return {**report, "reason": "cash_risk_over_cap"}
    if net_rr < config.min_net_reward_risk:
        return {**report, "reason": "after_cost_reward_risk_low"}
    return {**report, "qualified": True, "reason": "modeled_session_target_ok"}


class RangeVWAPReclaimStrategy(EliteStrategy):  # type: ignore[misc]
    """Session-VWAP reversal context; broker trigger only in a guarded pilot."""

    def __init__(
        self, config: RangeVWAPReclaimStrategyConfig, indicator_engine: Any
    ) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config
        self._seen: set[str] = set()
        self._regime_bars: dict[str, tuple[str, int]] = {}
        self._consumed_excursions: set[tuple[str, str, str]] = set()
        self._pending_excursions: dict[str, tuple[str, str, str]] = {}
        self._guard_owner: Any | None = None
        self._guard_date: str | None = None
        self._guard_lock = RLock()
        self._opportunities: dict[str, dict[str, Any]] = {}
        self._active_opportunity: str | None = None
        self._research_observed_minute: dict[tuple[str, str], str] = {}

    def attach_persistent_guard(self, position_manager: Any) -> None:
        """Reuse canonical position/risk persistence; never open a second store."""
        self._guard_owner = position_manager

    def _load_persistent_guard(self, decision_at: datetime) -> bool:
        """Restore only current-session reservations, or fail closed for LIVE."""
        if not range_vwap_live_enabled():
            return True
        owner = self._guard_owner
        reader = getattr(owner, "get_risk_circuit_state", None)
        writer = getattr(owner, "persist_range_vwap_setup_guard", None)
        today = getattr(owner, "_trading_date_ist", None)
        if not callable(reader) or not callable(writer) or not callable(today):
            return False
        day = decision_at.astimezone(_IST).date().isoformat()
        if today() != day:
            return False
        with self._guard_lock:
            if self._guard_date == day:
                return True
            try:
                parent = reader()
                if not isinstance(parent, Mapping):
                    return False
                saved = (
                    parent.get("range_vwap_setup_guard")
                    if parent.get("trading_date") == day
                    else None
                )
                if saved is None:
                    seen, consumed, pending = set(), set(), {}
                else:
                    if (
                        not isinstance(saved, Mapping)
                        or saved.get("version") != 1
                        or saved.get("trading_date") != day
                        or not isinstance(saved.get("seen"), list)
                        or not isinstance(saved.get("consumed"), list)
                        or not isinstance(saved.get("pending"), Mapping)
                    ):
                        return False
                    seen = set(saved["seen"])
                    consumed = {tuple(item) for item in saved["consumed"]}
                    pending = {
                        str(identity): tuple(item)
                        for identity, item in saved["pending"].items()
                    }
                    if (
                        len(seen) > 512
                        or len(consumed) > 128
                        or len(pending) > 512
                        or not all(
                            isinstance(identity, str)
                            and identity.startswith("range_session_vwap:")
                            for identity in seen
                        )
                        or not all(
                            len(key) == 3
                            and isinstance(key[0], str)
                            and key[1] in {"CE", "PE"}
                            and key[2] == day
                            for key in consumed
                        )
                        or not all(
                            identity in seen and key in consumed
                            for identity, key in pending.items()
                        )
                    ):
                        return False
                self._seen = seen
                self._consumed_excursions = consumed
                self._pending_excursions = pending
                self._guard_date = day
                return True
            except (OSError, TypeError, ValueError, AttributeError):
                return False

    def _persist_guard(self) -> bool:
        """Reserve BEFORE returning a live order signal, verify state on disk."""
        if not range_vwap_live_enabled():
            return True
        owner = self._guard_owner
        writer = getattr(owner, "persist_range_vwap_setup_guard", None)
        if not callable(writer) or self._guard_date is None:
            return False
        with self._guard_lock:
            if (
                len(self._seen) > 512
                or len(self._consumed_excursions) > 128
                or len(self._pending_excursions) > 512
            ):
                return False
            payload = {
                "version": 1,
                "trading_date": self._guard_date,
                "seen": sorted(self._seen),
                "consumed": [list(key) for key in sorted(self._consumed_excursions)],
                "pending": {
                    identity: list(key)
                    for identity, key in sorted(self._pending_excursions.items())
                },
            }
            try:
                return writer(payload) is True
            except (OSError, TypeError, ValueError, RuntimeError):
                return False

    def _observe_excursion(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        structural: Mapping[str, Any],
        *,
        decision_at: datetime,
        reason: str | None = None,
    ) -> None:
        """Research-only: log one immutable first observation per futures excursion."""
        from nifty_scalper_bot.utils.symbols import normalize_symbol

        side = symbol[-2:].upper()
        future = canonical_nifty_future_symbol(indicators.get("active_futures_symbol"))
        first = str(structural.get("excursion_first_bar_timestamp") or "")
        selected_symbol = indicators.get(
            "selected_ce" if side == "CE" else "selected_pe"
        )
        selected = indicators.get("is_selected_option") is True or normalize_symbol(
            str(selected_symbol or "")
        ) == normalize_symbol(symbol)
        if future is None or not selected:
            return
        # Observe local structure independently of session-VWAP excursions.
        # The existing research dedup map also owns one record per future/bar.
        bar_stamp = str(structural.get("bar_timestamp") or "")
        local_key = (future, "LOCAL")
        if bar_stamp and structural.get("local_range_state") is not None:
            with self._guard_lock:
                new_local_bar = self._research_observed_minute.get(local_key) != bar_stamp
                if new_local_bar:
                    self._research_observed_minute[local_key] = bar_stamp
            if new_local_bar:
                LOGGER.info(
                    "RANGE_VWAP_LOCAL_REGIME %s",
                    json.dumps(
                        {
                            "v": 1,
                            "decision_at": decision_at.isoformat(),
                            "futures_symbol": future,
                            "bar_timestamp": bar_stamp,
                            "regime": str(indicators.get("regime") or "UNKNOWN"),
                            "regime_raw": indicators.get("regime_raw"),
                            "regime_gate_allowed": indicators.get("regime_gate_allowed"),
                            "regime_transition_pending": indicators.get(
                                "regime_transition_pending"
                            ),
                            "session_vwap_setup_qualified": structural.get("qualified"),
                            **{
                                key: structural.get(key)
                                for key in (
                                    "source",
                                    "local_range_state",
                                    "local_range_evidence_fraction",
                                    "local_range_persistent",
                                    "local_lower_boundary_rejections",
                                    "local_upper_boundary_rejections",
                                    "local_reclaim_side",
                                    "local_range_breakout",
                                    "broader_60bar_directional_efficiency",
                                    "broader_60bar_displacement_atr",
                                    "futures_atr_bps",
                                    "range_width_atr",
                                    "prior_5m_to_14m_true_range_ratio",
                                    "local_rolling_center_slope_bps_per_min",
                                    "session_vwap_before_confirmation",
                                    "rolling_20m_secondary_center",
                                )
                            },
                        },
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                )
        if structural.get("excursion_side") != side or not first:
            return
        identity = f"{future}:{side}:{first}"
        with self._guard_lock:
            if identity in self._opportunities:
                self._opportunities[identity][
                    "_latest_evaluation_at"
                ] = decision_at.isoformat()
                self._active_opportunity = identity
                return
            bid = _value(indicators.get("bid"))
            ask = _value(indicators.get("ask"))
            lot = indicators.get("lot_size")
            if type(lot) is not int or lot <= 0:
                lot = None
            quote_ok = (
                bid is not None
                and ask is not None
                and 0 < bid <= ask
                and indicators.get("tradable_quote") is True
            )
            record = {
                "v": 1,
                "decision_at": decision_at.isoformat(),
                "excursion_id": identity,
                "futures_symbol": future,
                "side": side,
                "symbol": normalize_symbol(symbol),
                "excursion_first_bar": first,
                "structural_reason": str(structural.get("reason") or "unverified"),
                "structural_qualified": structural.get("qualified") is True,
                "regime": str(indicators.get("regime") or "UNKNOWN"),
                "regime_raw": indicators.get("regime_raw"),
                "regime_gate_allowed": indicators.get("regime_gate_allowed"),
                "local_range_candidate": structural.get("local_range_candidate"),
                "local_range_state": structural.get("local_range_state"),
                "local_range_evidence_fraction": structural.get(
                    "local_range_evidence_fraction"
                ),
                "local_reclaim_side": structural.get("local_reclaim_side"),
                "local_range_confirmed_in_trend": (
                    structural.get("local_range_confirmed") is True
                    and str(indicators.get("regime") or "").upper() == "TREND"
                    and str(indicators.get("regime_raw") or "").upper() == "TREND"
                    and indicators.get("regime_gate_allowed") is True
                    and indicators.get("regime_transition_pending") is not True
                ),
                "local_rolling_center_slope_bps_per_min": _value(
                    structural.get("local_rolling_center_slope_bps_per_min")
                ),
                "local_range_session_vwap_setup": (
                    structural.get("local_range_candidate") is True
                    and structural.get("qualified") is True
                    and str(indicators.get("regime") or "").upper() == "TREND"
                    and str(indicators.get("regime_raw") or "").upper() == "TREND"
                    and indicators.get("regime_gate_allowed") is True
                    and indicators.get("regime_transition_pending") is not True
                ),
                "regime_transition": (
                    indicators.get("regime_transition_pending") is True
                ),
                "session_vwap": _value(
                    structural.get("session_vwap_before_confirmation")
                ),
                "rolling_center": _value(
                    structural.get("rolling_20m_secondary_center")
                ),
                "center_gap_atr": _value(structural.get("secondary_divergence_atr")),
                "futures_atr": _value(structural.get("futures_atr")),
                "range_width_atr": _value(structural.get("range_width_atr")),
                "volatility_ratio": _value(
                    structural.get("prior_5m_to_14m_true_range_ratio")
                ),
                "directional_efficiency": _value(
                    structural.get("prior_5m_directional_efficiency")
                ),
                "quote": {
                    "bid": bid if quote_ok else None,
                    "ask": ask if quote_ok else None,
                    "lot_size": lot,
                    "tradable": quote_ok,
                    "source": str(indicators.get("quote_source") or "")[:32],
                    "age_seconds": _value(indicators.get("data_age_seconds")),
                },
                "status": "observed_not_traded",
                "dominant_rejection": None,
                "identity": identity,
                "observed_decisions": 0,
                "_latest_evaluation_at": decision_at.isoformat(),
                "shadow_net_pnl": None,
                "option_fills_verified": False,
            }
            self._opportunities[identity] = record
            self._active_opportunity = identity
            if len(self._opportunities) > 512:
                self._opportunities.pop(next(iter(self._opportunities)))
        LOGGER.info(
            "RANGE_VWAP_EXCURSION_OBSERVED %s",
            json.dumps(record, separators=(",", ":"), allow_nan=False),
        )
        if reason:
            self._record_excursion_decision(reason)

    def _record_excursion_decision(self, reason: str) -> None:
        """Log only distinct decision transitions, never each tick."""
        identity = self._active_opportunity
        if not identity or identity not in self._opportunities:
            return
        record = self._opportunities[identity]
        seen = record.setdefault("_research_logged_reasons", [])
        if reason in seen or len(seen) >= 16:
            return
        seen.append(reason)
        record["observed_decisions"] = len(seen)
        if reason not in {"cost_eligible", "signal_emitted"}:
            record["dominant_rejection"] = reason
            record["status"] = "rejected"
        elif reason == "signal_emitted":
            record["status"] = "signal_emitted"
        budget = record.get("option_cost_risk") or {}
        LOGGER.info(
            "RANGE_VWAP_EXCURSION_DECISION %s",
            json.dumps(
                {
                    "v": 1,
                    "decision_at": record["_latest_evaluation_at"],
                    "excursion_id": identity,
                    "reason": reason[:100],
                    "symbol": record["symbol"],
                    "modeled_stop_risk": _value(
                        budget.get("worst_stop_loss_including_costs")
                    ),
                    "modeled_net_rr": _value(budget.get("net_reward_risk")),
                },
                separators=(",", ":"),
                allow_nan=False,
            ),
        )

    def observe_research_excursion(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        *,
        blocked_reason: str,
    ) -> None:
        """Observe manager-vetoed futures excursions without admitting a trade."""
        from nifty_scalper_bot.utils.symbols import normalize_symbol

        if canonical_nifty_option_symbol(symbol) is None:
            return
        side = symbol[-2:].upper()
        selected_symbol = indicators.get(
            "selected_ce" if side == "CE" else "selected_pe"
        )
        if indicators.get("is_selected_option") is not True and normalize_symbol(
            str(selected_symbol or "")
        ) != normalize_symbol(symbol):
            return
        future = canonical_nifty_future_symbol(indicators.get("active_futures_symbol"))
        decision = _time(indicators.get("decision_at"))
        if future is None or decision is None:
            return
        minute = decision.replace(second=0, microsecond=0).isoformat()
        key = (future, symbol[-2:])
        with self._guard_lock:
            if self._research_observed_minute.get(key) == minute:
                return
            self._research_observed_minute[key] = minute
            if len(self._research_observed_minute) > 32:
                self._research_observed_minute.clear()
        getter = getattr(self._indicator_engine, "get_history", None)
        if not callable(getter):
            return
        try:
            bars = getter(future, count=400, field="bars")
            evidence = assess_range_vwap_reclaim(
                bars, decision_at=decision, config=self._cfg
            )
        except (TypeError, ValueError, KeyError, AttributeError):
            return
        # This is an observation only; NEVER call live lifecycle or reserve.
        self._active_opportunity = None
        self._observe_excursion(
            symbol,
            indicators,
            evidence,
            decision_at=decision,
            reason=blocked_reason,
        )
        self._active_opportunity = None

    def notify_entry_accepted(self, side: str, *, setup_id: str | None = None) -> None:
        """Finalize an emitted setup through the existing runner order callback."""
        with self._guard_lock:
            key = self._pending_excursions.get(str(setup_id or ""))
            if key is None or key[1] != str(side or "").upper():
                return
            self._pending_excursions.pop(str(setup_id), None)
            if not self._persist_guard():
                # Keep durable pending state if broker ACK persistence fails.
                self._pending_excursions[str(setup_id)] = key
        for row in self._opportunities.values():
            if row.get("setup_id") == setup_id:
                row["status"] = "broker_order_accepted"
                return

    def notify_entry_rejected(
        self, side: str, *, setup_id: str | None = None, reason: str | None = None
    ) -> None:
        """Release only proven pre-broker capacity rejects; never unknown orders."""
        if reason != "no_affordable_execution_candidate":
            return
        identity = str(setup_id or "")
        with self._guard_lock:
            key = self._pending_excursions.get(identity)
            if key is None or key[1] != str(side or "").upper():
                return
            self._pending_excursions.pop(identity, None)
            self._seen.discard(identity)
            self._consumed_excursions.discard(key)
            if not self._persist_guard():
                # Failed release must retain the original reservation.
                self._seen.add(identity)
                self._consumed_excursions.add(key)
                self._pending_excursions[identity] = key
                return
        for row in self._opportunities.values():
            if row.get("setup_id") == identity:
                row["status"] = "prebroker_rejected"
                row["prebroker_rejection_reason"] = reason
                return

    def _no_vote(self, reason: str) -> None:
        super()._no_vote(reason)
        key = self._active_opportunity
        if key and key in self._opportunities:
            row = self._opportunities[key]
            self._record_excursion_decision(reason)
            if row.get("modeled_cost_eligible"):
                row["post_cost_rejection"] = reason
            else:
                row["status"] = "rejected"
                row["dominant_rejection"] = reason

    def get_research_snapshot(self) -> dict[str, Any]:
        """Compact opportunity counts, explicitly not realized option P&L."""
        from collections import Counter

        records = list(self._opportunities.values())
        rejected = Counter(
            r.get("dominant_rejection")
            for r in records
            if r.get("status") == "rejected"
        )
        return {
            "validated_range_evaluations": len(records),
            "completed_band_reclaims": sum(
                r.get("structural_qualified") is True for r in records
            ),
            "cost_eligible_candidates": sum(
                r.get("modeled_cost_eligible") is True for r in records
            ),
            "broker_orders_accepted": sum(
                r.get("status") == "broker_order_accepted" for r in records
            ),
            "prebroker_capacity_rejections": sum(
                r.get("status") == "prebroker_rejected" for r in records
            ),
            "post_cost_rejection_reasons": list(
                Counter(
                    r.get("post_cost_rejection")
                    for r in records
                    if r.get("post_cost_rejection")
                ).most_common(5)
            ),
            "dominant_rejection": rejected.most_common(5),
            "shadow_net_expectancy": None,
            "shadow_mfe_mae": None,
            "option_fills_verified": False,
            "opportunities": records[-100:],
        }

    def get_required_indicators(self) -> list[str]:
        return []

    def _regime_bar_confirmed(self, future: str, timestamp: str) -> bool:
        """Require contiguous, forward-moving completed candle observations."""
        current = _time(timestamp)
        if current is None:
            return False
        required = max(1, int(self._cfg.min_regime_completed_bars))
        previous, n = self._regime_bars.get(future, ("", 0))
        earlier = _time(previous)
        if current == earlier:
            return bool(n >= required)
        if earlier is None or current - earlier != timedelta(minutes=1):
            n = 1
        else:
            n = min(required, n + 1)
        self._regime_bars[future] = timestamp, n
        return bool(n >= required)

    def _maybe_reset_excursions(self, future: str, evidence: Mapping[str, Any]) -> None:
        """Rearm only after futures actually return near session VWAP."""
        last = _value(evidence.get("underlying_latest_close"))
        center = _value(evidence.get("session_vwap_before_confirmation"))
        atr = _value(evidence.get("futures_atr"))
        if last is None or center is None or atr is None or atr <= 0:
            return
        if abs(last - center) > 0.25 * atr:
            return
        with self._guard_lock:
            previous = self._consumed_excursions
            # An uncertain broker outcome must not rearm just because futures
            # touched VWAP; retain it until existing entry callback resolves.
            unresolved = set(self._pending_excursions.values())
            retained = {
                key for key in previous if key[0] != future or key in unresolved
            }
            if retained == previous:
                return
            self._consumed_excursions = retained
            if not self._persist_guard():
                self._consumed_excursions = previous

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: dict[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
        self._active_opportunity = None
        if canonical_nifty_option_symbol(symbol) is None:
            self._no_vote("not_nifty_option")
            return None
        if position is not None:
            self._no_vote("range_position_already_present")
            return None
        side = symbol[-2:].upper()
        decision_at = _time(indicators.get("decision_at")) or datetime.now(_IST)
        minute = decision_at.hour * 60 + decision_at.minute
        if not 10 * 60 <= minute <= 14 * 60 + 30:
            self._no_vote("range_research_time_window")
            return None

        from nifty_scalper_bot.config.regime_ontology import (
            MarketRegime,
            normalize_regime,
        )

        regime = normalize_regime(
            indicators.get("regime") or indicators.get("market_regime")
        )
        future = canonical_nifty_future_symbol(indicators.get("active_futures_symbol"))
        if future is None:
            self._no_vote("range_futures_contract_missing")
            return None
        if regime is not MarketRegime.RANGE:
            self._regime_bars.pop(future, None)
            self._no_vote("not_verified_range_regime")
            return None
        if indicators.get("regime_transition_pending") is True:
            self._regime_bars.pop(future, None)
            self._no_vote("range_regime_transition_pending")
            return None
        getter = getattr(self._indicator_engine, "get_history", None)
        if not callable(getter):
            self._no_vote("range_futures_history_unavailable")
            return None
        try:
            # Up to 375 1m bars, with a small cushion for late provisional ticks.
            bars = getter(future, count=400, field="bars")
        except (TypeError, ValueError, AttributeError, KeyError):
            self._no_vote("range_futures_history_unavailable")
            return None
        structural = assess_range_vwap_reclaim(
            bars, decision_at=decision_at, config=self._cfg
        )
        if structural.get("reason") in {
            "session_open_anchor_missing",
            "insufficient_verified_futures_history",
            "futures_history_gap",
        }:
            restored, recovery_reason = _restore_authenticated_session_bars(
                bars,
                indicators.get("range_local_history_recovery"),
                future=future,
                decision_at=decision_at,
            )
            if recovery_reason == "authenticated_local_session_recovery":
                recovery = assess_range_vwap_reclaim(
                    restored, decision_at=decision_at, config=self._cfg
                )
                if recovery.get("reason") != "conflicting_futures_bar":
                    structural = recovery
                    structural["history_recovery"] = recovery_reason
            else:
                structural["history_recovery"] = recovery_reason
        self._observe_excursion(symbol, indicators, structural, decision_at=decision_at)
        if not self._load_persistent_guard(decision_at):
            self._no_vote("range_durable_setup_guard_unavailable")
            return None
        self._maybe_reset_excursions(future, structural)
        timestamp = str(structural.get("bar_timestamp") or "")
        if not timestamp or not self._regime_bar_confirmed(future, timestamp):
            self._no_vote("range_two_completed_regime_bars_required")
            return None
        if not structural.get("qualified"):
            self._no_vote(str(structural["reason"]))
            return None
        if structural.get("side") != side:
            self._no_vote("range_reversal_side_conflict")
            return None
        opportunity_key = self._active_opportunity
        if opportunity_key is not None and opportunity_key in self._opportunities:
            self._opportunities[opportunity_key]["structural_qualified"] = True

        if indicators.get("underlying_direction_state") == "TRANSITION":
            self._no_vote("range_underlying_transition")
            return None
        if indicators.get("direction_transition") is True:
            self._no_vote("range_spot_futures_direction_conflict")
            return None
        if indicators.get("stale_data_used") is True:
            self._no_vote("range_stale_data")
            return None
        age = _value(indicators.get("context_age_seconds"))
        quote_age = _value(indicators.get("data_age_seconds"))
        if age is None or not 0 <= age <= 2.0:
            self._no_vote("range_underlying_context_stale")
            return None
        if quote_age is None or not 0 <= quote_age <= 1.5:
            self._no_vote("range_option_quote_stale")
            return None

        selected = indicators.get("is_selected_option") is True
        near = _value(indicators.get("strike_distance_from_atm"))
        if not selected and not (
            near is not None and 0 <= near <= self._cfg.max_near_atm_distance_points
        ):
            self._no_vote("range_not_selected_or_near_atm")
            return None
        if indicators.get("tradable_quote") is not True:
            self._no_vote("range_quote_not_tradable")
            return None
        expiry_days = _value(indicators.get("days_to_expiry"))
        if expiry_days is None or expiry_days < self._cfg.minimum_days_to_expiry:
            self._no_vote("range_expiry_or_expiry_unverified")
            return None

        # A range reversal is independently defined by two completed futures
        # candles. The *older* directional classifier may oppose this side.
        # It still must not be in a transition/conflict state, and this
        # research-only context has no live manager override authority.
        fut_tick = _value(indicators.get("futures_ltp"))
        fut_close = _value(structural.get("underlying_entry_close"))
        fut_atr = _value(structural.get("futures_atr"))
        if (
            fut_tick is None
            or fut_close is None
            or fut_atr is None
            or abs(fut_tick - fut_close) > 0.50 * fut_atr
        ):
            self._no_vote("range_futures_quote_not_near_confirmed_bar")
            return None

        # The entry band is frozen at pre-confirmation bar close; the actual
        # trading TARGET is the independent exchange-reported session futures
        # VWAP from a fresh cached quote, never a rolling pseudo-VWAP.
        exchange_vwap = _value(indicators.get("futures_vwap"))
        calculated_vwap = _value(structural.get("session_vwap_before_confirmation"))
        if (
            exchange_vwap is None
            or calculated_vwap is None
            or indicators.get("futures_vwap_source") != "active_future_quote_vwap"
            or abs(exchange_vwap - calculated_vwap)
            > self._cfg.max_exchange_vwap_disagreement_atr * fut_atr
        ):
            self._no_vote("range_exchange_session_vwap_unverified_or_divergent")
            return None

        # The confirmed two-bar reversal is no longer tradeable if the
        # freshest futures quote has re-breached its frozen outer band, or
        # already reached the independent exchange VWAP target.
        lower = _value(structural.get("lower_band_before_confirmation"))
        upper = _value(structural.get("upper_band_before_confirmation"))
        if lower is None or upper is None:
            self._no_vote("range_frozen_outer_band_missing")
            return None
        if (side == "CE" and fut_tick <= lower) or (side == "PE" and fut_tick >= upper):
            self._no_vote("range_futures_rebroke_outer_band")
            return None
        if (side == "CE" and fut_tick >= exchange_vwap) or (
            side == "PE" and fut_tick <= exchange_vwap
        ):
            self._no_vote("range_futures_target_already_reached")
            return None

        # The option is a long-premium purchase for CE *and* PE. Require
        # positive option-book pressure; CE/PE signed futures directions must
        # not be incorrectly applied to the premium order book.
        proof = indicators.get("range_orderflow_confirmation")
        if not isinstance(proof, Mapping):
            self._no_vote("range_orderflow_handoff_missing")
            return None
        vote_stamp = _value(proof.get("vote_timestamp"))
        source_version = str(proof.get("quote_update_version") or "")
        expected_version = str(indicators.get("quote_update_version") or "")
        if (
            proof.get("candidate_symbol") != symbol
            or not source_version
            or not expected_version
            or source_version != expected_version
            or proof.get("quote_update_version_source") == "microstructure_fingerprint"
            or vote_stamp is None
            or not 0 <= decision_at.timestamp() - vote_stamp <= 2.0
            or proof.get("quote_readiness_allowed") is not True
            or proof.get("ofi_reversal_holdoff_evaluated") is not True
            or proof.get("range_research_holdoff_active") is not False
        ):
            self._no_vote("range_orderflow_quote_or_holdoff_unverified")
            return None
        ofi = _value(proof.get("ofi_3s_normalized"))
        ofi_1s = _value(proof.get("ofi_1s_normalized"))
        ofi_span = _value(proof.get("ofi_3s_span_seconds"))
        try:
            ofi_updates = int(proof.get("ofi_update_count_3s") or 0)
        except (TypeError, ValueError, OverflowError):
            ofi_updates = 0
        depth_imbalance = _value(proof.get("depth_imbalance"))
        if depth_imbalance is not None and depth_imbalance <= -0.15:
            self._no_vote("range_adverse_option_book_depth")
            return None
        if not (
            proof.get("ofi_ready") is True
            and proof.get("ofi_3s_complete") is True
            and ofi_span is not None
            and ofi_span >= 1.5
            and ofi_updates >= 3
            and ofi is not None
            and ofi >= 0.10
            and ofi_1s is not None
            and ofi_1s >= 0.0
        ):
            self._no_vote("range_independent_option_flow_unconfirmed")
            return None

        conditional_center = structural.get("conditional_center_exception_used") is True
        if conditional_center and (
            ofi is None or ofi < self._cfg.min_conditional_option_ofi_3s
        ):
            self._no_vote("range_conditional_option_flow_insufficient")
            return None

        stamp = decision_at.date().isoformat()
        excursion_key = (future, side, stamp)
        if excursion_key in self._consumed_excursions:
            self._no_vote("range_same_excursion_consumed")
            return None

        greeks = indicators.get("greeks")
        greek_map = greeks if isinstance(greeks, Mapping) else {}
        delta = indicators.get("delta")
        theta = indicators.get("theta")
        if delta is None:
            delta = (
                greek_map.get("delta")
                if isinstance(greeks, Mapping)
                else getattr(greeks, "delta", None)
            )
        if theta is None:
            theta = (
                greek_map.get("theta")
                if isinstance(greeks, Mapping)
                else getattr(greeks, "theta", None)
            )
        quote = {
            "bid": indicators.get("bid"),
            "ask": indicators.get("ask"),
            "atr": indicators.get("atr"),
            "lot_size": indicators.get("lot_size"),
            "delta": delta,
            "theta": theta,
            "theta_period": indicators.get("theta_period")
            or greek_map.get("theta_period"),
            "delta_underlying": indicators.get("delta_underlying")
            or greek_map.get("delta_underlying"),
            "gamma": indicators.get("gamma") or greek_map.get("gamma"),
            "gamma_underlying": indicators.get("gamma_underlying")
            or greek_map.get("gamma_underlying"),
            "vega_per_iv_point": indicators.get("vega_per_iv_point")
            or greek_map.get("vega_per_iv_point"),
            "vega_convention": indicators.get("vega_convention")
            or greek_map.get("vega_convention"),
            "spot_ltp": indicators.get("range_spot_ltp"),
            "futures_ltp": indicators.get("futures_ltp"),
            "futures_atr": structural.get("futures_atr"),
        }
        entry = _value(structural.get("underlying_entry_close"))
        target = exchange_vwap
        invalidation = _value(structural.get("invalidation_futures"))
        if entry is None or target is None or invalidation is None:
            self._no_vote("range_structural_prices_unavailable")
            return None
        if (side == "CE" and not invalidation < entry < target) or (
            side == "PE" and not target < entry < invalidation
        ):
            self._no_vote("range_structural_stop_target_order_invalid")
            return None
        # Rank economic feasibility against the freshest verified futures
        # quote, not the older confirmation-bar close. Never CREATE extra
        # expected reward from an adverse post-confirmation retracement.
        original_distance = abs(target - entry)
        quote_distance = abs(target - fut_tick)
        remaining_distance = min(original_distance, quote_distance)
        # The rolling center can obstruct price before session VWAP even
        # when the center gap is within the normal 1.0 ATR allowance.
        # Price the nearer barrier for both entry routes, not only for the
        # stricter conditional exception. A center already passed cannot
        # become a new forward target or manufacture additional reward.
        local_barrier_points: float | None = None
        local_center = _value(structural.get("rolling_20m_secondary_center"))
        if local_center is None:
            self._no_vote("range_local_center_unverified")
            return None
        local_center_ahead = (side == "CE" and fut_tick < local_center) or (
            side == "PE" and fut_tick > local_center
        )
        if conditional_center and not local_center_ahead:
            self._no_vote("range_conditional_local_center_already_reached")
            return None
        if local_center_ahead:
            local_barrier_points = abs(local_center - fut_tick)
            remaining_distance = min(remaining_distance, local_barrier_points)
        stop_distance = max(abs(entry - invalidation), abs(fut_tick - invalidation))
        budget = assess_option_cost_risk(
            quote,
            config=self._cfg,
            side=side,
            futures_target_distance=remaining_distance,
            futures_stop_distance=stop_distance,
        )
        budget["original_completed_bar_target_points"] = round(original_distance, 4)
        budget["fresh_futures_target_points"] = round(quote_distance, 4)
        budget["target_points_removed_due_to_quote_drift"] = round(
            original_distance - min(original_distance, quote_distance), 4
        )
        budget["conditional_center_exception"] = conditional_center
        budget["first_local_center_barrier_points"] = (
            round(local_barrier_points, 4) if local_barrier_points is not None else None
        )
        budget["local_center_binds_modeled_payoff"] = (
            local_barrier_points is not None
            and local_barrier_points < min(original_distance, quote_distance)
        )
        budget["modeled_futures_reward_points"] = round(remaining_distance, 4)
        # Optional economic priority evidence uses the existing canonical
        # cost and spread stressed budget. It must not supersede the selector.
        model_reward = _value(budget.get("target_net_reward_after_costs"))
        model_risk = _value(budget.get("worst_stop_loss_including_costs"))
        budget["net_cash_surplus_above_1_5r"] = (
            round(model_reward - 1.5 * model_risk, 2)
            if model_reward is not None and model_risk is not None
            else None
        )
        if not budget["qualified"]:
            self._no_vote(str(budget["reason"]))
            return None
        conditional_min_rr = max(
            self._cfg.min_net_reward_risk,
            self._cfg.min_conditional_net_reward_risk,
            1.75,
        )
        if conditional_center and (
            _value(budget.get("net_reward_risk")) is None
            or budget["net_reward_risk"] < conditional_min_rr
        ):
            self._no_vote("range_conditional_payoff_buffer_insufficient")
            return None
        live_pilot = range_vwap_live_enabled()
        if live_pilot and (
            budget["worst_stop_loss_including_costs"] > 773.0
            or budget["net_reward_risk"] < 1.5
        ):
            self._no_vote("range_live_cash_risk_or_net_rr")
            return None
        active_opportunity = self._active_opportunity
        if active_opportunity is not None and active_opportunity in self._opportunities:
            row = self._opportunities[active_opportunity]
            row["status"] = "cost_eligible"
            row["modeled_cost_eligible"] = True
            row["option_cost_risk"] = budget
            self._record_excursion_decision("cost_eligible")

        setup_id = (
            f"range_session_vwap:{future}:{side}:"
            f"{structural['previous_excursion_timestamp']}"
        )
        if setup_id in self._seen:
            self._no_vote("range_setup_already_observed")
            return None
        LOGGER.info(
            "RANGE_SESSION_VWAP_CANDIDATE symbol=%s setup_id=%s "
            "primary_vwap=%s rolling_center=%s net_rr=%s cash_risk=%s "
            "direction_proof=completed_futures_reclaim live_pilot=%s",
            symbol,
            setup_id,
            structural["session_vwap_before_confirmation"],
            structural["rolling_20m_secondary_center"],
            budget["net_reward_risk"],
            budget["worst_stop_loss_including_costs"],
            live_pilot,
        )
        if (
            str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper() == "LIVE"
            and not live_pilot
        ):
            self._no_vote("range_live_pilot_not_armed")
            return None
        stop_model = budget["stop_price_model"]
        target_model = budget["vwap_limited_target_price_model"]
        if live_pilot and not stop_model < current_price < target_model:
            self._no_vote("range_live_stop_target_vs_latest_price_invalid")
            return None
        # Reserve only an actually emitted vote. The manager/runner already
        # own broker deduplication; callbacks release proven pre-broker rejects.
        # Atomic reserve across simultaneous eligible CE/PE contract evaluations.
        # The broker/idempotency system remains the final order authority.
        with self._guard_lock:
            if setup_id in self._seen or excursion_key in self._consumed_excursions:
                self._no_vote("range_setup_reserved_concurrently")
                return None
            self._seen.add(setup_id)
            self._consumed_excursions.add(excursion_key)
            self._pending_excursions[setup_id] = excursion_key
            if not self._persist_guard():
                self._no_vote("range_durable_setup_reservation_failed")
                return None
        active_opportunity = self._active_opportunity
        if active_opportunity is not None and active_opportunity in self._opportunities:
            self._opportunities[active_opportunity]["setup_id"] = setup_id
        if not live_pilot:
            if len(self._seen) > 512:
                self._seen = {setup_id}
            if len(self._consumed_excursions) > 128:
                self._consumed_excursions = {excursion_key}
            if len(self._pending_excursions) > 512:
                self._pending_excursions.pop(next(iter(self._pending_excursions)))
        self._record_excursion_decision("signal_emitted")
        return EliteSignal(
            symbol=symbol,
            signal="BUY",
            confidence=1.0,
            entry_price=current_price,
            stop_loss=stop_model if live_pilot else None,
            target=target_model if live_pilot else None,
            quantity=1,
            strategy_name="RangeVWAPReclaim",
            metadata={
                "role": "trigger" if live_pilot else "context",
                "can_trigger": live_pilot,
                "setup_pass": True,
                "setup_id": setup_id,
                "setup_name": (
                    "session_vwap_reentry_live"
                    if live_pilot
                    else "session_vwap_reentry_shadow"
                ),
                "side": side,
                "trade_side": side,
                "contract_side": side,
                "source_domain": "futures_completed_bars",
                "research_only": not live_pilot,
                "range_option_flow_verified": True,
                # These are absolute option-premium price levels, not offsets.
                "bracket_anchor_mode": "absolute_level",
                "reversal_direction_source": "two_completed_futures_candles",
                "prior_trend_bias": indicators.get("underlying_direction_bias"),
                "requires_runner_execution_validation": True,
                # Reuse the canonical bracket time-stop machinery; never add
                # a second position manager for this strategy.
                "range_option_hold_minutes": self._cfg.assumed_hold_minutes,
                "range_option_model_source": indicators.get(
                    "range_option_model_source"
                ),
                "range_option_contract_expiry": indicators.get(
                    "range_option_contract_expiry"
                ),
                "range_futures_contract": future,
                "range_futures_side": side,
                "range_futures_target": target,
                "range_futures_invalidation": invalidation,
                "range_structure": structural,
                "exchange_session_vwap_at_decision": exchange_vwap,
                "option_cost_risk": budget,
                "range_failed_breakout_quality": structural.get(
                    "failed_breakout_quality"
                ),
                "range_conditional_center_exception": conditional_center,
                "range_economic_surplus_rupees": budget.get(
                    "net_cash_surplus_above_1_5r"
                ),
                "research_opportunity_id": opportunity_key,
                "setup_reasons": ["closed_session_vwap_band_reentry"],
                "required_data_present": True,
                "stale_data_used": False,
            },
        )
