"""Purpose: Research-only NIFTY futures SESSION-VWAP reclaim elite strategy.

Responsibilities: Verify all completed minute bars from 09:15, compute a
pre-confirmation session-VWAP outer band and a secondary 20-minute equilibrium,
then confirm a fully completed band reentry without look-ahead. The public
assess_option_cost_risk interface now consumes actual futures target/stop
distances and verified option delta to project conservative premium reward.
RangeVWAPReclaimStrategy owns distinct-bar regime confirmation and one
candidate per excursion, not any broker execution authority.
Safety: LIVE always produces no vote; no reversal may override the shared
underlying-direction, broker-stop or capital authorization contracts.
"""

from __future__ import annotations

import math
import os
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

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
    evidence: dict[str, Any] = {
        "qualified": False,
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
        "range_width_atr": round(width, 4),
        "volume_ratio": round(rvol, 4),
        "underlying_latest_close": float(latest["close"]),
    }
    if slope_bps > config.max_vwap_slope_bps_per_min:
        return {**evidence, "reason": "session_vwap_sloping"}
    if efficiency > config.max_directional_efficiency:
        return {**evidence, "reason": "directional_trend_not_range"}
    if not 1.5 <= width <= config.max_range_width_atr:
        return {**evidence, "reason": "range_width_unproven"}
    if midpoint_gap_atr > config.max_secondary_center_deviation_atr:
        return {**evidence, "reason": "session_rolling_center_disagree"}
    if rvol < config.min_reclaim_volume_ratio:
        return {**evidence, "reason": "reclaim_volume_weak"}
    if bar_range <= 0 or bar_range > 2.0 * atr:
        return {**evidence, "reason": "impulsive_or_invalid_reclaim_bar"}
    if body_fraction < config.min_reclaim_body_fraction:
        return {**evidence, "reason": "weak_reclaim_body"}

    reentry = max(config.min_band_reentry_atr * atr, 0.05 * band)
    remaining = config.min_remaining_center_gap_atr * atr
    last_close = float(latest["close"])
    bullish = bool(
        float(previous["low"]) < lower
        and float(previous["close"]) < lower
        and last_close >= lower + reentry
        and float(latest["low"]) >= float(previous["low"])
        and last_close > float(latest["open"])
        and last_close > float(previous["close"])
        and last_close + remaining < min(session_vwap, rolling_center)
    )
    bearish = bool(
        float(previous["high"]) > upper
        and float(previous["close"]) > upper
        and last_close <= upper - reentry
        and float(latest["high"]) <= float(previous["high"])
        and last_close < float(latest["open"])
        and last_close < float(previous["close"])
        and last_close - remaining > max(session_vwap, rolling_center)
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
    }


def assess_option_cost_risk(
    option_quote: Mapping[str, Any],
    *,
    config: RangeVWAPReclaimStrategyConfig,
    side: str | None = None,
    futures_target_distance: float | None = None,
    futures_stop_distance: float | None = None,
) -> dict[str, Any]:
    """Model reachable session-VWAP premium reward, not arbitrary option TP.

    Expected premium appreciation uses OBSERVED delta × remaining futures
    movement with an adverse haircut and theta decay. No gamma windfall is
    assumed. Actual future fill prices and option Greeks can change.
    """
    bid, ask, atr, lot = (
        _value(option_quote.get("bid")),
        _value(option_quote.get("ask")),
        _value(option_quote.get("atr")),
        _value(option_quote.get("lot_size")),
    )
    if lot is None or lot < 1 or lot > 1000 or not lot.is_integer():
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
    spread_pct = (ask - bid) / ask * 100.0
    if spread_pct > config.max_option_spread_pct:
        return {"qualified": False, "reason": "option_spread_too_wide"}

    theta = _value(option_quote.get("theta"))
    if theta is None:
        # No free assumption of zero theta: adverse haircut when unobserved.
        theta_loss = ask * config.unknown_theta_haircut_pct / 100
        theta_source = "unknown_theta_premium_haircut"
    else:
        theta_loss = max(0.0, -theta) * config.assumed_hold_minutes / 390.0
        theta_source = "observed_theta_daily_390minute_stress"
    reachable_premium_points = max(
        0.0,
        abs(delta) * distance * config.target_delta_haircut - theta_loss,
    )
    premium_stop_points = max(
        0.75 * atr,
        0.035 * ask,
        abs(delta) * invalidation_gap * 1.10,
    )
    if premium_stop_points >= ask:
        return {"qualified": False, "reason": "option_stop_geometry_invalid"}
    if reachable_premium_points <= 0:
        return {"qualified": False, "reason": "vwap_target_not_reachable_after_decay"}
    # Gross reward is capped by both the actual futures session-VWAP
    # distance and the existing conservative research target multiple.
    target_gain = min(
        reachable_premium_points,
        config.gross_target_r * premium_stop_points,
    )
    slippage = config.slippage_stress_bps / 10_000.0
    paid = ask * (1 + slippage)
    spread_points = ask - bid
    stop = (ask - premium_stop_points) * (1 - slippage) - spread_points
    target = (ask + target_gain) * (1 - slippage) - spread_points
    if stop <= 0 or target <= paid:
        return {"qualified": False, "reason": "liquidation_price_invalid"}
    stop_cost = estimate_round_trip_cost(
        entry_price=paid, exit_price=stop, quantity=quantity
    ).total
    target_cost = estimate_round_trip_cost(
        entry_price=paid, exit_price=target, quantity=quantity
    ).total
    loss = (paid - stop) * quantity + stop_cost
    reward = (target - paid) * quantity - target_cost
    net_rr = reward / loss if loss > 0 else 0.0
    report = {
        "qualified": False,
        "model_only": True,
        "observed_contract_lot_size": quantity,
        "observed_delta": round(delta, 5),
        "futures_session_target_points": round(distance, 4),
        "futures_structural_stop_points": round(invalidation_gap, 4),
        "assumed_hold_minutes": config.assumed_hold_minutes,
        "theta_cost_source": theta_source,
        "estimated_theta_cost_points": round(theta_loss, 4),
        "projected_premium_gain_points": round(reachable_premium_points, 4),
        "slippage_each_side_bps": config.slippage_stress_bps,
        "modeled_extra_exit_spread_points": round(spread_points, 4),
        "premium_stop_points": round(premium_stop_points, 4),
        "stop_price_model": round(stop, 4),
        "vwap_limited_target_price_model": round(target, 4),
        "worst_stop_loss_including_costs": round(loss, 2),
        "target_net_reward_after_costs": round(reward, 2),
        "net_reward_risk": round(net_rr, 4),
    }
    if loss > config.max_trade_cash_risk:
        return {**report, "reason": "cash_risk_over_cap"}
    if net_rr < config.min_net_reward_risk:
        return {**report, "reason": "after_cost_reward_risk_low"}
    return {**report, "qualified": True, "reason": "modeled_session_target_ok"}


class RangeVWAPReclaimStrategy(EliteStrategy):  # type: ignore[misc]
    """Research-only session-VWAP reversal context, never a LIVE trigger."""

    def __init__(
        self, config: RangeVWAPReclaimStrategyConfig, indicator_engine: Any
    ) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config
        self._seen: set[str] = set()
        self._regime_bars: dict[str, tuple[str, int]] = {}
        self._consumed_excursions: set[tuple[str, str, str]] = set()

    def get_required_indicators(self) -> list[str]:
        return []

    def _regime_bar_confirmed(self, future: str, timestamp: str) -> bool:
        """Require contiguous, forward-moving completed candle observations."""
        current = _time(timestamp)
        if current is None:
            return False
        previous, n = self._regime_bars.get(future, ("", 0))
        earlier = _time(previous)
        if current == earlier:
            return n >= self._cfg.min_regime_completed_bars
        if earlier is None or current - earlier != timedelta(minutes=1):
            n = 1
        else:
            n = min(self._cfg.min_regime_completed_bars, n + 1)
        self._regime_bars[future] = timestamp, n
        return n >= self._cfg.min_regime_completed_bars

    def _maybe_reset_excursions(self, future: str, evidence: Mapping[str, Any]) -> None:
        """Rearm only after futures actually return near session VWAP."""
        last = _value(evidence.get("underlying_latest_close"))
        center = _value(evidence.get("session_vwap_before_confirmation"))
        atr = _value(evidence.get("futures_atr"))
        if last is None or center is None or atr is None or atr <= 0:
            return
        if abs(last - center) > 0.25 * atr:
            return
        self._consumed_excursions = {
            key for key in self._consumed_excursions if key[0] != future
        }

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: dict[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
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

        # The option is a long-premium purchase for CE *and* PE. Require
        # positive option-book pressure; CE/PE signed futures directions must
        # not be incorrectly applied to the premium order book.
        ofi = _value(indicators.get("ofi_3s_normalized"))
        ofi_1s = _value(indicators.get("ofi_1s_normalized"))
        ofi_span = _value(indicators.get("ofi_3s_span_seconds"))
        try:
            ofi_updates = int(indicators.get("ofi_update_count_3s") or 0)
        except (TypeError, ValueError, OverflowError):
            ofi_updates = 0
        if not (
            indicators.get("ofi_ready") is True
            and indicators.get("ofi_3s_complete") is True
            and ofi_span is not None
            and ofi_span >= 1.5
            and ofi_updates >= 3
            and ofi is not None
            and ofi >= 0.10
            and ofi_1s is not None
            and ofi_1s >= 0.0
            and indicators.get("ofi_reversal_holdoff_active") is not True
        ):
            self._no_vote("range_independent_option_flow_unconfirmed")
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
        quote = {
            "bid": indicators.get("bid"),
            "ask": indicators.get("ask"),
            "atr": indicators.get("atr"),
            "lot_size": indicators.get("lot_size"),
            "delta": delta if delta is not None else greek_map.get("delta"),
            "theta": theta if theta is not None else greek_map.get("theta"),
        }
        entry = _value(structural.get("underlying_entry_close"))
        target = _value(structural.get("session_vwap_before_confirmation"))
        invalidation = _value(structural.get("invalidation_futures"))
        if entry is None or target is None or invalidation is None:
            self._no_vote("range_structural_prices_unavailable")
            return None
        if (side == "CE" and not invalidation < entry < target) or (
            side == "PE" and not target < entry < invalidation
        ):
            self._no_vote("range_structural_stop_target_order_invalid")
            return None
        budget = assess_option_cost_risk(
            quote,
            config=self._cfg,
            side=side,
            futures_target_distance=abs(target - entry),
            futures_stop_distance=abs(entry - invalidation),
        )
        if not budget["qualified"]:
            self._no_vote(str(budget["reason"]))
            return None

        setup_id = (
            f"range_session_vwap:{future}:{side}:"
            f"{structural['previous_excursion_timestamp']}"
        )
        if setup_id in self._seen:
            self._no_vote("range_setup_already_observed")
            return None
        self._seen.add(setup_id)
        self._consumed_excursions.add(excursion_key)
        if len(self._seen) > 512:
            self._seen = {setup_id}
        if len(self._consumed_excursions) > 128:
            self._consumed_excursions = {excursion_key}
        LOGGER.info(
            "RANGE_SESSION_VWAP_RESEARCH_CANDIDATE symbol=%s setup_id=%s "
            "primary_vwap=%s rolling_center=%s net_rr=%s cash_risk=%s "
            "direction_proof=completed_futures_reclaim policy=context_only",
            symbol,
            setup_id,
            structural["session_vwap_before_confirmation"],
            structural["rolling_20m_secondary_center"],
            budget["net_reward_risk"],
            budget["worst_stop_loss_including_costs"],
        )
        if str(os.getenv("EXECUTION_MODE", "SHADOW")).upper() == "LIVE":
            self._no_vote("range_research_only_no_live_trigger")
            return None
        return EliteSignal(
            symbol=symbol,
            signal="BUY",
            confidence=1.0,
            entry_price=current_price,
            stop_loss=None,
            target=None,
            quantity=1,
            strategy_name="RangeVWAPReclaim",
            metadata={
                "role": "context",
                "can_trigger": False,
                "setup_pass": True,
                "setup_id": setup_id,
                "setup_name": "session_vwap_reentry_shadow",
                "side": side,
                "trade_side": side,
                "contract_side": side,
                "source_domain": "futures_completed_bars",
                "research_only": True,
                "reversal_direction_source": "two_completed_futures_candles",
                "prior_trend_bias": indicators.get("underlying_direction_bias"),
                "requires_runner_execution_validation": True,
                "range_structure": structural,
                "option_cost_risk": budget,
                "setup_reasons": ["closed_session_vwap_band_reentry"],
                "required_data_present": True,
                "stale_data_used": False,
            },
        )
