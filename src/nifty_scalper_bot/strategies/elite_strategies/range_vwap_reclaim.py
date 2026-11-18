"""Purpose: Research-only NIFTY futures range / VWAP reclaim elite strategy.

Responsibilities: Read canonical completed 1m futures bars, require a genuinely
two-sided range and an observed turning candle, then estimate option-side
transaction costs and adverse risk without submitting a broker order.
Safety: LIVE always produces no vote; no reversal may override the existing
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
    return sum(
        ((float(item["high"]) + float(item["low"]) + float(item["close"])) / 3)
        * float(item["volume"])
        for item in rows
    ) / total


def _verified_completed_bars(
    observed: Any, *, decision_at: datetime, min_history: int
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(observed, Sequence) or isinstance(observed, (str, bytes)):
        return [], "missing_completed_futures_bars"
    decoded: dict[datetime, dict[str, Any]] = {}
    for raw in observed:
        if not isinstance(raw, Mapping):
            return [], "invalid_futures_bar"
        if raw.get("is_provisional") is True or raw.get("is_complete") is not True:
            continue
        stamp = _time(
            raw.get("timestamp") or raw.get("datetime") or raw.get("time")
        )
        numbers = {
            key: _value(raw.get(key))
            for key in ("open", "high", "low", "close", "volume")
        }
        if stamp is None or any(value is None for value in numbers.values()):
            return [], "invalid_futures_bar"
        assert all(value is not None for value in numbers.values())
        bar = {key: float(value) for key, value in numbers.items()}
        if (
            min(bar["open"], bar["high"], bar["low"], bar["close"]) <= 0
            or bar["volume"] <= 0
            or not (
                bar["low"] <= bar["open"] <= bar["high"]
                and bar["low"] <= bar["close"] <= bar["high"]
            )
        ):
            return [], "invalid_futures_bar"
        if stamp.date() != decision_at.date():
            continue
        if not (9 * 60 + 15 <= stamp.hour * 60 + stamp.minute < 15 * 60 + 30):
            continue
        if stamp + timedelta(minutes=1) > decision_at:
            continue
        existing = decoded.get(stamp)
        if existing is not None and existing != bar:
            return [], "conflicting_futures_bar"
        decoded[stamp] = bar
    selected = sorted(decoded.items())[-min_history:]
    if len(selected) < min_history:
        return [], "insufficient_verified_futures_history"
    most_recent = selected[-1][0] + timedelta(minutes=1)
    if not 0 <= (decision_at - most_recent).total_seconds() <= 90.0:
        return [], "futures_completed_bar_stale"
    for left, right in zip(selected, selected[1:]):
        if right[0] - left[0] != timedelta(minutes=1):
            return [], "futures_history_gap"
    return [
        {"timestamp": stamp.isoformat(), **bar}
        for stamp, bar in selected
    ], "verified"


def assess_range_vwap_reclaim(
    bars: Any,
    *,
    decision_at: datetime,
    config: RangeVWAPReclaimStrategyConfig,
) -> dict[str, Any]:
    """Causal completed-bar structural proof, separate from option execution."""
    rows, reason = _verified_completed_bars(
        bars, decision_at=decision_at, min_history=config.min_history_bars
    )
    if reason != "verified":
        return {"qualified": False, "reason": reason}
    previous, latest = rows[-2], rows[-1]
    prices = [float(r["close"]) for r in rows]
    ranges = [
        max(
            float(row["high"]) - float(row["low"]),
            abs(float(row["high"]) - float(rows[i - 1]["close"])),
            abs(float(row["low"]) - float(rows[i - 1]["close"])),
        )
        for i, row in enumerate(rows)
        if i > 0
    ]
    atr = statistics.mean(ranges[-14:])
    if atr <= 0:
        return {"qualified": False, "reason": "futures_atr_unavailable"}
    center = _weighted_center(rows[-20:])
    prior_center = _weighted_center(rows[-21:-1])
    older_center = _weighted_center(rows[-25:-5])
    if center is None or prior_center is None or older_center is None:
        return {"qualified": False, "reason": "futures_vwap_unavailable"}
    closes = prices[-21:-1]
    sigma = statistics.pstdev(closes)
    band = max(config.band_atr_multiple * atr, config.band_sigma_multiple * sigma)
    slope_bps = abs(center - older_center) / center * 10_000 / 5
    path = prices[-26:]
    travel = sum(abs(b - a) for a, b in zip(path, path[1:]))
    efficiency = abs(path[-1] - path[0]) / travel if travel > 0 else 1.0
    width_atr = (
        max(float(r["high"]) for r in rows[-30:])
        - min(float(r["low"]) for r in rows[-30:])
    ) / atr
    volume_baseline = statistics.mean(float(r["volume"]) for r in rows[-21:-1])
    rvol = float(latest["volume"]) / volume_baseline
    evidence = {
        "qualified": False,
        "source": "verified_completed_futures_1m",
        "bar_timestamp": latest["timestamp"],
        "futures_vwap": round(center, 4),
        "futures_atr": round(atr, 4),
        "futures_sigma": round(sigma, 4),
        "band_points": round(band, 4),
        "vwap_slope_bps_per_min": round(slope_bps, 4),
        "directional_efficiency": round(efficiency, 4),
        "range_width_atr": round(width_atr, 4),
        "volume_ratio": round(rvol, 4),
    }
    if slope_bps > config.max_vwap_slope_bps_per_min:
        return {**evidence, "reason": "futures_vwap_sloping"}
    if efficiency > config.max_directional_efficiency:
        return {**evidence, "reason": "directional_trend_not_range"}
    if not 1.5 <= width_atr <= config.max_range_width_atr:
        return {**evidence, "reason": "range_width_unproven"}
    if rvol < config.min_reclaim_volume_ratio:
        return {**evidence, "reason": "reclaim_volume_weak"}
    if float(latest["high"]) - float(latest["low"]) > 2.0 * atr:
        return {**evidence, "reason": "impulsive_bar_not_range"}
    lower = prior_center - band
    upper = prior_center + band
    gap = config.min_remaining_center_gap_atr * atr
    bullish = bool(
        float(previous["low"]) <= lower
        and float(latest["close"]) > float(previous["close"])
        and float(latest["close"]) > float(latest["open"])
        and float(latest["low"]) >= float(previous["low"])
        and float(latest["close"]) < center - gap
    )
    bearish = bool(
        float(previous["high"]) >= upper
        and float(latest["close"]) < float(previous["close"])
        and float(latest["close"]) < float(latest["open"])
        and float(latest["high"]) <= float(previous["high"])
        and float(latest["close"]) > center + gap
    )
    if bullish == bearish:
        return {**evidence, "reason": "no_completed_bar_reclaim"}
    return {
        **evidence,
        "qualified": True,
        "side": "CE" if bullish else "PE",
        "reason": "completed_bar_range_reclaim",
        "invalidation_futures": (
            float(previous["low"]) if bullish else float(previous["high"])
        ),
    }


def assess_option_cost_risk(
    option_quote: Mapping[str, Any],
    *,
    config: RangeVWAPReclaimStrategyConfig,
) -> dict[str, Any]:
    """Test the modeled option payoff; this is not a broker fill forecast."""
    bid, ask, atr, lot = (
        _value(option_quote.get("bid")),
        _value(option_quote.get("ask")),
        _value(option_quote.get("atr")),
        _value(option_quote.get("lot_size")),
    )
    if lot is None or lot < 1 or lot > 1000 or not lot.is_integer():
        return {"qualified": False, "reason": "historical_lot_size_unverified"}
    quantity = int(lot)
    if (
        bid is None or ask is None or atr is None or bid <= 0
        or ask <= bid or atr <= 0
    ):
        return {"qualified": False, "reason": "option_quote_or_atr_missing"}
    spread_pct = (ask - bid) / ask * 100
    if spread_pct > config.max_option_spread_pct:
        return {"qualified": False, "reason": "option_spread_too_wide"}
    stop_points = max(0.75 * atr, 0.035 * ask)
    if stop_points >= ask:
        return {"qualified": False, "reason": "option_stop_geometry_invalid"}
    slippage = config.slippage_stress_bps / 10_000
    paid = ask * (1 + slippage)
    # A long option is bought at the ask, but must be liquidated at the bid.
    # Include the *observed full entry spread* again at the exit as a
    # conservative stress proxy, on top of adverse slippage and exchange fees.
    # This is not a prediction of future spread or guaranteed stop liquidity.
    spread_points = ask - bid
    stop = (ask - stop_points) * (1 - slippage) - spread_points
    target = (
        (ask + config.gross_target_r * stop_points) * (1 - slippage)
        - spread_points
    )
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
        "slippage_each_side_bps": config.slippage_stress_bps,
        "modeled_extra_exit_spread_points": round(spread_points, 4),
        "premium_stop_points": round(stop_points, 4),
        "stop_price_model": round(stop, 4),
        "target_price_model": round(target, 4),
        "worst_stop_loss_including_costs": round(loss, 2),
        "target_net_reward_after_costs": round(reward, 2),
        "net_reward_risk": round(net_rr, 4),
    }
    if loss > config.max_trade_cash_risk:
        return {**report, "reason": "cash_risk_over_cap"}
    if net_rr < config.min_net_reward_risk:
        return {**report, "reason": "after_cost_reward_risk_low"}
    return {**report, "qualified": True, "reason": "modeled_geometry_ok"}


class RangeVWAPReclaimStrategy(EliteStrategy):
    """One elite STRATEGY CATALOG entry; never a live order trigger."""

    def __init__(
        self, config: RangeVWAPReclaimStrategyConfig, indicator_engine: Any
    ) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config
        self._seen: set[str] = set()

    def get_required_indicators(self) -> list[str]:
        return []

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: dict[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
        del position
        if canonical_nifty_option_symbol(symbol) is None:
            self._no_vote("not_nifty_option")
            return None
        side = symbol[-2:].upper()
        if str(indicators.get("regime") or "").lower() != "range":
            self._no_vote("not_verified_range_regime")
            return None
        if indicators.get("stale_data_used") is True:
            self._no_vote("range_stale_data")
            return None
        if str(indicators.get("underlying_direction_bias") or "").upper() != side:
            self._no_vote("range_underlying_direction_unconfirmed")
            return None
        if indicators.get("underlying_direction_state") == "TRANSITION":
            self._no_vote("range_direction_transition")
            return None
        if indicators.get("is_selected_option") is not True:
            self._no_vote("range_option_not_selected")
            return None
        if indicators.get("tradable_quote") is not True:
            self._no_vote("range_quote_not_tradable")
            return None
        age = _value(indicators.get("context_age_seconds"))
        quote_age = _value(indicators.get("data_age_seconds"))
        if age is None or not 0 <= age <= 2.0:
            self._no_vote("range_underlying_context_stale")
            return None
        if quote_age is None or not 0 <= quote_age <= 1.5:
            self._no_vote("range_option_quote_stale")
            return None
        future = canonical_nifty_future_symbol(
            indicators.get("active_futures_symbol")
        )
        getter = getattr(self._indicator_engine, "get_history", None)
        if future is None or not callable(getter):
            self._no_vote("range_futures_history_unavailable")
            return None
        try:
            bars = getter(future, count=50, field="bars")
        except (TypeError, ValueError, AttributeError, KeyError):
            self._no_vote("range_futures_history_unavailable")
            return None
        decision_at = _time(indicators.get("decision_at")) or datetime.now(_IST)
        structural = assess_range_vwap_reclaim(
            bars, decision_at=decision_at, config=self._cfg
        )
        if not structural.get("qualified"):
            self._no_vote(str(structural["reason"]))
            return None
        if structural.get("side") != side:
            self._no_vote("range_side_conflict")
            return None
        quote = {
            "bid": indicators.get("bid"),
            "ask": indicators.get("ask"),
            "atr": indicators.get("atr"),
            "lot_size": indicators.get("lot_size"),
        }
        budget = assess_option_cost_risk(quote, config=self._cfg)
        if not budget["qualified"]:
            self._no_vote(str(budget["reason"]))
            return None
        setup_id = (
            f"range_vwap:{future}:{side}:{structural['bar_timestamp']}"
        )
        if setup_id in self._seen:
            self._no_vote("range_setup_already_observed")
            return None
        self._seen.add(setup_id)
        if len(self._seen) > 512:
            self._seen.clear()
            self._seen.add(setup_id)
        LOGGER.info(
            "RANGE_VWAP_RESEARCH_CANDIDATE symbol=%s setup_id=%s "
            "regime=range net_rr=%s risk_cash=%s policy=shadow_only",
            symbol,
            setup_id,
            budget["net_reward_risk"],
            budget["worst_stop_loss_including_costs"],
        )
        # LIVE must never return a vote: no production strategy/arbitration
        # path has been authorized to treat a reversal as an execution trigger.
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
                "setup_name": "range_vwap_reclaim_shadow",
                "side": side,
                "trade_side": side,
                "contract_side": side,
                "source_domain": "futures_completed_bars",
                "research_only": True,
                "quality_probability": None,
                "requires_runner_execution_validation": True,
                "range_structure": structural,
                "option_cost_risk": budget,
                "setup_reasons": ["confirmed_futures_range_reclaim"],
                "required_data_present": True,
                "stale_data_used": False,
            },
        )
