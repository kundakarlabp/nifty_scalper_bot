"""Non-live directional hypothesis candidates for archived NIFTY research.

These deliberately do not implement EliteStrategy or enter the production catalog.
Signals are generated after completed futures bars and consumed only by the
existing archived option execution/cost simulator. Research proxy stops and targets
are not production entry/exit recommendations.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median, pstdev
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from nifty_scalper_bot.config.settings import get_settings
from nifty_scalper_bot.strategies.elite_strategies.orb_pro import ORBProStrategy
from nifty_scalper_bot.strategies.indicators import IndicatorEngine

_FAMILIES = {
    "trend_pullback": "PullbackResearch",
    "compression_breakout": "CompressionResearch",
    "momentum_acceleration": "MomentumResearch",
}


def _ema(values: Sequence[float], period: int) -> float:
    """Deterministic past-only exponential average for isolated research."""
    alpha = 2.0 / (period + 1.0)
    level = values[0]
    for value in values[1:]:
        level += alpha * (value - level)
    return level


def _band(values: Sequence[float]) -> tuple[float, float, float]:
    centre = sum(values) / len(values)
    deviation = pstdev(values)
    return (
        centre - 2.0 * deviation,
        centre + 2.0 * deviation,
        ((4.0 * deviation / centre) if centre > 0 else math.inf),
    )


def prior_futures_trend_efficiency(
    bars: Sequence[Mapping[str, Any]], side: str
) -> float | None:
    """Direction-signed efficiency of 20 *preceding* one-minute futures moves.

    Assumes the caller has validated 45 contiguous completed futures bars.
    Never includes the newly generated signal candle or any future bar.
    """
    if len(bars) < 22 or side not in {"CE", "PE"}:
        return None
    try:
        closes = [float(row["close"]) for row in bars[-22:-1]]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 0 for value in closes):
        return None
    path = sum(abs(right - left) for left, right in zip(closes, closes[1:]))
    if path <= 0:
        return 0.0
    displacement = closes[-1] - closes[0]
    aligned = displacement if side == "CE" else -displacement
    return max(0.0, aligned / path)


def classify_completed_futures_bars(
    bars: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Classify three predeclared hypotheses on the most recent closed bar.

    Bar timestamps are bar *starts*. Results are observations at bar close and
    are never eligible for a fill until the following minute. Missing, stale,
    provisional, gap-spanning, or cross-session histories fail closed.
    """
    if len(bars) < 45:
        return {}
    rows = bars[-45:]
    times: list[datetime] = []
    try:
        for row in rows:
            stamp = row["timestamp"]
            if not isinstance(stamp, datetime) or stamp.tzinfo is None:
                return {}
            if row.get("is_complete") is not True or row.get("is_provisional"):
                return {}
            numbers = [
                float(row[field])
                for field in ("open", "high", "low", "close", "volume")
            ]
            opn, high, low, close, volume = numbers
            if (
                not all(math.isfinite(value) for value in numbers)
                or min(opn, high, low, close) <= 0
                or volume < 0
                or not low <= min(opn, close) <= max(opn, close) <= high
            ):
                return {}
            times.append(stamp)
    except (KeyError, TypeError, ValueError):
        return {}
    if any(
        nxt.date() != prev.date() or nxt - prev != timedelta(minutes=1)
        for prev, nxt in zip(times, times[1:])
    ):
        return {}

    closes = [float(row["close"]) for row in rows]
    volumes = [float(row["volume"]) for row in rows]
    last, previous = rows[-1], rows[-2]
    true_ranges = [
        max(
            float(rows[index]["high"]) - float(rows[index]["low"]),
            abs(float(rows[index]["high"]) - closes[index - 1]),
            abs(float(rows[index]["low"]) - closes[index - 1]),
        )
        # Fourteen completed ranges ending with the preceding bar;
        # exclude the signal candle to prevent future-bar leakage.
        for index in range(len(rows) - 15, len(rows) - 1)
    ]
    atr = sum(true_ranges) / len(true_ranges)
    volume_mean = sum(volumes[-21:-1]) / 20.0
    if atr <= 0 or volume_mean <= 0:
        return {}

    prior_ema9 = _ema(closes[:-1], 9)
    prior_ema21 = _ema(closes[:-1], 21)
    older_ema21 = _ema(closes[:-6], 21)
    prior_trend = (
        "CE"
        if prior_ema9 > prior_ema21 > older_ema21
        else "PE" if prior_ema9 < prior_ema21 < older_ema21 else None
    )
    current_close = closes[-1]
    prev_close = closes[-2]
    prev_high, prev_low = float(previous["high"]), float(previous["low"])
    expansion = abs(current_close - prev_close)
    volume_ratio = volumes[-1] / volume_mean
    candidates: dict[str, str] = {}

    if prior_trend == "CE":
        pulled_back = (
            prev_close <= prior_ema9 + 0.2 * atr
            and prev_low <= prior_ema9
            and prev_close >= prior_ema21 - 0.3 * atr
        )
        if (
            pulled_back
            and current_close > prev_high + 0.05 * atr
            and current_close > float(last["open"])
            and volume_ratio >= 1.0
        ):
            candidates["trend_pullback"] = "CE"
    elif prior_trend == "PE":
        pulled_back = (
            prev_close >= prior_ema9 - 0.2 * atr
            and prev_high >= prior_ema9
            and prev_close <= prior_ema21 + 0.3 * atr
        )
        if (
            pulled_back
            and current_close < prev_low - 0.05 * atr
            and current_close < float(last["open"])
            and volume_ratio >= 1.0
        ):
            candidates["trend_pullback"] = "PE"

    # Compare the *preceding* 20-bar band with preceding band-width history.
    # The breakout candle cannot narrow its own reference bands.
    prior_lower, prior_upper, prior_width = _band(closes[-21:-1])
    widths = [
        _band(closes[end - 19 : end + 1])[2]
        for end in range(len(closes) - 21, len(closes) - 1)
    ]
    if (
        prior_width <= 0.7 * median(widths)
        and volume_ratio >= 1.3
        and expansion >= 0.45 * atr
    ):
        if current_close > prior_upper + 0.05 * atr:
            candidates["compression_breakout"] = "CE"
        elif current_close < prior_lower - 0.05 * atr:
            candidates["compression_breakout"] = "PE"

    previous_impulse = abs(closes[-2] - closes[-3])
    if (
        prior_trend is not None
        and expansion >= max(0.45 * atr, 1.4 * previous_impulse)
        and volume_ratio >= 1.4
    ):
        if prior_trend == "CE" and current_close > prev_high:
            candidates["momentum_acceleration"] = "CE"
        elif prior_trend == "PE" and current_close < prev_low:
            candidates["momentum_acceleration"] = "PE"

    return candidates


class ArchivedDirectionalHypothesis:
    """Minimal adapter for the existing research-only archived simulator."""

    def __init__(
        self,
        family: str,
        engine: IndicatorEngine,
        *,
        min_prior_trend_efficiency: float | None = None,
    ) -> None:
        if family not in _FAMILIES:
            raise ValueError("unknown_research_hypothesis")
        if min_prior_trend_efficiency is not None and (
            not math.isfinite(min_prior_trend_efficiency)
            or not 0 <= min_prior_trend_efficiency <= 1
        ):
            raise ValueError("research_trend_efficiency_threshold_invalid")
        self.min_prior_trend_efficiency = min_prior_trend_efficiency
        self.family = family
        self.name = _FAMILIES[family]
        self.engine = engine
        self.last_no_vote_reason = "not_evaluated"
        self.config = SimpleNamespace(enabled=True)

    @property
    def evaluation_health(self) -> dict[str, bool]:
        return {"healthy": True}

    def generate_signal(
        self,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any = None,
    ) -> SimpleNamespace | None:
        del position
        self.last_no_vote_reason = "research_conditions_absent"
        side = (
            "CE"
            if symbol.upper().endswith("CE")
            else ("PE" if symbol.upper().endswith("PE") else None)
        )
        future = str(indicators.get("futures_symbol") or "")
        stamp = indicators.get("bar_timestamp")
        if (
            side is None
            or not future.startswith("NFO:NIFTY")
            or not (future.upper().endswith("FUT") or future == "NFO:NIFTY_F1_CONTEXT")
            or not isinstance(stamp, datetime)
            or stamp.tzinfo is None
        ):
            self.last_no_vote_reason = "missing_nifty_futures_identity"
            return None
        rows = list(self.engine.get_history(future, count=45, field="bars") or [])
        if not rows or rows[-1].get("timestamp") != stamp:
            self.last_no_vote_reason = "futures_bar_not_current"
            return None
        detected = classify_completed_futures_bars(rows)
        if detected.get(self.family) != side:
            return None
        if (
            self.min_prior_trend_efficiency is not None
            and self.family in {"trend_pullback", "momentum_acceleration"}
        ):
            efficiency = prior_futures_trend_efficiency(rows, side)
            if efficiency is None or efficiency < self.min_prior_trend_efficiency:
                self.last_no_vote_reason = "prior_futures_trend_inefficient"
                return None

        try:
            price = float(current_price)
            option_atr = float(self.engine.get_atr(symbol) or 0.0)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (price, option_atr)):
            return None
        # Geometry is an explicitly non-live research proxy; never infer option
        # premium stops from futures price levels.
        stop_distance = 1.2 * option_atr
        if option_atr <= 0 or price <= stop_distance + 0.05:
            self.last_no_vote_reason = "option_research_geometry_unavailable"
            return None
        self.last_no_vote_reason = ""
        return SimpleNamespace(
            action="BUY",
            stop_loss=price - stop_distance,
            take_profit=price + 2.0 * stop_distance,
            metadata={
                "setup_id": f"research:{self.family}:{stamp.isoformat()}:{side}",
                "setup_name": self.family,
                "setup_type": self.family,
                "signal_family": "research_only",
                "research_geometry_source": "option_atr_proxy",
                "contract_side": side,
            },
        )

    def notify_entry_accepted(self, side: str, *, setup_id: str | None = None) -> None:
        """Research callbacks never touch live positions or brokers."""
        del side, setup_id


class ORBRangeScaledResearch(ORBProStrategy):
    """Research-only 15-minute range scaled to its 1-minute volatility horizon.

    Does not change the production ORB class, thresholds, volume confirmation,
    direction resolution, breakout logic, or broker execution behavior.
    """

    def __init__(self, config: Any, indicator_engine: IndicatorEngine) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)
        self.research_gate_counts: Counter[str] = Counter()

    def _structural_evidence(
        self,
        *,
        side: str,
        direction: str,
        penetration_atr: float,
        volume_ratio: float,
        opening_range_atr: float,
        indicators: Mapping[str, Any],
    ) -> tuple[bool, list[str], bool, float, float]:
        inputs = {
            "side": side,
            "direction": direction,
            "penetration_atr": penetration_atr,
            "volume_ratio": volume_ratio,
            "indicators": indicators,
        }
        raw = super()._structural_evidence(
            **inputs, opening_range_atr=opening_range_atr
        )
        normalized = super()._structural_evidence(
            **inputs,
            opening_range_atr=opening_range_atr
            / math.sqrt(max(1, int(self._cfg.orb_minutes))),
        )
        self.research_gate_counts["evaluated"] += 1
        self.research_gate_counts["raw_all_pass"] += int(raw[0])
        self.research_gate_counts["scaled_all_pass"] += int(normalized[0])
        self.research_gate_counts["raw_range_pass"] += int(raw[2])
        self.research_gate_counts["scaled_range_pass"] += int(normalized[2])
        for condition in (
            "underlying_direction_alignment",
            "underlying_volume_confirmation",
            "normalized_breakout_penetration",
            "futures_vwap_slope_alignment",
        ):
            if condition not in normalized[1]:
                self.research_gate_counts["failed_" + condition] += 1
        return normalized


def _research_strategies(
    engine: IndicatorEngine,
    *,
    min_prior_trend_efficiency: float | None = None,
) -> list[ArchivedDirectionalHypothesis]:
    return [
        ArchivedDirectionalHypothesis(
            family,
            engine,
            min_prior_trend_efficiency=min_prior_trend_efficiency,
        )
        for family in _FAMILIES
    ]


def run_directional_hypothesis_research(
    directory: Path, *, slippage_bps: float = 10.0
) -> dict[str, Any]:
    """Backtest fixed non-live hypotheses using canonical archived fills/costs."""
    from nifty_scalper_bot.backtesting.strategy_research import (
        _scenario,
        load_archive,
    )

    if os.getenv("EXECUTION_MODE", "SHADOW").upper() != "SHADOW":
        raise ValueError("research_requires_shadow_process")
    if not math.isfinite(slippage_bps) or not 0 <= slippage_bps <= 1000:
        raise ValueError("research_slippage_invalid")
    histories, instruments, digest = load_archive(directory)
    scenarios = [
        _scenario(
            histories,
            instruments,
            get_settings().elite,
            bps,
            components=set(_FAMILIES.values()),
            minimum_net_rr=1.5,
            strict_liquidity=True,
            lifecycle_proxy=True,
            strategy_factory=_research_strategies,
        )
        for bps in sorted(
            {slippage_bps, max(25.0, slippage_bps), max(50.0, slippage_bps)}
        )
    ]
    return {
        "scope": "research_only_intraday_directional_hypotheses",
        "evidence_label": "RESEARCH_CANDIDATE",
        "live_equivalent": False,
        "data_sha256": digest,
        "families": dict(_FAMILIES),
        "selected_for_live": None,
        "promotion_eligible": False,
        "assumptions": {
            "direction": "completed_NIFTY_futures_bars_only",
            "entry": "following_minute_open",
            "geometry": "1.2_option_atr_proxy_stop_2R_target",
            "costs": "canonical_cost_model_and_10_25_50bps_slippage_sensitivity",
            "lifecycle": "existing_causal_bar_proxy",
            "holding": "single_archived_lot_per_strategy_and_option",
        },
        "limitations": [
            "Retrospectively selected option contracts; no historical ATM rotation",
            "No historical executable bid/ask or depth",
            "Bar-based stop/target path sometimes unresolved; stress outcomes separate",
            "Historical replay lacks full live arbitration, risk and broker semantics",
            "Retrospective last 20 percent is not an untouched holdout",
            "Parameters are preregistered research hypotheses, not optimized alpha",
            "No strategy may enter live execution without independent validation",
        ],
        "scenarios": scenarios,
    }


__all__ = [
    "ArchivedDirectionalHypothesis",
    "ORBRangeScaledResearch",
    "classify_completed_futures_bars",
    "run_directional_hypothesis_research",
]
