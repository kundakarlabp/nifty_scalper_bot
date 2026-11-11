"""Compare failed-breakout reversal against SMC on matched historical events.

Responsibilities: Build a causal futures-confirmation cohort, replay both strategy
classes through the existing costed component simulator, and reconcile identical
option-contract/entry-minute trade outcomes.
Boundaries: Research-only; does not implement a new strategy, tune production,
fetch broker data, or replace the canonical replay/contract owners.
Safety: Incomplete history, missing exits and absent fills never imply profit.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Mapping

from nifty_scalper_bot.backtesting.directional_hypotheses import (
    ArchivedDirectionalHypothesis,
    classify_completed_futures_bars,
)
from nifty_scalper_bot.backtesting.strategy_research import (
    _archived_weekly_atm_rejection,
    _scenario,
    load_archive,
)
from nifty_scalper_bot.config.settings import get_settings
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy

_COMPONENTS = ("SMC", "FailedBreakoutResearch")
_SLIPPAGE_BPS = (10.0, 25.0, 50.0)


def completed_reversal_event_cohort(
    histories: Mapping[str, Mapping[datetime, Mapping[str, Any]]],
    instruments: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Identify futures-confirmed reversal/contract observations, without future bars.

    The following minute is merely the earliest possible entry. Its existence,
    prices, volume and future return are not examined when forming the cohort.
    """
    futures = [
        symbol
        for symbol, instrument in instruments.items()
        if instrument.get("instrument_type") == "FUT"
    ]
    if len(futures) != 1 or "NSE:NIFTY 50" not in histories:
        raise ValueError("reversal_matched_futures_or_spot_unavailable")
    future = futures[0]
    observed: list[dict[str, str]] = []
    completed: list[Mapping[str, Any]] = []
    for timestamp, bar in sorted(histories[future].items()):
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("reversal_matched_timestamp_naive")
        completed.append(bar)
        if not time(9, 30) <= timestamp.time() < time(14, 59):
            continue
        side = classify_completed_futures_bars(completed[-45:]).get(
            "failed_breakout_reversal"
        )
        spot = histories["NSE:NIFTY 50"].get(timestamp)
        if side is None or spot is None:
            continue
        for symbol, instrument in sorted(instruments.items()):
            if instrument.get("instrument_type") != side:
                continue
            option_bar = histories.get(symbol, {}).get(timestamp)
            if option_bar is None or float(option_bar.get("volume", 0)) <= 0:
                continue
            if _archived_weekly_atm_rejection(
                instrument, timestamp, float(spot["close"])
            ) is not None:
                continue
            observed.append(
                {
                    "event_bar_start": timestamp.isoformat(),
                    "earliest_entry": (timestamp + timedelta(minutes=1)).isoformat(),
                    "future": future,
                    "symbol": symbol,
                    "side": side,
                }
            )
    return observed


def _trade_index(
    results: Mapping[str, Any], *, unresolved: bool
) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = results["unresolved_trades" if unresolved else "trades"]
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        stamp = str(row.get("entry_time") or "")
        symbol = str(row.get("symbol") or "")
        try:
            entry_at = datetime.fromisoformat(stamp)
        except ValueError as exc:
            raise ValueError("reversal_matched_trade_timestamp_invalid") from exc
        if entry_at.tzinfo is None or not symbol:
            raise ValueError("reversal_matched_trade_identity_invalid")
        if not unresolved and (
            str(row.get("cost_source")) != "canonical_model_estimate"
            or not math.isfinite(float(row["net_pnl"]))
            or not math.isfinite(float(row["fees"]))
        ):
            raise ValueError("reversal_matched_trade_costs_unverified")
        key = (entry_at.isoformat(), symbol)
        if key in indexed:
            raise ValueError("reversal_matched_duplicate_trade")
        indexed[key] = row
    return indexed


def compare_smc_reversal_events(
    cohort: list[dict[str, str]],
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare resolved and unresolved next-minute fills on the SAME event/option.

    No fill does not mean the strategy voted against that event: its signal may
    have been suppressed, unfillable, or already in a position. No P&L is
    inferred for an unfilled or unpriceable opportunity.
    """
    strategies = scenario["strategies"]
    if set(strategies) != set(_COMPONENTS):
        raise ValueError("reversal_matched_component_set_invalid")
    resolved = {
        name: _trade_index(strategies[name], unresolved=False)
        for name in _COMPONENTS
    }
    missing_exit = {
        name: _trade_index(strategies[name], unresolved=True)
        for name in _COMPONENTS
    }
    event_keys: set[tuple[str, str]] = set()
    paired: list[dict[str, Any]] = []
    for event in cohort:
        key = (event["earliest_entry"], event["symbol"])
        if key in event_keys:
            raise ValueError("reversal_matched_duplicate_opportunity")
        event_keys.add(key)
        outcomes: dict[str, Any] = {}
        for name in _COMPONENTS:
            trade = resolved[name].get(key)
            ambiguous = missing_exit[name].get(key)
            if trade is not None and ambiguous is not None:
                raise ValueError("reversal_matched_ambiguous_exit")
            outcomes[name] = {
                "resolved_fill": trade is not None,
                "unresolved_exit": ambiguous is not None,
                "net_pnl": float(trade["net_pnl"]) if trade is not None else None,
                "costs": float(trade["fees"]) if trade is not None else None,
                "absence_means_no_signal": False,
            }
        paired.append({**event, "outcomes": outcomes})

    fills = {
        name: sum(row["outcomes"][name]["resolved_fill"] for row in paired)
        for name in _COMPONENTS
    }
    unresolved_counts = {
        name: sum(row["outcomes"][name]["unresolved_exit"] for row in paired)
        for name in _COMPONENTS
    }
    same_entry = sum(
        all(row["outcomes"][name]["resolved_fill"] for name in _COMPONENTS)
        for row in paired
    )
    reversal_only_wins = sum(
        row["outcomes"]["FailedBreakoutResearch"]["resolved_fill"]
        and not row["outcomes"]["SMC"]["resolved_fill"]
        and not row["outcomes"]["SMC"]["unresolved_exit"]
        and row["outcomes"]["FailedBreakoutResearch"]["net_pnl"] > 0
        for row in paired
    )
    return {
        "slippage_bps_per_side": scenario["slippage_bps_per_side"],
        "underlying_event_count": len(
            {(row["event_bar_start"], row["side"]) for row in paired}
        ),
        "matched_option_event_count": len(paired),
        "independent_session_count": len(
            {row["event_bar_start"][:10] for row in paired}
        ),
        "resolved_fills": fills,
        "unresolved_exits": unresolved_counts,
        "both_resolved_same_entry_count": same_entry,
        "reversal_only_positive_resolved_fill_count": reversal_only_wins,
        "research_only_missed_opportunity_proxy": True,
        "outperformance_inference_allowed": False,
        "matching": "same_completed_futures_event_option_symbol_and_next_minute",
        "events": paired,
    }


def run_failed_breakout_smc_comparison(directory: Path) -> dict[str, Any]:
    """Replay SMC and independent reversal on one common, research-only cohort."""
    if os.getenv("EXECUTION_MODE", "SHADOW").strip().upper() != "SHADOW":
        raise ValueError("reversal_smc_comparison_requires_shadow")
    histories, instruments, digest = load_archive(directory)
    cohort = completed_reversal_event_cohort(histories, instruments)
    settings = get_settings().elite

    def factory(engine: Any) -> list[Any]:
        return [
            SMCStrategy(settings.smc, engine),
            ArchivedDirectionalHypothesis("failed_breakout_reversal", engine),
        ]

    scenarios = [
        compare_smc_reversal_events(
            cohort,
            _scenario(
                histories,
                instruments,
                settings,
                bps,
                components=set(_COMPONENTS),
                minimum_net_rr=1.5,
                strict_liquidity=True,
                lifecycle_proxy=True,
                require_archived_weekly_atm=True,
                strategy_factory=factory,
            ),
        )
        for bps in _SLIPPAGE_BPS
    ]
    if any(
        row["underlying_event_count"] != scenarios[0]["underlying_event_count"]
        or row["matched_option_event_count"]
        != scenarios[0]["matched_option_event_count"]
        for row in scenarios
    ):
        raise ValueError("reversal_matched_cohort_drift")
    return {
        "scope": "same_event_smc_vs_failed_breakout_reversal",
        "evidence_label": "RESEARCH_CANDIDATE",
        "data_sha256": digest,
        "live_equivalent": False,
        "promotion_eligible": False,
        "selected_for_live": None,
        "scenarios": scenarios,
        "limitations": [
            "SMC may signal earlier; same-minute fills are incomplete attribution",
            "Missing fill cannot distinguish no-vote, rejection or pending entry",
            "Weekly-ATM proximity is not actual historical contract selection",
            "Archived OHLC lacks executable bid/ask depth and real broker queue/fills",
            "Replay omits live arbitration, contract rotation and some exits",
            "No untouched chronological holdout or prospective paper confirmation",
        ],
    }


__all__ = [
    "compare_smc_reversal_events",
    "completed_reversal_event_cohort",
    "run_failed_breakout_smc_comparison",
]
