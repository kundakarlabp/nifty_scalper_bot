"""Causal bar research of production strategy components, without broker orders.

This is deliberately not live-pipeline parity: today's archived contract basket,
minute OHLC and modeled execution cannot reproduce historical ATM selection,
depth, arbitration, dynamic sizing or the full protective-exit lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from nifty_scalper_bot.config.settings import get_settings
from nifty_scalper_bot.risk.cost_model import (
    estimate_round_trip_cost,
    evaluate_net_reward_risk,
)
from nifty_scalper_bot.strategies.elite_strategies.builder import build_elite_strategies
from nifty_scalper_bot.strategies.indicators import IndicatorEngine

IST = ZoneInfo("Asia/Kolkata")
COMPONENTS = {"ORBPro", "SMC", "VWAPPro"}


def load_archive(
    directory: Path,
) -> tuple[dict[str, dict[datetime, dict[str, Any]]], dict[str, dict[str, Any]], str]:
    """Validate and normalize raw Kite arrays/dicts; reject conflicting history."""
    histories: dict[str, dict[datetime, dict[str, Any]]] = {}
    instruments: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    for path in sorted((directory / "candles").glob("*.json")):
        raw = path.read_bytes()
        digest.update(raw)
        payload = json.loads(raw)
        if payload.get("timestamp_convention") != "bar_start":
            raise ValueError("research_timestamp_convention_invalid")
        symbol = payload["symbol"]
        instrument = payload["instrument"]
        if symbol in instruments and instruments[symbol] != instrument:
            raise ValueError("research_instrument_identity_conflict")
        instruments[symbol] = instrument
        history = histories.setdefault(symbol, {})
        for candle in payload["candles"]:
            row = (
                dict(candle)
                if isinstance(candle, dict)
                else dict(
                    zip(
                        ("date", "open", "high", "low", "close", "volume", "oi"), candle
                    )
                )
            )
            timestamp = datetime.fromisoformat(
                str(row.get("date") or row.get("timestamp")).replace("Z", "+00:00")
            )
            if timestamp.tzinfo is None:
                raise ValueError("research_timestamp_timezone_missing")
            timestamp = timestamp.astimezone(IST)
            if timestamp.second or timestamp.microsecond:
                raise ValueError("research_bar_alignment_invalid")
            bar = {
                key: float(row[key])
                for key in ("open", "high", "low", "close", "volume")
            }
            if (
                not all(math.isfinite(value) for value in bar.values())
                or min(bar[key] for key in ("open", "high", "low", "close")) <= 0
                or bar["volume"] < 0
            ):
                raise ValueError("research_bar_values_invalid")
            if (
                not bar["low"]
                <= min(bar["open"], bar["close"])
                <= max(bar["open"], bar["close"])
                <= bar["high"]
            ):
                raise ValueError("research_bar_geometry_invalid")
            normalized = {
                **bar,
                "timestamp": timestamp,
                "is_complete": True,
                "is_provisional": False,
            }
            if timestamp in history and history[timestamp] != normalized:
                raise ValueError("research_duplicate_bar_conflict")
            history[timestamp] = normalized
    kinds = {row.get("instrument_type") for row in instruments.values()}
    if (
        not histories
        or not {"CE", "PE", "FUT"}.issubset(kinds)
        or "NSE:NIFTY 50" not in histories
    ):
        raise ValueError("research_history_unavailable")
    for symbol, row in instruments.items():
        if row.get("instrument_type") in {"CE", "PE", "FUT"} and (
            row.get("name") != "NIFTY" or not symbol.startswith("NFO:NIFTY")
        ):
            raise ValueError("research_instrument_not_nifty")
        if row.get("instrument_type") in {"CE", "PE"}:
            if (
                not symbol.startswith("NFO:")
                or int(row.get("lot_size", 0)) <= 0
                or not histories[symbol]
            ):
                raise ValueError("research_option_identity_invalid")
            expiry = datetime.fromisoformat(str(row["expiry"])).date()
            if any(ts.date() > expiry for ts in histories[symbol]):
                raise ValueError("research_option_expired")
    return histories, instruments, digest.hexdigest()


def summarize(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Report post-cost realized metrics; undefined ratios remain null."""
    ordered = sorted(trades, key=lambda trade: trade["exit_time"])
    values = [float(trade["net_pnl"]) for trade in ordered]
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    equity = peak = drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "trade_count": len(values),
        "net_pnl": sum(values),
        "expectancy": sum(values) / len(values) if values else None,
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else None,
        "win_rate": len(wins) / len(values) if values else None,
        "average_win": sum(wins) / len(wins) if wins else None,
        "average_loss": sum(losses) / len(losses) if losses else None,
        "max_realized_drawdown": drawdown,
        "worst_trade": min(values) if values else None,
        "fees": sum(trade["fees"] for trade in trades),
        "turnover": sum(
            (trade["entry_price"] + trade["exit_price"]) * trade["quantity"]
            for trade in trades
        ),
        "exposure_minutes": sum(trade["duration_minutes"] for trade in trades),
    }


def _scenario(
    histories: dict[str, dict[datetime, dict[str, Any]]],
    instruments: dict[str, dict[str, Any]],
    settings: Any,
    slippage_bps: float,
    *,
    components: set[str] | None = None,
    minimum_net_rr: float | None = None,
    minimum_opening_rvol: float | None = None,
    compact_orb_context: bool = False,
    strict_liquidity: bool = False,
) -> dict[str, Any]:
    if compact_orb_context and components != {"ORBPro"}:
        raise ValueError("research_compact_context_requires_orb_only")
    options = sorted(
        symbol
        for symbol, row in instruments.items()
        if row.get("instrument_type") in {"CE", "PE"}
    )
    future = next(
        symbol
        for symbol, row in instruments.items()
        if row.get("instrument_type") == "FUT"
    )
    timestamps = sorted(set().union(*(set(history) for history in histories.values())))
    outcomes: dict[str, list[dict[str, Any]]] = {}
    unresolved_outcomes: dict[str, list[dict[str, Any]]] = {}
    stress_outcomes: dict[str, list[dict[str, Any]]] = {}
    rejections: dict[str, Counter[str]] = {}
    evaluated: Counter[str] = Counter()
    pending: dict[tuple[str, str], dict[str, Any]] = {}
    positions: dict[tuple[str, str], dict[str, Any]] = {}
    engine = IndicatorEngine()
    strategies = []
    current_day = None
    last_bars: dict[str, dict[str, Any]] = {}
    slip = slippage_bps / 10000.0

    def _trade_record(
        key: tuple[str, str],
        position: dict[str, Any],
        observed_at: datetime,
        price: float,
        reason: str,
    ) -> dict[str, Any]:
        exit_price = max(0.01, price * (1 - slip))
        cost = estimate_round_trip_cost(
            entry_price=position["entry_price"],
            exit_price=exit_price,
            quantity=position["quantity"],
        )
        gross = (exit_price - position["entry_price"]) * position["quantity"]
        exit_time = observed_at + timedelta(minutes=1)
        return {
            **position,
            "symbol": key[1],
            "entry_time": position["entry_time"].isoformat(),
            "exit_time": exit_time.isoformat(),
            "exit_price": exit_price,
            "exit_reason": reason,
            "gross_pnl": gross,
            "fees": cost.total,
            "net_pnl": gross - cost.total,
            "cost_source": "canonical_model_estimate",
            "duration_minutes": (
                exit_time - position["entry_time"]
            ).total_seconds()
            / 60,
        }

    def close(
        key: tuple[str, str], bar: dict[str, Any], price: float, reason: str
    ) -> None:
        position = positions.pop(key)
        trade = _trade_record(key, position, bar["timestamp"], price, reason)
        outcomes[key[0]].append(trade)
        stress_outcomes[key[0]].append(dict(trade))

    def mark_unresolved(
        key: tuple[str, str],
        observed_at: datetime,
        reason: str,
        *,
        stress_price: float = 0.01,
        stress_reason: str = "unpriced_full_premium_stress",
    ) -> None:
        """Remove an unpriceable position without inventing a primary P&L."""
        position = positions.pop(key)
        unresolved_outcomes[key[0]].append(
            {
                **position,
                "symbol": key[1],
                "entry_time": position["entry_time"].isoformat(),
                "observed_at": observed_at.isoformat(),
                "unresolved_reason": reason,
            }
        )
        stress_outcomes[key[0]].append(
            _trade_record(
                key,
                position,
                observed_at,
                stress_price,
                stress_reason,
            )
        )

    for timestamp in timestamps:
        if current_day != timestamp.date():
            for key in list(positions):
                observed = last_bars.get(key[1])
                mark_unresolved(
                    key,
                    (
                        observed["timestamp"]
                        if observed is not None
                        else timestamp - timedelta(minutes=1)
                    ),
                    "session_data_end_unresolved",
                    stress_reason="unpriced_session_end_stress",
                )
            pending.clear()
            engine = IndicatorEngine()
            strategies = [
                strategy
                for strategy in build_elite_strategies(settings, engine)
                if strategy.name
                in (components if components is not None else COMPONENTS)
            ]
            if not strategies:
                raise ValueError("research_components_unavailable")
            for strategy in strategies:
                outcomes.setdefault(strategy.name, [])
                unresolved_outcomes.setdefault(strategy.name, [])
                stress_outcomes.setdefault(strategy.name, [])
                rejections.setdefault(strategy.name, Counter())
            current_day = timestamp.date()
        bars = {
            symbol: history[timestamp]
            for symbol, history in histories.items()
            if timestamp in history
        }
        # Fill intents before exposing this minute's completed OHLC to strategies.
        for key, intent in list(pending.items()):
            pending.pop(key)
            bar = bars.get(key[1])
            if bar is None or timestamp != intent["available_at"]:
                rejections[key[0]]["next_minute_unavailable"] += 1
                continue
            if strict_liquidity and bar["volume"] <= 0:
                rejections[key[0]]["next_minute_has_no_trades"] += 1
                continue
            entry = bar["open"] * (1 + slip)
            if not intent["stop_loss"] < entry < intent["take_profit"]:
                rejections[key[0]]["gap_invalidates_geometry"] += 1
                continue
            quantity = int(instruments[key[1]]["lot_size"])
            if minimum_net_rr is not None:
                economics = evaluate_net_reward_risk(
                    entry_price=entry,
                    stop_price=intent["stop_loss"] * (1 - slip),
                    target_price=intent["take_profit"] * (1 - slip),
                    quantity=quantity,
                )
                if economics.net_rr < minimum_net_rr:
                    rejections[key[0]]["cost_net_rr_rejected"] += 1
                    continue
            positions[key] = {
                "entry_time": timestamp,
                "entry_price": entry,
                "stop_loss": intent["stop_loss"],
                "take_profit": intent["take_profit"],
                "quantity": quantity,
            }
            intent["strategy"].notify_entry_accepted(
                instruments[key[1]]["instrument_type"],
                setup_id=intent["setup_id"],
            )
        for key, position in list(positions.items()):
            bar = bars.get(key[1])
            if bar is None or (strict_liquidity and bar["volume"] <= 0):
                mark_unresolved(
                    key,
                    timestamp,
                    (
                        "zero_volume_exit_unresolved"
                        if bar is not None
                        else "history_gap_unresolved"
                    ),
                    stress_reason="unpriced_gap_full_premium_stress",
                )
                continue
            stop, target = position["stop_loss"], position["take_profit"]
            hit_stop = bar["low"] <= stop
            hit_target = bar["high"] >= target
            if bar["open"] <= stop:
                close(key, bar, bar["open"], "gap_stop")
            elif hit_stop and hit_target:
                mark_unresolved(
                    key,
                    timestamp,
                    "ambiguous_intrabar_stop_target",
                    stress_price=stop,
                    stress_reason="ambiguous_stop_first_stress",
                )
            elif hit_stop:
                close(key, bar, stop, "stop")
            elif hit_target:
                close(key, bar, target, "target")
            elif timestamp.time() >= time(14, 59):
                close(key, bar, bar["close"], "session_exit")
        for symbol, bar in bars.items():
            engine.ingest_historical_bar(symbol, bar)
            last_bars[symbol] = bar
        if not time(9, 30) <= timestamp.time() < time(14, 59):
            continue
        if compact_orb_context:
            range_end = timestamp.replace(hour=9, minute=15) + timedelta(
                minutes=settings.orb.orb_minutes
            )
            entry_end = range_end + timedelta(
                minutes=max(
                    1.0, float(os.getenv("ORB_MAX_ENTRY_MINUTES_AFTER_RANGE", "120"))
                )
            )
            if timestamp > entry_end:
                continue
        for symbol in options:
            if symbol not in bars or future not in bars or "NSE:NIFTY 50" not in bars:
                continue
            bar = bars[symbol]
            if strict_liquidity and bar["volume"] <= 0:
                continue
            # ORB's only option-derived numeric feature is canonical ATR.
            # Underlying structure still comes from the production strategy's
            # completed-history interface. Other components require full context.
            indicators = (
                {"atr": engine.get_atr(symbol)}
                if compact_orb_context
                else dict(engine.get_indicators(symbol))
            )
            indicators.update(
                history_count=engine.history_count(symbol),
                bar_timestamp=timestamp,
                timestamp=timestamp,
                session_date=timestamp.date().isoformat(),
                futures_symbol=future,
                spot_symbol="NSE:NIFTY 50",
                stale_data_used=False,
            )
            for strategy in strategies:
                key = (strategy.name, symbol)
                if key in positions:
                    continue
                evaluated[strategy.name] += 1
                signal = strategy.generate_signal(symbol, indicators, bar["close"])
                if not strategy.evaluation_health["healthy"]:
                    raise ValueError("research_strategy_evaluation_failed")
                if signal is None:
                    rejections[strategy.name][
                        getattr(strategy, "last_no_vote_reason", "no_signal")
                    ] += 1
                    continue
                if (
                    signal.action != "BUY"
                    or signal.stop_loss is None
                    or signal.take_profit is None
                    or not 0 < signal.stop_loss < bar["close"] < signal.take_profit
                ):
                    rejections[strategy.name]["invalid_signal_geometry"] += 1
                    continue
                if minimum_opening_rvol is not None:
                    rvol = opening_relative_volume(
                        histories[future], timestamp, settings.orb.orb_minutes
                    )
                    if rvol is None or rvol < minimum_opening_rvol:
                        rejections[strategy.name][
                            (
                                "opening_rvol_unavailable"
                                if rvol is None
                                else "opening_rvol_below_minimum"
                            )
                        ] += 1
                        continue
                pending[key] = {
                    "available_at": timestamp + timedelta(minutes=1),
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "strategy": strategy,
                    "setup_id": signal.metadata.get("setup_id"),
                }
    for key in list(positions):
        observed = last_bars.get(key[1])
        mark_unresolved(
            key,
            observed["timestamp"] if observed is not None else timestamps[-1],
            "session_data_end_unresolved",
            stress_reason="unpriced_session_end_stress",
        )
    days = sorted(
        {ts.date().isoformat() for symbol in options for ts in histories[symbol]}
    )
    cutoff = days[max(0, int(len(days) * 0.8))] if days else ""
    for trades in outcomes.values():
        trades.sort(key=lambda trade: (trade["exit_time"], trade["symbol"]))
    for trades in stress_outcomes.values():
        trades.sort(key=lambda trade: (trade["exit_time"], trade["symbol"]))
    return {
        "slippage_bps_per_side": slippage_bps,
        "strategies": {
            name: {
                "metrics": summarize(trades),
                "worst_case_stress_metrics": summarize(stress_outcomes[name]),
                "chronological_holdout_start": cutoff,
                "development_metrics": summarize(
                    [trade for trade in trades if trade["entry_time"][:10] < cutoff]
                ),
                "exit_reasons": dict(Counter(trade["exit_reason"] for trade in trades)),
                "worst_case_stress_exit_reasons": dict(
                    Counter(
                        trade["exit_reason"]
                        for trade in stress_outcomes[name]
                    )
                ),
                "data_quality": {
                    "resolved_exit_count": len(trades),
                    "unresolved_exit_count": len(unresolved_outcomes[name]),
                    "unresolved_exit_rate": (
                        len(unresolved_outcomes[name])
                        / (len(trades) + len(unresolved_outcomes[name]))
                        if trades or unresolved_outcomes[name]
                        else 0.0
                    ),
                    "primary_metrics_complete": not unresolved_outcomes[name],
                    "primary_metrics_exclude_unresolved": True,
                    "development_unresolved_exit_count": sum(
                        trade["entry_time"][:10] < cutoff
                        for trade in unresolved_outcomes[name]
                    ),
                    "holdout_unresolved_exit_count": sum(
                        trade["entry_time"][:10] >= cutoff
                        for trade in unresolved_outcomes[name]
                    ),
                },
                "holdout_metrics": summarize(
                    [trade for trade in trades if trade["entry_time"][:10] >= cutoff]
                ),
                "evaluations": evaluated[name],
                "no_vote_reasons": dict(rejections[name]),
                "unresolved_trades": unresolved_outcomes[name],
                "trades": trades,
                "worst_case_stress_trades": stress_outcomes[name],
            }
            for name, trades in outcomes.items()
        },
    }


def run_orb_session_research(
    directory: Path,
    *,
    overrides: dict[str, str],
    slippage_bps: float,
    minimum_net_rr: float | None = 1.5,
) -> dict[str, Any]:
    """Replay a preselected historical basket in an isolated research process.

    Contract selection and chronological settings selection belong to the
    research orchestrator. No broker or live execution path is instantiated.
    """
    if os.getenv("EXECUTION_MODE", "SHADOW").upper() != "SHADOW":
        raise ValueError("research_requires_shadow_process")
    if any(not key.startswith("ORB_") for key in overrides):
        raise ValueError("research_override_not_orb_setting")
    if not math.isfinite(slippage_bps) or not 0 <= slippage_bps <= 1000:
        raise ValueError("research_slippage_invalid")
    histories, instruments, _ = load_archive(directory)
    original = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        result = _scenario(
            histories,
            instruments,
            get_settings().elite,
            slippage_bps,
            components={"ORBPro"},
            minimum_net_rr=minimum_net_rr,
            compact_orb_context=True,
            strict_liquidity=True,
        )
        return result["strategies"]["ORBPro"]
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_archived_research(
    directory: Path, *, slippage_bps: float = 10.0
) -> dict[str, Any]:
    """Run a fixed baseline plus execution-cost sensitivity without tuning."""
    if os.getenv("EXECUTION_MODE", "SHADOW").upper() != "SHADOW":
        raise ValueError("research_requires_shadow_process")
    if not math.isfinite(slippage_bps) or slippage_bps < 0:
        raise ValueError("research_slippage_invalid")
    histories, instruments, digest = load_archive(directory)
    settings = get_settings().elite
    scenarios = [
        _scenario(histories, instruments, settings, value)
        for value in sorted(
            {slippage_bps, max(25.0, slippage_bps), max(50.0, slippage_bps)}
        )
    ]
    return {
        "scope": "active_contract_strategy_components",
        "evidence_label": "RESEARCH_CANDIDATE",
        "live_equivalent": False,
        "data_sha256": digest,
        "configuration": {
            name: asdict(getattr(settings, name)) for name in ("orb", "smc", "vwap")
        },
        "cost_environment": {
            key: os.environ[key]
            for key in (
                "COST_BROKERAGE_PER_ORDER",
                "COST_STT_SELL_PCT",
                "COST_EXCH_TXN_PCT",
                "COST_SEBI_PCT",
                "COST_GST_PCT",
                "COST_STAMP_BUY_PCT",
            )
            if key in os.environ
        },
        "coverage": {
            symbol: {
                "bar_count": len(history),
                "first": min(history).isoformat() if history else None,
                "last": max(history).isoformat() if history else None,
                "session_count": len({ts.date() for ts in history}),
            }
            for symbol, history in histories.items()
        },
        "assumptions": {
            "entry": "next_minute_open",
            "ambiguous_stop_target": (
                "unresolved_in_primary; stop_first only in worst_case_stress"
            ),
            "missing_exit_data": (
                "unresolved_in_primary; full-premium loss only in worst_case_stress"
            ),
            "quantity": "one_archived_lot_per_component_and_option",
            "costs": "canonical_model_estimate",
            "strategy_mode": "SHADOW",
            "holdout": "last_20_percent_sessions_no_parameter_tuning",
        },
        "limitations": [
            "Current contracts selected retrospectively; "
            "no historical ATM/expiry rotation",
            "No historical bid/ask depth; slippage is assumed and varied",
            "Component signals only; no live arbitration, risk sizing, "
            "trailing lifecycle or broker fill parity",
            "Missing underlying live direction context is left unavailable "
            "rather than fabricated",
            "Drawdown is realized trade drawdown, "
            "not mark-to-market portfolio drawdown",
            "Small or zero trade samples do not establish profitability",
        ],
        "scenarios": scenarios,
    }


def opening_relative_volume(
    history: dict[datetime, dict[str, Any]], timestamp: datetime, minutes: int
) -> float | None:
    """Same-clock opening volume / preceding 14 complete session volumes.

    Current/future sessions never enter the denominator; incomplete warmup is
    explicitly unavailable. This is a research filter, not live context.
    """
    opening = timestamp.replace(hour=9, minute=15, second=0, microsecond=0)
    if minutes < 1 or timestamp < opening + timedelta(minutes=minutes):
        return None
    prior_days = sorted({ts.date() for ts in history if ts.date() < timestamp.date()})[
        -14:
    ]
    if len(prior_days) < 14:
        return None
    volumes = []
    for day in [*prior_days, timestamp.date()]:
        start = opening.replace(year=day.year, month=day.month, day=day.day)
        keys = [start + timedelta(minutes=minute) for minute in range(minutes)]
        if any(key not in history for key in keys):
            return None
        volumes.append(sum(float(history[key]["volume"]) for key in keys))
    mean = sum(volumes[:-1]) / 14
    return volumes[-1] / mean if mean > 0 else None


def run_orb_comparison(directory: Path) -> dict[str, Any]:
    """Compare a registered bounded set; never promote a live configuration.

    The archive was already inspected. Its final 20% is a retrospective check,
    not an untouched holdout. Rank solely on development-period stress results.
    """
    if os.getenv("EXECUTION_MODE", "SHADOW").upper() != "SHADOW":
        raise ValueError("research_requires_shadow_process")
    histories, instruments, digest = load_archive(directory)
    settings = get_settings().elite
    # All candidates differ in one hypothesis from the cost-gated reference.
    variants: list[
        tuple[str, dict[str, str], int | None, float | None, float | None]
    ] = [
        ("raw_reference", {}, None, None, None),
        ("cost_gated_reference", {}, None, 1.5, None),
        ("retest_only", {"ORB_MOMENTUM_BRANCH_ENABLED": "false"}, None, 1.5, None),
        (
            "early_entry_60",
            {"ORB_MAX_ENTRY_MINUTES_AFTER_RANGE": "60"},
            None,
            1.5,
            None,
        ),
        ("target_rr_2_2", {"ORB_TARGET_RR": "2.2"}, None, 1.5, None),
        ("range_5", {}, 5, 1.5, None),
        ("range_10", {}, 10, 1.5, None),
        ("range_30", {}, 30, 1.5, None),
        ("opening_rvol_1", {}, None, 1.5, 1.0),
    ]
    environment_keys = (
        "ORB_MOMENTUM_BRANCH_ENABLED",
        "ORB_MAX_ENTRY_MINUTES_AFTER_RANGE",
        "ORB_TARGET_RR",
        "ORB_QUALITY_MIN_SCORE_SHADOW",
        "ORB_MOMENTUM_MIN_BODY_PCT",
        "ORB_MOMENTUM_MIN_PENETRATION_ATR",
        "ORB_MOMENTUM_MIN_VOLUME_RATIO",
        "ORB_MAX_EVENTS_PER_SIDE",
    )
    baseline_environment = {key: os.environ.get(key) for key in environment_keys}
    candidates: list[dict[str, Any]] = []
    for name, overrides, minutes, minimum_rr, rvol in variants:
        original = {key: os.environ.get(key) for key in overrides}
        try:
            os.environ.update(overrides)
            candidate_settings = (
                replace(settings, orb=replace(settings.orb, orb_minutes=minutes))
                if minutes is not None
                else settings
            )
            scenarios: list[dict[str, Any]] = []
            for slippage in (10.0, 25.0, 50.0):
                result = _scenario(
                    histories,
                    instruments,
                    candidate_settings,
                    slippage,
                    components={"ORBPro"},
                    minimum_net_rr=minimum_rr,
                    minimum_opening_rvol=rvol,
                )["strategies"]["ORBPro"]
                scenarios.append({"slippage_bps_per_side": slippage, **result})
            candidates.append(
                {
                    "name": name,
                    "environment_overrides": overrides,
                    "orb_minutes": candidate_settings.orb.orb_minutes,
                    "minimum_net_rr": minimum_rr,
                    "minimum_opening_rvol": rvol,
                    "scenarios": scenarios,
                }
            )
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    ranked = sorted(
        [
            candidate
            for candidate in candidates[1:]
            if all(
                row["development_metrics"]["trade_count"] >= 30
                and row["data_quality"]["development_unresolved_exit_count"] == 0
                for row in candidate["scenarios"]
            )
        ],
        key=lambda candidate: min(
            row["development_metrics"]["expectancy"] for row in candidate["scenarios"]
        ),
        reverse=True,
    )
    return {
        "scope": "bounded_orb_active_contract_component_comparison",
        "evidence_label": "RESEARCH_CANDIDATE",
        "live_equivalent": False,
        "data_sha256": digest,
        "baseline_configuration": asdict(settings.orb),
        "baseline_environment": baseline_environment,
        "candidate_count": len(candidates),
        "slippage_scenario_count": 3,
        "retrospective_check_is_untouched": False,
        "selection": {
            "ranking_rule": (
                "worst_development_expectancy_across_slippage; "
                ">=30 resolved development trades per scenario; "
                "zero unresolved development exits"
            ),
            "development_ranking": [candidate["name"] for candidate in ranked],
            "best_observed_research_candidate": ranked[0]["name"] if ranked else None,
            "minimum_development_trades_for_further_validation": 30,
            "selected_for_live": None,
            "promotion_eligible": False,
            "blockers": [
                "Retrospectively selected active contracts; no historical ATM rotation",
                "Reused archive is not an untouched holdout",
                "Full live pipeline and historical executable quotes unavailable",
                "Prospective chronological and paper validation required",
            ],
        },
        "assumptions": {
            "net_rr_filter": (
                "Research-only feasibility check at next open; "
                "slipped target/stop plus canonical fees; no target repair"
            ),
            "quantity": "one_archived_lot",
            "earliest_signal_bar_start": "09:30 IST",
            "rvol": (
                "Futures opening volume / prior 14 complete "
                "same-window session volumes"
            ),
            "check": "Last 20 percent of option sessions; excluded from ranking",
            "zero_trades": "Abstention, not demonstrated alpha",
        },
        "candidates": candidates,
    }
