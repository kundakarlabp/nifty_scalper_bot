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
from dataclasses import asdict
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from nifty_scalper_bot.config.settings import get_settings
from nifty_scalper_bot.risk.cost_model import estimate_round_trip_cost
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
) -> dict[str, Any]:
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
    rejections: dict[str, Counter[str]] = {}
    evaluated: Counter[str] = Counter()
    pending: dict[tuple[str, str], dict[str, Any]] = {}
    positions: dict[tuple[str, str], dict[str, Any]] = {}
    engine = IndicatorEngine()
    strategies = []
    current_day = None
    last_bars: dict[str, dict[str, Any]] = {}
    slip = slippage_bps / 10000.0

    def close(
        key: tuple[str, str], bar: dict[str, Any], price: float, reason: str
    ) -> None:
        position = positions.pop(key)
        exit_price = max(0.01, price * (1 - slip))
        cost = estimate_round_trip_cost(
            entry_price=position["entry_price"],
            exit_price=exit_price,
            quantity=position["quantity"],
        )
        gross = (exit_price - position["entry_price"]) * position["quantity"]
        exit_time = bar["timestamp"] + timedelta(minutes=1)
        outcomes[key[0]].append(
            {
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
                "duration_minutes": (exit_time - position["entry_time"]).total_seconds()
                / 60,
            }
        )

    for timestamp in timestamps:
        if current_day != timestamp.date():
            for key in list(positions):
                close(
                    key,
                    last_bars[key[1]],
                    last_bars[key[1]]["close"],
                    "session_data_end",
                )
            pending.clear()
            engine = IndicatorEngine()
            strategies = [
                strategy
                for strategy in build_elite_strategies(settings, engine)
                if strategy.name in COMPONENTS
            ]
            if not strategies:
                raise ValueError("research_components_unavailable")
            for strategy in strategies:
                outcomes.setdefault(strategy.name, [])
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
            entry = bar["open"] * (1 + slip)
            if not intent["stop_loss"] < entry < intent["take_profit"]:
                rejections[key[0]]["gap_invalidates_geometry"] += 1
                continue
            positions[key] = {
                "entry_time": timestamp,
                "entry_price": entry,
                "stop_loss": intent["stop_loss"],
                "take_profit": intent["take_profit"],
                "quantity": int(instruments[key[1]]["lot_size"]),
            }
            intent["strategy"].notify_entry_accepted(
                instruments[key[1]]["instrument_type"],
                setup_id=intent["setup_id"],
            )
        for key, position in list(positions.items()):
            bar = bars.get(key[1])
            if bar is None:
                close(key, last_bars[key[1]], last_bars[key[1]]["close"], "history_gap")
                continue
            stop, target = position["stop_loss"], position["take_profit"]
            if bar["open"] <= stop:
                close(key, bar, bar["open"], "gap_stop")
            elif bar["low"] <= stop:
                close(key, bar, stop, "stop")
            elif bar["high"] >= target:
                close(key, bar, target, "target")
            elif timestamp.time() >= time(14, 59):
                close(key, bar, bar["close"], "session_exit")
        for symbol, bar in bars.items():
            engine.ingest_historical_bar(symbol, bar)
            last_bars[symbol] = bar
        if not time(9, 30) <= timestamp.time() < time(14, 59):
            continue
        for symbol in options:
            if symbol not in bars or future not in bars or "NSE:NIFTY 50" not in bars:
                continue
            bar = bars[symbol]
            indicators = dict(engine.get_indicators(symbol))
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
                pending[key] = {
                    "available_at": timestamp + timedelta(minutes=1),
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "strategy": strategy,
                    "setup_id": signal.metadata.get("setup_id"),
                }
    for key in list(positions):
        close(key, last_bars[key[1]], last_bars[key[1]]["close"], "session_data_end")
    days = sorted(
        {ts.date().isoformat() for symbol in options for ts in histories[symbol]}
    )
    cutoff = days[max(0, int(len(days) * 0.8))] if days else ""
    for trades in outcomes.values():
        trades.sort(key=lambda trade: (trade["exit_time"], trade["symbol"]))
    return {
        "slippage_bps_per_side": slippage_bps,
        "strategies": {
            name: {
                "metrics": summarize(trades),
                "chronological_holdout_start": cutoff,
                "holdout_metrics": summarize(
                    [trade for trade in trades if trade["entry_time"][:10] >= cutoff]
                ),
                "evaluations": evaluated[name],
                "no_vote_reasons": dict(rejections[name]),
                "trades": trades,
            }
            for name, trades in outcomes.items()
        },
    }


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
            "ambiguous_stop_target": "stop_first",
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
