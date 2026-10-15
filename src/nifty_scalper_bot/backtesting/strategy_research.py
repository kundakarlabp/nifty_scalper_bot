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
from typing import Any, Callable
from zoneinfo import ZoneInfo

from nifty_scalper_bot.config.settings import get_settings
from nifty_scalper_bot.core.market_regime import MarketRegimeDetector
from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager
from nifty_scalper_bot.core.strategy_manager import (
    StrategyManager as RuntimeStrategyManager,
)
from nifty_scalper_bot.core.underlying_direction import (
    UnderlyingDirectionObservation,
    arbitrate_underlying_direction,
)
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
        "gross_pnl": sum(float(trade["gross_pnl"]) for trade in ordered),
        "gross_expectancy": (
            sum(float(trade["gross_pnl"]) for trade in ordered) / len(values)
            if values
            else None
        ),
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


def _summarize_by_setup(
    trades: list[dict[str, Any]],
    *,
    cutoff: str,
) -> dict[str, dict[str, Any]]:
    """Return post-cost outcome summaries by preserved structural setup subtype."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        setup_name = (
            str(trade.get("setup_name") or trade.get("setup_type") or "unknown").strip()
            or "unknown"
        )
        grouped.setdefault(setup_name, []).append(trade)
    return {
        setup_name: {
            "metrics": summarize(sample),
            "development_metrics": summarize(
                [trade for trade in sample if trade["entry_time"][:10] < cutoff]
            ),
            "holdout_metrics": summarize(
                [trade for trade in sample if trade["entry_time"][:10] >= cutoff]
            ),
            "exit_reasons": dict(Counter(trade["exit_reason"] for trade in sample)),
        }
        for setup_name, sample in sorted(grouped.items())
    }


def _resolve_research_signal_geometry(
    signal: Any,
    current_price: float,
) -> tuple[float | None, float | None, str]:
    """Resolve causal component-research geometry without taking live ownership."""
    try:
        stop = (
            float(signal.stop_loss)
            if getattr(signal, "stop_loss", None) is not None
            else None
        )
        target = (
            float(signal.take_profit)
            if getattr(signal, "take_profit", None) is not None
            else None
        )
    except (TypeError, ValueError):
        stop = target = None
    if stop is not None and target is not None:
        return stop, target, "signal"

    metadata = dict(getattr(signal, "metadata", {}) or {})
    proxy_stop_raw = metadata.get("setup_invalidation_premium")
    target_rr_raw = metadata.get("premium_target_rr")
    if proxy_stop_raw is None or target_rr_raw is None:
        return stop, target, "unavailable"
    try:
        proxy_stop = float(proxy_stop_raw)
        target_rr = float(target_rr_raw)
    except (TypeError, ValueError):
        return stop, target, "unavailable"
    if (
        not math.isfinite(proxy_stop)
        or not math.isfinite(target_rr)
        or not 0 < proxy_stop < current_price
        or target_rr <= 0
    ):
        return stop, target, "unavailable"
    proxy_target = current_price + (current_price - proxy_stop) * target_rr
    if not math.isfinite(proxy_target) or proxy_target <= current_price:
        return stop, target, "unavailable"
    return proxy_stop, proxy_target, "strategy_metadata_proxy"


def _research_direction_payload(
    engine: IndicatorEngine,
    symbol: str,
    *,
    role: str,
) -> tuple[dict[str, Any], UnderlyingDirectionObservation | None]:
    """Build the production direction inputs from completed historical bars."""
    bars = list(engine.get_history(symbol, count=100, field="bars") or [])
    if not bars:
        return {}, None
    current = bars[-1]
    previous = bars[-2] if len(bars) >= 2 else None
    current_ts = current.get("timestamp")
    session_bars = [
        bar
        for bar in bars
        if current_ts is not None
        and bar.get("timestamp") is not None
        and bar["timestamp"].date() == current_ts.date()
    ]
    day_open = (
        float(session_bars[0]["open"])
        if session_bars
        else float(current.get("open") or current.get("close") or 0.0)
    )
    volumes = [
        float(bar.get("volume") or 0.0)
        for bar in bars[-20:]
        if float(bar.get("volume") or 0.0) > 0
    ]
    avg_volume = sum(volumes) / len(volumes) if volumes else None
    current_volume = float(current.get("volume") or 0.0)
    payload: dict[str, Any] = {
        "close": float(current["close"]),
        "ltp": float(current["close"]),
        "open": day_open,
        "day_open": day_open,
        "volume": current_volume,
        "avg_volume": avg_volume,
        "session_vwap": engine.get_session_vwap(symbol),
        "vwap": engine.get_vwap(symbol),
        "vwap_slope": engine.get_session_vwap_slope(symbol, lookback=3),
        "ema_fast": engine.get_ema(symbol, period=9),
        "ema_slow": engine.get_ema(symbol, period=21),
        "ema_50": engine.get_ema(symbol, period=50),
    }
    if previous is not None:
        previous_close = float(previous["close"])
        payload["previous_close"] = previous_close
        payload["recent_ltp_delta"] = float(current["close"]) - previous_close
    if role == "futures_context" and avg_volume and avg_volume > 0:
        payload["futures_volume_ratio"] = current_volume / avg_volume

    direction, confidence, _reasons = (
        RuntimeStrategyManager._derive_context_direction(  # noqa: SLF001
            None,
            payload,
            role=role,
        )
    )
    if direction not in {"CE", "PE"}:
        return payload, None
    return payload, UnderlyingDirectionObservation(
        bias=direction,
        confidence=confidence,
        age_seconds=0.0,
        source=role,
    )


def _research_orb_structural_context(
    engine: IndicatorEngine,
    *,
    spot_symbol: str,
    futures_symbol: str,
) -> dict[str, Any]:
    """Return completed-bar underlying context shared by ORB and VWAP research."""
    spot_payload, spot_observation = _research_direction_payload(
        engine, spot_symbol, role="spot_context"
    )
    futures_payload, futures_observation = _research_direction_payload(
        engine, futures_symbol, role="futures_context"
    )
    resolution = arbitrate_underlying_direction(
        spot_observation,
        futures_observation,
    )
    observation = resolution.observation
    return {
        "underlying_direction_bias": observation.bias if observation else None,
        "direction_bias": observation.bias if observation else None,
        "underlying_direction_confidence": (
            observation.confidence if observation else 0.0
        ),
        "context_fresh": bool(spot_observation or futures_observation),
        # Closed minute bars establish bar-order causality, NOT live tick age.
        "context_age_seconds": 0.0,
        "research_context_age_proxy": "completed_bar_same_step",
        "futures_vwap_slope": futures_payload.get("vwap_slope"),
        "futures_volume_ratio": futures_payload.get("futures_volume_ratio"),
        "research_direction_resolution": resolution.reason,
        "research_spot_direction": (
            spot_observation.bias if spot_observation else None
        ),
        "research_futures_direction": (
            futures_observation.bias if futures_observation else None
        ),
        "research_spot_vwap_slope": spot_payload.get("vwap_slope"),
        "research_futures_vwap_slope": futures_payload.get("vwap_slope"),
    }


def _round_research_tick(price: float, tick_size: float = 0.05) -> float:
    """Round a modeled option stop to the broker tick without importing runtime I/O."""
    return round(round(float(price) / tick_size) * tick_size, 2)


def _apply_bar_lifecycle_proxy(
    position: dict[str, Any],
    bar: dict[str, Any],
    *,
    prior_atr: float | None,
) -> bool:
    """Causally mirror the production long-option trailing tiers on minute bars.

    The current completed bar may advance the MFE watermark and compute a new
    stop, but that stop is only available to the caller for the *next* bar.
    This avoids assuming whether the high or low happened first inside one OHLC
    candle. Production uses executable tick prices; this research proxy therefore
    remains lower-fidelity than recorded-feed runtime replay.
    """
    entry = float(position["entry_price"])
    initial_stop = float(position["initial_stop_loss"])
    current_stop = float(position["stop_loss"])
    quantity = int(position["quantity"])
    initial_risk = entry - initial_stop
    if initial_risk <= 0 or quantity <= 0:
        return False

    high_water = max(float(position.get("high_water", entry)), float(bar["high"]))
    position["high_water"] = high_water
    mfe = max(0.0, high_water - entry)
    mfe_r = mfe / initial_risk
    if mfe_r < 0.60:
        return False

    atr = float(prior_atr or 0.0)
    if not math.isfinite(atr) or atr <= 0:
        atr = entry * 0.02

    if mfe_r < 1.0:
        candidate = entry
    elif mfe_r < 2.0:
        candidate = entry + (mfe * 0.40)
    elif mfe_r < 3.0:
        candidate = max(entry + (mfe * 0.50), high_water - (atr * 1.50))
    else:
        candidate = max(entry + (mfe * 0.60), high_water - atr)

    breakeven_cost = (
        estimate_round_trip_cost(
            entry_price=entry,
            exit_price=entry,
            quantity=quantity,
        ).total
        / quantity
    )
    candidate = max(candidate, entry + breakeven_cost + (initial_risk * 0.10))
    candidate = _round_research_tick(candidate)
    # Production refuses a trail at/through the executable price. Using the bar
    # high as the favorable observation gives the proxy the same geometric guard.
    if candidate <= current_stop or candidate >= high_water:
        return False

    position["stop_loss"] = candidate
    position["trail_updates"] = int(position.get("trail_updates", 0)) + 1
    return True


def _bar_lifecycle_time_stop_due(
    position: dict[str, Any],
    timestamp: datetime,
) -> bool:
    """Mirror the 12-minute / <0.5R progress time stop using prior-bar MFE."""
    held_minutes = (timestamp - position["entry_time"]).total_seconds() / 60.0
    if held_minutes < 12.0:
        return False
    entry = float(position["entry_price"])
    initial_stop = float(position["initial_stop_loss"])
    initial_risk = entry - initial_stop
    if initial_risk <= 0:
        return False
    prior_mfe = max(0.0, float(position.get("high_water", entry)) - entry)
    return (prior_mfe / initial_risk) < 0.50


def _option_premium_confirmation_reason(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> str | None:
    """Causal research-only option confirmation on a completed signal bar.

    No intra-bar price, future bar, synthetic quote or trade outcome is used.
    A missing or untraded preceding minute cannot become a positive vote.
    """
    stamp = current.get("timestamp")
    if (
        previous is None
        or not isinstance(stamp, datetime)
        or not isinstance(previous.get("timestamp"), datetime)
        or previous["timestamp"] != stamp - timedelta(minutes=1)
        or previous["timestamp"].date() != stamp.date()
        or previous.get("is_complete") is not True
        or current.get("is_complete") is not True
        or previous.get("is_provisional")
        or current.get("is_provisional")
        or previous["volume"] <= 0
        or current["volume"] <= 0
    ):
        return "option_confirmation_history_unavailable"
    if current["close"] <= max(current["open"], previous["close"]):
        return "option_premium_not_confirmed"
    return None


def _archived_weekly_atm_rejection(
    instrument: dict[str, Any], timestamp: datetime, spot_close: float
) -> str | None:
    """Exclude plainly non-weekly or off-ATM archived option contracts."""
    try:
        expiry = datetime.fromisoformat(str(instrument["expiry"])).date()
    except (KeyError, TypeError, ValueError):
        return "archive_weekly_expiry_not_proven"
    days = (expiry - timestamp.date()).days
    if not 0 <= days <= 7:
        return "archive_weekly_expiry_not_proven"
    try:
        strike = float(instrument["strike"])
    except (KeyError, TypeError, ValueError):
        return "archive_strike_missing"
    if not (
        math.isfinite(strike)
        and strike > 0
        and math.isfinite(spot_close)
        and spot_close > 0
    ):
        return "archive_strike_missing"
    atm = round(spot_close / 50) * 50
    if abs(strike - atm) > 100:
        return "archive_strike_outside_atm_basket"
    return None


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
    lifecycle_proxy: bool = False,
    require_archived_weekly_atm: bool = False,
    strategy_factory: Callable[[IndicatorEngine], list[Any]] | None = None,
    require_option_premium_confirmation: bool = False,
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
    structural_gate_failures: Counter[str] = Counter()
    evaluated: Counter[str] = Counter()
    entry_funnels: dict[str, Counter[str]] = {}
    orb_replay_regimes: Counter[str] = Counter()
    eligible_option_days: set[str] = set()
    eligible_option_bars = 0
    pending: dict[tuple[str, str], dict[str, Any]] = {}
    positions: dict[tuple[str, str], dict[str, Any]] = {}
    engine = IndicatorEngine()
    regime_detector: MarketRegimeDetector | None = None
    regime_manager: MarketRegimeManager | None = None
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
            "duration_minutes": (exit_time - position["entry_time"]).total_seconds()
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
            if compact_orb_context or os.getenv(
                "VWAP_TREND_MOMENTUM_ENABLED", ""
            ).lower() in {"1", "true", "yes", "on"}:
                # A RANGE ablation without completed-bar regime data is inert.
                # Populate the same causal detector for *all* compact ORB arms
                # so the baseline and experiment receive identical context.
                regime_detector = MarketRegimeDetector()
                regime_manager = MarketRegimeManager(regime_detector)
            else:
                regime_detector = None
                regime_manager = None
            available_strategies = (
                strategy_factory(engine)
                if strategy_factory is not None
                else build_elite_strategies(settings, engine)
            )
            strategies = [
                strategy
                for strategy in available_strategies
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
                entry_funnels.setdefault(strategy.name, Counter())
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
            entry_funnels[key[0]]["accepted_fills"] += 1
            positions[key] = {
                "entry_time": timestamp,
                "entry_price": entry,
                "stop_loss": intent["stop_loss"],
                "initial_stop_loss": intent["stop_loss"],
                "take_profit": intent["take_profit"],
                "quantity": quantity,
                "high_water": entry,
                "trail_updates": 0,
                **intent.get("research_signal_metadata", {}),
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
            initial_stop = position["initial_stop_loss"]
            stop_reason = "trailing_stop" if stop > initial_stop else "stop"
            gap_reason = "gap_trailing_stop" if stop > initial_stop else "gap_stop"
            hit_stop = bar["low"] <= stop
            hit_target = bar["high"] >= target
            if bar["open"] <= stop:
                close(key, bar, bar["open"], gap_reason)
            elif lifecycle_proxy and _bar_lifecycle_time_stop_due(position, timestamp):
                close(key, bar, bar["open"], "time_stop")
            elif hit_stop and hit_target:
                mark_unresolved(
                    key,
                    timestamp,
                    "ambiguous_intrabar_stop_target",
                    stress_price=stop,
                    stress_reason="ambiguous_stop_first_stress",
                )
            elif hit_stop:
                close(key, bar, stop, stop_reason)
            elif hit_target:
                close(key, bar, target, "target")
            elif timestamp.time() >= time(14, 59):
                close(key, bar, bar["close"], "session_exit")
            elif lifecycle_proxy:
                prior_atr = engine.get_atr(key[1])
                _apply_bar_lifecycle_proxy(position, bar, prior_atr=prior_atr)
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
        if future not in bars or "NSE:NIFTY 50" not in bars:
            continue
        eligible_options = [symbol for symbol in options if symbol in bars]
        if require_archived_weekly_atm:
            scoped_options = []
            for symbol in eligible_options:
                reason = _archived_weekly_atm_rejection(
                    instruments[symbol], timestamp, bars["NSE:NIFTY 50"]["close"]
                )
                if reason is not None:
                    for strategy in strategies:
                        rejections[strategy.name][reason] += 1
                else:
                    scoped_options.append(symbol)
            eligible_options = scoped_options
        if not eligible_options:
            continue
        underlying_context = _research_orb_structural_context(
            engine,
            spot_symbol="NSE:NIFTY 50",
            futures_symbol=future,
        )
        if regime_detector is not None and regime_manager is not None:
            # Recompute from *already ingested* completed futures bars only.
            # Use the same detector and stable-state manager as production;
            # historical minute bars remain a lower-fidelity regime proxy.
            regime_detector.evaluate(
                future,
                {
                    "price": bars[future]["close"],
                    "price_series": engine.get_history(future, count=20),
                    "ema_fast": engine.get_ema(future, 12),
                    "ema_slow": engine.get_ema(future, 26),
                    "adx": engine.get_adx(future),
                    "atr": engine.get_atr(future),
                    "volume_ratio": underlying_context.get("futures_volume_ratio"),
                },
            )
            stable_regime = regime_manager.get_latest_snapshot()
            if stable_regime is not None:
                underlying_context["regime"] = stable_regime.regime
        if compact_orb_context:
            regime = underlying_context.get("regime")
            regime_label = str(getattr(regime, "value", regime) or "UNKNOWN")
            orb_replay_regimes[regime_label] += 1
        for symbol in eligible_options:
            bar = bars[symbol]
            if strict_liquidity and bar["volume"] <= 0:
                continue
            eligible_option_days.add(timestamp.date().isoformat())
            eligible_option_bars += 1
            # ORB's only option-derived numeric feature is canonical ATR.
            # Underlying structure still comes from the production strategy's
            # completed-history interface. Other components require full context.
            indicators = (
                {"atr": engine.get_atr(symbol)}
                if compact_orb_context
                else dict(engine.get_indicators(symbol))
            )
            if compact_orb_context:
                indicators.update(underlying_context)
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
                strategy_indicators = dict(indicators)
                if strategy.name in {"ORBPro", "VWAPPro"}:
                    strategy_indicators.update(underlying_context)
                signal = strategy.generate_signal(
                    symbol,
                    strategy_indicators,
                    bar["close"],
                )
                if not strategy.evaluation_health["healthy"]:
                    raise ValueError("research_strategy_evaluation_failed")
                if signal is None:
                    no_vote = getattr(strategy, "last_no_vote_reason", "no_signal")
                    rejections[strategy.name][no_vote] += 1
                    if (
                        strategy.name == "ORBPro"
                        and no_vote == "orb_structural_contract_not_passed"
                    ):
                        for reason in getattr(
                            strategy, "last_structural_failure_reasons", ()
                        ):
                            structural_gate_failures[reason] += 1
                    continue
                entry_funnels[strategy.name]["signals_emitted"] += 1
                metadata = dict(signal.metadata or {})
                stop_loss, take_profit, geometry_source = (
                    _resolve_research_signal_geometry(signal, bar["close"])
                )
                if (
                    signal.action != "BUY"
                    or stop_loss is None
                    or take_profit is None
                    or not 0 < stop_loss < bar["close"] < take_profit
                ):
                    rejections[strategy.name]["invalid_signal_geometry"] += 1
                    continue
                entry_funnels[strategy.name]["geometry_valid"] += 1
                if require_option_premium_confirmation:
                    rejection = _option_premium_confirmation_reason(
                        bar,
                        histories[symbol].get(timestamp - timedelta(minutes=1)),
                    )
                    if rejection is not None:
                        rejections[strategy.name][rejection] += 1
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
                entry_funnels[strategy.name]["next_bar_intents"] += 1
                pending[key] = {
                    "available_at": timestamp + timedelta(minutes=1),
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "strategy": strategy,
                    "setup_id": metadata.get("setup_id"),
                    "research_signal_metadata": {
                        "research_geometry_source": geometry_source,
                        **{
                            field: metadata.get(field)
                            for field in (
                                "setup_id",
                                "setup_name",
                                "setup_type",
                                "signal_family",
                                "setup_reasons",
                                "entry_branch",
                                "retest_confirmed",
                                "underlying_volume_ratio",
                                "underlying_penetration_atr",
                                "opening_range_width_atr",
                                "opening_range_balanced",
                                "underlying_breakout_body_pct",
                                "underlying_direction_confidence",
                                "vwap_event_subtype",
                                "vwap_distance_atr",
                                "vwap_distance_sigma",
                                "futures_vwap_slope_bps",
                                "option_volume_ratio",
                                "futures_volume_ratio",
                                "volume_confirmation_source",
                                "days_to_expiry",
                                "strike_distance_from_atm",
                                "minutes_since_open",
                                "setup_invalidation_premium",
                                "premium_target_rr",
                                "direction_contract",
                                "setup_contract",
                                "confirmation_contract",
                            )
                        },
                    },
                }
    for key in list(positions):
        observed = last_bars.get(key[1])
        mark_unresolved(
            key,
            observed["timestamp"] if observed is not None else timestamps[-1],
            "session_data_end_unresolved",
            stress_reason="unpriced_session_end_stress",
        )
    source_days = sorted(
        {ts.date().isoformat() for symbol in options for ts in histories[symbol]}
    )
    # Holdout sessions must come from the admitted historical option universe.
    source_eligible_days = sorted(eligible_option_days)
    days = source_eligible_days if require_archived_weekly_atm else source_days
    cutoff = days[max(0, int(len(days) * 0.8))] if days else ""
    for trades in outcomes.values():
        trades.sort(key=lambda trade: (trade["exit_time"], trade["symbol"]))
    for trades in stress_outcomes.values():
        trades.sort(key=lambda trade: (trade["exit_time"], trade["symbol"]))
    return {
        "slippage_bps_per_side": slippage_bps,
        "archive_weekly_atm_proxy_enabled": require_archived_weekly_atm,
        "orb_replay_regimes": dict(orb_replay_regimes),
        "strategies": {
            name: {
                "metrics": summarize(trades),
                "worst_case_stress_metrics": summarize(stress_outcomes[name]),
                "chronological_holdout_start": cutoff,
                "development_metrics": summarize(
                    [trade for trade in trades if trade["entry_time"][:10] < cutoff]
                ),
                "setup_metrics": _summarize_by_setup(trades, cutoff=cutoff),
                "exit_reasons": dict(Counter(trade["exit_reason"] for trade in trades)),
                "worst_case_stress_exit_reasons": dict(
                    Counter(trade["exit_reason"] for trade in stress_outcomes[name])
                ),
                "data_quality": {
                    "source_option_session_count": len(source_days),
                    "eligible_session_count": len(eligible_option_days),
                    "eligible_session_dates": sorted(eligible_option_days),
                    "eligible_option_bar_count": eligible_option_bars,
                    "development_eligible_session_count": sum(
                        day < cutoff for day in days
                    ),
                    "holdout_eligible_session_count": sum(
                        day >= cutoff for day in days
                    ),
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
                "entry_funnel": {
                    "evaluations": evaluated[name],
                    "signals_emitted": entry_funnels[name]["signals_emitted"],
                    "geometry_valid": entry_funnels[name]["geometry_valid"],
                    "next_bar_intents": entry_funnels[name]["next_bar_intents"],
                    "accepted_fills": entry_funnels[name]["accepted_fills"],
                },
                "no_vote_reasons": dict(rejections[name]),
                "structural_gate_failures": (
                    dict(structural_gate_failures) if name == "ORBPro" else {}
                ),
                "research_gate_counts": dict(
                    getattr(
                        next(
                            strategy for strategy in strategies if strategy.name == name
                        ),
                        "research_gate_counts",
                        {},
                    )
                ),
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
    lifecycle_proxy: bool = False,
    normalize_opening_range: bool = False,
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
        research_factory: Callable[[IndicatorEngine], list[Any]] | None = None
        if normalize_opening_range:
            from nifty_scalper_bot.backtesting.directional_hypotheses import (
                ORBRangeScaledResearch,
            )

            def _make_scaled(engine: IndicatorEngine) -> list[Any]:
                return [ORBRangeScaledResearch(get_settings().elite.orb, engine)]

            research_factory = _make_scaled

        result = _scenario(
            histories,
            instruments,
            get_settings().elite,
            slippage_bps,
            components={"ORBPro"},
            minimum_net_rr=minimum_net_rr,
            compact_orb_context=True,
            strict_liquidity=True,
            lifecycle_proxy=lifecycle_proxy,
            strategy_factory=research_factory,
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


def run_vwap_comparison(directory: Path) -> dict[str, Any]:
    """Compare a small prespecified VWAP hypothesis set without live promotion."""
    if os.getenv("EXECUTION_MODE", "SHADOW").upper() != "SHADOW":
        raise ValueError("research_requires_shadow_process")
    histories, instruments, digest = load_archive(directory)
    settings = get_settings().elite
    variants: list[tuple[str, dict[str, str]]] = [
        ("current_reference", {}),
        ("distance_2_atr", {"VWAP_TREND_QUALITY_MAX_DISTANCE_ATR": "2.0"}),
        ("legacy_distance_5_atr", {"VWAP_TREND_QUALITY_MAX_DISTANCE_ATR": "5.0"}),
        ("slope_floor_0_5bp", {"VWAP_FUTURES_SLOPE_MIN_BPS": "0.5"}),
        ("slope_floor_2bp", {"VWAP_FUTURES_SLOPE_MIN_BPS": "2.0"}),
        ("futures_rvol_1_2", {"VWAP_FUTURES_VOLUME_MIN_RATIO": "1.2"}),
        ("penetration_opt_in", {"VWAP_ALLOW_PENETRATION_ONLY_ENTRY": "true"}),
        (
            "trend_premium_rvol_1_5",
            {
                "VWAP_TREND_MOMENTUM_ENABLED": "true",
                "VWAP_TREND_MOMENTUM_OPTION_RVOL_MIN": "1.5",
            },
        ),
        (
            "trend_premium_vwap_retest",
            {
                "VWAP_TREND_MOMENTUM_ENABLED": "true",
                "VWAP_TREND_MOMENTUM_OPTION_RVOL_MIN": "1.5",
                "VWAP_TREND_REQUIRE_RETEST": "true",
            },
        ),
    ]
    environment_keys = (
        "VWAP_TREND_QUALITY_MAX_DISTANCE_ATR",
        "VWAP_FUTURES_SLOPE_MIN_BPS",
        "VWAP_FUTURES_VOLUME_MIN_RATIO",
        "VWAP_ALLOW_PENETRATION_ONLY_ENTRY",
        "VWAP_TREND_MOMENTUM_ENABLED",
        "VWAP_TREND_MOMENTUM_OPTION_RVOL_MIN",
        "VWAP_TREND_REQUIRE_RETEST",
    )
    baseline_environment = {key: os.environ.get(key) for key in environment_keys}
    candidates: list[dict[str, Any]] = []
    for name, overrides in variants:
        # Keep the baseline independent of any inherited experimental toggle.
        keys = {
            "VWAP_TREND_MOMENTUM_ENABLED",
            "VWAP_TREND_REQUIRE_RETEST",
            *overrides,
        }
        original = {key: os.environ.get(key) for key in keys}
        try:
            os.environ["VWAP_TREND_MOMENTUM_ENABLED"] = "false"
            os.environ["VWAP_TREND_REQUIRE_RETEST"] = "false"
            os.environ.update(overrides)
            scenarios: list[dict[str, Any]] = []
            for slippage in (10.0, 25.0, 50.0):
                result = _scenario(
                    histories,
                    instruments,
                    settings,
                    slippage,
                    components={"VWAPPro"},
                    minimum_net_rr=1.5,
                    strict_liquidity=True,
                    lifecycle_proxy=True,
                    require_archived_weekly_atm=True,
                )["strategies"]["VWAPPro"]
                scenarios.append({"slippage_bps_per_side": slippage, **result})
            candidates.append(
                {
                    "name": name,
                    "environment_overrides": overrides,
                    "minimum_net_rr": 1.5,
                    "lifecycle_proxy": True,
                    "scenarios": scenarios,
                }
            )
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    eligible = [
        candidate
        for candidate in candidates
        if all(
            row["development_metrics"]["trade_count"] >= 30
            and row["data_quality"]["development_unresolved_exit_count"] == 0
            for row in candidate["scenarios"]
        )
    ]
    ranked = sorted(
        eligible,
        key=lambda candidate: min(
            row["development_metrics"]["expectancy"]
            for row in candidate["scenarios"]
            if row["development_metrics"]["expectancy"] is not None
        ),
        reverse=True,
    )
    return {
        "scope": "bounded_vwap_active_contract_component_comparison",
        "evidence_label": "RESEARCH_CANDIDATE",
        "live_equivalent": False,
        "data_sha256": digest,
        "baseline_configuration": asdict(settings.vwap),
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
                "Archived basket incomplete; weekly near-ATM proximity only",
                "Reused archive is not an untouched holdout",
                "Historical bid/ask depth and full live arbitration unavailable",
                "Bar-close context age is a proxy, not verified live tick freshness",
                "Prospective chronological and paper validation required",
            ],
        },
        "assumptions": {
            "geometry": (
                "Explicit signal stop/target when available; otherwise existing "
                "strategy structural invalidation and target-RR metadata"
            ),
            "entry": "next_minute_open",
            "net_rr_filter": "minimum 1.5 after modeled slippage and canonical fees",
            "lifecycle": "existing causal 0.6R minute-bar trailing/time-stop proxy",
            "quantity": "one_archived_lot",
            "historical_eligibility": (
                "Research-only expiry 0-7 days, strike within two 50-point steps "
                "of historical completed spot close. Missing archived contracts "
                "and intrabar ATM rotation cannot be recovered."
            ),
            "check": "Last 20 percent of option sessions; excluded from ranking",
            "zero_trades": "Abstention, not demonstrated alpha",
        },
        "candidates": candidates,
    }


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
        (
            "range_transition_research",
            {"ORB_RESEARCH_ALLOW_RANGE_TRANSITION": "true"},
            None,
            1.5,
            None,
        ),
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
        "ORB_RESEARCH_ALLOW_RANGE_TRANSITION",
        "ORB_MOMENTUM_BRANCH_ENABLED",
        "ORB_MAX_ENTRY_MINUTES_AFTER_RANGE",
        "ORB_TARGET_RR",
        "ORB_MOMENTUM_MIN_BODY_PCT",
        "ORB_MOMENTUM_MIN_PENETRATION_ATR",
        "ORB_MOMENTUM_MIN_VOLUME_RATIO",
    )
    baseline_environment = {key: os.environ.get(key) for key in environment_keys}
    candidates: list[dict[str, Any]] = []
    for name, overrides, minutes, minimum_rr, rvol in variants:
        # Keep every reference arm free of experimental overrides, even if
        # the calling research process has the feature flag in its environment.
        original = {
            key: os.environ.get(key)
            for key in {"ORB_RESEARCH_ALLOW_RANGE_TRANSITION", *overrides}
        }
        try:
            os.environ["ORB_RESEARCH_ALLOW_RANGE_TRANSITION"] = "false"
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
                    compact_orb_context=True,
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


def run_entry_confirmation_comparison(directory: Path) -> dict[str, Any]:
    """Pre-registered, SHADOW-only admission/confirmation ablation for all three.

    Keep option contracts, timestamps, slippage and strategy classes identical
    across arms. This does not infer live quote depth, choose a winning
    configuration, or change production entry policy.
    """
    if os.getenv("EXECUTION_MODE", "SHADOW").upper() != "SHADOW":
        raise ValueError("research_requires_shadow_process")
    histories, instruments, digest = load_archive(directory)
    settings = get_settings().elite
    execution_proxy = {
        "minimum_net_rr": 1.5,
        "strict_liquidity": True,
        "lifecycle_proxy": True,
        "require_archived_weekly_atm": True,
    }
    variants: list[tuple[str, dict[str, Any]]] = [
        ("current_component_reference", {}),
        ("execution_feasibility_proxy", execution_proxy),
        (
            "execution_plus_completed_premium_confirmation",
            {**execution_proxy, "require_option_premium_confirmation": True},
        ),
    ]
    candidates: list[dict[str, Any]] = []
    for name, constraints in variants:
        scenarios = []
        for slippage in (10.0, 25.0, 50.0):
            report = _scenario(
                histories,
                instruments,
                settings,
                slippage,
                components=COMPONENTS,
                **constraints,
            )
            # The per-entry ledger remains in the saved component baseline;
            # status summaries contain metrics and funnel diagnostics only.
            summaries = {}
            for strategy_name, result in report["strategies"].items():
                summaries[strategy_name] = {
                    key: value
                    for key, value in result.items()
                    if key
                    not in {
                        "trades",
                        "unresolved_trades",
                        "worst_case_stress_trades",
                    }
                }
            scenarios.append(
                {
                    "slippage_bps_per_side": slippage,
                    "strategies": summaries,
                }
            )
        candidates.append(
            {
                "name": name,
                "research_only_constraints": constraints,
                "scenarios": scenarios,
            }
        )
    return {
        "scope": "paired_three_strategy_entry_confirmation_component_ablation",
        "evidence_label": "RESEARCH_CANDIDATE",
        "live_equivalent": False,
        "data_sha256": digest,
        "baseline_configuration": {
            name: asdict(getattr(settings, name)) for name in ("orb", "smc", "vwap")
        },
        "candidate_count": len(candidates),
        "slippage_scenario_count": 3,
        "hypothesis": (
            "Among opportunities passing the same modeled execution proxy, "
            "an additional completed-option-bar premium follow-through check "
            "may improve net expectancy, at a cost in accepted trades."
        ),
        "comparison_rule": (
            "Compare execution_feasibility_proxy against "
            "execution_plus_completed_premium_confirmation at each same "
            "slippage level; current_component_reference explains the "
            "different admission funnel, not incremental confirmation alpha."
        ),
        "selection": {
            "selected_for_live": None,
            "promotion_eligible": False,
            "retrospective_check_is_untouched": False,
            "minimum_development_trades_for_further_validation": 30,
            "blockers": [
                "Active-contract archive is not causal historical ATM rotation",
                "Weekly/ATM scope is a research proximity proxy, not contract parity",
                "Historical executable bid/ask depth and full live arbitration missing",
                "Research dates reused; no untouched chronological test",
                "Prospective recorded-feed and paper validation still required",
            ],
        },
        "assumptions": {
            "signal": "actual ORBPro, SMC and VWAPPro classes",
            "premium_confirmation": (
                "Current and preceding contiguous complete traded minute bars; "
                "current premium close > max(current open, previous close)"
            ),
            "entry": "next_minute_open_after_complete_signal_bar",
            "fill_cost_stress": "10/25/50 bps per side and canonical modeled fees",
            "eligibility": (
                "Research-only weekly DTE 0-7 and <=100 points from completed "
                "spot-close ATM; never a reconstructed historical selection"
            ),
            "unresolved_exits": "excluded from primary outcomes; separate stress",
        },
        "candidates": candidates,
    }
