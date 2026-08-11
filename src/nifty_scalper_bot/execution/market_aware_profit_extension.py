"""Market-aware profit protection and winner extension for canonical brackets.

The supervisor is deliberately subordinate to the canonical bracket lifecycle:
- hard SL, TP1, time-stop, broker truth, and exit reconciliation stay authoritative;
- market context may only tighten an existing stop or defer/extend FINAL_TP;
- extension is allowed only after the trade already has a protected profit floor;
- missing/stale/contradictory evidence fails closed to the existing FINAL_TP.

No broker calls are introduced.  Evidence is read from the already-wired
IndicatorEngine and market-data facade/cache.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
import logging
import math
import os
import time
from typing import Any, Callable

from nifty_scalper_bot.core.active_basket import active_contract_selection_from_basket

LOGGER = logging.getLogger("nifty_scalper_bot.execution.market_aware_profit_extension")
_TICK_SIZE = 0.05

_OPTION_INDICATORS = (
    "ltp",
    "price",
    "close",
    "exchange_vwap",
    "session_vwap",
    "vwap",
    "ema_fast",
    "ema_9",
    "ema_slow",
    "ema_21",
    "adx",
    "atr",
    "volume",
    "avg_volume",
    "volume_ratio",
)
_CONTEXT_INDICATORS = (
    *_OPTION_INDICATORS,
    "futures_volume_ratio",
    "direction_bias",
    "underlying_direction_bias",
    "underlying_direction_confidence",
    "direction_confidence",
    "regime",
    "market_regime",
    "context_age_seconds",
)


@dataclass(frozen=True, slots=True)
class ContinuationDecision:
    """Read-only continuation assessment for one open option position."""

    extend: bool
    score: float
    evidence_count: int
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    snapshot: Mapping[str, Any]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive(value: Any) -> float | None:
    parsed = _as_float(value)
    return parsed if parsed is not None and parsed > 0 else None


def _round_tick(value: float) -> float:
    return round(round(float(value) / _TICK_SIZE) * _TICK_SIZE, 2)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    items = getattr(value, "items", None)
    if callable(items):
        with suppress(Exception):
            return dict(items())
    return {}


def _quote(manager: Any, symbol: str) -> dict[str, Any]:
    source = getattr(manager, "_market_data", None)
    if source is None:
        return {}
    for name in ("get_quote", "get_latest_tick", "get_tick", "get_ltp_snapshot"):
        getter = getattr(source, name, None)
        if not callable(getter):
            continue
        try:
            value = getter(symbol, allow_pull=False) if name == "get_quote" else getter(symbol)
        except TypeError:
            with suppress(Exception):
                value = getter(symbol)
                mapped = _mapping(value)
                if mapped:
                    return mapped
            continue
        except Exception:
            continue
        mapped = _mapping(value)
        if mapped:
            return mapped
    return {}


def _indicator_snapshot(manager: Any, symbol: str, names: Sequence[str]) -> dict[str, Any]:
    engine = getattr(manager, "_indicator_engine", None)
    getter = getattr(engine, "get_indicators", None)
    if not callable(getter):
        return {}
    try:
        return _mapping(getter(symbol, list(names)))
    except Exception as exc:  # noqa: BLE001 - missing context must fail closed
        LOGGER.debug("PROFIT_CONTEXT_INDICATORS_UNAVAILABLE symbol=%s error=%s", symbol, exc)
        return {}


def _field(payload: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = _positive(payload.get(name))
        if value is not None:
            return value
    return None


def _raw_float(payload: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = _as_float(payload.get(name))
        if value is not None:
            return value
    return None


def _active_basket(manager: Any) -> Any | None:
    source = getattr(manager, "_market_data", None)
    candidates = [source, getattr(source, "market_data_manager", None), getattr(source, "_mdm", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        getter = getattr(candidate, "get_active_contract_basket", None)
        if callable(getter):
            with suppress(Exception):
                basket = getter()
                if basket is not None:
                    return basket
        for attr in ("_active_contract_basket", "active_contract_basket"):
            basket = getattr(candidate, attr, None)
            if basket is not None:
                return basket
    return None


def _selection(manager: Any):
    return active_contract_selection_from_basket(_active_basket(manager))


def _market_metric(manager: Any, symbol: str, kind: str, quote: Mapping[str, Any]) -> float | None:
    source = getattr(manager, "_market_data", None)
    method_names = {
        "oi": ("get_oi",),
        "iv": ("get_iv", "get_implied_volatility"),
    }
    for method_name in method_names.get(kind, ()):
        getter = getattr(source, method_name, None)
        if callable(getter):
            with suppress(Exception):
                value = _positive(getter(symbol))
                if value is not None:
                    return value
    aliases = {
        "oi": ("oi", "open_interest"),
        "iv": ("iv", "implied_volatility"),
    }
    return _field(quote, *aliases.get(kind, (kind,)))


def _greeks(manager: Any, symbol: str) -> dict[str, Any]:
    source = getattr(manager, "_market_data", None)
    getter = getattr(source, "get_greeks", None)
    if callable(getter):
        with suppress(Exception):
            return _mapping(getter(symbol))
    return {}


def _chain_pcr(manager: Any) -> float | None:
    selection = _selection(manager)
    symbols = list(selection.option_symbols or ())[:24]
    if not symbols:
        for symbol in (selection.selected_ce, selection.selected_pe):
            if symbol:
                symbols.append(symbol)
    ce_oi = 0.0
    pe_oi = 0.0
    for symbol in symbols:
        normalized = str(symbol or "").upper()
        if not normalized.endswith(("CE", "PE")):
            continue
        quote = _quote(manager, normalized)
        oi = _market_metric(manager, normalized, "oi", quote)
        if oi is None:
            continue
        if normalized.endswith("CE"):
            ce_oi += oi
        else:
            pe_oi += oi
    if ce_oi <= 0 or pe_oi <= 0:
        return None
    return pe_oi / ce_oi


def _quote_age_seconds(quote: Mapping[str, Any]) -> float | None:
    raw = quote.get("exchange_timestamp") or quote.get("timestamp") or quote.get("received_at")
    if raw is None:
        return None
    try:
        if isinstance(raw, datetime):
            stamp = raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
            return max(0.0, (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds())
        if isinstance(raw, (int, float)):
            seconds = float(raw) / 1000.0 if float(raw) > 1e12 else float(raw)
            return max(0.0, time.time() - seconds)
        stamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OSError):
        return None


def _spread_pct(quote: Mapping[str, Any]) -> float | None:
    bid = _field(quote, "bid", "best_bid", "bid_price")
    ask = _field(quote, "ask", "best_ask", "ask_price")
    if bid is None or ask is None or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 100.0 if mid > 0 else None


def _option_momentum(manager: Any, symbol: str) -> float | None:
    ticks = getattr(manager, "_recent_ticks", {}).get(symbol)
    if ticks is None or len(ticks) < 5:
        return None
    try:
        first = float(ticks[-5])
        last = float(ticks[-1])
    except (TypeError, ValueError, IndexError):
        return None
    if first <= 0:
        return None
    return ((last - first) / first) * 100.0


def _side_alignment(option_symbol: str, price: float | None, reference: float | None) -> int:
    if price is None or reference is None:
        return 0
    bullish = str(option_symbol).upper().endswith("CE")
    if bullish:
        return 1 if price > reference else -1 if price < reference else 0
    return 1 if price < reference else -1 if price > reference else 0


def _ema_alignment(option_symbol: str, payload: Mapping[str, Any]) -> int:
    fast = _field(payload, "ema_fast", "ema_9", "ema9")
    slow = _field(payload, "ema_slow", "ema_21", "ema21")
    if fast is None or slow is None:
        return 0
    bullish = str(option_symbol).upper().endswith("CE")
    if bullish:
        return 1 if fast > slow else -1 if fast < slow else 0
    return 1 if fast < slow else -1 if fast > slow else 0


def _add_vote(
    label: str,
    value: float,
    *,
    positive: list[str],
    negative: list[str],
    score_box: list[float],
) -> None:
    score_box[0] += float(value)
    if value > 0:
        positive.append(label)
    elif value < 0:
        negative.append(label)


def capture_entry_market_baseline(manager: Any, bracket: Any) -> bool:
    """Persist entry-time OI/IV/volume/chain baselines for later comparisons."""
    provenance = getattr(bracket, "trade_provenance", None)
    if not isinstance(provenance, dict):
        provenance = dict(provenance or {}) if isinstance(provenance, Mapping) else {}
        bracket.trade_provenance = provenance
    if provenance.get("profit_extension_baseline_captured"):
        return False

    quote = _quote(manager, str(bracket.symbol))
    indicators = _indicator_snapshot(manager, str(bracket.symbol), _OPTION_INDICATORS)
    oi = _market_metric(manager, str(bracket.symbol), "oi", quote)
    iv = _market_metric(manager, str(bracket.symbol), "iv", quote)
    volume = _field(quote, "volume", "volume_traded") or _field(indicators, "volume")
    chain_pcr = _chain_pcr(manager)
    greeks = _greeks(manager, str(bracket.symbol))

    if oi is not None:
        provenance["profit_extension_entry_oi"] = oi
    if iv is not None:
        provenance["profit_extension_entry_iv"] = iv
    if volume is not None:
        provenance["profit_extension_entry_volume"] = volume
    if chain_pcr is not None:
        provenance["profit_extension_entry_chain_pcr"] = chain_pcr
    delta = _as_float(greeks.get("delta"))
    if delta is not None:
        provenance["profit_extension_entry_delta"] = delta
    provenance["profit_extension_baseline_captured"] = True
    provenance["profit_extension_policy_version"] = 1
    return True


def assess_continuation(manager: Any, bracket: Any, ltp: float) -> ContinuationDecision:
    """Score live continuation evidence without changing bracket state."""
    if not _env_bool("MARKET_AWARE_PROFIT_EXTENSION_ENABLED", True):
        return ContinuationDecision(False, 0.0, 0, (), (), {"reason": "disabled"})
    if str(getattr(bracket, "side", "BUY") or "BUY").upper() != "BUY":
        return ContinuationDecision(False, 0.0, 0, (), (), {"reason": "non_long_option"})
    symbol = str(getattr(bracket, "symbol", "") or "").upper()
    if not symbol.endswith(("CE", "PE")):
        return ContinuationDecision(False, 0.0, 0, (), (), {"reason": "non_option"})

    quote = _quote(manager, symbol)
    option = _indicator_snapshot(manager, symbol, _OPTION_INDICATORS)
    selection = _selection(manager)
    spot_symbol = "NSE:NIFTY"
    future_symbol = str(selection.futures_symbol or "")
    spot = _indicator_snapshot(manager, spot_symbol, _CONTEXT_INDICATORS)
    future = (
        _indicator_snapshot(manager, future_symbol, _CONTEXT_INDICATORS)
        if future_symbol
        else {}
    )
    spot_quote = _quote(manager, spot_symbol)
    future_quote = _quote(manager, future_symbol) if future_symbol else {}

    positive: list[str] = []
    negative: list[str] = []
    score = [0.0]
    evidence_count = 0
    critical_block = False

    momentum = _option_momentum(manager, symbol)
    if momentum is not None:
        evidence_count += 1
        if momentum >= 0.20:
            _add_vote("premium_momentum", 1.5, positive=positive, negative=negative, score_box=score)
        elif momentum <= -0.15:
            _add_vote("premium_momentum_reversal", -1.5, positive=positive, negative=negative, score_box=score)

    premium_vwap = _field(option, "exchange_vwap", "session_vwap", "vwap")
    if premium_vwap is not None:
        evidence_count += 1
        _add_vote(
            "premium_vwap_hold" if ltp >= premium_vwap else "premium_vwap_lost",
            1.0 if ltp >= premium_vwap else -1.0,
            positive=positive,
            negative=negative,
            score_box=score,
        )

    fast = _field(option, "ema_fast", "ema_9", "ema9")
    slow = _field(option, "ema_slow", "ema_21", "ema21")
    if fast is not None and slow is not None:
        evidence_count += 1
        _add_vote(
            "premium_ema_support" if fast >= slow else "premium_ema_weak",
            1.0 if fast >= slow else -1.0,
            positive=positive,
            negative=negative,
            score_box=score,
        )

    adx = _raw_float(option, "adx")
    if adx is not None:
        evidence_count += 1
        if adx >= 20.0:
            _add_vote("premium_adx", 0.5, positive=positive, negative=negative, score_box=score)
        elif adx < 15.0:
            _add_vote("premium_adx_weak", -0.5, positive=positive, negative=negative, score_box=score)

    volume = _field(quote, "volume", "volume_traded") or _field(option, "volume")
    avg_volume = _field(option, "avg_volume")
    volume_ratio = _raw_float(option, "volume_ratio")
    provenance = getattr(bracket, "trade_provenance", {})
    provenance = provenance if isinstance(provenance, Mapping) else {}
    entry_volume = _positive(provenance.get("profit_extension_entry_volume"))
    if volume_ratio is None and volume is not None:
        reference = avg_volume or entry_volume
        if reference is not None and reference > 0:
            volume_ratio = volume / reference
    if volume_ratio is not None:
        evidence_count += 1
        if volume_ratio >= 1.20:
            _add_vote("premium_volume_expansion", 1.0, positive=positive, negative=negative, score_box=score)
        elif volume_ratio <= 0.70:
            _add_vote("premium_volume_fade", -0.5, positive=positive, negative=negative, score_box=score)

    current_oi = _market_metric(manager, symbol, "oi", quote)
    entry_oi = _positive(provenance.get("profit_extension_entry_oi"))
    oi_change_pct = None
    if current_oi is not None and entry_oi is not None:
        evidence_count += 1
        oi_change_pct = ((current_oi - entry_oi) / entry_oi) * 100.0
        if oi_change_pct >= 5.0 and (momentum is None or momentum >= 0):
            _add_vote("option_oi_build_with_price", 0.5, positive=positive, negative=negative, score_box=score)

    current_iv = _market_metric(manager, symbol, "iv", quote)
    entry_iv = _positive(provenance.get("profit_extension_entry_iv"))
    iv_change_pct = None
    if current_iv is not None and entry_iv is not None:
        evidence_count += 1
        iv_change_pct = ((current_iv - entry_iv) / entry_iv) * 100.0
        if iv_change_pct >= 3.0:
            _add_vote("iv_support", 0.5, positive=positive, negative=negative, score_box=score)
        elif iv_change_pct <= -5.0:
            _add_vote("iv_crush", -0.5, positive=positive, negative=negative, score_box=score)

    greeks = _greeks(manager, symbol)
    delta = _as_float(greeks.get("delta"))
    if delta is not None:
        evidence_count += 1
        if 0.35 <= abs(delta) <= 0.85:
            _add_vote("responsive_delta", 0.25, positive=positive, negative=negative, score_box=score)

    spread_pct = _spread_pct(quote)
    if spread_pct is not None:
        evidence_count += 1
        if spread_pct <= 1.0:
            _add_vote("executable_spread", 0.5, positive=positive, negative=negative, score_box=score)
        elif spread_pct > _env_float("PROFIT_EXTENSION_MAX_SPREAD_PCT", 1.5):
            _add_vote("wide_spread", -2.0, positive=positive, negative=negative, score_box=score)
            critical_block = True

    quote_age = _quote_age_seconds(quote)
    if quote_age is not None:
        evidence_count += 1
        if quote_age > _env_float("PROFIT_EXTENSION_QUOTE_MAX_AGE_SEC", 3.0):
            _add_vote("held_quote_stale", -3.0, positive=positive, negative=negative, score_box=score)
            critical_block = True

    context_votes = 0
    context_available = 0
    for payload, live_quote in ((spot, spot_quote), (future, future_quote)):
        if not payload and not live_quote:
            continue
        price = _field(live_quote, "ltp", "last_price", "price") or _field(payload, "ltp", "price", "close")
        vwap = _field(payload, "exchange_vwap", "session_vwap", "vwap")
        alignment = _side_alignment(symbol, price, vwap)
        if price is not None and vwap is not None:
            context_available += 1
            context_votes += alignment
    if context_available:
        evidence_count += 1
        if context_votes == context_available:
            _add_vote("underlying_vwap_alignment", 1.5, positive=positive, negative=negative, score_box=score)
        elif context_votes <= -context_available:
            _add_vote("underlying_vwap_conflict", -1.5, positive=positive, negative=negative, score_box=score)

    ema_votes = [_ema_alignment(symbol, payload) for payload in (spot, future) if payload]
    ema_votes = [vote for vote in ema_votes if vote != 0]
    if ema_votes:
        evidence_count += 1
        if all(vote > 0 for vote in ema_votes):
            _add_vote("underlying_ema_alignment", 1.5, positive=positive, negative=negative, score_box=score)
        elif all(vote < 0 for vote in ema_votes):
            _add_vote("underlying_ema_conflict", -1.5, positive=positive, negative=negative, score_box=score)

    futures_volume_ratio = _raw_float(future, "futures_volume_ratio", "volume_ratio")
    if futures_volume_ratio is None:
        fut_volume = _field(future, "volume")
        fut_avg = _field(future, "avg_volume")
        if fut_volume is not None and fut_avg is not None and fut_avg > 0:
            futures_volume_ratio = fut_volume / fut_avg
    if futures_volume_ratio is not None:
        evidence_count += 1
        if futures_volume_ratio >= 1.10:
            _add_vote("futures_volume_confirmation", 0.75, positive=positive, negative=negative, score_box=score)
        elif futures_volume_ratio <= 0.80:
            _add_vote("futures_volume_fade", -0.5, positive=positive, negative=negative, score_box=score)

    direction_bias = str(
        future.get("direction_bias")
        or future.get("underlying_direction_bias")
        or spot.get("direction_bias")
        or spot.get("underlying_direction_bias")
        or ""
    ).upper()
    direction_conf = _raw_float(
        future,
        "underlying_direction_confidence",
        "direction_confidence",
    ) or _raw_float(spot, "underlying_direction_confidence", "direction_confidence")
    if direction_bias in {"CE", "PE"} and direction_conf is not None and direction_conf >= 0.55:
        evidence_count += 1
        if direction_bias == symbol[-2:]:
            _add_vote("underlying_direction_context", 1.0, positive=positive, negative=negative, score_box=score)
        else:
            _add_vote("underlying_direction_conflict", -2.0, positive=positive, negative=negative, score_box=score)
            critical_block = True

    regime = str(
        future.get("regime")
        or future.get("market_regime")
        or spot.get("regime")
        or spot.get("market_regime")
        or ""
    ).upper()
    if regime:
        evidence_count += 1
        aligned_regime = "TREND_UP" if symbol.endswith("CE") else "TREND_DOWN"
        if regime == aligned_regime:
            _add_vote("trend_regime", 1.0, positive=positive, negative=negative, score_box=score)
        elif regime in {"RANGE", "CHOPPY", "LOW_VOLATILITY"}:
            _add_vote("nontrend_regime", -0.75, positive=positive, negative=negative, score_box=score)

    current_pcr = _chain_pcr(manager)
    entry_pcr = _positive(provenance.get("profit_extension_entry_chain_pcr"))
    pcr_change_pct = None
    if current_pcr is not None:
        evidence_count += 1
        if entry_pcr is not None:
            pcr_change_pct = ((current_pcr - entry_pcr) / entry_pcr) * 100.0
        if symbol.endswith("CE") and (
            current_pcr >= 1.05 or (pcr_change_pct is not None and pcr_change_pct >= 3.0)
        ):
            _add_vote("chain_put_support", 0.5, positive=positive, negative=negative, score_box=score)
        elif symbol.endswith("PE") and (
            current_pcr <= 0.95 or (pcr_change_pct is not None and pcr_change_pct <= -3.0)
        ):
            _add_vote("chain_call_pressure", 0.5, positive=positive, negative=negative, score_box=score)

    threshold = _env_float("MARKET_AWARE_PROFIT_EXTENSION_SCORE", 5.0)
    min_evidence = max(1, int(_env_float("MARKET_AWARE_PROFIT_MIN_EVIDENCE", 4.0)))
    should_extend = (
        not critical_block
        and evidence_count >= min_evidence
        and score[0] >= threshold
    )
    snapshot = {
        "momentum_pct": momentum,
        "volume_ratio": volume_ratio,
        "oi_change_pct": oi_change_pct,
        "iv_change_pct": iv_change_pct,
        "spread_pct": spread_pct,
        "quote_age_s": quote_age,
        "futures_volume_ratio": futures_volume_ratio,
        "regime": regime or None,
        "direction_bias": direction_bias or None,
        "direction_confidence": direction_conf,
        "chain_pcr": current_pcr,
        "chain_pcr_change_pct": pcr_change_pct,
        "future_symbol": future_symbol or None,
    }
    return ContinuationDecision(
        extend=should_extend,
        score=round(score[0], 4),
        evidence_count=evidence_count,
        positive=tuple(positive),
        negative=tuple(negative),
        snapshot=snapshot,
    )


def _cached_assessment(manager: Any, bracket: Any, ltp: float) -> ContinuationDecision:
    now = time.monotonic()
    cache_seconds = max(0.0, _env_float("PROFIT_EXTENSION_CONTEXT_CACHE_SEC", 0.75))
    previous_at = _as_float(getattr(bracket, "_profit_extension_assessed_mono", None))
    previous = getattr(bracket, "_profit_extension_decision", None)
    previous_ltp = _as_float(getattr(bracket, "_profit_extension_assessed_ltp", None))
    if (
        previous_at is not None
        and isinstance(previous, ContinuationDecision)
        and (now - previous_at) <= cache_seconds
        and previous_ltp is not None
        and abs(previous_ltp - ltp) <= max(_TICK_SIZE, abs(ltp) * 0.002)
    ):
        return previous
    decision = assess_continuation(manager, bracket, ltp)
    bracket._profit_extension_assessed_mono = now
    bracket._profit_extension_assessed_ltp = float(ltp)
    bracket._profit_extension_decision = decision
    return decision


def _initial_risk(bracket: Any) -> float:
    entry = _positive(getattr(bracket, "entry_price", None))
    initial_sl = _positive(
        getattr(bracket, "initial_sl_trigger_price", None)
        or getattr(bracket, "sl_trigger_price", None)
    )
    if entry is None or initial_sl is None:
        return 0.0
    return abs(entry - initial_sl)


def _mfe_points(bracket: Any, ltp: float) -> float:
    entry = float(getattr(bracket, "entry_price", 0.0) or 0.0)
    if str(getattr(bracket, "side", "BUY")).upper() == "BUY":
        return max(0.0, max(float(getattr(bracket, "highest_ltp", entry) or entry), ltp) - entry)
    return max(0.0, entry - min(float(getattr(bracket, "lowest_ltp", entry) or entry), ltp))


def _cost_floor(manager: Any, bracket: Any) -> float:
    getter = getattr(manager, "_breakeven_cost_per_unit", None)
    if callable(getter):
        with suppress(Exception):
            return max(0.0, float(getter(bracket) or 0.0))
    return 0.0


def _protected_profit_floor(manager: Any, bracket: Any) -> bool:
    entry = float(getattr(bracket, "entry_price", 0.0) or 0.0)
    current_sl = float(getattr(bracket, "sl_trigger_price", 0.0) or 0.0)
    cost = _cost_floor(manager, bracket)
    if entry <= 0 or current_sl <= 0:
        return False
    if str(getattr(bracket, "side", "BUY")).upper() == "BUY":
        return current_sl >= entry + max(cost, _TICK_SIZE) - 1e-9
    return current_sl <= entry - max(cost, _TICK_SIZE) + 1e-9


def _ratchet_stop(manager: Any, bracket: Any, candidate: float, ltp: float, *, reason: str) -> bool:
    current = float(getattr(bracket, "sl_trigger_price", 0.0) or 0.0)
    side = str(getattr(bracket, "side", "BUY") or "BUY").upper()
    candidate = _round_tick(candidate)
    if side == "BUY":
        candidate = min(candidate, _round_tick(ltp - _TICK_SIZE))
        if candidate <= current or candidate >= ltp:
            return False
    else:
        candidate = max(candidate, _round_tick(ltp + _TICK_SIZE))
        if current > 0 and candidate >= current:
            return False
        if candidate <= ltp:
            return False

    validator = getattr(manager, "_is_trail_candidate_allowed", None)
    if callable(validator):
        with suppress(Exception):
            if not bool(validator(bracket, candidate, ltp)):
                return False

    lock = getattr(manager, "_lock", None)
    context = lock if hasattr(lock, "__enter__") else nullcontext()
    with context:
        current = float(getattr(bracket, "sl_trigger_price", 0.0) or 0.0)
        if side == "BUY" and candidate <= current:
            return False
        if side != "BUY" and current > 0 and candidate >= current:
            return False
        bracket.sl_trigger_price = candidate
        bracket.updated_at = time.time()
        bracket.last_trail_price = float(ltp)
        bracket.trail_revision = int(getattr(bracket, "trail_revision", 0) or 0) + 1

    LOGGER.info(
        "MARKET_AWARE_PROFIT_FLOOR_RATCHET symbol=%s old_sl=%.2f new_sl=%.2f ltp=%.2f reason=%s",
        getattr(bracket, "symbol", ""),
        current,
        candidate,
        ltp,
        reason,
        extra={
            "event": "MARKET_AWARE_PROFIT_FLOOR_RATCHET",
            "symbol": getattr(bracket, "symbol", ""),
            "old_sl": current,
            "new_sl": candidate,
            "ltp": ltp,
            "reason": reason,
        },
    )
    return True


def extend_final_target_if_supported(
    manager: Any,
    bracket: Any,
    ltp: float,
    final_action: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return original FINAL_TP action unless a protected strong winner can run."""
    action = dict(final_action)
    if str(action.get("type") or "") != "FINAL_TP":
        return action
    if not _env_bool("MARKET_AWARE_PROFIT_EXTENSION_ENABLED", True):
        return action

    risk = _initial_risk(bracket)
    entry = float(getattr(bracket, "entry_price", 0.0) or 0.0)
    if risk <= 0 or entry <= 0 or not _protected_profit_floor(manager, bracket):
        return action

    side = str(getattr(bracket, "side", "BUY") or "BUY").upper()
    current_target = float(getattr(bracket, "tp_trigger_price", 0.0) or 0.0)
    max_r = max(1.0, _env_float("PROFIT_EXTENSION_MAX_R", 4.0))
    max_target = entry + risk * max_r if side == "BUY" else entry - risk * max_r
    at_cap = current_target >= max_target - (_TICK_SIZE / 2) if side == "BUY" else current_target <= max_target + (_TICK_SIZE / 2)
    if at_cap:
        return action

    decision = _cached_assessment(manager, bracket, ltp)
    if not decision.extend:
        return action

    step_r = max(0.25, _env_float("PROFIT_EXTENSION_STEP_R", 0.75))
    candidate_target = current_target + risk * step_r if side == "BUY" else current_target - risk * step_r
    if side == "BUY":
        new_target = min(candidate_target, max_target)
        if new_target <= current_target + (_TICK_SIZE / 2):
            return action
    else:
        new_target = max(candidate_target, max_target)
        if new_target >= current_target - (_TICK_SIZE / 2):
            return action
    new_target = _round_tick(new_target)

    mfe = _mfe_points(bracket, ltp)
    lock_fraction = min(0.90, max(0.0, _env_float("PROFIT_EXTENSION_LOCK_FRACTION", 0.50)))
    cost = _cost_floor(manager, bracket)
    protected_points = max(cost, mfe * lock_fraction)
    candidate_sl = entry + protected_points if side == "BUY" else entry - protected_points

    old_sl = float(getattr(bracket, "sl_trigger_price", 0.0) or 0.0)
    old_target = current_target
    stop_changed = _ratchet_stop(
        manager,
        bracket,
        candidate_sl,
        ltp,
        reason="strong_continuation_extension",
    )

    lock = getattr(manager, "_lock", None)
    context = lock if hasattr(lock, "__enter__") else nullcontext()
    with context:
        bracket.tp_trigger_price = new_target
        bracket.updated_at = time.time()
        provenance = getattr(bracket, "trade_provenance", None)
        if not isinstance(provenance, dict):
            provenance = dict(provenance or {}) if isinstance(provenance, Mapping) else {}
            bracket.trade_provenance = provenance
        provenance.setdefault("profit_extension_original_tp", old_target)
        provenance["profit_extension_count"] = int(provenance.get("profit_extension_count", 0) or 0) + 1
        provenance["profit_extension_last_score"] = decision.score
        provenance["profit_extension_last_evidence_count"] = decision.evidence_count
        provenance["profit_extension_last_positive"] = list(decision.positive)
        provenance["profit_extension_last_negative"] = list(decision.negative)
        provenance["profit_extension_last_snapshot"] = dict(decision.snapshot)
        provenance["profit_extension_last_target"] = new_target
        provenance["profit_extension_last_sl"] = float(getattr(bracket, "sl_trigger_price", old_sl) or old_sl)
        provenance["profit_extension_last_at"] = time.time()

    saver = getattr(manager, "save_state", None)
    if callable(saver):
        with suppress(Exception):
            saver()
    notifier = getattr(manager, "_notify_event", None)
    payload = {
        "symbol": getattr(bracket, "symbol", ""),
        "score": decision.score,
        "evidence_count": decision.evidence_count,
        "old_tp": round(old_target, 2),
        "new_tp": round(new_target, 2),
        "old_sl": round(old_sl, 2),
        "new_sl": round(float(getattr(bracket, "sl_trigger_price", old_sl) or old_sl), 2),
        "stop_changed": stop_changed,
        "positive": list(decision.positive),
        "negative": list(decision.negative),
    }
    LOGGER.info(
        "PROFIT_TARGET_EXTENDED symbol=%s score=%.2f evidence=%s old_tp=%.2f new_tp=%.2f old_sl=%.2f new_sl=%.2f",
        payload["symbol"],
        decision.score,
        decision.evidence_count,
        old_target,
        new_target,
        old_sl,
        float(getattr(bracket, "sl_trigger_price", old_sl) or old_sl),
        extra={"event": "PROFIT_TARGET_EXTENDED", **payload},
    )
    if callable(notifier):
        with suppress(Exception):
            notifier("PROFIT_TARGET_EXTENDED", payload)
    return None


def tighten_market_aware_floor(manager: Any, bracket: Any, ltp: float) -> bool:
    """Tighten an already-profitable trade when continuation evidence deteriorates."""
    if not _env_bool("MARKET_AWARE_PROFIT_TIGHTEN_ENABLED", True):
        return False
    risk = _initial_risk(bracket)
    if risk <= 0:
        return False
    mfe = _mfe_points(bracket, ltp)
    if mfe < risk:
        return False
    decision = _cached_assessment(manager, bracket, ltp)
    threshold = _env_float("MARKET_AWARE_PROFIT_TIGHTEN_SCORE", -2.0)
    min_evidence = max(1, int(_env_float("MARKET_AWARE_PROFIT_MIN_EVIDENCE", 4.0)))
    if decision.evidence_count < min_evidence or decision.score > threshold:
        return False

    lock_fraction = min(0.90, max(0.0, _env_float("PROFIT_WEAK_LOCK_FRACTION", 0.65)))
    entry = float(getattr(bracket, "entry_price", 0.0) or 0.0)
    cost = _cost_floor(manager, bracket)
    protected_points = max(cost, mfe * lock_fraction)
    side = str(getattr(bracket, "side", "BUY") or "BUY").upper()
    candidate = entry + protected_points if side == "BUY" else entry - protected_points
    changed = _ratchet_stop(
        manager,
        bracket,
        candidate,
        ltp,
        reason="weak_continuation_profit_lock",
    )
    if not changed:
        return False

    provenance = getattr(bracket, "trade_provenance", None)
    if isinstance(provenance, dict):
        provenance["profit_tighten_last_score"] = decision.score
        provenance["profit_tighten_last_at"] = time.time()
    saver = getattr(manager, "save_state", None)
    if callable(saver):
        with suppress(Exception):
            saver()
    return True


def adapt_evaluate_exit_fast(original: Callable[..., Any]) -> Callable[..., Any]:
    """Let only FINAL_TP consult market continuation; all safety exits pass through."""

    @wraps(original)
    def wrapped(
        self: Any,
        bracket: Any,
        ltp: float,
        *,
        committed_sl: float | None = None,
    ) -> Any:
        action = original(self, bracket, ltp, committed_sl=committed_sl)
        if not isinstance(action, Mapping) or str(action.get("type") or "") != "FINAL_TP":
            return action
        try:
            return extend_final_target_if_supported(self, bracket, float(ltp), action)
        except Exception as exc:  # noqa: BLE001 - fail closed to canonical FINAL_TP
            LOGGER.error(
                "PROFIT_EXTENSION_EVALUATION_FAILED symbol=%s error=%s",
                getattr(bracket, "symbol", ""),
                exc,
                extra={
                    "event": "PROFIT_EXTENSION_EVALUATION_FAILED",
                    "symbol": getattr(bracket, "symbol", ""),
                    "error_type": type(exc).__name__,
                },
                exc_info=exc,
            )
            return action

    return wrapped


def adapt_trailing_math(original: Callable[..., Any]) -> Callable[..., Any]:
    """Preserve canonical trailing, then optionally tighten weak profitable trades."""

    @wraps(original)
    def wrapped(self: Any, bracket: Any) -> bool:
        original_changed = bool(original(self, bracket))
        ltp = _positive(getattr(bracket, "last_ltp", None))
        if ltp is None:
            return original_changed
        try:
            market_changed = tighten_market_aware_floor(self, bracket, ltp)
        except Exception as exc:  # noqa: BLE001 - canonical trailing remains authoritative
            LOGGER.debug(
                "PROFIT_TIGHTEN_EVALUATION_FAILED symbol=%s error=%s",
                getattr(bracket, "symbol", ""),
                exc,
            )
            market_changed = False
        return original_changed or market_changed

    return wrapped


def adapt_confirm_entry_fill(original: Callable[..., Any]) -> Callable[..., Any]:
    """Capture cached market baselines only after canonical fill activation succeeds."""

    @wraps(original)
    def wrapped(self: Any, order_id: str, fill_price: float, filled_qty: int | None = None) -> Any:
        result = original(self, order_id, fill_price, filled_qty)
        bracket = None
        getter = getattr(self, "get_bracket", None)
        if callable(getter):
            with suppress(Exception):
                bracket = getter(order_id)
        if bracket is None or not bool(getattr(bracket, "entry_confirmed", False)):
            return result
        try:
            changed = capture_entry_market_baseline(self, bracket)
            if changed:
                saver = getattr(self, "save_state", None)
                if callable(saver):
                    saver()
        except Exception as exc:  # noqa: BLE001 - baseline is optional, protection is not
            LOGGER.debug(
                "PROFIT_EXTENSION_BASELINE_CAPTURE_FAILED symbol=%s error=%s",
                getattr(bracket, "symbol", ""),
                exc,
            )
        return result

    return wrapped


def apply_patches(target_cls: type[Any]) -> None:
    """Install the market-aware supervisor on the single canonical bracket class."""
    if bool(getattr(target_cls, "_market_aware_profit_extension_installed", False)):
        return

    evaluate = getattr(target_cls, "_evaluate_exit_fast", None)
    trailing = getattr(target_cls, "_apply_trailing_math", None)
    confirm = getattr(target_cls, "confirm_entry_fill", None)
    if not callable(evaluate) or not callable(trailing) or not callable(confirm):
        raise RuntimeError("Canonical bracket lifecycle methods unavailable")

    target_cls._evaluate_exit_fast = adapt_evaluate_exit_fast(evaluate)
    target_cls._apply_trailing_math = adapt_trailing_math(trailing)
    target_cls.confirm_entry_fill = adapt_confirm_entry_fill(confirm)
    target_cls._market_aware_profit_extension_installed = True


__all__ = [
    "ContinuationDecision",
    "adapt_confirm_entry_fill",
    "adapt_evaluate_exit_fast",
    "adapt_trailing_math",
    "apply_patches",
    "assess_continuation",
    "capture_entry_market_baseline",
    "extend_final_target_if_supported",
    "tighten_market_aware_floor",
]
