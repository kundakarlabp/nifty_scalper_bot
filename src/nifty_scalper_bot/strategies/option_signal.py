"""Option-native candidate scoring: IV richness, OI/price confirmation, depth imbalance.

Underlying direction is owned elsewhere. This module only answers whether a
candidate option is a good expression of that direction. OI is therefore not
treated as directional by itself: rising OI supports a long-premium candidate
only when the premium also moves meaningfully higher, while rising OI against a
meaningful premium fall is adverse divergence. Sub-noise premium changes are
neutral. Static book imbalance remains a small confirmation, not entry authority.

Every input is optional. Missing evidence is neutral and never crashes or
manufactures a signal.

Env overrides:
- OPTION_SIGNAL_ENABLED          (default true)
- OPTION_IV_RICH                 (default 0.60: IV above this is penalized)
- OPTION_IV_CHEAP                (default 0.30: IV below this is rewarded)
- OPTION_DEPTH_IMBALANCE         (default 1.5: bid/ask qty ratio for support)
- OPTION_OI_MIN_PRICE_MOVE_PCT   (default 0.002: minimum premium move fraction)
"""

from __future__ import annotations

import os
from typing import Any

from nifty_scalper_bot.config.env_utils import parse_bool_env, parse_float_env

_PRIOR_OI: dict[str, float] = {}
_PRIOR_PRICE: dict[str, float] = {}
_MAX_CACHE = 64


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _depth_totals(depth: Any) -> tuple[float, float]:
    """Args: kite-style depth dict. Returns: (total_bid_qty, total_ask_qty)."""
    if not isinstance(depth, dict):
        return 0.0, 0.0
    bid_qty = sum(_f(level.get("quantity")) or 0.0 for level in depth.get("buy") or [] if isinstance(level, dict))
    ask_qty = sum(_f(level.get("quantity")) or 0.0 for level in depth.get("sell") or [] if isinstance(level, dict))
    return bid_qty, ask_qty


def _premium(metrics: dict[str, Any]) -> float | None:
    price = _f(metrics.get("ltp") or metrics.get("last_price"))
    if price is not None and price > 0:
        return price
    bid, ask = _f(metrics.get("bid")), _f(metrics.get("ask"))
    if bid is not None and ask is not None and bid > 0 and ask >= bid:
        return (bid + ask) / 2.0
    return None


def _premium_noise_floor(metrics: dict[str, Any], price: float) -> float:
    """Return minimum meaningful premium move as a fraction of prior premium."""
    configured = max(
        0.0,
        parse_float_env(os.getenv("OPTION_OI_MIN_PRICE_MOVE_PCT"), 0.002),
    )
    bid, ask = _f(metrics.get("bid")), _f(metrics.get("ask"))
    if bid is None or ask is None or bid <= 0 or ask < bid or price <= 0:
        return configured
    half_spread_fraction = ((ask - bid) / 2.0) / price
    return max(configured, half_spread_fraction)


def _prune_cache() -> None:
    if len(_PRIOR_OI) <= _MAX_CACHE:
        return
    for stale in list(_PRIOR_OI)[: len(_PRIOR_OI) - _MAX_CACHE]:
        _PRIOR_OI.pop(stale, None)
        _PRIOR_PRICE.pop(stale, None)


def score_option_candidate(symbol: str, metrics: dict[str, Any] | None) -> tuple[float, list[str]]:
    """Return option-expression score delta in [-1.5, +1.5] and reasons."""
    if not parse_bool_env(os.getenv("OPTION_SIGNAL_ENABLED"), True):
        return 0.0, ["option_signal_disabled"]
    if not metrics:
        return 0.0, ["option_metrics_unavailable"]

    delta = 0.0
    reasons: list[str] = []

    iv = _f(metrics.get("iv"))
    if iv is not None and iv > 0:
        iv_rich = parse_float_env(os.getenv("OPTION_IV_RICH"), 0.60)
        iv_cheap = parse_float_env(os.getenv("OPTION_IV_CHEAP"), 0.30)
        if iv >= iv_rich:
            delta -= 1.0
            reasons.append(f"iv_rich_{iv:.2f}")
        elif iv <= iv_cheap:
            delta += 0.5
            reasons.append(f"iv_reasonable_{iv:.2f}")

    price = _premium(metrics)
    oi = _f(metrics.get("oi"))
    if oi is not None and oi > 0:
        prior_oi = _PRIOR_OI.get(symbol)
        prior_price = _PRIOR_PRICE.get(symbol)
        if prior_oi is not None and prior_oi > 0:
            oi_change = (oi - prior_oi) / prior_oi
            if abs(oi_change) >= 0.01:
                if price is None or prior_price is None or prior_price <= 0:
                    reasons.append("oi_change_unconfirmed")
                elif oi_change > 0:
                    price_change = (price - prior_price) / prior_price
                    noise_floor = _premium_noise_floor(metrics, prior_price)
                    if abs(price_change) < noise_floor:
                        reasons.append("oi_buildup_price_noise")
                    elif price_change > 0:
                        delta += 0.5
                        reasons.append("oi_buildup_price_confirmed")
                    else:
                        delta -= 0.5
                        reasons.append("oi_buildup_price_divergence")
                else:
                    # Falling OI can represent either long liquidation or short
                    # covering. Without participant-side data it is context,
                    # not a directional long-premium signal.
                    reasons.append("oi_unwinding_context")
        _PRIOR_OI[symbol] = oi
        if price is not None:
            _PRIOR_PRICE[symbol] = price
        _prune_cache()

    bid_qty, ask_qty = _depth_totals(metrics.get("depth"))
    if bid_qty > 0 and ask_qty > 0:
        ratio_min = parse_float_env(os.getenv("OPTION_DEPTH_IMBALANCE"), 1.5)
        ratio = bid_qty / ask_qty
        if ratio >= ratio_min:
            delta += 0.5
            reasons.append("depth_buy_support")
        elif ratio <= 1.0 / ratio_min:
            delta -= 0.5
            reasons.append("depth_sell_pressure")

    if not reasons:
        reasons.append("option_neutral")
    return max(-1.5, min(1.5, delta)), reasons
