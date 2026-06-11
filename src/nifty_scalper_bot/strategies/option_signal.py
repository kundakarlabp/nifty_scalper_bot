"""Option-native candidate scoring: IV richness, OI buildup, depth imbalance.

All current strategy signals read the underlying price series. This module
adds the option-specific information that actually differentiates option
trades: whether the premium is expensive (IV), whether positioning supports
the move (open-interest buildup), and whether the order book leans with the
trade (bid/ask depth imbalance).

Design: one pure scoring function plus a tiny module-level OI cache. Every
input is optional — missing data contributes nothing, so the scorer can
never block trading or crash on sparse feeds.

Env overrides:
- OPTION_SIGNAL_ENABLED   (default true)
- OPTION_IV_RICH          (default 0.60: IV above this is penalized)
- OPTION_IV_CHEAP         (default 0.30: IV below this is rewarded)
- OPTION_DEPTH_IMBALANCE  (default 1.5: bid/ask qty ratio for support)
"""

from __future__ import annotations

import os
from typing import Any

from nifty_scalper_bot.config.env_utils import parse_bool_env, parse_float_env

# prior OI per symbol within this process; small and self-pruning
_PRIOR_OI: dict[str, float] = {}
_MAX_CACHE = 64


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # reject NaN


def _depth_totals(depth: Any) -> tuple[float, float]:
    """Args: kite-style depth dict. Returns: (total_bid_qty, total_ask_qty)."""
    if not isinstance(depth, dict):
        return 0.0, 0.0
    bid_qty = sum(_f(level.get("quantity")) or 0.0 for level in depth.get("buy") or [] if isinstance(level, dict))
    ask_qty = sum(_f(level.get("quantity")) or 0.0 for level in depth.get("sell") or [] if isinstance(level, dict))
    return bid_qty, ask_qty


def score_option_candidate(symbol: str, metrics: dict[str, Any] | None) -> tuple[float, list[str]]:
    """Args: option symbol, DataHub.get_option_metrics payload (or None).
    Returns: (score_delta in [-1.5, +1.5], reasons). Raises: none.
    """
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

    oi = _f(metrics.get("oi"))
    if oi is not None and oi > 0:
        prior = _PRIOR_OI.get(symbol)
        if prior is not None and prior > 0:
            change = (oi - prior) / prior
            if change >= 0.01:
                delta += 0.5
                reasons.append("oi_buildup")
            elif change <= -0.01:
                delta -= 0.25
                reasons.append("oi_unwinding")
        _PRIOR_OI[symbol] = oi
        if len(_PRIOR_OI) > _MAX_CACHE:
            for stale in list(_PRIOR_OI)[: len(_PRIOR_OI) - _MAX_CACHE]:
                _PRIOR_OI.pop(stale, None)

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
