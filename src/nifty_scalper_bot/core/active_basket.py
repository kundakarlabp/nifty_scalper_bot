"""Lightweight active basket helpers without app boot dependencies.

Runtime role:
- Normalizes provided active basket schemas.
- Does not infer missing futures/options.
- Must not select contracts."""

from __future__ import annotations

import os
import re
from typing import Mapping

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

# Matches the 4-6 digit strike immediately before the CE/PE suffix.
# For NFO:NIFTY2660923500CE  →  group("strike") == "23500"
# The expiry code (e.g. "26609") is longer than 6 digits so it's excluded.
_OPTION_STRIKE_RE = re.compile(r"(?P<strike>\d{4,6})(?:CE|PE)$", re.IGNORECASE)


def extract_symbol_strike(symbol: str) -> int | None:
    """Extract option strike from Zerodha NIFTY option symbols.

    Examples:
    - NFO:NIFTY2660923450CE -> 23450
    - NFO:NIFTY2660923500PE -> 23500
    - NFO:NIFTY26JUN23950CE -> 23950
    """
    raw = str(symbol or "").strip().upper()
    if not raw:
        return None

    bare = raw.split(":")[-1]
    if not (bare.endswith("CE") or bare.endswith("PE")):
        return None

    body = bare[:-2]
    match = re.search(r"(\d+)$", body)
    if not match:
        return None

    digits = match.group(1)

    if len(digits) >= 5:
        strike = int(digits[-5:])
        if 10000 <= strike <= 50000 and strike % 50 == 0:
            return strike

    if len(digits) >= 4:
        strike = int(digits[-4:])
        if 1000 <= strike <= 9999 and strike % 50 == 0:
            return strike

    return None


def pick_atm_option_symbols_from_basket(
    basket: Mapping[str, object],
) -> tuple[str | None, str | None]:
    """Pick ATM CE/PE from basket. Args: basket. Returns: ce/pe symbols. Raises: none."""
    selected_ce = str(basket.get('selected_ce') or basket.get('atm_ce') or '') or None
    selected_pe = str(basket.get('selected_pe') or basket.get('atm_pe') or '') or None
    option_symbols = [
        str(s)
        for s in list(basket.get('option_symbols') or basket.get('symbols') or [])
        if str(s).endswith(('CE', 'PE'))
    ]
    atm_raw = basket.get('atm_strike')
    try:
        atm_strike = int(float(atm_raw)) if atm_raw is not None else None
    except Exception:
        atm_strike = None
    valid_symbols = set(option_symbols)
    if selected_ce and (not selected_ce.endswith('CE') or selected_ce not in valid_symbols):
        selected_ce = None
    if selected_pe and (not selected_pe.endswith('PE') or selected_pe not in valid_symbols):
        selected_pe = None
    if selected_ce and selected_pe:
        return selected_ce, selected_pe

    ce_candidates = [s for s in option_symbols if s.endswith('CE')]
    pe_candidates = [s for s in option_symbols if s.endswith('PE')]
    if not selected_ce and ce_candidates:
        selected_ce = min(
            ce_candidates,
            key=lambda s: abs((extract_symbol_strike(s) or 0) - (atm_strike or (extract_symbol_strike(s) or 0))),
        )
    if not selected_pe and pe_candidates:
        selected_pe = min(
            pe_candidates,
            key=lambda s: abs((extract_symbol_strike(s) or 0) - (atm_strike or (extract_symbol_strike(s) or 0))),
        )
    return selected_ce, selected_pe


def normalize_active_basket_schema(basket: Mapping[str, object]) -> dict[str, object]:
    """Return canonical basket dict with guaranteed context and option fields.

    Key invariants:
    - If the input already has ``selected_ce`` / ``selected_pe``, those SSOT
      values are KEPT.  ``pick_atm_option_symbols_from_basket`` is only called
      as a fallback when they are absent.
    - Token fields (``token_by_symbol``, ``all_tokens``, ``all_symbols``,
      ``selected_ce_token``, ``selected_pe_token``) are passed through
      unchanged so MDM never sees an empty token map.
    """
    out = dict(basket or {})
    spot_symbol = str(out.get("spot_symbol") or "NSE:NIFTY")
    futures_symbol = str(out.get("futures_symbol") or out.get("future_symbol") or "")
    option_symbols = [
        str(s)
        for s in list(out.get("option_symbols") or out.get("symbols") or [])
        if str(s).endswith(("CE", "PE"))
    ]
    option_symbols = list(dict.fromkeys(option_symbols))
    ce_symbols = list(
        dict.fromkeys(
            [str(s) for s in list(out.get("ce_symbols") or []) if str(s).endswith("CE")]
            or [s for s in option_symbols if s.endswith("CE")]
        )
    )
    pe_symbols = list(
        dict.fromkeys(
            [str(s) for s in list(out.get("pe_symbols") or []) if str(s).endswith("PE")]
            or [s for s in option_symbols if s.endswith("PE")]
        )
    )
    # Preserve SSOT selected_ce/selected_pe if already present.  Only fall
    # back to pick_atm_option_symbols_from_basket when both are absent.
    ssot_ce = str(out.get("selected_ce") or out.get("atm_ce") or "") or None
    ssot_pe = str(out.get("selected_pe") or out.get("atm_pe") or "") or None
    if ssot_ce and ssot_pe:
        selected_ce, selected_pe = ssot_ce, ssot_pe
    else:
        selected_ce, selected_pe = pick_atm_option_symbols_from_basket(
            {
                **out,
                "option_symbols": option_symbols,
                "symbols": option_symbols,
                "ce_symbols": ce_symbols,
                "pe_symbols": pe_symbols,
            }
        )
    out["spot_symbol"] = spot_symbol
    out["futures_symbol"] = futures_symbol
    out["option_symbols"] = option_symbols
    out["ce_symbols"] = ce_symbols
    out["pe_symbols"] = pe_symbols
    out["selected_ce"] = selected_ce
    out["selected_pe"] = selected_pe
    out["atm_ce"] = out.get("atm_ce") or selected_ce
    out["atm_pe"] = out.get("atm_pe") or selected_pe
    out["symbols"] = list(dict.fromkeys([s for s in [spot_symbol, futures_symbol, *option_symbols] if s]))
    # Token fields must survive normalization untouched so MDM can consume them.
    # They are already in `out` from the input dict; no action needed beyond
    # the comment to make the invariant explicit.
    return out


def build_active_trading_basket_symbols(ctx: object, basket: Mapping[str, object]) -> list[str]:
    """Build deterministic active basket. Args: ctx,basket. Returns: ordered symbols. Raises: none."""
    _ = ctx
    max_active_options = max(2, int(os.getenv('MAX_ACTIVE_OPTION_SYMBOLS', '6') or 6))
    spot = str(basket.get('spot_symbol') or 'NSE:NIFTY')
    fut = str(basket.get('futures_symbol') or '')
    selected_ce, selected_pe = pick_atm_option_symbols_from_basket(basket)
    option_symbols = [
        str(s)
        for s in list(basket.get('option_symbols') or basket.get('symbols') or [])
        if str(s).endswith(('CE', 'PE'))
    ]
    option_symbols = list(dict.fromkeys(option_symbols))
    core = [s for s in (selected_ce, selected_pe) if s]
    nearby = [s for s in option_symbols if s not in core]
    selected_options = (core + nearby)[:max_active_options]
    out = list(dict.fromkeys([s for s in (spot, fut, *selected_options) if s]))
    LOGGER.info(
        'ACTIVE_TRADING_BASKET_SELECTED count=%d selected_ce=%s selected_pe=%s symbols=%s',
        len(out),
        selected_ce,
        selected_pe,
        out,
    )
    return out


__all__ = ['build_active_trading_basket_symbols', 'pick_atm_option_symbols_from_basket', 'extract_symbol_strike', 'normalize_active_basket_schema']
