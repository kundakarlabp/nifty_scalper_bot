"""Lightweight active basket helpers without app boot dependencies."""

from __future__ import annotations

import os
from typing import Mapping

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


def extract_symbol_strike(symbol: str) -> int | None:
    """Extract option strike from symbol. Args: symbol. Returns: strike or None. Raises: none."""
    digits = ''
    base = symbol[:-2] if symbol.endswith(('CE', 'PE')) else symbol
    for ch in reversed(base):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    return int(digits) if digits else None


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


__all__ = ['build_active_trading_basket_symbols', 'pick_atm_option_symbols_from_basket', 'extract_symbol_strike']
