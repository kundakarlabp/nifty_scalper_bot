from __future__ import annotations

from nifty_scalper_bot.options.strike_selector import (
    _extract_expiry_token,
    _parse_strike_from_symbol,
)


def test_parse_strike_from_zerodha_symbol() -> None:
    assert _parse_strike_from_symbol('NIFTY26FEB22500CE') == 22500.0
    assert _parse_strike_from_symbol('NFO:NIFTY26FEB22500PE') == 22500.0


def test_parse_strike_from_weekly_digit_expiry() -> None:
    assert _parse_strike_from_symbol('NIFTY25022722400CE') == 22400.0


def test_extract_expiry_token_matches_supported_formats() -> None:
    assert _extract_expiry_token('NIFTY26FEB22500CE') == '26FEB22'
    assert _extract_expiry_token('NIFTY25022722400PE') == '250227'
