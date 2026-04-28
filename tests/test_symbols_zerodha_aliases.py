"""Tests for Zerodha REST quote alias helpers."""

from __future__ import annotations

import pytest

from nifty_scalper_bot.utils.symbols import (
    to_zerodha_quote_symbol,
    zerodha_quote_aliases,
)


class TestToZerodhaQuoteSymbol:
    def test_canonical_nifty_maps_to_kite_index_name(self) -> None:
        assert to_zerodha_quote_symbol("NSE:NIFTY") == "NSE:NIFTY 50"

    def test_bare_nifty_maps_to_kite_index_name(self) -> None:
        assert to_zerodha_quote_symbol("NIFTY") == "NSE:NIFTY 50"

    def test_nifty50_alias_maps_to_kite_index_name(self) -> None:
        assert to_zerodha_quote_symbol("NIFTY50") == "NSE:NIFTY 50"

    def test_already_kite_form_passes_through(self) -> None:
        assert to_zerodha_quote_symbol("NSE:NIFTY 50") == "NSE:NIFTY 50"

    def test_banknifty_maps_to_kite_index_name(self) -> None:
        assert to_zerodha_quote_symbol("NSE:BANKNIFTY") == "NSE:NIFTY BANK"
        assert to_zerodha_quote_symbol("BANKNIFTY") == "NSE:NIFTY BANK"

    def test_finnifty_maps_to_kite_index_name(self) -> None:
        assert to_zerodha_quote_symbol("NSE:FINNIFTY") == "NSE:NIFTY FIN SERVICE"
        assert to_zerodha_quote_symbol("FINNIFTY") == "NSE:NIFTY FIN SERVICE"

    def test_option_symbol_passes_through_unchanged(self) -> None:
        assert (
            to_zerodha_quote_symbol("NFO:NIFTY2612025700CE")
            == "NFO:NIFTY2612025700CE"
        )

    def test_equity_symbol_passes_through_unchanged(self) -> None:
        assert to_zerodha_quote_symbol("NSE:RELIANCE") == "NSE:RELIANCE"

    def test_empty_input_returns_empty(self) -> None:
        assert to_zerodha_quote_symbol("") == ""
        assert to_zerodha_quote_symbol(None) == ""  # type: ignore[arg-type]

    def test_double_space_collapses(self) -> None:
        assert to_zerodha_quote_symbol("NSE:NIFTY  50") == "NSE:NIFTY 50"


class TestZerodhaQuoteAliases:
    def test_nifty_aliases_include_kite_form_and_bare_name(self) -> None:
        aliases = zerodha_quote_aliases("NSE:NIFTY")
        assert "NSE:NIFTY 50" in aliases
        assert "NIFTY 50" in aliases
        # Kite-preferred form must come first.
        assert aliases[0] == "NSE:NIFTY 50"

    def test_aliases_preserve_original_for_round_trip(self) -> None:
        aliases = zerodha_quote_aliases("NSE:NIFTY")
        assert "NSE:NIFTY" in aliases

    def test_aliases_dedupe(self) -> None:
        aliases = zerodha_quote_aliases("NSE:NIFTY 50")
        assert aliases.count("NSE:NIFTY 50") == 1

    def test_option_symbol_aliases_unchanged(self) -> None:
        aliases = zerodha_quote_aliases("NFO:NIFTY2612025700CE")
        assert aliases[0] == "NFO:NIFTY2612025700CE"
        assert "NIFTY2612025700CE" in aliases

    def test_empty_input_returns_empty(self) -> None:
        assert zerodha_quote_aliases("") == []
        assert zerodha_quote_aliases(None) == []  # type: ignore[arg-type]

    def test_banknifty_aliases(self) -> None:
        aliases = zerodha_quote_aliases("NSE:BANKNIFTY")
        assert "NSE:NIFTY BANK" in aliases
        assert "NIFTY BANK" in aliases
