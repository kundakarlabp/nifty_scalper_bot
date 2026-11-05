"""Protective MARKET exits retain point-in-time WS book evidence without I/O."""

import time

from nifty_scalper_bot.execution.bracket_core import BracketManager


SYMBOL = "NFO:NIFTY26N0924350CE"


def _manager() -> BracketManager:
    manager = BracketManager.__new__(BracketManager)
    manager._exit_protective_order_mode = "MARKET"
    manager._exit_quote_max_age = 2.0
    manager._exit_quotes = {}
    return manager


def _price(manager: BracketManager):
    return manager._price_exit_order(
        bracket=None,
        symbol=SYMBOL,
        side="SELL",
        reason="HARD_SL_BREACH",
        preferred_order_type="LIMIT",
        qty=65,
    )


def test_market_stop_uses_fresh_cached_book_for_evidence_only() -> None:
    manager = _manager()
    manager._exit_quotes[SYMBOL] = (177.50, 177.85, time.time() - 0.1)

    def no_quote_fetch(_symbol: str):
        raise AssertionError("protective MARKET exit must not fetch a quote")

    manager._extract_exit_quote = no_quote_fetch
    order_type, price, metadata = _price(manager)

    assert order_type == "MARKET"
    assert price is None
    assert metadata["bid"] == 177.50
    assert metadata["ask"] == 177.85
    assert 0 <= metadata["quote_age_ms"] <= 2000
    assert metadata["fallback"] is False


def test_stale_cached_book_does_not_masquerade_as_fresh_exit_quote() -> None:
    manager = _manager()
    manager._exit_quotes[SYMBOL] = (177.50, 177.85, time.time() - 30.0)
    order_type, price, metadata = _price(manager)

    assert order_type == "MARKET"
    assert price is None
    assert metadata["bid"] is None
    assert metadata["ask"] is None
    assert metadata["quote_age_ms"] is None
