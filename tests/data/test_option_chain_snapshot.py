from __future__ import annotations

from datetime import date

import pytest

from nifty_scalper_bot.data.option_chain import OptionChainSnapshot, OptionQuote


def build_quote(
    strike: int, ce_bid: float, ce_ask: float, pe_bid: float, pe_ask: float, ts: int
) -> OptionQuote:
    return OptionQuote(
        strike=strike,
        ce_bid=ce_bid,
        ce_ask=ce_ask,
        pe_bid=pe_bid,
        pe_ask=pe_ask,
        ce_oi=50_000,
        pe_oi=60_000,
        ltt_ns=ts,
    )


def test_atm_detection_rounds_to_nearest_50() -> None:
    chain = OptionChainSnapshot(
        symbol="NIFTY",
        spot=22_785.0,
        expiry=date(2024, 5, 2),
        quotes=[build_quote(22_800, 91.0, 92.5, 108.0, 109.5, 1)],
    )
    assert chain.atm_strike() == 22_800


def test_filter_liquid_removes_wide_spread_and_low_oi() -> None:
    rich = build_quote(22_750, 101.0, 102.5, 97.0, 98.5, 1)
    illiquid = OptionQuote(
        strike=22_700,
        ce_bid=50.0,
        ce_ask=54.0,
        pe_bid=149.0,
        pe_ask=152.5,
        ce_oi=10_000,
        pe_oi=70_000,
        ltt_ns=2,
    )
    chain = OptionChainSnapshot(
        symbol="NIFTY",
        spot=22_785.0,
        expiry=date(2024, 5, 2),
        quotes=[rich, illiquid],
    )
    liquid = chain.filter_liquid()
    assert all(q.strike != illiquid.strike for q in liquid.quotes)


def test_best_otm_handles_out_of_order_quotes() -> None:
    quotes = [
        build_quote(22_900, 60.0, 61.0, 140.0, 141.0, 3),
        build_quote(22_800, 92.0, 93.0, 109.0, 110.0, 2),
        build_quote(22_850, 75.0, 76.0, 124.0, 125.0, 4),
        build_quote(22_750, 107.0, 108.0, 94.0, 95.0, 1),
    ]
    chain = OptionChainSnapshot(
        symbol="NIFTY",
        spot=22_785.0,
        expiry=date(2024, 5, 2),
        quotes=quotes,
    )
    call = chain.best_ce_otm(0)
    put = chain.best_pe_otm(0)
    assert call.strike == 22_850
    assert put.strike == 22_750


def test_sanity_check_detects_timestamp_inversion() -> None:
    chain = OptionChainSnapshot(
        symbol="NIFTY",
        spot=22_785.0,
        expiry=date(2024, 5, 2),
        quotes=[
            build_quote(22_800, 92.0, 93.0, 109.0, 110.0, 10),
            build_quote(22_850, 75.0, 76.0, 124.0, 125.0, 5),
        ],
    )
    with pytest.raises(ValueError):
        chain.sanity_check()
