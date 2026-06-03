"""Regression tests for live-entry spread_unknown fix.

Root cause: _symbol_live_entry_ready extracted spread_pct only via
    spread_pct = _extract_float(quote, "spread_pct")
Zerodha WS full-quote ticks do NOT carry a pre-computed "spread_pct" key.
They carry bid/ask at top-level OR inside depth.buy[0]/sell[0].
So spread_pct was always None → "spread_unknown" → final_ready=False.
OrderFlow resolved it fine by computing from indicators['bid']/['ask'].

Fix: canonical _resolve_quote_bid_ask_spread helper that tries
    top-level bid/ask → best_bid/best_ask → depth.buy[0]/sell[0] → derives spread_pct.
"""
from __future__ import annotations
import pytest
from nifty_scalper_bot.strategies.runner import _resolve_quote_bid_ask_spread


# ─── 1. Top-level bid/ask ──────────────────────────────────────────────────

def test_resolves_top_level_bid_ask() -> None:
    quote = {"bid": 120.5, "ask": 121.0, "ltp": 120.7, "tradable_quote": True}
    bid, ask, spread_pct, source = _resolve_quote_bid_ask_spread(quote)
    assert bid == 120.5
    assert ask == 121.0
    assert spread_pct is not None
    assert round(spread_pct, 4) == round(((121.0 - 120.5) / 120.75) * 100.0, 4)
    assert source == "top_level"


# ─── 2. best_bid / best_ask ───────────────────────────────────────────────

def test_resolves_best_bid_best_ask() -> None:
    quote = {"best_bid": 115.0, "best_ask": 115.8, "ltp": 115.4}
    bid, ask, spread_pct, source = _resolve_quote_bid_ask_spread(quote)
    assert bid == 115.0
    assert ask == 115.8
    assert spread_pct is not None and spread_pct > 0
    assert source == "best_bid_ask"


# ─── 3. depth.buy[0] / depth.sell[0] — the Zerodha WS full-quote layout ──

def test_resolves_spread_from_depth_levels() -> None:
    """
    Zerodha WS full-quote ticks look like:
        {"ltp": 120.0, "depth": {"buy": [{"price": 119.5, "quantity": 75}],
                                  "sell": [{"price": 120.5, "quantity": 50}]}}
    NO top-level bid/ask. Live-entry gate was returning spread_unknown here.
    """
    quote = {
        "ltp": 120.0,
        "tradable_quote": False,  # not pre-computed — we derive from depth
        "depth": {
            "buy": [{"price": 119.5, "quantity": 75}, {"price": 119.0, "quantity": 100}],
            "sell": [{"price": 120.5, "quantity": 50}, {"price": 121.0, "quantity": 80}],
        },
        "depth_available": True,
        # No "bid", "ask", "spread_pct" key — exactly the failing scenario
    }
    bid, ask, spread_pct, source = _resolve_quote_bid_ask_spread(quote)
    assert bid == 119.5, f"bid should be depth.buy[0].price=119.5, got {bid}"
    assert ask == 120.5, f"ask should be depth.sell[0].price=120.5, got {ask}"
    assert spread_pct is not None and spread_pct > 0, "spread_pct must be derived from depth bid/ask"
    assert source == "depth", f"source should be 'depth', got {source}"


def test_depth_derived_spread_matches_orderflow_formula() -> None:
    """spread_pct from depth must match the OrderFlow formula: (ask-bid)/mid * 100."""
    bid_price, ask_price = 119.5, 120.5
    quote = {
        "ltp": 120.0,
        "depth": {
            "buy": [{"price": bid_price, "quantity": 75}],
            "sell": [{"price": ask_price, "quantity": 50}],
        },
        "depth_available": True,
    }
    _, _, spread_pct, _ = _resolve_quote_bid_ask_spread(quote)
    mid = (bid_price + ask_price) / 2.0
    expected = ((ask_price - bid_price) / mid) * 100.0
    assert spread_pct is not None
    assert abs(spread_pct - expected) < 1e-9, f"spread_pct={spread_pct} != expected={expected}"


# ─── 4. spread_pct key with fresh depth proof ─────────────────────────────

def test_pre_computed_spread_pct_accepted_with_depth_proof() -> None:
    """If spread_pct key exists AND depth_available=True, accept it."""
    quote = {
        "ltp": 120.0,
        "spread_pct": 0.24,
        "tradable_quote": True,
        "depth_available": True,
    }
    bid, ask, spread_pct, source = _resolve_quote_bid_ask_spread(quote)
    # No bid/ask in quote — bid and ask are None
    # But spread_pct should be accepted since depth_available=True (has_quote_proof)
    assert spread_pct == 0.24


# ─── 5. Stale derived spread_pct rejected without fresh quote/depth proof ─

def test_stale_spread_pct_rejected_without_quote_proof() -> None:
    """
    If spread_pct key is present but NO tradable_quote/depth_available/depth,
    the value may be stale (from indicators_ctx). Must NOT trust it.
    final_ready=False if other gates would pass but spread is from stale indicator.
    """
    quote = {
        "ltp": 120.0,
        "spread_pct": 0.24,     # derived from indicators — no fresh proof
        "tradable_quote": False,
        "depth_available": False,
        # no "depth" key
    }
    bid, ask, spread_pct, source = _resolve_quote_bid_ask_spread(quote)
    assert bid is None
    assert ask is None
    assert spread_pct is None, (
        f"spread_pct should be None when no fresh quote/depth proof exists, got {spread_pct}. "
        "Trusting a stale derived spread_pct without live bid/ask proof would allow bad entries."
    )


# ─── 6. Invalid bid/ask (inverted or zero) ────────────────────────────────

def test_inverted_bid_ask_not_accepted() -> None:
    """ask < bid is invalid — should not derive a spread from it."""
    quote = {"bid": 121.0, "ask": 119.0}  # inverted
    bid, ask, spread_pct, source = _resolve_quote_bid_ask_spread(quote)
    # bid/ask at top level are rejected because ask <= bid
    assert spread_pct is None


def test_zero_bid_not_accepted() -> None:
    quote = {"bid": 0.0, "ask": 120.0}
    bid, ask, spread_pct, source = _resolve_quote_bid_ask_spread(quote)
    # bid=0 is invalid
    assert bid is None or spread_pct is None


def test_missing_quote_returns_all_none() -> None:
    bid, ask, spread_pct, source = _resolve_quote_bid_ask_spread({})
    assert bid is None
    assert ask is None
    assert spread_pct is None
    assert source == "missing"


# ─── 7. Priority: top-level takes precedence over depth ───────────────────

def test_top_level_bid_ask_preferred_over_depth() -> None:
    """When top-level bid/ask are valid, depth prices are ignored."""
    quote = {
        "bid": 120.0, "ask": 121.0,
        "depth": {
            "buy": [{"price": 118.0, "quantity": 75}],
            "sell": [{"price": 123.0, "quantity": 50}],
        },
        "depth_available": True,
    }
    bid, ask, spread_pct, source = _resolve_quote_bid_ask_spread(quote)
    assert bid == 120.0
    assert ask == 121.0
    assert source == "top_level"


# ─── 8. Full scenario: WS tick with only depth — live entry would have failed

def test_orderflow_depth_quote_live_entry_scenario() -> None:
    """
    Scenario from production logs:
        ORDERFLOW_TRIGGER_DECISION spread_pct=0.24 trigger_conditions_met=True
        SYMBOL_LIVE_ENTRY_READY_CHECK final_ready=False reason=spread_unknown

    OrderFlow got spread_pct=0.24 from indicators (which has bid/ask from
    the indicator pipeline including depth).
    Live-entry _get_cached_quote_for_live_entry returned the raw DataHub tick
    which has depth but no "spread_pct" key.
    """
    # This is what the raw DataHub tick looks like for a Zerodha WS full-quote
    raw_datahub_tick = {
        "symbol": "NFO:NIFTY2660923200PE",
        "instrument_token": 10824962,
        "ltp": 120.5,
        "last_price": 120.5,
        "depth": {
            "buy": [
                {"price": 120.25, "quantity": 75, "orders": 3},
                {"price": 119.75, "quantity": 150, "orders": 5},
            ],
            "sell": [
                {"price": 120.75, "quantity": 50, "orders": 2},
                {"price": 121.25, "quantity": 100, "orders": 4},
            ],
        },
        "depth_available": True,
        "tradable_quote": True,
        # NO "bid" key, NO "ask" key, NO "spread_pct" key — raw Zerodha tick
    }

    bid, ask, spread_pct, source = _resolve_quote_bid_ask_spread(raw_datahub_tick)

    assert bid == 120.25, f"Expected 120.25 from depth.buy[0].price, got {bid}"
    assert ask == 120.75, f"Expected 120.75 from depth.sell[0].price, got {ask}"
    assert spread_pct is not None and spread_pct > 0, (
        f"spread_pct must be non-None/non-zero for depth quote, got {spread_pct}. "
        "This is the exact fix for the spread_unknown bug."
    )
    assert spread_pct < 1.0, f"spread_pct={spread_pct:.4f}% should be < 1.0% for tight option quote"
    assert source == "depth"
    # Sanity: would pass the 0.75% gate
    assert spread_pct <= 0.75, f"spread_pct={spread_pct:.4f}% exceeds LIVE_MAX_SPREAD_PCT=0.75"
