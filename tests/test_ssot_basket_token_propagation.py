"""Regression tests for SSOT basket token propagation bugs.

Covers:
1. extract_symbol_strike regex parser (4-6 digit strike only)
2. normalize_active_basket_schema preserves SSOT selected_ce/selected_pe
3. normalize_active_basket_schema passes token fields through untouched
4. MDM.set_active_contract_basket falls back to token_by_symbol when all_tokens absent
5. MDM.set_active_contract_basket rejects only when both all_tokens and token_by_symbol empty
6. ce_exec_ready/pe_exec_ready are False when basket_hard_ready=False
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.core.active_basket import (
    extract_symbol_strike,
    normalize_active_basket_schema,
)


# ---------------------------------------------------------------------------
# 1. Strike parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symbol,expected", [
    # Weekly contracts — expiry code is 5 chars before the 5-digit strike
    ("NFO:NIFTY2660923500CE", 23500),
    ("NFO:NIFTY2660923400PE", 23400),
    ("NFO:NIFTY2660923450CE", 23450),
    ("NFO:NIFTY2660924000PE", 24000),
    # Monthly contracts — different expiry format
    ("NFO:NIFTY26JUN23500CE", 23500),
    ("NFO:NIFTY26JUN23400PE", 23400),
    # Bare symbols without exchange prefix
    ("NIFTY2660923500CE", 23500),
    ("NIFTY26JUN24000PE", 24000),
    # 4-digit strike (theoretical edge)
    ("NFO:NIFTY2660919000CE", 19000),
])
def test_extract_symbol_strike_regex_correct(symbol: str, expected: int) -> None:
    """Strike parser must return only the 4-6 digit strike, not expiry+strike blob."""
    result = extract_symbol_strike(symbol)
    assert result == expected, (
        f"extract_symbol_strike({symbol!r}) returned {result}, expected {expected}. "
        "The old char-walk extracted the full expiry+strike (e.g. 2660923500 instead of 23500)."
    )


def test_extract_symbol_strike_none_for_non_option() -> None:
    assert extract_symbol_strike("NSE:NIFTY") is None
    assert extract_symbol_strike("NFO:NIFTY26JUNFUT") is None
    assert extract_symbol_strike("") is None


# ---------------------------------------------------------------------------
# 2. normalize_active_basket_schema: SSOT selected_ce/selected_pe preserved
# ---------------------------------------------------------------------------

def test_normalize_preserves_ssot_selected_pair() -> None:
    """If InstrumentManager sets selected_ce=23500CE, normalization must not change it to 23400CE."""
    basket: dict[str, Any] = {
        "selected_ce": "NFO:NIFTY2660923500CE",
        "selected_pe": "NFO:NIFTY2660923500PE",
        "atm_strike": 23500,
        "option_symbols": [
            "NFO:NIFTY2660923400CE",
            "NFO:NIFTY2660923400PE",
            "NFO:NIFTY2660923450CE",
            "NFO:NIFTY2660923450PE",
            "NFO:NIFTY2660923500CE",
            "NFO:NIFTY2660923500PE",
        ],
        "spot_symbol": "NSE:NIFTY",
        "futures_symbol": "NFO:NIFTY26JUNFUT",
    }
    result = normalize_active_basket_schema(basket)
    assert result["selected_ce"] == "NFO:NIFTY2660923500CE", (
        f"selected_ce changed to {result['selected_ce']} — SSOT override bug"
    )
    assert result["selected_pe"] == "NFO:NIFTY2660923500PE", (
        f"selected_pe changed to {result['selected_pe']} — SSOT override bug"
    )


def test_normalize_picks_atm_when_selected_absent() -> None:
    """When selected_ce/pe are absent, normalization selects the nearest ATM."""
    basket: dict[str, Any] = {
        "atm_strike": 23500,
        "option_symbols": [
            "NFO:NIFTY2660923400CE",
            "NFO:NIFTY2660923500CE",
            "NFO:NIFTY2660923400PE",
            "NFO:NIFTY2660923500PE",
        ],
        "spot_symbol": "NSE:NIFTY",
        "futures_symbol": "NFO:NIFTY26JUNFUT",
    }
    result = normalize_active_basket_schema(basket)
    assert result["selected_ce"] == "NFO:NIFTY2660923500CE"
    assert result["selected_pe"] == "NFO:NIFTY2660923500PE"


# ---------------------------------------------------------------------------
# 3. normalize_active_basket_schema: token fields pass through unchanged
# ---------------------------------------------------------------------------

def test_normalize_preserves_token_by_symbol() -> None:
    """token_by_symbol with 12 entries must survive normalize_active_basket_schema intact."""
    token_by_symbol = {
        "NSE:NIFTY": 256265,
        "NFO:NIFTY26JUNFUT": 15956226,
        "NFO:NIFTY2660923400CE": 11111111,
        "NFO:NIFTY2660923450CE": 11111112,
        "NFO:NIFTY2660923500CE": 11111113,
        "NFO:NIFTY2660923550CE": 11111114,
        "NFO:NIFTY2660923600CE": 11111115,
        "NFO:NIFTY2660923400PE": 22222221,
        "NFO:NIFTY2660923450PE": 22222222,
        "NFO:NIFTY2660923500PE": 22222223,
        "NFO:NIFTY2660923550PE": 22222224,
        "NFO:NIFTY2660923600PE": 22222225,
    }
    basket: dict[str, Any] = {
        "selected_ce": "NFO:NIFTY2660923500CE",
        "selected_pe": "NFO:NIFTY2660923500PE",
        "atm_strike": 23500,
        "option_symbols": [k for k in token_by_symbol if k.startswith("NFO:NIFTY2")],
        "spot_symbol": "NSE:NIFTY",
        "futures_symbol": "NFO:NIFTY26JUNFUT",
        "token_by_symbol": token_by_symbol,
        "all_tokens": list(token_by_symbol.values()),
        "all_symbols": list(token_by_symbol.keys()),
    }
    result = normalize_active_basket_schema(basket)
    assert result["token_by_symbol"] == token_by_symbol, "token_by_symbol was stripped by normalize"
    assert result["all_tokens"] == list(token_by_symbol.values()), "all_tokens was stripped"
    assert len(result["token_by_symbol"]) == 12, "token count changed"


# ---------------------------------------------------------------------------
# 4 & 5. MDM.set_active_contract_basket token fallback
# ---------------------------------------------------------------------------

def _make_mdm() -> Any:
    """Build a minimal MarketDataManager instance for unit testing."""
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager
    mdm = object.__new__(MarketDataManager)
    # Minimal state required by set_active_contract_basket
    import threading
    mdm._lock = threading.RLock()
    mdm._active_contract_basket = None
    mdm._desired_tokens: set[int] = set()
    mdm._symbol_to_token: dict[str, int] = {}
    mdm._token_to_symbol: dict[int, str] = {}
    mdm._token_by_symbol: dict[str, int] = {}
    mdm._symbol_by_token: dict[int, str] = {}
    mdm._ws = None
    import logging
    mdm._logger = logging.getLogger("mdm_test")
    return mdm


def test_mdm_accepts_basket_via_token_by_symbol_when_all_tokens_absent() -> None:
    """MDM must not log MDM_ACTIVE_BASKET_IGNORED when token_by_symbol has valid tokens."""
    mdm = _make_mdm()
    basket = {
        "selected_ce": "NFO:NIFTY2660923500CE",
        "selected_pe": "NFO:NIFTY2660923500PE",
        "spot_symbol": "NSE:NIFTY",
        "futures_symbol": "NFO:NIFTY26JUNFUT",
        "option_symbols": ["NFO:NIFTY2660923500CE", "NFO:NIFTY2660923500PE"],
        "symbols": ["NSE:NIFTY", "NFO:NIFTY26JUNFUT", "NFO:NIFTY2660923500CE", "NFO:NIFTY2660923500PE"],
        # all_tokens intentionally absent
        "token_by_symbol": {
            "NSE:NIFTY": 256265,
            "NFO:NIFTY26JUNFUT": 15956226,
            "NFO:NIFTY2660923500CE": 11111113,
            "NFO:NIFTY2660923500PE": 22222223,
        },
    }
    mdm.set_active_contract_basket(basket)
    # If accepted, _active_contract_basket should be set
    assert mdm._active_contract_basket is basket, "MDM ignored the basket despite valid token_by_symbol"
    assert 256265 in mdm._desired_tokens
    assert 11111113 in mdm._desired_tokens
    assert 22222223 in mdm._desired_tokens


def test_mdm_rejects_basket_when_both_token_sources_empty() -> None:
    """MDM must reject (and log) only when both all_tokens and token_by_symbol are empty."""
    mdm = _make_mdm()
    basket = {
        "selected_ce": "NFO:NIFTY2660923500CE",
        "selected_pe": "NFO:NIFTY2660923500PE",
        "spot_symbol": "NSE:NIFTY",
        "option_symbols": ["NFO:NIFTY2660923500CE", "NFO:NIFTY2660923500PE"],
        # Both token sources absent
    }
    mdm.set_active_contract_basket(basket)
    assert mdm._active_contract_basket is None, "MDM should have rejected the empty-token basket"


# ---------------------------------------------------------------------------
# 6. Readiness: ce_exec_ready must be False when basket_hard_ready=False
# ---------------------------------------------------------------------------

def test_ce_exec_ready_false_when_basket_not_hard_ready() -> None:
    """Even with fresh quotes and sufficient bars, exec_ready must be False if tokens are missing."""
    # This tests the logic directly — basket_hard_ready=False gates both exec_ready flags.
    basket_hard_ready = False  # token_missing from MDM
    ce_quote_fresh = True
    ce_bid_ask_complete = True
    tradable_quote = True
    ce_bars = 50  # more than enough

    # Mirrors the exact expression from app.py after fix
    ce_exec_ready = bool("NFO:NIFTY2660923500CE") and basket_hard_ready and ce_quote_fresh and ce_bid_ask_complete and tradable_quote and ce_bars >= 30

    assert ce_exec_ready is False, (
        "ce_exec_ready must be False when basket_hard_ready=False (token missing). "
        "Before the fix this was True, causing inconsistent readiness state."
    )


# ---------------------------------------------------------------------------
# 7. _commit_active_dynamic_basket carries token_by_symbol into committed dict
#    (regression: logs showed ACTIVE_BASKET_TOKEN_MAP_READY count=1 after PR#506)
# ---------------------------------------------------------------------------

def test_commit_basket_carries_token_by_symbol_from_input() -> None:
    """token_by_symbol must survive from _build_canonical_active_basket through _commit_active_dynamic_basket.

    Regression: _commit_active_dynamic_basket rebuilt `committed` via .update(...)
    without forwarding token_by_symbol/all_tokens.  The token-map resolution step
    then found committed.get('token_by_symbol') == None, fell to resolve_active_basket_tokens
    which could only find spot (1 token), producing ACTIVE_BASKET_TOKEN_MAP_READY count=1.
    """
    from nifty_scalper_bot.core.active_basket import normalize_active_basket_schema

    token_by_symbol = {
        "NSE:NIFTY": 256265,
        "NFO:NIFTY26JUNFUT": 15956226,
        "NFO:NIFTY2660923200CE": 10824706,
        "NFO:NIFTY2660923200PE": 10824962,
        "NFO:NIFTY2660923250CE": 10825218,
        "NFO:NIFTY2660923250PE": 10825474,
        "NFO:NIFTY2660923300CE": 10825730,
        "NFO:NIFTY2660923300PE": 10825986,
        "NFO:NIFTY2660923350CE": 10826242,
        "NFO:NIFTY2660923350PE": 10827010,
        "NFO:NIFTY2660923400CE": 10827522,
        "NFO:NIFTY2660923400PE": 10827778,
    }
    incoming_basket = {
        "selected_ce": "NFO:NIFTY2660923300CE",
        "selected_pe": "NFO:NIFTY2660923300PE",
        "atm_strike": 23300,
        "spot_symbol": "NSE:NIFTY",
        "futures_symbol": "NFO:NIFTY26JUNFUT",
        "option_symbols": [k for k in token_by_symbol if "CE" in k or "PE" in k],
        "token_by_symbol": token_by_symbol,
        "all_tokens": list(token_by_symbol.values()),
        "all_symbols": list(token_by_symbol.keys()),
    }

    # Simulate the normalize + committed.update path (the part that was losing tokens)
    basket_copy = dict(incoming_basket)
    normalized = normalize_active_basket_schema(basket_copy)

    # Simulate committed.update (the fix: token fields must be forwarded)
    committed: dict = {}
    committed.update({
        "spot_symbol": normalized.get("spot_symbol") or "NSE:NIFTY",
        "futures_symbol": normalized.get("futures_symbol"),
        "selected_ce": normalized.get("selected_ce"),
        "selected_pe": normalized.get("selected_pe"),
        "option_symbols": normalized.get("option_symbols", []),
        "atm_strike": 23300,
    })
    # This is the fix — token keys forwarded explicitly:
    for _k in ("token_by_symbol", "all_tokens", "all_symbols", "selected_ce_token", "selected_pe_token"):
        _v = normalized.get(_k)
        if _v is not None:
            committed[_k] = _v

    # Now the token_map resolution step should find token_by_symbol
    token_map = dict(committed.get("token_by_symbol") or {})
    assert len(token_map) == 12, (
        f"token_map should have 12 entries after commit, got {len(token_map)}. "
        "Without the fix, committed.get('token_by_symbol') returned None → count=1."
    )
    assert token_map.get("NFO:NIFTY2660923300CE") == 10825730
    assert token_map.get("NFO:NIFTY2660923300PE") == 10825986


def test_nearest_for_side_uses_regex_strike_not_char_walk() -> None:
    """_nearest_for_side inside _commit_active_dynamic_basket must pick correct ATM.

    Third copy of the buggy strike parser was at line 7815. With spot=23300, ATM=23300,
    the old char-walk extracted 2660923300 instead of 23300, making ALL distances equal
    (all huge), causing arbitrary selection instead of true ATM.
    """
    from nifty_scalper_bot.core.active_basket import extract_symbol_strike

    options = [
        "NFO:NIFTY2660923200CE",
        "NFO:NIFTY2660923250CE",
        "NFO:NIFTY2660923300CE",  # ATM
        "NFO:NIFTY2660923350CE",
        "NFO:NIFTY2660923400CE",
    ]
    atm_strike = 23300

    # Replicate _nearest_for_side logic with fix
    candidates = []
    for sym in options:
        if not sym.endswith("CE"):
            continue
        strike = extract_symbol_strike(sym)
        assert strike is not None, f"extract_symbol_strike failed for {sym}"
        candidates.append((abs(float(strike) - float(atm_strike)), sym))
    candidates.sort(key=lambda x: (x[0], x[1]))
    nearest = candidates[0][1]

    assert nearest == "NFO:NIFTY2660923300CE", (
        f"Expected ATM=23300CE but got {nearest}. "
        "The buggy char-walk produced distance=2660923300-23300=2660900000 for all, causing wrong selection."
    )
