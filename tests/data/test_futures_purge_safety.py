from __future__ import annotations

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def _complete_basket() -> dict[str, object]:
    ce = "NFO:NIFTY26JUN25000CE"
    pe = "NFO:NIFTY26JUN25000PE"
    fut = "NFO:NIFTY26JUNFUT"
    return {
        "spot_symbol": "NSE:NIFTY",
        "spot_token": 256265,
        "futures_symbol": fut,
        "futures_token": 222,
        "selected_ce": ce,
        "selected_pe": pe,
        "selected_ce_token": 11,
        "selected_pe_token": 12,
        "option_symbols": [ce, pe],
        "option_tokens": [11, 12],
        "all_symbols": ["NSE:NIFTY", fut, ce, pe],
        "all_tokens": [256265, 222, 11, 12],
        "token_by_symbol": {"NSE:NIFTY": 256265, fut: 222, ce: 11, pe: 12},
        "symbol_by_token": {256265: "NSE:NIFTY", 222: fut, 11: ce, 12: pe},
        "atm_strike": 25000,
    }


def test_purge_never_removes_spot_active_future_or_selected_options() -> None:
    mdm = MarketDataManager(websocket=None)
    basket = _complete_basket()
    mdm.set_active_contract_basket(basket)

    purged = mdm.purge_stale_nifty_futures("NFO:NIFTY26JUNFUT", reason="unit_test")

    assert purged == []
    assert 256265 in mdm._desired_tokens  # noqa: SLF001
    assert 222 in mdm._desired_tokens  # noqa: SLF001
    assert 11 in mdm._desired_tokens and 12 in mdm._desired_tokens  # noqa: SLF001


def test_purge_removes_only_old_nifty_future_token() -> None:
    mdm = MarketDataManager(websocket=None)
    basket = _complete_basket()
    mdm.set_active_contract_basket(basket)
    old = "NFO:NIFTY26MAYFUT"
    mdm._symbol_by_token[333] = old  # noqa: SLF001
    mdm._token_to_symbol[333] = old  # noqa: SLF001
    mdm._token_by_symbol[old] = 333  # noqa: SLF001
    mdm._symbol_to_token[old] = 333  # noqa: SLF001
    mdm._desired_tokens.add(333)  # noqa: SLF001

    purged = mdm.purge_stale_nifty_futures("NFO:NIFTY26JUNFUT", reason="unit_test")

    assert purged == [333]
    assert 333 not in mdm._desired_tokens  # noqa: SLF001
    assert 256265 in mdm._desired_tokens  # noqa: SLF001


def test_unknown_token_not_purged_without_symbol_identity() -> None:
    mdm = MarketDataManager(websocket=None)
    basket = _complete_basket()
    mdm.set_active_contract_basket(basket)
    mdm._desired_tokens.add(999)  # noqa: SLF001

    purged = mdm.purge_stale_nifty_futures("NFO:NIFTY26JUNFUT", reason="unit_test")

    assert purged == []
    assert 999 in mdm._desired_tokens  # noqa: SLF001
