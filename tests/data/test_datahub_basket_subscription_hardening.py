from __future__ import annotations

from typing import Any

from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.data.data_hub_subscription_hardening import (
    install_data_hub_subscription_hardening,
)


class _FakeMDM:
    def __init__(self) -> None:
        self.subscriptions: list[tuple[str, Any]] = []
        self.unsubscriptions: list[tuple[str, Any]] = []

    def attach_tick_bus(self, _bus: Any) -> None:
        return None

    def subscribe(self, symbol: str, callback: Any) -> None:
        self.subscriptions.append((symbol, callback))

    def unsubscribe(self, symbol: str, callback: Any) -> None:
        self.unsubscriptions.append((symbol, callback))


def _basket(strike: int) -> dict[str, Any]:
    ce = f"NFO:NIFTY26804{strike}CE"
    pe = f"NFO:NIFTY26804{strike}PE"
    symbols = ["NSE:NIFTY", "NFO:NIFTY26AUGFUT", ce, pe]
    return {
        "all_symbols": symbols,
        "option_symbols": [ce, pe],
        "spot_symbol": "NSE:NIFTY",
        "futures_symbol": "NFO:NIFTY26AUGFUT",
        "selected_ce": ce,
        "selected_pe": pe,
        "token_by_symbol": {symbol: index + 1 for index, symbol in enumerate(symbols)},
    }


def test_live_basket_rotation_registers_new_direct_mdm_ingress_once() -> None:
    install_data_hub_subscription_hardening(DataHub)
    mdm = _FakeMDM()
    hub = DataHub(mdm, defer_live_symbol_subscriptions=False)
    try:
        first = _basket(24600)
        second = _basket(24650)

        hub.set_active_contract_basket(first)
        hub.set_active_contract_basket(second)
        hub.set_active_contract_basket(second)

        symbols = [symbol for symbol, _callback in mdm.subscriptions]
        assert symbols.count(first["selected_ce"]) == 1
        assert symbols.count(first["selected_pe"]) == 1
        assert symbols.count(second["selected_ce"]) == 1
        assert symbols.count(second["selected_pe"]) == 1
        assert symbols.count("NSE:NIFTY") == 1
        assert symbols.count("NFO:NIFTY26AUGFUT") == 1
        assert all(callback == hub.ingest_tick_sync for _symbol, callback in mdm.subscriptions)
    finally:
        hub.close()


def test_startup_basket_remains_deferred_until_explicit_flush() -> None:
    install_data_hub_subscription_hardening(DataHub)
    mdm = _FakeMDM()
    hub = DataHub(mdm, defer_live_symbol_subscriptions=True)
    try:
        basket = _basket(24600)

        hub.set_active_contract_basket(basket)

        assert mdm.subscriptions == []
        assert set(basket["all_symbols"]).issubset(hub._pending_live_symbols)

        flushed = hub.flush_pending_live_subscriptions()

        assert flushed == len(set(basket["all_symbols"]))
        assert {symbol for symbol, _callback in mdm.subscriptions} == set(
            basket["all_symbols"]
        )
    finally:
        hub.close()


def test_installer_is_idempotent() -> None:
    install_data_hub_subscription_hardening(DataHub)
    wrapped = DataHub.set_active_contract_basket

    install_data_hub_subscription_hardening(DataHub)

    assert DataHub.set_active_contract_basket is wrapped
