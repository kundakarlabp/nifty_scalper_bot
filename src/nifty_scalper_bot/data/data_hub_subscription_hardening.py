"""Keep DataHub's direct MDM ingress aligned with the active basket."""

from __future__ import annotations

import logging
from typing import Any, Mapping

_INSTALLED_ATTR = "_active_basket_subscription_hardening_installed"
_ORIGINAL_ATTR = "_active_basket_subscription_hardening_original_set_basket"


def _basket_get(basket: Any, key: str, default: Any = None) -> Any:
    if isinstance(basket, Mapping):
        return basket.get(key, default)
    return getattr(basket, key, default)


def _basket_symbols(basket: Any) -> list[str]:
    values: list[Any] = []
    values.extend(_basket_get(basket, "all_symbols", ()) or ())
    values.extend(_basket_get(basket, "symbols", ()) or ())
    values.extend(_basket_get(basket, "option_symbols", ()) or ())
    values.extend(
        [
            _basket_get(basket, "spot_symbol"),
            _basket_get(basket, "futures_symbol"),
            _basket_get(basket, "selected_ce"),
            _basket_get(basket, "selected_pe"),
            _basket_get(basket, "atm_ce"),
            _basket_get(basket, "atm_pe"),
        ]
    )
    return [str(value) for value in dict.fromkeys(values) if value]


def install_data_hub_subscription_hardening(data_hub_cls: type[Any]) -> None:
    """Subscribe every active-basket symbol through DataHub's direct MDM path."""
    if bool(getattr(data_hub_cls, _INSTALLED_ATTR, False)):
        return

    original = getattr(data_hub_cls, "set_active_contract_basket")
    setattr(data_hub_cls, _ORIGINAL_ATTR, original)

    def _set_active_contract_basket(self: Any, basket: Any) -> None:
        original(self, basket)
        token_by_symbol = dict(_basket_get(basket, "token_by_symbol", {}) or {})
        deferred = bool(getattr(self, "_defer_live_symbol_subscriptions", True))
        subscribe_ticks = getattr(self, "subscribe_ticks", None)
        if not callable(subscribe_ticks):
            return

        failed: list[str] = []
        subscribed: list[str] = []
        for symbol in _basket_symbols(basket):
            token = token_by_symbol.get(symbol)
            try:
                subscribe_ticks(
                    symbol,
                    token=int(token) if token is not None else None,
                    force_live=not deferred,
                )
                subscribed.append(symbol)
            except Exception as exc:  # noqa: BLE001 - one symbol must not abort basket commit
                failed.append(symbol)
                logging.getLogger(__name__).error(
                    "DATAHUB_BASKET_SUBSCRIPTION_FAILED symbol=%s error=%r",
                    symbol,
                    exc,
                    exc_info=True,
                    extra={
                        "event": "DATAHUB_BASKET_SUBSCRIPTION_FAILED",
                        "symbol": symbol,
                        "error_type": type(exc).__name__,
                    },
                )
        logging.getLogger(__name__).info(
            "DATAHUB_BASKET_SUBSCRIPTIONS_SYNCED symbols=%d failed=%d deferred=%s",
            len(subscribed),
            len(failed),
            deferred,
            extra={
                "event": "DATAHUB_BASKET_SUBSCRIPTIONS_SYNCED",
                "symbols": subscribed,
                "failed_symbols": failed,
                "deferred": deferred,
            },
        )

    setattr(data_hub_cls, "set_active_contract_basket", _set_active_contract_basket)
    setattr(data_hub_cls, _INSTALLED_ATTR, True)


__all__ = ["install_data_hub_subscription_hardening"]
