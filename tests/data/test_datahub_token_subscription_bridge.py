from __future__ import annotations

from typing import Any

from nifty_scalper_bot.data.data_hub import DataHub


class StubMdm:
    def __init__(self) -> None:
        self._token_by_symbol: dict[str, int] = {}
        self._symbol_to_token: dict[str, int] = {}


def test_token_subscription_bridges_symbol_alias_without_duplication() -> None:
    hub = DataHub(StubMdm())
    received: list[dict[str, Any]] = []
    callback = lambda t: received.append(t)
    hub.subscribe_ticks('NFO:NIFTY26MAY23950CE', callback, token=123)

    hub.ingest_tick_sync({'symbol': 'NFO:NIFTY2650523950CE', 'instrument_token': 123, 'ltp': 100.0})

    assert len(received) == 1


def test_alias_and_token_match_still_emits_once_and_unsubscribe_cleans() -> None:
    hub = DataHub(StubMdm())
    received: list[dict[str, Any]] = []

    def cb(tick: dict[str, Any]) -> None:
        received.append(tick)

    hub.subscribe_ticks('NFO:NIFTY26MAY23950CE', cb, token=123)
    hub.ingest_tick_sync({'symbol': 'NFO:NIFTY2650523950CE', 'instrument_token': 123, 'ltp': 101.0})
    assert len(received) == 1

    status_before = hub.debug_subscription_status('NFO:NIFTY26MAY23950CE', 123)
    assert status_before['token_callbacks'] > 0

    hub.unsubscribe_ticks('NFO:NIFTY26MAY23950CE', cb)
    status_after = hub.debug_subscription_status('NFO:NIFTY26MAY23950CE', 123)
    assert status_after['token_callbacks'] == 0
