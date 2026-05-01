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
    hub.subscribe_ticks('NFO:NIFTY26MAY23950CE', lambda t: received.append(t), token=123)

    hub.ingest_tick_sync({'symbol': 'NFO:NIFTY2650523950CE', 'instrument_token': 123, 'ltp': 100.0})

    assert len(received) == 1
    assert hub.get_quote('NFO:NIFTY26MAY23950CE', allow_pull=False) is not None
    assert hub.get_tick_by_token(123) is not None
