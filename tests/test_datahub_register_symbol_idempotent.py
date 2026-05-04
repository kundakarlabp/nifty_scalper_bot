from __future__ import annotations

import logging

from nifty_scalper_bot.data.data_hub import DataHub


class _StubMDM:
    def register_symbol(self, *_args, **_kwargs):
        return True


def test_datahub_register_symbol_idempotent(caplog) -> None:
    hub = DataHub(_StubMDM())
    with caplog.at_level(logging.INFO):
        hub.subscribe_ticks('NSE:NIFTY', token=256265)
        hub.subscribe_ticks('NSE:NIFTY', token=256265)
    deferred = [r for r in caplog.records if 'datahub_symbol_registered_deferred' in r.message]
    assert len(deferred) == 1
