from __future__ import annotations

import logging

from nifty_scalper_bot.data.data_hub import DataHub


class MDM:
    def get_active_contract_basket(self):
        return {"selected_ce": "NFO:ABC25000CE", "selected_pe": "NFO:ABC25000PE"}


def test_datahub_missing_quote_logs_are_throttled(caplog):
    hub = DataHub(MDM())
    hub._missing_data_log_cooldown_sec = 30.0
    caplog.set_level(logging.WARNING)
    assert hub.get_quote("NFO:ABC25000CE", allow_pull=False) is None
    assert hub.get_quote("NFO:ABC25000CE", allow_pull=False) is None
    assert caplog.text.count("DATAHUB_QUOTE_MISSING") == 1
