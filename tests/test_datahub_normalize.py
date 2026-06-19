"""Regression test: DataHub.normalize must exist and match normalize_symbol.

~25 call sites across execution/risk/notifications use ``DataHub.normalize``
as a static helper, but it was never defined on the class — every reached
call raised ``AttributeError: type object 'DataHub' has no attribute
'normalize'``, silently aborting position sync/adoption.
"""

from __future__ import annotations

from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.utils.symbols import normalize_symbol


async def test_datahub_normalize_exists() -> None:
    assert hasattr(DataHub, "normalize")


async def test_datahub_normalize_matches_helper() -> None:
    cases = [
        "NIFTY2662324050PE",
        "NFO:NIFTY26JUNFUT",
        "NIFTY",
        "nse:nifty",
        "",
        "256265",
    ]
    for sym in cases:
        assert DataHub.normalize(sym) == normalize_symbol(sym)


async def test_datahub_normalize_callable_on_instance() -> None:
    # Some call sites invoke it on an instance (hub.normalize(sym)); a
    # staticmethod must work both ways without needing instance state.
    hub = DataHub.__new__(DataHub)
    assert hub.normalize("NIFTY2662324050CE") == "NFO:NIFTY2662324050CE"
