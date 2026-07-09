from __future__ import annotations

import importlib
import sys


def _reset_data_modules() -> None:
    for name in list(sys.modules):
        if name == "nifty_scalper_bot.data" or name.startswith("nifty_scalper_bot.data.data_hub"):
            sys.modules.pop(name, None)


def test_data_package_import_does_not_eagerly_import_datahub() -> None:
    _reset_data_modules()

    importlib.import_module("nifty_scalper_bot.data")

    assert "nifty_scalper_bot.data.data_hub" not in sys.modules


def test_direct_datahub_import_installs_synthetic_timestamp_guard() -> None:
    _reset_data_modules()

    module = importlib.import_module("nifty_scalper_bot.data.data_hub")

    assert getattr(module.DataHub, "_synthetic_timestamp_guard_installed", False) is True


def test_direct_datahub_import_synthetic_quote_is_guarded() -> None:
    _reset_data_modules()
    module = importlib.import_module("nifty_scalper_bot.data.data_hub")

    class _MdmNoop:
        def attach_tick_bus(self, _tick_bus) -> None:
            return None

    hub = module.DataHub(_MdmNoop())
    symbol = "NFO:NIFTY26JUL24000CE"
    hub.store_quote(
        symbol,
        {
            "instrument_token": 12345,
            "ltp": 100.0,
            "bid": 99.5,
            "ask": 100.5,
            "depth_available": True,
        },
        source="ws",
    )
    quote = hub.get_quote(symbol, allow_pull=False)

    assert quote is not None
    assert quote["timestamp_quality"] == "synthetic"
    assert quote["hard_readiness_eligible"] is False
    assert hub.get_cached_ltp(symbol, max_age_seconds=2.0) is None
