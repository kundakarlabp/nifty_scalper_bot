"""Data layer exports used across the runtime."""

from __future__ import annotations

import importlib

from nifty_scalper_bot.brokers.instrument_lookup import Instrument
from nifty_scalper_bot.data.instrument_resolver import InstrumentResolver
from nifty_scalper_bot.data.instrument_loader import (
    InstrumentUniverseStatus,
    ensure_sqlite,
    load_rows_for_resolver,
    parse_kite_csv,
    refresh_from_csv,
    sync_instrument_csv_from_broker,
    upsert_instruments,
    write_instrument_rows_to_csv,
)
from nifty_scalper_bot.utils.time_apply import apply as _apply_time_adapter

try:
    _apply_time_adapter(importlib.import_module("nifty_scalper_bot.data." + "source"))
except Exception:
    pass

try:
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager
    from nifty_scalper_bot.data.market_data_hardening import (
        install_market_data_manager_hardening,
    )

    install_market_data_manager_hardening(MarketDataManager)
except Exception:
    pass

__all__ = [
    "Instrument",
    "InstrumentResolver",
    "InstrumentUniverseStatus",
    "ensure_sqlite",
    "load_rows_for_resolver",
    "parse_kite_csv",
    "refresh_from_csv",
    "sync_instrument_csv_from_broker",
    "upsert_instruments",
    "write_instrument_rows_to_csv",
]
