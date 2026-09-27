"""Data layer exports used across the runtime.

MarketDataManager, WebSocket, and DataHub hardening are owned at their class
definition sites. Importing this package has no runtime patching side effects.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

from nifty_scalper_bot.brokers.instrument_lookup import Instrument
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
from nifty_scalper_bot.data.instrument_resolver import InstrumentResolver


def __getattr__(name: str) -> ModuleType:
    if name == "rest":
        module = importlib.import_module("nifty_scalper_bot.data.rest")
        setattr(sys.modules[__name__], name, module)
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
