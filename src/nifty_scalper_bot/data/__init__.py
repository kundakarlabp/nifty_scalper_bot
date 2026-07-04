"""Data layer exports used across the runtime."""

from __future__ import annotations

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

# Import for side effect: DataHub quotes are stamped with canonical symbol,
# token, update-version and tick-age metadata before execution can consume them.
from nifty_scalper_bot.data import quote_identity_extension as _quote_identity_extension

_quote_identity_extension.apply_patches()

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
