"""Data layer exports used across the runtime.

Hardening note: MarketDataManager hardening is installed explicitly in
``market_data_manager.py`` and the IST time adapter in ``source.py`` — at
their definition sites. DataHub quote timestamp-quality guarding is installed
when this package is imported so direct ``data_hub`` imports get the same live
readiness safety.
"""

from __future__ import annotations

import logging

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

try:
    from nifty_scalper_bot.data.data_hub import DataHub as _DataHub
    from nifty_scalper_bot.data.data_hub_synthetic_guard import (
        install_datahub_synthetic_timestamp_guard as _install_datahub_synthetic_guard,
    )

    _install_datahub_synthetic_guard(_DataHub)
except Exception as exc:  # noqa: BLE001 - data package imports must remain usable
    logging.getLogger(__name__).error(
        "DATAHUB_SYNTHETIC_TIMESTAMP_GUARD_FAILED error=%s",
        exc,
        extra={"event": "DATAHUB_SYNTHETIC_TIMESTAMP_GUARD_FAILED", "error_type": type(exc).__name__},
    )

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
