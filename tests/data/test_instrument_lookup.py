from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.brokers.instrument_lookup import InstrumentLookup


def test_instrument_lookup_missing_real_csv_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        InstrumentLookup(str(missing))


def test_instrument_lookup_magicmock_enables_in_memory_mode() -> None:
    resolver = InstrumentLookup(MagicMock())
    resolver.upsert("NIFTY25OCT25900CE", 123, exchange="NFO")
    assert resolver.resolve("NIFTY25OCT25900CE", exchange="NFO") == 123


def test_upsert_populates_option_field_lookup() -> None:
    resolver = InstrumentLookup(None)
    resolver.upsert(
        "NIFTY25OCT25900CE",
        123,
        exchange="NFO",
        expiry="2025-10-30",
        strike=25900,
        option_type="CE",
    )
    token = resolver.get_token_by_fields(
        exchange="NFO",
        underlying="NIFTY",
        expiry="2025-10-30",
        strike=25900,
        option_type="CE",
    )
    assert token == 123
