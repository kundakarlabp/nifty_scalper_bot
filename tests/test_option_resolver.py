from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.options.instruments_cache import InstrumentsCache, nearest_weekly_expiry
from src.options.resolver import OptionResolver
from src.execution.micro_filters import micro_from_quote


def test_nearest_weekly_expiry():
    dt = datetime(2024, 5, 27, tzinfo=timezone.utc)  # Monday
    assert nearest_weekly_expiry(dt) == "2024-05-28"


def test_nearest_weekly_expiry_after_close():
    dt = datetime(2024, 5, 28, 10, 30, tzinfo=timezone.utc)  # Tuesday 16:00 IST
    assert nearest_weekly_expiry(dt) == "2024-06-04"


def test_option_resolver_and_micro():
    instruments = [
        {
            "name": "NIFTY",
            "expiry": datetime(2024, 5, 28),
            "strike": 22500,
            "instrument_type": "CE",
            "instrument_token": 123,
            "tradingsymbol": "NIFTY24MAY22500CE",
            "lot_size": 50,
        },
        {
            "name": "NIFTY",
            "expiry": datetime(2024, 5, 28),
            "strike": 22500,
            "instrument_type": "PE",
            "instrument_token": 456,
            "tradingsymbol": "NIFTY24MAY22500PE",
            "lot_size": 50,
        },
    ]
    cache = InstrumentsCache(instruments=instruments)
    resolver = OptionResolver(cache)
    now = datetime(2024, 5, 27)
    opt = resolver.resolve_atm("NIFTY", 22510, "CE", now)
    assert opt["token"] == 123 and opt["strike"] == 22500 and opt["tradingsymbol"]
    q = {"depth": {"buy": [{"price": 100.0, "quantity": 100}], "sell": [{"price": 100.5, "quantity": 100}]}}
    sp, ok = micro_from_quote(q, lot_size=50, depth_min_lots=1)
    assert ok is True and sp is not None


def test_instruments_cache_refreshes_from_kite():
    dump1 = [
        {
            "name": "NIFTY",
            "expiry": datetime(2024, 5, 28),
            "strike": 22500,
            "instrument_type": "CE",
            "instrument_token": 123,
            "tradingsymbol": "NIFTY24MAY22500CE",
            "lot_size": 50,
        }
    ]
    dump2 = dump1 + [
        {
            "name": "NIFTY",
            "expiry": datetime(2024, 5, 28),
            "strike": 22500,
            "instrument_type": "PE",
            "instrument_token": 456,
            "tradingsymbol": "NIFTY24MAY22500PE",
            "lot_size": 50,
        }
    ]
    kite = MagicMock()
    kite.instruments.side_effect = [dump1, dump2]
    cache = InstrumentsCache(kite=kite)
    assert cache.get("NIFTY", "2024-05-28", 22500, "CE")["token"] == 123
    assert cache.get("NIFTY", "2024-05-28", 22500, "PE")["token"] == 456
    assert kite.instruments.call_count == 2
