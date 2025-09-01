from datetime import datetime, timezone

from src.options.instruments_cache import InstrumentsCache, nearest_weekly_expiry
from src.options.resolver import OptionResolver
from src.execution.micro_filters import micro_from_l1


def test_nearest_weekly_expiry():
    dt = datetime(2024, 5, 27, tzinfo=timezone.utc)  # Monday
    assert nearest_weekly_expiry(dt) == "2024-05-30"


def test_option_resolver_and_micro():
    instruments = [
        {
            "name": "NIFTY",
            "expiry": datetime(2024, 5, 30),
            "strike": 22500,
            "instrument_type": "CE",
            "instrument_token": 123,
            "lot_size": 50,
        },
        {
            "name": "NIFTY",
            "expiry": datetime(2024, 5, 30),
            "strike": 22500,
            "instrument_type": "PE",
            "instrument_token": 456,
            "lot_size": 50,
        },
    ]
    cache = InstrumentsCache(instruments=instruments)
    resolver = OptionResolver(cache)
    now = datetime(2024, 5, 27)
    opt = resolver.resolve_atm("NIFTY", 22510, "CE", now)
    assert opt["token"] == 123 and opt["strike"] == 22500
    l1 = {"depth": {"buy": [{"price": 100.0, "quantity": 100}], "sell": [{"price": 100.5, "quantity": 100}]}}
    sp, ok, extra = micro_from_l1(l1, lot_size=50, depth_min_lots=1)
    assert ok is True and sp is not None and extra["bid"] == 100.0
