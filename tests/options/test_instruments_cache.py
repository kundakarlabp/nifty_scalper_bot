from src.options.instruments_cache import InstrumentsCache


def test_instruments_cache_skips_nan_entries() -> None:
    bad_instrument = {
        "name": "NIFTY",
        "expiry": "2024-05-23",
        "strike": float("nan"),
        "instrument_type": "CE",
        "instrument_token": float("nan"),
        "tradingsymbol": "NIFTY24MAYBAD",
        "lot_size": float("nan"),
    }
    good_instrument = {
        "name": "NIFTY",
        "expiry": "2024-05-30",
        "strike": 18500,
        "instrument_type": "CE",
        "instrument_token": 123456,
        "tradingsymbol": "NIFTY24MAY18500CE",
        "lot_size": 50,
    }

    cache = InstrumentsCache(instruments=[bad_instrument, good_instrument])

    assert cache.get("NIFTY", "2024-05-23", 0, "CE") is None
    good_entry = cache.get("NIFTY", "2024-05-30", 18500, "CE")
    assert good_entry is not None
    assert good_entry["token"] == 123456
    assert good_entry["lot_size"] == 50
