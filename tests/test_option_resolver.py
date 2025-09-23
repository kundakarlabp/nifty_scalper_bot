from datetime import datetime
from unittest.mock import MagicMock, patch

from src.execution.micro_filters import micro_from_quote
from src.options.contract_registry import InstrumentRegistry
from src.options.instruments_cache import InstrumentsCache
from src.options.resolver import OptionResolver


def _write_registry(tmp_path) -> InstrumentRegistry:
    csv_content = """instrument_token,tradingsymbol,name,expiry,strike,instrument_type,lot_size\n"""
    csv_content += "123,NIFTY24MAY22500CE,NIFTY,2024-05-28,22500,CE,75\n"
    csv_content += "456,NIFTY24MAY22500PE,NIFTY,2024-05-28,22500,PE,75\n"
    path = tmp_path / "contracts.csv"
    path.write_text(csv_content)
    registry = InstrumentRegistry(source=path)
    registry.refresh(force=True)
    return registry


def test_registry_next_weekly_expiry(tmp_path):
    registry = _write_registry(tmp_path)
    expiry = registry.next_weekly_expiry("NIFTY", datetime(2024, 5, 27))
    assert expiry is not None
    assert expiry.date().isoformat() == "2024-05-28"
    assert registry.lot_size("NIFTY", expiry.date().isoformat()) == 75


def test_option_resolver_and_micro(tmp_path):
    instruments = [
        {
            "name": "NIFTY",
            "expiry": datetime(2024, 5, 28),
            "strike": 22500,
            "instrument_type": "CE",
            "instrument_token": 123,
            "tradingsymbol": "NIFTY24MAY22500CE",
            "lot_size": 75,
        },
        {
            "name": "NIFTY",
            "expiry": datetime(2024, 5, 28),
            "strike": 22500,
            "instrument_type": "PE",
            "instrument_token": 456,
            "tradingsymbol": "NIFTY24MAY22500PE",
            "lot_size": 75,
        },
    ]
    cache = InstrumentsCache(instruments=instruments)
    registry = _write_registry(tmp_path)
    resolver = OptionResolver(cache, registry=registry)
    opt = resolver.resolve_atm("NIFTY", 22510, "CE", datetime(2024, 5, 27))
    assert opt["token"] == 123
    assert opt["lot_size"] == 75
    q = {"depth": {"buy": [{"price": 100.0, "quantity": 100}], "sell": [{"price": 100.5, "quantity": 100}]}}
    sp, ok = micro_from_quote(q, lot_size=opt["lot_size"], depth_min_lots=1)
    assert ok is True and sp is not None


def test_option_resolver_fetches_missing_token():
    """Falls back to Kite LTP when cache lacks the option token."""
    cache = InstrumentsCache(instruments=[])
    kite = MagicMock()
    kite.ltp.return_value = {"NFO:NIFTY24MAY22500CE": {"instrument_token": 789}}
    with patch("src.options.resolver.resolve_weekly_atm") as r:
        r.return_value = {"ce": ("NIFTY24MAY22500CE", 50)}
        resolver = OptionResolver(cache, kite)
        opt = resolver.resolve_atm("NIFTY", 22510, "CE", datetime(2024, 5, 27))
    assert opt["token"] == 789


def test_option_resolver_fetches_instruments(monkeypatch):
    """Fetches instrument dump when cache lacks option token."""
    cache = InstrumentsCache(instruments=[])
    kite = MagicMock()
    inst_dump = [
        {
            "name": "NIFTY",
            "segment": "NFO-OPT",
            "expiry": datetime(2024, 5, 28).date(),
            "instrument_type": "CE",
            "strike": 22500,
            "tradingsymbol": "NIFTY24MAY22500CE",
            "lot_size": 50,
        }
    ]
    monkeypatch.setattr(
        "src.options.resolver._fetch_instruments_nfo", lambda _k: inst_dump
    )
    kite.ltp.return_value = {
        "NFO:NIFTY24MAY22500CE": {"instrument_token": 321}
    }
    resolver = OptionResolver(cache, kite)
    opt = resolver.resolve_atm("NIFTY", 22510, "CE", datetime(2024, 5, 27))
    assert opt["token"] == 321
