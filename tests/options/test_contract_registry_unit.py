from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import zipfile
from zoneinfo import ZoneInfo

from src.options.contract_registry import (
    InstrumentRegistry,
    _normalize_option_type,
    _parse_expiry,
    _is_file_path,
)


def _write_csv(path: Path, rows: list[str]) -> None:
    header = "instrument_type,name,strike,instrument_token,lot_size,expiry,tradingsymbol"
    path.write_text("\n".join([header] + rows))


def test_registry_refresh_and_lookup_from_csv(tmp_path):
    csv_path = tmp_path / "contracts.csv"
    _write_csv(
        csv_path,
        [
            "CALL,NIFTY,22500,1001,75,2025-09-02,NIFTY25SEP22500CE",
            "PUT,NIFTY,22500,1002,75,2025-09-02,NIFTY25SEP22500PE",
        ],
    )

    registry = InstrumentRegistry(source=csv_path)
    registry.refresh(force=True)

    meta = registry.lookup("nifty", "2025-09-02", 22500, "CE")
    assert meta is not None
    assert meta["token"] == 1001
    assert meta["tradingsymbol"] == "NIFTY25SEP22500CE"
    assert meta["lot_size"] == 75
    assert registry.lot_size("NIFTY") == 75
    assert registry.lookup("NIFTY", "2025-09-02", 99999, "CE") is None
    assert registry.next_expiry("UNKNOWN") is None
    assert registry.next_weekly_expiry("UNKNOWN") is None
    assert (
        registry.next_weekly_expiry(
            "NIFTY", datetime(2025, 8, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
        ).date().isoformat()
        == "2025-09-02"
    )


def test_registry_weekly_skips_monthly_contract(tmp_path):
    csv_path = tmp_path / "contracts_weekly.csv"
    _write_csv(
        csv_path,
        [
            "CALL,NIFTY,22500,2001,75,2025-09-30,NIFTY25SEP22500CE",
            "CALL,NIFTY,22500,2002,75,2025-10-07,NIFTY25OCT22500CE",
        ],
    )

    registry = InstrumentRegistry(source=csv_path)
    registry.refresh(force=True)

    now = datetime(2025, 9, 15, 12, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    expiry = registry.next_weekly_expiry("NIFTY", now)
    assert expiry is not None
    assert expiry.date().isoformat() == "2025-10-07"
    assert registry.next_expiry("NIFTY", datetime(2026, 1, 1, tzinfo=ZoneInfo("Asia/Kolkata"))).date().isoformat() == "2025-10-07"


def test_registry_refresh_from_env_zip(monkeypatch, tmp_path):
    csv_payload = "\n".join(
        [
            "instrument_type,name,strike,instrument_token,lot_size,expiry,tradingsymbol",
            "CALL,NIFTY,22600,3001,50,2025-10-07,NIFTY25OCT22600CE",
            "PUT,NIFTY,22600,3002,,2025-10-07,NIFTY25OCT22600PE",
        ]
    )
    zip_path = tmp_path / "contracts.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("contracts.csv", csv_payload)

    monkeypatch.setenv("INSTRUMENTS__CSV", str(zip_path))

    registry = InstrumentRegistry()
    registry.refresh(force=True)

    naive_now = datetime(2025, 10, 1, 9, 0)
    expiry = registry.next_expiry("NIFTY", naive_now)
    assert expiry is not None
    assert expiry.tzinfo is not None

    meta = registry.lookup("NIFTY", "2025-10-07", 22600, "CE")
    assert meta is not None
    assert meta["weekly"] is True
    assert registry.lot_size("NIFTY", "2025-10-07") == 50
    assert registry.last_refresh is not None


def test_registry_refresh_skips_when_checksum_matches(monkeypatch, tmp_path):
    csv_path = tmp_path / "contracts_checksum.csv"
    _write_csv(
        csv_path,
        ["CALL,NIFTY,22500,4001,75,2025-11-04,NIFTY25NOV22500CE"],
    )
    registry = InstrumentRegistry(source=csv_path)
    registry.refresh(force=True)

    def boom(_raw: bytes):
        raise AssertionError("_iter_rows should not run when checksum matches")

    monkeypatch.setattr(registry, "_iter_rows", boom)
    registry.refresh()


def test_registry_handles_empty_file(tmp_path):
    csv_path = tmp_path / "contracts_empty.csv"
    csv_path.write_text("instrument_type,name,strike,instrument_token,lot_size,expiry,tradingsymbol\n")
    registry = InstrumentRegistry(source=csv_path)
    registry.refresh(force=True)
    assert registry.lookup("NIFTY", "2025-01-01", 1, "CE") is None


def test_registry_skips_invalid_rows(tmp_path):
    csv_path = tmp_path / "contracts_invalid.csv"
    _write_csv(
        csv_path,
        [
            "INVALID,NIFTY,22500,5001,75,2025-10-07,NIFTY25OCT22500XX",
            ",,,,",
        ],
    )
    registry = InstrumentRegistry(source=csv_path)
    registry.refresh(force=True)
    assert registry.lookup("NIFTY", "2025-10-07", 22500, "CE") is None


def test_registry_load_bytes_from_url(monkeypatch):
    registry = InstrumentRegistry()
    response = SimpleNamespace(content=b"csv-data", raise_for_status=lambda: None)
    monkeypatch.setattr(
        "src.options.contract_registry.requests.get",
        lambda url, timeout=10: response,
    )
    data = registry._load_bytes("https://example.com/contracts.csv")
    assert data == b"csv-data"


def test_iter_rows_handles_zip_without_csv(tmp_path):
    zip_path = tmp_path / "no_csv.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("notes.txt", "not a csv")

    registry = InstrumentRegistry(source=zip_path)
    rows = list(registry._iter_rows(zip_path.read_bytes()))
    assert rows == []


def test_normalize_row_accepts_datetime_expiry(tmp_path):
    registry = InstrumentRegistry()
    row = {
        "name": "NIFTY",
        "instrument_type": "CE",
        "instrument_token": "7001",
        "strike": "22500",
        "lot_size": "75",
        "tradingsymbol": "NIFTY25OCT22500CE",
        "expiry": datetime(2025, 10, 7, 15, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
    }
    contract = registry._normalize_row(row)
    assert contract is not None
    assert contract.weekly is True


def test_normalize_row_requires_symbol():
    registry = InstrumentRegistry()
    assert registry._normalize_row({"instrument_type": "CE"}) is None


def test_coerce_tz_handles_none_and_naive():
    registry = InstrumentRegistry()
    now_default = registry._coerce_tz(None)
    assert now_default.tzinfo is not None
    naive = datetime(2025, 1, 1, 10, 0)
    adjusted = registry._coerce_tz(naive)
    assert adjusted.tzinfo is not None


def test_normalize_option_type_variants():
    assert _normalize_option_type("BUY") == "CE"
    assert _normalize_option_type("SELL") == "PE"
    assert _normalize_option_type("symbolCE") == "CE"
    assert _normalize_option_type("symbolPE") == "PE"


def test_parse_expiry_fallback_isoformat():
    parsed = _parse_expiry("2025-10-07T15:30:00")
    assert parsed.year == 2025 and parsed.month == 10 and parsed.day == 7


def test_is_file_path_handles_oserror():
    class BadPath:
        def __fspath__(self) -> str:
            raise OSError("bad path")

    assert _is_file_path(BadPath()) is False
