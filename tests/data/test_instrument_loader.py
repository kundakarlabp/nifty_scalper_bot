"""Tests for instrument universe CSV ingestion and SQLite cache."""

from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.data import (
    ensure_sqlite,
    load_rows_for_resolver,
    refresh_from_csv,
    sync_instrument_csv_from_broker,
    write_instrument_rows_to_csv,
)


def test_refresh_and_load_from_csv(tmp_path: Path) -> None:
    """CSV refresh should persist NFO NIFTY options and expose resolver rows."""

    csv_content = (
        "instrument_token,exchange,tradingsymbol,lot_size,expiry,strike,"
        "instrument_type\n"
        "1001,NFO,NIFTY25OCT26000CE,50,2025-10-30,26000.0,CE\n"
        "1002,NFO,NIFTY25OCT26000PE,50,2025-10-30,26000.0,PE\n"
        "1003,NFO,NIFTY25OCT26000FUT,50,2025-10-30,0.0,FUT\n"
        "2001,NSE,NIFTY,1,,0.0,EQ\n"
    )
    csv_path = tmp_path / "instruments.csv"
    csv_path.write_text(csv_content)
    db_path = tmp_path / "cache.sqlite"
    conn = ensure_sqlite(str(db_path))
    try:
        summary = refresh_from_csv(conn, str(csv_path))
        assert summary["stored"] == 2
        rows = load_rows_for_resolver(conn)
        assert len(rows) == 2
        first = rows[0]
        assert "tradingsymbol" in first
        assert "exchange" in first
        assert "instrument_token" in first
        assert "lot_size" in first
    finally:
        conn.close()


def test_write_instrument_rows_to_csv_creates_file(tmp_path: Path) -> None:
    """Broker rows should be persisted into a CSV consumable by refresh logic."""

    csv_path = tmp_path / "instruments.csv"
    written = write_instrument_rows_to_csv(
        [
            {
                "instrument_token": 1001,
                "exchange": "NFO",
                "tradingsymbol": "NIFTY25OCT26000CE",
                "lot_size": 50,
                "expiry": "2025-10-30",
                "strike": 26000,
                "instrument_type": "CE",
                "segment": "NFO-OPT",
                "name": "NIFTY",
            }
        ],
        str(csv_path),
    )
    assert written == 1
    assert csv_path.exists()
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "instrument_token" in header


def test_sync_instrument_csv_from_broker_uses_loader(tmp_path: Path) -> None:
    """Broker loader should bootstrap a missing warm-up CSV file."""

    class _Broker:
        def load_instruments(self, exchange: str) -> list[dict[str, object]]:
            assert exchange == "NFO"
            return [
                {
                    "instrument_token": 2001,
                    "exchange": "NFO",
                    "tradingsymbol": "NIFTY25OCT26100PE",
                    "lot_size": 50,
                    "expiry": "2025-10-30",
                    "strike": 26100,
                    "instrument_type": "PE",
                    "segment": "NFO-OPT",
                    "name": "NIFTY",
                }
            ]

    csv_path = tmp_path / "bootstrap.csv"
    summary = sync_instrument_csv_from_broker(_Broker(), str(csv_path), exchange="NFO")
    assert summary["written"] == 1
    assert csv_path.exists()
