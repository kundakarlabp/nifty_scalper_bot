"""Tests for instrument universe CSV ingestion and SQLite cache."""

from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.data import (
    ensure_sqlite,
    load_rows_for_resolver,
    refresh_from_csv,
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
