"""Causal, read-only research adapter for NIFTY EOD contract references.

Responsibilities: Serve available-after-day historical contract and index bars.
Ownership: Offline research input, not production execution.
Safety: Reject same-day lookahead and unsupported quotes or fills.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


class EvidenceUnavailable(ValueError):
    """Requested execution evidence is not present in the EOD archive."""


class EodReferenceReader:
    """Read-only dated contract metadata with no same-session lookahead."""

    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError("historical_eod_research_db_missing")
        self.path = path.resolve()

    def _db(self):
        # Read-only URI also prevents any accidental SQLite write locks.
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)

    @staticmethod
    def _asof_day(asof: datetime):
        if asof.tzinfo is None or asof.utcoffset() is None:
            raise ValueError("asof_timezone_required")
        return asof.astimezone(IST).date()

    def readiness(self) -> dict:
        with self._db() as db:
            row = db.execute(
                "SELECT COUNT(*), SUM(option_contracts), "
                "SUM(futures_contracts) FROM sessions"
            ).fetchone()
            index = db.execute("SELECT COUNT(*) FROM daily_index").fetchone()[0]
        return {
            "observed_sessions": int(row[0]),
            "observed_option_contract_days": int(row[1] or 0),
            "observed_future_contract_days": int(row[2] or 0),
            "valid_index_daily_bars": int(index),
            "research_only": True,
            "data_tier": "EOD_EX_POST_REFERENCE",
            "full_historical_option_universe_verified": False,
            "historical_intraday_quote_depth_present": False,
            "execution_grade_ready": False,
            "promotion_eligible": False,
        }

    def reference_contracts(
        self, *, asof: datetime, day: date, kind: str, expiry: date | None = None
    ):
        """Return EOD row facts only after their next-calendar-day availability.

        The source does NOT prove intraday listing, broker token, or execution.
        """
        if kind not in {"CE", "PE", "FUT"}:
            raise ValueError("unsupported_reference_kind")
        seen_at = self._asof_day(asof)
        if seen_at <= day:
            raise EvidenceUnavailable("same_day_eod_reference_leakage")
        sql = (
            "SELECT c.instrument_id,c.instrument_name,c.kind,c.expiry,c.strike,"
            "c.lot_size,c.volume,c.transactions,c.open_interest,c.close,"
            "c.last,s.available_not_before "
            "FROM contract_eod AS c JOIN sessions s ON s.day=c.day "
            "WHERE c.day=? AND c.kind=? AND s.available_not_before<=?"
        )
        params = [day.isoformat(), kind, seen_at.isoformat()]
        if expiry is not None:
            sql += " AND c.expiry=?"
            params.append(expiry.isoformat())
        sql += " ORDER BY c.expiry,c.strike,c.instrument_name"
        with self._db() as db:
            rows = db.execute(sql, params).fetchall()
        return [
            {
                "instrument_id": r[0],
                "name": r[1],
                "kind": r[2],
                "expiry": r[3],
                "strike": r[4],
                "reported_lot": r[5],
                "day_eod_volume": r[6],
                "day_eod_transactions": r[7],
                "day_eod_open_interest": r[8],
                "day_close": r[9],
                "day_last": r[10],
                "available_not_before": r[11],
                "ex_post_only": True,
                "executable_quote": False,
            }
            for r in rows
        ]

    def daily_index_bars(self, *, asof: datetime, limit: int = 100) -> list[dict]:
        if not 1 <= limit <= 10000:
            raise ValueError("invalid_bar_limit")
        asof_day = self._asof_day(asof)
        with self._db() as db:
            rows = db.execute(
                "SELECT day,open,high,low,close,volume,available_not_before "
                "FROM daily_index WHERE available_not_before<=? "
                "ORDER BY day DESC LIMIT ?",
                (asof_day.isoformat(), limit),
            ).fetchall()
        return [
            {
                "day": r[0],
                "open": r[1],
                "high": r[2],
                "low": r[3],
                "close": r[4],
                "volume": r[5],
                "available_not_before": r[6],
                "execution_eligible": False,
                "frequency": "1d",
            }
            for r in reversed(rows)
        ]

    def executable_quotes(self, *args, **kwargs):
        raise EvidenceUnavailable("historical_bid_ask_and_depth_unavailable")

    def simulated_fills(self, *args, **kwargs):
        raise EvidenceUnavailable("historical_executable_fills_unavailable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--asof", type=datetime.fromisoformat, required=True)
    args = parser.parse_args()
    reader = EodReferenceReader(args.database)
    recent = reader.daily_index_bars(asof=args.asof, limit=1)
    print(
        json.dumps(
            {
                **reader.readiness(),
                "asof_date_ist": reader._asof_day(args.asof).isoformat(),
                "latest_research_index_day": recent[-1]["day"] if recent else None,
                "quotes_available": False,
                "fills_available": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
