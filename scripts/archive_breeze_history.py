#!/usr/bin/env python3
"""Archive named Breeze NIFTY options for offline research only.

Responsibilities: Authenticate read-only and archive exact-contract minute bars.
Ownership: Isolated offline research importer, not a production broker adapter.
Safety: Never place orders, infer contracts, simulate fills or promote to replay.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
SESSION_START = dt.time(9, 15)
SESSION_END = dt.time(15, 30)
FIELDS = ("open", "high", "low", "close", "volume", "open_interest")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _numbers(row: dict[str, Any]) -> list[float | int | None]:
    try:
        prices = [float(row[key]) for key in FIELDS[:4]]
        volume = float(row["volume"])
        oi = None if row.get("open_interest") is None else float(row["open_interest"])
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise ValueError("breeze_candle_values_invalid") from exc
    if (
        not all(math.isfinite(value) and value > 0 for value in prices)
        or not math.isfinite(volume)
        or volume < 0
        or not volume.is_integer()
        or oi is not None
        and (not math.isfinite(oi) or oi < 0 or not oi.is_integer())
        or prices[1] < max(prices[0], prices[3])
        or prices[2] > min(prices[0], prices[3])
        or prices[1] < prices[2]
    ):
        raise ValueError("breeze_candle_values_invalid")
    return [*prices, int(volume), None if oi is None else int(oi)]


def normalize(
    response: Any,
    *,
    day: dt.date,
    expiry: dt.date,
    strike: int,
    side: str,
    timestamp_convention: str,
) -> tuple[list[list[Any]], list[dict[str, Any]], list[str]]:
    """Verify provider contract and observed minute; never fill missing data."""
    if timestamp_convention not in {"bar_start", "bar_end"}:
        raise ValueError("breeze_timestamp_convention_required")
    if (
        not isinstance(response, dict)
        or response.get("Status") != 200
        or response.get("Error") is not None
        or not isinstance(response.get("Success"), list)
    ):
        raise ValueError("breeze_provider_error")
    raw_rows = response["Success"]
    candles: list[list[Any]] = []
    observed: set[dt.datetime] = set()
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ValueError("breeze_candle_schema_invalid")
        expected = {
            "stock_code": "NIFTY",
            "exchange_code": "NFO",
            "product_type": "OPTIONS",
            "right": "CALL" if side == "CE" else "PUT",
        }
        if any(
            str(row.get(key, "")).upper() != value
            for key, value in expected.items()
        ):
            raise ValueError("breeze_contract_identity_mismatch")
        try:
            actual_expiry = dt.datetime.strptime(
                str(row["expiry_date"]).upper(), "%d-%b-%Y"
            ).date()
            actual_strike = float(row["strike_price"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("breeze_contract_identity_mismatch") from exc
        if actual_expiry != expiry or actual_strike != strike:
            raise ValueError("breeze_contract_identity_mismatch")
        try:
            stamp = dt.datetime.strptime(str(row["datetime"]), "%Y-%m-%d %H:%M:%S")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("breeze_timestamp_invalid") from exc
        stamp = stamp.replace(tzinfo=IST)
        if timestamp_convention == "bar_end":
            stamp -= dt.timedelta(minutes=1)
        if (
            stamp.date() != day
            or not SESSION_START <= stamp.timetz().replace(tzinfo=None) < SESSION_END
            or stamp.second
            or stamp.microsecond
            or stamp in observed
            or candles
            and stamp.isoformat() <= candles[-1][0]
        ):
            raise ValueError("breeze_timestamp_invalid")
        observed.add(stamp)
        candles.append([stamp.isoformat(), *_numbers(row)])
    session = dt.datetime.combine(day, SESSION_START, tzinfo=IST)
    missing = [
        (session + dt.timedelta(minutes=minute)).isoformat()
        for minute in range(375)
        if session + dt.timedelta(minutes=minute) not in observed
    ]
    return candles, raw_rows, missing


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("breeze_archive_conflict")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    if temporary.exists():
        raise ValueError("breeze_partial_archive_exists")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def collect_pair(
    client: Any,
    *,
    day: dt.date,
    expiry: dt.date,
    strike: int,
    outdir: Path,
    timestamp_convention: str,
) -> dict[str, Any]:
    """At most two read-only V2 requests for one completed session."""
    if (
        day >= dt.datetime.now(IST).date()
        or expiry < day
        or type(strike) is not int
        or strike <= 0
        or strike % 50
    ):
        raise ValueError("breeze_research_contract_or_date_invalid")
    if timestamp_convention not in {"bar_start", "bar_end"}:
        raise ValueError("breeze_timestamp_convention_required")
    root = outdir / "breeze_history" / day.isoformat()
    report: dict[str, Any] = {
        "source": "icici_breeze_historical_v2",
        "day": day.isoformat(),
        "expiry": expiry.isoformat(),
        "strike": strike,
        "research_only": True,
        "historical_atm_selection_verified": False,
        "historical_depth_verified": False,
        "historical_lot_size_verified": False,
        "promotion_eligible": False,
        "requests_made": 0,
        "contracts": {},
    }
    for side, right in (("CE", "call"), ("PE", "put")):
        path = root / f"NIFTY_{expiry.isoformat()}_{strike}_{side}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if (
                cached.get("schema_version") != 1
                or cached.get("side") != side
                or cached.get("day") != day.isoformat()
                or cached.get("expiry") != expiry.isoformat()
                or cached.get("strike") != strike
                or cached.get("source_timestamp_convention") != timestamp_convention
                or cached.get("provider_sha256") != _digest(cached.get("provider_rows"))
                or cached.get("candles_sha256") != _digest(cached.get("candles"))
            ):
                raise ValueError("breeze_archive_conflict")
            candles, _, missing = normalize(
                {"Status": 200, "Error": None, "Success": cached["provider_rows"]},
                day=day,
                expiry=expiry,
                strike=strike,
                side=side,
                timestamp_convention=timestamp_convention,
            )
            if cached["candles"] != candles or cached["missing_minutes"] != missing:
                raise ValueError("breeze_archive_conflict")
            status = "cached"
        else:
            report["requests_made"] += 1
            response = client.get_historical_data_v2(
                interval="1minute",
                from_date=f"{day.isoformat()}T00:00:00.000Z",
                to_date=f"{day.isoformat()}T23:59:59.000Z",
                stock_code="NIFTY",
                exchange_code="NFO",
                product_type="options",
                expiry_date=f"{expiry.isoformat()}T07:00:00.000Z",
                right=right,
                strike_price=str(strike),
            )
            candles, raw_rows, missing = normalize(
                response,
                day=day,
                expiry=expiry,
                strike=strike,
                side=side,
                timestamp_convention=timestamp_convention,
            )
            payload = {
                "schema_version": 1,
                "source": "icici_breeze_historical_v2",
                "day": day.isoformat(),
                "expiry": expiry.isoformat(),
                "strike": strike,
                "side": side,
                "interval": "1minute",
                "source_timezone": "Asia/Kolkata",
                "source_timestamp_convention": timestamp_convention,
                "normalized_timestamp_convention": "bar_start",
                "provider_rows": raw_rows,
                "provider_sha256": _digest(raw_rows),
                "candles": candles,
                "candles_sha256": _digest(candles),
                "missing_minutes": missing,
                "promotion_eligible": False,
            }
            _write_once(path, payload)
            status = "collected" if candles else "no_data"
        report["contracts"][side] = {
            "path": str(path.relative_to(outdir)),
            "bars": len(candles),
            "missing_minutes": len(missing),
            "status": status,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-only", action="store_true")
    parser.add_argument("--day", type=dt.date.fromisoformat)
    parser.add_argument("--expiry", type=dt.date.fromisoformat)
    parser.add_argument("--strike", type=int)
    parser.add_argument("--timestamp-convention", choices=("bar_start", "bar_end"))
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    values = [
        os.environ.get(name, "").strip()
        for name in ("BREEZE_API_KEY", "BREEZE_API_SECRET", "BREEZE_SESSION_TOKEN")
    ]
    if not all(values):
        parser.error("Breeze credentials missing or empty (values not displayed)")
    if not args.auth_only and not all(
        (args.day, args.expiry, args.strike, args.outdir, args.timestamp_convention)
    ):
        parser.error(
            "--day, --expiry, --strike, --timestamp-convention, --outdir required"
        )
    try:
        from breeze_connect import BreezeConnect

        client = BreezeConnect(api_key=values[0])
        client.generate_session(api_secret=values[1], session_token=values[2])
    except Exception:
        parser.exit(1, "Breeze authentication failed (details suppressed)\n")
    if args.auth_only:
        print("Breeze authentication succeeded (read-only; no order sent)")
        return
    try:
        report = collect_pair(
            client,
            day=args.day,
            expiry=args.expiry,
            strike=args.strike,
            outdir=args.outdir,
            timestamp_convention=args.timestamp_convention,
        )
    except Exception:
        parser.exit(1, "Breeze history collection failed (details suppressed)\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
