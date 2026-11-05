#!/usr/bin/env python3
"""Archive identified Breeze NIFTY index/futures/options for offline research.

Responsibilities: Validate identities and seal raw/normalized historical bars.
Ownership: Isolated offline research importer, not a production broker adapter.
Safety: Never place orders, infer contracts, simulate fills or promote to replay.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import logging
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
    expiry: dt.date | None,
    strike: int | None,
    side: str,
    timestamp_convention: str,
    product_type: str = "options",
) -> tuple[list[list[Any]], list[dict[str, Any]], list[str]]:
    """Verify provider contract and observed minute; never fill missing data."""
    if timestamp_convention not in {"bar_start", "bar_end"}:
        raise ValueError("breeze_timestamp_convention_required")
    if product_type not in {"cash", "futures", "options"}:
        raise ValueError("breeze_product_invalid")
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
            "exchange_code": "NSE" if product_type == "cash" else "NFO",
        }
        if product_type != "cash" or "product_type" in row:
            expected["product_type"] = product_type.upper()
        if product_type == "options":
            expected["right"] = "CALL" if side == "CE" else "PUT"
        if any(
            str(row.get(key, "")).upper() != value for key, value in expected.items()
        ):
            raise ValueError("breeze_contract_identity_mismatch")
        try:
            actual_expiry = (
                dt.datetime.strptime(str(row["expiry_date"]).upper(), "%d-%b-%Y").date()
                if product_type != "cash"
                else None
            )
            actual_strike = (
                float(row["strike_price"]) if product_type == "options" else None
            )
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


def collect_contract(
    client: Any,
    *,
    day: dt.date,
    expiry: dt.date | None,
    strike: int | None,
    side: str,
    product_type: str,
    outdir: Path,
    timestamp_convention: str,
) -> dict[str, Any]:
    """Seal one named contract; cache verification never makes a broker request."""
    if (
        day >= dt.datetime.now(IST).date()
        or product_type not in {"cash", "futures", "options"}
        or product_type != "cash"
        and (expiry is None or expiry < day)
        or product_type == "cash"
        and (expiry is not None or strike is not None or side != "INDEX")
        or product_type == "futures"
        and (strike is not None or side != "FUT")
        or product_type == "options"
        and (
            type(strike) is not int
            or strike <= 0
            or strike % 50
            or side not in {"CE", "PE"}
        )
    ):
        raise ValueError("breeze_research_contract_or_date_invalid")
    if timestamp_convention not in {"bar_start", "bar_end"}:
        raise ValueError("breeze_timestamp_convention_required")
    identity = (
        f"NIFTY_{expiry.isoformat()}_{strike}_{side}"
        if product_type == "options" and expiry is not None
        else (
            f"NIFTY_{expiry.isoformat()}_FUT"
            if product_type == "futures" and expiry is not None
            else "NIFTY_INDEX"
        )
    )
    path = outdir / "breeze_history" / day.isoformat() / f"{identity}.json"
    arguments = {
        "day": day,
        "expiry": expiry,
        "strike": strike,
        "side": side,
        "product_type": product_type,
        "timestamp_convention": timestamp_convention,
    }
    expected = {
        "schema_version": 1,
        "day": day.isoformat(),
        "expiry": expiry.isoformat() if expiry is not None else None,
        "strike": strike,
        "side": side,
        "source_timestamp_convention": timestamp_convention,
    }
    if path.exists():
        cached = json.loads(path.read_text())
        if (
            any(cached.get(key) != value for key, value in expected.items())
            or cached.get("product_type", "options") != product_type
            or cached.get("provider_sha256") != _digest(cached.get("provider_rows"))
            or cached.get("candles_sha256") != _digest(cached.get("candles"))
        ):
            raise ValueError("breeze_archive_conflict")
        candles, _, missing = normalize(
            {"Status": 200, "Error": None, "Success": cached["provider_rows"]},
            **arguments,
        )
        if cached["candles"] != candles or cached["missing_minutes"] != missing:
            raise ValueError("breeze_archive_conflict")
        status = "cached"
        requests_made = 0
    else:
        request = {
            "interval": "1minute",
            "from_date": f"{day.isoformat()}T09:15:00.000Z",
            "to_date": f"{day.isoformat()}T15:29:00.000Z",
            "stock_code": "NIFTY",
            "exchange_code": "NSE" if product_type == "cash" else "NFO",
            "product_type": product_type,
        }
        if expiry is not None:
            request["expiry_date"] = f"{expiry.isoformat()}T07:00:00.000Z"
            request["right"] = (
                ("call" if side == "CE" else "put")
                if product_type == "options"
                else "others"
            )
            request["strike_price"] = str(strike) if strike is not None else "0"
        response = client.get_historical_data_v2(**request)
        try:
            candles, raw_rows, missing = normalize(response, **arguments)
        except ValueError:
            if isinstance(response, dict) and isinstance(response.get("Success"), list):
                _write_once(
                    outdir / "quarantine" / day.isoformat() / f"{identity}.json",
                    {
                        "request": request,
                        "provider_rows": response["Success"],
                        "validation_failed": True,
                        "promotion_eligible": False,
                    },
                )
            raise
        payload = {
            **expected,
            "source": "icici_breeze_historical_v2",
            "product_type": product_type,
            "interval": "1minute",
            "source_timezone": "Asia/Kolkata",
            "source_timezone_verified": False,
            "source_timestamp_convention_verified": False,
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
        requests_made = 1
    return {
        "path": str(path.relative_to(outdir)),
        "bars": len(candles),
        "missing_minutes": len(missing),
        "status": status,
        "requests_made": requests_made,
    }


def collect_pair(
    client: Any,
    *,
    day: dt.date,
    expiry: dt.date,
    strike: int,
    outdir: Path,
    timestamp_convention: str,
) -> dict[str, Any]:
    """Preserve the original option-pair CLI and archive contract."""
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
    for side in ("CE", "PE"):
        result = collect_contract(
            client,
            day=day,
            expiry=expiry,
            strike=strike,
            side=side,
            product_type="options",
            outdir=outdir,
            timestamp_convention=timestamp_convention,
        )
        report["requests_made"] += result.pop("requests_made")
        report["contracts"][side] = result
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
        # Official SDK loggers can write outside redirected stdout/stderr.
        logging.disable(logging.CRITICAL)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
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
    except ValueError as exc:
        known_codes = {
            "breeze_provider_error",
            "breeze_contract_identity_mismatch",
            "breeze_candle_schema_invalid",
            "breeze_candle_values_invalid",
            "breeze_timestamp_invalid",
            "breeze_archive_conflict",
            "breeze_partial_archive_exists",
        }
        code = str(exc)
        code = code if code in known_codes else "breeze_validation_failed"
        parser.exit(1, f"Breeze history collection failed: {code}\n")
    except Exception:
        parser.exit(1, "Breeze history SDK/transport failed (details suppressed)\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
