#!/usr/bin/env python3
"""Archive active Kite contract history for offline research; never submit orders."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient

IST = ZoneInfo("Asia/Kolkata")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(path)


def archive_history(
    client: ZerodhaKiteClient,
    symbols: list[str],
    start: dt.date,
    end: dt.date,
    outdir: Path,
) -> dict:
    """Save raw minute/OI responses and contemporaneous instrument identity.

    This is an offline exporter, not a runtime history store or contract selector.
    Only complete past calendar dates may be cached. Empty/failed responses are
    reported and retried on subsequent runs rather than claimed as full coverage.
    """
    if start > end or end >= dt.datetime.now(IST).date():
        raise ValueError("Require start <= end and end before today's IST date")
    requested = list(dict.fromkeys(symbol.strip().upper() for symbol in symbols))
    if not requested or any(":" not in symbol for symbol in requested):
        raise ValueError("Specify exchange-qualified NIFTY contract symbols")
    exchanges = {symbol.split(":", 1)[0] for symbol in requested}
    if not exchanges.issubset({"NSE", "NFO"}):
        raise ValueError("Only NSE/NFO NIFTY context/options are supported")
    instruments = {}
    for exchange in sorted(exchanges):
        for row in client.instruments(exchange):
            key = f"{row.get('exchange', exchange)}:{row.get('tradingsymbol', '')}"
            if key in requested:
                instruments[key] = row
    # Validate the entire requested universe before any history request.
    for symbol in requested:
        row = instruments.get(symbol)
        if row is None:
            raise ValueError(f"Not in current instrument master: {symbol}")
        option_or_future = (
            row.get("name") == "NIFTY"
            and row.get("instrument_type") in {"CE", "PE", "FUT"}
            and symbol.startswith("NFO:")
        )
        spot = symbol == "NSE:NIFTY 50"
        if not (option_or_future or spot):
            raise ValueError(f"Not a NIFTY option/context instrument: {symbol}")
        if int(row["instrument_token"]) <= 0:
            raise ValueError(f"Invalid current instrument token: {symbol}")
    report = {
        "source": "kite_historical",
        "interval": "minute",
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "symbols": requested,
        "saved_requests": 0,
        "cached_requests": 0,
        "empty_requests": [],
        "failed_requests": [],
        "contains_historical_bid_ask_depth": False,
        "coverage_is_request_level": True,
    }
    _write_json(
        outdir / "instruments.json",
        {
            "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "instruments": instruments,
        },
    )
    for symbol in requested:
        instrument = instruments[symbol]
        token = int(instrument["instrument_token"])
        cursor = start
        while cursor <= end:
            last = min(cursor + dt.timedelta(days=29), end)
            first_ts = dt.datetime.combine(cursor, dt.time.min, IST)
            last_ts = dt.datetime.combine(last, dt.time(23, 59, 59), IST)
            identity = {
                "symbol": symbol,
                "instrument_token": token,
                "interval": "minute",
                "oi": True,
                "continuous": False,
                "from": first_ts.isoformat(),
                "to": last_ts.isoformat(),
            }
            name = f"{symbol.replace(':', '_').replace(' ', '_')}_{cursor}_{last}.json"
            path = outdir / "candles" / name
            cached = None
            if path.exists():
                try:
                    cached = json.loads(path.read_text())
                except (OSError, ValueError):
                    pass
            if (
                isinstance(cached, dict)
                and all(cached.get(k) == v for k, v in identity.items())
                and isinstance(cached.get("candles"), list)
                and cached["candles"]
            ):
                report["cached_requests"] += 1
            else:
                try:
                    candles = client.historical_data(
                        token, first_ts, last_ts, "minute", continuous=False, oi=True
                    )
                    if not isinstance(candles, list):
                        raise ValueError("Historical response is not a candle list")
                    if not candles:
                        report["empty_requests"].append(identity)
                    else:
                        _write_json(
                            path,
                            {
                                **identity,
                                "instrument": instrument,
                                "captured_at": dt.datetime.now(
                                    dt.timezone.utc
                                ).isoformat(),
                                "timestamp_convention": "bar_start",
                                "candles": candles,
                            },
                        )
                        report["saved_requests"] += 1
                except Exception as exc:
                    # Do not persist exception text that may contain auth/request data.
                    report["failed_requests"].append(
                        {**identity, "error_type": type(exc).__name__}
                    )
            cursor = last + dt.timedelta(days=1)
    _write_json(outdir / "coverage.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", type=dt.date.fromisoformat, required=True)
    parser.add_argument(
        "--end",
        type=dt.date.fromisoformat,
        default=dt.datetime.now(IST).date() - dt.timedelta(days=1),
    )
    parser.add_argument("--outdir", type=Path, default=Path("data/kite_history"))
    args = parser.parse_args(argv)
    load_dotenv(override=False)
    client = ZerodhaKiteClient()
    try:
        report = archive_history(
            client, args.symbols, args.start, args.end, args.outdir
        )
    finally:
        client.close()
    print(json.dumps(report, indent=2))
    return 2 if report["failed_requests"] or report["empty_requests"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
