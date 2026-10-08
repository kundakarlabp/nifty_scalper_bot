#!/usr/bin/env python3
"""Archive active Kite contract history for offline research; never submit orders."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import time
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from nifty_scalper_bot.config.env_utils import parse_int_env
from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
from nifty_scalper_bot.instruments.active_contracts import cap_option_universe

IST = ZoneInfo("Asia/Kolkata")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(path)


def _candle_digest(candles: list) -> str:
    return hashlib.sha256(
        json.dumps(candles, default=str, sort_keys=True).encode()
    ).hexdigest()


def _validate_candles(candles: list, first: dt.datetime, last: dt.datetime) -> None:
    previous: dt.datetime | None = None
    for row in candles:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            raise ValueError("historical_candle_shape_invalid")
        timestamp = dt.datetime.fromisoformat(str(row[0]))
        if (
            timestamp.tzinfo is None
            or not first <= timestamp <= last
            or (previous is not None and timestamp <= previous)
        ):
            raise ValueError("historical_candle_timestamp_invalid")
        values = [float(value) for value in row[1:6]]
        opening, high, low, close, volume = values
        if (
            not all(math.isfinite(value) for value in values)
            or min(values[:4]) <= 0
            or volume < 0
            or high < max(opening, low, close)
            or low > min(opening, high, close)
        ):
            raise ValueError("historical_candle_values_invalid")
        previous = timestamp


def history_universe(rows: list[dict], selected: dict, future: str) -> list[str]:
    """Mirror the live capped current-expiry basket; never infer past baskets."""
    nominated = ["NSE:NIFTY 50", future, selected["ce"], selected["pe"]]
    ce = next(
        (row for row in rows if f"NFO:{row.get('tradingsymbol')}" == selected["ce"]),
        None,
    )
    if (
        ce is None
        or not ce.get("strike")
        or not ce.get("expiry")
        or not ce.get("instrument_token")
    ):
        return nominated
    atm = float(ce["strike"])
    expiry = str(ce["expiry"])
    option_items: list[tuple[str, int, float, str]] = []
    for row in rows:
        side = str(row.get("instrument_type") or "").upper()
        if (
            row.get("name") != "NIFTY"
            or side not in {"CE", "PE"}
            or str(row.get("expiry") or "") != expiry
        ):
            continue
        try:
            token = int(row.get("instrument_token") or 0)
            strike = float(row.get("strike") or 0)
        except (TypeError, ValueError):
            continue
        if token <= 0 or strike <= 0:
            continue
        option_items.append(
            (f"NFO:{row['tradingsymbol']}", token, strike, side)
        )
    max_options = parse_int_env(
        os.getenv("MAX_ACTIVE_OPTION_SYMBOLS")
        or os.getenv("MAX_LIVE_OPTION_SYMBOLS"),
        8,
    )
    capped = cap_option_universe(
        option_items,
        selected_ce=selected["ce"],
        selected_pe=selected["pe"],
        atm_strike=atm,
        max_options=max_options,
    )
    return list(dict.fromkeys(nominated[:2] + [item[0] for item in capped] + nominated[2:]))


def archive_history(
    client: ZerodhaKiteClient,
    symbols: list[str],
    start: dt.date,
    end: dt.date,
    outdir: Path,
    *,
    cache_dir: Path | None = None,
    allow_completed_today: bool = False,
) -> dict:
    """Save raw minute/OI responses and contemporaneous instrument identity.

    This is an offline exporter, not a runtime history store or contract selector.
    Only complete past calendar dates may be cached. Empty/failed responses are
    reported and retried on subsequent runs rather than claimed as full coverage.
    """
    now = dt.datetime.now(IST)
    latest = (
        now.date()
        if allow_completed_today and now.time() >= dt.time(15, 35)
        else now.date() - dt.timedelta(days=1)
    )
    if start > end or end > latest:
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
            cache_path = (
                (cache_dir / "candles" / name) if cache_dir is not None else path
            )
            cached = None
            if cache_path.exists():
                try:
                    cached = json.loads(cache_path.read_text())
                except (OSError, ValueError):
                    pass
            if (
                isinstance(cached, dict)
                and all(cached.get(k) == v for k, v in identity.items())
                and isinstance(cached.get("candles"), list)
                and cached["candles"]
                and cached.get("instrument")
                == json.loads(json.dumps(instrument, default=str))
                and cached.get("candles_sha256") == _candle_digest(cached["candles"])
            ):
                report["cached_requests"] += 1
                if cache_path != path:
                    _write_json(path, cached)
            else:
                try:
                    for attempt in range(3):
                        try:
                            candles = client.historical_data(
                                token,
                                first_ts,
                                last_ts,
                                "minute",
                                continuous=False,
                                oi=True,
                            )
                            break
                        except Exception:
                            if attempt == 2:
                                raise
                            time.sleep(0.35 * (attempt + 1))
                    if not isinstance(candles, list):
                        raise ValueError("Historical response is not a candle list")
                    if not candles:
                        report["empty_requests"].append(identity)
                    else:
                        _validate_candles(candles, first_ts, last_ts)
                        payload = {
                            **identity,
                            "instrument": instrument,
                            "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "timestamp_convention": "bar_start",
                            "candles": candles,
                            "candles_sha256": _candle_digest(candles),
                        }
                        _write_json(path, payload)
                        if cache_path != path:
                            _write_json(cache_path, payload)
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
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Existing operator env file; does not override environment values",
    )
    args = parser.parse_args(argv)
    if args.env_file is not None and not args.env_file.is_file():
        raise FileNotFoundError("Environment file does not exist")
    load_dotenv(dotenv_path=args.env_file, override=False)
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
