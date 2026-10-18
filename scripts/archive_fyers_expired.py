#!/usr/bin/env python3
"""Read-only FYERS expired NIFTY option importer for secondary bar research.

Source contract identity is obtained from FYERS expiry and contract discovery.
No trading APIs, historical token fabrication, inferred lot sizes, synthetic
spot/futures, or live strategy configuration are permitted here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
UTC = dt.timezone.utc
UNDERLYING = "NSE:NIFTY50-INDEX"
CORE_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
OPTION_NAME = re.compile(r"^NSE:NIFTY(?:\d{2}[A-Z]{3}|\d{2}[A-Z]\d{2})\d{4,6}(?:CE|PE)$")


def _response_ok(response: Any, label: str) -> dict[str, Any]:
    if not isinstance(response, dict) or response.get("s") != "ok":
        raise ValueError(f"fyers_{label}_unavailable")
    return response


def discover_pair(client: Any, expiry: dt.date, strike: int) -> dict[str, str]:
    """Use the provider's exact symbols; never synthesize expiry/strike codes."""
    if type(strike) is not int or strike <= 0 or strike % 50:
        raise ValueError("fyers_strike_invalid")
    expiry_response = _response_ok(
        client.expiry_dates(
            {
                "symbol": UNDERLYING,
                "range_from": expiry.isoformat(),
                "range_to": expiry.isoformat(),
                "date_format": "1",
            }
        ),
        "expiry_catalog",
    )
    expiry_dates = expiry_response.get("data", {}).get("expiry_dates", {})
    if expiry.isoformat() not in expiry_dates.get("options", []):
        raise ValueError("fyers_option_expiry_not_listed")
    listing = _response_ok(
        client.history_underlying_symbols(
            {"symbol": UNDERLYING, "expiry_date": expiry.isoformat()}
        ),
        "contract_catalog",
    )
    data = listing.get("data", {})
    if data.get("expiry_date") != expiry.isoformat():
        raise ValueError("fyers_contract_catalog_expiry_conflict")
    contracts = data.get("contracts", {}).get("options")
    if not isinstance(contracts, list):
        raise ValueError("fyers_contract_catalog_invalid")
    result: dict[str, str] = {}
    for side in ("CE", "PE"):
        matched = [
            symbol
            for symbol in contracts
            if isinstance(symbol, str)
            and OPTION_NAME.fullmatch(symbol)
            and symbol.endswith(f"{strike}{side}")
        ]
        if len(matched) != 1:
            raise ValueError("fyers_expired_pair_missing_or_ambiguous")
        result[side] = matched[0]
    if result["CE"] == result["PE"]:
        raise ValueError("fyers_expired_pair_identity_conflict")
    return result


def normalize_option_candles(
    response: Any,
    *,
    symbol: str,
    expiry: dt.date,
    start: dt.date,
    end: dt.date,
    timestamp_convention: str,
) -> list[list[Any]]:
    """Strict FYERS named-column to timezone-aware research-bar conversion."""
    if timestamp_convention not in {"bar_start", "bar_end"}:
        raise ValueError("fyers_timestamp_convention_unverified")
    if not isinstance(response, dict) or response.get("s") not in {"ok", "no_data"}:
        raise ValueError("fyers_historical_data_unavailable")
    if response.get("symbol") != symbol or response.get("resolution") != "1":
        raise ValueError("fyers_candle_contract_or_resolution_mismatch")
    columns = response.get("columns")
    if (
        not isinstance(columns, list)
        or tuple(columns[:6]) != CORE_COLUMNS
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("fyers_candle_columns_invalid")
    candles = response.get("candles")
    if not isinstance(candles, list):
        raise ValueError("fyers_candle_rows_invalid")
    if response["s"] == "no_data":
        if candles:
            raise ValueError("fyers_no_data_has_candles")
        return []
    previous: dt.datetime | None = None
    oi_index = columns.index("open_interest") if "open_interest" in columns else None
    normalized: list[list[Any]] = []
    for row in candles:
        if not isinstance(row, (list, tuple)) or len(row) != len(columns):
            raise ValueError("fyers_candle_shape_invalid")
        epoch = row[0]
        if type(epoch) is not int or not 1_000_000_000 < epoch < 3_000_000_000:
            raise ValueError("fyers_candle_epoch_invalid")
        timestamp = dt.datetime.fromtimestamp(epoch, tz=UTC).astimezone(IST)
        if timestamp_convention == "bar_end":
            timestamp -= dt.timedelta(minutes=1)
        if (
            timestamp.second
            or timestamp.microsecond
            or not start <= timestamp.date() <= end
            or timestamp.date() > expiry
            or not dt.time(9, 15) <= timestamp.time() < dt.time(15, 30)
            or previous is not None
            and timestamp <= previous
        ):
            raise ValueError("fyers_candle_timestamp_invalid")
        try:
            opening, high, low, closing, volume = (float(value) for value in row[1:6])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("fyers_candle_values_invalid") from exc
        if (
            not all(math.isfinite(x) for x in (opening, high, low, closing, volume))
            or min(opening, high, low, closing) <= 0
            or volume < 0
            or high < max(opening, low, closing)
            or low > min(opening, high, closing)
        ):
            raise ValueError("fyers_candle_values_invalid")
        oi: int | None = None
        if oi_index is not None and row[oi_index] is not None:
            value = row[oi_index]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or value != int(value)
            ):
                raise ValueError("fyers_open_interest_invalid")
            oi = int(value)
        normalized.append(
            [timestamp.isoformat(), opening, high, low, closing, volume, oi]
        )
        previous = timestamp
    return normalized


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    """Refuse conflicting input instead of overwriting historical evidence."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("fyers_historical_archive_conflict")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    if temporary.exists():
        raise ValueError("fyers_historical_partial_write")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def collect_expired_pair(
    client: Any,
    *,
    expiry: dt.date,
    strike: int,
    start: dt.date,
    end: dt.date,
    outdir: Path,
    timestamp_convention: str,
    max_intraday_requests: int = 10,
) -> dict[str, Any]:
    """Download only two explicitly discovered expired contracts, without orders.

    Selection of the strike is user-specified, NOT a proven historical ATM
    basket. Save provider evidence separate from the canonical replay tree.
    """
    if not start <= end <= expiry or end >= dt.datetime.now(IST).date():
        raise ValueError("fyers_historical_date_range_invalid")
    if type(max_intraday_requests) is not int or not 2 <= max_intraday_requests <= 100:
        raise ValueError("fyers_intraday_request_budget_invalid")
    if timestamp_convention not in {"bar_start", "bar_end"}:
        raise ValueError("fyers_timestamp_convention_unverified")
    pair = discover_pair(client, expiry, strike)
    report: dict[str, Any] = {
        "source": "fyers_expired_fo",
        "source_underlying": UNDERLYING,
        "expiry": expiry.isoformat(),
        "strike": strike,
        "requested_period": [start.isoformat(), end.isoformat()],
        "provider_contract_symbols": pair,
        "timestamp_convention": timestamp_convention,
        "historical_contracts_provider_verified": True,
        "historical_atm_selection_verified": False,
        "spot_futures_coverage_verified": False,
        "historical_bid_ask_depth_available": False,
        "historical_lot_size_verified": False,
        "promotion_eligible": False,
        "intraday_requests_made": 0,
        "max_intraday_requests": max_intraday_requests,
        "contracts": {},
    }
    for side, symbol in pair.items():
        filename = symbol.replace(":", "_") + ".json"
        path = outdir / "fyers_expired" / expiry.isoformat() / filename
        if path.exists():
            cached = json.loads(path.read_text())
            if (
                cached.get("source") != "fyers_expired_fo"
                or cached.get("symbol") != symbol
                or cached.get("expiry") != expiry.isoformat()
                or cached.get("strike") != strike
                or cached.get("requested_period") != report["requested_period"]
                or cached.get("timestamp_convention") != timestamp_convention
                or cached.get("sha256") != _stable_digest(cached.get("candles"))
            ):
                raise ValueError("fyers_historical_archive_conflict")
            report["contracts"][side] = {
                "path": str(path.relative_to(outdir)),
                "bars": len(cached["candles"]),
                "status": "cached",
            }
            continue
        if report["intraday_requests_made"] >= max_intraday_requests:
            raise ValueError("fyers_intraday_request_budget_exhausted")
        report["intraday_requests_made"] += 1
        response = client.fno_historical_data(
            {
                "symbol": symbol,
                "resolution": "1",
                "date_format": "1",
                "range_from": start.isoformat(),
                "range_to": end.isoformat(),
                "include_oi": "1",
                "include_greeks": "0",
            }
        )
        candles = normalize_option_candles(
            response,
            symbol=symbol,
            expiry=expiry,
            start=start,
            end=end,
            timestamp_convention=timestamp_convention,
        )
        if not candles:
            report["contracts"][side] = {"bars": 0, "status": "no_data"}
            continue
        payload = {
            "source": "fyers_expired_fo",
            "symbol": symbol,
            "expiry": expiry.isoformat(),
            "strike": strike,
            "side": side,
            "requested_period": report["requested_period"],
            "timestamp_convention": timestamp_convention,
            "source_columns": response["columns"],
            "candles": candles,
            "sha256": _stable_digest(candles),
            "historical_bid_ask_depth_available": False,
            "promotion_eligible": False,
        }
        _write_once(path, payload)
        report["contracts"][side] = {
            "path": str(path.relative_to(outdir)),
            "bars": len(candles),
            "status": "archived",
        }
    # Reporting is not part of the source cache: safe to refresh after retries.
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = outdir / "fyers_expired" / expiry.isoformat() / "collection.json"
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def stage_pair_with_verified_context(
    provider_root: Path,
    context_directory: Path,
    destination: Path,
    *,
    day: dt.date,
    expiry: dt.date,
    strike: int,
    lot_size: int,
    lot_size_source: str,
    min_synchronized_minutes: int = 60,
) -> dict[str, Any]:
    """Bridge imported options into the existing *secondary* bar research format.

    Requires independent real spot and futures candles, a supplied historical
    lot size with provenance, and synchronized completed-minute data. No
    fabricated instruments/tokens or inferred historical ATM basket allowed.
    """
    from nifty_scalper_bot.backtesting.strategy_research import load_archive

    if (
        type(lot_size) is not int
        or lot_size <= 0
        or not lot_size_source.strip()
        or not 1 <= min_synchronized_minutes <= 375
        or not day <= expiry
        or destination.exists()
    ):
        raise ValueError("fyers_research_stage_prerequisites_invalid")
    manifest_file = provider_root / "fyers_expired" / expiry.isoformat() / "collection.json"
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("expiry") != expiry.isoformat() or manifest.get("strike") != strike:
        raise ValueError("fyers_research_stage_identity_conflict")
    normalized: dict[str, dict[str, Any]] = {}
    for side in ("CE", "PE"):
        item = manifest["contracts"][side]
        if item.get("status") not in {"archived", "cached"}:
            raise ValueError("fyers_research_stage_option_data_missing")
        path = provider_root / item["path"]
        if path.resolve().is_relative_to(provider_root.resolve()) is False:
            raise ValueError("fyers_research_stage_path_invalid")
        payload = json.loads(path.read_text())
        if (
            payload.get("source") != "fyers_expired_fo"
            or payload.get("symbol") != manifest["provider_contract_symbols"][side]
            or payload.get("expiry") != expiry.isoformat()
            or payload.get("strike") != strike
            or payload.get("side") != side
            or payload.get("timestamp_convention") != "bar_start"
            or payload.get("sha256") != _stable_digest(payload.get("candles"))
        ):
            raise ValueError("fyers_research_stage_source_untrusted")
        symbol = "NFO:" + payload["symbol"].split(":", 1)[1]
        normalized[symbol] = {
            "symbol": symbol,
            "timestamp_convention": "bar_start",
            "source": "fyers_expired_fo",
            "provider_symbol": payload["symbol"],
            "provider_sha256": payload["sha256"],
            "instrument": {
                "exchange": "NFO",
                "tradingsymbol": symbol.split(":", 1)[1],
                "instrument_type": side,
                "name": "NIFTY",
                "expiry": expiry.isoformat(),
                "strike": strike,
                "lot_size": lot_size,
                "historical_lot_size_source": lot_size_source,
            },
            "candles": [
                bar for bar in payload["candles"]
                if dt.datetime.fromisoformat(bar[0]).date() == day
            ],
        }
    context: dict[str, dict[str, Any]] = {}
    for source in sorted((context_directory / "candles").glob("*.json")):
        raw = json.loads(source.read_text())
        symbol = raw.get("symbol")
        if symbol != "NSE:NIFTY 50" and raw.get("instrument", {}).get(
            "instrument_type"
        ) != "FUT":
            continue
        if (
            raw.get("timestamp_convention") != "bar_start"
            or raw.get("instrument", {}).get("instrument_type") not in {"EQ", "FUT"}
        ):
            raise ValueError("fyers_research_context_untrusted")
        if symbol in context:
            raise ValueError("fyers_research_context_identity_ambiguous")
        if raw["instrument"]["instrument_type"] == "FUT":
            if not symbol.startswith("NFO:NIFTY") or dt.date.fromisoformat(
                str(raw["instrument"]["expiry"])
            ) < day:
                raise ValueError("fyers_research_context_future_invalid")
        context[symbol] = {
            **raw,
            "candles": [
                bar for bar in raw["candles"]
                if dt.datetime.fromisoformat(str(bar[0])).date() == day
            ],
        }
    futures = [
        symbol for symbol in context
        if context[symbol]["instrument"]["instrument_type"] == "FUT"
    ]
    if len(futures) != 1 or "NSE:NIFTY 50" not in context:
        raise ValueError("fyers_research_context_missing_or_ambiguous")
    all_candles = {**context, **normalized}
    if any(not payload["candles"] for payload in all_candles.values()):
        raise ValueError("fyers_research_stage_empty_participant")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "session"
        for symbol, payload in all_candles.items():
            name = symbol.replace(":", "_").replace(" ", "_") + ".json"
            _write_once(staging / "candles" / name, payload)
        histories, _, digest = load_archive(staging)
        synchronized = set.intersection(
            *(set(history) for history in histories.values())
        )
        anchors = sorted(
            stamp for stamp in synchronized
            if stamp.date() == day
            and dt.time(9, 30) <= stamp.time() <= dt.time(9, 45)
        )
        if not any(
            all(
                anchor + dt.timedelta(minutes=offset) in synchronized
                for offset in range(min_synchronized_minutes)
            )
            for anchor in anchors
        ):
            raise ValueError("fyers_research_stage_synchronized_minutes_missing")
        result = {
            "source": "fyers_expired_fo_plus_independent_spot_future",
            "evidence_label": "RESEARCH_CANDIDATE",
            "session": day.isoformat(),
            "expiry": expiry.isoformat(),
            "strike": strike,
            "lot_size": lot_size,
            "lot_size_source": lot_size_source,
            "data_sha256": digest,
            "minimum_synchronized_minutes": min_synchronized_minutes,
            "historical_atm_selection_verified": False,
            "historical_order_book_depth_available": False,
            "live_equivalent": False,
            "promotion_eligible": False,
        }
        _write_once(staging / "provenance.json", result)
        staging.rename(destination)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expiry", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--strike", type=int, required=True)
    parser.add_argument("--start", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--end", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("data/fyers_history"))
    parser.add_argument(
        "--timestamp-convention",
        choices=("bar_start", "bar_end"),
        required=True,
        help="Confirm FYERS minute label convention independently; never guess",
    )
    parser.add_argument("--max-intraday-requests", type=int, default=10)
    args = parser.parse_args(argv)
    client_id = os.getenv("FYERS_CLIENT_ID", "")
    access_token = os.getenv("FYERS_ACCESS_TOKEN", "")
    if not client_id or not access_token:
        raise ValueError("fyers_read_only_credentials_unavailable")
    try:
        from fyers_apiv3 import fyersModel
    except ImportError as exc:
        raise RuntimeError("fyers_apiv3_sdk_not_installed") from exc
    client = fyersModel.FyersModel(
        client_id=client_id, is_async=False, token=access_token, log_path=""
    )
    report = collect_expired_pair(
        client,
        expiry=args.expiry,
        strike=args.strike,
        start=args.start,
        end=args.end,
        outdir=args.outdir,
        timestamp_convention=args.timestamp_convention,
        max_intraday_requests=args.max_intraday_requests,
    )
    print(
        json.dumps(
            {
                "source": report["source"],
                "expiry": report["expiry"],
                "contracts": report["contracts"],
                "promotion_eligible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
