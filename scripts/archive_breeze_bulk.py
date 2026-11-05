"""Collect bounded NIFTY research history, never production orders.

Responsibilities: source dated NSE identities, acquire index/futures/option
minutes through the canonical Breeze importer, checkpoint coverage and hashes.
Ownership: offline acquisition orchestration; archive_breeze_history owns bars.
Safety: no invented expiry calendars, public price logs, fills or live imports.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from scripts.archive_breeze_history import (
    IST,
    authentication_error_code,
    collect_contract,
)

NSE_REPO = "SantoshSrinivas79/NSE-FNO-Data-bank"
NSE_REVISION = "ccf4aa6cb90f27428dd9f090556efb64286e5ad8"
NSE_TREE = (
    f"https://api.github.com/repos/{NSE_REPO}/git/trees/{NSE_REVISION}?recursive=1"
)


def fetch(url: str, maximum: int) -> bytes:
    headers = {"User-Agent": "Nifty-private-research/1.0"}
    if url.startswith("https://api.github.com/") and os.environ.get("GH_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GH_TOKEN"]
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=45
    ) as response:
        data = response.read(maximum + 1)
    if not data or len(data) > maximum:
        raise ValueError("source_size_invalid")
    return data


def catalog_path(day: dt.date) -> str:
    return f"data/{day:%Y/%m}/BhavCopy_NSE_FO_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"


def parse_catalog(data: bytes, day: dt.date) -> dict[str, Any]:
    """Preserve all published NIFTY F&O EOD rows as ex-post references."""
    if not 50 <= len(data) <= 9_000_000:
        raise ValueError("catalog_size_invalid")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = [
            m
            for m in archive.infolist()
            if not m.is_dir() and m.filename.endswith(".csv")
        ]
        if len(members) != 1 or members[0].file_size > 90_000_000:
            raise ValueError("catalog_zip_invalid")
        with archive.open(members[0]) as stream:
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
            needed = {
                "BizDt",
                "TckrSymb",
                "FinInstrmTp",
                "XpryDt",
                "StrkPric",
                "OptnTp",
            }
            if reader.fieldnames is None or not needed <= set(reader.fieldnames):
                raise ValueError("catalog_schema_invalid")
            rows = []
            futures, options = set(), set()
            for row in reader:
                if row.get("TckrSymb") != "NIFTY" or row.get("FinInstrmTp") not in {
                    "IDF",
                    "IDO",
                }:
                    continue
                if row["BizDt"] != day.isoformat():
                    raise ValueError("catalog_date_mismatch")
                expiry = dt.date.fromisoformat(row["XpryDt"])
                if expiry < day:
                    raise ValueError("catalog_expired_identity")
                if row["FinInstrmTp"] == "IDF":
                    if expiry.isoformat() in futures:
                        raise ValueError("catalog_duplicate_future")
                    futures.add(expiry.isoformat())
                else:
                    strike = float(row["StrkPric"])
                    side = row["OptnTp"]
                    if (
                        not strike.is_integer()
                        or strike <= 0
                        or side not in {"CE", "PE"}
                    ):
                        raise ValueError("catalog_option_invalid")
                    identity = (expiry.isoformat(), int(strike), side)
                    if identity in options:
                        raise ValueError("catalog_duplicate_option")
                    options.add(identity)
                rows.append(row)
    if not futures or not options:
        raise ValueError("catalog_nifty_contracts_missing")
    return {
        "day": day.isoformat(),
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "all_nifty_eod_rows": rows,
        "future_expiries": sorted(futures),
        "option_identities": sorted(options),
        "available_not_before": (day + dt.timedelta(days=1)).isoformat(),
        "ex_post_reference_only": True,
    }


def select_options(
    catalog: dict[str, Any], opening: float
) -> list[tuple[str, int, str]]:
    """Bound downloads around the observed opening; never authorize a trade."""
    identities = {(str(e), int(k), str(s)) for e, k, s in catalog["option_identities"]}
    selected = []
    for expiry in sorted({e for e, _, _ in identities})[:2]:
        paired = sorted(
            {
                k
                for e, k, _ in identities
                if e == expiry
                and (e, k, "CE") in identities
                and (e, k, "PE") in identities
            }
        )
        if paired:
            strike = min(paired, key=lambda k: (abs(k - opening), k))
            selected.extend((expiry, strike, side) for side in ("CE", "PE"))
    return selected


class ResearchClient:
    """One acquisition budget and rate limiter around SDK historical GETs only."""

    def __init__(self, client: Any, maximum: int, deadline: float):
        if not 1 <= maximum <= 4400:
            raise ValueError("request_budget_invalid")
        self.client, self.maximum, self.deadline = client, maximum, deadline
        self.calls, self.last = 0, 0.0
        self.provider_errors = 0

    def get_historical_data_v2(self, **kwargs: Any) -> Any:
        if self.calls >= self.maximum or time.monotonic() >= self.deadline:
            raise RuntimeError("acquisition_budget_exhausted")
        time.sleep(max(0.0, self.last + 1.1 - time.monotonic()))
        self.last = time.monotonic()
        self.calls += 1
        # No SDK response or exception body is sent to stdout/logging.
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            response = self.client.get_historical_data_v2(**kwargs)
        if (
            not isinstance(response, dict)
            or response.get("Status") != 200
            or response.get("Error") is not None
        ):
            self.provider_errors += 1
        else:
            self.provider_errors = 0
        if self.provider_errors >= 5:
            raise RuntimeError("acquisition_provider_unavailable")
        return response


def write_gzip(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = gzip.compress(json.dumps(value, sort_keys=True).encode(), mtime=0)
    if path.exists() and path.read_bytes() != encoded:
        raise ValueError("catalog_archive_conflict")
    path.write_bytes(encoded)


def acquire(
    client: ResearchClient, first: dt.date, stop: dt.date, outdir: Path, paths: set[str]
) -> dict[str, Any]:
    if not first < stop <= dt.datetime.now(IST).date() or (stop - first).days > 735:
        raise ValueError("completed_study_window_invalid")
    outdir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "first": str(first),
        "stop_exclusive": str(stop),
        "source": "icici_breeze_historical_v2",
        "catalog_mirror_commit": NSE_REVISION,
        "interval": "1minute",
        "completed_days": [],
        "source_unavailable": [],
        "contracts": [],
        "failures": [],
        "requests_made": 0,
        "all_historical_strikes_collected": False,
        "quotes_or_depth_reconstructed": False,
        "option_policy": (
            "opening_nearest_paired_strike_near_two_expiries_"
            "ex_post_acquisition_only"
        ),
        "promotion_eligible": False,
        "plan_walk_completed": False,
        "collection_complete": False,
    }
    try:
        for offset in range((stop - first).days):
            day = first + dt.timedelta(days=offset)
            source = catalog_path(day)
            if day.weekday() >= 5 and source not in paths:
                continue
            if shutil.disk_usage(outdir).free < 3 * 1024**3:
                raise RuntimeError("acquisition_storage_floor")
            catalog = None
            try:
                if source in paths:
                    url = f"https://raw.githubusercontent.com/{NSE_REPO}/{NSE_REVISION}/{source}"
                else:
                    url = (
                        "https://nsearchives.nseindia.com/content/fo/"
                        + source.rsplit("/", 1)[1]
                    )
                catalog = parse_catalog(fetch(url, 9_000_000), day)
                catalog["source_url"] = url
                write_gzip(outdir / "catalog" / f"{day}.json.gz", catalog)
            except (OSError, ValueError, zipfile.BadZipFile):
                report["source_unavailable"].append(str(day))

            def one(
                kind: str, expiry: str | None, strike: int | None, side: str
            ) -> dict[str, Any] | None:
                try:
                    result = collect_contract(
                        client,
                        day=day,
                        expiry=dt.date.fromisoformat(expiry) if expiry else None,
                        strike=strike,
                        side=side,
                        product_type=kind,
                        outdir=outdir,
                        timestamp_convention="bar_start",
                    )
                    report["contracts"].append(
                        {
                            "day": str(day),
                            "kind": kind,
                            "expiry": expiry,
                            "strike": strike,
                            "side": side,
                            **result,
                        }
                    )
                    return result
                except (ValueError, OSError, TypeError, KeyError, OverflowError):
                    report["failures"].append(
                        {
                            "day": str(day),
                            "kind": kind,
                            "expiry": expiry,
                            "strike": strike,
                            "side": side,
                            "reason": "provider_or_validation_failed",
                        }
                    )
                    return None

            spot = one("cash", None, None, "INDEX")
            if catalog is not None:
                for expiry in catalog["future_expiries"]:
                    one("futures", expiry, None, "FUT")
                if spot and spot["bars"]:
                    data = json.loads((outdir / spot["path"]).read_text())
                    opening = float(data["candles"][0][4])
                    for expiry, strike, side in select_options(catalog, opening):
                        one("options", expiry, strike, side)
            report["completed_days"].append(str(day))
            report["requests_made"] = client.calls
            (outdir / "coverage.json").write_text(json.dumps(report, sort_keys=True))
            if len(report["completed_days"]) % 20 == 0:
                print(
                    json.dumps(
                        {
                            "processed_days": len(report["completed_days"]),
                            "calls": client.calls,
                        }
                    ),
                    flush=True,
                )
        report["plan_walk_completed"] = True
    except RuntimeError:
        report["stopped_early"] = True
    finally:
        report["requests_made"] = client.calls
        report["total_bars"] = sum(c["bars"] for c in report["contracts"])
        report["nonempty_contract_days"] = sum(
            c["bars"] > 0 for c in report["contracts"]
        )
        report["empty_contract_days"] = sum(c["bars"] == 0 for c in report["contracts"])
        report["collection_complete"] = bool(
            report["plan_walk_completed"]
            and not report["source_unavailable"]
            and not report["failures"]
            and not report["empty_contract_days"]
        )
        (outdir / "coverage.json").write_text(json.dumps(report, sort_keys=True))
        entries = []
        for path in sorted(outdir.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                entries.append(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                    f"{path.relative_to(outdir)}\n"
                )
        (outdir / "SHA256SUMS").write_text("".join(entries))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    report = None
    authenticated = False
    try:
        # Suppress SDK file loggers as well as console output, before import.
        logging.disable(logging.CRITICAL)
        values = [
            os.environ.get(k, "").strip()
            for k in ("BREEZE_API_KEY", "BREEZE_API_SECRET", "BREEZE_SESSION_TOKEN")
        ]
        if not all(values):
            raise ValueError("breeze_credentials_missing")
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            from breeze_connect import BreezeConnect

            sdk = BreezeConnect(api_key=values[0])
            sdk.generate_session(api_secret=values[1], session_token=values[2])
        authenticated = True
        print("Breeze archive authentication succeeded", flush=True)
        tree = json.loads(fetch(NSE_TREE, 5_000_000))
        if tree.get("truncated") or not isinstance(tree.get("tree"), list):
            raise ValueError("catalog_index_incomplete")
        paths = {e["path"] for e in tree["tree"] if e.get("type") == "blob"}
        report = acquire(
            ResearchClient(
                sdk, int(request["max_requests"]), time.monotonic() + 90 * 60
            ),
            dt.date.fromisoformat(request["first"]),
            dt.date.fromisoformat(request["stop_exclusive"]),
            outdir,
            paths,
        )
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "requests_made",
                        "total_bars",
                        "nonempty_contract_days",
                        "collection_complete",
                    )
                }
            ),
            flush=True,
        )
    except Exception as exc:
        reason = (
            "source_or_transport_failed"
            if authenticated
            else authentication_error_code(exc)
        )
        (outdir / "acquisition-error.json").write_text(
            json.dumps(
                {
                    "reason": reason,
                    "details_suppressed": True,
                }
            )
        )
        raise SystemExit(
            f"Breeze acquisition stopped: {reason}; exception details suppressed"
        ) from None
    if report is None or not report["collection_complete"] or report["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
