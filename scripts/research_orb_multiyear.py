#!/usr/bin/env python3
"""Checksum-verified public monthly-option research; never changes live settings.

Context and early option labels are interpreted as bar ends; later options as
bar starts. Conventions and continuous-future roll identity are unverified, so
results are conditional research evidence and cannot authorize live promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import sys
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nifty_scalper_bot.backtesting.strategy_research import (  # noqa: E402
    run_orb_session_research,
    summarize,
)

IST = ZoneInfo("Asia/Kolkata")
SOURCES = (
    ("Nifty spot and futures data.zip", "240aecc77a7275ec2f05a092b976bf91"),
    ("Nifty Options Data.zip", "717d15361af52d1d654b4909d998967e"),
)
SLIPPAGE = (10.0, 25.0, 50.0)


def candidates() -> list[dict[str, Any]]:
    """Register a small geometry grid and two entry hypotheses before outcomes."""
    result = [{"name": "raw_reference", "overrides": {}, "minimum_net_rr": None}]
    for stop in (0.75, 1.0, 1.25):
        for target in (1.8, 2.2, 2.5):
            result.append(
                {
                    "name": f"stop_{stop}_rr_{target}",
                    "overrides": {
                        "ORB_PREMIUM_STOP_ATR_MULT": str(stop),
                        "ORB_TARGET_RR": str(target),
                    },
                    "minimum_net_rr": 1.5,
                }
            )
    result.extend(
        [
            {
                "name": "retest_only",
                "overrides": {"ORB_MOMENTUM_BRANCH_ENABLED": "false"},
                "minimum_net_rr": 1.5,
            },
            {
                "name": "early_entry_60",
                "overrides": {"ORB_MAX_ENTRY_MINUTES_AFTER_RANGE": "60"},
                "minimum_net_rr": 1.5,
            },
        ]
    )
    return result


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False))
    temporary.replace(path)


def verify_source(path: Path, checksum: str) -> None:
    digest = hashlib.md5()  # Publisher integrity checksum, not a security primitive.
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != checksum:
        raise ValueError("public_archive_checksum_mismatch")


def download_sources(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, checksum in SOURCES:
        path = directory / name.replace(" ", "_")
        if not path.exists():
            encoded = name.replace(" ", "%20")
            url = f"https://zenodo.org/api/records/10899828/files/{encoded}/content"
            temporary = path.with_suffix(".part")
            with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            verify_source(temporary, checksum)
            temporary.replace(path)
        verify_source(path, checksum)


def read_frame(stream: Any, *, option: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(
        stream,
        header=None,
        names=(
            "symbol",
            "date",
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "oi",
        ),
    )
    ignored = int((frame["volume"] == 0).sum()) if option else 0
    if option:
        frame = frame.loc[frame["volume"] != 0].copy()
    frame.attrs["zero_volume_rows_removed"] = ignored
    values = frame[["open", "high", "low", "close", "volume"]]
    invalid = (
        values.isna().any(axis=1)
        | (values[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (frame["volume"] < 0)
        | (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    if invalid.any() and not option:
        raise ValueError("public_archive_invalid_ohlcv")
    frame.attrs["invalid_rows_removed"] = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    frame["date"] = frame["date"].str.replace("/", "-", regex=False)
    if frame.duplicated(["date", "time"]).any():
        raise ValueError("public_archive_duplicate_bars")
    return frame


def normalize_rows(frame: pd.DataFrame, *, option: bool = False) -> list[list[Any]]:
    rows = []
    for row in frame.itertuples(index=False):
        end = datetime.fromisoformat(f"{row.date}T{row.time}:00").replace(tzinfo=IST)
        start = (
            end
            if option and frame.attrs.get("bar_start_labels")
            else end - timedelta(minutes=1)
        )
        if "09:15" <= start.strftime("%H:%M") < "15:30":
            rows.append(
                [start.isoformat(), row.open, row.high, row.low, row.close, row.volume]
            )
    return rows


def select_atm_pair(
    spot: float, expiry: str, available: dict[tuple[int, str], pd.DataFrame], day: str
) -> tuple[int, pd.DataFrame, pd.DataFrame] | None:
    """Fixed opening ATM, nearest monthly expiry; never inspect later liquidity."""
    strike = int(round(spot / 50) * 50)
    if expiry < day:
        return None
    selected = []
    for side in ("CE", "PE"):
        frame = available.get((strike, side))
        if frame is None:
            return None
        session = frame[frame["date"] == day]
        # 09:16 is the assumed end of the opening minute. Later completeness
        # cannot authorize selection; no fallback to a better hindsight strike.
        cutoff = (
            "09:15" if str(frame.iloc[0]["symbol"]).startswith("NIFTY") else "09:16"
        )
        known = session[(session["time"] <= cutoff) & (session["volume"] > 0)]
        if known.empty:
            return None
        selected.append(session)
    return strike, selected[0], selected[1]


def option_members(content: bytes) -> list[tuple[str, bytes]]:
    """Read one textual representation; nested CSV copies must not double count."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        nested = [name for name in archive.namelist() if name.endswith(".zip")]
        if nested:
            selected = [name for name in nested if "TXT" in name.upper()]
            if len(selected) != 1:
                raise ValueError("public_archive_text_representation_ambiguous")
            return option_members(archive.read(selected[0]))
        return [
            (name, archive.read(name))
            for name in archive.namelist()
            if name.endswith((".txt", ".csv"))
        ]


def prepare(source: Path, output: Path) -> dict[str, Any]:
    """Build rotating daily archives without extracting untrusted ZIP paths."""
    download_sources(source)
    context: dict[str, dict[str, pd.DataFrame]] = {}
    with zipfile.ZipFile(source / "Nifty_spot_and_futures_data.zip") as archive:
        for year_file in archive.namelist():
            with zipfile.ZipFile(io.BytesIO(archive.read(year_file))) as yearly:
                for filename in yearly.namelist():
                    if not filename.endswith(".csv"):
                        continue
                    with yearly.open(filename) as stream:
                        frame = read_frame(stream)
                    kind = "FUT" if "_F1" in filename else "EQ"
                    for day, session in frame.groupby("date", sort=True):
                        context.setdefault(str(day), {})[kind] = session
    prepared: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    months: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(source / "Nifty_Options_Data.zip") as archive:
        for year_file in sorted(archive.namelist()):
            with zipfile.ZipFile(io.BytesIO(archive.read(year_file))) as yearly:
                for month_file in yearly.namelist():
                    if month_file.endswith(".zip"):
                        label = Path(month_file).stem
                        if not label[-4:].isdigit():
                            label += " " + Path(year_file).stem[-4:]
                        months.append((label, yearly.read(month_file)))
    for name, content in sorted(
        months, key=lambda item: datetime.strptime(item[0], "%B %Y")
    ):
        month_start = datetime.strptime(name, "%B %Y")
        prior_expiry = month_start - timedelta(days=1)
        while prior_expiry.weekday() != 3:
            prior_expiry -= timedelta(days=1)
        previous_expiry = prior_expiry.date().isoformat()
        available: dict[tuple[int, str], pd.DataFrame] = {}
        expiry = "0000-00-00"
        for filename, raw in option_members(content):
            frame = read_frame(io.BytesIO(raw), option=True)
            frame.attrs["bar_start_labels"] = month_start >= datetime(2018, 3, 1)
            excluded["zero_volume_option_rows_removed"] += frame.attrs[
                "zero_volume_rows_removed"
            ]
            excluded["invalid_option_rows_removed"] += frame.attrs[
                "invalid_rows_removed"
            ]
            label = str(frame.iloc[0]["symbol"]).strip()
            identity = re.fullmatch(r"(CE|PE)\s+(\d+)", label)
            compact = re.fullmatch(r"NIFTY(\d+)(CE|PE)", label)
            dated = re.fullmatch(
                r"NIFTY(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)", label.upper()
            )
            if identity:
                key = (int(identity[2]), identity[1])
            elif compact:
                key = (int(compact[1]), compact[2])
            elif dated:
                key = (int(dated[2]), dated[3])
                named_expiry = datetime.strptime(dated[1], "%d%b%y")
                if named_expiry.strftime("%Y-%m") != month_start.strftime("%Y-%m"):
                    raise ValueError("public_archive_named_expiry_conflict")
            else:
                raise ValueError(
                    f"public_archive_option_identity_invalid: {name}/{filename}: "
                    f"{label}"
                )
            if key in available or not (frame["symbol"].str.strip() == label).all():
                raise ValueError("public_archive_option_identity_conflict")
            available[key] = frame
            expiry = max(expiry, str(frame["date"].max()))
        print(
            f"Parsed {name}: expiry label {expiry}, {len(available)} contracts",
            flush=True,
        )
        if expiry[:7] != month_start.strftime("%Y-%m"):
            raise ValueError("public_archive_expiry_month_mismatch")
        days = sorted(day for day in context if previous_expiry < day <= expiry)
        for day in days:
            session_context = context[day]
            if not {"EQ", "FUT"} <= session_context.keys():
                excluded["missing_underlying"] += 1
                continue
            spot_open = session_context["EQ"].query("time == '09:16'")
            if len(spot_open) != 1:
                excluded["missing_opening_spot"] += 1
                continue
            pair = select_atm_pair(
                float(spot_open.iloc[0]["close"]), expiry, available, day
            )
            if pair is None:
                excluded["opening_atm_pair_unavailable"] += 1
                continue
            strike, ce, pe = pair
            directory = output / "sessions" / day / "candles"
            symbols = {
                "EQ": "NSE:NIFTY 50",
                "FUT": "NFO:NIFTY_F1_CONTEXT",
                "CE": f"NFO:NIFTY{expiry.replace('-', '')}{strike}CE",
                "PE": f"NFO:NIFTY{expiry.replace('-', '')}{strike}PE",
            }
            frames = {**session_context, "CE": ce, "PE": pe}
            counts = {}
            for kind, frame in frames.items():
                rows = normalize_rows(frame, option=kind in {"CE", "PE"})
                counts[kind] = len(rows)
                write_json(
                    directory / f"{kind}.json",
                    {
                        "symbol": symbols[kind],
                        "timestamp_convention": "bar_start",
                        "instrument": {
                            "instrument_type": kind,
                            "name": "NIFTY",
                            "lot_size": 75,
                            "expiry": expiry,
                            "strike": strike if kind in {"CE", "PE"} else 0,
                        },
                        "candles": rows,
                    },
                )
            prepared.append(
                {"day": day, "expiry": expiry, "strike": strike, "bars": counts}
            )
    manifest = {
        "source": "https://doi.org/10.5281/zenodo.10899828",
        "source_md5": dict(SOURCES),
        "sessions": prepared,
        "excluded_sessions": dict(excluded),
        "underlying_sessions": len(context),
        "unprepared_underlying_sessions": len(context) - len(prepared),
        "assumptions": [
            "Context and options before March 2018 assumed bar-end; "
            "later options assumed bar-start; publisher-unverified",
            "Monthly expiry label inferred from last traded date across that folder",
            "Continuous NIFTY_F1 context; historical future roll identity unverified",
            "Fixed daily ATM from opening-minute spot close; "
            "no intraday basket rotation",
            "Quantity standardized to 75 units; current modeled fees, "
            "not historical account returns",
            "No executable bid/ask or depth; slippage scenarios do not prove fills",
        ],
    }
    write_json(output / "data_manifest.json", manifest)
    return manifest


def run_task(task: tuple[str, dict[str, Any], float, list[str]]) -> dict[str, Any]:
    root, candidate, slippage, days = task
    os.environ.update(
        EXECUTION_MODE="SHADOW",
        STRATEGY_MODE="directional_scalp",
        ORB_ENABLED="true",
        BROKER_API_KEY="offline_research_unused",
        BROKER_API_SECRET="offline_research_unused",
    )
    logging.disable(logging.CRITICAL)
    trades: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for index, day in enumerate(days):
        result = run_orb_session_research(
            Path(root) / "sessions" / day,
            overrides=candidate["overrides"],
            slippage_bps=slippage,
            minimum_net_rr=candidate["minimum_net_rr"],
        )
        trades.extend(result["trades"])
        reasons.update(result["no_vote_reasons"])
        if (index + 1) % 100 == 0:
            print(
                f"{candidate['name']} / {slippage} bps: "
                f"{index + 1}/{len(days)} sessions",
                flush=True,
            )
    return {
        "candidate": candidate["name"],
        "slippage_bps_per_side": slippage,
        "metrics": summarize(trades),
        "yearly_metrics": {
            year: summarize(
                [trade for trade in trades if trade["entry_time"].startswith(year)]
            )
            for year in sorted({day[:4] for day in days})
        },
        "no_vote_reasons": dict(reasons),
        "exit_reasons": dict(Counter(trade["exit_reason"] for trade in trades)),
        "trades": trades,
    }


def select_candidate(results: list[dict[str, Any]]) -> str | None:
    """Only development results may select; require a meaningful stress sample."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if row["candidate"] != "raw_reference":
            grouped.setdefault(row["candidate"], []).append(row)
    eligible = [
        (min(row["metrics"]["expectancy"] for row in rows), name)
        for name, rows in grouped.items()
        if {row["slippage_bps_per_side"] for row in rows} == set(SLIPPAGE)
        and len(rows) == len(SLIPPAGE)
        and all(row["metrics"]["trade_count"] >= 100 for row in rows)
    ]
    return max(eligible)[1] if eligible else None


def run_phase(output: Path, phase: str, workers: int) -> None:
    protocol_path = output / "protocol.json"
    protocol = {
        "candidates": candidates(),
        "slippage_bps_per_side": SLIPPAGE,
        "development_years": ["2017", "2018"],
        "validation_years": ["2019"],
        "final_years": ["2020"],
        "selection_rule": (
            "Max worst-slippage development expectancy; "
            ">=100 trades at every stress level"
        ),
        "entry_window": (
            "09:30 through 120 minutes after the 15-minute range; " "early candidate 60"
        ),
        "exit_rule": (
            "Next-minute open; stop-first ambiguity; "
            "missing/zero-volume exit charged full-premium loss"
        ),
        "promotion_eligible": False,
        "replay_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
            + (
                ROOT / "src/nifty_scalper_bot/backtesting/strategy_research.py"
            ).read_bytes()
        ).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            (output / "data_manifest.json").read_bytes()
        ).hexdigest(),
    }
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != json.loads(
        json.dumps(protocol)
    ):
        raise ValueError("research_protocol_changed")
    write_json(protocol_path, protocol)
    manifest = json.loads((output / "data_manifest.json").read_text())
    years = protocol[f"{phase}_years"]
    days = [row["day"] for row in manifest["sessions"] if row["day"][:4] in years]
    selected = protocol["candidates"]
    if phase != "development":
        frozen = json.loads((output / "selection.json").read_text())
        selected = [
            row
            for row in selected
            if row["name"] in {"raw_reference", "stop_0.75_rr_1.8", frozen["candidate"]}
        ]
    tasks = [
        (str(output), candidate, slip, days)
        for candidate in selected
        for slip in SLIPPAGE
    ]
    results = []
    phase_dir = output / phase
    pending = []
    for task in tasks:
        path = phase_dir / f"{task[1]['name']}_{task[2]}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
        else:
            pending.append(task)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_task, task): task for task in pending}
        for future in as_completed(futures):
            row = future.result()
            write_json(
                phase_dir / f"{row['candidate']}_{row['slippage_bps_per_side']}.json",
                row,
            )
            results.append(row)
            print(
                f"Completed {phase}: {row['candidate']} / "
                f"{row['slippage_bps_per_side']}: {row['metrics']}",
                flush=True,
            )
    results.sort(key=lambda row: (row["candidate"], row["slippage_bps_per_side"]))
    write_json(output / f"{phase}_results.json", results)
    if phase == "development":
        write_json(
            output / "selection.json",
            {
                "candidate": select_candidate(results),
                "selected_from": "2017–2018 only",
                "protocol_sha256": hashlib.sha256(
                    protocol_path.read_bytes()
                ).hexdigest(),
                "promotion_eligible": False,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "development", "validation", "final")
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    if args.action == "prepare":
        manifest = prepare(args.source, args.output)
        print(
            f"Prepared {len(manifest['sessions'])} sessions; "
            f"exclusions {manifest['excluded_sessions']}"
        )
    else:
        run_phase(args.output, args.action, args.workers)


if __name__ == "__main__":
    main()
