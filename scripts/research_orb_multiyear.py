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
import math
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

from nifty_scalper_bot.backtesting.directional_hypotheses import (  # noqa: E402
    _research_strategies,
)
from nifty_scalper_bot.backtesting.strategy_research import (  # noqa: E402
    _scenario,
    load_archive,
    run_orb_session_research,
    summarize,
)
from nifty_scalper_bot.config.settings import get_settings  # noqa: E402
from nifty_scalper_bot.risk.cost_model import estimate_round_trip_cost  # noqa: E402

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
    unresolved: list[dict[str, Any]] = []
    stress_trades: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for index, day in enumerate(days):
        result = run_orb_session_research(
            Path(root) / "sessions" / day,
            overrides=candidate["overrides"],
            slippage_bps=slippage,
            minimum_net_rr=candidate["minimum_net_rr"],
        )
        trades.extend(result["trades"])
        unresolved.extend(result["unresolved_trades"])
        stress_trades.extend(result["worst_case_stress_trades"])
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
        "worst_case_stress_metrics": summarize(stress_trades),
        "data_quality": {
            "resolved_exit_count": len(trades),
            "unresolved_exit_count": len(unresolved),
            "unresolved_exit_rate": (
                len(unresolved) / (len(trades) + len(unresolved))
                if trades or unresolved
                else 0.0
            ),
            "primary_metrics_complete": not unresolved,
            "primary_metrics_exclude_unresolved": True,
        },
        "yearly_metrics": {
            year: summarize(
                [trade for trade in trades if trade["entry_time"].startswith(year)]
            )
            for year in sorted({day[:4] for day in days})
        },
        "no_vote_reasons": dict(reasons),
        "exit_reasons": dict(Counter(trade["exit_reason"] for trade in trades)),
        "unresolved_trades": unresolved,
        "trades": trades,
        "worst_case_stress_trades": stress_trades,
    }


def _group_candidate_results(
    results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if row["candidate"] != "raw_reference":
            grouped.setdefault(row["candidate"], []).append(row)
    return grouped


def select_candidate(results: list[dict[str, Any]]) -> str | None:
    """Only development results may select; require a meaningful stress sample."""
    grouped = _group_candidate_results(results)
    eligible = [
        (min(row["metrics"]["expectancy"] for row in rows), name)
        for name, rows in grouped.items()
        if {row["slippage_bps_per_side"] for row in rows} == set(SLIPPAGE)
        and len(rows) == len(SLIPPAGE)
        and all(row["metrics"]["trade_count"] >= 100 for row in rows)
        and all(row["data_quality"]["unresolved_exit_count"] == 0 for row in rows)
    ]
    return max(eligible)[1] if eligible else None


def select_exploratory_top5(results: list[dict[str, Any]]) -> list[str]:
    """Freeze five development-only candidates for later out-of-sample comparison.

    This is exploratory evidence, never a live-promotion rule. Candidates need
    at least 100 resolved trades in the 10 bps development scenario. Ranking
    uses 10 bps post-cost expectancy, then the worst expectancy across the three
    registered slippage scenarios as a robustness tie-breaker.
    """
    ranked: list[tuple[float, float, str]] = []
    for name, rows in _group_candidate_results(results).items():
        if {row["slippage_bps_per_side"] for row in rows} != set(SLIPPAGE):
            continue
        by_slip = {row["slippage_bps_per_side"]: row for row in rows}
        base = by_slip[10.0]
        if base["metrics"]["trade_count"] < 100:
            continue
        expectancy = base["metrics"]["expectancy"]
        if expectancy is None:
            continue
        worst = min(
            row["metrics"]["expectancy"]
            for row in rows
            if row["metrics"]["expectancy"] is not None
        )
        ranked.append((float(expectancy), float(worst), name))
    ranked.sort(reverse=True)
    return [name for _, _, name in ranked[:5]]


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
            ">=100 resolved trades at every stress level; "
            "zero unresolved exits"
        ),
        "entry_window": (
            "09:30 through 120 minutes after the 15-minute range; " "early candidate 60"
        ),
        "exit_rule": (
            "Next-minute open; ambiguous stop/target and missing/zero-volume "
            "exits are unresolved in primary metrics; stop-first/full-premium "
            "losses are reported only as worst-case stress"
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
        selected_names = {
            "raw_reference",
            "stop_0.75_rr_1.8",
            *frozen.get("exploratory_top5", []),
        }
        if frozen["candidate"] is not None:
            selected_names.add(frozen["candidate"])
        selected = [row for row in selected if row["name"] in selected_names]
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
                "exploratory_top5": select_exploratory_top5(results),
                "exploratory_top5_rule": (
                    "development-only 10 bps expectancy; >=100 resolved trades "
                    "at 10 bps; worst three-slippage expectancy tie-breaker"
                ),
                "selected_from": "2017–2018 only",
                "protocol_sha256": hashlib.sha256(
                    protocol_path.read_bytes()
                ).hexdigest(),
                "promotion_eligible": False,
            },
        )


def run_fixed_hypothesis_day(task: tuple[str, str]) -> dict[str, Any]:
    """Evaluate research-only signals and cost-gated ORB on one fixed daily basket."""
    from nifty_scalper_bot.backtesting.directional_hypotheses import (
        run_directional_hypothesis_research,
    )

    root, day = task
    os.environ.update(
        EXECUTION_MODE="SHADOW",
        STRATEGY_MODE="directional_scalp",
        ORB_ENABLED="true",
        BROKER_API_KEY="offline_research_unused",
        BROKER_API_SECRET="offline_research_unused",
    )
    logging.disable(logging.CRITICAL)
    directory = Path(root) / "sessions" / day
    research = run_directional_hypothesis_research(directory)
    for scenario in research["scenarios"]:
        slippage = scenario["slippage_bps_per_side"]
        baseline = run_orb_session_research(
            directory,
            overrides={},
            slippage_bps=slippage,
            minimum_net_rr=1.5,
            lifecycle_proxy=True,
        )
        scenario["strategies"]["ORBNetGateBaseline"] = baseline
        scaled = run_orb_session_research(
            directory,
            overrides={},
            slippage_bps=slippage,
            minimum_net_rr=1.5,
            lifecycle_proxy=True,
            normalize_opening_range=True,
        )
        scenario["strategies"]["ORBRangeScaledResearch"] = scaled
    # Policy-level ablation, fixed geometry and costs, no strategy tuning.
    # Turning trailing off can change later admissions; aggregate and matched
    # entry comparisons are therefore both explicitly reported below.
    histories, instruments, _ = load_archive(directory)
    no_trailing = _scenario(
        histories,
        instruments,
        get_settings().elite,
        10.0,
        components={"PullbackResearch", "CompressionResearch", "MomentumResearch"},
        minimum_net_rr=1.5,
        strict_liquidity=True,
        lifecycle_proxy=False,
        strategy_factory=_research_strategies,
    )
    # A fixed cost-economics hypothesis, not parameter searching: compare
    # original 1.8R with 2.2R under the same ORB range-scaling experiment.
    # The 1.5 net-RR gate and all structural confirmations remain intact.
    target_2_2 = run_orb_session_research(
        directory,
        overrides={"ORB_TARGET_RR": "2.2"},
        slippage_bps=10.0,
        minimum_net_rr=1.5,
        lifecycle_proxy=True,
        normalize_opening_range=True,
    )
    return {
        "day": day,
        "scenarios": research["scenarios"],
        "exit_ablation": no_trailing["strategies"],
        "orb_target_2_2": target_2_2,
    }


def summarize_fixed_hypothesis_days(
    daily: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep each chronological period distinct; never select on later years."""
    names = (
        "ORBNetGateBaseline",
        "ORBRangeScaledResearch",
        "PullbackResearch",
        "CompressionResearch",
        "MomentumResearch",
    )
    years = {
        "development": {"2017", "2018"},
        "validation": {"2019"},
        "final": {"2020"},
    }
    if len({row["day"] for row in daily}) != len(daily):
        raise ValueError("hypothesis_duplicate_session")
    results: list[dict[str, Any]] = []
    for phase, years_in_phase in years.items():
        sessions = [row for row in daily if row["day"][:4] in years_in_phase]
        for slippage in SLIPPAGE:
            for name in names:
                trades: list[dict[str, Any]] = []
                unresolved: list[dict[str, Any]] = []
                stressed: list[dict[str, Any]] = []
                rejections: Counter[str] = Counter()
                gate_counts: Counter[str] = Counter()
                for session in sessions:
                    matches = [
                        row
                        for row in session["scenarios"]
                        if row["slippage_bps_per_side"] == slippage
                    ]
                    if len(matches) != 1 or set(matches[0]["strategies"]) != set(names):
                        raise ValueError("hypothesis_scenario_incomplete")
                    observed = matches[0]["strategies"][name]
                    trades.extend(observed["trades"])
                    unresolved.extend(observed["unresolved_trades"])
                    stressed.extend(observed["worst_case_stress_trades"])
                    rejections.update(observed["no_vote_reasons"])
                    gate_counts.update(observed.get("research_gate_counts", {}))
                results.append(
                    {
                        "phase": phase,
                        "candidate": name,
                        "slippage_bps_per_side": slippage,
                        "session_count": len(sessions),
                        "metrics": summarize(trades),
                        "worst_case_stress_metrics": summarize(stressed),
                        "data_quality": {
                            "resolved_exit_count": len(trades),
                            "unresolved_exit_count": len(unresolved),
                            "primary_metrics_complete": not unresolved,
                            "primary_metrics_exclude_unresolved": True,
                        },
                        "no_vote_reasons": dict(rejections),
                        "research_gate_counts": dict(gate_counts),
                        "exit_reasons": dict(
                            Counter(trade["exit_reason"] for trade in trades)
                        ),
                        "unresolved_trades": unresolved,
                        "trades": trades,
                        "worst_case_stress_trades": stressed,
                    }
                )
    return results


def summarize_exit_lifecycle_ablation(
    daily: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare original vs non-trailing proxy; keep shared entries identifiable.

    This is not a randomized or current-live exit comparison. Holding positions
    longer changes which subsequent entry intents the simulator can admit.
    """
    years = {
        "development": {"2017", "2018"},
        "validation": {"2019"},
        "final": {"2020"},
    }
    families = ("PullbackResearch", "CompressionResearch", "MomentumResearch")
    if len({row["day"] for row in daily}) != len(daily):
        raise ValueError("hypothesis_duplicate_session")
    summary: list[dict[str, Any]] = []
    for phase, year_set in years.items():
        sessions = [row for row in daily if row["day"][:4] in year_set]
        for family in families:
            original: list[dict[str, Any]] = []
            alternate: list[dict[str, Any]] = []
            unresolved = [0, 0]
            for session in sessions:
                before = [
                    item
                    for item in session["scenarios"]
                    if item["slippage_bps_per_side"] == 10.0
                ]
                after = session.get("exit_ablation", {})
                if len(before) != 1 or family not in after:
                    raise ValueError("exit_ablation_scenario_incomplete")
                old = before[0]["strategies"][family]
                new = after[family]
                original.extend(old["trades"])
                alternate.extend(new["trades"])
                unresolved[0] += len(old["unresolved_trades"])
                unresolved[1] += len(new["unresolved_trades"])

            def by_entry(
                trades: list[dict[str, Any]],
            ) -> dict[tuple[str, str], dict[str, Any]]:
                keyed = {(t["symbol"], t["entry_time"]): t for t in trades}
                if len(keyed) != len(trades):
                    raise ValueError("exit_ablation_duplicate_trade_identity")
                return keyed

            a, b = by_entry(original), by_entry(alternate)
            matched = sorted(a.keys() & b.keys())
            summary.append(
                {
                    "phase": phase,
                    "candidate": family,
                    "slippage_bps_per_side": 10.0,
                    "method": "one_policy_change_no_bar_lifecycle_proxy",
                    "original": {
                        "metrics": summarize(original),
                        "unresolved_exit_count": unresolved[0],
                    },
                    "no_trailing": {
                        "metrics": summarize(alternate),
                        "unresolved_exit_count": unresolved[1],
                    },
                    "identical_entry_count": len(matched),
                    "identical_entry_original_metrics": summarize(
                        [a[k] for k in matched]
                    ),
                    "identical_entry_no_trailing_metrics": summarize(
                        [b[k] for k in matched]
                    ),
                    "caveat": "nonmatched_accepted_entries_change_with_holding_time",
                }
            )
    return summary


def summarize_orb_target_sensitivity(
    daily: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Contrast the existing 1.8R and fixed 2.2R target after identical costs.

    Changed targets alter which entries pass net-RR and subsequent position
    occupancy; shared-entry outcomes are therefore reported separately.
    """
    years = {
        "development": {"2017", "2018"},
        "validation": {"2019"},
        "final": {"2020"},
    }
    if len({row["day"] for row in daily}) != len(daily):
        raise ValueError("hypothesis_duplicate_session")
    results: list[dict[str, Any]] = []
    for phase, period in years.items():
        baseline: list[dict[str, Any]] = []
        candidate: list[dict[str, Any]] = []
        missing = [0, 0]
        rejected = [0, 0]
        for day in (row for row in daily if row["day"][:4] in period):
            matched = [
                row
                for row in day["scenarios"]
                if row["slippage_bps_per_side"] == 10.0
            ]
            if len(matched) != 1 or not isinstance(
                day.get("orb_target_2_2"), dict
            ):
                raise ValueError("orb_target_sensitivity_incomplete")
            base = matched[0]["strategies"]["ORBRangeScaledResearch"]
            alt = day["orb_target_2_2"]
            baseline.extend(base["trades"])
            candidate.extend(alt["trades"])
            missing[0] += len(base["unresolved_trades"])
            missing[1] += len(alt["unresolved_trades"])
            rejected[0] += base["no_vote_reasons"].get("cost_net_rr_rejected", 0)
            rejected[1] += alt["no_vote_reasons"].get("cost_net_rr_rejected", 0)

        def index_by_entry(
            trades: list[dict[str, Any]],
        ) -> dict[tuple[str, str], dict[str, Any]]:
            indexed = {(t["symbol"], t["entry_time"]): t for t in trades}
            if len(indexed) != len(trades):
                raise ValueError("orb_target_duplicate_entry")
            return indexed

        original = index_by_entry(baseline)
        changed = index_by_entry(candidate)
        common = sorted(original.keys() & changed.keys())
        results.append(
            {
                "phase": phase,
                "method": "same_scaled_orb_structural_rules_1_8R_vs_2_2R",
                "slippage_bps_per_side": 10.0,
                "minimum_net_rr": 1.5,
                "baseline_target_rr": 1.8,
                "research_target_rr": 2.2,
                "baseline": {
                    "metrics": summarize(baseline),
                    "unresolved_exit_count": missing[0],
                    "cost_net_rr_rejections": rejected[0],
                },
                "target_2_2": {
                    "metrics": summarize(candidate),
                    "unresolved_exit_count": missing[1],
                    "cost_net_rr_rejections": rejected[1],
                },
                "identical_entry_count": len(common),
                "identical_entry_baseline": summarize([original[k] for k in common]),
                "identical_entry_target_2_2": summarize([changed[k] for k in common]),
                "caveat": "target_distance_changes_admission_and_realized_exits",
            }
        )
    return results


def fixed_cohort_slippage_stress(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reprice one fixed 10-bps filled-trade cohort at higher adverse slippage.

    This changes only execution prices and canonical fees, never accepted
    trades, timestamps, stop/target path, or strategy signal decisions. It is
    a cost-only counterfactual, not alternative executable fills.
    """
    stress: list[dict[str, Any]] = []
    for row in results:
        if row["slippage_bps_per_side"] != 10.0:
            continue
        trades = row["trades"]
        if any(t.get("cost_source") != "canonical_model_estimate" for t in trades):
            raise ValueError("research_cost_source_unverified")
        for bps in SLIPPAGE:
            repriced: list[dict[str, Any]] = []
            for trade in trades:
                entry = float(trade["entry_price"]) / 1.001
                exit_price = float(trade["exit_price"]) / 0.999
                qty = int(trade["quantity"])
                if not (
                    math.isfinite(entry)
                    and math.isfinite(exit_price)
                    and entry > 0
                    and exit_price > 0
                    and qty > 0
                ):
                    raise ValueError("research_trade_geometry_invalid")
                adverse = bps / 10000.0
                bought = entry * (1.0 + adverse)
                sold = exit_price * (1.0 - adverse)
                cost = estimate_round_trip_cost(
                    entry_price=bought, exit_price=sold, quantity=qty
                )
                gross = (sold - bought) * qty
                repriced.append(
                    {
                        **trade,
                        "entry_price": bought,
                        "exit_price": sold,
                        "gross_pnl": gross,
                        "fees": cost.total,
                        "net_pnl": gross - cost.total,
                    }
                )
            stress.append(
                {
                    "phase": row["phase"],
                    "candidate": row["candidate"],
                    "slippage_bps_per_side": bps,
                    "method": "same_10bps_resolved_trade_cohort",
                    "resolved_trade_count": len(trades),
                    "unresolved_exit_count": len(row["unresolved_trades"]),
                    "metrics": summarize(repriced),
                }
            )
    return stress


def run_fixed_hypothesis_study(output: Path, workers: int) -> None:
    """Run prespecified comparative research without altering ORB's protocol."""
    if os.getenv("EXECUTION_MODE", "SHADOW").upper() != "SHADOW":
        raise ValueError("research_requires_shadow_process")
    manifest_path = output / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    days = sorted(row["day"] for row in manifest["sessions"])
    if not days:
        raise ValueError("research_history_unavailable")
    directory = output / "fixed_hypotheses"
    pending = [
        (str(output), day) for day in days if not (directory / f"{day}.json").exists()
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_fixed_hypothesis_day, task): task for task in pending
        }
        for future in as_completed(futures):
            row = future.result()
            if row["day"] != futures[future][1]:
                raise ValueError("hypothesis_session_identity_conflict")
            write_json(directory / f"{row['day']}.json", row)
    daily = [json.loads((directory / f"{day}.json").read_text()) for day in days]
    results = summarize_fixed_hypothesis_days(daily)
    cost_stress = fixed_cohort_slippage_stress(results)
    exit_ablation = summarize_exit_lifecycle_ablation(daily)
    target_sensitivity = summarize_orb_target_sensitivity(daily)
    write_json(
        output / "fixed_hypothesis_results.json",
        {
            "scope": "fixed_monthly_option_research_hypotheses",
            "evidence_label": "RESEARCH_CANDIDATE",
            "live_equivalent": False,
            "promotion_eligible": False,
            "selected_for_live": None,
            "data_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "coverage": {
                "session_count": len(days),
                "first_session": days[0],
                "last_session": days[-1],
            },
            "assumptions": {
                "families": [
                    "ORBNetGateBaseline",
                    "ORBRangeScaledResearch",
                    "PullbackResearch",
                    "CompressionResearch",
                    "MomentumResearch",
                ],
                "basket": "fixed_session_open_ATM_monthly_options",
                "direction": "closed_NIFTY_continuous_futures_bars",
                "selection": "fixed_rules_no_parameter_search",
                "orb_range_normalization": (
                    "research_only_15min_range_divided_by_sqrt_of_minutes_"
                    "before_comparing_with_1min_ATR"
                ),
                "execution": "next_minute_open_canonical_costs_and_slippage",
                "comparisons": "1.5_net_rr_gate_and_causal_bar_lifecycle_proxy",
                "orb_target_sensitivity": "fixed_1.8R_vs_2.2R_10bps_scaled_ORB",
                "periods": {
                    "development": ["2017", "2018"],
                    "validation": ["2019"],
                    "final": ["2020"],
                },
            },
            "limitations": [
                "Retrospective archive, including previously examined years",
                "Publisher-unverified early timestamp conventions",
                "Continuous futures roll identity is not reconstructed",
                "Monthly options differ from current weekly products",
                "No bid/ask depth or execution queue; modeled fills only",
                "No production basket selection, arbitration or broker parity",
                "Unresolved exits excluded from primary and shown separately",
                "Opening-range normalization was proposed after earlier ORB results",
                "No-trailing changes admission by keeping positions open longer",
                "2.2R ORB target is post-hoc research, not an untouched holdout",
                "No setting selected or authorized for live trading",
            ],
            "results": results,
            "fixed_10bps_cohort_slippage_stress": cost_stress,
            "fixed_10bps_exit_lifecycle_ablation": exit_ablation,
            "fixed_10bps_orb_target_sensitivity": target_sensitivity,
            "research_quality": {
                "matched_orb_baseline_has_trades": any(
                    row["candidate"] == "ORBNetGateBaseline"
                    and row["metrics"]["trade_count"] > 0
                    for row in results
                ),
                "trade_cohorts_vary_across_replayed_slippage": True,
                "fixed_cohort_stress_is_not_reexecuted_market_path": True,
                "all_strategies_research_only": True,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("prepare", "development", "validation", "final", "hypotheses"),
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
    elif args.action == "hypotheses":
        run_fixed_hypothesis_study(args.output, args.workers)
    else:
        run_phase(args.output, args.action, args.workers)


if __name__ == "__main__":
    main()
