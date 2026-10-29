"""Read-only observation-quality audit for saved live market-data events.

Strictly validate archive writer counters, unbroken sequence and immutable
runtime snapshot using the bot's existing validators. This does not
retroactively recover order-book messages, exchange fills, or missing quotes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from nifty_scalper_bot.backtesting.runtime_research import validate_replay_epochs
from nifty_scalper_bot.storage.replay_archive import archive_fingerprint, iter_session

IST = ZoneInfo("Asia/Kolkata")


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _depth(tick: Mapping[str, Any]) -> tuple[bool, bool]:
    depth = tick.get("depth")
    if not isinstance(depth, dict):
        return False, False
    counts = []
    first = []
    for side in ("buy", "sell"):
        levels = depth.get(side)
        if not isinstance(levels, list) or not levels:
            return False, False
        valid = [
            row
            for row in levels[:5]
            if isinstance(row, dict)
            and _number(row.get("price"))
            and type(row.get("quantity")) is int
            and row["quantity"] > 0
        ]
        if len(valid) != min(len(levels), 5):
            return False, False
        counts.append(len(valid))
        first.append(valid[0]["price"])
    top = first[0] < first[1]
    return top, top and counts == [5, 5]


def _selected(snapshot: Mapping[str, Any]) -> dict[str, str]:
    basket = snapshot.get("basket")
    if not isinstance(basket, dict):
        return {}
    ce = basket.get("selected_ce")
    pe = basket.get("selected_pe")
    future = basket.get("futures_symbol")
    if not (
        isinstance(ce, str)
        and ce
        and isinstance(pe, str)
        and pe
        and isinstance(future, str)
        and future
    ):
        return {}
    selected: dict[str, str] = {
        "CE": ce,
        "PE": pe,
        "FUT": future,
        "SPOT": "NSE:NIFTY 50",
    }
    if not all(
        selected[k].startswith("NFO:") and selected[k].endswith(k)
        for k in ("CE", "PE", "FUT")
    ):
        return {}
    return selected


def audit_session(path: Path, *, gap_seconds: float = 120) -> dict[str, Any]:
    if not math.isfinite(gap_seconds) or not 0 < gap_seconds <= 900:
        raise ValueError("replay_audit_invalid_gap")
    events, first, snapshots = validate_replay_epochs(path)
    active = _selected(first)
    if not active:
        raise ValueError("replay_audit_missing_selected_contracts")
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    minutes: dict[str, set[str]] = defaultdict(set)
    previous: dict[str, datetime] = {}
    max_gap: dict[str, float] = defaultdict(float)
    basket_versions: set[tuple[str, str, str]] = set()
    # A snapshot can replace the selected CE/PE while preserving replay
    # configuration. Validate each recorded segment rather than treating all
    # quote coverage as if one basket were active for the entire process run.
    basket_segments: list[dict[str, Counter[str]]] = []
    current_segment: dict[str, Counter[str]] | None = None
    current_selection: tuple[str, str, str] | None = None
    unmatched = 0
    missing_symbol = 0
    for event in iter_session(path):
        t = datetime.fromisoformat(event["available_at"]).astimezone(IST)
        if t.date().isoformat() != path.stem:
            raise ValueError("replay_audit_session_date_mismatch")
        if event["kind"] == "snapshot":
            active = _selected(event["payload"])
            if not active:
                raise ValueError("replay_audit_snapshot_basket_unidentified")
            selection = (active["CE"], active["PE"], active["FUT"])
            basket_versions.add(selection)
            # A repeated snapshot of the SAME selected basket is not a new
            # selection epoch and cannot create a synthetic zero-tick segment.
            if current_segment is None or selection != current_selection:
                current_segment = defaultdict(Counter)
                basket_segments.append(current_segment)
                previous.clear()
            current_selection = selection
            continue
        tick = event["payload"]
        symbol = tick.get("symbol")
        if not isinstance(symbol, str):
            missing_symbol += 1
            continue
        role = next((k for k, v in active.items() if v == symbol), None)
        if role is None and symbol == "NSE:NIFTY":
            role = "SPOT"
        if role is None:
            unmatched += 1
            continue
        c = counts[role]
        c["ticks"] += 1
        if current_segment is not None:
            current_segment[role]["ticks"] += 1
        if 9 * 60 + 15 <= t.hour * 60 + t.minute < 15 * 60 + 30:
            c["market_session_ticks"] += 1
            minutes[role].add(t.strftime("%H:%M"))
        if str(tick.get("source") or "").lower() in ("ws", "websocket"):
            c["websocket_ticks"] += 1
            if current_segment is not None:
                current_segment[role]["websocket_ticks"] += 1
        if _number(tick.get("ltp")):
            c["positive_ltp_ticks"] += 1
        top, full = _depth(tick)
        if top:
            c["uncrossed_two_sided_depth_ticks"] += 1
            if current_segment is not None:
                current_segment[role]["uncrossed_two_sided_depth_ticks"] += 1
        if full:
            c["five_level_depth_ticks"] += 1
            if current_segment is not None:
                current_segment[role]["five_level_depth_ticks"] += 1
        if type(tick.get("instrument_token")) is int and tick["instrument_token"] > 0:
            c["token_identified_ticks"] += 1
        prior = previous.get(role)
        if prior:
            seconds = (t - prior).total_seconds()
            max_gap[role] = max(max_gap[role], seconds)
            if seconds > gap_seconds:
                c["gaps_over_threshold"] += 1
        previous[role] = t
    roles: dict[str, dict[str, Any]] = {}
    for role in ("CE", "PE", "FUT", "SPOT"):
        c = counts[role]
        n = c["ticks"]
        roles[role] = {
            "observed_ticks": n,
            "market_session_ticks": c["market_session_ticks"],
            "observed_market_minutes": len(minutes[role]),
            "websocket_ticks": c["websocket_ticks"],
            "uncrossed_two_sided_depth_ticks": c["uncrossed_two_sided_depth_ticks"],
            "five_level_depth_ticks": c["five_level_depth_ticks"],
            "token_identified_ticks": c["token_identified_ticks"],
            "positive_ltp_ticks": c["positive_ltp_ticks"],
            "gaps_over_threshold": c["gaps_over_threshold"],
            "longest_observed_intertick_gap_seconds": max_gap[role] if n > 1 else None,
            "share_with_five_level_depth": (
                c["five_level_depth_ticks"] / n if n else None
            ),
        }
    # Rotation itself is not a data-quality defect. A snapshot creates a
    # fresh coverage obligation for the new selected contracts, even if the
    # same basket was seen previously. Incomplete segments remain blocked.
    missing_requirements: Counter[str] = Counter()
    verified_segments = 0
    for segment in basket_segments:
        segment_missing = []
        for role in ("CE", "PE", "FUT", "SPOT"):
            if segment[role]["ticks"] == 0:
                segment_missing.append(f"{role}_no_ticks")
        for role in ("CE", "PE", "FUT"):
            if segment[role]["websocket_ticks"] == 0:
                segment_missing.append(f"{role}_no_websocket_ticks")
        for role in ("CE", "PE"):
            if segment[role]["uncrossed_two_sided_depth_ticks"] == 0:
                segment_missing.append(f"{role}_no_two_sided_depth")
            if segment[role]["five_level_depth_ticks"] == 0:
                segment_missing.append(f"{role}_no_five_level_depth")
        if segment_missing:
            missing_requirements.update(segment_missing)
        else:
            verified_segments += 1
    # Verified *observations* do not establish complete exchange message
    # delivery, simulated broker fill feasibility or executable alpha.
    usable_context = bool(basket_segments) and (
        verified_segments == len(basket_segments)
    )
    return {
        "profile": "replay_capture_fidelity_v1",
        "evidence_label": "RESEARCH_CANDIDATE",
        "session_date": path.stem,
        "capture_sha256": archive_fingerprint(path),
        "event_count": events,
        "snapshot_count": snapshots,
        "basket_versions": len(basket_versions),
        "basket_segments": len(basket_segments),
        "basket_segments_with_quote_context": verified_segments,
        "basket_segments_missing_quote_context": (
            len(basket_segments) - verified_segments
        ),
        "basket_segments_missing_by_requirement": dict(missing_requirements),
        "unselected_ticks": unmatched,
        "missing_symbol_ticks": missing_symbol,
        "gap_threshold_seconds": gap_seconds,
        "roles": roles,
        "archive_integrity_verified": True,
        "observed_selected_context_available": usable_context,
        "actual_exchange_stream_completeness_proven": False,
        "historical_exchange_fills_available": False,
        "live_equivalent": False,
        "promotion_eligible": False,
        "account_pnl": None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gap-seconds", type=float, default=120)
    args = p.parse_args()
    report = audit_session(args.session, gap_seconds=args.gap_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "events": report["event_count"],
                "context_available": report["observed_selected_context_available"],
                "live_equivalent": False,
            }
        )
    )


if __name__ == "__main__":
    main()
