#!/usr/bin/env python3
"""Audit matched NIFTY events against independently supplied as-of evidence.

Responsibilities: Reject inferred archive identities, unverified clocks and
directional conflicts before using the canonical observed-depth simulator.
Ownership: Historical SHADOW research only; never broker execution or tuning.
Safety: Missing point-in-time books remain UNRESOLVED, not OHLC proxy fills.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from nifty_scalper_bot.backtesting.execution_evidence import (
    BookLevel,
    ObservedBook,
    OrderIntent,
    replay_market_order,
)
from nifty_scalper_bot.core.underlying_direction import (
    UnderlyingDirectionObservation,
    arbitrate_underlying_direction,
)


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("pit_timestamp_unverified")
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("pit_timestamp_unverified") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("pit_timestamp_unverified")
    return stamp


def _sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _contract(
    event: Mapping[str, Any], record: Mapping[str, Any], decision: datetime
) -> tuple[str, Mapping[str, Any] | None]:
    """Require an independently dated exchange-master assertion, not a candle label."""
    if not record:
        return "HISTORICAL_EXCHANGE_MASTER_MISSING", None
    try:
        source_time = _time(record["source_observed_at"])
        listed = _time(record["listed_at"])
        expiry = date.fromisoformat(record["expiry"])
        token = str(record["token"])
        lot_size = record["lot_size"]
        strike = float(record["strike"])
        if (
            record.get("source_kind") != "NSE_POINT_IN_TIME_INSTRUMENT_MASTER"
            or record.get("independent_source_verified") is not True
            or not _sha(record.get("source_sha256"))
            or record.get("symbol") != event["symbol"]
            or record.get("right") != event["side"]
            or not token.isdigit()
            or not int(token) > 0
            or type(lot_size) is not int
            or lot_size <= 0
            or not math.isfinite(strike)
            or strike <= 0
            or listed > decision
            or source_time > decision
            or expiry < decision.date()
            or (expiry - decision.date()).days > 7
        ):
            return "POINT_IN_TIME_CONTRACT_PROOF_INVALID", None
        archived = event["archive_instrument"]
        if (
            archived.get("expiry") != expiry.isoformat()
            or float(archived.get("strike")) != strike
            or archived.get("lot_size") != lot_size
        ):
            return "ARCHIVED_CONTRACT_LABEL_CONFLICT", None
    except (KeyError, TypeError, ValueError, OverflowError):
        return "POINT_IN_TIME_CONTRACT_PROOF_INVALID", None
    return "POINT_IN_TIME_CONTRACT_ASSERTED", record


def _direction(
    event: Mapping[str, Any], evidence: Mapping[str, Any], decision: datetime
) -> tuple[str, str | None]:
    """Adjudicate fresh underlying observations with the runtime owner."""
    if not evidence:
        return "UNDERLYING_SOURCE_PROVENANCE_MISSING", None
    future = evidence.get("futures_contract") or {}
    if (
        evidence.get("clock_verified") is not True
        or evidence.get("futures_roll_identity_verified") is not True
        or not _sha(evidence.get("source_sha256"))
        or future.get("continuous_alias") != event["future"]
        or not str(future.get("actual_symbol", "")).startswith("NFO:NIFTY")
        or not str(future.get("instrument_token", "")).isdigit()
        or not int(future["instrument_token"]) > 0
    ):
        return "UNDERLYING_SOURCE_PROVENANCE_UNVERIFIED", None
    observations = {}
    for role in ("spot", "futures"):
        row = evidence.get(role) or {}
        if not row:
            observations[role] = None
            continue
        try:
            observed = _time(row["source_observed_at"])
            if (
                row.get("independent_source_verified") is not True
                or observed > decision
                or decision - observed > timedelta(seconds=120)
                or not str(row.get("source", "")).startswith(
                    ("NSE_SPOT", "NSE_FUTURES")
                )
            ):
                return "UNDERLYING_SOURCE_STALE_OR_UNVERIFIED", None
            if role == "spot" and not row["source"].startswith("NSE_SPOT"):
                return "UNDERLYING_ROLE_MISMATCH", None
            if role == "futures" and not row["source"].startswith("NSE_FUTURES"):
                return "UNDERLYING_ROLE_MISMATCH", None
            observations[role] = UnderlyingDirectionObservation(
                bias=row["bias"],
                confidence=float(row["confidence"]),
                age_seconds=(decision - observed).total_seconds(),
                source=row["source"],
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return "UNDERLYING_OBSERVATION_INVALID", None
    result = arbitrate_underlying_direction(
        observations["spot"], observations["futures"]
    )
    if result.conflict:
        return "FRESH_SPOT_FUTURES_CONFLICT", None
    if result.executable_bias is None:
        return "NO_FRESH_UNDERLYING_DIRECTION", None
    if result.executable_bias != event["side"]:
        return "UNDERLYING_SIDE_MISMATCH", result.executable_bias
    return "UNDERLYING_DIRECTION_ALIGNED", result.executable_bias


def _book(
    row: Mapping[str, Any], contract: Mapping[str, Any]
) -> ObservedBook:
    """Reject malformed depth rather than backfill it from option OHLC."""
    if not _sha(row.get("source_sha256")):
        raise ValueError("pit_depth_source_unverified")
    if any(
        type(level[1]) is not int
        for side in ("bid", "ask")
        for level in row[side]
    ):
        raise ValueError("pit_depth_quantity_unverified")
    return ObservedBook(
        symbol=str(row["symbol"]),
        token=str(row["token"]),
        expiry=date.fromisoformat(row["expiry"]),
        lot_size=row["lot_size"],
        observed_at=_time(row["observed_at"]),
        bid=tuple(BookLevel(float(p), q) for p, q in row["bid"]),
        ask=tuple(BookLevel(float(p), q) for p, q in row["ask"]),
        source=row["source"],
        full_depth_captured=(row.get("full_depth_captured") is True),
        historical_identity_verified=(
            row.get("independent_source_verified") is True
            and row["symbol"] == contract["symbol"]
            and str(row["token"]) == str(contract["token"])
        ),
        clock_convention_verified=(row.get("clock_verified") is True),
    )


def _event_status(
    event: Mapping[str, Any], proof: Mapping[str, Any]
) -> dict[str, Any]:
    decision = _time(event["earliest_entry"])
    event_bar_start = _time(event["event_bar_start"])
    if decision != event_bar_start + timedelta(minutes=1):
        raise ValueError("pit_event_decision_clock_invalid")
    contract_status, contract = _contract(
        event, proof.get("contract") or {}, decision
    )
    direction_status, direction = _direction(
        event, proof.get("direction") or {}, decision
    )
    quote_rows = proof.get("entry_books") or []
    if not isinstance(quote_rows, list):
        raise ValueError("pit_book_payload_invalid")
    entry_status = "NOT_ATTEMPTED_MISSING_PROOF"
    entry_price = None
    entry_fee = None
    if contract and direction_status == "UNDERLYING_DIRECTION_ALIGNED":
        signal = proof.get("strategy_intent") or {}
        if (
            signal.get("verified") is not True
            or signal.get("symbol") != event["symbol"]
            or signal.get("side") != event["side"]
            or signal.get("strategy") not in {"SMC", "FailedBreakoutResearch"}
            or not _sha(signal.get("source_sha256"))
            or "emitted_at" not in signal
            or not _time(signal["emitted_at"]) == decision
        ):
            entry_status = "VERIFIED_STRATEGY_INTENT_MISSING"
        elif not quote_rows:
            entry_status = "HISTORICAL_FIVE_LEVEL_QUOTES_MISSING"
        else:
            fee = proof.get("historical_exchange_rate") or {}
            if (
                fee.get("independent_source_verified") is not True
                or not _sha(fee.get("source_sha256"))
            ):
                entry_status = "HISTORICAL_EXCHANGE_FEE_UNVERIFIED"
            else:
                try:
                    quotes = [_book(row, contract) for row in quote_rows]
                    intent = OrderIntent(
                        symbol=contract["symbol"],
                        token=str(contract["token"]),
                        expiry=date.fromisoformat(contract["expiry"]),
                        lot_size=contract["lot_size"],
                        decision_at=decision,
                        side="BUY",
                        quantity=contract["lot_size"],
                        latency_ms=100,
                    )
                    result = replay_market_order(
                        intent,
                        quotes,
                        historical_exchange_rate=float(fee["rate"]),
                    )
                    entry_status = result.reason
                    if result.status == "MODELED_DEPTH":
                        entry_price, entry_fee = result.price, result.fees
                except (KeyError, TypeError, ValueError, OverflowError):
                    entry_status = "HISTORICAL_QUOTE_OR_FEE_INVALID"
    blockers = [
        status
        for status in (contract_status, direction_status, entry_status)
        if status not in {
            "POINT_IN_TIME_CONTRACT_ASSERTED",
            "UNDERLYING_DIRECTION_ALIGNED",
            "observed_depth_model_only",
        }
    ]
    return {
        "event_id": event["event_id"],
        "phase": event["phase"],
        "symbol": event["symbol"],
        "side": event["side"],
        "contract_status": contract_status,
        "underlying_status": direction_status,
        "underlying_bias": direction,
        "entry_status": entry_status,
        "modeled_entry_price": entry_price,
        "modeled_entry_fee": entry_fee,
        "blockers": blockers,
        "broker_fill_verified": False,
        "exit_cost_ledger_verified": False,
        "net_expectancy_verified": False,
        "live_eligible": False,
    }


def audit(
    pilot: Mapping[str, Any], evidence: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Classify each *same* archived event; never synthesize missing quotes."""
    if os.getenv("EXECUTION_MODE", "").upper() != "SHADOW":
        raise ValueError("pit_audit_requires_shadow")
    if pilot.get("promotion_eligible") is not False:
        raise ValueError("pit_pilot_must_be_nonpromotable")
    events = pilot.get("matched_reversal_events")
    if not isinstance(events, list):
        raise ValueError("pit_event_level_roster_missing")
    proof_map = (evidence or {}).get("events") or {}
    if not isinstance(proof_map, Mapping):
        raise ValueError("pit_proof_payload_invalid")
    rows = []
    seen = set()
    for event in events:
        event_id = event["event_id"]
        if not _sha(event_id) or event_id in seen:
            raise ValueError("pit_event_identity_invalid")
        seen.add(event_id)
        rows.append(_event_status(event, proof_map.get(event_id) or {}))
    return {
        "scope": "point_in_time_directional_observed_book_gate",
        "original_manifest_sha256": pilot.get("manifest_sha256"),
        "session_archive_digests": pilot.get("session_archive_digests"),
        "matched_event_count": len(rows),
        "results": rows,
        "contract_status": dict(Counter(row["contract_status"] for row in rows)),
        "underlying_status": dict(Counter(row["underlying_status"] for row in rows)),
        "entry_status": dict(Counter(row["entry_status"] for row in rows)),
        "complete_modeled_round_trip_count": 0,
        "actual_broker_fill_count": 0,
        "live_promotion_eligible": False,
        "execution_grade_ready": False,
        "conclusion": "ENTRY_FEASIBILITY_AUDIT_NOT_HISTORICAL_EXECUTABLE_PNL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing audit")
    report = json.loads(args.pilot.read_text())
    proof = json.loads(args.evidence.read_text()) if args.evidence else {}
    result = audit(report, proof)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "events": result["matched_event_count"],
                "contract_status": result["contract_status"],
                "underlying_status": result["underlying_status"],
                "entry_status": result["entry_status"],
                "live_promotion_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
