"""Read-only evidence-quality audit for canonical trade-ledger rows.

This report does not reconcile against broker orders, balances or contract notes;
non-terminal historical rows are never interpreted as current open positions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

_FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def summarize_ledger_integrity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Count provenance gaps without inventing missing fills or charges."""
    states: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    complete_count = 0
    arithmetic_mismatch = 0
    arithmetic_unverifiable = 0
    closed_incomplete = 0
    incomplete_ids: list[str] = []
    for row in rows:
        state = str(row.get("state") or "UNKNOWN").upper()
        states[state] += 1
        complete = state == "CLOSED" and row.get("ledger_complete") in (1, True)
        if not complete:
            if state == "CLOSED":
                closed_incomplete += 1
            elif len(incomplete_ids) < 20:
                incomplete_ids.append(str(row.get("trade_id") or ""))
            continue
        complete_count += 1
        for field in (
            "entry_filled_at",
            "exit_filled_at",
            "closed_at",
            "entry_order_id",
            "exit_order_id",
            "entry_price",
            "exit_price",
        ):
            if row.get(field) is None:
                missing[field] += 1
        sha = str(row.get("build_sha") or "").strip()
        if not _FULL_SHA.fullmatch(sha):
            missing["valid_build_sha"] += 1

        outcome = _mapping(row.get("outcome"))
        source = str(outcome.get("cost_source") or "unattributed").strip()
        sources[source] += 1
        # estimated_costs is a separate model estimate. When present,
        # effective_costs.total is the actual value used to derive net P&L;
        # do NOT compare the broker-derived net to the model estimate.
        effective = _mapping(outcome.get("effective_costs"))
        used_cost = _finite(effective.get("total"))
        if used_cost is None:
            used_cost = _finite(row.get("estimated_costs"))
        gross = _finite(row.get("gross_pnl"))
        net = _finite(row.get("net_pnl"))
        if gross is None or net is None or used_cost is None:
            arithmetic_unverifiable += 1
        elif abs((gross - used_cost) - net) > 0.05:
            arithmetic_mismatch += 1

    return {
        "total_ledger_rows": len(rows),
        "state_counts": dict(sorted(states.items())),
        "completed_closed_rows": complete_count,
        "closed_incomplete_rows": closed_incomplete,
        "nonclosed_historical_rows": len(rows)
        - complete_count
        - closed_incomplete,
        "completed_missing": dict(sorted(missing.items())),
        "completed_cost_source_counts": dict(sorted(sources.items())),
        "completed_cost_arithmetic_mismatch": arithmetic_mismatch,
        "completed_cost_arithmetic_unverifiable": arithmetic_unverifiable,
        "nonclosed_sample_ids": incomplete_ids,
        "broker_reconciled": False,
        "live_open_positions_inferred": False,
        "pnl_attributed_to_deployment": False,
        "interpretation": (
            "historical_event_ledger_only; broker_order_fill_and_contract_note_"
            "reconciliation_required_before_release_attribution"
        ),
    }
