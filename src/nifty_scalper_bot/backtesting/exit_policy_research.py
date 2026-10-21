"""Read-only, causal exit-overlay research on captured exchange quote observations.

The current live entry and protective-exit code are never imported or modified.
These single-position overlays are NOT broker-fill parity: marketable sizes,
latency, queue priority and differing transaction taxes remain unverified.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from nifty_scalper_bot.storage.replay_archive import iter_session

_POLICIES = (
    "target_0.6r",
    "target_0.8r",
    "target_1.0r",
    "lock_40pct",
    "lock_50pct",
    "lock_60pct",
)


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("exit_replay_timestamp_missing")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("exit_replay_timezone_missing")
    return stamp.astimezone(timezone.utc)


def _usable_bid(row: Mapping[str, Any], symbol: str, quantity: int) -> float | None:
    if row.get("kind") != "tick":
        return None
    payload = row.get("payload")
    if not isinstance(payload, Mapping) or payload.get("symbol") != symbol:
        return None
    depth = payload.get("depth")
    if not isinstance(depth, Mapping):
        return None
    buys = depth.get("buy")
    if not isinstance(buys, list) or not buys or not isinstance(buys[0], Mapping):
        return None
    try:
        bid = float(buys[0]["price"])
        visible_quantity = int(buys[0]["quantity"])
    except (KeyError, ValueError, TypeError):
        return None
    if not math.isfinite(bid) or bid <= 0 or visible_quantity < quantity:
        return None
    return bid


def compare_trade(
    trade: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
    *,
    slippage_points: float = 0.5,
    max_quote_gap_seconds: float = 10.0,
) -> dict[str, Any]:
    """Evaluate stop/target rules in observation order, bounded by actual exit.

    Target variants inherit the verified actual exit if they did not hit a TP.
    Trailing variants without an observed fill are UNRESOLVED, not assigned
    the observed exit. This avoids price-invention beyond the recorded position.
    """
    if not math.isfinite(slippage_points) or slippage_points < 0:
        raise ValueError("exit_replay_invalid_slippage")
    try:
        if trade.get("side") != "BUY" or not trade.get("ledger_complete"):
            raise ValueError("exit_replay_incomplete_trade")
        symbol = str(trade["symbol"])
        entry = float(trade["entry_price"])
        initial_sl = float(trade["initial_stop"])
        qty = int(trade["quantity"])
        net = float(trade["net_pnl"])
        observed_costs = trade.get("effective_costs")
        if observed_costs is None:
            outcome = trade.get("outcome")
            effective = (
                outcome.get("effective_costs") if isinstance(outcome, Mapping) else None
            )
            observed_costs = (
                effective.get("total")
                if isinstance(effective, Mapping)
                else trade.get("estimated_costs")
            )
        costs = float(observed_costs)
        entry_at = _timestamp(trade["entry_filled_at"])
        exit_at = _timestamp(trade["exit_filled_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("exit_replay_invalid_trade_geometry") from exc
    if not (
        all(math.isfinite(x) for x in (entry, initial_sl, net, costs))
        and 0 < initial_sl < entry
        and qty > 0
        and costs >= 0
        and exit_at > entry_at
    ):
        raise ValueError("exit_replay_invalid_trade_geometry")

    bids: list[tuple[datetime, float]] = []
    for row in events:
        if row.get("kind") != "tick":
            continue
        bid = _usable_bid(row, symbol, qty)
        if bid is None:
            continue
        at = _timestamp(row.get("available_at"))
        if entry_at < at <= exit_at:
            bids.append((at, bid))

    # Strict data-quality gate: short gaps alone do not prove fill feasibility,
    # but holes invalidate even the modeled quote-level overlay.
    # Preserve observation sequence; sorting by timestamp would use a later
    # delivery retroactively and create causal leakage.
    gaps = [(bids[i][0] - bids[i - 1][0]).total_seconds() for i in range(1, len(bids))]
    valid_coverage = (
        len(bids) >= 3
        and (bids[0][0] - entry_at).total_seconds() <= max_quote_gap_seconds
        and (exit_at - bids[-1][0]).total_seconds() <= max_quote_gap_seconds
        and all(0 < gap <= max_quote_gap_seconds for gap in gaps)
    )
    result: dict[str, Any] = {
        "trade_id": str(trade.get("trade_id") or ""),
        "symbol": symbol,
        "observed_net": round(net, 2),
        "bid_observations": len(bids),
        "coverage_ready": bool(valid_coverage),
        "evidence": "RESEARCH_CANDIDATE_MODELED_QUOTES_NOT_BROKER_FILLS",
        "outcomes": {},
    }
    if not valid_coverage:
        result["outcomes"] = {
            key: {
                "status": "UNRESOLVED",
                "reason": "insufficient_executable_bid_coverage",
            }
            for key in _POLICIES
        }
        return result

    risk = entry - initial_sl
    for policy in _POLICIES:
        is_target = policy.startswith("target_")
        target_r = float(policy.split("_")[1][:-1]) if is_target else None
        lock_pct = int(policy.split("_")[1][:-3]) / 100.0 if not is_target else None
        target = entry + risk * target_r if target_r is not None else None
        watermark = entry
        stop = initial_sl
        simulated = None
        for at, bid in bids:
            # The working stop was set on an earlier observation. Check it
            # BEFORE advancing the current tick's favorable watermark.
            if not is_target and bid <= stop:
                # Exchange SL-LIMIT can remain unfilled through a large gap;
                # conservatively mark this alternative as unresolved.
                if bid < stop * 0.95:
                    simulated = {
                        "status": "UNRESOLVED",
                        "reason": "stop_limit_gap_risk",
                    }
                else:
                    exit_price = bid - slippage_points
                    simulated = {
                        "status": "MODELED_STOP",
                        "at": at.isoformat(),
                        "exit_price": round(exit_price, 2),
                        "net": round((exit_price - entry) * qty - costs, 2),
                    }
                break
            if is_target and target is not None and bid >= target:
                # Target limit is only modeled if observed full-size bid
                # crosses it; apply execution slippage conservatively.
                exit_price = target - slippage_points
                simulated = {
                    "status": "MODELED_TARGET",
                    "at": at.isoformat(),
                    "exit_price": round(exit_price, 2),
                    "net": round((exit_price - entry) * qty - costs, 2),
                }
                break
            if not is_target and lock_pct is not None:
                watermark = max(watermark, bid)
                # Approximate the canonical tier-1 rule (0.6R activation):
                # recover round-trip fees and lock 0.10R before tier 2.
                cost_per_unit = costs / qty
                if watermark - entry >= 0.6 * risk:
                    floor = entry + cost_per_unit + 0.10 * risk
                    if floor < bid - cost_per_unit:
                        stop = max(stop, floor)
                if watermark - entry >= risk:  # 1R tier-2 lock
                    proposed = entry + lock_pct * (watermark - entry)
                    if proposed < bid - cost_per_unit:
                        stop = max(stop, proposed)

        if simulated is None:
            simulated = (
                {"status": "UNCHANGED_ACTUAL", "net": round(net, 2)}
                if is_target
                else {
                    "status": "UNRESOLVED",
                    "reason": "hypothetical_position_survives_observed_exit",
                }
            )
        result["outcomes"][policy] = simulated
    return result


def run_exit_comparison(
    session: Path,
    trades_json: Path,
    *,
    slippage_points: float = 0.5,
) -> dict[str, Any]:
    """Load checksum-validated one-session observation tape and local trade JSON."""
    raw = json.loads(trades_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("exit_replay_trade_list_required")
    events = list(iter_session(session))
    results = []
    excluded_dates = 0
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError("exit_replay_trade_row_invalid")
        if str(row.get("trading_date") or "") not in {"", session.stem}:
            excluded_dates += 1
            continue
        try:
            compared = compare_trade(row, events, slippage_points=slippage_points)
        except ValueError as exc:
            compared = {
                "trade_id": str(row.get("trade_id") or ""),
                "coverage_ready": False,
                "observed_net": row.get("net_pnl"),
                "outcomes": {
                    policy: {"status": "UNRESOLVED", "reason": str(exc)}
                    for policy in _POLICIES
                },
            }
        results.append(compared)
    scenarios = {}
    for policy in _POLICIES:
        resolved = [
            r["outcomes"][policy]
            for r in results
            if r["outcomes"][policy]["status"]
            in {"MODELED_TARGET", "MODELED_STOP", "UNCHANGED_ACTUAL"}
        ]
        unresolved = len(results) - len(resolved)
        scenarios[policy] = {
            "resolved": len(resolved),
            "unresolved": unresolved,
            "net_sum_if_complete": (
                round(sum(x["net"] for x in resolved), 2)
                if not unresolved and resolved
                else None
            ),
            "promotion_eligible": False,
        }
    return {
        "scope": "single-position-causal-bid-exit-overlay",
        "session": session.name,
        "trade_count": len(results),
        "excluded_other_dates": excluded_dates,
        "slippage_points": slippage_points,
        "scenario_summary": scenarios,
        "trades": results,
        "live_settings_modified": False,
        "promotion_eligible": False,
        "limitations": [
            "exchange queue/fill unverified",
            "fixed actual transaction costs, not re-priced",
            "no trading beyond actual observed exit",
            "requires complete captured bid depth and lot quantity",
            "single-position comparisons do not replay subsequent entries",
            "tier-1 and tier-2 rules are quote-level approximations; "
            "not production parity",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only recorded-bid exit overlays")
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--trades", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--slippage-points", default=0.5, type=float)
    args = parser.parse_args()
    report = run_exit_comparison(
        args.session, args.trades, slippage_points=args.slippage_points
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
