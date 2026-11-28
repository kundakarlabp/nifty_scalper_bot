"""Read-only, spread-aware forward observations for canonical setup decisions.

Responsibilities: Deduplicate setup identities; compare observed option
ask/bid quotes and load selected, exact-token FULL-depth option books and
futures WS price paths for the canonical session-excursion research report.
load_excursion_quote_paths also returns causal, scope-specific contract lots.
Optionally reuse validated chronological quote-time indexes for large sessions.
Boundaries: No broker calls, strategy overrides, or new setup identity owner.
Safety: Missing/rotated contracts, untradable quotes, and archive gaps are
unlabelled; markouts are not proof of fills or post-cost profitability.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def _positive(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def observed_markout(
    ticks: Sequence[tuple[float, float, float]],
    *,
    decision_ts: float,
    horizon_s: int,
    max_entry_age_s: float = 3.0,
    terminal_tolerance_s: float = 5.0,
    min_forward_quotes: int = 3,
    timestamps: Sequence[float] | None = None,
) -> tuple[dict[str, float | int] | None, str]:
    """Use only a known prior ask and subsequent bid quotes; never future entry."""
    if not math.isfinite(decision_ts) or horizon_s <= 0:
        raise ValueError("invalid_markout_window")
    if not ticks:
        return None, "no_tradable_exact_contract_quotes"
    if timestamps is not None and len(timestamps) != len(ticks):
        raise ValueError("invalid_quote_timestamp_index")
    chronology = timestamps if timestamps is not None else [tick[0] for tick in ticks]
    previous = bisect.bisect_right(chronology, decision_ts) - 1
    if previous < 0 or decision_ts - chronology[previous] > max_entry_age_s:
        return None, "no_point_in_time_entry_ask"
    entry_ask = ticks[previous][2]
    left = bisect.bisect_right(chronology, decision_ts)
    right = bisect.bisect_right(chronology, decision_ts + horizon_s)
    forward = ticks[left:right]
    if len(forward) < min_forward_quotes:
        return None, "insufficient_forward_quotes"
    terminal_at, terminal_bid, _ask = forward[-1]
    if decision_ts + horizon_s - terminal_at > terminal_tolerance_s:
        return None, "stale_terminal_bid"
    high_bid = max(row[1] for row in forward)
    low_bid = min(row[1] for row in forward)
    return (
        {
            "entry_ask": round(entry_ask, 4),
            "terminal_bid": round(terminal_bid, 4),
            "gross_spread_adjusted_markout_pct": round(
                100 * (terminal_bid / entry_ask - 1), 5
            ),
            "mfe_markout_pct": round(100 * (high_bid / entry_ask - 1), 5),
            "mae_markout_pct": round(100 * (low_bid / entry_ask - 1), 5),
            "forward_quote_count": len(forward),
            "entry_quote_age_seconds": round(decision_ts - chronology[previous], 3),
            "terminal_quote_age_seconds": round(
                decision_ts + horizon_s - terminal_at, 3
            ),
        },
        "observed_bid_path_not_broker_fill",
    )


def load_archive_quotes(
    path: Path,
) -> tuple[dict[str, list[tuple[float, float, float]]], str, int]:
    """Reject incomplete archives and exclude degraded/non-current WS quotes."""
    from nifty_scalper_bot.storage.replay_archive import (
        archive_fingerprint,
        iter_session,
    )
    from nifty_scalper_bot.utils.symbols import normalize_symbol

    quotes: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    count = 0
    for event in iter_session(path):
        if event["kind"] != "tick":
            continue
        tick = event["payload"]
        if not isinstance(tick, Mapping):
            continue
        # Normalized LIVE full-depth WebSocket ticks are tagged "ws_full".
        # Reject REST/poll data, but do not discard the actual executable tape.
        if str(tick.get("source") or "").lower() not in {"ws", "ws_full"}:
            continue
        if tick.get("depth_available") is not True:
            continue
        if tick.get("tradable_quote") is not True:
            continue
        symbol = normalize_symbol(str(tick.get("symbol") or ""))
        if not symbol.startswith("NFO:") or not symbol.endswith(("CE", "PE")):
            continue
        bid = _positive(tick.get("bid"))
        ask = _positive(tick.get("ask"))
        if bid is None or ask is None or bid > ask:
            continue
        at = datetime.fromisoformat(event["available_at"]).timestamp()
        if not math.isfinite(at):
            continue
        quotes[symbol].append((at, bid, ask))
        count += 1
    for path_quotes in quotes.values():
        path_quotes.sort(key=lambda tick: tick[0])
    return dict(quotes), archive_fingerprint(path), count


def load_excursion_quote_paths(
    path: Path,
) -> tuple[
    dict[str, list[tuple[float, float, str]]],
    dict[str, list[tuple[float, float, float, int, int, str]]],
    dict[tuple[str, str], int],
]:
    """Observe selected futures, exact-token FULL books and causal catalog lots.

    The ReplayArchive remains the sole capture owner. Missing token, selected
    contract, full-depth, or best-level quantity does not become a fill.
    """
    from nifty_scalper_bot.instruments.active_contracts import (
        canonical_nifty_future_symbol,
    )
    from nifty_scalper_bot.storage.replay_archive import iter_session
    from nifty_scalper_bot.utils.symbols import normalize_symbol

    futures: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    books: dict[str, list[tuple[float, float, float, int, int, str]]] = defaultdict(
        list
    )
    selected: dict[str, str] = {}
    tokens: dict[str, int] = {}
    scope_lots: dict[tuple[str, str], int] = {}
    phase = 0
    previous_phase_key: tuple[str, ...] | None = None
    for event in iter_session(path):
        payload = event["payload"]
        if event["kind"] == "snapshot":
            basket = payload.get("basket")
            if not isinstance(basket, Mapping):
                selected = {}
                previous_phase_key = None
                continue
            selected = {
                side: normalize_symbol(
                    str(basket.get(f"selected_{side.lower()}") or "")
                )
                for side in ("CE", "PE")
            }
            selected["FUT"] = normalize_symbol(str(basket.get("futures_symbol") or ""))
            # Each snapshot owns its own identity evidence; never carry stale
            # catalog rows across a restart or basket replacement.
            tokens.clear()
            catalog_lots: dict[str, tuple[int, int]] = {}
            conflicting_lots: set[str] = set()
            catalog = payload.get("instruments")
            if isinstance(catalog, list):
                for item in catalog:
                    if not isinstance(item, Mapping):
                        continue
                    symbol = normalize_symbol(
                        f"{item.get('exchange')}:{item.get('tradingsymbol')}"
                    )
                    token = item.get("instrument_token")
                    if type(token) is int and token > 0:
                        tokens[symbol] = token
                        lot = item.get("lot_size")
                        if (
                            type(lot) is int
                            and 0 < lot < 10000
                            and str(item.get("name") or "").upper() == "NIFTY"
                            and str(item.get("instrument_type") or "").upper()
                            == symbol[-2:]
                            and symbol in (selected["CE"], selected["PE"])
                        ):
                            previous_lot = catalog_lots.get(symbol)
                            evidence = (token, lot)
                            if previous_lot is not None and previous_lot != evidence:
                                conflicting_lots.add(symbol)
                            catalog_lots[symbol] = evidence
            mapping = basket.get("token_by_symbol")
            if isinstance(mapping, Mapping):
                for raw_symbol, token in mapping.items():
                    if type(token) is int and token > 0:
                        tokens[normalize_symbol(str(raw_symbol))] = token
            selected_lots = {
                symbol: evidence[1]
                for symbol, evidence in catalog_lots.items()
                if symbol not in conflicting_lots and evidence[0] == tokens.get(symbol)
            }
            phase_key = (
                str(event["run_id"]),
                selected.get("CE", ""),
                selected.get("PE", ""),
                selected.get("FUT", ""),
                *(
                    str(tokens.get(selected.get(k, "")) or "")
                    for k in ("CE", "PE", "FUT")
                ),
                *(str(selected_lots.get(selected[k]) or "") for k in ("CE", "PE")),
            )
            if phase_key != previous_phase_key:
                phase += 1
                previous_phase_key = phase_key
            scope = f"{event['run_id']}:{phase}"
            for symbol, lot in selected_lots.items():
                scope_lots[(scope, symbol)] = lot
            continue
        if str(payload.get("source") or "").lower() not in {"ws", "ws_full"}:
            continue
        symbol = normalize_symbol(str(payload.get("symbol") or ""))
        scope = f"{event['run_id']}:{phase}"
        at = datetime.fromisoformat(event["available_at"]).timestamp()
        if not math.isfinite(at):
            continue
        if canonical_nifty_future_symbol(symbol) is not None and symbol == selected.get(
            "FUT"
        ):
            price = _positive(payload.get("ltp") or payload.get("last_price"))
            if price is not None and type(payload.get("instrument_token")) is int:
                if payload["instrument_token"] == tokens.get(symbol):
                    futures[symbol].append((at, price, scope))
            continue
        side = symbol[-2:]
        if (
            side not in {"CE", "PE"}
            or symbol != selected.get(side)
            or str(payload.get("source") or "").lower() != "ws_full"
            or payload.get("tradable_quote") is not True
            or payload.get("depth_available") is not True
            or type(payload.get("instrument_token")) is not int
            or payload["instrument_token"] != tokens.get(symbol)
        ):
            continue
        depth = payload.get("depth")
        if not isinstance(depth, Mapping):
            continue
        buy = depth.get("buy")
        sell = depth.get("sell")
        if not isinstance(buy, list) or not isinstance(sell, list):
            continue
        if (
            not buy
            or not sell
            or not isinstance(buy[0], Mapping)
            or not isinstance(sell[0], Mapping)
        ):
            continue
        bid = _positive(buy[0].get("price"))
        ask = _positive(sell[0].get("price"))
        bid_qty = buy[0].get("quantity")
        ask_qty = sell[0].get("quantity")
        top_bid = _positive(payload.get("bid"))
        top_ask = _positive(payload.get("ask"))
        if (
            bid is None
            or ask is None
            or bid >= ask
            or type(bid_qty) is not int
            or type(ask_qty) is not int
            or bid_qty <= 0
            or ask_qty <= 0
            or top_bid is None
            or top_ask is None
            or not math.isclose(bid, top_bid, abs_tol=1e-6)
            or not math.isclose(ask, top_ask, abs_tol=1e-6)
        ):
            continue
        books[symbol].append((at, bid, ask, bid_qty, ask_qty, scope))
    for rows in futures.values():
        rows.sort(key=lambda row: row[0])
    for rows in books.values():
        rows.sort(key=lambda row: row[0])
    return dict(futures), dict(books), scope_lots


def observed_futures_markout(
    ticks: Sequence[tuple[float, float]],
    *,
    decision_ts: float,
    horizon_s: int,
    side: str,
) -> tuple[dict[str, float | int] | None, str]:
    """Require a dense same-futures, causal observed WS path to the horizon."""
    if side not in {"CE", "PE"} or horizon_s <= 0 or not math.isfinite(decision_ts):
        raise ValueError("invalid_futures_markout_request")
    if not ticks:
        return None, "futures_ws_path_unavailable"
    times = [row[0] for row in ticks]
    prior = bisect.bisect_right(times, decision_ts) - 1
    end = bisect.bisect_right(times, decision_ts + horizon_s)
    if prior < 0 or not 0 <= decision_ts - times[prior] <= 3:
        return None, "futures_entry_price_unavailable"
    path = ticks[prior:end]
    if len(path) < 4:
        return None, "futures_forward_path_sparse"
    if decision_ts + horizon_s - path[-1][0] > 5:
        return None, "futures_terminal_price_stale"
    if any(b[0] - a[0] > 10 or b[0] < a[0] for a, b in zip(path, path[1:])):
        return None, "futures_forward_path_gapped"
    initial = path[0][1]
    direction = 1 if side == "CE" else -1
    moves = [(price - initial) * direction for _, price in path[1:]]
    return {
        "entry_futures_price": round(initial, 4),
        "terminal_futures_price": round(path[-1][1], 4),
        "signed_terminal_points": round(moves[-1], 4),
        "favorable_points": round(max(0.0, *moves), 4),
        "adverse_points": round(min(0.0, *moves), 4),
        "quote_count": len(path),
    }, "observed_futures_ws_not_a_trade"


def evaluate_opportunities(
    opportunities: Sequence[Any],
    quotes: Mapping[str, Sequence[tuple[float, float, float]]],
    horizons: Sequence[int],
) -> list[dict[str, Any]]:
    """Preserve every independent decision including all unlabelled reasons."""
    from nifty_scalper_bot.utils.symbols import normalize_symbol

    results = []
    for opportunity in opportunities:
        contract = normalize_symbol(opportunity.symbol)
        decision_ts = getattr(opportunity, "decision_ts", opportunity.first_seen_ts)
        same_contract = (
            len(opportunity.observed_symbols) == 1
            and normalize_symbol(opportunity.observed_symbols[0]) == contract
            and contract.startswith("NFO:")
            and contract.endswith(("CE", "PE"))
        )
        for horizon in horizons:
            markout, status = (
                observed_markout(
                    quotes.get(contract, ()),
                    decision_ts=decision_ts,
                    horizon_s=horizon,
                )
                if same_contract
                else (None, "contract_rotation_or_missing_identity")
            )
            results.append(
                {
                    "opportunity_id": opportunity.opportunity_id,
                    "strategy": opportunity.strategy,
                    "side": opportunity.side,
                    "contract": contract,
                    "approved": opportunity.approved,
                    "reason": opportunity.final_reason,
                    "blocked_reasons": list(opportunity.blocked_reasons),
                    "first_seen_timestamp": opportunity.first_seen_ts,
                    "decision_timestamp": decision_ts,
                    "horizon_seconds": horizon,
                    "status": status,
                    "markout": markout,
                }
            )
    return results


def summarize_groups(
    rows: Sequence[Mapping[str, Any]], *, min_labelled: int
) -> list[dict[str, Any]]:
    """Report independent denominators and unlabelled evidence explicitly."""
    if min_labelled < 1:
        raise ValueError("min_labelled must be positive")
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        group = "approved" if row["approved"] else "blocked"
        groups[(group, str(row["reason"]), int(row["horizon_seconds"]))].append(row)
    result = []
    for (group, reason, horizon), observations in sorted(groups.items()):
        valid = [
            float(row["markout"]["gross_spread_adjusted_markout_pct"])
            for row in observations
            if isinstance(row.get("markout"), Mapping)
        ]
        sufficient = len(valid) >= min_labelled
        unlabelled_statuses = Counter(
            str(row.get("status") or "missing_status")[:64]
            for row in observations
            if not isinstance(row.get("markout"), Mapping)
        )
        result.append(
            {
                "group": group,
                "reason": reason,
                "horizon_seconds": horizon,
                "independent_setups": len(observations),
                "labelled": len(valid),
                "unlabelled": len(observations) - len(valid),
                "unlabelled_status_counts": dict(sorted(unlabelled_statuses.items())),
                "label_coverage_fraction": round(len(valid) / len(observations), 4),
                "mean_gross_markout_pct": (
                    round(statistics.mean(valid), 5) if sufficient else None
                ),
                "median_gross_markout_pct": (
                    round(statistics.median(valid), 5) if sufficient else None
                ),
                "minimum_labelled": min_labelled,
                "causal_or_profitability_inference_ready": False,
            }
        )
    return result


def main(argv: list[str] | None = None) -> int:
    """Perform offline analysis on explicitly supplied, immutable local paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades-db", required=True, type=Path)
    parser.add_argument("--replay-session", required=True, type=Path)
    parser.add_argument("--horizon-s", type=int, action="append")
    parser.add_argument("--min-labelled", type=int, default=10)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    horizons = sorted(set(args.horizon_s or (60, 180, 300)))
    if any(horizon <= 0 for horizon in horizons):
        parser.error("--horizon-s values must be positive")
    if args.min_labelled < 1:
        parser.error("--min-labelled must be positive")

    from nifty_scalper_bot.backtesting.setup_opportunity_analysis import (
        canonicalize_setup_opportunities,
    )
    from nifty_scalper_bot.journal.trade_ledger import load_candidate_decision_rows

    decision_rows = load_candidate_decision_rows(args.trades_db)
    opportunities = canonicalize_setup_opportunities(decision_rows)
    quotes, fingerprint, quote_count = load_archive_quotes(args.replay_session)
    outcomes = evaluate_opportunities(opportunities, quotes, horizons)
    covered = sum(opportunity.decision_count for opportunity in opportunities)
    report = {
        "schema_version": 1,
        "analysis": "observational_ask_to_bid_not_trade_replay",
        "replay_sha256": fingerprint,
        "complete_replay_verified": True,
        "tradeable_ws_quotes": quote_count,
        "decision_rows": len(decision_rows),
        "decisions_with_setup_identity": covered,
        "setup_identity_coverage_fraction": (
            round(covered / len(decision_rows), 4) if decision_rows else 0.0
        ),
        "independent_setups": len(opportunities),
        "independent_approved": sum(op.approved for op in opportunities),
        "independent_blocked": sum(not op.approved for op in opportunities),
        "costs_and_broker_fills_covered": False,
        "strategy_promotion_supported": False,
        "strata": summarize_groups(outcomes, min_labelled=args.min_labelled),
        "outcomes": outcomes,
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Analysis saved to {args.out}; "
        f"setups={len(opportunities)} "
        f"labelled={sum(row['markout'] is not None for row in outcomes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
