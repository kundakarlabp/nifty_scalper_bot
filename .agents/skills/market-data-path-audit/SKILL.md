---
name: market-data-path-audit
description: Audit the NIFTY scalper market-data path end to end when investigating stale quotes, missing depth, LTP-only degradation, subscription gaps, timestamp/freshness defects, reconnects, or downstream loss of normalized WebSocket FULL data.
---

# Market-Data Path Audit

Use this skill for data-path integrity problems. Read `AGENTS.md`, `docs/REPO_MAP.md`, and the current market-data implementation before proposing changes.

## Core invariant

WebSocket FULL data is the canonical live source. Once a fresh normalized quote exists, downstream code must not silently replace it with older, lower-quality, LTP-only, polling, synthetic, or partially normalized data.

Preserve the canonical fields required by the repository:

`symbol, token, timestamp, timestamp_ms, bid, ask, spread, depth, OI, source, freshness, stale state, tradable_quote`.

## Canonical trace

Trace one real symbol/token through the actual runtime path:

```text
broker/KiteTicker payload
→ streaming/websocket_manager.py
→ normalized quote/tick
→ data/market_data_manager.py
→ data/data_hub.py
→ strategies/runner.py and strategy consumers
→ readiness/risk/execution consumers
→ execution/bracket_manager.py where current quote/depth is required
```

Do not infer correctness from an upstream object alone. Prove the required fields and freshness semantics survive every boundary that consumes them.

## Audit procedure

1. **Choose one concrete event**
   - exact symbol and token
   - exchange timestamp and receive timestamp
   - source/mode
   - full bid/ask/depth/OI payload where available

2. **Verify normalization**
   - token/symbol identity remains stable
   - units/timezone are explicit
   - bid/ask/spread/depth are not discarded
   - source quality is labeled
   - normalization occurs once at the correct owner rather than repeatedly downstream

3. **Verify freshness ordering**
   - compare timestamps in compatible units
   - older fallback data cannot overwrite fresher WebSocket state
   - stale state is explicit rather than inferred differently by each consumer
   - a stale or incomplete quote cannot become tradable merely because an LTP exists

4. **Verify subscription lifecycle**
   - active option basket is subscribed
   - ATM/expiry rotation removes/adds the correct tokens
   - reconnect restores required subscriptions
   - duplicate subscription events remain harmless
   - required symbols become observable as hydrated or specifically blocked

5. **Verify fallback behavior**
   - fallback activates only for a real degradation condition
   - fallback source is explicit
   - recovery back to WebSocket FULL data is deterministic
   - fallback never masquerades as FULL depth
   - polling does not become the default live path by accident

6. **Verify downstream consumers**
   - DataHub remains a read facade, not a second market-data owner
   - strategies receive prepared data rather than fetching broker data
   - readiness uses the canonical freshness/quality state
   - execution and bracket logic consume the same authoritative current quote contract where required

## Golden replay

For a market-data defect that can be represented safely, extend or derive a minimal case from `tests/fixtures/replay/golden_market_path.csv` and run it through the existing `ReplayHarness`. Keep the fixture sanitized, deterministic, and broker-free. Do not create a second replay engine.

The documented tick contract is `docs/contracts/canonical_market_tick.schema.json`; use it to check stable field meaning while the live implementation remains authoritative.

## Regression matrix

Add the smallest tests needed for the defect, selecting from:

- FULL quote survives end to end
- LTP-only quote is explicitly degraded
- stale polling cannot replace fresher WS FULL
- out-of-order tick cannot move freshness backward
- reconnect restores subscriptions and depth
- ATM/expiry rotation updates symbol/token identity
- missing bid/ask/depth produces the expected blocker
- duplicate tick/update remains idempotent
- no HTTP/REST fetch is introduced into the strategy evaluation loop

Prefer fixture replay or public-interface tests over private-state assertions.

## Report

Return:

```markdown
Observed defect:
First incorrect boundary:
Authoritative owner:
Fields preserved/lost:
Freshness/source result:
Fallback result:
Smallest correction:
Regression proving it:
Residual uncertainty:
```

Do not redesign the market-data stack merely because a local conversion is awkward. Fix the earliest incorrect owner or boundary.
