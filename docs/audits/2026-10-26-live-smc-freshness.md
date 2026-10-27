# 2026-10-26 second trade: causal SMC freshness contract

## Verified production evidence (read-only Supabase ledger + market logs)

- Trade: `TRD_ec52bc896669f9ad`, 14:09:29–14:10:31 IST, `NFO:NIFTY26OCT22700CE`, SMC.
- Fill 149.85, initial stop 141.20, exit 141.15; 65 units; net approximately -634.41.
- Approval: `smc_sustained_orderflow_confirmed`. No independent trigger consensus. Futures direction was fresh and bullish; recorded regime was `RANGE`.
- Structural sweep identity was anchored at 14:03:00 IST; entry occurred about **6 minutes 29 seconds** later. Strategy vote was fresh but its underlying structural opportunity was not.
- Manager-enriched BOS at the latest completed bar was the sole logged range-local support; retest and premium reclaim were false. OrderFlow confirmed twice at 14:09:22 and 14:09:27 after a negative reading at 14:09:17.
- Protective SL was exchange-armed; stop filled without a demonstrated trailing malfunction. Max favourable excursion 0.393R, below 0.60R activation.

## Mechanism and changes

1. SMC's existing event-age check uses a latest-completed-bar timestamp, possibly supplemented by a quote timestamp. When the quote timestamp is absent or lagging, the check can admit a live sweep older than the intended five-minute ceiling. A fresh *signal emission* and an up-to-date *option quote* do not reset structural setup age.
2. The **existing** `strategy_live_safety` boundary now compares the original `sweep_timestamp` with the current wall clock for every approved live SMC buy/sell. It fails closed on missing/nonfinite/future/stale proof and limits age to at most 5 minutes (respecting stricter operator settings). It does not alter non-SMC or exit decisions. The guard runs before order candidate approval, after existing strategy/orchestrator filters.
3. In real LIVE mode SMC can no longer count untimestamped locally prepared BOS, CHOCH, retest, or premium reclaim as confirmation. Timestamped manager enrichment must place each event **after its own sweep** and **not after its confirmation bar**. This is not another entry arbiter.
4. Bracket/trade provenance records the original sweep timestamp, regime and support sources for audit/replay. Tests cover boundary ages, invalid/future evidence, bounds, epoch formats and protected non-SMC decisions.

## Research, rollback, safety

- **No alpha claim:** The live modification enforces causal relevance, not a proved net-expectancy improvement. The current `SMC + sustained OFI` path has only two observed trades under today's production configuration.
- RANGE-only admission tightening, stronger OFI persistence, and trailing activation thresholds **require chronology-preserving, executable bid/ask, after-cost out-of-sample comparisons**. Do not tune these parameters to today's one loser alone.
- Mandatory validation: exact GitHub CI at PR head; replay the 14:03–14:10 case to confirm the new block; check candidate count and suppressed true positives; inspect daily trade-event reasons; verify deployed SHA after merge.
- Merging while a live position or protective order exists must not interrupt execution. Automatic deploy should remain broker-flat gated. This code is fail-closed for new entries only; it does not cancel or alter existing stops.

## 27 October follow-up: admission-stage chronology (production archive)

Read-only SQL against the finalized 26 October `nifty_daily_market_logs`
independently extracted manager qualification and broker acceptance markers,
and recovered the original UTC sweep timestamp from each `smcv2` setup ID:

| Contract | Regime in closed ledger | Original sweep IST | Manager qualified IST | Sweep age at manager | Broker accepted IST | Sweep age at broker |
| --- | --- | --- | --- | ---: | --- | ---: |
| 22650CE | TREND | 13:43:00 | 13:45:26 | 146 seconds | 13:45:28 | 148 seconds |
| 22700CE | RANGE | 14:03:00 | 14:09:27 | 387 seconds | 14:09:29 | 389 seconds |

The **manager qualification age** (387 seconds) is distinct from the **entry
fill age** (approximately 389 seconds). A reference rule allowing up to 300
seconds would have rejected the second manager-approved setup while retaining
the first on sweep-age evidence alone. This does **not** prove the first would
still have passed all modern live admission checks, nor that the second would
have lost under every alternative execution policy.

Existing `scripts/reporting/rejected_setup_replay.py` now reports these
historical ages and a three-way five-minute *reference*, not a trading
recommendation. An invalid/future original event, a missing approval marker or
an unparseable setup ID is not counted as a confirmed passing trade. The
chronological proof is anchored to the strategy's own structural setup identity
rather than to a fresh option quote or repeated strategy vote.

The first trade logged regime TREND; the second logged RANGE. Both used
`smc_sustained_orderflow_confirmed` with no independent trigger-strategy
consensus. Only two SMC manager-qualified approvals were observed in 21–27
October logs, and policies/deployed builds changed during that interval.
Do **not** infer that tightening all RANGE admissions improves expected P&L
from these two observations.

The journal does not contain a continuous executable quote/depth path for
rejected trades. Consequently neither a trade-level P&L replay nor statistical
performance comparison across alternative RANGE and OrderFlow rules is
supported. The production five-minute live sweep safeguard was already in
`main` before this research extension; this follow-up changes research tooling
and tests only.

### Reproduction and promotion rule

Run the existing read-only command on a locally exported unmodified journal:

```bash
python scripts/reporting/rejected_setup_replay.py journal_2026-10-26.txt \
  --output smc_admission_replay_2026-10-26.json
```

Review `manager_qualified_5m_reference_rejections`, the per-setup original
age, missing provenance and actual recorded broker lifecycle before making
changes. A new LIVE regime/OFI restriction needs independent time-aligned
bid/ask replay, fees, held-out sessions and drawdown comparison. Never
retroactively tune production thresholds to this one losing trade.
