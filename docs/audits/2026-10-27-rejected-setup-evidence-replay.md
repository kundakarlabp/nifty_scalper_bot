# 27 October 2026 — setup-level logged-evidence replay (research only)

## Decision

**No change to LIVE SMC, VWAP, OrderFlow or safety gates.** The 120-second
per-setup retrospective did not identify a complete, non-adverse two-observation
confirmation sequence. A *naive* variant ignoring any unaligned intervening
observation would produce one extra pair; its interruption had reason
'ofi_3s_not_aligned' and must not be treated as neutral. The conservative
research variant retains directional-reset safety and produces zero extra pairs.

This is an observational journal proxy, not a tick-exact P&L backtest.

## Reproduce

Run:

    python scripts/reporting/rejected_setup_replay.py exported-journald.txt --output setup_replay.json

The tool deduplicates lifecycle-stage and vote-only SMC evidence by setup ID, matches
a same-symbol OrderFlow event to a trigger vote within one second, and requires
two distinct qualifying book quote versions 4–12 seconds apart. A research
comparator permits one non-directional missing-confirmation observation between
the two votes but resets on reversal holdoff, adverse flow, OFI 3s misalignment,
direction conflict, incomplete coverage, or invalid provenance.
Unmatched votes and vote-only setups without a lifecycle transition return an\nindeterminate null, never a fabricated negative or a fabricated stage.

## Observed 27 October session

Source: production Supabase project dehdptgkqbrkyzyodicd, read-only table
public.nifty_daily_market_logs (Lightsail journald). Archive captured 16,719
lines by 15:27 IST. Four 120-second windows from distinct SMC first-trigger
markers contained 44 same-setup vote/book pairs.

| First trigger IST | Contract | Matched votes | Qualifying book observations | Confirmation resets | Neutral observations | Strict logged pair | Conservative neutral-tolerant logged pair |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 11:34:05 | 22750 CE | 12 | 3 | 6 | 3 | No | No |
| 11:58:00 | 22750 CE | 12 | 1 | 7 | 4 | No | No |
| 13:04:06 | 22750 PE | 9 | 1 | 6 | 2 | No | No |
| 13:46:06 | 22700 PE | 11 | 2 | 9 | 0 | No | No |\n| 14:15:04 | 22650 PE | 2 | 1 | 1 | 0 | Indeterminate (stage missing) | Indeterminate (stage missing) |

The final setup has two different quote versions (252 at 14:15:04 and\n412 at 14:17:00); the first book observation was aligned, but the second\nwas explicitly adverse (`fresh_opposing_context`). The vote-only setup is\nreported without inventing an unobserved lifecycle transition.\n\nA log marker is not a complete quote trace. These results do not prove that
the actual trading state machine invoked the confirmation method on every
book update. The live day logged zero SMC_SUSTAINED_FLOW_PROOF events,
zero manager-qualified events, and zero order submissions.

## Earlier session comparison — event-marker counts, not profitability inference

| Date 2026 | SMC first-trigger stage markers | SMC trigger-vote log lines | SMC proof logs | All-strategy manager-qualified stages | All-strategy broker-accepted stages | Server starts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oct 21 | 1 | 6 | 0 | 1 | 1 | 4 |
| Oct 22 | 5 | 78 | 0 | 1 | 0 | 2 |
| Oct 23 | 7 | 115 | 0 | 1 | 1 | 4 |
| Oct 26 | 5 | 71 | 3 | 2 | 2 | 12 |
| Oct 27 | 4 | 46 | 0 | 0 | 0 | 6 |

Policy versions changed during the week. All-strategy stage counts
are not attributable to SMC without the matching setup ID. Separately,
the two real trades on 26 October lost INR 577.42 net in the ledger;
more approvals do not automatically imply better net outcomes.

## Missing data and promotion criteria

The archived journald lacks complete rejected-candidate contemporaneous
executable ask/bid, finalized premium candles and VWAP/ATR provenance,
unthrottled book events, broker fill/latency, stop/exit trajectory and
slippage/fees. A credible P&L comparison needs time-synchronised rejected
candidate snapshots and forward executable-side quote paths with strict
contract identity, risk geometry and capital contention.

Collect a bounded, allow-listed research snapshot keyed by unique setup ID,
not every evaluation loop. Compare alternative admission rules
one at a time across development and later held-out chronological sessions.
Require improved after-cost expectancy and acceptable maximum drawdown
before live promotion.

Six server starts on Oct 27 and 12 on Oct 26 warrant a separate deployment
audit against systemd invocation IDs and release history. Graceful shutdown
markers alone do not identify the issuer. The existing flat-broker and
validated-CI deployment guards are unchanged by this research.

## Verification

Sixteen focused tests cover lifecycle/vote-only deduplication, rejected side conflicts,
distinct versions and time separation, adverse resets, three-second OFI
misalignment, missing evidence and a truly neutral research variant.
The tool is research-only and makes no order or live configuration change.
