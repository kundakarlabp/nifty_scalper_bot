# Exit-policy research using recorded executable bids

This tool is research-only. It cannot submit Zerodha orders, change a live
strategy, or turn an MFE-based hindsight calculation into a causal backtest.

## Question

For one already-filled NIFTY option BUY, would a specified exit overlay have
closed the position **during the actual observed holding window**, using the
bid and visible size that the bot had already received?

It compares the canonical ledger's verified net exit with:

- Sell targets at 0.6R, 0.8R and 1.0R.
- Tier-1 cost-adjusted protection from 0.6R and subsequent 40%, 50%, 60%
  achieved-executable-bid peak locks from 1R.

A target overlay not reached during the original holding window uses the
original **actual** exit (no change in exit logic). A trailing variant that
does not exit before the original exit is **UNRESOLVED**, not assumed flat.
An exchange SL-LIMIT gap through its reference limit is **UNRESOLVED**.

This tool does **not** reconstruct the full live stop owner, model exchange
queue priority, changing costs, exact order-modification confirmation, or
subsequent entries. It cannot justify live parameter promotion.

## Inputs

The original session is the validated local `ReplayArchive` JSONL file.
The separate `trades.json` is an export of canonical `nifty_trade_ledger`
CLOSED BUY rows. Required fields per trade:

`trade_id, trading_date, symbol, side, entry_price, initial_stop, quantity,
entry_filled_at, exit_filled_at, net_pnl, ledger_complete`

Costs come from `effective_costs`, or
`outcome.effective_costs.total`, or `estimated_costs`; do not invent
missing costs or stops. Rows for another date are excluded. Other missing
fields become per-trade UNRESOLVED outcomes.

```sh
PYTHONPATH=src python -m nifty_scalper_bot.backtesting.exit_policy_research \
  --session /private/replay_archive/2026-10-21.jsonl \
  --trades /private/exit_research/2026-10-21-trades.json \
  --output /private/exit_research/2026-10-21-exits.json \
  --slippage-points 0.5
```

The tool is local and read-only; never copy production credentials, quote
recordings or private trade files to a public repository. The session
must exist, pass ReplayArchive integrity validation and contain
full-size bid-depth quotes within 10-second observation gaps. An input
basename is illustrative and does not assert availability.

## Interpretation and promotion boundary

The result is `RESEARCH_CANDIDATE_MODELED_QUOTES_NOT_BROKER_FILLS`. A
scenario is never "profitable" or promotable merely because its modeled net
is higher. Never aggregate only resolved winners: show unresolved count and
leave total net null for incomplete scenarios.

Before selecting live changes, rerun a deterministic full-execution replay
with canonical broker costs and safety state, check baseline-to-live stop
parity, and test chronological holdouts across regimes. Stress execution
slippage, partial depth, stop-limit gaps, and periods without quote data.
Preserve position sizing, exchange-protection and the 5% day-loss circuit.
