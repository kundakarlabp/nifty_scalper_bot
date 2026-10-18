# Canonical backtesting framework

## Authority hierarchy

1. **Production-composition recorded-feed replay** (`backtesting/runtime_research.py`) is the canonical bot replay for dates captured by `ReplayArchive`. It uses the production component graph and replaces only external network/broker execution with isolated archived adapters.
2. **Historical component bar research** (`backtesting/strategy_research.py`) is secondary evidence for older dates. It uses production strategy classes but cannot prove historical contract, quote/depth, arbitration, dynamic risk-sizing or broker-fill parity.
3. **Independent Backtrader oracle** verifies resolved-trade gross-P&L bookkeeping outside the bot implementation.
4. The generic dataframe `BacktestEngine` is an engineering fixture. It must not be presented as current-bot profitability evidence.

No layer may silently upgrade its evidence label.

Several requested professional controls already existed and are reused rather
than reimplemented: `HistoricalContractCatalog` provides point-in-time basket
resolution, `ReplayHarness` delays bar-start observations until availability,
`PaperFillEngine` supports quote/depth-aware execution and calibration from
measured completed-trade slippage/latency, and the canonical completed-trade
analysis already provides chronological walk-forward, execution-quality checks,
PBO/deflated-Sharpe evidence and broker-cost provenance. The admin dashboard
already exposes one-click component research and recorded-session replay.

## Data and causality rules

- Every bar is acted on only after it is complete; entries occur no earlier than the next executable observation.
- Historical contract selection must use only information available at that decision time. Current instrument masters are never a historical universe.
- Missing option observations are data-quality events, not prices.
- A position whose exit cannot be priced is `UNRESOLVED` in primary metrics. A full-premium-loss assumption is allowed only in a separately labelled worst-case stress ledger.
- A one-minute candle touching both stop and target is unresolved in primary metrics because OHLC does not reveal event ordering. Stop-first is retained only as a pessimistic stress assumption.
- Primary reports expose resolved count, unresolved count/rate and whether the primary metrics are complete.
- Parameter selection may not use a scenario containing unresolved exits.
- Costs remain the shared canonical Indian options cost model; slippage is reported independently and stressed at multiple levels.

## Promotion rules

A strategy/configuration is not promotable from a profitable headline alone. Promotion requires causal historical selection, complete primary exits, adequate sample size, transaction-cost/slippage stress, development-only parameter selection, untouched later-period evaluation and prospective recorded-feed replay. Runtime/live settings are never changed by a research workflow.

## Legacy committed result artifact

`docs/research/orb_monthly_2017_2020_results.json` was generated before the
unresolved-exit correction and therefore contains the former full-premium
penalty inside headline P&L. It is retained only as historical audit evidence.
Do not use it for current strategy selection. Post-change workflow artifacts
(`development_results.json`, `validation_results.json`, `final_results.json`
and `external_backtrader.json`) are authoritative for this protocol.

## Ordinary ChatGPT chat operation

Two isolated routes are available without ChatGPT Work:

- **Current bot / recorded sessions:** update `deploy/research_request.json` with a fresh validated ID and mode `all`, `components` or `runtime`. The existing AWS updater launches the fixed read-only research worker. Status/results are readable through the existing production status/relay endpoints. No shell command or credential is placed in the request.
- **Free independent historical ORB:** create a repository-owner issue titled `[Backtest] ORB ...`. GitHub Actions downloads the checksum-verified public 2017-2020 dataset, runs the fixed historical protocol, then validates resolved gross P&L with Backtrader. The public repository uses standard free Actions runners.

TradingView and Streak remain useful visual/manual signal checks, but neither is the canonical automation backend because a free stable programmatic interface for this multi-contract workflow is not available through ChatGPT.

## FYERS expired-option import (separate, secondary evidence)

The read-only scripts/archive_fyers_expired.py importer is an **optional**
offline source, not an execution broker or a substitute for recorded-feed
runtime replay. Official FYERS API v3 documentation (updated 14 October 2026)
defines the sequence expiry_dates -> history_underlying_symbols ->
fno_historical_data. The API uses explicit columns to name optional OI and
Greek values. Documentation:
https://fyers.in/api-docs/v3/api/market-data/expired-fo-data/

- Requires **operator-provided** FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN in
  the environment and the optional fyers-apiv3 Python SDK. Never put tokens
  in repository files, CLI flags, logs or research reports.
- The explicitly chosen strike and expiry are checked against FYERS'
  provider-reported contract catalogue. A nominated strike is **not** proof of
  the historically traded ATM basket. No expired contract or token is invented.
- Normalization checks minute timestamps, source bar-label convention, OHLC,
  volume, and nullable open interest. The source bar_start/bar_end labels
  must be independently verified by the operator; normalized output is always
  completed-minute bar-start. Bad/empty responses are not interpolated.
- Collection keeps provider identifiers, original columns, source hashes and
  provenance outside the canonical candles/ tree until independent spot and
  futures context and a historically verified lot size are supplied.
- Optional staging (--context-archive, --stage-output,
  --historical-lot-size, --lot-size-source) calls the **existing**
  backtesting.strategy_research.load_archive validator and requires at least
  60 consecutive synchronized observed minutes across spot, futures, CE and
  PE. It never fills gaps or fetches a later contract to replace missing bars.
- Stage and collection results are always RESEARCH_CANDIDATE,
  promotion_eligible=false and live_equivalent=false. Historical bid/ask
  depth, queueing, real fill feasibility and complete point-in-time option
  selection are **not** established by this import.
- A FYERS Standard plan allows at most 10 expired F&O intraday API requests
  per day according to its October 2026 documentation. The importer makes
  at most two uncached minute-candle requests per CE/PE pair, with a per-run
  request budget; account-wide requests made elsewhere are not measurable.

Example for an **already expired** provider-listed strike (date/strike are
illustrative, not an assertion that that exact provider contract exists):

~~~bash
export FYERS_CLIENT_ID="..." FYERS_ACCESS_TOKEN="..."
python scripts/archive_fyers_expired.py \
  --expiry 2025-03-27 --strike 23000 \
  --start 2025-03-26 --end 2025-03-26 \
  --timestamp-convention bar_start --outdir data/fyers_history
~~~

This collection alone does **not** enable historical strategy promotion. Use
the staging arguments only with independently verified historical spot and
futures data and exchange-derived historical lot-size provenance. Do not
equate a successful importer regression suite with a downloaded or profitable
2025–2026 weekly-options backtest.
