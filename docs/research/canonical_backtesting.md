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


## Dhan five-year rolling-options archive (offline component research)

The separate **private** repository `kundakarlabp/Nifty_options_backtest_data`
stores checksum-locked Dhan near- and next-expiry ATM-relative NIFTY series
(2021–2026). Rolling ATM is **not** a historically identified option contract:
an actual strike can switch within a session. Do not convert these rows into
invented Zerodha symbols, tokens, expiries, lots, broker quotes, or fills.

Once its independent validation workflow publishes the
`backtest-quality-report.json`, `backtest-catalog.json`,
`backtest-segments.jsonl.gz`, `source-archive-sha256.txt` and
`backtest-readiness.sha256` release assets, extract the ten private half-year
shards into one private local directory. The research-only reader can export
one near-expiry ATM context session (spot, near future, call and put):

~~~bash
python -m nifty_scalper_bot.backtesting.dhan_rolling_archive \
  --shards /private/dhan-extracted \
  --readiness /private/dhan-readiness \
  --date 2025-01-02 \
  --output /private/dhan-session-research.json
~~~

The reader verifies the five readiness assets, each selected normalized task
against its receipt digest and every session row's segment membership. It
retains the real strike and provider timestamp, delays signal availability at
least 60 seconds, and reports synchronized observed minutes plus the longest
contiguous unswitched CE/PE interval. It rejects missing or corrupted inputs.
`component_research_eligible` requires at least 60 such synchronized minutes;
it does **not** mean the market was executable, signal generation passed, or
P&L was established.

**Evidence boundary:** `RESEARCH_CANDIDATE` only. The output is suitable for
offline signal-context studies, debugging strategy no-trade funnels and
identifying candidate days. It is not input to `load_archive`, which correctly
requires named historical NFO contracts. Never use it for order-level backtests,
overnight holding, exact-contract return attribution, or promotion of live
settings. Historical contract identity, point-in-time expiry and lot-size
calendars, execution-quality quotes and prospective recorded-feed replay remain
separate prerequisites. Keep the private data and any exports private; no
production credentials, services, settings or live trading code are modified.
