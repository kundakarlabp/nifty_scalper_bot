# Execution evidence hardening for NIFTY historical research — 9 November 2026

## Status
**Research-only patch; not a certified executable backtest or a LIVE release.**
All 25 original encrypted Breeze windows (2024-11-06 to 2026-11-05) are privately backed up in Google Drive. Original collection: 6/25 windows meet the strict collector completion flag, 19/25 remain partial; 1,454,867 one-minute candles reported. 4,491 archived files passed SHA-256 and candle integrity checks. This is not historical full-option-chain coverage.

Targeted re-acquisition run [42635897210](https://github.com/kundakarlabp/nifty_scalper_bot/actions/runs/42635897210) made 158 authenticated Breeze history requests. 139 returned data, all **identical** to their original contract series by SHA-256; 19 requests failed again. No additional historical candles were recovered. The encrypted repair ZIP is also in the private Drive folder. Repeating the same requests without changing the upstream coverage source is not a backfill solution.

## Source hierarchy and no-leakage policy
1. **Recorded production-composition feed** with exact logged contract identity, historical instrument master, synchronized FULL-depth bid/ask and timestamps is the highest *simulated* execution evidence. It still does not certify broker acknowledgments or queue priority.
2. **Read-only observed depth simulator** (`backtesting/execution_evidence.py`) requires exactly five ordered, positive levels each side, uncrossed book, exact symbol/token/expiry/lot, explicit historical identity and clock provenance, and a fresh post-latency quote. Orders consume only displayed quantities; remainder is PARTIAL, not filled. No same-bar OHLC fills, no hypothetical midpoint improvement, and no silent market-depth extrapolation.
3. **Breeze identified OHLCV bars** (`backtesting/breeze_bar_research.py`) support research-only causal indicators and 1m/5m/15m observations with explicit session hours. Missing one-minute bars invalidate the containing 5m/15m aggregate; do not forward-fill.
4. **Generic dataframe backtester** is an engineering fixture, not an account-profitability oracle.

`ReplayHarness(..., strict_execution_evidence=True)` marks all historical OHLCV-derived quotes `tradable_quote=False` and `replay_reject_reason=historical_ohlcv_not_executable` before publishing them to paper execution. Earlier modeled-bar behavior is retained by default for compatibility, but outputs expose `BAR_PROXY_ONLY`, `execution_grade_ready=False` and `live_promotion_eligible=False`. Recorded-feed replay remains a separate route.

## Dated charges, contracts and risk
Historical option STT **sell side**: 0.1% of premium through 2026-03-31; 0.15% from 2026-04-01 (Zerodha bulletin, 1 Apr 2026). NSE option transaction charge was 0.03503% on 2024-10-01; the currently published charge is 0.03553%. The precise intervening effective-date schedule must be documented, so `historical_option_fee` requires an externally verified exchange-fee rate for each evaluation date; it never applies today's rate indiscriminately to two years of history. Model is explicitly **Zerodha**; Breeze supplies market research bars, not broker execution/account charges. Verify actual historical contract lot size per expiry (NIFTY 25/75/65 transition), price ticks, exercise/settlement rules and brokerage/fees against the broker contract note. Short-option opening requires external margin evidence; a SELL-to-close requires an independently verified existing long position ledger.

Reference links:
- https://zerodha.com/marketintel/bulletin/391488/revision-in-transactions-charges-from-1st-october-2024
- https://zerodha.com/marketintel/bulletin/445377/revision-in-stt-securities-transaction-tax-from-1st-april-2026
- https://zerodha.com/charges/
- https://nsearchives.nseindia.com/content/circulars/FAOP70616.pdf

## Reproducible read-only checks
Run after installing the project in an isolated research environment:
```bash
pytest -q tests/backtests/test_execution_evidence.py tests/backtests/test_breeze_bar_research.py tests/backtests/test_replay_strict_execution_evidence.py tests/backtests/test_breeze_execution_readiness.py tests/backtest/test_execution_sim.py
python -m scripts.audit_breeze_execution_readiness \
  --coverage /private/Breeze_2year_coverage_audit_20261109.json \
  --validation /private/Breeze_2year_candle_validation_20261109.json \
  --repair-comparison /private/Breeze_repair_comparison_20261109.json \
  --output /private/execution_readiness.json
```
This report must state `execution_grade_ready=false` for the actual Breeze OHLCV archive even when its SHA-256 integrity is valid. Do not publish raw data, account credentials, tokens or private decryption keys.

## Outstanding blockers to account-level fill equivalence
- True historical contemporaneous bid/ask (prefer five levels), quote freshness, tradeable strikes for decisions and broker-side queue/partial-fill evidence; none can be reverse-engineered from one-minute OHLCV.
- Earlier periods lack full historical strikes and verified source clock convention. The special 2025-02-01 Budget session requires separately verified intraday records; 2025-10-21 Muhurat must use its special-session clock.
- 34 empty futures contract-days, 19 persistent failed requests and 105 normal-session gap series remain; re-requests recovered **zero** bars. Some far-month empties may be genuine illiquidity.
- Cost-note reconciliation, calibrated per-leg latency and slippage, margin and atomic/multi-leg partial-fill handling, chronological train/validation/test partitions, stress tests and paper-trading comparison remain unverified.
- No research result may be promoted to LIVE automatically. A passed code CI tests software mechanics, not historical executable market evidence or profitability.

## Next controlled milestone
Record FULL-depth snapshots and actual broker acknowledgments/fill times for prospective sessions in the already existing ReplayArchive; independently check symbol/token/master/expiry, align event clocks, calibrate slippage and latency on held-out days, validate basket partial-fill recovery and observed P&L net of dated costs. Re-run out-of-sample strategy selection only after the evidence gate passes. Do not infer an available order book from OHLCV or EOD rows.
