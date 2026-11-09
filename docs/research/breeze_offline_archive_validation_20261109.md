# Offline Breeze historical validation — 9 November 2026

## Data provenance and evidence tier

- Source: **25 privately encrypted ICICI Breeze acquisition windows**, 2024-11-06 through 2026-11-05, held in the user's private Google Drive historical folder. The original ZIPs were not modified.
- Local independent check verified each encrypted ZIP's outer `history.cms.sha256`, decrypted its AES-256-GCM CMS payload in a temporary directory and checked its inner `SHA256SUMS` against archived members. No broker keys or raw candles were committed to GitHub.
- Result: 25/25 sealed windows verifiable, 6 flagged complete, 19 partial, **zero checksum or coverage-total disagreements**.
- Normalized 1-minute rows counted: NIFTY index 184,786; futures 536,695; selected options 733,386; total **1,454,867**.
- Observed contract-days: index 522, dated futures 1,472, selected options 1,960. **This is not the historical full options chain**.
- A separate 5m and 15m completeness check accepts only windows containing *every expected source minute* under an explicitly supplied session grid; **290,070 complete 5-minute windows** and **96,399 complete 15-minute windows**. Incomplete windows, respectively 5,976 and 2,283, are not forward-filled or given synthetic closes. These totals are pooled across underlying/index/futures/option contracts, not independent calendar candles.
- The 2025-10-21 Muhurat derivatives session was classified with a 13:45–14:45 IST sixty-minute grid. **Eight contract series contain an extra 14:45 timestamp**, which must remain a flagged boundary observation until source timestamp conventions are verified; it is not assumed to be a post-close executable minute. The index series for that date requires independent source reconciliation.
- Across the archives, **zero** contract records declare `source_timestamp_convention_verified=true` or `source_timezone_verified=true`. The 2025-02-01 Budget session remains absent as validated intraday research data.
- SHA-256 of the independently generated private-safe JSON validation report: `132bc982b6154b159561362dffb98976fdf06dadea626757f531083f141cf252`.

## Repository implementation

- `src/nifty_scalper_bot/backtesting/breeze_bar_research.py`: strict interval aggregation with explicit session start/end and rejection of partial constituent windows.
- `src/nifty_scalper_bot/backtesting/execution_evidence.py`: fail-closed observed-book, point-in-time contract, latency, depth-limited partial-fill and dated fee research; output is labelled **modeled**, never broker-confirmed.
- `src/nifty_scalper_bot/backtesting/replay.py`: opt-in `strict_execution_evidence=True` marks OHLCV-derived replay quotes untradable rather than assigning fills using the close/LTP.
- `scripts/audit_breeze_execution_readiness.py`: CLI consumes inspected aggregate audit manifests and emits an explicit, non-promotable evidence-readiness report. It does **not** acquire market data or contain credentials.
- Unit and normal CI shards plus deterministic E2E checks verify the isolated research components. Research code is not automatically invoked by live StrategyManager, exchange execution, or a production deployment.

## Backfill conclusion

Run `42635897210` retried 158 flagged contract-days. 139 returned series match the existing candles byte-for-byte and 19 requests still failed. Net recovered one-minute bars: **0**. Stated gaps remain gaps; no default forward-fill, interpolation, invented strike, bid/ask, depth, or quote-derived Greeks is permitted.

## Practical readiness

**Causal OHLCV research: available with missing-session and source-convention caveats.**
**Historical execution-level fill validation: NOT READY.**
The recorded quote/depth stream, order acknowledgments, measured latency/partial fills, historically verified contract master, historical broker fee changes and point-in-time listed option universe remain prerequisite evidence. No backtest result is authorized to promote to LIVE without these and chronological out-of-sample/prospective checks.

The main bot is not redeployed or reconfigured by this research audit.
