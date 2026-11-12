# NIFTY option historical execution dataset — procurement and acceptance contract
Date: 2026-11-12
Owner: offline research only. No broker credentials, exchange data, private keys, production changes, or LIVE promotion.

## Decision
**NOT EXECUTION-GRADE.** The existing Breeze acquisition is useful for causal OHLCV research, not historical executable option fills. Re-requesting the same 158 contract-days is not a fix: 139 repeated series were SHA-256 identical and 19 still failed. Original 25 encrypted archives are immutable and privately backed up. Do not copy them into the GitHub repository.

## Historical target
- NIFTY underlying index, dated NIFTY futures and the **point-in-time NIFTY option universe** from 2024-11-06 to 2026-11-05, inclusive.
- Instrument listing and delisting, each dated expiry, strike, CE/PE, exchange token, lot size, tick size, reference index and session calendar **as valid at the historical instant**, not as known today.
- Preserve the special 2025-02-01 Budget session (09:15–15:30 IST) and 2025-10-21 Muhurat derivatives session (13:45–14:45 IST) as separate calendars; verify timestamp semantics and the eight boundary observations; reconcile anomalous spot timestamps before any joint futures/spot hypothesis.
- Official holidays are explicit CLOSED, not missing data; unknown and no-trade intervals must remain distinguishable.
- No silent interpolation, forward-fill, bar-to-quote conversion, synthetic strike listing, IV, Greeks, or assumptions about fills.

## Required provider deliverables
1. **Historical contemporaneous quotes/order events:** for every listed NIFTY option and relevant dated futures, time-stamped best bid/ask prices and quantities at minimum, preferably top five levels on both sides. For event-data products, require order addition/modification/cancellation/trades and sufficient initial book state + sequence/reset rules to reconstruct bid/ask snapshots deterministically.
2. **Point-in-time masters:** historical daily contract and listing files, expiry transitions, symbol/token changes, CE/PE, strike, tick, lot size, trading status, settlement and contract exercise rules.
3. **Independent clocks and events:** timezone/UTC offset, start/end-of-bar convention, event ordering/sequence, source and publish timestamps, market open/close, halted sessions and special session rules.
4. **Completeness and rights:** exchange/provider provenance, archive SHA-256 checksums, per-day/per-contract counts and known losses, licensed research/storage rights, retention and API/download limits, coverage for the entire target window.
5. **Execution calibration:** contemporaneous broker order submit/ack/modify/reject/partial fill/cancel/fill timestamps and commissions/charges/margin ledger from prospective *read-only* records; exchange quote data alone cannot establish broker execution.
6. **Dated costs:** verifiable historical fee/rate changes, exchange fees, STT, GST, stamp/SEBI charges, brokerage, bid/ask/spread, and short-option margin inputs for the actual broker, not estimated from one-minute volume.

**Preferred primary source:** NSE Data & Analytics historical F&O Order & Trade Data. Confirm the product's exact fields, whether a complete book can actually be reconstructed from the archive, time horizon, costs, licensed usage and achievable volume with an official sample before purchasing. Real-time NSE Level 2/tick descriptions do **not** imply those historical fields are included in the order-and-trade product. Do not infer them.

- Historical order/trade product: https://www.nseindia.com/static/market-data/eod-historical-data-subscription
- NSE quote feed depths: https://www.nseindia.com/static/market-data/real-time-data-subscription
- NSE contact: marketdata@nse.co.in

## Secondary research/data-access routes
- Quantsapp advertises historical NIFTY option-chain bid/ask, greeks, order flow and point-in-time expiry/lot data at one-second resolution in its own backtester. It does **not** establish from this public description that raw history is downloadable, API-accessible, top-five depth, or licensed for an external bot. Verify export rights, quote field definition, timestamp controls and sample data before committing: https://www.quantsapp.com/backtesting/data
- Global Datafeeds documents NFO historical TICK calls and bid/ask L1 fields; this is a plausible independent historical best-bid/ask source but must be checked for available 2024–2026 contracts, export limits, timestamp semantics, volume and historical quote completeness. Its listed historical format does not prove five-level depth: https://globaldatafeeds.in/global-datafeeds-apis/global-datafeeds-apis/introduction/type-of-data-available/
- Reject third-party claims of complete tick/L2 archives without a verifiable licensed sample and exchange provenance; do not use scraped/live option-chain pages as historical bid/ask evidence.

## Mandatory pilot before full procurement
Request a small but representative paid/free sample including: a normal high-volume Tuesday/weekly expiry, a volatile intraday session, an illiquid/wide-spread strike, and exact special-session samples if retained. Require contemporaneous order/quote updates with source timestamp, historical master, sample count and gap audit, and a documented reconstruction example.
Reject a trade-only/EOD/1-minute-candle-only offering as **insufficient for execution-grade depth replay**.

## Data normalization and safety
- Keep encrypted immutable raw artifacts and independent SHA-256 logs in private storage. Never upload raw exchange files to the source repository.
- Partition research store by trading date, venue, symbol/token, expiry, right and strike. Preserve raw event order, source timezone, ingest time, correction flags and versioned history.
- Store per-session quote completeness, contract-days, exact gaps, stale/crossed books, timestamps and provenance. Reconstruct top-five depth only from validated events; refuse snapshots with unknown book state, sequence loss, quote staleness or mismatched identity.
- Join underlying decisions **as-of decision time**, then select the first valid post-latency quote; no same-bar close/OHLC fills, favorable midpoint improvements or extrapolation past observed quantity.
- Model multi-leg orders as non-atomic unless supported by broker evidence; require margin, hedge confirmations and partial-leg unwind paths.
- Reconcile historical date-specific fees, overnight/expiry risk, time-of-day spread regimes, slippage and known broker fills.
- Never mark historical simulated fills as confirmed broker fills.

## Release gates
| Gate | Pass requirement | Current |
|---|---|---|
| G0 archive integrity | Hash-verified historical archives, no duplicate overwrite | 25/25 archives validated; original 19 partial windows |
| G1 strategy candles | Strict 1/5/15m causal samples, missing periods exposed | Conditional research only |
| G2 historical universe | Point-in-time all candidate strikes, lot/tick/expiry, special dates | FAIL |
| G3 historical execution quotes | Complete synchronized contemporaneous bid/ask or verified 5-level reconstruction for studied events | FAIL |
| G4 event quality | Measured sequence gaps, timestamp convention, invalid book/quotes, quote age, excluded sessions | FAIL |
| G5 simulated executable P&L | Date-specific costs, latency/depth, margin, true long/short ledger, basket partial fills | FAIL |
| G6 external validation | Broker-recorded calibration, held-out chronological OOS and prospective paper | FAIL |
| G7 LIVE promotion | Separate human and operational signoff; never automatic | BLOCKED |

The final delivered report must publish counts for expected/observed contract-days, missing-minute and invalid-depth ranges, rejected strategy events (with causes), date-specific universe coverage, archive digests, date-specific charges, and net OOS statistics. A positive gross candle-only P&L is not certification.

## Baseline, held apart from genuine trading gaps
- 29 normal-session exchange holidays, 34 empty futures contract-days, 19 persistent failed requests, 105 nonempty missing-minute contract-days after removing the 2025-10-21 Muhurat anomaly.
- The 2025-02-01 Budget session is absent as verified intraday research data.
- The 158-request targeted Breeze repair recovered **zero** new bars.
- Existing source: 184,786 index + 536,695 futures + 733,386 selected-options minute bars = 1,454,867, not a full option chain.
- The existing `scripts/audit_breeze_execution_readiness.py` and `backtesting/execution_evidence.py` must remain fail-closed. Do not loosen the LIVE gate to force research completeness.

## Vendor enquiry (request for quotation only)
Please provide availability, technical sample, coverage, licensing terms, volume and price for NSE NIFTY options historical **Order & Trade** data for 2024-11-06 through 2026-11-05. We require point-in-time instrument masters (including contract lot/tick/expiry), historical event timestamps, cancellations/modifications/trades, and a verified means of reconstructing at least top-five bid/ask depth or contemporaneous depth snapshots. Please specify whether records include enough initial book state and sequence/reset information for exact book reconstruction, identify any unavailable dates/contract subsets, and confirm permitted private research/backtesting use. Quote sample versus complete-history cost separately. No order or payment is authorized by this enquiry.
