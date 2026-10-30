# 30 October 2026 — P3 failed-breakout reversal research candidate

## Purpose

Create a separable, **non-live** reversal hypothesis instead of weakening the
canonical SMC/ORB/VWAP direction gate. The pattern is not presumed profitable,
and a rejected wick is not automatically a liquidity grab.

## Point-in-time contract

The existing archived hypothesis classifier already requires 45 contiguous,
same-session, final NIFTY futures one-minute bars with valid OHLC and finite
nonnegative traded volume. Candidate detection uses two **separately completed**
bars after an immutable level taken from the preceding 20 futures candles.

- For a **PE** candidate: the first bar opens inside and probes above the prior
  20-bar high by more than 0.10 of the available 14-bar underlying ATR, but
  closes back inside. A later completed candle fails to reclaim the boundary,
  closes bearish and below the sweep candle low by at least 0.05 ATR.
- For a **CE** candidate: mirror at the preceding 20-bar low, require rejection
  back inside, then a bullish confirmation above the sweep candle high.
- The sweep and confirmation require genuine futures participation relative
  to the already-observed prior 20 bars (>=1.1 and >=0.8 of their reference
  mean respectively). These numbers are **research hypotheses**, not
  established optimal NIFTY thresholds.
- The reference excludes both the sweep and confirmation candle. Missing
  completed bars, gaps, provisional observations, malformed OHLC, unavailable
  volume, or a breakout that holds outside cannot authorize this vote.
- The existing archive simulator fills no earlier than the next minute, and
  applies its existing option ATR proxy stop/target and full modeled
  costs/slippage. The next-minute OHLC is **not** available at signal time.

## Isolation

Set `EXECUTION_MODE=SHADOW` and
`RESEARCH_FAILED_BREAKOUT_REVERSAL_ENABLED=true` **only** in the
`run_directional_hypothesis_research` archived replay. By default the
existing three families and their reports remain unchanged.
The additional component is `FailedBreakoutResearch`; it is not an
`EliteStrategy`, is not registered in live builder or manager, never relaxes
underlying-direction authority, and never authorizes a real broker order.

## Pre-promotion evidence requirements

Use an immutable development/test split; historical date-correct near-ATM
weekly contracts; complete futures context and bid/ask provenance; executable
option economics; failed- and successful-breakout opportunity denominators;
conservative slippage and intrabar ambiguity reporting; full chronological
out-of-sample and prospective SHADOW comparison. Contrast simple boundary
rejection, rejection plus confirmation, and existing SMC on identical events.
Measure after-cost expectancy and expected shortfall, not only win rate.

The existing archive is incomplete for fully realistic historical option
execution; neither a passing unit test nor a positive proxy return establishes
live profitability. No live activation is authorized by this change.
