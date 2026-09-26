---
name: strategy-research-validation
description: Evaluate and improve NIFTY option strategy logic, scoring, thresholds, filters, or parameters using code-correctness review, external evidence, realistic backtesting, walk-forward validation, and anti-overfitting safeguards before any live strategy change.
---

# Strategy Research and Validation

Use this skill for ORB, VWAP, SMC, EnhancedScalping, signal scoring, confidence thresholds, filters, exits, or requests framed as improving profitability.

## Separate three questions

Never collapse these into one:

1. **Engineering correctness** — does the code implement the intended rule without timestamp, state, or execution defects?
2. **Mechanistic plausibility** — is there a defensible market/microstructure reason the rule could help?
3. **Empirical evidence** — does chronological out-of-sample evidence support changing the live strategy?

Correct code is not proof of alpha. A higher historical P&L is not sufficient evidence of improvement.

## Research workflow

```text
current implementation
→ precise hypothesis
→ relevant external/repository evidence
→ falsifiable expected effect
→ smallest candidate change
→ deterministic replay/backtest
→ chronological holdout / walk-forward validation
→ paper/shadow evidence
→ only then consider live-small rollout
```

Prefer primary exchange/broker documentation for execution mechanics, peer-reviewed or otherwise rigorous market-microstructure research for mechanisms, and reproducible quantitative evidence for parameter decisions. Label practitioner heuristics as such. If current external evidence is unavailable, do not describe the change as evidence-backed.

## Before changing strategy code

Establish:

- exact current signal and scoring path
- which data authorizes underlying direction
- completed-vs-forming candle semantics
- timestamp and session alignment
- setup identity and same-bar behavior
- interaction with global gates, regime filters, risk, and execution
- baseline parameter set and current live/replay parity

Change one coherent hypothesis at a time. Do not combine strategy tuning with market-data fixes or architecture cleanup.

## Backtest validity gate

Reject or correct a result containing:

- look-ahead or future-bar leakage
- use of incomplete candles when live logic uses completed candles
- overlapping train/test windows
- test-period reuse during tuning
- data leakage through normalization or feature fitting
- unrealistic fills at candle extrema or untradeable prices
- missing bid/ask, slippage, brokerage, taxes, or option liquidity assumptions
- ignored rejected/partial fills when execution feasibility matters
- expiry/roll mishandling
- cherry-picked dates, regimes, strikes, or successful parameter runs
- backtest features that differ materially from live feature computation

Walk-forward windows must be chronological and non-overlapping on the test side. Keep a final untouched holdout where feasible.

## Compare against baseline

A candidate should be compared with the current validated baseline, not with zero.

Report at minimum:

- net expectancy per trade after costs
- total net P&L
- maximum drawdown
- profit factor
- win rate together with average win/loss
- trade count
- exposure/turnover
- performance across time/regimes where sample size permits
- parameter sensitivity around the chosen value

When many variants are tried, explicitly account for selection/multiple-testing risk. Prefer a stable plateau over a single sharp optimum.

## Decision labels

Use evidence labels instead of profitability claims:

- `ENGINEERING_CORRECT`
- `RESEARCH_CANDIDATE`
- `HOLDOUT_SUPPORTED`
- `PAPER_SUPPORTED`
- `LIVE_SMALL_OBSERVED`

Do not call a strategy `profitable`, `validated alpha`, or `improved` solely from an in-sample or single-period result.

## Implementation rule

If evidence supports a code change:

1. add a red-capable regression or parity test;
2. implement the smallest owner-consistent change;
3. run strategy-focused tests and replay/backtest parity checks;
4. run the repository validation ring;
5. record what changed and what deliberately did not change.

Risk limits, execution safeguards, and market-data quality gates are not tuning knobs.
