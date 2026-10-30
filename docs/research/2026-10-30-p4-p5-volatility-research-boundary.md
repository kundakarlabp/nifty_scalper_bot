# 30 October 2026 — P4/P5 volatility strategy research boundary

## Scope, evidence and trading status

These are **quote-geometry research utilities**, not trading strategies or profit
validation. `src/nifty_scalper_bot/backtesting/volatility_research.py`
does not register an EliteStrategy, connect to Zerodha, authorize positions,
alter entry/exit arbitration or reactivate the retired short-strangle strategy.

Both assessors require **explicit `EXECUTION_MODE=SHADOW`**. A caller supplies
point-in-time NIFTY instrument-token-identified CE/PE bid/ask observations with
FULL or archived FULL provenance, actual quoted bid/ask quantities, a same
future expiry, a common contract lot size, and subsecond observation alignment.
Missing/contradictory fields, stale quotes, cross-session observations, inverted
markets, mixed expiries and unsynchronized legs fail closed. They do **not**
generate orders or infer executable data from LTP or candles.

The repository's existing `OptionsChainManager` is **not** an acceptable
live source for these assessments as currently implemented: it can set expiry
from the local date and its refresh path is not a synchronized validated
multi-leg FULL bid/ask/depth snapshot. The `PremiumDecayStrategy` old short
strangle is also explicitly not a LIVE execution authority.

## P4 — defined-risk credit condor

Order of legs: long put < short put < short call < long call. Credit estimates
sell at **bid**, protective wings buy at **ask**. Compute:
- `credit = short_put_bid + short_call_bid - long_put_ask - long_call_ask`
- `gross_best = credit × contract_units`
- `gross_worst = (max(put_width,call_width) - credit) × contract_units`
- canonical **entry** exchange, GST, brokerage, STT, stamp and SEBI fees, by
  calling the existing `estimate_order_cost` for each of four hypothetical
  orders. These are model outputs only.

The reported loss is **gross terminal payoff loss before transaction/settlement
fees**. A net maximum loss is not certified: short-leg execution before
protective wing acceptance can expose uncapped directional risk; margin,
broker partial fills, mark-to-market liquidation, assignment/settlement,
exercise taxes, liquidity disappearance, gap risk and exit-side fees are not
simulated. Never use this geometry for a real four-leg order.

## P5 — long ATM straddle

Same-expiry, same-strike CE and PE both purchased at **ask**; the supplied
point-in-time NIFTY spot must be within 25 points of the common strike to
qualify as ATM. Compute the
gross expiry breakevens `strike ± (CE_ask + PE_ask)`, premium outlay,
canonical modeled entry fees, and the diagnostic difference between
a separately supplied annualized realized-vol forecast and annualized IV.
Require the forecast's *as-of* timestamp to be no more than 60 seconds before
the snapshot; reject future dates, missing/invalid IV and stale forecasts.
The forecast itself is **not** validated for calibration, historical
availability or fairness by the calculation. It is not an entry recommendation.

The potential payoff is nonlinear and holding before expiry may behave very
differently; bid/ask liquidation, IV crush, theta, expiry settlement and dynamic
hedging are not modeled. The fee-inclusive premium at risk excludes further
exit, exercise and settlement charges.

## Required subsequent work before live evaluation

1. Obtain timestamped, licensed, source-verifiable historical **FULL**
   option books with all leg tokens, correct expiry and quantities, plus
   NIFTY futures context. Do not synthesize executable prices from LTP.
2. Extend the canonical **one** execution owner to handle linked four-leg
   intent IDs, capital/margin reservations, protective wings first, exact
   partial/failed leg recovery, cross-leg reconciliation, timeout and forced
   flattening, venue/broker order controls, and total portfolio stop criteria.
   Real exchange/broker product support must be established first.
3. Add a real position/portfolio simulator with legwise fills, actual
   slippage/fees, expiry/settlement, margin and stress (gaps, IV spikes,
   loss of depth); stress scenarios include one or more missing wing fills.
4. Evaluate prespecified chronological OOS samples with unchanged contract
   selection, conservative costs, realistic event-day/week-by-week outcomes,
   and prospective SHADOW/PAPER observation. Assess drawdown, tail loss and
   probability of stressed negative outcomes, not just win rate.
5. Only after these conditions, propose a separately reviewed design for a
   supported live policy. **No production flag is enabled by this PR.**

This research is deliberately narrower than a deployable strategy, and no
claim of positive after-cost expectancy is made.
