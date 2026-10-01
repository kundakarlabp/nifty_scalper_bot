# Backtest annualization: session equity rather than minute row count

The generic `BacktestEngine` treated every input bar as a daily observation
while using 252 observations per year. Identical session-end equity paths
therefore produced different CAGR, volatility and Sharpe/Sortino solely when
flat intraday bars were inserted. This could mis-rank optimization candidates
with different sampling densities even when economic outcomes were identical.

The existing performance owner now resamples observed daily closing equity,
includes the first session's entry costs relative to initial capital, and
annualizes those daily returns with 252 sessions/year. Empty calendar days are
excluded. Per-bar equity, return series, drawdown, orders, trade accounting and
JSON daily PnL retain their behavior. Single-session standard deviation is not
estimable; the existing zero-volatility/undefined-ratio reporting convention
now yields finite zeros instead of NaN. These zeros are not evidence of safety
or statistical significance.

Test-first public `BacktestEngine.run()` regressions compare one, two and five
flat bars per session across Thursday, Friday and Monday. Total return and
all annualized metrics must agree, with no synthetic weekend observations.
A one-session flat test verifies finite metrics.

## Before/after fixture

Initial capital ₹10,000, fixed quantity 10, prices 100 → 110 → 90 over three
observed sessions, constant long position and a deliberately illustrative 1%
entry commission. This is engineering data, not a NIFTY strategy backtest or
an estimate of actual Zerodha fees.

| Metric | Before, one bar/session | Before, two bars/session | Corrected, either density |
|---|---:|---:|---:|
| Total return | -1.10% | -1.10% | -1.10% |
| Annualized growth | -60.51% | -37.16% | -60.51% |
| Annualized volatility | 23.95% | 15.47% | 23.95% |
| Sharpe, zero risk-free fixture | -3.7924 | -2.9360 | -3.7924 |

The large annualized loss is the mathematical extrapolation of a tiny losing
fixture, not a reliable forecast. Real optimization still needs sufficient
complete sessions, actual option prices, realistic costs/fills, synchronized
context, chronological holdouts, and stability across expiries/regimes. This
patch corrects reporting units and does not demonstrate increased profitability.
