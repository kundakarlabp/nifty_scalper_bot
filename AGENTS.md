# NIFTY Scalper Bot — Repository Guidance

## Mission

Maintain a deterministic, observable, capital-protective Python 3.12+ NIFTY options trading system using Zerodha KiteConnect, live market data, Telegram controls, strategy gating, risk management, and automated execution.

Capital protection takes priority over trade frequency and apparent profitability.

Success means:
- only valid NIFTY option contracts can reach execution;
- market data and readiness states are explicit;
- strategies consume prepared context;
- risk and broker constraints cannot be bypassed;
- failures and blocks are actionable;
- changed behavior is covered by focused validation.

Do not optimize for guaranteed profitability. Strategy performance must be evaluated separately through reproducible backtesting, costs, slippage, out-of-sample testing, paper trading, and live metrics.

## Product boundaries

- Trade NIFTY options only.
- NIFTY spot and futures are context only.
- Never place spot or futures orders.
- Every executable signal must resolve to a broker-validated NIFTY option symbol and instrument token.
- Prefer fewer valid trades over noisy signals.
- Do not weaken safety controls to increase trade count.

## Source-of-truth ownership

| Domain | Owner |
|---|---|
| Instrument discovery, contracts, symbol/token mapping | `core/instrument_manager.py` |
| Runtime basket commit | `core/app.py` |
| Subscription, quote, depth, OI and hydration state | `data/market_data_manager.py` |
| Tick-to-OHLC bars | `data/candle_engine.py` |
| Strategy-facing data facade | `data/data_hub.py` |
| Strategy orchestration and evaluation | `core/strategy_manager.py`, `strategies/*` |
| Risk policy and sizing | `risk/*` |
| Order construction and broker execution | `execution/*` |
| Telegram controls and diagnostics | `notifications/*` |

Do not create competing selectors, instrument caches, contract generators, readiness owners, or execution paths.

Strategies must not:
- select or generate contracts;
- fetch broker instruments;
- call broker historical data in the live evaluation loop;
- bypass data, risk, or execution readiness.

Execution must not bypass:
- execution mode;
- instrument validation;
- risk and daily-loss limits;
- margin and lot-size checks;
- open-position and cooldown checks;
- SL/TP validation;
- broker order-state confirmation.

## Runtime flow

```text
InstrumentManager selects validated basket
→ App commits active basket
→ MarketDataManager subscribes and hydrates
→ CandleEngine builds bars
→ DataHub exposes prepared context
→ StrategyManager evaluates option candidates
→ Risk validates and sizes
→ Execution submits option orders