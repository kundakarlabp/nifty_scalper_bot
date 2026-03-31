# MDM-First Codex Prompts

The following prompts provide implementation guidance for maintaining Market Data Master (MDM) as the single source of truth across the codebase. Use each prompt as-is when planning focused pull requests.

---

1) Enforce MDM in OrderExecutor (no broker reads, exactly-once, atomic legs)

Goal:
Refactor OrderExecutor to depend on an injected MDM/DataHub for ALL reads and use broker ONLY for order submission/cancel/reconcile. Keep exactly-once semantics and atomic multi-leg entries.

Files:
- src/nifty_scalper_bot/execution/order_executor.py
- src/nifty_scalper_bot/execution/order_manager.py (if present)
- tests/execution/test_options_execution_policy.py (adjust doubles)

Tasks:
- OrderExecutor.__init__(..., mdm) -> store self._mdm
- Replace _quote_context() to use self._mdm.get_last_quote(symbol) ONLY (remove broker reads).
- Multi-leg atomic entry: if any leg fails/partial > tolerance, cancel-all, reconcile.
- Reconnect: reconcile_open_orders() and re-attach by client_order_id.
- Preserve OptionsExecutionPolicy hooks (tick/lot/spread/latency/notional) BEFORE submit.

MDM contract (mock in tests):
mdm.get_last_quote(symbol) -> {"bid": float, "ask": float, "ts_ns": int}

Tests:
- Idempotent retry recovers order via client_order_id with MDM-only quotes.
- Atomic two-leg: one leg rejection -> other canceled.
- No code path performs broker reads.

Acceptance:
- Pytest green; grep shows no "get_last_quote" calls on broker.

---

2) DataHub interface + adapters (central read path)

Goal:
Define a clean MDM/DataHub interface and adapters for your current market data source (WS/REST). All strategies/utilities read via MDM.

Files:
- src/nifty_scalper_bot/datahub/interfaces.py
- src/nifty_scalper_bot/datahub/adapters/<provider>.py
- tests/datahub/test_interfaces.py

Interface:
class MarketDataGateway(Protocol):
  def get_last_quote(self, symbol: str) -> dict: ...
  def get_option_quote(self, symbol: str, strike: int, right: Literal["CE","PE"]) -> dict: ...
  def get_orderbook_top(self, instrument: str) -> dict: ...
  def get_depth(self, instrument: str) -> dict: ...
  def now_ns(self) -> int: ...

Adapter tasks:
- Implement adapter mapping provider payloads → canonical dicts with keys: bid, ask, ltp, ts_ns, depth, oi.
- Add staleness guard: if provider ts age>config, mark stale in a field for RiskState.

Tests:
- Contract tests for shape/keys; staleness computed correctly.

---

3) Live OptionChainProvider using MDM (weekly expiry + fast roll)

Goal:
Build OptionChainProvider that constructs OptionChainSnapshot strictly from MDM, selects current weekly expiry, and fast-rolls near close.

Files:
- src/nifty_scalper_bot/data/option_chain_provider.py
- tests/data/test_option_chain_provider.py

Tasks:
- OptionChainProvider(mdm: MarketDataGateway, config) with:
  - current_week_expiry(today_ist: date) -> date
  - fast_roll_window(now_ist: datetime) -> bool
  - get_snapshot(symbol: Literal["NIFTY"]) -> OptionChainSnapshot
    * gathers per-strike quotes+OI via mdm
    * creates OptionQuote, runs sanity_check, then filter_liquid()

Constraints:
- No direct broker reads. All from mdm.
- Handles partial/missing strikes; stable sort.

Tests:
- Expiry selection across Wed/Thu.
- Fast roll window (e.g., 15:10–15:20 IST) returns True.
- Missing strikes handled gracefully.

---

4) Strategy driver wired to MDM (depth + LTT + warm-up)

Goal:
Feed OptionMicroSignal with depth & last trade-through computed from MDM; enforce warm-up; live-reload thresholds from config.

Files:
- src/nifty_scalper_bot/strategies/option_micro_driver.py
- tests/strategies/test_option_micro_driver.py

Tasks:
- OptionMicroDriver(mdm, signal: OptionMicroSignal, config):
  - Pull CE/PE top-of-book + depth + last trade.
  - Derive LTT: last trade at bid/ask within tick_size/2.
  - Warm-up gate: require N_ticks AND N_ms elapsed before entries.
  - Params (snmom_threshold, microvol_threshold, spread_limit, min_depth) read on each call.

Tests:
- No signals during warm-up.
- Proper LTT derivation with synthetic ticks.
- Param change at runtime affects decisions immediately.

---

5) Session/EOD guard + holiday calendar (MDM time source)

Goal:
Authoritative session control using IST time from MDM.now_ns (or a time gateway), with holiday/early-close handling.

Files:
- src/nifty_scalper_bot/infra/session_guard.py
- src/nifty_scalper_bot/infra/holidays_india.json
- tests/infra/test_session_guard.py

Tasks:
- TradingSessionGuard(mdm_time, config):
  is_market_open(now_ist) -> bool
  allow_new_entries(now_ist) -> bool
  must_square_off(now_ist) -> bool
  is_holiday(d: date) -> bool
- Buffers for pre-close “no-new-entries” and square-off deadlines.

Tests:
- Open/close boundaries, holiday blocks, forced square-off near close.

---

6) Precise India options fees + conservative PnL marking

Goal:
Implement accurate Indian options fees and configurable marking mode; ensure all PnL reads prices from MDM/DataHub.

Files:
- src/nifty_scalper_bot/fees/india.py
- src/nifty_scalper_bot/portfolio/pnl.py
- tests/fees/test_india_fees.py
- tests/portfolio/test_marking.py

Fees:
- Brokerage (configurable per-order or per-lot)
- Exchange txn charges
- SEBI fees
- Stamp duty (BUY only)
- STT (SELL only)
- GST on brokerage + txn charges
calc(side: "BUY"|"SELL", qty, price) -> float

PNL:
- MarkMode in settings: MID | CONSERVATIVE
- mark_price(side, bid, ask, mid, mode) -> float
- PnL uses prices pulled only from MDM.

Tests:
- BUY/SELL asymmetry.
- CONSERVATIVE ≤ MID for longs; ≥ for shorts.

---

7) Observability: Prometheus metrics + health using MDM data

Goal:
Expose core metrics and health endpoints that reflect MDM staleness and RiskState reasons.

Files:
- src/nifty_scalper_bot/infra/metrics.py
- src/nifty_scalper_bot/infra/health.py
- tests/infra/test_metrics_health.py

Metrics:
- ticks_per_sec, spread, quote_age_ms, orders_submitted, orders_filled, rejects, partial_fills, rtt_submit_ms, pnl_realized, pnl_unrealized, drawdown, consecutive_losses.
- start_http_server(port).

Health:
- /healthz: OK + latest mdm ts, quote_age_ms, risk reasons.
- /readyz: broker session + mdm connectivity.

Tests:
- Metrics emit increments.
- Health JSON contains mdm-sourced timestamps and reason codes.

---

8) Reconnect/recovery & cancel/replace (MDM-consistent)

Goal:
Recovery flows that never read market data from broker; all staleness detection via MDM; cancel/replace policy with id lineage.

Files:
- src/nifty_scalper_bot/execution/order_manager.py
- src/nifty_scalper_bot/streaming/supervisor.py
- tests/execution/test_reconnect_reconcile.py
- tests/streaming/test_staleness_guard.py

Tasks:
- on_reconnect(): reconcile_open_orders(); dedupe by client_order_id.
- cancel_replace(order_id, new_price): generate new client_order_id with lineage.
- streaming/supervisor: backoff+jitter; re-subscribe; mark quotes stale when mdm gap > threshold.

Tests:
- Outage + recovery → no dup orders; positions coherent.
- Stale detection via MDM time gap, not wall-clock.

---

9) Replay parity (single pipeline for live/backtest)

Goal:
Make tick replay use the same signal→risk→execution pipeline with MDM shim, guaranteeing deterministic parity.

Files:
- src/nifty_scalper_bot/replay/replayer.py
- src/nifty_scalper_bot/backtest/fill_model.py (wrap ExecutionSimulator)
- tests/replay/test_replay_parity.py

Tasks:
- Replayer injects a FakeMDM that serves ticks (ts_ns,bid,ask,ltp,depth,oi).
- Live and backtest both consume via MDM interface.
- FillModel (microstructure-based) used for paper/shadow and backtests.

Tests:
- Two runs with same CSV → byte-identical trade logs.
- Live-sim (FakeMDM + FillModel) equals backtest fills given the same ticks.

---

10) Exposure caps & shadow mode (reads from MDM only)

Goal:
Add net lot/delta caps and SHADOW mode that simulates fills with FillModel, with all reads via MDM.

Files:
- src/nifty_scalper_bot/risk/risk_manager.py (exposure caps)
- src/nifty_scalper_bot/mode/modes.py
- src/nifty_scalper_bot/execution/path_shadow.py
- tests/risk/test_exposure_caps.py
- tests/mode/test_shadow.py

Tasks:
- Risk caps: max_net_lots, approx max_net_delta (near-ATM delta estimate OK), reason codes LOT_CAP/DELTA_CAP.
- SHADOW: do not call broker; route to FillModel; emit metrics; optional sampler to place a fraction live.
- All price/greeks inputs pulled from MDM.

Tests:
- Caps block entries when exceeded.
- Shadow mode records signals, produces fills without broker calls; sampled live placement when enabled.

---

Implementation notes (shared across prompts)

Single source of truth: NO market reads from broker anywhere. Inject mdm: MarketDataGateway into every runtime component that needs prices/depth/ts.

Use OptionsExecutionPolicy before every submit; preserve client_order_id idempotency.

Keep deterministic replay (monotonic ts_ns, no wall-clock randomness in tests).

Each PR must include docstrings + unit tests and keep pytest green.
