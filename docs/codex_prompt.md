# Codex Prompt for Nifty Scalper Bot Enhancements

```
✳️ PROJECT INSTRUCTIONS (for Codex) — NIFTY_SCALPER_BOT

Context

You are editing a Python repo for a Zerodha-based Nifty options scalper bot. We need world-class, robust, production-grade execution with:

Bracket-order mimic (since Zerodha has no native BO): immediate SL + TP on every entry, OCO behavior, partial profit, trailing SL, safe order resizing, and consistent cleanup.

Margin/risk guard before entries; consistent follow-up after fills (no “entry without exits” ever).

Telegram diagnostics & control commands that are simple and useful on mobile.

Zero new dependencies, no public signature changes unless absolutely unavoidable.

CI-clean: black + isort + ruff; keep typing tight (mypy-friendly) where touched.

Small, focused commits using Conventional Commits; open a PR with clear “Why” and risk notes.


Repo Constraints (HARD)

Do NOT add dependencies (requirements.txt/pyproject.lock must not change).

Do NOT change public function/class signatures unless there is no other way to fix a bug; if changed, keep compatibility shims and document it in the PR “Risk.”

Do NOT break existing tests; if we must adjust, keep coverage ≥ previous.

Do NOT leak secrets in logs; redact tokens/keys.

Keep structure/flow intact; make surgical edits.


Objectives (Deliverables)

1. Bracket Order Mimic (core execution)

On entry fill, place:

SL: stop-market (trigger) for full qty.

TP1: limit for partial fraction (configurable; default off = 0.0).


When TP1 fully/partially fills:

Immediately reduce SL quantity to current open position.

If remainder > 0 and second_target_price is defined, place TP2 for the remainder.


When SL fills (full/partial):

Cancel any TP orders (OCO) and stop trailing; ensure we never oversell.


Always handle partial fills on both SL/TP by resizing opposite leg to exact remaining position.

Trailing stop controller:

Optional per-order; parameters: trail_by, min_gap.

Moves SL trigger only in favorable direction (long: up; short: down).

Uses broker modify_order safely, throttled, idempotent.


Robust subscription to order updates (or polling fallback) and single place for OCO logic.



2. Risk & Margin

Before any entry, compute required margin (conservative) including the likely worst-case (entry + SL exposure).

Abort or down-size quantity if insufficient margin or risk cap exceeded.

Log clear, single-line structured reasons for any denial.



3. Order Consistency Guarantees

Never leave a filled position without a protecting SL.

If SL placement fails after entry fill, market-exit immediately and log CRITICAL.

Centralize cleanup: cancel stale legs, stop trailing, free callbacks.



4. Telegram

Keep existing commands. Add or wire robust ones (or aliases) with clear responses:

/mode, /live, /flat, /brk, /diag, /uptime, /limits, /net, /save

Market data helpers: /book <SYM>, /ohlc <SYM>, /chain <ROOT>, /atm <ROOT>, /spot <ROOT>, /greeks <SYM>, /iv <ROOT>, /prevclose <SYM>, /session, /holiday

Strategy/Risk: /sig, /state, /score, /gate_why, /size BUY|SELL price stop atr qty [symbol], /trail, /riskstate, /journal


Ensure aliases don’t override existing commands. Use safe registration.

Redact secrets in error messages. No stack traces to users; log internally.



5. Logging & Observability

Every major state change logs a single, unambiguous line: event=... order_id=... symbol=... qty=... reason=...

Add trailing controller debug logs with backoff to avoid noise.

No secrets in logs (use existing redact helpers).



6. Tests (minimal & surgical)

If edits touch order routing or Telegram commands, add/update small tests to cover:

OCO behavior (cancel opposite leg on fill).

Partial TP reduces SL qty.

SL fills cancel TPs.

Trailing modifies SL only in favorable direction.

Margin guard blocks oversized orders.


Keep coverage ≥ previous; no large fixtures; fast unit style.




---

Files to Touch (expected)

src/nifty_scalper_bot/execution/order_manager.py (or equivalent): add execute_bracket_trade(...) + glue to existing flows.

src/nifty_scalper_bot/execution/trailing.py (new small helper if needed) or integrate into existing SL modifier—no new deps.

src/nifty_scalper_bot/notifications/telegram_commands.py (or existing controller) for safe command registration (if not present).

src/nifty_scalper_bot/notifications/telegram_utils.py for redact/reply helpers (if not present).

Minimal tests under tests/ for the scenarios above (mock broker).

Important: Keep existing public APIs stable; if adding, prefer new private helpers and wire them in the current flow.


---

Core Implementation Details

A) Bracket Trade Orchestrator

Add non-breaking method (adapt names to the repo, keep existing types):

# Inside OrderManager (keep existing signatures untouched; add this safely)
def execute_bracket_trade(
    self,
    symbol: str,
    side: Literal["BUY", "SELL"],
    quantity: int,
    entry_price: float | None,
    stop_loss_price: float,
    take_profit_price: float,
    *,
    partial_profit_fraction: float = 0.0,         # 0.0 => all at TP1
    second_target_price: float | None = None,     # optional TP2
    trailing: dict[str, float] | None = None,     # {"trail_by": x, "min_gap": y}
    tag: str | None = None,
) -> str:
    """
    One-shot execution with SL + TP (and optional TP2) + optional trailing.
    Places SL/TP only *after* entry fill is confirmed.
    Maintains OCO & partial-fill resizing guarantees.
    Returns entry order_id.
    """

Flow (must-haves):

Pre-check: _validate_quantity, _ensure_trading_allowed, margin via broker helper; downsize or abort.

Place entry (market/limit) → wait for fill (subscribe or poll).

Immediately place SL (for full qty) + TP1 (qty split if partial enabled).

Subscribe to broker order updates (or polling fallback) to run OCO rules:

On SL FILLED → cancel TP1/TP2, stop trailing.

On TP1 FILLED → reduce SL qty to remainder; optionally place TP2; keep trailing.

On partial (TP or SL) → resize opposite qty to current net position.

On any placement/modification failure → fail safe: exit position and log CRITICAL.


Trailing (if provided) attaches to SL:

Long: only raises trigger (never lowers).

Short: only lowers trigger.

Honors min_gap; throttles broker modify_order calls; dedup by last effective trigger.


Robust subscription to order updates (or polling fallback) and single place for OCO logic.



B) Trailing Controller (lightweight, no deps)

A small class (or integrate into OrderManager) that:

Keeps last_trigger, best_price.

on_tick(price) decides if a modify is needed.

start(entry_id, sl_order_id, side, spec) / stop(entry_id) lifecycle.

Logs: event=trail_update side=BUY new_trigger=... best=... order_id=...


Reuse existing event loop/ticker callbacks (already present in your data stream manager).

Ensure thread-safety / asyncio correctness for broker calls, with simple lock.


C) Broker Adapters

Use existing broker client methods. If modify quantity isn’t supported, do:

Cancel old SL → place new SL for remainder → re-attach trailing (with state carry).


Provide idempotent cancel (ignore if already done).


D) Telegram Wiring

Ensure all commands listed below are registered only if not already registered (no collisions).

Use redacted error replies; detailed errors go to logs.

Commands (single list—must exist or be safely no-op if features absent):

pos, mode, live, flat, brk, diag, uptime, limits, net, save, book, ohlc, chain, atm, spot, greeks, iv, prevclose, session, holiday, sig, state, score, gate_why, size, trail, riskstate, journal


Keep responses short & clear (mobile-friendly).



---

Logging & Errors (strict)

Single-line structured logs (compatible with your logger):
event=bracket.place entry_id=... symbol=... side=... qty=... tp=... sl=...

NEVER log tokens/keys; use redact helper.

On any exception placing SL/TP/modify: log .critical and flatten position if entry was filled.



---

Tests You Must Add/Adjust (unit, minimal)

Create/adjust fast tests using a mock broker:

1. oco_cancel_on_sl_fill: SL fill cancels TP(s), stops trailing.


2. partial_tp_reduces_sl: Partial TP fill reduces SL qty.


3. tp2_flow: TP1 filled → SL resized → TP2 placed → TP2 fill cancels SL.


4. trailing_directional: Trailing raises long SL trigger on rising prices; never lowers; respects min_gap.


5. margin_guard_blocks: Oversized order throws/returns error with clear message.


6. telegram_register_safe: Commands add if missing; do not override existing.


Keep tests tiny; mock time/ticks; no network.



---

Git / PR Flow

1. Create branch: fix/bracket-trailing-oco.


2. Commits (small, focused):

feat(execution): add bracket trade orchestrator with OCO + trailing

feat(execution): partial TP logic and SL resizing

fix(broker): safe modify-or-replace for SL qty

feat(telegram): safe command registration + diagnostics

test(execution): OCO, partial, trailing, margin



3. Format & lint:

ruff check src --fix

black . && isort .

mypy src

pytest -q (or your existing test runner flags)



4. PR title: feat: bracket order mimic with trailing & partial TP

Why: Zerodha no BO; ensure protected exits; remove sell-follow failures; maximize profit w/ trailing + partials.

Risk: Broker compatibility for modify qty; race on rapid fills; resolved via cancel-and-replace, idempotent cancel, throttled trailing.

Labels: area:execution, type:feat, risk:medium

Block merge if CI fails or deps changed.



---

Acceptance Criteria (must pass)

All tests green; coverage ≥ previous.

No new deps; no signature breaks (unless documented).

On live/paper sim:

Entry → SL+TP placed immediately after fill.

Partial TP reduces SL qty.

SL fill cancels TPs.

Trailing moves SL trigger only in favorable direction, respects min_gap.

Logs are clear, single lines, no secrets.



---

Helpful Stubs (you can adapt to the repo)

Hook points you can rely on (rename if needed to match existing files):

OrderManager._calculate_required_margin(symbol, side, qty)

Broker.place_order(...), modify_order(...), cancel_order(...), subscribe_orders([...], callback)

MarketData.on_tick(symbol, callback) (or equivalent) to drive trailing.

Telegram: register_telegram_commands(bot, application, services) ensures safe add.


OPTIONS-ONLY ADDENDUM (Zerodha NFO)

Scope
- Trade ONLY NFO option contracts (CE/PE). No futures, no equities.
- Product: MIS (intraday) unless config says NRML. Variety: "regular".
- Quantity must be in exchange lot size multiples (e.g., NIFTY=75). Validate & round down.

Reduce-only exits (no naked shorts)
- Any SELL must be reduce-only against an existing LONG of the EXACT same tradingsymbol/product.
- Implement place_reduce_only_exit(...) that:
  - Reads live positions -> finds net long qty for tradingsymbol+product.
  - Caps exit qty to that long qty.
  - Sends SELL only for that exact symbol; never flip legs (CE↔PE) or strikes.
  - If long_qty==0 → skip and log event=exit_skip_no_position.
- If broker returns “Insufficient funds” for that SELL, treat as symbol/product mismatch (critical log) and DO NOT retry as fresh short.

Bracket mimic (SL/TP) for options
- On ENTRY fill of an OPTION symbol:
  - Place SL as STOP-MARKET (trigger) for the option price (not underlying).
  - Place TP1 as LIMIT. If partial fraction>0, split qty; else full qty at TP1.
  - Optional TP2: only for the remainder after TP1 fills (OCO behavior).
- OCO engine rules:
  - If SL (any qty) fills → cancel TP1/TP2 leftovers and stop trailing.
  - If TP1 (any qty) fills → reduce SL qty to current position, then place TP2 if configured.
  - Always re-read current net position before modifying/cancelling to avoid oversell.
  - If modify-qty is unsupported, use cancel-and-replace pattern; idempotent cancels.

Trailing stop (option LTP)
- Trailing is applied to the OPTION instrument (use option LTP).
- Longs: raise SL trigger only; Shorts (rare, and disabled by default): lower only.
- Respect min_gap (no churn). Throttle modify_order. Keep last effective trigger to dedupe.

Margin & sizing (options)
- For entries only, call order_margins (or adapter equivalent) with OPTION payload.
- If margin insufficient → downsize to next lower lot multiple; if still insufficient → block entry with event=margin_block details.
- Never margin-check exits (reduce-only).

Symbol resolution
- Resolver must return a single tradingsymbol string like "NIFTY20OCT25450CE".
- Store entry context {order_id, tradingsymbol, product, qty, avg_price}.
- All downstream actions (SL/TP/trailing/exit) use that exact tradingsymbol.

Validation & safety
- Enforce MIS/NRML consistency between entry and exits.
- Enforce exchange="NFO".
- Refuse sending SELL for different strike/month/option type than the entry (log event=reduce_only_violation).
- Lot guard: qty % lot_size == 0 else floor to valid multiple (log event=qty_rounded).

Telegram (options helpers)
- /pos → show NFO option positions with tradingsymbol, product, net_qty.
- /flat → iterate place_reduce_only_exit(...) over every long option until flat.
- /chain <ROOT>, /atm <ROOT>, /greeks <SYMBOL> must operate on options only.
- Errors must be short; details go to logs; redact tokens.

Tests (options)
- Cover: OCO cancel on SL fill (options), partial TP reduces SL (options), trailing raises only for longs (option LTP), margin guard blocks oversize, reduce-only prevents naked shorts.

Implementation notes Codex should follow (practical)

Order placement:

transaction_type: "BUY" on entry; "SELL" only via place_reduce_only_exit(...).

order_type: "MARKET" for entry (or configurable limit), "SL-M" (stop-market) for SL, "LIMIT" for TP.

validity="DAY", exchange="NFO", product from config (MIS default).


Price math:

TP/SL can be absolute or % of entry; define config:

tp_type: "percent"|"abs", tp_value, sl_type, sl_value.

Compute trigger/limit on the option price.


Lot size:

Get lot size from your instruments cache/contract registry; default NIFTY=75 (config fallback). Always qty = (qty // lot) * lot.


Data wiring:

Trailing uses option LTP updates from your market data manager; if no push, poll at safe interval.


Logging:

Single-line events: bracket.place, sl.place, tp.place, oco.cancel, sl.resize, trail.update, exit.reduce_only, with order_id, symbol, qty, price/trigger, reason.
```

