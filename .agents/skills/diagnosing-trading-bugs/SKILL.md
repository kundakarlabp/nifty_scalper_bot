---
name: diagnosing-trading-bugs
description: Diagnose hard bugs, intermittent failures, wrong signals, stale-data problems, duplicate orders, broker mismatches, startup failures, and performance regressions in the NIFTY scalper. Use before proposing a fix when the user reports something broken, inconsistent, slow, unsafe, or unexplained.
---

# Diagnosing Trading-Bot Bugs

Follow this sequence. Skip a phase only when the reason is explicit and evidence-based. Read `AGENTS.md` and the relevant repository map before editing.

## 0. Protect capital first

Before reproducing any execution-path problem:

- Prefer unit tests, fixture replay, paper mode, shadow mode, or a broker mock.
- Keep live execution disabled unless the user explicitly requires a controlled live check.
- Never print access tokens, API secrets, session cookies, or complete broker payloads containing credentials.
- Do not bypass readiness, risk, cooldown, capital, position, or instrument-resolution guards to make a reproduction easier.

## 1. Build a tight pass/fail loop

Create one deterministic command that can detect the exact symptom. Prefer, in order:

1. A focused failing test through the real public interface.
2. A replay of captured ticks, candles, order updates, or broker responses.
3. A small fixture-driven harness around the affected module.
4. A dry-run startup command with assertions.
5. A differential run comparing known-good and current behavior.
6. A stress loop for timing, reconnect, race, or duplicate-event failures.

The loop is acceptable only when it is:

- **Red-capable:** it catches the reported symptom, not merely any exception.
- **Deterministic:** repeated runs give the same verdict, or a flaky failure has been amplified to a high reproduction rate.
- **Fast:** seconds where practical.
- **Agent-runnable:** no manual broker interaction unless unavoidable and explicitly documented.

Do not start theorizing from code inspection alone when a red-capable loop can be built.

## 2. Reproduce and minimize

Run the loop and confirm the same failure the user reported. Then remove inputs, modules, configuration, and steps one at a time until every remaining element is necessary.

Capture:

- exact symbol and instrument token
- instrument type
- event timestamps and timezone
- quote source and freshness
- bid, ask, spread, depth, and LTP-only status
- candle interval and bar identity
- readiness and rejection reasons
- order request, broker acknowledgement, fills, and position state
- live, paper, or shadow mode

## 3. Rank falsifiable hypotheses

Generate three to five hypotheses before testing. Each must predict an observable result.

Use this form:

> If X is the cause, changing or observing Y will produce Z.

Trading-bot hypothesis categories commonly include:

- stale or misparsed timestamps; UTC/IST mismatch
- same-bar duplicate evaluation
- option symbol/token or expiry mismatch
- partial or failed option-basket subscription
- quote-quality downgrade from FULL depth to LTP-only
- OHLC hydration gaps or ownership violations
- stale cached context overriding fresh option data
- race between WebSocket, polling fallback, and strategy evaluation
- order retry or reconnect causing duplicate placement
- broker acknowledgement not reconciled with local position state
- hidden config/default drift
- test/backtest path differing from the live path

## 4. Instrument one hypothesis at a time

Prefer debugger or direct state inspection. Otherwise add narrow logs at the seam that separates competing hypotheses.

- Prefix temporary diagnostics with a unique marker such as `[DEBUG-7f3a]`.
- Include module, function, symbol, token, stage, timestamp, event id, and reason.
- Do not log everything and grep later.
- For latency regressions, establish a timing baseline and profile before changing code.

## 5. Fix through the correct seam

Turn the minimal reproduction into a regression test before applying the fix when a valid test seam exists.

Then:

1. Observe the test fail.
2. Apply the smallest architecture-consistent correction.
3. Observe the test pass.
4. Re-run the original unminimized reproduction.
5. Run the relevant focused suite and the repository-required validation commands.

Do not create a parallel selector, hidden fallback, side-channel execution path, or broad exception suppression.

## 6. Verify trading-specific failure modes

For affected areas, explicitly test the relevant cases:

- missing or stale spot context
- unresolved option token
- unsubscribed option symbol
- absent or insufficient option OHLC
- LTP-only quote when bid/ask is required
- wide spread or missing depth
- duplicate tick, candle, signal, order update, or retry
- partial fill, rejected order, delayed acknowledgement, and broker timeout
- reconnect and process restart
- open-position and pending-order recovery
- expiry/day rollover and exchange-session boundaries
- paper/live mode separation

## 7. Cleanup and report

Before declaring completion:

- Original reproduction is green.
- Regression test is green.
- Required compile and test commands were run, or exact blockers are documented.
- Temporary `[DEBUG-...]` instrumentation is removed.
- Throwaway fixtures are deleted or clearly retained as regression assets.
- Root cause is stated precisely with file and function references.
- The PR description states what changed, what did not change, runtime impact, and residual risk.

This workflow is adapted from Matt Pocock's MIT-licensed `diagnosing-bugs` skill and tailored to this repository's live-trading safety constraints.
